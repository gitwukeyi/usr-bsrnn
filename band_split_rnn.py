import torch
from thop import profile
from torch import nn, Tensor


class BandSplit(nn.Module):
    def __init__(self, up_sample_dim: int, split_bands: int, band_width: int):
        super(BandSplit, self).__init__()

        self.split_bands = split_bands
        self.band_width = band_width

        self.layer_list = nn.ModuleList()

        for _ in range(split_bands):
            layer = nn.Sequential(
                nn.LayerNorm([band_width*2]),
                nn.Linear(in_features=band_width * 2, out_features=up_sample_dim, bias=False)
            )
            self.layer_list.append(layer)

    def forward(self, real: Tensor, imag: Tensor):
        """
        :param imag: (B, T, F)
        :param real: (B, T, F)
        :return: (B, N, K, T)
        """
        band_out = list()
        for idx in range(self.split_bands):
            real_sub_band = real[..., idx * self.band_width: (idx + 1) * self.band_width]
            imag_sub_band = imag[..., idx * self.band_width: (idx + 1) * self.band_width]

            real_imag = torch.cat([real_sub_band, imag_sub_band], dim=-1)  # (B, T, 2band)
            sub_out = self.layer_list[idx](real_imag)  # (B, T, N)
            band_out.append(sub_out)

        out = torch.stack(band_out, dim=1)  # (B, K, T, N)
        out = torch.permute(out, (0, 3, 1, 2))

        return out  # (B, N, K ,T)


class FeatureAcross(nn.Module):
    def __init__(self, up_sample_dim: int):
        super(FeatureAcross, self).__init__()

        self.norm_t = nn.BatchNorm2d(num_features=up_sample_dim)
        self.lstm_t = nn.LSTM(input_size=up_sample_dim, hidden_size=up_sample_dim * 2,
                              bidirectional=False, batch_first=True)
        self.proj_t = nn.Linear(up_sample_dim * 2, up_sample_dim, bias=False)

        self.norm_k = nn.BatchNorm2d(num_features=up_sample_dim)
        self.lstm_k = nn.LSTM(input_size=up_sample_dim, hidden_size=up_sample_dim * 2,
                              bidirectional=True, batch_first=True)
        self.proj_k = nn.Linear(up_sample_dim * 4, up_sample_dim, bias=False)

    def forward(self, inputs):
        """
        :param inputs: (B, N, K, T)
        :return: (B, N, K, T)
        """
        B, N, K, T = inputs.shape
        norm_out = self.norm_t(inputs)

        norm_out = torch.permute(norm_out, (0, 2, 3, 1))  # (B, K, T, N)
        norm_out = torch.reshape(norm_out, (B * K, T, N))

        lstm_out, _ = self.lstm_t(norm_out)
        lstm_out = self.proj_t(lstm_out)
        lstm_out = torch.reshape(lstm_out, (B, K, T, N))
        lstm_out = torch.permute(lstm_out, (0, 3, 1, 2))
        lstm_out_t = lstm_out + inputs  # (B, N, K, T)

        norm_out = self.norm_k(lstm_out_t)
        lstm_out = torch.permute(norm_out, (0, 2, 3, 1))
        lstm_out = torch.reshape(lstm_out, (B * T, K, N))
        lstm_out, _ = self.lstm_k(lstm_out)
        lstm_out = self.proj_k(lstm_out)
        lstm_out = torch.reshape(lstm_out, (B, T, K, N))  # (B, T, K, N)
        lstm_out = torch.permute(lstm_out, (0, 3, 2, 1))  # (B, N, K, T)

        out = lstm_out + lstm_out_t

        return out  # (B, N, K, T)


class Transpose(nn.Module):
    def forward(self, inputs):
        return torch.transpose(inputs, 1, 2)


class BandMerge(nn.Module):
    def __init__(self, up_sample_dim: int, split_bands: int, band_width: int):
        super(BandMerge, self).__init__()
        self.split_bands = split_bands
        self.band_width = band_width
        self.layer_list = nn.ModuleList()
        for idx in range(split_bands):
            layer = nn.Sequential(
                Transpose(),
                nn.BatchNorm1d(num_features=up_sample_dim),
                Transpose(),
                nn.Linear(in_features=up_sample_dim, out_features=up_sample_dim * 4),
                nn.Tanh(),
                nn.Linear(in_features=up_sample_dim * 4, out_features=band_width * 4),
                nn.GLU(dim=-1)
            )
            self.layer_list.append(layer)

    def forward(self, inputs: Tensor):
        """
        :param inputs: (B, N, K, T)
        :return: (B, T, F, 2)
        """
        B, _, _, T = inputs.shape
        inputs = torch.permute(inputs, (0, 3, 1, 2))  # (B, T, N, K)

        out_list = list()
        for idx in range(self.split_bands):
            sub_band = inputs[..., idx]  # (B, T, N)
            sub_band = self.layer_list[idx](sub_band)  # (B, T, 2band)
            sub_band = torch.reshape(sub_band, (B, T, self.band_width, 2))  # (B, T, band_width, 2)
            out_list.append(sub_band)

        out = torch.cat(out_list, dim=2)  # (B, T, F, 2)

        return out


class BandSplitRnn(nn.Module):
    def __init__(self, up_sample_dim: int, fre_dim: int, band_width: int, enhance_blocks: int = 6):
        """
        An implement of paper: computational efficient monaural speech enhancement with universal sample rate
        band-split RNN
        :param up_sample_dim:
        :param fre_dim:
        :param band_width:
        :param enhance_blocks:
        """
        super().__init__()
        assert (fre_dim-1) % band_width == 0, f"(fre_dim-1)/band_width, {fre_dim-1} is not divisible by {band_width}"
        split_bands = (fre_dim - 1)//band_width

        self.band_split = BandSplit(up_sample_dim=up_sample_dim, split_bands=split_bands, band_width=band_width)

        self.enhance_layer = nn.Sequential(
            *[FeatureAcross(up_sample_dim=up_sample_dim) for _ in range(enhance_blocks)]
        )

        self.band_merge = BandMerge(up_sample_dim=up_sample_dim, split_bands=split_bands, band_width=band_width)
        self.pad_dc = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

    def forward(self, real: Tensor, imag: Tensor):
        """
        :param real:  (B, F, T)
        :param imag:
        :return: (B, 2, T, F)
        """
        real = real[:, 1:, :]
        imag = imag[:, 1:, :]  # remove dc part
        real = torch.transpose(real, 1, 2)
        imag = torch.transpose(imag, 1, 2)  # (B, T, F)

        out = self.band_split(real, imag)
        out = self.enhance_layer(out)
        mask = self.band_merge(out)

        real_mask = mask[..., 0]  # (B, T, F)
        imag_mask = mask[..., 1]

        out_real = real * real_mask
        out_imag = imag * imag_mask

        out = torch.stack([out_real, out_imag], dim=1)  # (B, 2, T, F)
        out = self.pad_dc(out)

        return out


if __name__ == "__main__":
    test_data = torch.randn(1, 7 * 2048)
    test_spec = torch.stft(test_data, n_fft=2048, hop_length=512, center=False, return_complex=True)
    test_real = torch.real(test_spec)
    test_imag = torch.imag(test_spec)
    test_layer = BandSplitRnn(up_sample_dim=16, fre_dim=1025, band_width=8)
    import time

    start_time = time.time()
    test_out = test_layer(test_real, test_imag)
    end_time = time.time()
    macs, params = profile(test_layer, inputs=(test_real, test_imag))
    print(f"mac: {macs / 1e9} G \nparams: {params / 1e6}M")
    print(f"total time = {(end_time - start_time) * 1000}ms")
