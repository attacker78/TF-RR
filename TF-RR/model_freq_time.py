import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from s4 import S4Block as S4
import numpy as np
from Embed import TokenEmbedding


class dialted_residual_inception_module_down(nn.Module):
    def __init__(self, d_model=128, d_input=2, in_channels=128, out_channels=128):
        super(dialted_residual_inception_module_down, self).__init__()

        # Define Conv1D layer
        self.encoder = nn.Conv1d(d_model, d_model, kernel_size=1)
        # Define encoder layer
        # self.encoder = nn.Linear(d_input, d_model)
        # Define S4 layer
        self.s4 = S4(d_model=d_model)

        # self.drop = nn.Dropout(0.2)
        # Define activation function
        self.activation = nn.LeakyReLU(0.3)
        # Dilated convolutions
        self.dilated_conv1d_layers1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1)
        self.dilated_conv1d_layers2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=2, dilation=2)
        self.dilated_conv1d_layers3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=4, dilation=4)
        self.dilated_conv1d_layers4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=8, dilation=8)

        # BatchNormalization
        self.normalize = nn.BatchNorm1d(out_channels)

        self.conv = nn.Conv1d(in_channels=out_channels * 4, out_channels=out_channels * 2, kernel_size=1)

    def forward(self, x):
        """

        :param x: (B, in_channels, length)
        :return:
        """
        # x = self.encoder(x)
        # x = x.permute(0, 2, 1)
        x, _ = self.s4(x)
        # x = self.drop(x)

        dilated_layer1 = self.dilated_conv1d_layers1(x)
        # dilated_layer1 = self.activation(dilated_layer1)
        # dilated_layer1, _ = self.s4(dilated_layer1)  # (B, H, L) ---> (B, H, L)
        dilated_layer1 = self.normalize(dilated_layer1)

        dilated_layer2 = self.dilated_conv1d_layers2(x)
        dilated_layer2 = self.activation(dilated_layer2)
        # dilated_layer2, _ = self.s4(dilated_layer2)  # (B, H, L) ---> (B, H, L)
        dilated_layer2 = self.normalize(dilated_layer2)

        dilated_layer3 = self.dilated_conv1d_layers3(x)
        dilated_layer3 = self.activation(dilated_layer3)
        # dilated_layer3, _ = self.s4(dilated_layer3)  # (B, H, L) ---> (B, H, L)
        dilated_layer3 = self.normalize(dilated_layer3)

        dilated_layer4 = self.dilated_conv1d_layers4(x)
        dilated_layer4 = self.activation(dilated_layer4)
        # dilated_layer4, _ = self.s4(dilated_layer4)  # (B, H, L) ---> (B, H, L)
        dilated_layer4 = self.normalize(dilated_layer4)

        dilated = torch.cat([dilated_layer1, dilated_layer2, dilated_layer3, dilated_layer4], dim=1)
        dilated = self.conv(dilated)

        return dilated


class dialted_residual_inception_module_up(nn.Module):
    def __init__(self, d_model=128, d_input=2, in_channels=128, out_channels=128):
        super(dialted_residual_inception_module_up, self).__init__()

        # Deconvolution
        # self.dilated_conv1d_layers4 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=4, output_padding=1)
        # self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, dilation=1, output_padding=0)
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, dilation=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, dilation=2)
        self.deconv3 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, dilation=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, dilation=8)

        # BatchNormalization
        self.normalize = nn.BatchNorm1d(out_channels)

        self.activation = nn.LeakyReLU(0.3)

        # Define S4 layer
        self.s4 = S4(d_model=d_model)

        # self.drop = nn.Dropout(0.2)

        self.conv = nn.Conv1d(in_channels=out_channels * 4, out_channels=out_channels * 2, kernel_size=1)

    def forward(self, x):
        """

        :param x: (B, in_channels, length)
        :return:
        """

        dilated_layer1 = self.deconv1(x)[:, :, :x.size(2) * 2]
        dilated_layer1 = self.normalize(dilated_layer1)

        dilated_layer2 = self.deconv2(x)[:, :, :x.size(2) * 2]
        dilated_layer2 = self.activation(dilated_layer2)
        # dilated_layer2, _ = self.s4(dilated_layer2)  # (B, H, L) ---> (B, H, L)
        dilated_layer2 = self.normalize(dilated_layer2)

        dilated_layer3 = self.deconv3(x)[:, :, :x.size(2) * 2]
        dilated_layer3 = self.activation(dilated_layer3)
        # dilated_layer3, _ = self.s4(dilated_layer3)  # (B, H, L) ---> (B, H, L)
        dilated_layer3 = self.normalize(dilated_layer3)

        dilated_layer4 = self.deconv4(x)[:, :, :x.size(2) * 2]
        dilated_layer4 = self.activation(dilated_layer4)
        # dilated_layer4, _ = self.s4(dilated_layer4)  # (B, H, L) ---> (B, H, L)
        dilated_layer4 = self.normalize(dilated_layer4)

        dilated = torch.cat([dilated_layer1, dilated_layer2, dilated_layer3, dilated_layer4], dim=1)
        dilated = self.conv(dilated)

        x, _ = self.s4(dilated)  # (B, H, L) ---> (B, H, L)
        # x = self.drop(x)

        return x


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # try:
        x = self.value_embedding(x)
        # except:
        #     a = 1
        return self.dropout(x)


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        # Use torch's random sampling to avoid np.random issues in multi-process environments
        import torch
        index = list(range(0, seq_len // 2))
        # Use torch's random permutation
        perm = torch.randperm(len(index)).tolist()
        index = [index[i] for i in perm[:modes]]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


### Frequency model
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=32, mode_select_method='list'):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.scale = (1 / (self.in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=32, mode_select_method='list',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        # Get modes for queries and keys (& values) on frequency domain
        # self.index_q = list(range(0, modes))
        # self.index_kv = list(range(0, modes))
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):  # Non-random, self.index_q only serves truncation purpose
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        rq = xq_ft_[:, :, :, :self.modes1].abs()  # Magnitude information of q
        pq = xq_ft_[:, :, :, :self.modes1].angle()  # Phase information of q

        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        rk = xk_ft_[:, :, :, :self.modes1].abs()  # Magnitude information of k
        pk = xk_ft_[:, :, :, :self.modes1].angle()  # Phase information of k

        # Perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} activation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)


class AutoCorrelationLayer_encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer_encoder, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = SpectralConv1d(in_channels=d_model, out_channels=d_model, seq_len=2000)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AutoCorrelationLayer_decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer_decoder, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = FourierCrossAttention(in_channels=d_model, out_channels=d_model, seq_len_q=2000,
                                                        seq_len_kv=2000, mode_select_method='list')
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(Encoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AutoCorrelationLayer_encoder(d_model=d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.encoder_layers = 2

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(Decoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AutoCorrelationLayer_decoder(d_model=d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.encoder_layers = 2

    def forward(self, q, x):  # q == seasonal
        new_x, attn = self.attention(q, x, x)

        # new_x = new_x + q
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Frequency_module(nn.Module):
    def __init__(self, d_input=2, d_model=64):
        super(Frequency_module, self).__init__()

        # Encoder
        # Fre_embedding
        self.encoder_layers = 2
        self.freq_embedding = DataEmbedding_wo_pos(c_in=2, d_model=d_model)
        self.freq_embedding1 = DataEmbedding_wo_pos(c_in=1, d_model=d_model)
        # self.encoder_module = nn.Sequential(Encoder(d_model),Encoder(d_model))
        self.encoder_module = Encoder(d_model)

        # Define decoder
        self.raw_frequency = nn.Parameter(torch.randn(1))  # Raw frequency parameter, initialized with random value
        self.decoder_module = Decoder(d_model=d_model)

    def forward(self, x):
        """

        :param x: input x(batch, length, d_input)
        :param labels:
        :param current_epoch:
        :return:
        """
        ### Process in frequency domain

        frequency = 0.1 + 0.5 * F.sigmoid(self.raw_frequency)
        # frequency = 0.1 + 0.5 * self.raw_frequency
        # Use trainable frequency parameter to generate sine wave here
        t = torch.arange(0, 16, 1 / 125, dtype=torch.float32, device=x.device)
        t = t.unsqueeze(0).expand(x.size(0), -1)
        seasonal_init = torch.cos(2 * torch.pi * frequency * t).unsqueeze(-1)
        seasonal_init = self.freq_embedding1(seasonal_init)

        f_x_ = self.freq_embedding(x)
        ef = self.encoder_module(f_x_)

        ef, _ = self.decoder_module(seasonal_init, ef)

        return ef


class FED_s4(nn.Module):
    def __init__(self, d_input=2, d_model=64):
        super(FED_s4, self).__init__()

        # Define encoder layer
        # Fre_embedding
        self.freq_embedding = DataEmbedding_wo_pos(c_in=2, d_model=d_model)
        self.freq_module = Frequency_module()

        self.dilated_block1 = dialted_residual_inception_module_down(d_model=64, in_channels=64, out_channels=64)
        self.dilated_block2 = dialted_residual_inception_module_down(d_model=64 * 2, in_channels=64 * 2, out_channels=128)

        self.dilated_block3 = dialted_residual_inception_module_up(d_model=64 * 2, in_channels=64 * 4, out_channels=64)
        self.dilated_block4 = dialted_residual_inception_module_up(d_model=64, in_channels=64 * 2, out_channels=32)

        # self.dilated_block4 = dialted_residual_inception_module_up(d_model=64, in_channels=64, out_channels=64)
        self.decoder = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.decoder1 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        self.decoder2 = nn.Conv1d(256, 64, kernel_size=1)

    def forward(self, x):
        ef = self.freq_module(x)

        f_x_ = self.freq_embedding(x)

        et1 = self.dilated_block1(f_x_.permute(0, 2, 1))
        et2 = self.dilated_block2(et1)

        et3 = self.dilated_block3(et2)
        et4 = self.dilated_block4(et1 + et3)
        et = et4.permute(0, 2, 1)

        # Version 1: Direct concatenation of time domain and frequency domain results
        # src_recon = torch.concat([ef, et], -1).permute(0, 2, 1)
        # src_recon = F.normalize(src_recon)

        # Version 2: Hadamard product connecting time domain and frequency domain models
        # src_recon = torch.einsum('bld,bld->bld', ef, et).permute(0, 2, 1)
        # src_recon = F.normalize(src_recon)
        # src_recon = self.decoder(src_recon)
        # src_recon = F.sigmoid(src_recon)
        # src_recon = self.decoder1(src_recon)
        # src_recon = F.sigmoid(src_recon)

        # Version 3: DNN connecting time domain and frequency domain results - not feasible, matrix too large

        # Version 4: Direct addition
        src_recon = (ef + et).permute(0, 2, 1)
        src_recon = self.decoder(src_recon)
        src_recon = self.decoder1(src_recon)

        return src_recon.squeeze()


if __name__ == '__main__':
    # Randomly generate input data
    batch_size = 32

    model = FED_s4()

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, 2000, 2])
    out = model.forward(enc)
    print(out)