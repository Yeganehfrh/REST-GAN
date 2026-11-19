"""Building blocks of the generator and the critic. Inside the networks, signals are (batch, channels, time)."""
import math

import torch
import torch.nn.functional as F
from torch import nn


def init_normal(module, std=0.02):
    """Weights ~ N(0, std), biases zero: the initialisation of every conv and linear layer."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class SameConv1d(nn.Conv1d):
    """Conv1d with output length ceil(T / stride). Odd padding puts the extra sample at the end (Keras 'same')."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        init_normal(self)

    def forward(self, x):
        span = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        stride, length = self.stride[0], x.shape[-1]
        total = span - 1 if stride == 1 else max(0, (math.ceil(length / stride) - 1) * stride + span - length)
        left, right = total // 2, total - total // 2
        if left != right:
            x, left = F.pad(x, (left, right)), 0
        return F.conv1d(x, self.weight, self.bias, self.stride, left, self.dilation)


class DualFiLM(nn.Module):
    """FiLM by participant and eye state: out = gamma * x + beta, one value per channel."""

    def __init__(self, n_channels, d_sub=32, d_state=16):
        super().__init__()
        self.sub_film = nn.Linear(d_sub, 2 * n_channels)
        self.state_film = nn.Linear(d_state, 2 * n_channels)
        self.g_sub = nn.Parameter(torch.tensor(1.0))
        self.g_state = nn.Parameter(torch.tensor(1.0))
        # zero init: the block starts as the identity
        for layer in (self.sub_film, self.state_film):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, subj_emb, state_emb):
        """x: (B, C, T); subj_emb: (B, d_sub); state_emb: (B, d_state)."""
        gamma_sub, beta_sub = self.sub_film(subj_emb).chunk(2, dim=-1)
        gamma_state, beta_state = self.state_film(state_emb).chunk(2, dim=-1)
        # gamma is centred at 1, both terms are down-scaled by 0.1; g_sub and g_state are learnable gates
        gamma = 1.0 + 0.1 * (self.g_sub * gamma_sub + self.g_state * gamma_state)
        beta = 0.1 * (self.g_sub * beta_sub + self.g_state * beta_state)
        return gamma[:, :, None] * x + beta[:, :, None]


class ConditionProjection(nn.Module):
    """Linear map of the two embeddings for the projection term of the critic (Miyato & Koyama, 2018)."""

    def __init__(self, d_sub, d_state, dim):
        super().__init__()
        self.sub = init_normal(nn.Linear(d_sub, dim, bias=False))
        self.state = init_normal(nn.Linear(d_state, dim, bias=False))

    def forward(self, subj_emb, state_emb):
        return self.sub(subj_emb) + self.state(state_emb)


class SecondDifference(nn.Module):
    """Fixed filter [-1, 2, -1] / 2 on every channel: emphasises high frequencies. No parameters."""

    def __init__(self):
        super().__init__()
        self.register_buffer('kernel', torch.tensor([-1.0, 2.0, -1.0]).view(1, 1, 3) / 2.0, persistent=False)

    def forward(self, x):
        channels = x.shape[1]
        return F.conv1d(x, self.kernel.to(x.dtype).expand(channels, 1, 3), padding=1, groups=channels)


class MinibatchStdDev(nn.Module):
    """Append one feature map with the mean standard deviation across the batch (Karras et al., 2018)."""

    def __init__(self, eps=1e-8, with_grad=False):
        super().__init__()
        self.eps = eps
        self.with_grad = with_grad

    def forward(self, x):
        batch, _, length = x.shape
        if batch <= 1:
            value = x.new_zeros(())
        else:
            # without gradient by default: the value is then a constant per batch
            source = x.float() if self.with_grad else x.detach().float()
            value = torch.sqrt(source.var(dim=0, unbiased=False) + self.eps).mean().to(x.dtype)
        return torch.cat([x, value.expand(batch, 1, length)], dim=1)


class ParticipantEncoder(nn.Module):
    """Participant embedding from reference segments of one recording: refs (B, K, T, C) -> (B, d_out)."""

    def __init__(self, n_channels, d_out=32):
        super().__init__()
        c = n_channels
        self.convs = nn.ModuleList([SameConv1d(c, 4 * c, 7, stride=2), SameConv1d(4 * c, 8 * c, 5, stride=2),
                                    SameConv1d(8 * c, 16 * c, 5, stride=2)])
        self.act = nn.LeakyReLU(0.2)
        self.out = init_normal(nn.Linear(32 * c, d_out))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, refs):
        batch, k = refs.shape[:2]
        h = refs.flatten(0, 1).transpose(1, 2)
        for conv in self.convs:
            h = self.act(conv(h))
        h = h.float()
        # mean and SD of every feature map over time, averaged over the reference segments
        stats = torch.cat([h.mean(2), torch.sqrt(h.var(2, unbiased=False) + 1e-5)], dim=1)
        return self.norm(self.out(stats.view(batch, k, -1).mean(1)))


class ChannelAffine(nn.Module):
    """Learned scale and shift per channel."""

    def __init__(self, n_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_channels))
        self.shift = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]


class SpectralFilter(nn.Module):
    """Linear filter that is specific to the participant and the eye state: one FIR kernel per pair of channels.
    The filter is the identity plus a correction, which starts at zero. x: (B, C, L) -> (B, C, L - n_taps + 1).
    """

    def __init__(self, n_channels, d_sub=32, d_state=16, n_taps=129):
        super().__init__()
        if n_taps % 2 == 0:
            raise ValueError(f'n_taps must be odd, got {n_taps}')
        self.n_channels = n_channels
        self.n_taps = n_taps
        self.n_bins = (n_taps - 1) // 2 + 1
        n_out = n_channels * n_channels * self.n_bins * 2
        self.sub = nn.Linear(d_sub, n_out)  # its bias is the correction that all participants share
        self.state = nn.Linear(d_state, n_out, bias=False)
        for layer in (self.sub, self.state):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.sub.bias)
        identity = torch.zeros(n_channels, n_channels, n_taps)
        identity[torch.arange(n_channels), torch.arange(n_channels), n_taps // 2] = 1.0
        self.register_buffer('identity', identity, persistent=False)
        self.register_buffer('window', torch.hann_window(n_taps + 2, periodic=False)[1:-1].clone(), persistent=False)

    def response(self, subj_emb, state_emb):
        """Frequency response of the correction before tapering, complex (B, C_out, C_in, n_bins)."""
        r = (self.sub(subj_emb) + self.state(state_emb)).view(-1, self.n_channels, self.n_channels, self.n_bins, 2)
        return torch.complex(r[..., 0], r[..., 1])

    def kernels(self, subj_emb, state_emb):
        """Kernel of every item (B, C_out, C_in, n_taps), as `conv1d` weights."""
        taps = torch.fft.irfft(self.response(subj_emb, state_emb), n=self.n_taps - 1, dim=-1)
        taps = F.pad(torch.roll(taps, self.n_taps // 2, dims=-1), (0, 1))  # lags -(n-1)/2 ... (n-1)/2, the last one zero
        return self.identity + (taps * self.window).flip(-1)

    def forward(self, x, subj_emb, state_emb):
        batch, channels, length = x.shape
        dtype = self.sub.weight.dtype  # full precision, also under mixed precision
        with torch.autocast(x.device.type, enabled=False):
            kernels = self.kernels(subj_emb.to(dtype), state_emb.to(dtype))
            # one group per item: every item is filtered with its own kernel
            y = F.conv1d(x.to(dtype).reshape(1, batch * channels, length),
                         kernels.reshape(batch * channels, channels, self.n_taps), groups=batch)
        return y.view(batch, channels, length - self.n_taps + 1)
