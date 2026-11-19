"""Generator, critic and losses of REST-GAN (conditional WGAN-GP for resting-state EEG).

EEG segments are (batch, time, channels). `sub` is the participant index and `state` the eye state
(0 = eyes closed, 1 = eyes open).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .layers import (SameConv1d, DualFiLM, ConditionProjection, SecondDifference, MinibatchStdDev, ChannelAffine,
                     ParticipantEncoder, SpectralFilter, init_normal)
from .spectral import N_BINS, segment_spectra

D_SUB = 32      # size of the participant embedding
D_STATE = 16    # size of the eye-state embedding
DEEP_LAYERS = 4      # extra strided layers of the deep critic
FILTER_TAPS = 129    # taps of the generator's output filter (1 s at 128 Hz)
NOISE_DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2, 4)  # dilations of the generator (receptive field 141 samples, 1.1 s)
SLOW_TAPS = 257      # the slow noise channels are white noise smoothed over 2 s
SPECTRAL_HEAD = (128, 64)  # filters of the two convolutions of the spectral head
HEAD_BINS = 44       # the spectral head uses the bins from 0 to 43 Hz (the data are low-passed at 45 Hz)


def _float32(fn, *tensors):
    """Run `fn` in full precision, also under mixed precision (the critic's score is too large for float16)."""
    with torch.autocast(tensors[0].device.type, enabled=False):
        return fn(*(t.float() for t in tensors))


class Critic(nn.Module):
    """Critic: a convolutional encoder with the score, the reconstruction branch and the spectral head."""

    def __init__(self,
                 time_dim,
                 feature_dim,
                 n_subjects,
                 use_recon=True,          # reconstruction branch
                 use_mbsd=True,           # minibatch standard deviation layer
                 use_second_diff=True,    # second difference of the input, concatenated to it
                 critic_cond='projection',  # 'projection' (embeddings enter at the output) or 'film' (FiLM on the input)
                 use_contrastive=False,   # head for the contrastive loss
                 subject_cond='table',    # 'table' (one embedding per participant) or 'encoder' (from reference segments)
                 proj_cov=False,          # projection also on the covariance of the second layer's feature maps
                 deep=False,              # DEEP_LAYERS more strided layers (receptive field 0.15 s -> 2 s)
                 spectral_head=True,      # second path to the score, from the 1-Hz cross-spectrum of the segment
                 d_sub=D_SUB,
                 head_bins=HEAD_BINS,     # the spectral head uses the bins from 0 to head_bins - 1 Hz
                 mbsd_grad=False):        # minibatch standard deviation with gradient
        super().__init__()
        self.deep = deep
        self.spectral_head = spectral_head
        self.d_sub = d_sub
        self.head_bins = head_bins
        self.mbsd_grad = mbsd_grad
        if not 1 <= head_bins <= N_BINS:
            raise ValueError(f'head_bins must be between 1 and {N_BINS}, got {head_bins}')
        if proj_cov and critic_cond != 'projection':
            raise ValueError("proj_cov needs critic_cond='projection'.")
        self.proj_cov = proj_cov
        if critic_cond not in ('film', 'projection'):
            raise ValueError(f"critic_cond must be 'film' or 'projection', got {critic_cond!r}")
        if use_contrastive and critic_cond != 'projection':
            raise ValueError("The contrastive auxiliary loss needs critic_cond='projection': with FiLM on the "
                             "input the encoder is told the identity, which makes the task trivial.")
        if subject_cond not in ('table', 'encoder'):
            raise ValueError(f"subject_cond must be 'table' or 'encoder', got {subject_cond!r}")
        if subject_cond == 'encoder' and critic_cond != 'projection':
            raise ValueError("subject_cond='encoder' needs critic_cond='projection'.")
        self.subject_cond = subject_cond
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.n_subjects = n_subjects
        self.use_recon = use_recon
        self.use_mbsd = use_mbsd
        self.use_second_diff = use_second_diff
        self.critic_cond = critic_cond
        self.use_contrastive = use_contrastive
        c = feature_dim
        self.pooled_dim = 2 * c + 16 * c + (16 * c if deep else 0)  # time-averaged h1 and h (and the deep features)

        # participant and eye-state embeddings
        if subject_cond == 'table':
            self.sub_emb = nn.Embedding(n_subjects, d_sub)
        else:
            self.ref_head = nn.Sequential(init_normal(nn.Linear(self.pooled_dim, d_sub)), nn.LayerNorm(d_sub))
        self.state_emb = nn.Embedding(2, D_STATE)

        # conditioning: FiLM on the input, or projection at the output
        if critic_cond == 'film':
            self.film = DualFiLM(c, d_sub, D_STATE)
        else:
            self.projection = ConditionProjection(d_sub, D_STATE, self.pooled_dim)
            if proj_cov:
                self.cov_dim = 4 * c * (4 * c + 1) // 2  # upper triangle of the covariance of the 4C feature maps
                self.projection_cov = ConditionProjection(d_sub, D_STATE, self.cov_dim)

        # spectral head
        if spectral_head:
            self.spec_conv1 = SameConv1d(2 * c + c * (c - 1), SPECTRAL_HEAD[0], 5)
            self.spec_conv2 = SameConv1d(SPECTRAL_HEAD[0], SPECTRAL_HEAD[1], 5, stride=2)
            self.spec_dim = SPECTRAL_HEAD[1] * math.ceil(head_bins / 2)
            self.spec_final = init_normal(nn.Linear(self.spec_dim, 1))
            self.spec_projection = ConditionProjection(d_sub, D_STATE, self.spec_dim)

        # encoder
        if use_second_diff:
            self.second_diff = SecondDifference()
        self.conv1 = SameConv1d(2 * c if use_second_diff else c, 2 * c, 5)
        self.conv2 = SameConv1d(2 * c, 4 * c, 5, stride=2)
        self.conv3 = SameConv1d(4 * c, 16 * c, 5, stride=2)
        self.act = nn.LeakyReLU(0.1)

        # adversarial branch
        if use_mbsd:
            self.mbsd = MinibatchStdDev(with_grad=mbsd_grad)
        coarse_time = math.ceil(math.ceil(time_dim / 2) / 2)
        n_score_inputs = (16 * c + int(use_mbsd)) * coarse_time + 2 * c * time_dim
        if deep:
            self.deep_convs = nn.ModuleList(SameConv1d(16 * c, 16 * c, 5, stride=2) for _ in range(DEEP_LAYERS))
            deep_time = coarse_time
            for _ in range(DEEP_LAYERS):
                deep_time = math.ceil(deep_time / 2)
            n_score_inputs += 16 * c * deep_time
        self.final = init_normal(nn.Linear(n_score_inputs, 1))

        # reconstruction branch
        if use_recon:
            self.recon_conv1 = SameConv1d(16 * c, 16 * c, 3)
            self.recon_conv2 = SameConv1d(16 * c, 4 * c, 3)
            self.recon_out = SameConv1d(4 * c, c, 3)

        # contrastive head
        if use_contrastive:
            self.con_fc1 = init_normal(nn.Linear(self.pooled_dim, self.pooled_dim))
            self.con_fc2 = init_normal(nn.Linear(self.pooled_dim, 64))

    def embed_refs(self, refs):
        """Participant embedding (B, d_sub) from reference segments (B, K, T, C), through the critic's encoder."""
        h1, h = self.encode_features(refs.flatten(0, 1))
        pooled = self.pooled_features(h1, h).view(*refs.shape[:2], -1).mean(1)
        return _float32(self.ref_head, pooled)

    def _embeddings(self, sub, state, subj_emb, refs=None):
        if subj_emb is None:
            subj_emb = self.embed_refs(refs) if self.subject_cond == 'encoder' else self.sub_emb(sub.view(-1))
        return subj_emb, self.state_emb(state.view(-1))

    def encode_features(self, x, sub=None, state=None, subj_emb=None, with_second_layer=False, film=True):
        """Encoder features of x (B, T, C): the first layer h1 and the last layer h. film=False skips the FiLM step."""
        x = x.transpose(1, 2)
        if self.critic_cond == 'film' and film:
            x = self.film(x, *self._embeddings(sub, state, subj_emb))
        if self.use_second_diff:
            x = torch.cat([x, self.second_diff(x)], dim=1)
        h1 = self.act(self.conv1(x))    # (B, 2C, T)
        h2 = self.act(self.conv2(h1))   # (B, 4C, T/2)
        h = self.act(self.conv3(h2))    # (B, 16C, T/4)
        return (h1, h2, h) if with_second_layer else (h1, h)

    def deep_features(self, h):
        """Output of the extra layers of the deep critic (B, 16C, T/64), in full precision."""
        def layers(h):
            for conv in self.deep_convs:
                h = self.act(conv(h))
            return h
        return _float32(layers, h)

    def pooled_features(self, h1, h, top=None):
        """Time-averaged feature maps of the first and the last layer (B, 18C; 34C for the deep critic)."""
        parts = [h1.mean(dim=2), h.mean(dim=2)]
        if self.deep:
            parts.append((self.deep_features(h) if top is None else top).mean(dim=2))
        return torch.cat(parts, dim=1)

    def score_from_features(self, h1, h, top=None):
        parts = [(self.mbsd(h) if self.use_mbsd else h).flatten(1), h1.flatten(1)]
        if self.deep:
            parts.append((self.deep_features(h) if top is None else top).flatten(1))
        return _float32(self.final, torch.cat(parts, dim=1))

    def projection_score(self, h1, h, sub=None, state=None, subj_emb=None, refs=None, top=None):
        subj_emb, state_emb = self._embeddings(sub, state, subj_emb, refs)
        return _float32(lambda s, e, p: (self.projection(s, e) * p).sum(dim=1, keepdim=True),
                        subj_emb, state_emb, self.pooled_features(h1, h, top))

    def second_order_features(self, h2):
        """Covariance over time of the second layer's feature maps, relative to its mean diagonal (upper triangle)."""
        def features(h):
            h = h - h.mean(dim=2, keepdim=True)
            cov = h @ h.transpose(1, 2) / h.shape[2]
            size = cov.diagonal(dim1=1, dim2=2).mean(dim=1)
            # floor of 1% of the batch average: a segment that is almost zero gives zeros, not amplified noise
            cov = cov / (size + 0.01 * size.mean().detach() + 1e-20)[:, None, None]
            iu = torch.triu_indices(cov.shape[1], cov.shape[1], device=cov.device)
            return cov[:, iu[0], iu[1]]
        return _float32(features, h2)

    def projection_cov_score(self, h2, sub=None, state=None, subj_emb=None, refs=None):
        subj_emb, state_emb = self._embeddings(sub, state, subj_emb, refs)
        return _float32(lambda s, e, c: (self.projection_cov(s, e) * c).sum(dim=1, keepdim=True),
                        subj_emb, state_emb, self.second_order_features(h2))

    def spectral_head_features(self, x):
        """Features of the spectral head for the segments x (B, T, C): (B, spec_dim), float32."""
        def features(x):
            return self.act(self.spec_conv2(self.act(self.spec_conv1(segment_spectra(x, self.head_bins))))).flatten(1)
        return _float32(features, x)

    def spectral_score(self, x, sub=None, state=None, subj_emb=None, refs=None):
        """Share of the spectral head in the score (B, 1): an unconditional and a projection term."""
        subj_emb, state_emb = self._embeddings(sub, state, subj_emb, refs)
        return _float32(lambda f, s, e: self.spec_final(f) + (self.spec_projection(s, e) * f).sum(dim=1, keepdim=True),
                        self.spectral_head_features(x), subj_emb, state_emb)

    def contrastive_embedding(self, h1, h):
        """Unit-norm embedding (B, 64) of the pooled features, for the contrastive loss."""
        if not self.use_contrastive:
            raise RuntimeError('This critic was built without the contrastive head (use_contrastive=False).')
        z = _float32(self.con_fc2, self.act(self.con_fc1(self.pooled_features(h1, h))))
        return z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def reconstruct_from_features(self, h):
        """Decode h back to a signal (B, T, C)."""
        if not self.use_recon:
            raise RuntimeError('This critic was built without the reconstruction branch (use_recon=False).')
        x = self.act(self.recon_conv1(F.interpolate(h, scale_factor=2, mode='nearest')))
        x = self.act(self.recon_conv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        return _float32(self.recon_out, x).transpose(1, 2)

    def forward(self, x, sub=None, state=None, subj_emb=None, refs=None):
        """Critic score (B, 1), float32. The participant is given by `sub`, `subj_emb` or `refs`."""
        h1, h2, h = self.encode_features(x, sub, state, subj_emb, with_second_layer=True)
        top = self.deep_features(h) if self.deep else None
        score = self.score_from_features(h1, h, top)
        if self.critic_cond == 'projection':
            score = score + self.projection_score(h1, h, sub, state, subj_emb, refs, top)
            if self.proj_cov:
                score = score + self.projection_cov_score(h2, sub, state, subj_emb, refs)
        if self.spectral_head:  # spectral head: its own unconditional and projection terms
            score = score + self.spectral_score(x, sub, state, subj_emb, refs)
        return score


class Generator(nn.Module):
    """Generator: random input -> EEG segment (B, T, C), conditioned on participant and eye state."""

    def __init__(self,
                 time_dim,
                 feature_dim,
                 latent_dim,
                 n_subjects,
                 use_dilation=True,       # dilated convolutions (arch='noise': without them every dilation is 1)
                 subject_cond='table',    # 'table' (one embedding per participant) or 'encoder' (from reference segments)
                 arch='noise',            # 'noise' (noise sequence, dilated convolutions) or 'latent' (latent vector and upsampling)
                 spectral_filter=False,   # the last layer is a participant-specific linear filter
                 d_sub=D_SUB,
                 slow_noise=True):        # arch='noise': C of the 3C noise channels are smoothed over 2 s
        super().__init__()
        self.d_sub = d_sub
        self.slow_noise = slow_noise
        if arch not in ('latent', 'noise'):
            raise ValueError(f"arch must be 'latent' or 'noise', got {arch!r}")
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.n_subjects = n_subjects
        self.use_dilation = use_dilation
        self.subject_cond = subject_cond
        self.arch = arch
        self.spectral_filter = spectral_filter
        c = feature_dim
        filter_reach = FILTER_TAPS // 2 if spectral_filter else 0

        # participant and eye-state embeddings
        if subject_cond == 'table':
            self.sub_emb = nn.Embedding(n_subjects, d_sub)
        else:
            self.ref_encoder = ParticipantEncoder(c, d_sub)
        self.state_emb = nn.Embedding(2, D_STATE)
        self.act = nn.LeakyReLU(0.2)

        if arch == 'latent':
            # latent vector -> T/4 time steps -> two upsampling stages
            self.stack_reach = 0
            self.margin = 4 * math.ceil(filter_reach / 4)
            # with 8 channels and 512 samples: a 4096-unit projection reshaped to 128 time steps x 32 filters
            self.base_time = (time_dim + 2 * self.margin) // 4
            self.base_filters = 4 * c
            self.fc = init_normal(nn.Linear(latent_dim, self.base_time * self.base_filters))
            self.film = DualFiLM(self.base_filters, d_sub, D_STATE)
            self.conv_a = SameConv1d(self.base_filters, 8 * c, 3)
            self.affine_a = ChannelAffine(8 * c)
            self.conv_b = SameConv1d(8 * c, 8 * c, 3)
            self.affine_b = ChannelAffine(8 * c)
            self.to_channels = SameConv1d(8 * c, c, 3)
            if use_dilation:
                self.dil_conv1 = SameConv1d(c, c, 3, dilation=2)
                self.dil_conv2 = SameConv1d(c, c, 3, dilation=4)
            if spectral_filter:  # centres the signal after the last nonlinearity
                self.out_affine = ChannelAffine(c)
            else:
                self.out_film = DualFiLM(c, d_sub, D_STATE)
        else:
            # noise sequence -> dilated convolutions, FiLM before each
            self.n_fast = 2 * c     # white noise channels
            self.n_slow = c         # slow noise channels
            self.width = 8 * c
            dilations = NOISE_DILATIONS if use_dilation else (1,) * len(NOISE_DILATIONS)
            self.stack_reach = sum(dilations)  # samples at each end that see the padding
            self.margin = self.stack_reach + filter_reach
            self.noise_in = SameConv1d(self.n_fast + self.n_slow, self.width, 1)
            self.films = nn.ModuleList(DualFiLM(self.width, d_sub, D_STATE) for _ in dilations)
            self.blocks = nn.ModuleList(SameConv1d(self.width, self.width, 3, dilation=d) for d in dilations)
            self.to_channels = SameConv1d(self.width, c, 1)
            # first and last layer keep the scale of their input: training starts from a signal of unit variance
            nn.init.normal_(self.noise_in.weight, std=(self.n_fast + self.n_slow) ** -0.5)
            nn.init.normal_(self.to_channels.weight, std=self.width ** -0.5)
            # kernel that smooths the slow noise channels (2-s Hann window, unit variance)
            taper = torch.hann_window(SLOW_TAPS, periodic=False)
            self.register_buffer('slow_kernel', (taper / taper.pow(2).sum().sqrt()).view(1, 1, -1), persistent=False)
        if spectral_filter:
            self.out_filter = SpectralFilter(c, d_sub, D_STATE, FILTER_TAPS)

    @staticmethod
    def _upsample(x):
        return F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

    def sample_latent(self, n, randn=torch.randn, length=None):
        """Random input of n segments: (n, latent_dim) for arch='latent', (n, 3C, length + 2 * margin) for arch='noise'."""
        device = self.state_emb.weight.device
        if self.arch == 'latent':
            if length not in (None, self.time_dim):
                raise ValueError("arch='latent' generates segments of a fixed length.")
            return randn(n, self.latent_dim).to(device)
        total = (self.time_dim if length is None else length) + 2 * self.margin
        # white noise channels
        fast = randn(n, self.n_fast, total).to(device)
        if not self.slow_noise:
            return torch.cat([fast, randn(n, self.n_slow, total).to(device)], dim=1)
        # slow channels: white noise smoothed over 2 s
        slow = randn(n, self.n_slow, total + SLOW_TAPS - 1).to(device)
        slow = F.conv1d(slow, self.slow_kernel.to(slow.dtype).expand(self.n_slow, 1, -1), groups=self.n_slow)
        return torch.cat([fast, slow], dim=1)

    def embed_refs(self, refs):
        """Participant embedding (B, d_sub) from reference segments (B, K, T, C)."""
        return self.ref_encoder(refs)

    def forward(self, z, sub=None, state=None, subj_emb=None, refs=None):
        """Generate segments from the random input z (see `sample_latent`). `subj_emb` (B, d_sub) replaces `sub` and `refs`."""
        if subj_emb is None:
            subj_emb = self.embed_refs(refs) if self.subject_cond == 'encoder' else self.sub_emb(sub.view(-1))
        state_emb = self.state_emb(state.view(-1))
        if self.arch == 'latent':
            x = self.act(self.fc(z)).view(len(z), self.base_time, self.base_filters).transpose(1, 2)
            x = self.film(x, subj_emb, state_emb)
            x = self.act(self.affine_a(self.conv_a(self._upsample(x))))
            x = self.act(self.affine_b(self.conv_b(self._upsample(x))))
            x = self.to_channels(x)
            if self.use_dilation:
                x = self.act(self.dil_conv2(self.act(self.dil_conv1(x))))
            if self.spectral_filter:
                x = self.out_affine(x)
        else:
            h = self.noise_in(z)
            # residual dilated convolutions, FiLM before each
            for film, conv in zip(self.films, self.blocks):
                h = h + conv(self.act(film(h, subj_emb, state_emb)))
            x = self.to_channels(self.act(h))
            x = x[:, :, self.stack_reach:x.shape[2] - self.stack_reach]  # drop the samples that saw the padding
        if self.spectral_filter:
            x = self.out_filter(x, subj_emb, state_emb)
        elif self.arch == 'latent':
            x = self.out_film(x, subj_emb, state_emb)
        return x.transpose(1, 2).float()


# The defaults are the model of the paper (Section 2.2).
class FiLMGAN(nn.Module):
    """Generator and critic together. `config` holds the constructor arguments and is saved in checkpoints."""

    def __init__(self,
                 time_dim=512,
                 feature_dim=8,
                 latent_dim=128,
                 n_subjects=1,
                 use_recon=True,
                 use_mbsd=True,
                 use_second_diff=True,
                 use_dilation=True,
                 critic_cond='projection',
                 use_contrastive=False,
                 subject_cond='table',
                 recon_mask='channel',    # 'channel' (whole channels masked) or 'time' (spans masked in all channels)
                 proj_cov=False,
                 gen_arch='noise',
                 spectral_filter=False,
                 deep_critic=False,
                 spectral_head=True,
                 d_sub=D_SUB,
                 head_bins=HEAD_BINS,
                 slow_noise=True,
                 mbsd_grad=False):
        super().__init__()
        if recon_mask not in ('time', 'channel'):
            raise ValueError(f"recon_mask must be 'time' or 'channel', got {recon_mask!r}")
        self.config = dict(time_dim=time_dim,
                           feature_dim=feature_dim,
                           latent_dim=latent_dim,
                           n_subjects=n_subjects,
                           use_recon=use_recon,
                           use_mbsd=use_mbsd,
                           use_second_diff=use_second_diff,
                           use_dilation=use_dilation,
                           critic_cond=critic_cond,
                           use_contrastive=use_contrastive,
                           subject_cond=subject_cond,
                           recon_mask=recon_mask,
                           proj_cov=proj_cov,
                           gen_arch=gen_arch,
                           spectral_filter=spectral_filter,
                           deep_critic=deep_critic,
                           spectral_head=spectral_head,
                           d_sub=d_sub,
                           head_bins=head_bins,
                           slow_noise=slow_noise,
                           mbsd_grad=mbsd_grad)
        for key, value in self.config.items():
            setattr(self, key, value)
        self.generator = Generator(time_dim,
                                   feature_dim,
                                   latent_dim,
                                   n_subjects,
                                   use_dilation=use_dilation,
                                   subject_cond=subject_cond,
                                   arch=gen_arch,
                                   spectral_filter=spectral_filter,
                                   d_sub=d_sub,
                                   slow_noise=slow_noise)
        self.critic = Critic(time_dim,
                             feature_dim,
                             n_subjects,
                             use_recon=use_recon,
                             use_mbsd=use_mbsd,
                             use_second_diff=use_second_diff,
                             critic_cond=critic_cond,
                             use_contrastive=use_contrastive,
                             subject_cond=subject_cond,
                             proj_cov=proj_cov,
                             deep=deep_critic,
                             spectral_head=spectral_head,
                             d_sub=d_sub,
                             head_bins=head_bins,
                             mbsd_grad=mbsd_grad)

    def forward(self, x, sub=None, state=None):
        return self.critic(x, sub, state)


# Losses

def gradient_penalty(critic, real, fake, sub, state, eps=None, subj_emb=None):
    """WGAN-GP penalty (Gulrajani et al., 2017) on random interpolations of real and generated segments."""
    if eps is None:
        eps = torch.rand(len(real), 1, 1, device=real.device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(interpolated, sub, state, subj_emb=subj_emb)
    gradients = torch.autograd.grad(score.sum(), interpolated, create_graph=True)[0]
    return ((gradients.float().flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def make_masked_batch(x, mask_ratio=0.15, max_spans=3, generator=None):
    """Mask 1 to `max_spans` time spans per segment in all channels, about `mask_ratio` of its length.
    Returns the masked batch and the mask (1 = masked), both (B, T, C).
    """
    B, T, C = x.shape
    device = x.device
    total_masked = max(1, int(T * mask_ratio))
    min_span = max(4, T // 32)

    def randint(low, high):  # integers in [low, high), with bounds per sample
        return low + torch.floor(torch.rand(B, device=device, generator=generator) * (high - low)).long()

    time = torch.arange(T, device=device)[None, :]
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    remaining = torch.full((B,), total_masked, dtype=torch.long, device=device)
    n_spans = torch.randint(1, max_spans + 1, (B,), device=device, generator=generator)
    for span_idx in range(max_spans):
        active = (span_idx < n_spans) & (remaining > 0)
        spans_left = (n_spans - span_idx).clamp(min=1)
        span_len = (remaining // spans_left).clamp(min=min_span)
        jitter = (span_len // 3).clamp(min=1)
        low = (span_len - jitter).clamp(min=min_span)
        high = (span_len + jitter + 1).clamp(max=T)
        span_len = torch.minimum(randint(low, high), remaining).clamp(max=T)
        start = randint(torch.zeros_like(span_len), T - span_len + 1)
        span = (time >= start[:, None]) & (time < (start + span_len)[:, None]) & active[:, None]
        mask = mask | span
        remaining = torch.where(active, remaining - span_len, remaining)

    mask = mask[:, :, None].expand(B, T, C).to(x.dtype)
    return x * (1.0 - mask), mask


def make_channel_mask(x, max_channels=2, generator=None):
    """Mask 1 to `max_channels` whole channels per segment. Returns the mask (1 = masked), (B, T, C)."""
    B, T, C = x.shape
    n_masked = torch.randint(1, max_channels + 1, (B,), device=x.device, generator=generator)
    order = torch.rand(B, C, device=x.device, generator=generator).argsort(dim=1).argsort(dim=1)  # random rank per channel
    return (order < n_masked[:, None])[:, None, :].expand(B, T, C).to(x.dtype)


def reconstruction_loss(critic, x, sub, state, mask=None):
    """Mean absolute error of the critic's reconstruction over the masked part of real segments."""
    if mask is None:
        x_masked, mask = make_masked_batch(x)
    else:
        x_masked = x * (1.0 - mask)
    _, h = critic.encode_features(x_masked, sub, state)
    recon = critic.reconstruct_from_features(h)[:, :x.shape[1]]
    return ((recon - x).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def contrastive_loss(critic, x, x2, sub, state, temperature=0.1, recording=None):
    """Same-recording contrastive loss on real segments (supervised-contrastive form of InfoNCE).
    x and x2 are two segments of the same recording; the segments of other recordings in the batch are negatives.
    Returns the loss and the accuracy (is the most similar other segment from the same recording?).
    """
    h1, h = critic.encode_features(torch.cat([x, x2], dim=0))
    z = critic.contrastive_embedding(h1, h)  # (2B, 64), unit norm, float32
    with torch.autocast(z.device.type, enabled=False):
        logits = (z @ z.T) / temperature
        # without `recording`, a recording is the pair (sub, state)
        if recording is None:
            recording = sub.view(-1) * 2 + state.view(-1)
        recording = recording.view(-1).repeat(2)
        eye = torch.eye(len(recording), dtype=torch.bool, device=logits.device)
        positives = (recording[:, None] == recording[None, :]) & ~eye
        logits = logits.masked_fill(eye, float('-inf'))  # an anchor is never its own candidate
        log_prob = torch.log_softmax(logits, dim=1)
        loss = -(log_prob.masked_fill(~positives, 0.0).sum(1) / positives.sum(1).clamp(min=1)).mean()
        accuracy = positives.gather(1, logits.argmax(1, keepdim=True)).float().mean()
    return loss, accuracy.detach()
