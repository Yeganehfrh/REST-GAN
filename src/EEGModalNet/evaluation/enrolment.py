"""Fit an embedding to a new participant (Section 2.4.3).

The weights of the model stay unchanged and no labels are used. Starting from the mean of the learned embeddings,
the embedding is optimised so that the generated EEG matches the segments of the participant.
"""
import numpy as np
import torch

from .models import generate_from_embedding, load_checkpoint
from ..models.spectral import spectral_distance, spectral_features


def _spectrum_and_corr(x):
    """x: (participants, segments, time, channels) -> log-spectra (P, 44, C) and correlation matrices (P, C, C)."""
    p, n, t, c = x.shape
    window = torch.hann_window(t)
    power = (torch.fft.rfft(x * window[None, None, :, None], dim=2).abs() ** 2).mean(1)  # (P, freq, C)
    step = t // 128                                                                      # bins per Hz
    power = power[:, step:45 * step].reshape(p, 44, step, c).mean(2)                     # 1-45 Hz in 1-Hz bins
    xc = x - x.mean(2, keepdim=True)
    cov = torch.einsum('pnti,pntj->pij', xc, xc) / (n * t)
    sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-8)
    return torch.log(power + 1e-8), cov / (sd[:, :, None] * sd[:, None, :])  # channel power is in the spectrum term


def fit_embeddings(checkpoint, recordings, state, steps=150, batch=32, lr=0.05, prior=0.01, seed=0, model=None,
                   reference=None):
    """Fit one embedding per recording. Returns the embeddings and the final losses.
    recordings: list of (n_segments, T, C) arrays of the same shape. reference: the level of the spectral loss for
    real segments; if given, the spectral loss of training is used, otherwise log spectrum and channel correlation.
    """
    model = load_checkpoint(checkpoint) if model is None else model
    generator = model.generator
    # mean and SD of the learned embeddings
    table = generator.sub_emb.weight.detach()
    mean = table.mean(0)
    sd = table.std(0) + 1e-6 if len(table) > 1 else torch.ones_like(mean)

    real = torch.as_tensor(np.stack(recordings), dtype=torch.float32)
    n = len(real)
    # targets from the segments of the participants
    with torch.no_grad():
        target_spec, target_corr = _spectrum_and_corr(real)
        targets = spectral_features(real) if reference is not None else None
    g = torch.Generator().manual_seed(seed)
    # every embedding starts at the mean of the learned embeddings
    emb = mean[None, :].repeat(n, 1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([emb], lr=lr)
    for _ in range(steps):
        # generate `batch` segments per participant and compare them with the targets
        with torch.no_grad():
            z = generator.sample_latent(n * batch, lambda *shape: torch.randn(*shape, generator=g))
        fake = generate_from_embedding(generator, emb.repeat_interleave(batch, dim=0), z, state)
        groups = fake.reshape(n, batch, *fake.shape[1:])
        if reference is not None:
            power, coherency = spectral_distance(spectral_features(groups), targets)
            fit = power / reference[0] + coherency / reference[1]
        else:
            spec, corr = _spectrum_and_corr(groups)
            fit = ((spec - target_spec) ** 2).mean((1, 2)) + ((corr - target_corr) ** 2).mean((1, 2))
        # weak prior that keeps the embedding close to the learned ones
        per_participant = fit + prior * (((emb - mean) / sd) ** 2).mean(1)
        optimizer.zero_grad()
        per_participant.sum().backward()
        optimizer.step()
    return emb.detach().numpy(), per_participant.detach().numpy()
