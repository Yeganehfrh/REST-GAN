"""Spectral features of groups of segments: the spectral loss of the generator and the input of the spectral head.

A group is a set of segments of one recording, or generated for one participant and eye state. Its
cross-spectrum is estimated with 1-s Hann windows that overlap by half (1-Hz bins).
"""
import torch

N_WINDOW = 128      # 1-s windows at 128 Hz (1-Hz bins)
HOP = 64            # the windows overlap by half
N_BINS = N_WINDOW // 2 + 1
FLOOR = 1e-4        # share of a channel's total power that is added to every bin
POWER_WEIGHT = 0.5  # coherency term: share of the weight that follows the power of the target
FISHER = True       # size of the coherency on Fisher's z scale


def cross_spectrum(x):
    """Cross-spectrum of x (groups, segments, T, C): real and imaginary part, each (groups, bins, C, C)."""
    with torch.autocast(x.device.type, enabled=False):
        # the mean is not removed: bin 0 also holds the offset of the segments
        windows = x.float().unfold(2, N_WINDOW, HOP)  # (G, N, W, C, n_window)
        coef = torch.fft.rfft(windows * torch.hann_window(N_WINDOW, device=x.device), dim=-1)
        a, b = coef.real, coef.imag  # (G, N, W, C, F)
        n = windows.shape[1] * windows.shape[2]
        real = (torch.einsum('gnwif,gnwjf->gfij', a, a) + torch.einsum('gnwif,gnwjf->gfij', b, b)) / n
        imag = (torch.einsum('gnwif,gnwjf->gfij', b, a) - torch.einsum('gnwif,gnwjf->gfij', a, b)) / n
    return real, imag


def spectral_features(x):
    """Log power (G, F, C) and coherency (G, F, pairs, 2: real, imaginary) of x (groups, segments, T, C)."""
    real, imag = cross_spectrum(x)
    power = torch.diagonal(real, dim1=2, dim2=3)  # (G, F, C)
    # floor: below it the log power is constant and the coherency goes to zero
    power = power + FLOOR * power.sum(dim=1, keepdim=True) + 1e-20
    i, j = torch.triu_indices(x.shape[3], x.shape[3], offset=1, device=x.device)
    norm = torch.sqrt(power[:, :, i] * power[:, :, j])
    coherency = torch.stack([real[:, :, i, j] / norm, imag[:, :, i, j] / norm], dim=-1)
    if FISHER:  # size of the coherency on Fisher's z scale, its direction kept
        size = coherency.pow(2).sum(-1, keepdim=True).add(1e-12).sqrt()
        coherency = coherency * torch.atanh(size.clamp(max=0.995)) / size
    return torch.log(power), coherency


def segment_spectra(x, bins=None):
    """Input of the critic's spectral head: x (B, T, C) -> (B, 2C + C(C-1), bins), float32.
    Per 1-Hz bin: the relative and the total log power of every channel and the coherency of every pair.
    """
    real, imag = (part[:, :bins] for part in cross_spectrum(x[:, None]))
    power = torch.diagonal(real, dim1=2, dim2=3)  # (B, F, C)
    total = power.sum(dim=1, keepdim=True)
    # floor of 1% of the batch average: a segment that is almost zero gives constants, not amplified noise
    total = total + 0.01 * total.mean(dim=0, keepdim=True).detach() + 1e-20
    power = power + FLOOR * total
    i, j = torch.triu_indices(x.shape[2], x.shape[2], offset=1, device=x.device)
    amplitude = torch.sqrt(power)  # the product of two powers can underflow in single precision
    norm = amplitude[:, :, i] * amplitude[:, :, j]
    unit = N_WINDOW * torch.hann_window(N_WINDOW, device=x.device).pow(2).sum() / 2  # total power of unit white noise
    features = [torch.log(power / total * power.shape[1]) / 3, torch.log(total / unit).expand_as(power),
                real[:, :, i, j] / norm, imag[:, :, i, j] / norm]
    return torch.cat(features, dim=2).transpose(1, 2)


def spectral_distance(features, targets):
    """Squared difference from the targets per group: the log-power and the coherency term, each (G,)."""
    (log_power, coherency), (target_power, target_coherency) = features, targets
    error = ((coherency - target_coherency) ** 2).sum(-1)  # (G, F, pairs)
    weight = torch.full_like(error, 1.0 / error.shape[1])
    # half of the weight follows the power of the target in the two channels (geometric mean)
    if POWER_WEIGHT > 0:
        n_ch = target_power.shape[2]
        i, j = torch.triu_indices(n_ch, n_ch, offset=1, device=error.device)
        by_power = torch.exp(0.5 * (target_power[:, :, i] + target_power[:, :, j]))
        weight = (1 - POWER_WEIGHT) * weight + POWER_WEIGHT * by_power / by_power.sum(dim=1, keepdim=True)
    return ((log_power - target_power) ** 2).mean((1, 2)), (weight * error).sum(1).mean(1)


def spectral_loss(features, targets, reference):
    """Both terms relative to their value for real groups (`reference`), added: real data give 2."""
    power, coherency = (d.mean() for d in spectral_distance(features, targets))
    return power / reference[0] + coherency / reference[1], power, coherency


def recording_targets(x, seg_len=512, block_len=None, chunk=16):
    """Spectral features of whole recordings x (recordings, channels, time), cut into segments within blocks."""
    n_rec, n_ch, length = x.shape
    block = length if block_len is None else block_len
    per_block = block // seg_len
    out = []
    with torch.no_grad():
        for start in range(0, n_rec, chunk):
            part = x[start:start + chunk].reshape(-1, n_ch, length // block, block)[..., :per_block * seg_len]
            part = part.reshape(len(part), n_ch, -1, seg_len).permute(0, 2, 3, 1)  # (r, segments, T, C)
            out.append(spectral_features(part))
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def spectral_embeddings(targets, reference, sub, state, n_subjects, dim):
    """Starting values of the participant embeddings: scores on the principal components of the spectral features."""
    log_power, coherency = targets
    # both kinds of features in units of their level for real groups, as in the spectral loss
    feats = torch.cat([log_power.flatten(1) / (reference[0] * log_power[0].numel()) ** 0.5,
                       coherency.flatten(1) / (reference[1] * coherency[0].numel() / 2) ** 0.5], dim=1)
    blocks = []
    for value in torch.unique(state).tolist():
        rows, f = sub[state == value], feats[state == value]
        block = f.mean(0).repeat(n_subjects, 1)  # a participant without this state: the mean
        # several recordings in one state (--no-state-cond): their mean
        total = torch.zeros_like(block).index_add_(0, rows, f)
        count = torch.zeros(n_subjects, dtype=f.dtype, device=f.device).index_add_(0, rows, torch.ones_like(rows, dtype=f.dtype))
        present = count > 0
        block[present] = total[present] / count[present, None]
        blocks.append(block)
    per_subject = torch.cat(blocks, dim=1).cpu()  # on the CPU
    # scores on the first k principal components, with unit variance
    k = min(dim, n_subjects - 1)
    u, _, _ = torch.linalg.svd(per_subject - per_subject.mean(0), full_matrices=False)
    return (u[:, :k] * (n_subjects - 1) ** 0.5).to(log_power.device)
