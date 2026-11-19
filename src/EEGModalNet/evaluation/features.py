"""Features of one recording, computed from its 4-s segments (Section 2.4.1).

A recording is an array (n_segments, n_times, n_channels). Every feature is computed within segments and then
combined over segments, so the joins between segments do not affect it.
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis

FS = 128
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}  # [lo, hi) Hz


def _fourier(seg):
    """Hann-windowed FFT of each segment: frequencies and coefficients (n_segments, n_freqs, n_channels)."""
    n = seg.shape[1]
    window = np.hanning(n)[None, :, None]
    return np.fft.rfftfreq(n, 1 / FS), np.fft.rfft((seg - seg.mean(1, keepdims=True)) * window, axis=1)


def band_power(seg):
    """Fractional band power per channel and band (share of the 1-45 Hz power)."""
    freqs, coef = _fourier(seg)
    psd = (np.abs(coef) ** 2).mean(0)  # (n_freqs, n_channels)
    total = psd[(freqs >= 1) & (freqs < 45)].sum(0)
    return np.concatenate([psd[(freqs >= lo) & (freqs < hi)].sum(0) / total for lo, hi in BANDS.values()])


def coherence(seg):
    """Magnitude-squared coherence per channel pair and band."""
    freqs, coef = _fourier(seg)
    # auto- and cross-spectra are averaged over segments and summed over the band before the ratio is taken
    cross = np.einsum('sfi,sfj->fij', coef, coef.conj()) / len(seg)
    iu = np.triu_indices(seg.shape[2], k=1)
    out = []
    for lo, hi in BANDS.values():
        band = cross[(freqs >= lo) & (freqs < hi)].sum(0)
        power = np.real(np.diag(band))
        out.append((np.abs(band) ** 2 / np.outer(power, power))[iu])
    return np.concatenate(out)


def hjorth(seg):
    """Log activity, mobility and complexity per channel."""
    d1, d2 = np.diff(seg, axis=1), np.diff(seg, n=2, axis=1)
    v0, v1, v2 = seg.var(1), d1.var(1), d2.var(1)
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return np.concatenate([np.log(v0).mean(0), mobility.mean(0), complexity.mean(0)])


def log_cov(seg, window=256):
    """Mean and SD over 2-s windows of the log-covariance descriptor (upper triangle, log-Euclidean)."""
    n_seg, n_times, n_ch = seg.shape
    win = seg[:, :(n_times // window) * window].reshape(-1, window, n_ch)
    win = win - win.mean(1, keepdims=True)
    cov = np.einsum('wti,wtj->wij', win, win) / window
    cov = cov + 1e-6 * np.trace(cov, axis1=1, axis2=2)[:, None, None] / n_ch * np.eye(n_ch)
    values, vectors = np.linalg.eigh(cov)
    logc = (vectors * np.log(values)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
    iu = np.triu_indices(n_ch)
    desc = logc[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return np.concatenate([desc.mean(0), desc.std(0)])


def hjorth_shape(seg):
    """Hjorth mobility and complexity per channel (no activity)."""
    return hjorth(seg)[seg.shape[2]:]


def log_corr(seg, window=256):
    """Mean and SD over 2-s windows of the log-correlation descriptor (upper triangle, log-Euclidean)."""
    n_seg, n_times, n_ch = seg.shape
    win = seg[:, :(n_times // window) * window].reshape(-1, window, n_ch)
    win = win - win.mean(1, keepdims=True)
    cov = np.einsum('wti,wtj->wij', win, win) / window
    sd = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    corr = cov / (sd[:, :, None] * sd[:, None, :]) + 1e-6 * np.eye(n_ch)
    values, vectors = np.linalg.eigh(corr)
    logc = (vectors * np.log(values)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
    iu = np.triu_indices(n_ch)
    desc = logc[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return np.concatenate([desc.mean(0), desc.std(0)])


SECOND_ORDER = ('band_power', 'coherence', 'hjorth_shape', 'log_corr')  # the four feature spaces of the paper
REGIONS = {'frontal': ('F1', 'F2'), 'central': ('C1', 'C2'), 'parietal': ('P1', 'P2'), 'occipital': ('O1', 'O2')}
REGION_FEATURES = ['activity', 'mobility', 'complexity'] + list(BANDS)


def psd(seg):
    """Power spectrum averaged over segments: frequencies (1-45 Hz, 0.25-Hz bins at 512 samples) and (n_freqs, n_channels)."""
    freqs, coef = _fourier(seg)
    keep = (freqs >= 1) & (freqs <= 45)
    return freqs[keep], (np.abs(coef) ** 2).mean(0)[keep]


def region_features(seg, channels):
    """Hjorth parameters and fractional band power averaged over the channels of each region: (n_regions, n_features)."""
    channels = list(channels)
    n = len(channels)
    per_channel = np.concatenate([hjorth(seg).reshape(3, n), band_power(seg).reshape(len(BANDS), n)])  # (8, n)
    return np.stack([per_channel[:, [channels.index(c) for c in members]].mean(1) for members in REGIONS.values()])


def alpha_peak_frequency(freqs, power, channels, posterior=('O1', 'O2', 'P1', 'P2'), fmin=8.0, fmax=13.0,
                         min_prominence=0.08, min_height_over_median=0.05, edge_buffer=0.3):
    """Individual alpha frequency from the posterior spectrum (dB), or NaN without a clear peak in 8-13 Hz."""
    idx = [list(channels).index(c) for c in posterior]
    band = (freqs >= fmin) & (freqs <= fmax)
    y = 10 * np.log10(np.clip(power[:, idx].mean(1), 1e-12, None))[band]
    f = freqs[band]
    # smooth the spectrum with a Savitzky-Golay filter
    window = min(9, len(y) - (1 - len(y) % 2))
    if window > 3:
        y = savgol_filter(y, window_length=window, polyorder=3, mode='interp')
    # most prominent peak, at least `edge_buffer` Hz from the band edges
    peaks, props = find_peaks(y, prominence=min_prominence, height=np.median(y) + min_height_over_median)
    valid = (f[peaks] >= fmin + edge_buffer) & (f[peaks] <= fmax - edge_buffer)
    if not valid.any():
        return np.nan
    return float(f[peaks[valid][np.argmax(props['prominences'][valid])]])


def amplitude_stats(seg):
    """RMS and excess kurtosis per channel."""
    flat = np.asarray(seg, dtype=np.float64).reshape(-1, seg.shape[2])
    return {'rms': np.sqrt((flat ** 2).mean(0)), 'kurtosis': kurtosis(flat, axis=0)}


HIGHER_ORDER = ('kurtosis', 'alpha_env_cv', 'alpha_sd_segments')
ENVELOPE_CV_GAUSSIAN = float(np.sqrt(4 / np.pi - 1))  # CV of the envelope of a narrow-band Gaussian process (Rayleigh)


def beyond_second_order(seg, margin=64):
    """Three non-Gaussian properties per channel (Supplementary Section S5), ordered as HIGHER_ORDER.
    kurtosis: excess kurtosis of the amplitudes. alpha_env_cv: coefficient of variation of the 8-13 Hz envelope.
    alpha_sd_segments: SD across segments of the log share of 8-13 Hz in the 1-45 Hz power.
    """
    n_seg, n, n_ch = seg.shape
    freqs = np.fft.rfftfreq(n, 1 / FS)
    alpha = (freqs >= 8) & (freqs < 13)
    coef = np.fft.rfft(seg - seg.mean(1, keepdims=True), axis=1)
    analytic = np.zeros((n_seg, n, n_ch), dtype=complex)
    analytic[:, :len(freqs)][:, alpha] = 2 * coef[:, alpha]
    envelope = np.abs(np.fft.ifft(analytic, axis=1))[:, margin:n - margin]
    power = np.abs(coef) ** 2
    share = np.log(power[:, alpha].sum(1) / power[:, (freqs >= 1) & (freqs < 45)].sum(1))
    return np.concatenate([kurtosis(seg.reshape(-1, n_ch), axis=0), envelope.std((0, 1)) / envelope.mean((0, 1)),
                           share.std(0)])


def higher_order_summary(values, channels, posterior=('O1', 'O2', 'P1', 'P2')):
    """One value per recording and property: kurtosis over all channels, the alpha properties over the posterior channels."""
    channels = list(channels)
    values = np.asarray(values).reshape(len(values), len(HIGHER_ORDER), len(channels))
    idx = [channels.index(c) for c in posterior]
    return np.stack([values[:, 0].mean(1), values[:, 1][:, idx].mean(1), values[:, 2][:, idx].mean(1)], axis=1)


def segment_artefacts(recordings):
    """Properties of generated segments that real EEG does not have, over all segments of a list of recordings.
    mean_share: share of the power in the segment mean. fixed_waveform: RMS of the average standardised segment.
    position_sd: how much the power depends on the position within the segment.
    """
    a = np.concatenate([np.asarray(r, dtype=np.float64) for r in recordings])
    centred = a - a.mean(1, keepdims=True)
    z = centred / centred.std((1, 2), keepdims=True)
    profile = (z ** 2).mean((0, 2))
    return {'mean_share': float((a.mean(1) ** 2).mean() / (a ** 2).mean()),
            'fixed_waveform': float(np.sqrt((z.mean(0) ** 2).mean())),
            'waveform_noise_level': float(1 / np.sqrt(len(a))),
            'position_sd': float(profile.std() / profile.mean())}


def recording_features(seg, scale_free=True):
    """All feature sets of one recording: dict name -> 1-D vector.
    scale_free=True gives the features of the paper, which do not depend on the amplitude of a channel: band
    power, coherence, Hjorth mobility and complexity, log-correlation and the non-Gaussian properties.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if scale_free:
        return {'band_power': band_power(seg), 'coherence': coherence(seg), 'hjorth_shape': hjorth_shape(seg),
                'log_corr': log_corr(seg), 'beyond_second_order': beyond_second_order(seg)}
    return {'band_power': band_power(seg), 'coherence': coherence(seg), 'hjorth': hjorth(seg), 'log_cov': log_cov(seg)}


def join_ratio(seg):
    """Jump between consecutive segments relative to the sample-to-sample change within them (1: as smooth as elsewhere)."""
    seg = np.asarray(seg, dtype=np.float64)
    within = np.abs(np.diff(seg, axis=1)).mean()
    joins = np.abs(seg[1:, 0] - seg[:-1, -1]).mean()
    return float(joins / within)
