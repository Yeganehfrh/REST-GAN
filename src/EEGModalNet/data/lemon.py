"""Loading the LEMON resting-state recordings used for training and evaluation."""
from typing import Dict, List

import numpy as np
import torch
import xarray as xr
from meegkit import dss
from scipy.signal import butter, ellip, sosfiltfilt

from ..preprocessing.preprocessing import preprocess_data, zscore_recordings

LEMON_BLOCK = 7680  # each LEMON recording joins eight 60-s blocks (128 Hz) with a step at every join


def highpass_filter(x, cutoff=0.5, sfreq=128, block_len=None):
    """Zero-phase 4th-order Butterworth high-pass along time; with `block_len`, each block separately."""
    sos = butter(4, cutoff, btype='high', fs=sfreq, output='sos')
    if block_len is None:
        return sosfiltfilt(sos, x, axis=-1)
    if x.shape[-1] % block_len:
        raise ValueError(f'{x.shape[-1]} samples are not a whole number of {block_len}-sample blocks.')
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_len, block_len)
    return sosfiltfilt(sos, blocks, axis=-1).reshape(x.shape)


def remove_line_noise_dss(x, fline=50, sfreq=128):
    """Remove the line-noise component with DSS (de Cheveigné, 2010), one spatial filter for all recordings."""
    y, _ = dss.dss_line(x.T, fline=fline, sfreq=sfreq, nremove=1)
    return y.T


def lowpass_filter(x, cutoff=45.0, sfreq=128, block_len=None):
    """Zero-phase elliptic low-pass along time (order 8); with `block_len`, each block separately."""
    sos = ellip(8, 0.1, 60, cutoff, btype='low', fs=sfreq, output='sos')
    if block_len is None:
        return sosfiltfilt(sos, x, axis=-1)
    if x.shape[-1] % block_len:
        raise ValueError(f'{x.shape[-1]} samples are not a whole number of {block_len}-sample blocks.')
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_len, block_len)
    return sosfiltfilt(sos, blocks, axis=-1).reshape(x.shape)


def _line_noise_step(line_noise):
    """'dss', 'lowpass' or None, from the values that `line_noise` may take (True means 'dss')."""
    if line_noise in (True, 'dss'):
        return 'dss'
    if line_noise in (False, None, 'none'):
        return None
    if line_noise == 'lowpass':
        return 'lowpass'
    raise ValueError(f"line_noise must be 'dss', 'lowpass' or None, got {line_noise!r}")


def preprocess_recordings(x, filter_first=True, block_len=None, high_pass=True, line_noise='lowpass', clip=20.0):
    """Preprocess recordings x (recordings, channels, time).
    filter_first=True (the preprocessing of the paper, with line_noise='lowpass'): filter each 60-s block, then
    z-score. filter_first=False: z-score first, then filter the whole recordings.
    """
    step = _line_noise_step(line_noise)
    if not filter_first:  # whole recordings are filtered: block_len is not used
        if step == 'lowpass':
            raise ValueError("line_noise='lowpass' belongs to the revised pipeline: use filter_first=True.")
        x = zscore_recordings(x, clip=clip)
        x = highpass_filter(x) if high_pass else x
        return remove_line_noise_dss(x) if step == 'dss' else x
    # Step 1: high-pass filter at 0.5 Hz
    x = highpass_filter(x, block_len=block_len) if high_pass else x
    # Step 2: line noise (DSS, or a low-pass filter at 45 Hz)
    if step == 'dss':
        x = remove_line_noise_dss(x)
    elif step == 'lowpass':
        x = lowpass_filter(x, block_len=block_len)
    # Step 3: z-score each channel over the whole recording and clamp values beyond `clip` standard deviations
    return zscore_recordings(x, clip=clip)


def _resolve_condition_path(data_path: str, target_condition: str) -> str:
    """Return a path matching target_condition by swapping EC/EO tokens when needed."""
    target = target_condition.upper()
    other = 'EO' if target == 'EC' else 'EC'
    if target in data_path:
        return data_path
    if other in data_path:
        return data_path.replace(other, target)
    return data_path


def load_data(data_path: str,
              channels: List[str] | str = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
              n_subjects: int = 202,
              condition: str | None = None,
              exclude_sub_ids=None,
              preprocess=True,
              highpass=True,
              remove_line_noise='lowpass',
              filter_first=True,
              block_len=LEMON_BLOCK) -> Dict:
    """Load the LEMON recordings and preprocess them (see `preprocess_recordings`).
    Returns x (recordings, channels, time), sub (participant index), state (0 = eyes closed, 1 = eyes open)
    and `subjects`, the participant id of each index.
    """
    condition = condition.lower() if isinstance(condition, str) else None
    if condition not in {None, 'both', 'ec', 'eo'}:
        raise ValueError(f"Unsupported condition={condition!r}. Use one of: 'both', 'EC', 'EO', or None.")

    if condition in {'ec', 'eo'}:
        data_path = _resolve_condition_path(data_path, condition)
    xarray = xr.open_dataarray(data_path, engine='h5netcdf')

    if n_subjects < xarray.sizes['subject']:
        xarray = xarray.sel(subject=xarray.subject[:n_subjects])

    if condition == 'both':
        if 'EC' in data_path:
            data_path_2 = data_path.replace('EC', 'EO')
        elif 'EO' in data_path:
            data_path_2 = data_path.replace('EO', 'EC')
        else:
            raise ValueError("For condition='both', data_path must include either 'EC' or 'EO'.")
        xarray_2 = xr.open_dataarray(data_path_2, engine='h5netcdf')
        if n_subjects < xarray_2.sizes['subject']:
            xarray_2 = xarray_2.sel(subject=xarray_2.subject[:n_subjects])
        xarray_2 = xarray_2.rename({"time": "timestep"})  # the two files name their time dimension differently
        if list(xarray.subject.values) != list(xarray_2.subject.values):
            raise ValueError("EC and EO files do not list the same subjects in the same order.")
        xarray = xr.concat([xarray, xarray_2], dim='subject')

    if channels != 'all':
        xarray = xarray.sel(channel=channels)

    if exclude_sub_ids is not None:
        exclude_sub_ids = [str(s) for s in exclude_sub_ids]
        missing = sorted(set(exclude_sub_ids) - set(xarray.subject.values.astype(str)))
        if missing:
            raise ValueError(f"Held-out participants not found in the data: {missing}")
        keep = ~np.isin(xarray.subject.values.astype(str), exclude_sub_ids)
        xarray = xarray.isel(subject=np.flatnonzero(keep))

    subject_ids = [str(s) for s in xarray.subject.values]
    x = xarray.to_numpy()

    if preprocess == 'legacy':  # baseline correction, RobustScaler and StandardScaler
        if filter_first:
            raise ValueError("preprocess='legacy' reproduces the old pipeline: use filter_first=False.")
        if _line_noise_step(remove_line_noise) == 'lowpass':
            raise ValueError("preprocess='legacy' reproduces the old pipeline, which had no low-pass.")
        x = preprocess_data(x, sampling_rate=128)
        x = highpass_filter(x) if highpass else x
        x = remove_line_noise_dss(x) if _line_noise_step(remove_line_noise) == 'dss' else x
    elif preprocess:
        x = preprocess_recordings(x, filter_first=filter_first, block_len=block_len, high_pass=highpass,
                                  line_noise=remove_line_noise)
    else:  # no normalisation
        x = highpass_filter(x, block_len=block_len if filter_first else None) if highpass else x
        step = _line_noise_step(remove_line_noise)
        if step == 'dss':
            x = remove_line_noise_dss(x)
        elif step == 'lowpass':
            x = lowpass_filter(x, block_len=block_len if filter_first else None)

    x = torch.tensor(x.copy(), dtype=torch.float32)
    if condition == 'both':
        if x.shape[0] % 2 != 0:
            raise ValueError("Expected an even number of subjects for condition='both'.")
        n_per_condition = x.shape[0] // 2
        sub = torch.arange(n_per_condition, dtype=torch.long).repeat(2)
        state = torch.cat([torch.zeros(n_per_condition, dtype=torch.long),
                           torch.ones(n_per_condition, dtype=torch.long)])
        subject_ids = subject_ids[:n_per_condition]
    else:
        sub = torch.arange(x.shape[0], dtype=torch.long)
        state = torch.full((x.shape[0],), 0 if condition != 'eo' else 1, dtype=torch.long)

    return {'x': x, 'sub': sub, 'state': state, 'subjects': subject_ids}
