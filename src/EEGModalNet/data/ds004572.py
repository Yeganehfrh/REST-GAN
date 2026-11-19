"""ds004572 (OTKA study 1): eyes-closed resting baseline recordings, raw and only downsampled."""
import numpy as np
import mne
import xarray as xr

MODEL_CHANNELS = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2']


def create_baseline_dataset(bids_root,
                            output_path,
                            subject_ids,
                            tasks=('baseline1', 'baseline2'),
                            channels=MODEL_CHANNELS,
                            sfreq=128,
                            n_samples=38400):
    """Create a dataset of the raw baseline (eyes-closed rest) recordings, only downsampled to 128 Hz, in volts."""
    data = np.full((len(subject_ids), len(tasks), len(channels), n_samples), np.nan)
    for i, sub in enumerate(subject_ids):
        for j, task in enumerate(tasks):
            raw = mne.io.read_raw_brainvision(f'{bids_root}/{sub}/ses-01/eeg/{sub}_ses-01_task-{task}_eeg.vhdr',
                                              preload=True, verbose=0)
            x = raw.pick(channels).reorder_channels(channels).resample(sfreq, verbose=0).get_data()
            if x.shape[-1] < n_samples:
                raise ValueError(f'{sub} {task} has only {x.shape[-1]} samples at {sfreq} Hz')
            data[i, j] = x[:, :n_samples]
    dataset = xr.DataArray(data, dims=['subject', 'task', 'channel', 'time'],
                           coords={'subject': list(subject_ids), 'task': list(tasks), 'channel': list(channels),
                                   'time': np.arange(n_samples)},
                           attrs={'units': 'V', 'sfreq': sfreq, 'source': 'OpenNeuro ds004572, raw BIDS',
                                  'processing': 'channel selection, mne resample to 128 Hz, first 300 s; nothing else'})

    # save
    dataset.to_netcdf(output_path, engine='h5netcdf')
    return dataset
