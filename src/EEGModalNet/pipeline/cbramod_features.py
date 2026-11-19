"""Compute CBraMod features of the segments used for decoding (Section 2.5).

Run from the repository root (a CPU is enough):

    python -m src.EEGModalNet.pipeline.cbramod_features --cbramod ../CBraMod-benchmark \\
        --edf-dir data/ds005385/raw --subjects-from data/ds005385/ds005385.nc5 --out logs/decoding/cbramod

--subjects-from can be given several times, for participants that are listed in several files.

--cbramod is a directory with the authors' code as `CBraMod/` (github.com/wjq-learning/CBraMod) and
`pretrained_weights/pretrained_weights.pth` (huggingface.co/weighting666/CBraMod).

The input is prepared as for the pretraining of the model: 200 Hz, band-pass 0.3-75 Hz, 50-Hz notch, units of
100 microvolts, patches of 1 s. One 4-s segment is given to the model as channels x 4 patches x 200 samples.

Writes one row per segment: <out>_pooled.npy (averaged over channels and patches, 200 values),
<out>_channels.npy (averaged over patches for each channel, 1,600 values), with --full also <out>_full.npy
(every patch embedding, 6,400 values), and <out>_subjects.npy (the participant of every recording).
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

PATCH = 200  # samples per patch: 1 s at 200 Hz
UNIT = 1e6 / 100  # volts -> units of 100 microvolts, as in the pretraining of the model


def edf_path(edf_dir, subject):
    """Path of the recording of `subject` in a directory that holds the recordings side by side."""
    return os.path.join(edf_dir, f'{subject}_ses-1_task-EyesClosed_acq-pre_eeg.edf')


def four_second_segments(epochs):
    """(participants, 2-s epochs, channels, 2 patches, 200) -> (participants, 4-s segments, channels, 4 patches, 200).

    Two consecutive epochs make one segment; within a segment every channel keeps its own four seconds in order.
    """
    n, n_epochs, n_ch, per_epoch, size = epochs.shape
    n_seg = n_epochs // 2
    x = epochs[:, :2 * n_seg].reshape(n, n_seg, 2, n_ch, per_epoch, size)
    return np.ascontiguousarray(x.transpose(0, 1, 3, 2, 4, 5)).reshape(n, n_seg, n_ch, 2 * per_epoch, size)


def load_backbone(directory):
    import torch
    sys.path.insert(0, os.path.abspath(directory))
    from CBraMod.models.cbramod import CBraMod
    backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    backbone.load_state_dict(torch.load(os.path.join(directory, 'pretrained_weights', 'pretrained_weights.pth'),
                                        map_location='cpu', weights_only=True))
    backbone.proj_out = torch.nn.Identity()
    return backbone.eval()


def pooled_features(backbone, segments, batch=128, full=False):
    """CBraMod readouts of segments (segments, channels, patches, 200): averaged over channels and patches, and
    averaged over patches for each channel. With full=True also every patch embedding, flattened.
    """
    import torch
    pooled, per_channel, everything = [], [], []
    with torch.no_grad():
        for start in range(0, len(segments), batch):
            features = backbone(torch.as_tensor(segments[start:start + batch], dtype=torch.float32))
            pooled.append(features.mean((1, 2)).numpy())
            per_channel.append(features.mean(2).flatten(1).numpy())
            if full:
                everything.append(features.flatten(1).numpy())
            print(f'  {min(start + batch, len(segments))} of {len(segments)} segments', flush=True)
    out = np.concatenate(pooled).astype(np.float32), np.concatenate(per_channel).astype(np.float32)
    return out + (np.concatenate(everything).astype(np.float32),) if full else out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cbramod', required=True, help="directory with the authors' code and pretrained weights")
    parser.add_argument('--input', default=None, help='recordings already prepared for the model (see above)')
    parser.add_argument('--edf-dir', default=None, help='directory with the recordings (.edf), instead of --input')
    parser.add_argument('--subjects-from', default=None, action='append',
                        help='with --edf-dir: a data file whose participants are used, in its order (can be given several times)')
    parser.add_argument('--out', required=True, help='prefix of the two output files')
    parser.add_argument('--max-participants', type=int, default=None, help='smoke tests: use a subset')
    parser.add_argument('--full', action='store_true',
                        help='also write <out>_full.npy: every patch embedding, flattened')
    parser.add_argument('--check-inputs', action='store_true',
                        help='only check that every recording is there (--edf-dir) or the file has the expected '
                             'shape (--input), then stop; nothing is written')
    args = parser.parse_args(argv)
    if (args.input is None) == (args.edf_dir is None) or (args.edf_dir and not args.subjects_from):
        raise SystemExit('Give either --input, or --edf-dir with --subjects-from.')
    if os.path.exists(args.out + '_pooled.npy') and not args.check_inputs:
        raise SystemExit(f'{args.out}_pooled.npy exists; results are not overwritten.')

    from ...EEGModalNet.data.ds005385 import process_ds005385_for_CBraMod, MODEL_CHANNELS
    if args.input:
        data = xr.open_dataarray(args.input, engine='h5netcdf')
        if data.dims != ('subject', 'epoch', 'channel', 'segment', 'time') or data.shape[3:] != (2, PATCH):
            raise SystemExit(f'{args.input}: expected subject x epoch x channel x 2 patches x {PATCH} samples; found {data.dims} {data.shape}.')
        if [str(c) for c in data.channel.values] != MODEL_CHANNELS:
            raise SystemExit(f'{args.input}: channels {list(data.channel.values)}, expected {MODEL_CHANNELS} in this order.')
        subjects = data.subject.values.astype(str)[:args.max_participants]
        if args.check_inputs:
            print(f'{args.input}: {len(subjects)} participants, shape {data.shape}')
            return None
        epochs = data.values[:len(subjects)]
    else:
        subjects = np.concatenate([xr.open_dataarray(path, engine='h5netcdf').subject.values.astype(str)
                                   for path in args.subjects_from])[:args.max_participants]
        missing = [s for s in subjects if not os.path.isfile(edf_path(args.edf_dir, s))]
        if missing:
            raise SystemExit(f'{len(missing)} of {len(subjects)} recordings missing in {args.edf_dir}, e.g. {edf_path(args.edf_dir, missing[0])}')
        if args.check_inputs:
            print(f'all {len(subjects)} recordings present in {args.edf_dir}')
            return None
        epochs = np.stack([process_ds005385_for_CBraMod(s, MODEL_CHANNELS, edf_dir=args.edf_dir) for s in subjects])
    segments = four_second_segments(epochs * UNIT)
    print(f'{len(subjects)} participants, {segments.shape[1]} segments of {segments.shape[3]} s, {segments.shape[2]} channels', flush=True)
    result = pooled_features(load_backbone(args.cbramod), segments.reshape(-1, *segments.shape[2:]), full=args.full)
    features, per_channel = result[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out + '_pooled.npy', features)
    np.save(args.out + '_channels.npy', per_channel)
    if args.full:
        np.save(args.out + '_full.npy', result[2])
        print(f'wrote {args.out}_full.npy {result[2].shape}')
    np.save(args.out + '_subjects.npy', np.asarray(subjects))
    print(f'wrote {args.out}_pooled.npy {features.shape}, {args.out}_channels.npy {per_channel.shape} and {args.out}_subjects.npy')
    return features


if __name__ == '__main__':
    main()
