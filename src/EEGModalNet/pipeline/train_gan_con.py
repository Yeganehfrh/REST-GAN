## Seed: fixed before any framework import; override with `--seed N` or EEGMODALNET_SEED
import os
import sys
import random
import numpy as np


def _early_seed(default=42):
    for i, arg in enumerate(sys.argv):
        if arg == '--seed' and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith('--seed='):
            return int(arg.split('=', 1)[1])
    return int(os.environ.get('EEGMODALNET_SEED', default))


SEED = _early_seed()

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

## Setting Keras backend to PyTorch and configuring CUDA devices
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

print("SDP backends:",
      "flash =", torch.backends.cuda.flash_sdp_enabled(),
      "mem_efficient =", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "math =", torch.backends.cuda.math_sdp_enabled())

import keras
keras.utils.set_random_seed(SEED)

import argparse
import json
import subprocess
from datetime import datetime

from keras.optimizers.schedules import ExponentialDecay
from ...EEGModalNet import (CustomModelCheckpoint, preprocess_data, zscore_recordings, FiLMGAN,
                            RandomCropEEGDataset)
from ...EEGModalNet.utils.utils import load_training_state, get_resume_state_path
from typing import List, Dict
import xarray as xr
from meegkit import dss
from scipy.signal import butter, sosfiltfilt


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
              preprocess=False,
              highpass=False,
              remove_line_noise=True) -> Dict:
    """Load and preprocess the training recordings.

    preprocess: True -> per-recording, per-channel z-score clipped at 20 SD (`zscore_recordings`);
                'legacy' -> the original baseline + RobustScaler + StandardScaler chain (same result).
    exclude_sub_ids: participants held out from training (dropped from both conditions).
    Returns x, sub (embedding index), pos (0 = EC, 1 = EO) and `subjects`, the participant id of each
    embedding index.
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
        xarray_2  = xarray_2.rename({"time": "timestep"}) # we know that the naming of the time dimensions are not the same
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

    if preprocess == 'legacy':
        x = preprocess_data(x, sampling_rate=128)
    elif preprocess:
        x = zscore_recordings(x, clip=20.0)

    if highpass:
        sos = butter(4, 0.5, btype='high', fs=128, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    if remove_line_noise:
        x, _ = dss.dss_line(x.T, fline=50, sfreq=128, nremove=1)
        x = x.T

    x = torch.tensor(x.copy(), dtype=torch.float32)
    if condition == 'both':
        if x.shape[0] % 2 != 0:
            raise ValueError("Expected an even number of subjects for condition='both'.")
        n_per_condition = x.shape[0] // 2
        sub = torch.arange(n_per_condition, dtype=torch.long)[:, None].repeat(2, 1)
        state_ids = torch.cat([
            torch.zeros(n_per_condition, dtype=torch.long),
            torch.ones(n_per_condition, dtype=torch.long)
        ], dim=0)[:, None]
        subject_ids = subject_ids[:n_per_condition]
    else:
        sub = torch.arange(x.shape[0], dtype=torch.long)[:, None]
        state_value = 0 if condition != 'eo' else 1
        state_ids = torch.full((x.shape[0], 1), state_value, dtype=torch.long)

    return {'x': x, 'sub': sub, 'pos': state_ids, 'subjects': subject_ids}


def build_model(n_subjects, n_channels, latent_dim, steps_per_epoch):
    return FiLMGAN(time_dim=512,
                   feature_dim=n_channels,
                   latent_dim=latent_dim,
                   n_subjects=n_subjects,
                   use_sublayer_generator=True,
                   use_sublayer_critic=False,
                   use_channel_merger_g=False,
                   use_channel_merger_c=False,
                   interpolation='bilinear',
                   steps_per_epoch=steps_per_epoch)


def materialize(model, sample_batch, latent_dim, device):
    """Run one forward pass through every branch so that all lazily built layers own their weights."""
    if isinstance(sample_batch, (tuple, list)):
        sample_batch = sample_batch[0]
    sample_x = sample_batch['x'].to(device)
    sample_sub = sample_batch['sub'].to(device)
    sample_pos = sample_batch['pos'].to(device)
    with torch.no_grad():
        sample_h1, sample_h = model.critic.encode_features({'x': sample_x, 'sub': sample_sub, 'pos': sample_pos})
        _ = model.critic.score_from_features(sample_h1, sample_h)
        _ = model.critic.reconstruct_from_features(sample_h)
        sample_noise = keras.random.normal((sample_x.size(0), latent_dim), dtype=sample_x.dtype)
        _ = model.generator((sample_noise, sample_sub, sample_pos))


def compile_model(model):
    lr_schedule_g = ExponentialDecay(0.0002, decay_steps=100000, decay_rate=0.90, staircase=True)
    lr_schedule_d = ExponentialDecay(0.0006, decay_steps=100000, decay_rate=0.90, staircase=True)
    model.compile(d_optimizer=keras.optimizers.Adam(lr_schedule_d, beta_1=0.0, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(lr_schedule_g, beta_1=0.0, beta_2=0.9),
                  gradient_penalty_weight=1.0,
                  recon_weight=0.1)


def prepare_model(sample_batch, n_subjects, n_channels, latent_dim, steps_per_epoch, device,
                  checkpoint_path=None):
    """Build (or restore) a compiled model. Returns (model, initial_epoch, resume_status).

    Restoring: the critic's conv layers are built lazily, so `keras.saving.load_model` alone leaves them
    freshly initialised. The weights are therefore loaded again after every layer has been built.
    """
    if checkpoint_path is not None:
        print(checkpoint_path)
        model = keras.saving.load_model(checkpoint_path, custom_objects={'FiLMGAN': FiLMGAN}, compile=False)
    else:
        model = build_model(n_subjects, n_channels, latent_dim, steps_per_epoch)

    model.to(device)
    print(f'>>>> Model is on {device}')
    model.steps_per_epoch = steps_per_epoch

    # Materialize all lazy weights before loading weights and building/restoring optimizer slots.
    materialize(model, sample_batch, latent_dim, device)
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    compile_model(model)

    initial_epoch, resume_status = 0, None
    if checkpoint_path is not None:
        initial_epoch, resume_status = load_training_state(model, checkpoint_path)
        if resume_status == 'exact':
            print(f'>>>> Resuming exactly from epoch {initial_epoch} using {get_resume_state_path(checkpoint_path)}')
        elif resume_status == 'weights_only':
            print(f'>>>> Resuming from checkpoint weights at epoch {initial_epoch}; optimizer state was not restored.')
        else:
            print(f'>>>> No resume sidecar found at {get_resume_state_path(checkpoint_path)}; continuing from weights only.')
    return model, initial_epoch, resume_status


def run(train_loader,
        n_subjects,
        channels,
        max_epochs=100_000,
        latent_dim=64,
        batch_size=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        checkpoint_path=None,
        shuffle=False,
        steps_per_epoch=500,
        save_freq=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, initial_epoch, _ = prepare_model(next(iter(train_loader)), n_subjects, len(channels), latent_dim,
                                            steps_per_epoch, device,
                                            checkpoint_path=checkpoint_path if reuse_model else None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for model to be loaded

    # step_loss_history = StepLossHistory()
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield (batch,)

    _ = model.fit(infinite_loader(train_loader),
                  batch_size=batch_size,
                  epochs=max_epochs,
                  initial_epoch=initial_epoch,
                  shuffle=shuffle,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=[
                      CustomModelCheckpoint(model_path, save_freq=save_freq, save_training_state=True),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_gloss.model.keras', monitor='2 g_loss', save_best_only=True, mode='min'),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_dloss.model.keras', monitor='1 d_loss', save_best_only=True, mode='min'),
                      keras.callbacks.CSVLogger(cvloger_path, append=reuse_model),
                      keras.callbacks.TerminateOnNaN()
                      # step_loss_history
                  ])

    return model


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], capture_output=True, text=True, check=False).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the conditional WGAN-GP on LEMON resting-state EEG.')
    parser.add_argument('--seed', type=int, default=SEED, help='random seed (read before framework import)')
    parser.add_argument('--run-name', type=str, default=None, help='output prefix under logs/; default: date_seed')
    parser.add_argument('--heldout-split', type=str, default=None,
                        help='JSON file with a "heldout" list of participant ids excluded from training')
    parser.add_argument('--data-path', type=str, default='data/LEMON_DATA/EC_ch-8_sf-128.nc5')
    parser.add_argument('--condition', type=str, default='both', choices=['both', 'EC', 'EO'])
    parser.add_argument('--max-epochs', type=int, default=420)
    parser.add_argument('--steps-per-epoch', type=int, default=500)
    parser.add_argument('--save-freq', type=int, default=10, help='checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None, help='checkpoint (.model.keras) to resume from')
    parser.add_argument('--allow-cpu', action='store_true', help='run without a GPU (smoke tests only)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    CHANNELS = {
        8: ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
        16: ['O1', 'O2', 'P3', 'P1', 'Pz', 'P2', 'P4',
             'C3', 'C1', 'C2', 'C4', 'F1', 'F2', 'AF3', 'AFz', 'AF4'],
        32: ['O1', 'O2', 'P3', 'P1', 'Pz', 'P2', 'P4',
             'C3', 'C1', 'C2', 'C4', 'F1', 'F2', 'AF3', 'AFz', 'AF4',
             'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
             'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'POz'],
        56: ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
             'C3', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz', 'P7',
             'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'AF7', 'AF3',
             'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8', 'C5',
             'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2',
             'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
    }
    args = parse_args()
    if args.seed != SEED:
        raise SystemExit(f'--seed must be given as "--seed N" or "--seed=N" (parsed {SEED}, got {args.seed}).')

    N_SUBJECTS_FILE = 202  # participants in the LEMON training files
    LATENT_DIM = 128
    BATCH_SIZE = 128
    RUN_NAME = args.run_name or f'{datetime.now():%Y%m%d}_seed{SEED}'
    OUTPUT_PATH = f'logs/{RUN_NAME}'
    CONDITION = args.condition  # supported: 'both', 'EC', or 'EO'

    heldout = None
    if args.heldout_split is not None:
        with open(args.heldout_split) as f:
            heldout = json.load(f)['heldout']

    data = load_data(args.data_path,
                     channels='all',
                     n_subjects=N_SUBJECTS_FILE,
                     condition=CONDITION,
                     exclude_sub_ids=heldout,
                     preprocess=True,
                     highpass=True,
                     remove_line_noise=True)
    N_SUBJECTS = len(data['subjects'])  # size of the subject-embedding tables

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    # the original run config is kept; a resumed job writes its own file next to it
    config_path = f'{OUTPUT_PATH}.config.json' if args.resume_from is None else f'{OUTPUT_PATH}.resume.config.json'
    with open(config_path, 'w') as f:
        json.dump({'run_name': RUN_NAME, 'started': datetime.now().isoformat(timespec='seconds'),
                   'args': vars(args), 'seed': SEED, 'git': _git_state(),
                   'keras': keras.__version__, 'torch': torch.__version__,
                   'latent_dim': LATENT_DIM, 'batch_size': BATCH_SIZE,
                   'n_subjects': N_SUBJECTS, 'n_heldout': 0 if heldout is None else len(heldout),
                   'subjects': data['subjects'],  # participant id of each embedding index
                   'heldout': heldout}, f, indent=1)

    # Set random generator for DataLoader
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = torch.utils.data.DataLoader(
    RandomCropEEGDataset(
        data['x'], data['sub'], data['pos'],
        seg_len=512,
        n_samples=100_000   # 100k random crops per epoch
    ),
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=True,
    generator=g
    )

    if torch.cuda.is_available():
        print('GPU is available')
        print(f'Running on {torch.cuda.device_count()} GPUs')

        # Explicitly set the CUDA device
        torch.cuda.set_device(0)

        # preload CUDA libraries with a dummy tensor
        _ = torch.randn(1, device="cuda")

        # Apply mixed precision policy
        keras.mixed_precision.set_global_policy('mixed_float16')
        print(f'Global policy is {keras.mixed_precision.global_policy().name}')
    elif args.allow_cpu:
        print('GPU is not available: running on CPU (smoke test only)')
        # keep Keras off MPS so that its tensors and the model live on the same device
        from keras.src.backend.common import global_state
        from keras.src.backend.torch import core as _torch_core
        global_state.set_global_attribute('torch_device', 'cpu')
        _torch_core.DEFAULT_DEVICE = 'cpu'
    else:
        print('GPU is not available!!')
        exit()

    model = run(train_loader,
                n_subjects=N_SUBJECTS,
                channels=CHANNELS[8],
                max_epochs=args.max_epochs,
                latent_dim=LATENT_DIM,
                batch_size=BATCH_SIZE,
                cvloger_path=f'{OUTPUT_PATH}.csv',
                model_path=OUTPUT_PATH,
                reuse_model=args.resume_from is not None,
                checkpoint_path=args.resume_from,
                shuffle=False,
                steps_per_epoch=args.steps_per_epoch,
                save_freq=args.save_freq)
