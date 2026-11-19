"""REST-GAN: a conditional WGAN-GP for resting-state EEG. The earlier Keras experiments are loaded on first use."""
import importlib
import os

import numpy  # noqa: F401  (before torch: avoids a duplicate-OpenMP crash on macOS)

from .models.FiLMGAN import FiLMGAN, Generator, Critic
from .training import Trainer, load_model
from .data.sampler import RecordingSampler

__all__ = ['FiLMGAN', 'Generator', 'Critic', 'Trainer', 'load_model', 'RecordingSampler',
           'preprocess_data', 'zscore_recordings', 'load_data']

_LAZY = {'load_data': '.data.lemon', 'preprocess_data': '.preprocessing.preprocessing',
         'zscore_recordings': '.preprocessing.preprocessing'}
_LEGACY_KERAS = {
    'WGAN_GP': '.models.WGAN', 'WGAN_GP_V0': '.models.WGAN_v0', 'TCNWGAN': '.models.TCN_WGAN',
    'ProgressBarCallback': '.utils.utils', 'CustomModelCheckpoint': '.utils.utils',
    'StepLossHistory': '.utils.utils', 'BalancedAccuracy': '.utils.utils',
    'get_averaged_data': '.preprocessing.utils',
    'SubjectLayers_v2': '.models.common', 'convBlock': '.models.common', 'ChannelMerger': '.models.common',
    'ResidualBlock': '.models.common', 'build_eeg_transformer': '.models.common',
    'RandomCropEEGDataset': '.data.hpc_data_loader',
    'extract_features_batched': '.utils.extractor', 'extract_features_batched_deterministic': '.utils.extractor',
}


def __getattr__(name):
    if name in _LEGACY_KERAS:
        os.environ.setdefault('KERAS_BACKEND', 'torch')
        return getattr(importlib.import_module(_LEGACY_KERAS[name], __name__), name)
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
