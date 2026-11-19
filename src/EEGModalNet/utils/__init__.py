"""Helpers of the earlier Keras experiments (`utils`, `extractor`), loaded on first use."""
import importlib
import os

_LEGACY_KERAS = ['ProgressBarCallback', 'CustomModelCheckpoint', 'StepLossHistory', 'BalancedAccuracy',
                 'load_training_state', 'get_resume_state_path']


def __getattr__(name):
    if name in _LEGACY_KERAS:
        os.environ.setdefault('KERAS_BACKEND', 'torch')
        return getattr(importlib.import_module('.utils', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
