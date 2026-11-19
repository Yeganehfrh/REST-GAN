"""The model is in `FiLMGAN`, `layers` and `spectral`. The other modules are earlier Keras experiments, loaded on first use."""
import importlib
import os

_LEGACY_KERAS = {'MultiInputModel': '.multi_input', 'SimpleGAN': '.gan', 'WGAN_GP': '.WGAN'}


def __getattr__(name):
    if name in _LEGACY_KERAS:
        os.environ.setdefault('KERAS_BACKEND', 'torch')
        return getattr(importlib.import_module(_LEGACY_KERAS[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
