"""Read checkpoints of the earlier Keras implementation (`.model.keras`) into the PyTorch model.

A `.model.keras` file is a zip archive with the configuration and the weights; neither Keras nor TensorFlow is
needed to read it. Every weight in the file must find its place in the PyTorch model.
"""
import io
import json
import re
import zipfile

import h5py
import numpy as np
import torch

from .FiLMGAN import FiLMGAN, D_SUB, D_STATE

BN_EPSILON = 1e-3  # Keras default


def _read(path):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))['config']
        weights = {}
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as f:
            f.visititems(lambda name, obj: weights.update({name: np.array(obj)})
                         if isinstance(obj, h5py.Dataset) else None)
    return config, weights


def _numbered(weights, prefix, stem):
    """Layer groups `prefix/stem`, `prefix/stem_1`, ... in creation order."""
    names = {m.group(1) for k in weights if (m := re.match(rf'{re.escape(prefix)}/({stem}(?:_\d+)?)/vars/', k))}
    return [f'{prefix}/{n}' for n in sorted(names, key=lambda n: int(n.rsplit('_', 1)[1]) if '_' in n[len(stem):] else -1)]


def keras_config(config):
    """Constructor arguments of the PyTorch model from a Keras model configuration."""
    if config.get('use_sublayer_critic') or not config.get('use_sublayer_generator', True) \
            or config.get('use_channel_merger_g') or config.get('use_channel_merger_c'):
        raise ValueError('This checkpoint uses options the PyTorch model does not implement '
                         '(critic sub-layer, no generator sub-layer, or channel merger).')
    defaults = dict(use_recon=True, use_mbsd=True, use_second_diff=True, use_dilation=True,
                    critic_cond='film', use_contrastive=False)
    # the Keras model: a latent vector per segment, reconstruction of masked time spans, no spectral head
    keras_model = dict(recon_mask='time', gen_arch='latent', spectral_head=False, head_bins=65)
    return {**{k: config[k] for k in ('time_dim', 'feature_dim', 'latent_dim', 'n_subjects')},
            **{k: config.get(k, v) for k, v in defaults.items()}, **keras_model}


def keras_state_dict(config, weights):
    """PyTorch state dict from the weights of a Keras checkpoint. `config` as returned by `keras_config`."""
    weights = {k: v for k, v in weights.items() if k.startswith(('critic/', 'generator/'))}
    used, state = set(), {}

    def take(key):
        used.add(key)
        return torch.from_numpy(weights[key])

    def conv(group, name):
        state[f'{name}.weight'] = take(f'{group}/vars/0').permute(2, 1, 0).contiguous()
        state[f'{name}.bias'] = take(f'{group}/vars/1')

    def dense(group, name):
        state[f'{name}.weight'] = take(f'{group}/vars/0').T.contiguous()
        state[f'{name}.bias'] = take(f'{group}/vars/1')

    def torch_module(group, name):  # modules that already were PyTorch: same parameter names and layout
        for key in [k for k in weights if k.startswith(f'{group}/vars/')]:
            state[f"{name}.{key.rsplit('/', 1)[1]}"] = take(key)

    def wrapped_modules(part):  # embeddings, projection and the generator's output FiLM
        for group in _numbered(weights, f'{part}/layers', 'torch_module_wrapper'):
            keys = {k.rsplit('/', 1)[1] for k in weights if k.startswith(f'{group}/vars/')}
            if keys == {'weight'}:
                width = weights[f'{group}/vars/weight'].shape[1]
                torch_module(group, f"{part}.{ {D_SUB: 'sub_emb', D_STATE: 'state_emb'}[width] }")
            elif keys == {'sub.weight', 'state.weight'}:
                torch_module(group, f'{part}.projection')
            elif 'sub_film.weight' in keys:
                torch_module(group, f'{part}.out_film')
            else:
                raise ValueError(f'Unrecognised module in the checkpoint: {group} with {sorted(keys)}')

    c, t = config['feature_dim'], config['time_dim']

    # critic
    for i in (1, 2, 3):
        conv(f'critic/conv{i}', f'critic.conv{i}')
    if config['critic_cond'] == 'film':
        torch_module('critic/film_block', 'critic.film')
    wrapped_modules('critic')
    if config['use_recon']:
        for group, name in zip(_numbered(weights, 'critic/layers', 'conv1d'), ('recon_conv1', 'recon_conv2', 'recon_out'),
                               strict=True):
            conv(group, f'critic.{name}')
    if config['use_contrastive']:
        dense('critic/con_dense1', 'critic.con_fc1')
        dense('critic/con_dense2', 'critic.con_fc2')
    # final layer: Keras flattened [h, h1] as (time, features); the PyTorch model as (features, time)
    kernel = take('critic/final_dense/vars/0')[:, 0]
    n_h1 = 2 * c * t
    coarse = kernel[:-n_h1].reshape(-1, 16 * c + int(config['use_mbsd'])).T
    fine = kernel[-n_h1:].reshape(t, 2 * c).T
    state['critic.final.weight'] = torch.cat([coarse.reshape(-1), fine.reshape(-1)])[None, :].contiguous()
    state['critic.final.bias'] = take('critic/final_dense/vars/1')

    # generator
    dense('generator/layers/sequential/layers/dense', 'generator.fc')
    torch_module('generator/film_block', 'generator.film')
    wrapped_modules('generator')
    for group, name in zip(_numbered(weights, 'generator/cov_block/layers', 'conv1d'), ('conv_a', 'conv_b', 'to_channels'),
                           strict=True):
        conv(group, f'generator.{name}')
    for group, name in zip(_numbered(weights, 'generator/cov_block/layers', 'batch_normalization'),
                           ('affine_a', 'affine_b'), strict=True):
        gamma, beta, mean, variance = (take(f'{group}/vars/{i}') for i in range(4))
        scale = gamma / torch.sqrt(variance + BN_EPSILON)
        state[f'generator.{name}.scale'] = scale
        state[f'generator.{name}.shift'] = beta - mean * scale
    if config['use_dilation']:
        for group, name in zip(_numbered(weights, 'generator/dil_block/layers', 'conv1d'), ('dil_conv1', 'dil_conv2'),
                               strict=True):
            conv(group, f'generator.{name}')

    unused = sorted(set(weights) - used)
    if unused:
        raise ValueError(f'Weights in the checkpoint that the PyTorch model has no place for: {unused}')
    return state


def load_keras_checkpoint(path):
    """Build the PyTorch model from a `.model.keras` checkpoint of the Keras implementation."""
    raw_config, weights = _read(path)
    config = keras_config(raw_config)
    model = FiLMGAN(**config)
    model.load_state_dict(keras_state_dict(config, weights), strict=True)
    return model
