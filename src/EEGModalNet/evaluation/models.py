"""Loading trained models and generating recordings from them for the evaluation, on CPU."""
import json
import os
import re

import torch

from ..training import load_model


def load_checkpoint(path, ema=True):
    """Load a checkpoint (`.pt` or `.model.keras`), frozen and in eval mode."""
    model = load_model(path, ema=ema)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def checkpoint_name(path):
    """`run_epoch_150` for `logs/run_epoch_150.pt` (or `.model.keras`)."""
    return re.sub(r'(\.pt|\.model\.keras)$', '', os.path.basename(path))


def run_prefix(path):
    """`logs/run` for `logs/run_epoch_150.pt` (or `.model.keras`), and for the run's own `logs/run.config.json`."""
    if path.endswith('.config.json'):
        return path[:-len('.config.json')]
    return re.sub(r'_epoch_\d+(\.pt|\.model\.keras)$', '', path)


def preprocessing_of(path):
    """(filter_first, block_len) of the data a run was trained on."""
    prefix = run_prefix(path)
    if os.path.exists(f'{prefix}.config.json'):
        with open(f'{prefix}.config.json') as f:
            p = json.load(f).get('preprocessing')
        if p:
            return p['filter_first'], p['block_len']
    return False, None


def line_noise_of(path):
    """Line-noise step of the data a run was trained on ('dss', 'lowpass' or 'none')."""
    prefix = run_prefix(path)
    if os.path.exists(f'{prefix}.config.json'):
        with open(f'{prefix}.config.json') as f:
            p = json.load(f).get('preprocessing')
        if p:
            return p.get('line_noise', 'dss')
    return 'dss'


def _randn(generator):
    return lambda *shape: torch.randn(*shape, generator=generator)


def run_config(path, all_subjects, heldout_split=None):
    """Training participants (in embedding order), held-out participants and conditioning flags of a run."""
    prefix = run_prefix(path)
    if os.path.exists(f'{prefix}.config.json'):
        with open(f'{prefix}.config.json') as f:
            cfg = json.load(f)
        return {'train': cfg['subjects'], 'heldout': cfg.get('heldout') or [], 'truly_heldout': True,
                'no_subject_cond': bool(cfg['args'].get('no_subject_cond')),
                'no_state_cond': bool(cfg['args'].get('no_state_cond'))}
    heldout = []
    if heldout_split:
        with open(heldout_split) as f:
            heldout = json.load(f)['heldout']
    print(f'!! no config found for {path}: assuming all {len(all_subjects)} participants were trained on')
    return {'train': list(all_subjects), 'heldout': heldout, 'truly_heldout': False,
            'no_subject_cond': False, 'no_state_cond': False}


def generate_from_embedding(generator, subj_emb, z, state):
    """Generator forward pass with an explicit participant embedding (n, d_sub); differentiable in it."""
    return generator(z, state=torch.full((len(z),), int(state), dtype=torch.long), subj_emb=subj_emb)


def generate(model, embedding_index, state, n_seg, seed):
    """One generated recording (n_seg segments) per entry of `embedding_index`: list of (n_seg, T, C)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for idx in embedding_index:
            z = model.generator.sample_latent(n_seg, _randn(g))
            sub = torch.full((n_seg,), int(idx), dtype=torch.long)
            out.append(model.generator(z, sub, torch.full((n_seg,), int(state), dtype=torch.long)).numpy())
    return out


def generate_continuous(model, embedding_index, state, n_samples, seed):
    """One continuous signal (n_samples, C) per entry of `embedding_index`, generated in a single pass (Section 2.4.5)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for idx in embedding_index:
            z = model.generator.sample_latent(1, _randn(g), length=n_samples)
            sub = torch.full((1,), int(idx), dtype=torch.long)
            out.append(model.generator(z, sub, torch.full((1,), int(state), dtype=torch.long))[0].numpy())
    return out


def generate_with_embeddings(model, embeddings, state, n_seg, seed):
    """As `generate`, but from explicit embedding vectors (n_recordings, d_sub)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for emb in torch.as_tensor(embeddings, dtype=torch.float32):
            z = model.generator.sample_latent(n_seg, _randn(g))
            out.append(generate_from_embedding(model.generator, emb[None, :].expand(n_seg, -1), z, state).numpy())
    return out


def embed_recordings(part, recordings):
    """Participant embedding of each recording, inferred from all its segments (subject_cond='encoder')."""
    with torch.no_grad():
        return torch.cat([part.embed_refs(torch.as_tensor(rec, dtype=torch.float32)[None])
                          for rec in recordings]).numpy()


def generate_like(model, recordings, state, n_seg, seed):
    """One generated recording per real one, the participant inferred from the real recording (encoder models)."""
    return generate_with_embeddings(model, embed_recordings(model.generator, recordings), state, n_seg, seed)
