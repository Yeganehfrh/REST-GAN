"""Measure the cost of training with 8 to 128 channels: time per training step, peak GPU memory and parameters
(Section 2.6).

Run on a GPU from the repository root:

    python -m src.EEGModalNet.pipeline.benchmark_cost --channels 8 16 32 64 128 --out logs/benchmark_cost.json

The model is the model of the paper, in mixed precision, with batches of 128 segments. Two further variants
(without the spectral head, and without the head and the spectral loss) show which parts grow with the number
of channels. The recordings are synthetic (white noise of the shape of the LEMON data).
"""
import argparse
import json
import time

import torch

from ..models.FiLMGAN import FiLMGAN
from ..training import Trainer
from ..data.sampler import RecordingSampler

VARIANTS = {
    'design': dict(spectral_head=True, spectral_weight=1.0),
    'without spectral head': dict(spectral_head=False, spectral_weight=1.0),
    'without spectral head and loss': dict(spectral_head=False, spectral_weight=0.0),
}


def measure(n_channels, variant, device, steps, warmup, n_recordings, length, seed=0):
    flags = VARIANTS[variant]
    torch.manual_seed(seed)
    n_subjects = n_recordings // 2
    model = FiLMGAN(time_dim=512, feature_dim=n_channels, latent_dim=128, n_subjects=n_subjects,
                    spectral_head=flags['spectral_head'])
    # random starting values of the embeddings: the recordings are white noise, and the start does not change the cost
    trainer = Trainer(model, device=device, seed=seed, spectral_weight=flags['spectral_weight'], embedding_init='random')
    # synthetic recordings
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_recordings, n_channels, length, generator=g)
    sub = torch.arange(n_recordings) // 2
    state = torch.arange(n_recordings) % 2
    sampler = RecordingSampler(x, sub, state, seg_len=512, batch_size=128, device=device,
                               n_groups=trainer.hparams['spectral_groups'] if flags['spectral_weight'] > 0 else 0)
    trainer.prepare(sampler)
    row = {'channels': n_channels, 'variant': variant,
           'parameters_generator': sum(p.numel() for p in model.generator.parameters()),
           'parameters_critic': sum(p.numel() for p in model.critic.parameters())}
    cuda = device.startswith('cuda')
    try:
        # warm-up steps (not timed)
        for _ in range(warmup):
            trainer.train_step(sampler.batch())
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # timed steps
        start = time.perf_counter()
        for _ in range(steps):
            trainer.train_step(sampler.batch(), sync=False)
        if cuda:
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / steps
        trainer._raise_if_nonfinite()
        row.update(seconds_per_step=seconds,
                   hours_per_100_epochs=seconds * trainer.hparams['steps_per_epoch'] * 100 / 3600,
                   peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
    except torch.OutOfMemoryError:
        row.update(seconds_per_step=None, hours_per_100_epochs=None, peak_memory_gb='out of memory')
    finally:
        del trainer, model, sampler, x
        if cuda:
            torch.cuda.empty_cache()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--channels', type=int, nargs='+', default=[8, 16, 32, 64, 128])
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--steps', type=int, default=50, help='timed training steps per setting')
    parser.add_argument('--warmup', type=int, default=10, help='untimed steps before them')
    parser.add_argument('--recordings', type=int, default=20, help='synthetic recordings (two per participant)')
    parser.add_argument('--length', type=int, default=61440, help='samples per synthetic recording (8 min at 128 Hz)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True, help='JSON file for the results')
    args = parser.parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('No GPU: the benchmark measures GPU cost (use --device cpu only for a smoke test).')

    info = {'device': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
            'torch': torch.__version__, 'steps': args.steps, 'warmup': args.warmup}
    rows = []
    for variant in args.variants:
        for n in args.channels:
            row = measure(n, variant, args.device, args.steps, args.warmup, args.recordings, args.length)
            rows.append(row)
            mem = row['peak_memory_gb']
            print(f"{variant:32s} {n:4d} channels: "
                  + (f"{row['seconds_per_step']:.3f} s/step, {row['hours_per_100_epochs']:.1f} h per 100 epochs, "
                     f"peak memory {mem if isinstance(mem, str) else f'{mem:.2f} GB' if mem is not None else '-'}"
                     if row['seconds_per_step'] is not None else 'out of memory')
                  + f", parameters G {row['parameters_generator']:,} / D {row['parameters_critic']:,}", flush=True)
            with open(args.out, 'w') as f:
                json.dump({'info': info, 'rows': rows}, f, indent=1)
    return rows


if __name__ == '__main__':
    main()
