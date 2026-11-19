"""Train REST-GAN on LEMON resting-state EEG.

Run from the repository root (the defaults are the model of the paper):

    python -m src.EEGModalNet.pipeline.train_gan_con --seed 1 --run-name main_s1 \\
        --heldout-split splits/lemon_heldout40.json

Add `--resume-from logs/main_s1_epoch_50.pt` to resume a run.
Writes logs/<run>.config.json, logs/<run>.csv and logs/<run>_epoch_<n>.pt.
"""
import argparse
import json
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import torch

from ...EEGModalNet.models.FiLMGAN import FiLMGAN
from ...EEGModalNet.training import Trainer
from ...EEGModalNet.data.sampler import RecordingSampler
from ...EEGModalNet.data.lemon import load_data, LEMON_BLOCK

N_SUBJECTS_FILE = 202  # participants in the LEMON training files
LATENT_DIM = 128
BATCH_SIZE = 128
SEG_LEN = 512


def seed_everything(seed, deterministic=False):
    """Seed every random generator. deterministic=True uses deterministic cuDNN algorithms (slower)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def preprocessing_config(legacy=False, line_noise='lowpass'):
    """How the training data are preprocessed; stored in the run config and read by the evaluation."""
    if legacy:
        return {'filter_first': False, 'block_len': None, 'line_noise': 'dss',
                'steps': 'z-score per recording and channel, clip at 20 SD, 0.5-Hz high-pass (whole recording), '
                         '50-Hz DSS line-noise removal'}
    step = {'dss': '50-Hz DSS line-noise removal, ', 'lowpass': '45-Hz low-pass per 60-s block, ', 'none': ''}[line_noise]
    return {'filter_first': True, 'block_len': LEMON_BLOCK, 'line_noise': line_noise,
            'steps': f'0.5-Hz high-pass per 60-s block, {step}z-score per recording and '
                     'channel, clip at 20 SD; training segments cut within blocks'}


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], capture_output=True, text=True, check=False).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train REST-GAN on LEMON resting-state EEG.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', type=str, default=None, help='output prefix under logs/ (default: date_seed)')
    parser.add_argument('--heldout-split', type=str, default=None,
                        help='JSON file with a "heldout" list of participants excluded from training')
    parser.add_argument('--data-path', type=str, default='data/LEMON_DATA/EC_ch-8_sf-128.nc5')
    parser.add_argument('--condition', type=str, default='both', choices=['both', 'EC', 'EO'])
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=500)
    parser.add_argument('--save-freq', type=int, default=10, help='checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='checkpoint (.pt) to resume from; the model and training options come from it')
    parser.add_argument('--allow-cpu', action='store_true', help='run without a GPU (smoke tests only)')
    parser.add_argument('--no-amp', action='store_true', help='full precision on the GPU (default: mixed precision)')
    parser.add_argument('--legacy-preprocessing', action='store_true',
                        help='z-score before the filters and filter the whole recordings')
    parser.add_argument('--deterministic', action='store_true',
                        help='deterministic cuDNN algorithms: bit-identical repeats on the same GPU, but slower')
    parser.add_argument('--critic-cond', type=str, default='projection', choices=['projection', 'film'],
                        help='conditioning of the critic: projection at the output, or FiLM on the input')
    parser.add_argument('--contrastive', action='store_true',
                        help='add the same-recording contrastive loss to the critic (needs --critic-cond projection)')
    parser.add_argument('--contrastive-weight', type=float, default=0.5,
                        help='norm of the contrastive gradient relative to the adversarial gradient')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1)
    parser.add_argument('--recon-mask', type=str, default='channel', choices=['channel', 'time'],
                        help='the reconstruction branch reconstructs masked channels, or masked time spans')
    parser.add_argument('--proj-cov', action='store_true',
                        help="the projection term also uses the covariance of the second layer's feature maps")
    parser.add_argument('--line-noise', type=str, default='lowpass', choices=['lowpass', 'dss', 'none'],
                        help='after the high-pass: a 45-Hz low-pass, DSS line-noise removal, or nothing')
    parser.add_argument('--gen-arch', type=str, default='noise', choices=['noise', 'latent'],
                        help='generator: a noise sequence through dilated convolutions, or a latent vector with '
                             'upsampling')
    parser.add_argument('--spectral-filter', action='store_true',
                        help='the generator ends with a participant-specific linear filter')
    parser.add_argument('--spectral-weight', type=float, default=1.0,
                        help='spectral loss of the generator: norm of its gradient relative to the adversarial one '
                             '(0 = off)')
    parser.add_argument('--spectral-group-size', type=int, default=8,
                        help='generated segments per recording in the spectral loss')
    parser.add_argument('--spectral-groups', type=int, default=16, help='recordings per step in the spectral loss')
    parser.add_argument('--embedding-init', type=str, default='spectral', choices=['spectral', 'random'],
                        help='starting values of the participant embeddings: from the spectral features of the '
                             'participants, or random')
    parser.add_argument('--deep-critic', action='store_true',
                        help='four more strided layers in the critic: receptive field 2 s instead of 0.15 s')
    parser.add_argument('--head-bins', type=int, default=44,
                        help='the spectral head uses the bins from 0 to N - 1 Hz (at most 65)')
    parser.add_argument('--d-sub', type=int, default=32, help='size of the participant embedding')
    parser.add_argument('--ema-decay', type=float, default=0.0,
                        help="running average of the generator's weights, saved for evaluation (0 = off)")
    parser.add_argument('--subject-encoder', action='store_true',
                        help='infer the participant embedding from other segments of the recording '
                             '(needs --spectral-weight 0 --embedding-init random)')
    parser.add_argument('--n-refs', type=int, default=4, help='reference segments per item for --subject-encoder')

    # ablations of Table 1 (A2: --spectral-weight 0, A7: --embedding-init random)
    ablation = parser.add_argument_group('ablations (each removes one component of the full model)')
    ablation.add_argument('--no-spectral-head', action='store_true', help='A1: no spectral head in the critic')
    ablation.add_argument('--no-recon', action='store_true', help='A8: no reconstruction branch')
    ablation.add_argument('--no-mbsd', action='store_true', help='A6: no minibatch standard deviation layer')
    parser.add_argument('--mbsd-grad', action='store_true',
                        help='compute the minibatch standard deviation with gradient')
    ablation.add_argument('--no-second-diff', action='store_true', help='A5: no second-difference input to the critic')
    ablation.add_argument('--no-slow-noise', action='store_true',
                          help='G1: the smoothed noise channels are replaced by white noise')
    ablation.add_argument('--no-dilation', action='store_true',
                          help='G2: all dilation rates of the generator set to 1')
    ablation.add_argument('--no-subject-cond', action='store_true',
                          help='A3: no participant conditioning (one embedding shared by all participants)')
    ablation.add_argument('--no-state-cond', action='store_true',
                          help='A4: no eye-state conditioning (one state embedding for both eye states)')
    args = parser.parse_args(argv)
    if args.contrastive and args.critic_cond != 'projection':
        parser.error('--contrastive needs --critic-cond projection.')
    if args.subject_encoder and args.critic_cond != 'projection':
        parser.error('--subject-encoder needs --critic-cond projection.')
    if args.subject_encoder and args.no_subject_cond:
        parser.error('--subject-encoder and --no-subject-cond exclude each other.')
    if (args.spectral_weight > 0 or args.embedding_init == 'spectral') and args.subject_encoder:
        parser.error('--spectral-weight and --embedding-init spectral need participant tables, not --subject-encoder.')
    if args.proj_cov and args.critic_cond != 'projection':
        parser.error('--proj-cov needs --critic-cond projection.')
    if args.mbsd_grad and args.no_mbsd:
        parser.error('--mbsd-grad and --no-mbsd exclude each other.')
    if args.no_slow_noise and args.gen_arch != 'noise':
        parser.error('--no-slow-noise needs --gen-arch noise.')
    if args.legacy_preprocessing and args.line_noise != 'dss':
        parser.error('--legacy-preprocessing needs --line-noise dss.')
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed, deterministic=args.deterministic)
    run_name = args.run_name or f'{datetime.now():%Y%m%d}_seed{args.seed}'
    out_prefix = f'logs/{run_name}'

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Running on {torch.cuda.get_device_name(0)}')
    elif args.allow_cpu:
        device = torch.device('cpu')
        print('GPU is not available: running on CPU (smoke test only)')
    else:
        raise SystemExit('GPU is not available (use --allow-cpu for a smoke test).')

    # held-out participants
    heldout = None
    if args.heldout_split is not None:
        with open(args.heldout_split) as f:
            heldout = json.load(f)['heldout']

    # load and preprocess the training data
    preprocessing = preprocessing_config(legacy=args.legacy_preprocessing, line_noise=args.line_noise)
    data = load_data(args.data_path,
                     channels='all',
                     n_subjects=N_SUBJECTS_FILE,
                     condition=args.condition,
                     exclude_sub_ids=heldout,
                     preprocess=True,
                     highpass=True,
                     remove_line_noise=preprocessing['line_noise'],
                     filter_first=preprocessing['filter_first'],
                     block_len=preprocessing['block_len'])
    if args.no_subject_cond:  # one shared embedding: the model cannot tell participants apart
        data['sub'] = torch.zeros_like(data['sub'])
    if args.no_state_cond:  # one shared state embedding: the model cannot tell EC from EO
        data['state'] = torch.zeros_like(data['state'])
    n_subjects = 1 if args.no_subject_cond else len(data['subjects'])  # rows of the participant-embedding tables
    n_channels = data['x'].shape[1]

    # build the model and the trainer, or restore them from a checkpoint
    if args.resume_from is not None:
        trainer = Trainer.load(args.resume_from, device=device, amp=False if args.no_amp else None)
        saved = trainer.model.config
        if (saved['n_subjects'], saved['feature_dim']) != (n_subjects, n_channels):
            raise SystemExit(f"The checkpoint was trained with {saved['n_subjects']} embedding rows and "
                             f"{saved['feature_dim']} channels; this run has {n_subjects} and {n_channels}.")
        print(f'Resuming exactly from epoch {trainer.epoch} (step {trainer.global_step}) of {args.resume_from}; '
              'the model and training options are those of the checkpoint.')
    else:
        model = FiLMGAN(time_dim=SEG_LEN,
                        feature_dim=n_channels,
                        latent_dim=LATENT_DIM,
                        n_subjects=n_subjects,
                        use_recon=not args.no_recon,
                        use_mbsd=not args.no_mbsd,
                        use_second_diff=not args.no_second_diff,
                        use_dilation=not args.no_dilation,
                        critic_cond=args.critic_cond,
                        use_contrastive=args.contrastive,
                        subject_cond='encoder' if args.subject_encoder else 'table',
                        recon_mask=args.recon_mask,
                        proj_cov=args.proj_cov,
                        gen_arch=args.gen_arch,
                        spectral_filter=args.spectral_filter,
                        deep_critic=args.deep_critic,
                        spectral_head=not args.no_spectral_head,
                        d_sub=args.d_sub,
                        head_bins=args.head_bins,
                        slow_noise=not args.no_slow_noise,
                        mbsd_grad=args.mbsd_grad)
        trainer = Trainer(model,
                          device=device,
                          amp=False if args.no_amp else None,
                          seed=args.seed,
                          steps_per_epoch=args.steps_per_epoch,
                          contrastive_weight=args.contrastive_weight,
                          contrastive_temperature=args.contrastive_temperature,
                          spectral_weight=args.spectral_weight,
                          ema_decay=args.ema_decay,
                          spectral_groups=args.spectral_groups,
                          spectral_group_size=args.spectral_group_size,
                          embedding_init=args.embedding_init)
    print(f"Mixed precision is {'on (float16)' if trainer.amp else 'off'}")
    hp = trainer.hparams
    print(f"Critic updates per generator update: {hp['n_critic']}")
    print(f"cuDNN: {'deterministic algorithms' if args.deterministic else 'autotuned algorithms (not bit-reproducible)'}")

    # save the run configuration
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    # the original run config is kept; a resumed job writes its own file next to it
    config_path = f'{out_prefix}.config.json' if args.resume_from is None else f'{out_prefix}.resume.config.json'
    with open(config_path, 'w') as f:
        json.dump({'run_name': run_name, 'started': datetime.now().isoformat(timespec='seconds'),
                   'args': vars(args), 'seed': args.seed, 'git': _git_state(), 'torch': str(torch.__version__),
                   'model': trainer.model.config, 'training': trainer.hparams, 'amp': trainer.amp,
                   'cudnn_deterministic': args.deterministic, 'preprocessing': preprocessing,
                   'latent_dim': LATENT_DIM, 'batch_size': BATCH_SIZE,
                   'n_subjects': n_subjects, 'n_heldout': 0 if heldout is None else len(heldout),
                   'subjects': data['subjects'],  # participant id of each embedding index
                   'heldout': heldout}, f, indent=1)

    # train
    sampler = RecordingSampler(data['x'],
                               data['sub'],
                               data['state'],
                               seg_len=SEG_LEN,
                               batch_size=BATCH_SIZE,
                               paired=trainer.model.use_contrastive,
                               n_refs=args.n_refs if trainer.model.subject_cond == 'encoder' else 0,
                               seed=args.seed,
                               device=device,
                               block_len=preprocessing['block_len'],
                               n_groups=hp['spectral_groups'] if hp['spectral_weight'] > 0 else 0)
    trainer.fit(sampler, args.max_epochs, out_prefix, save_freq=args.save_freq)


if __name__ == '__main__':
    main()
