"""Evaluate the generated EEG of one trained model (Section 2.4).

Run from the repository root (a CPU is enough):

    python -m src.EEGModalNet.pipeline.evaluate logs/main_s1_epoch_100.pt

Writes <out>/<run>_epoch_<N>/metrics.json and arrays.npz (arrays per participant, for the figures):

  generation    generated vs real recordings of the training participants, per eye state and feature space:
                Fréchet distance, precision and recall, RDM correlation, similarity and identification
  region_fd     Fréchet distance per region and feature (region_difference: difference of the group means)
  alpha_peak    individual alpha frequency, generated vs real
  joins         join ratio of a 60-s signal generated in a single pass, of 15 generated segments joined end to
                end, and of the first 60 s of the real recording
  ceiling       the same comparisons between the two halves of the real recordings (the reference)
  heldout       the generated population against the held-out participants, and the test of memorisation
  enrolment     held-out participants: an embedding is fitted to the first half of the recording, and the
                generated recording is compared with the second half
"""
import os

import argparse
import json

import numpy as np
import xarray as xr
from scipy.stats import spearmanr, wasserstein_distance

from ...EEGModalNet.evaluation import (REGIONS, REGION_FEATURES, SECOND_ORDER, HIGHER_ORDER, recording_features,
                                       join_ratio, psd, region_features, alpha_peak_frequency, amplitude_stats,
                                       beyond_second_order, higher_order_summary, segment_artefacts,
                                       compare_populations, precision_recall_sweep, mismatched_similarity,
                                       univariate_fd, heldout_population, identification, frechet_distance,
                                       bootstrap_metrics, bootstrap_index, interval)
from ...EEGModalNet.evaluation.models import (load_checkpoint, checkpoint_name, run_config, preprocessing_of,
                                              line_noise_of, generate, generate_continuous, generate_with_embeddings,
                                              generate_like, embed_recordings)
from ...EEGModalNet.evaluation.enrolment import fit_embeddings
from ...EEGModalNet.data.lemon import load_data

CONDITIONS = ('EC', 'EO')
SCALE_FREE = True  # features that do not depend on the amplitude of a channel (--no-scale-free: with amplitude)
SUMMARY = ('fd', 'precision', 'recall', 'rdm_rho', 'identification')
N_BOOT = 2000  # bootstrap samples of participants for the 95% intervals (--n-boot)
CONTINUOUS = 15  # 4-s pieces of a 60-s signal, for the join ratio


def describe(recordings, channels):
    """Features of a list of recordings: feature sets, regional features, spectra, alpha peaks, amplitude, joins."""
    feats = [recording_features(r, scale_free=SCALE_FREE) for r in recordings]
    out = {name: np.stack([f[name] for f in feats]) for name in feats[0]}
    spectra = [psd(r) for r in recordings]
    amp = [amplitude_stats(r) for r in recordings]
    higher = out['beyond_second_order'] if 'beyond_second_order' in out else np.stack(
        [beyond_second_order(np.asarray(r, dtype=np.float64)) for r in recordings])
    return {'families': out, 'higher_order': higher_order_summary(higher, channels),
            'artefacts': segment_artefacts(recordings),
            'regions': np.stack([region_features(r, channels) for r in recordings]),
            'freqs': spectra[0][0], 'psd': np.stack([p for _, p in spectra]),
            'iaf': np.array([alpha_peak_frequency(f, p, channels) for f, p in spectra]),
            'rms': np.stack([a['rms'] for a in amp]), 'kurtosis': np.stack([a['kurtosis'] for a in amp]),
            'join_ratio': np.array([join_ratio(r[:CONTINUOUS]) for r in recordings])}  # first 60 s


def _interval(name, boots, estimate):
    """95% interval of a measure from its bootstrap values (Fréchet distance: estimate +- 1.96 SE, not below 0)."""
    return interval(boots, point=estimate, lower=0.0) if name == 'fd' else interval(boots)


def compare(real, gen, rows=None):
    """Compare two descriptions (rows of `real` select the same participants as `gen`)."""
    sel = (lambda a: a) if rows is None else (lambda a: a[rows])
    out = {'families': {}}
    boots = {}
    index = bootstrap_index(len(next(iter(gen['families'].values()))), N_BOOT) if N_BOOT else None
    for fam, values in gen['families'].items():
        r = sel(real['families'][fam])
        m = compare_populations(r, values)
        m['mismatched_similarity'] = mismatched_similarity(r, values)
        m['pr_by_k'] = precision_recall_sweep(r, values)
        if N_BOOT:  # 95% intervals over participants
            boots[fam] = bootstrap_metrics(r, values, index)
            m['ci95'] = {name: _interval(name, v, m[name]) for name, v in boots[fam].items()}
        out['families'][fam] = m
    n_regions, n_feats = len(REGIONS), len(REGION_FEATURES)
    fd = univariate_fd(sel(real['regions']).reshape(-1, n_regions * n_feats), gen['regions'].reshape(-1, n_regions * n_feats))
    out['region_fd'] = {reg: dict(zip(REGION_FEATURES, map(float, fd.reshape(n_regions, n_feats)[i])))
                        for i, reg in enumerate(REGIONS)}
    # difference between the group means, in standard deviations of the real values (Figure 2e)
    difference = (gen['regions'].mean(0) - sel(real['regions']).mean(0)) / sel(real['regions']).std(0)
    out['region_difference'] = {reg: dict(zip(REGION_FEATURES, map(float, difference[i]))) for i, reg in enumerate(REGIONS)}
    ri, gi = sel(real['iaf']), gen['iaf']
    both = np.isfinite(ri) & np.isfinite(gi)
    out['alpha_peak'] = {'valid_real': float(np.isfinite(ri).mean()), 'valid_generated': float(np.isfinite(gi).mean()),
                         'wasserstein_hz': float(wasserstein_distance(ri[np.isfinite(ri)], gi[np.isfinite(gi)]))
                         if np.isfinite(ri).any() and np.isfinite(gi).any() else float('nan'),
                         'paired_r': float(np.corrcoef(ri[both], gi[both])[0, 1]) if both.sum() > 2 else float('nan')}
    out['joins'] = {'generated': float(gen['join_ratio'].mean()), 'real': float(sel(real['join_ratio']).mean())}
    out['amplitude'] = {'kurtosis_generated': float(gen['kurtosis'].mean()), 'kurtosis_real': float(sel(real['kurtosis']).mean()),
                        'rms_ratio': float((gen['rms'] / sel(real['rms'])).mean())}
    rh, gh = sel(real['higher_order']), gen['higher_order']  # medians over participants; rank agreement between them
    out['beyond_second_order'] = {name: {'generated': float(np.median(gh[:, k])), 'real': float(np.median(rh[:, k])),
                                         'spearman': float(spearmanr(gh[:, k], rh[:, k]).statistic)}
                                  for k, name in enumerate(HIGHER_ORDER)}
    out['artefacts'] = gen['artefacts']
    if all(f in out['families'] for f in SECOND_ORDER):
        out['summary'] = {m: float(np.mean([out['families'][f][m] for f in SECOND_ORDER])) for m in SUMMARY}
        if N_BOOT:
            out['summary_ci95'] = {m: _interval(m, np.mean([boots[f][m] for f in SECOND_ORDER], axis=0), out['summary'][m])
                                   for m in SUMMARY}
    return out


class ModelSource:
    """Generated recordings from a trained model."""

    def __init__(self, model, checkpoint, embedding, enrol_reference, args):
        self.model = model
        self.checkpoint = checkpoint
        self.embedding = embedding
        self.reference = enrol_reference
        self.args = args

    def train(self, state, real_train, seed):
        if self.model.subject_cond == 'encoder':  # the participant is inferred from their own recording
            return generate_like(self.model, real_train, state, self.args.n_seg, seed)
        return generate(self.model, self.embedding, state, self.args.n_seg, seed)

    def enrol(self, state, first_halves, real_train, n_seg, seed):
        """Recordings for held-out participants from an embedding fitted to their first half, and from the mean embedding."""
        model = self.model
        if model.subject_cond == 'encoder':  # the embedding is one pass of the generator's encoder
            emb, loss = embed_recordings(model.generator, first_halves), np.full(len(first_halves), np.nan)
            mean_emb = embed_recordings(model.generator, real_train).mean(0)
        else:
            emb, loss = fit_embeddings(self.checkpoint, first_halves, state, steps=self.args.enrol_steps,
                                       batch=self.args.enrol_batch, model=model, reference=self.reference)
            mean_emb = model.generator.sub_emb.weight.detach().mean(0).numpy()
        return (generate_with_embeddings(model, emb, state, n_seg, seed),
                generate_with_embeddings(model, np.tile(mean_emb, (len(emb), 1)), state, n_seg, seed + 2), emb, loss)

    def continuous(self, state, n_samples, seed):
        """Signals generated in a single pass for the training participants (noise-sequence generator only)."""
        if self.model.gen_arch != 'noise' or self.model.subject_cond != 'table':
            return None
        return generate_continuous(self.model, self.embedding, state, n_samples, seed)


def average(dicts):
    """Mean and SD over generation seeds of a list of identically structured nested dicts."""
    first = dicts[0]
    if isinstance(first, dict):
        return {k: average([d[k] for d in dicts]) for k in first}
    if isinstance(first, (tuple, list)):
        return [average([d[i] for d in dicts]) for i in range(len(first))]
    values = np.array(dicts, dtype=float)
    return {'mean': float(np.nanmean(values)), 'sd': float(np.nanstd(values))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', help='checkpoint (.pt) of the run to evaluate')
    parser.add_argument('--data-path', default='data/LEMON_DATA/EC_ch-8_sf-128.nc5')
    parser.add_argument('--heldout-split', default='splits/lemon_heldout40.json',
                        help='used only when the checkpoint has no .config.json next to it')
    parser.add_argument('--gen-seeds', type=int, default=3, help='number of independent generation seeds')
    parser.add_argument('--n-seg', type=int, default=120, help='4-s segments per generated recording')
    parser.add_argument('--enrol-steps', type=int, default=150)
    parser.add_argument('--enrol-batch', type=int, default=32, help='generated segments per participant and step')
    parser.add_argument('--skip-enrolment', action='store_true')
    parser.add_argument('--enrol-loss', default='auto', choices=['auto', 'spectral', 'correlation'],
                        help="loss for fitting the embeddings: 'spectral' (the spectral loss of training), "
                             "'correlation' (log spectrum and channel correlation) or 'auto' (spectral if the "
                             "checkpoint has the reference level of the loss)")
    parser.add_argument('--max-participants', type=int, default=None, help='smoke tests: use a subset')
    parser.add_argument('--out', default='logs/eval')
    parser.add_argument('--scale-free', action=argparse.BooleanOptionalAction, default=True,
                        help='features that do not depend on the amplitude of a channel (the features of the paper)')
    parser.add_argument('--n-boot', type=int, default=2000,
                        help='bootstrap samples of participants for the 95%% intervals of the population metrics (0: none)')
    parser.add_argument('--live-weights', action='store_true',
                        help="evaluate the generator's weights as they were being trained, not their running average "
                             '(output: <run>_epoch_<N>_live)')
    args = parser.parse_args(argv)
    global SCALE_FREE, N_BOOT
    SCALE_FREE, N_BOOT = args.scale_free, args.n_boot

    # real data: all participants, both eye states, preprocessed as for training
    channels = [str(c) for c in xr.open_dataarray(args.data_path, engine='h5netcdf').channel.values]
    filter_first, block_len = preprocessing_of(args.checkpoint)
    line_noise = line_noise_of(args.checkpoint)
    name = checkpoint_name(args.checkpoint) + ('_live' if args.live_weights else '')
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    data = load_data(args.data_path,
                     channels='all',
                     n_subjects=202,
                     condition='both',
                     exclude_sub_ids=None,
                     preprocess=True,
                     highpass=True,
                     remove_line_noise=line_noise,
                     filter_first=filter_first,
                     block_len=block_len)
    subjects, x = data['subjects'], data['x'].numpy()
    n, seg_len = len(subjects), 512
    n_real_seg = x.shape[-1] // seg_len
    half = n_real_seg // 2

    # load the model
    model = load_checkpoint(args.checkpoint, ema=not args.live_weights)
    enrol_reference = None
    if args.enrol_loss != 'correlation' and args.checkpoint.endswith('.pt'):
        import torch
        enrol_reference = torch.load(args.checkpoint, map_location='cpu', weights_only=True).get('spectral_reference')
    if args.enrol_loss == 'spectral' and enrol_reference is None:
        raise SystemExit('--enrol-loss spectral needs a checkpoint of a model trained with the spectral loss.')
    print('enrolment loss:', 'spectral loss of training' if enrol_reference is not None else 'log spectrum and channel correlation')
    cfg = run_config(args.checkpoint, subjects, args.heldout_split)
    index = {s: i for i, s in enumerate(subjects)}
    # training and held-out participants of the run
    train_rows = [index[s] for s in cfg['train'] if s not in set(cfg['heldout'])]
    held_rows = [index[s] for s in cfg['heldout']]
    if args.max_participants:
        train_rows, held_rows = train_rows[:args.max_participants], held_rows[:max(6, args.max_participants // 4)]
    embedding = [0 if cfg['no_subject_cond'] else cfg['train'].index(subjects[r]) for r in train_rows]

    def real_recordings(state, rows):
        block = x[state * n:(state + 1) * n][rows][:, :, :n_real_seg * seg_len]
        return [r.reshape(r.shape[0], n_real_seg, seg_len).transpose(1, 2, 0) for r in block]

    source = ModelSource(model, args.checkpoint, embedding, enrol_reference, args)

    results = {'run': name, 'checkpoint': args.checkpoint, 'n_train': len(train_rows),
               'n_heldout': len(held_rows), 'heldout_is_unseen': cfg['truly_heldout'], 'gen_seeds': args.gen_seeds,
               'n_seg': args.n_seg, 'critic_cond': model.critic_cond, 'subject_cond': model.subject_cond, 'channels': channels,
               'preprocessing': {'line_noise': line_noise, 'filter_first': filter_first, 'block_len': block_len},
               'scale_free': SCALE_FREE, 'averaged_weights': not args.live_weights, 'generation': {}, 'ceiling': {}, 'heldout': {}, 'enrolment': {}}
    arrays = {'train_subjects': np.array([subjects[r] for r in train_rows]),
              'heldout_subjects': np.array([subjects[r] for r in held_rows])}

    # evaluate each eye state
    for state, cond in enumerate(CONDITIONS):
        gen_state = 0 if cfg['no_state_cond'] else state
        real_train = real_recordings(state, train_rows)
        real = describe(real_train, channels)
        first, second = describe([r[:half] for r in real_train], channels), describe([r[half:] for r in real_train], channels)
        results['ceiling'][cond] = compare(second, first)  # real first half vs real second half

        per_seed, gen0 = [], None
        for seed in range(args.gen_seeds):
            generated = source.train(gen_state, real_train, 1000 * seed + state)
            gen = describe(generated, channels)
            comparison = compare(real, gen)
            # 60-s signals generated in a single pass: are they as smooth where the segments would join?
            continuous = source.continuous(gen_state, CONTINUOUS * seg_len, 1000 * seed + state)
            if continuous is not None:
                pieces = [c.reshape(CONTINUOUS, seg_len, -1) for c in continuous]
                comparison['joins']['single_pass'] = float(np.mean([join_ratio(p) for p in pieces]))
            per_seed.append(comparison)
            if gen0 is None:  # keep the arrays of the first seed and a few example segments for the figures
                gen0 = gen
                arrays[f'{cond}/example_real'] = np.stack([r[0] for r in real_train[:6]])
                arrays[f'{cond}/example_gen'] = np.stack([g[0] for g in generated[:6]])
        results['generation'][cond] = average(per_seed)

        arrays.update({f'{cond}/freqs': real['freqs'], f'{cond}/psd_real': real['psd'], f'{cond}/psd_gen': gen0['psd'],
                       f'{cond}/iaf_real': real['iaf'], f'{cond}/iaf_gen': gen0['iaf'],
                       f'{cond}/higher_order_real': real['higher_order'], f'{cond}/higher_order_gen': gen0['higher_order'],
                       f'{cond}/regions_real': real['regions'], f'{cond}/regions_gen': gen0['regions']})
        for fam in real['families']:
            arrays[f'{cond}/{fam}_real'], arrays[f'{cond}/{fam}_gen'] = real['families'][fam], gen0['families'][fam]

        # generated population against the held-out participants
        if held_rows:
            real_held_rec = real_recordings(state, held_rows)
            held = describe(real_held_rec, channels)
            results['heldout'][cond] = {fam: heldout_population(real['families'][fam], held['families'][fam], gen0['families'][fam])
                                        for fam in real['families']}
            for fam in held['families']:
                arrays[f'{cond}/{fam}_heldout_real'] = held['families'][fam]

            # fit an embedding to the first half and compare the generated recording with the second half
            if not args.skip_enrolment:
                held_a = describe([r[:half] for r in real_held_rec], channels)
                held_b = describe([r[half:] for r in real_held_rec], channels)
                enrolled, from_mean, emb, loss = source.enrol(gen_state, [r[:half] for r in real_held_rec], real_train,
                                                              half, 7 + state)
                enrolled = describe(enrolled, channels)
                from_mean = describe(from_mean, channels)
                res = {}
                for fam in held_b['families']:
                    b = held_b['families'][fam]
                    row = {}
                    for label, other in (('enrolled', enrolled), ('mean_embedding', from_mean), ('real_first_half', held_a)):
                        top1, sim = identification(b, other['families'][fam])
                        row[label] = {'identification': top1, 'similarity': sim,
                                      'mismatched_similarity': mismatched_similarity(b, other['families'][fam]),
                                      'fd': frechet_distance(b, other['families'][fam])}
                    res[fam] = row
                results['enrolment'][cond] = {'families': res, 'fit_loss_median': float(np.median(loss)),
                                              'chance_identification': 1.0 / len(held_rows)}
                if all(f in res for f in SECOND_ORDER):
                    results['enrolment'][cond]['summary'] = {
                        label: {m: float(np.mean([res[f][label][m] for f in SECOND_ORDER])) for m in ('identification', 'fd')}
                        for label in ('enrolled', 'mean_embedding', 'real_first_half')}
                arrays[f'{cond}/enrolled_embeddings'] = emb
                for fam in enrolled['families']:
                    arrays[f'{cond}/{fam}_enrolled'] = enrolled['families'][fam]
                    arrays[f'{cond}/{fam}_from_mean_embedding'] = from_mean['families'][fam]
                    arrays[f'{cond}/{fam}_heldout_first_half'] = held_a['families'][fam]
                    arrays[f'{cond}/{fam}_heldout_second_half'] = held_b['families'][fam]
        print(f'{name}: {cond} done', flush=True)

    # save
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    np.savez_compressed(os.path.join(out_dir, 'arrays.npz'), **arrays)

    # short printed summary
    for cond in CONDITIONS:
        print(f'\n{cond}: generated vs real (mean over {args.gen_seeds} generation seeds) | ceiling = real halves')
        for fam, m in results['generation'][cond]['families'].items():
            c = results['ceiling'][cond]['families'][fam]
            print(f"  {fam:11s} FD {m['fd']['mean']:.3f} ({c['fd']:.3f}) | precision {m['precision']['mean']:.2f} ({c['precision']:.2f})"
                  f" | recall {m['recall']['mean']:.2f} ({c['recall']:.2f}) | RDM {m['rdm_rho']['mean']:.2f} ({c['rdm_rho']:.2f})"
                  f" | identification {m['identification']['mean']:.2f} ({c['identification']:.2f})")
        g, c = results['generation'][cond], results['ceiling'][cond]
        if 'summary' in g:
            print('  four feature spaces: ' + ' | '.join(f"{m} {g['summary'][m]['mean']:.3f} ({c['summary'][m]:.3f})" for m in SUMMARY))
        print('  non-Gaussian properties, generated (real): ' + ' | '.join(
            f"{k} {v['generated']['mean']:.3f} ({v['real']['mean']:.3f})" for k, v in g['beyond_second_order'].items()))
        print('  join ratio: ' + ' | '.join(f"{k} {v['mean']:.2f}" for k, v in g['joins'].items()))
        if cond in results['heldout']:
            print('  memorisation (0.5: none): ' + ' | '.join(
                f"{fam} {h['memorisation']:.2f}" for fam, h in results['heldout'][cond].items()))
        if cond in results['enrolment']:
            for fam, row in results['enrolment'][cond]['families'].items():
                print(f"  enrolment {fam:11s} identification: enrolled {row['enrolled']['identification']:.2f}, "
                      f"mean embedding {row['mean_embedding']['identification']:.2f}, real halves {row['real_first_half']['identification']:.2f}")
    print(f'\nwrote {out_dir}/metrics.json and arrays.npz')
    return results


if __name__ == '__main__':
    main()
