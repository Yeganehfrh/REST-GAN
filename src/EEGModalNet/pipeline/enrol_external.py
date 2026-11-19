"""Fit the model to the participants of another data set (Section 2.4.3).

Run from the repository root (a CPU is enough):

    python -m src.EEGModalNet.pipeline.enrol_external --data data/ds005385/ds005385.nc5 --out logs/enrol_external \\
        logs/main_s1_epoch_100.pt logs/main_s2_epoch_100.pt logs/main_s3_epoch_100.pt

The checkpoints are training seeds of one model. For every checkpoint, an embedding is fitted to the first
--n-fit segments of each participant (evaluation/enrolment.py). The recording generated from it is compared
with the last --n-fit segments, next to two references: recordings generated from the mean of the learned
embeddings, and the participant's own first segments.

Measures, per feature space and as the mean over the four feature spaces of the paper:
  similarity      median correlation between the features of the generated recording and of the own last segments
  identification  share of participants whose generated recording is closest to their own last segments
  fd              Fréchet distance between the generated recordings and the last segments

Writes metrics.json, report.md, arrays.npz and fits/<run>.npz per checkpoint to <out>. An existing metrics.json
is never overwritten; --fit-only stops after the fits. --data can be given several times, for recordings that are
stored in several files.
"""
import argparse
import json
import os

import numpy as np

from .decode import load_segments
from ...EEGModalNet.evaluation import (SECOND_ORDER, recording_features, paired_similarity, mismatched_similarity,
                                       frechet_distance, bootstrap_index)
from ...EEGModalNet.evaluation.enrolment import fit_embeddings
from ...EEGModalNet.evaluation.models import load_checkpoint, checkpoint_name, line_noise_of, generate_with_embeddings

STATES = ('EC', 'EO')
MODEL = 'fitted embedding'
MEAN = 'mean embedding'
REAL = 'real first segments'


def families(recordings):
    """Feature sets of a list of recordings (n_segments, T, C): dict name -> (recordings, features)."""
    feats = [recording_features(np.asarray(r), scale_free=True) for r in recordings]
    return {name: np.stack([f[name] for f in feats]) for name in feats[0]}


def fit_model(checkpoint, first, state, n_gen, chunk=40, steps=150, batch=32):
    """Fit embeddings to the first segments of every participant, in groups of `chunk`.
    Returns the embeddings and the features of what is generated from them and from the mean embedding.
    """
    import torch
    model = load_checkpoint(checkpoint)
    reference = torch.load(checkpoint, map_location='cpu', weights_only=True).get('spectral_reference')
    mean = model.generator.sub_emb.weight.detach().mean(0).numpy()
    emb, loss, fitted, from_mean = [], [], [], []
    for c, start in enumerate(range(0, len(first), chunk)):
        e, l = fit_embeddings(checkpoint, list(first[start:start + chunk]), state, steps=steps, batch=batch, seed=c,
                              model=model, reference=reference)
        fitted += generate_with_embeddings(model, e, state, n_gen, 7 + state + 1000 * c)
        from_mean += generate_with_embeddings(model, np.tile(mean, (len(e), 1)), state, n_gen, 9 + state + 1000 * c)
        emb.append(e), loss.append(l)
        print(f'  {checkpoint_name(checkpoint)}: {min(start + chunk, len(first))} of {len(first)} participants', flush=True)
    out = {'embeddings': np.concatenate(emb), 'fit_loss': np.concatenate(loss),
           'spectral_loss': np.array(reference is not None)}
    for label, recs in ((MODEL, fitted), (MEAN, from_mean)):
        out.update({f'{label}/{fam}': v for fam, v in families(recs).items()})
    return out


def summarise(last, sources, n_boot, seed=0):
    """Measures per source, the bootstrap values of the summaries and the similarity of every participant.
    last: name -> (n, features) of the last segments. sources: label -> list (one per checkpoint) of such dicts.
    """
    fams = list(last)
    n = len(next(iter(last.values())))
    index = bootstrap_index(n, n_boot, seed)
    second = [fams.index(f) for f in SECOND_ORDER]
    results, boots, per_participant = {}, {}, {}
    for label, seeds in sources.items():
        sim, hit = (np.empty((len(seeds), len(fams), n)) for _ in range(2))
        floor, fd = (np.empty((len(seeds), len(fams))) for _ in range(2))
        for s, feats in enumerate(seeds):
            for f, fam in enumerate(fams):
                sim[s, f], hit[s, f] = paired_similarity(last[fam], feats[fam])
                floor[s, f], fd[s, f] = mismatched_similarity(last[fam], feats[fam]), frechet_distance(last[fam], feats[fam])
        per_family = {'similarity': np.median(sim, -1).mean(0), 'floor': floor.mean(0),
                      'identification': hit.mean(-1).mean(0), 'fd': fd.mean(0)}
        b_sim = np.median(sim[:, second][:, :, index], -1).mean((0, 1))        # (n_boot,)
        b_hit = hit[:, second][:, :, index].mean(-1).mean((0, 1))
        results[label] = {
            'summary': {m: float(v[second].mean()) for m, v in per_family.items()},
            'summary_ci95': {'similarity': np.percentile(b_sim, [2.5, 97.5]).tolist(),
                             'identification': np.percentile(b_hit, [2.5, 97.5]).tolist()},
            'families': {fam: {m: float(v[f]) for m, v in per_family.items()} for f, fam in enumerate(fams)},
            'per_checkpoint': {'similarity': np.median(sim[:, second], -1).mean(1).tolist(),
                               'identification': hit[:, second].mean((1, 2)).tolist()}}
        boots[label] = {'similarity': b_sim, 'identification': b_hit}
        per_participant[label] = sim[:, second].mean((0, 1))
    return results, boots, per_participant


def compare(results, boots, a, b):
    """Summary of source a minus that of source b, with the paired bootstrap."""
    out = {}
    for m in ('similarity', 'identification'):
        d = boots[a][m] - boots[b][m]
        out[m] = {'value': results[a]['summary'][m] - results[b]['summary'][m],
                  'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'p_greater': float((d <= 0).mean())}
    return out


def write_report(path, res):
    rows = ['| Source | Similarity to own last segments [95% interval] | Floor (other participants) | Identification [95% interval] | Fréchet distance |',
            '|---|---|---|---|---|']
    for label, r in res['sources'].items():
        s, ci = r['summary'], r['summary_ci95']
        rows.append(f"| {label} | {s['similarity']:.3f} [{ci['similarity'][0]:.3f}, {ci['similarity'][1]:.3f}] | {s['floor']:.3f} | "
                    f"{s['identification']:.3f} [{ci['identification'][0]:.3f}, {ci['identification'][1]:.3f}] | {s['fd']:.3f} |")
    fam_rows = ['| Source | ' + ' | '.join(res['families']) + ' |', '|---|' + '---|' * len(res['families'])]
    for label, r in res['sources'].items():
        fam_rows.append(f'| {label} | ' + ' | '.join(f"{r['families'][f]['similarity']:.2f} / {r['families'][f]['identification']:.2f}"
                                                    for f in res['families']) + ' |')
    tests = ['| Comparison | Measure | Difference [95% interval] | p (one-sided) |', '|---|---|---|---|']
    for label, t in res['comparisons'].items():
        for m, d in t.items():
            tests.append(f"| {label} | {m} | {d['value']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] | {d['p_greater']:.4f} |")
    with open(path, 'w') as f:
        f.write(f"The model fitted to {res['n_participants']} participants of {res['data']} ({res['state']}): embeddings "
                f"fitted to the first {res['n_fit']} segments, generated recordings compared with the last {res['n_fit']}. "
                f"{len(res['checkpoints'])} checkpoint(s); mean over the four feature spaces of the paper; intervals from "
                f"{res['n_boot']} bootstrap samples of participants. Chance level of identification: "
                f"{res['chance_identification']:.4f}.\n\n"
                + '\n'.join(rows) + '\n\n## Per feature space: similarity / identification\n\n' + '\n'.join(fam_rows)
                + '\n\n## Comparisons\n\n' + '\n'.join(tests) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', help='checkpoints (.pt) of the training seeds of one model')
    parser.add_argument('--data', required=True, action='append',
                        help='recordings (subject, channel, time) at 128 Hz, not preprocessed (can be given several times)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--state', default='EC', choices=STATES, help='the eye state of the recordings')
    parser.add_argument('--n-fit', type=int, default=22, help='segments to fit on, and to compare with')
    parser.add_argument('--chunk', type=int, default=40, help='participants fitted together')
    parser.add_argument('--enrol-steps', type=int, default=150)
    parser.add_argument('--enrol-batch', type=int, default=32)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--training-data', default='data/LEMON_DATA/EC_ch-8_sf-128.nc5',
                        help='the training data of the model: its channels, in its order, are taken from --data')
    parser.add_argument('--max-participants', type=int, default=None, help='smoke tests: use a subset')
    parser.add_argument('--fit-only', action='store_true', help='fit the given checkpoints and stop')
    args = parser.parse_args(argv)

    result_path = os.path.join(args.out, 'metrics.json')
    if os.path.exists(result_path):
        raise SystemExit(f'{result_path} exists; results are not overwritten.')
    os.makedirs(os.path.join(args.out, 'fits'), exist_ok=True)
    # recordings of the participants, cut into segments
    line_noise = line_noise_of(args.checkpoints[0])
    if any(line_noise_of(c) != line_noise for c in args.checkpoints):
        raise SystemExit('The checkpoints were trained on data with different preprocessing.')
    subjects, segments = load_segments(args.data, args.training_data, line_noise)
    if args.max_participants:
        subjects, segments = subjects[:args.max_participants], segments[:args.max_participants]
    if segments.shape[1] < 2 * args.n_fit:
        raise SystemExit(f'{segments.shape[1]} segments per participant: too few for {args.n_fit} to fit and {args.n_fit} to compare.')
    first, last = segments[:, :args.n_fit], segments[:, -args.n_fit:]
    state = STATES.index(args.state)
    print(f'{len(subjects)} participants, {segments.shape[1]} segments each; first and last {args.n_fit} used', flush=True)

    def cached(name, compute):
        """Computed once per output directory; a stored result must belong to the same participants."""
        path = os.path.join(args.out, 'fits', name + '.npz')
        if os.path.exists(path):
            stored = dict(np.load(path, allow_pickle=True))
            if list(stored['subjects'].astype(str)) != list(subjects) or int(stored['n_fit']) != args.n_fit:
                raise SystemExit(f'{path} belongs to other participants or segments.')
            print(f'{name}: reused', flush=True)
            return stored
        result = dict(compute(), subjects=np.array(subjects), n_fit=np.array(args.n_fit))
        np.savez(path + '.tmp.npz', **result)
        os.replace(path + '.tmp.npz', path)
        return result

    # fit the embeddings (stored per checkpoint)
    fits = [cached(checkpoint_name(c), lambda c=c: fit_model(c, first, state, args.n_fit, args.chunk,
                                                              args.enrol_steps, args.enrol_batch))
            for c in args.checkpoints]
    if args.fit_only:
        return None

    # compare every source with the last segments
    real_first, real_last = families(first), families(last)
    fams = list(real_last)
    pick = lambda stored, label: {fam: stored[f'{label}/{fam}'] for fam in fams}
    sources = {MODEL: [pick(f, MODEL) for f in fits], MEAN: [pick(f, MEAN) for f in fits], REAL: [real_first]}
    results, boots, per_participant = summarise(real_last, sources, args.n_boot)
    res = {'data': ', '.join(args.data), 'state': args.state, 'n_participants': len(subjects), 'n_fit': args.n_fit,
           'n_segments': int(segments.shape[1]), 'checkpoints': list(args.checkpoints), 'n_boot': args.n_boot,
           'line_noise': line_noise, 'spectral_loss': bool(all(f['spectral_loss'] for f in fits)),
           'chance_identification': 1.0 / len(subjects), 'families': fams, 'sources': results,
           'fit_loss_median': [float(np.median(f['fit_loss'])) for f in fits],
           'comparisons': {f'{MODEL} - {MEAN}': compare(results, boots, MODEL, MEAN)}}

    # save
    with open(result_path, 'w') as f:
        json.dump(res, f, indent=1, default=float)
    np.savez_compressed(os.path.join(args.out, 'arrays.npz'), subjects=np.array(subjects),
                        embeddings=np.stack([f['embeddings'] for f in fits]),
                        **{f'similarity/{label}': v for label, v in per_participant.items()})
    write_report(os.path.join(args.out, 'report.md'), res)
    with open(os.path.join(args.out, 'report.md')) as f:
        print('\n' + f.read())
    return res


if __name__ == '__main__':
    main()
