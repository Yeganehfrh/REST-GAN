"""Decode sex and age from features of EEG segments: the critic against established features (Section 2.5).

Run from the repository root. Three steps, each a subcommand:

    python -m src.EEGModalNet.pipeline.decode extract --data data/ds005385/ds005385.nc5 \\
        --demographics data/ds005385/demographic.csv --out logs/decoding/features.npz \\
        critic_s1=logs/main_s1_epoch_100.pt
    python -m src.EEGModalNet.pipeline.decode cv --features logs/decoding/features.npz \\
        --sets bandpower,filterbank,critic_s1,filterbank+critic_s1 --out logs/decoding/cv.jsonl --jobs 3
    python -m src.EEGModalNet.pipeline.decode report --features logs/decoding/features.npz \\
        --results logs/decoding/cv.jsonl --critic critic_s1 --holm critic_s1 bandpower --out logs/decoding/report

extract   cuts the recordings into 4-s segments and computes per segment: band power, filter-bank covariance,
          the 1-Hz cross-spectrum (xspec) and, for every NAME=checkpoint, the readouts of the critic (covariance
          of the second layer as NAME, time averages as NAME:mean). NAME=checkpoint@mean, @zero or @none sets
          the stand-in for the participant of a critic with FiLM at its input. Features computed elsewhere
          (e.g. CBraMod) are added with --plain NAME=file.npy. --data and --demographics can be given several
          times, for recordings that are stored in several files.
cv        5-fold cross-validation over participants, repeated 10 times, for every feature set and target.
          '+' combines sets; NAME=a|b|c is a choice between the sets a, b and c made inside every training fold.
          Results are appended, so an interrupted call can be continued.
report    scores with bootstrap intervals (AUROC; R2 and mean absolute error for age in years) and paired
          comparisons between feature sets (--compare; --holm for the comparisons whose p values are corrected
          together with Holm's method).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from ...EEGModalNet.data.lemon import preprocess_recordings
from ...EEGModalNet.evaluation import decoding as dec
from ...EEGModalNet.evaluation.models import load_checkpoint, line_noise_of

TARGETS = ('age_group', 'sex', 'age')
N_REP = 10


def load_segments(data, channels_from, line_noise):
    """Recordings (subject, channel, time; 128 Hz, not preprocessed) -> participant ids and 4-s segments
    (participants, n, 512, channels), in the channels of the model and preprocessed like the training data.
    data: one file, or a list of files whose participants are joined in the given order.
    """
    reference = [str(c) for c in xr.open_dataarray(channels_from, engine='h5netcdf').channel.values]
    subjects, segments = [], []
    for path in [data] if isinstance(data, str) else data:
        eeg = xr.open_dataarray(path, engine='h5netcdf').sel(channel=reference)
        x = preprocess_recordings(eeg.to_numpy().astype(np.float64), filter_first=True, block_len=None, line_noise=line_noise)
        subjects.append(eeg.subject.values.astype(str))
        segments.append(dec.cut_segments(x))
    return np.concatenate(subjects), np.concatenate(segments)


def extract(args):
    subjects, segments = load_segments(args.data, args.channels_from, args.line_noise)
    demog = pd.concat([pd.read_csv(path) for path in args.demographics]).set_index('participant_id').loc[subjects]
    n = segments.shape[1]
    out = dict(np.load(args.out, allow_pickle=True)) if os.path.exists(args.out) else {}
    if 'groups' in out and not (out['groups'].astype(str) == np.repeat(subjects, n)).all():
        raise SystemExit(f'{args.out} holds features of other recordings or segments.')

    # participant and targets of every segment
    out.update(groups=np.repeat(subjects, n), line_noise=np.array(args.line_noise),
               age_group=np.repeat((demog['age_group'].to_numpy() != 'young').astype(int), n),
               sex=np.repeat((demog['sex'].to_numpy() == 'M').astype(int), n),
               age=np.repeat(demog['age'].to_numpy(float), n))

    # band power and filter-bank covariance
    if 'bandpower' not in out:
        out['bandpower'] = dec.band_power_features(segments)
        out['filterbank'] = dec.filter_bank_covariances(segments)
        print('band power and filter-bank covariances done', flush=True)

    # 1-Hz cross-spectrum
    if 'xspec' not in out:
        out['xspec'] = dec.cross_spectrum_features(segments)
        print('1-Hz cross-spectrum done', flush=True)

    # readouts of the critics
    for item in args.checkpoints:
        name, path = item.split('=', 1)
        path, _, background = path.partition('@')
        if line_noise_of(path) != args.line_noise:
            raise SystemExit(f'{path} was trained on data with line-noise step {line_noise_of(path)!r}, not {args.line_noise!r}.')
        readouts = dec.critic_readouts(load_checkpoint(path).critic, segments, background or None, args.extra_readouts)
        out[name], out[f'{name}:mean'] = readouts['cov_h2'], readouts['mean']
        for k in dec.EXTRA_READOUTS if args.extra_readouts else ():
            out[f'{name}:{k}'] = readouts[k]
        print(name, 'done', flush=True)

    # features computed elsewhere (e.g. CBraMod)
    if args.plain_subjects and not (np.repeat(np.load(args.plain_subjects).astype(str), n) == out['groups']).all():
        raise SystemExit(f'The rows behind {args.plain_subjects} are not the segments of {args.data}, in its order.')
    for item in args.plain:
        name, path = item.split('=', 1)
        values = np.load(path)
        if len(values) != len(out['groups']):
            raise SystemExit(f'{path}: {len(values)} rows for {len(out["groups"])} segments.')
        out[name] = values.astype(np.float32)

    # save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f'wrote {args.out}: {len(subjects)} participants, {n} segments each; feature sets:',
          sorted(k for k in out if k not in ('groups', 'line_noise', *TARGETS)))


def blocks_of(features, fset):
    """A feature set ('a+b') as blocks: 4-D arrays are covariance blocks, 2-D arrays plain features."""
    return [('cov' if features[part].ndim == 4 else 'plain', features[part]) for part in fset.split('+')]


def parse_set(fset):
    """'a+b' -> ('a+b', None); 'NAME=a|b+c' -> ('NAME', ['a', 'b+c']): a choice between sets, made in every fold."""
    name, _, alternatives = fset.partition('=')
    return (name, alternatives.split('|')) if alternatives else (fset, None)


def _job(path, fset, target, rep):
    name, alternatives = parse_set(fset)
    parts = {p for s in (alternatives or [fset]) for p in s.split('+')}
    features = np.load(path, allow_pickle=True)
    features = {k: features[k] for k in parts | {'groups', target}}
    groups = features['groups'].astype(str)
    by_participant = pd.Series(features[target], index=groups).groupby(level=0).first()
    if alternatives is None:
        run = dec.regress if target == 'age' else dec.classify
        return {'target': target, 'fset': fset, 'rep': rep, **run(blocks_of(features, fset), groups, by_participant, rep)}
    run = dec.regress_choice if target == 'age' else dec.classify_choice
    result = run({s: blocks_of(features, s) for s in alternatives}, groups, by_participant, rep)
    return {'target': target, 'fset': name, 'rep': rep, 'candidates': alternatives, **result}


def cross_validate(args):
    from joblib import Parallel, delayed
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, '1')  # one thread per job: the jobs run side by side
    # skip what is already in the results file
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r['target'], r['fset'], r['rep']) for r in map(json.loads, f)}
    todo = [(args.features, fset, target, rep) for target in args.targets for fset in args.sets.split(',')
            for rep in range(args.repeats) if (target, parse_set(fset)[0], rep) not in done]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for result in Parallel(n_jobs=args.jobs, return_as='generator_unordered')(delayed(_job)(*job) for job in todo):
        with open(args.out, 'a') as f:
            f.write(json.dumps(result, default=float) + '\n')
    print(len(todo), 'jobs done')


def load_predictions(results_path, participants, repeats=N_REP):
    """{(target, fset): participant-level predictions averaged over repeats, in the order of `participants`}."""
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]
    out = {}
    for key in {(r['target'], r['fset']) for r in rows}:
        mine = [r for r in rows if (r['target'], r['fset']) == key]
        if len({r['rep'] for r in mine}) >= repeats:
            out[key] = pd.DataFrame([r['p'] for r in mine]).mean(0).loc[participants].to_numpy()
    return out


def choice_counts(results_path):
    """How often each set was chosen inside the training folds, for the choices (NAME=a|b|c) in a results file."""
    with open(results_path) as f:
        rows = [r for r in map(json.loads, f) if 'choice' in r]
    counts = [{'target': r['target'], 'features': r['fset'], 'chosen': c} for r in rows for c in r['choice']]
    if not counts:
        return pd.DataFrame(columns=['target', 'features', 'chosen', 'folds', 'share'])
    table = pd.DataFrame(counts).value_counts().rename('folds').reset_index()
    table['share'] = table.folds / table.groupby(['target', 'features']).folds.transform('sum')
    return table.sort_values(['target', 'features', 'folds'], ascending=[True, True, False], ignore_index=True)


def report(args):
    features = np.load(args.features, allow_pickle=True)
    groups = features['groups'].astype(str)
    participants = np.array(sorted(set(groups)))
    truth = {t: pd.Series(features[t], index=groups).groupby(level=0).first().loc[participants].to_numpy() for t in TARGETS}
    pred = load_predictions(args.results, participants, args.repeats)
    rows = {t: dec.bootstrap_rows(truth[t], args.n_boot, seed=0, stratified=t != 'age') for t in TARGETS}
    score = {t: dec.r_squared if t == 'age' else dec.auroc for t in TARGETS}
    have = lambda t, names: bool(names) and all((t, n) in pred for n in names)
    get = lambda t, names: [pred[t, n] for n in names]

    # scores
    table = []
    for (target, fset) in sorted(pred):
        e = dec.estimate(truth[target], pred[target, fset], score[target], rows[target])
        row = {'target': target, 'features': fset, 'score': e['value'], 'ci_low': e['ci95'][0], 'ci_high': e['ci95'][1]}
        if target == 'age':
            row['mae_years'] = float(np.abs(truth['age'] - pred[target, fset]).mean())
        table.append(row)
    groups_of_sets = {'critic (mean over seeds)': args.critic,
                      'filter-bank covariance + critic (mean over seeds)': [f'{args.filterbank}+{c}' for c in args.critic]}
    for target in TARGETS:
        for label, names in groups_of_sets.items():
            if have(target, names):
                e = dec.estimate(truth[target], get(target, names), score[target], rows[target])
                table.append({'target': target, 'features': label, 'score': e['value'], 'ci_low': e['ci95'][0], 'ci_high': e['ci95'][1]})
    table = pd.DataFrame(table)

    # paired comparisons
    comparisons = [(a, b, True) for a, b in args.holm] + [(a, b, False) for a, b in args.compare]
    tests = []
    for target in TARGETS:
        for a, b, corrected in comparisons:
            names_a, names_b = a.split(','), b.split(',')
            if have(target, names_a) and have(target, names_b):
                d = dec.difference(truth[target], get(target, names_a), get(target, names_b), score[target], rows[target])
                tests.append({'target': target, 'comparison': f'{a} - {b}', 'difference': d['value'],
                              'ci_low': d['ci95'][0], 'ci_high': d['ci95'][1], 'p_two_sided': d['p_two_sided'],
                              'p_holm': np.nan, 'corrected': corrected and target in args.holm_targets})
    tests = pd.DataFrame(tests)

    # Holm's correction over the --holm comparisons (p values below 2 / n_boot cannot be resolved)
    if len(tests):
        family = tests.pop('corrected').to_numpy(bool)
        if family.any():
            tests.loc[family, 'p_holm'] = dec.holm(np.maximum(tests.loc[family, 'p_two_sided'].to_numpy(), 2 / args.n_boot))

    choices = choice_counts(args.results)

    # save
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, 'scores.csv'), index=False)
    tests.to_csv(os.path.join(args.out, 'comparisons.csv'), index=False)
    if len(choices):
        choices.to_csv(os.path.join(args.out, 'choices.csv'), index=False)
    with open(os.path.join(args.out, 'report.md'), 'w') as md:
        md.write(f'Decoding, {len(participants)} participants; {args.repeats} repeats of 5-fold cross-validation over '
                 f'participants; 95% intervals from {args.n_boot} bootstrap samples of participants.\n'
                 'Score: AUROC for age group and sex, R2 for age in years. p_two_sided: bootstrap p value of the paired '
                 "difference; p_holm: corrected with Holm's method over the --holm comparisons.\n\n## Scores\n\n"
                 + table.to_markdown(index=False, floatfmt='.3f') + '\n\n## Comparisons\n\n'
                 + (tests.to_markdown(index=False, floatfmt='.3f') if len(tests) else 'none') + '\n'
                 + ('\n## Sets chosen inside the training folds\n\n' + choices.to_markdown(index=False, floatfmt='.2f') + '\n'
                    if len(choices) else ''))
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    if len(tests):
        print('\n' + tests.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {args.out}/report.md, scores.csv and comparisons.csv')
    return table, tests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--data', required=True, action='append',
                   help='recordings (subject, channel, time) at 128 Hz, not preprocessed (can be given several times)')
    p.add_argument('--demographics', required=True, action='append',
                   help='CSV with participant_id, age_group, sex, age (can be given several times)')
    p.add_argument('--out', required=True)
    p.add_argument('--line-noise', default='lowpass', choices=['lowpass', 'dss'])
    p.add_argument('--extra-readouts', action='store_true',
                   help="also the critic's further readouts (evaluation/decoding.py: EXTRA_READOUTS), as NAME:readout")
    p.add_argument('--channels-from', default='data/LEMON_DATA/EC_ch-8_sf-128.nc5',
                   help="the training data file: its channels, in its order, are taken from --data")
    p.add_argument('--plain', action='append', default=[], metavar='NAME=file.npy',
                   help='further features, one row per segment (the option can be given several times)')
    p.add_argument('--plain-subjects', default=None, metavar='file.npy',
                   help='the participant of every recording behind the rows of the --plain files, to check their order')
    p.add_argument('checkpoints', nargs='*',
                   help='NAME=checkpoint.pt; for a critic with FiLM at its input NAME=checkpoint.pt@BACKGROUND, the '
                        'stand-in for the participant: mean, zero or none (evaluation/decoding.py: critic_readouts)')
    p.set_defaults(run=extract)
    p = sub.add_parser('cv')
    p.add_argument('--features', required=True)
    p.add_argument('--sets', required=True, help="comma-separated feature sets; '+' combines sets; NAME=a|b|c is "
                   "a choice between sets made inside every training fold, stored as NAME")
    p.add_argument('--targets', nargs='+', default=list(TARGETS), choices=TARGETS)
    p.add_argument('--repeats', type=int, default=N_REP)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--out', required=True, help='results file (.jsonl); existing results are kept')
    p.set_defaults(run=cross_validate)
    p = sub.add_parser('report')
    p.add_argument('--features', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--critic', nargs='*', default=[], help='feature sets of the critic, one per training seed')
    p.add_argument('--filterbank', default='filterbank')
    p.add_argument('--compare', nargs=2, action='append', default=[], metavar=('A', 'B'),
                   help='a paired comparison, A - B (the option can be given several times); A and B can be '
                        'comma-separated lists, one set per seed, scored per seed and averaged')
    p.add_argument('--holm', nargs=2, action='append', default=[], metavar=('A', 'B'),
                   help="as --compare; the p values of all --holm comparisons are corrected together (Holm's method)")
    p.add_argument('--holm-targets', nargs='+', default=['sex', 'age'], choices=TARGETS,
                   help='targets of the --holm comparisons')
    p.add_argument('--repeats', type=int, default=N_REP)
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--out', required=True, help='directory for report.md, scores.csv and comparisons.csv')
    p.set_defaults(run=report)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == '__main__':
    main()
