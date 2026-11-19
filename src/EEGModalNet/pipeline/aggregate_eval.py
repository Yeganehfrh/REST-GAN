"""Combine the evaluations of several runs into tables.

    python -m src.EEGModalNet.pipeline.aggregate_eval \\
        --main logs/eval/main_s1_epoch_100 logs/eval/main_s2_epoch_100 logs/eval/main_s3_epoch_100 \\
        --ablation "A6 no minibatch SD=logs/eval/abl_A6_s1_epoch_100"

Each directory is an output of `evaluate.py`. `--main` lists the training seeds of the full model; an ablation
counts as different when its value falls outside the range of the training seeds. Writes one CSV per table and
tables.md to --out.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

METRICS = ['fd', 'precision', 'recall', 'rdm_rho', 'identification']
LOWER_IS_BETTER = {'fd'}


def load(directory):
    with open(os.path.join(directory, 'metrics.json')) as f:
        return json.load(f)


def _value(node):
    return node['mean'] if isinstance(node, dict) and 'mean' in node else node


def mean_sd(values, digits=3):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f'{values[0]:.{digits}f}'
    return f'{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}'


def generation_table(runs):
    rows = []
    for cond in runs[0]['generation']:
        for fam in runs[0]['generation'][cond]['families']:
            row = {'eye state': cond, 'features': fam}
            for m in METRICS + ['similarity', 'mismatched_similarity']:
                row[m] = mean_sd([_value(r['generation'][cond]['families'][fam][m]) for r in runs])
                row[f'{m} (real halves)'] = mean_sd([r['ceiling'][cond]['families'][fam][m] for r in runs[:1]])
            rows.append(row)
    return pd.DataFrame(rows)


def region_fd_table(runs):
    rows = []
    for cond in runs[0]['generation']:
        regions = runs[0]['generation'][cond]['region_fd']
        for region in regions:
            row = {'eye state': cond, 'region': region}
            for feat in regions[region]:
                row[feat] = mean_sd([_value(r['generation'][cond]['region_fd'][region][feat]) for r in runs])
            rows.append(row)
        for region in regions:  # real-vs-real reference
            rows.append({'eye state': cond, 'region': f'{region} (real halves)',
                         **{feat: f"{runs[0]['ceiling'][cond]['region_fd'][region][feat]:.3f}" for feat in regions[region]}})
    return pd.DataFrame(rows)


def k_table(runs):
    rows = []
    for cond in runs[0]['generation']:
        for fam, m in runs[0]['generation'][cond]['families'].items():
            for k in m['pr_by_k']:
                pr = [[_value(v) for v in r['generation'][cond]['families'][fam]['pr_by_k'][k]] for r in runs]
                ceil = runs[0]['ceiling'][cond]['families'][fam]['pr_by_k'][k]
                rows.append({'eye state': cond, 'features': fam, 'k': int(k),
                             'precision': mean_sd([p[0] for p in pr]), 'recall': mean_sd([p[1] for p in pr]),
                             'precision (real halves)': f'{ceil[0]:.3f}', 'recall (real halves)': f'{ceil[1]:.3f}'})
    return pd.DataFrame(rows)


def other_table(runs):
    rows = []
    for cond in runs[0]['generation']:
        g = [r['generation'][cond] for r in runs]
        c = runs[0]['ceiling'][cond]
        rows.append({'eye state': cond,
                     'alpha peak found, real': mean_sd([_value(x['alpha_peak']['valid_real']) for x in g], 2),
                     'alpha peak found, generated': mean_sd([_value(x['alpha_peak']['valid_generated']) for x in g], 2),
                     'alpha peak Wasserstein (Hz)': mean_sd([_value(x['alpha_peak']['wasserstein_hz']) for x in g], 2),
                     'alpha peak paired r': mean_sd([_value(x['alpha_peak']['paired_r']) for x in g], 2),
                     'alpha peak paired r (real halves)': f"{c['alpha_peak']['paired_r']:.2f}",
                     'join ratio, single pass': mean_sd([_value(x['joins']['single_pass']) for x in g], 2)
                     if all('single_pass' in x['joins'] for x in g) else '',
                     'join ratio, joined segments': mean_sd([_value(x['joins']['generated']) for x in g], 2),
                     'join ratio, real': mean_sd([_value(x['joins']['real']) for x in g], 2),
                     'kurtosis, generated': mean_sd([_value(x['amplitude']['kurtosis_generated']) for x in g], 2),
                     'kurtosis, real': mean_sd([_value(x['amplitude']['kurtosis_real']) for x in g], 2)})
    return pd.DataFrame(rows)


def heldout_table(runs):
    rows = []
    for cond in runs[0].get('heldout', {}):
        for fam in runs[0]['heldout'][cond]:
            row = {'eye state': cond, 'features': fam}
            for key in runs[0]['heldout'][cond][fam]:
                row[key] = mean_sd([r['heldout'][cond][fam][key] for r in runs])
            rows.append(row)
    return pd.DataFrame(rows)


def enrolment_table(runs):
    rows = []
    for cond in runs[0].get('enrolment', {}):
        for fam in runs[0]['enrolment'][cond]['families']:
            row = {'eye state': cond, 'features': fam}
            for source in ('enrolled', 'mean_embedding', 'real_first_half'):
                for m in ('identification', 'similarity'):
                    row[f'{m}, {source}'] = mean_sd([r['enrolment'][cond]['families'][fam][source][m] for r in runs], 2)
            row['chance identification'] = f"{runs[0]['enrolment'][cond]['chance_identification']:.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def summary(run, section='generation'):
    """One value per measure: the mean over eye states and feature spaces (the four feature spaces of the paper)."""
    conds = run[section].values()
    if all('summary' in cond for cond in conds):
        return {m: float(np.mean([_value(cond['summary'][m]) for cond in conds])) for m in METRICS}
    return {m: float(np.mean([_value(fam[m]) for cond in conds for fam in cond['families'].values()])) for m in METRICS}


def _interval(run, section='generation'):
    """95% bootstrap interval over participants of each summary metric (mean over eye states of the bounds)."""
    conds = run[section].values()
    if not all('summary_ci95' in cond for cond in conds):
        return None
    return {m: [float(np.mean([_value(cond['summary_ci95'][m][i]) for cond in conds])) for i in (0, 1)] for m in METRICS}


def _beyond(run, section='generation'):
    """Median over participants of the three non-Gaussian properties (mean over eye states)."""
    conds = [cond['beyond_second_order'] for cond in run[section].values() if 'beyond_second_order' in cond]
    if not conds:
        return {}
    return {name: {k: float(np.mean([_value(c[name][k]) for c in conds])) for k in ('generated', 'real', 'spearman')}
            for name in conds[0]}


def _alpha_r(run, section='generation'):
    return float(np.mean([_value(cond['alpha_peak']['paired_r']) for cond in run[section].values()]))


def headline_table(main_runs):
    """The full model and the two halves of the real recordings."""
    def row(label, summaries, intervals, alpha, beyond):
        out = {'source': label}
        for m in METRICS:
            out[m] = mean_sd([x[m] for x in summaries])
            if len(summaries) == 1 and intervals[0]:
                out[m] += f' [{intervals[0][m][0]:.3f}, {intervals[0][m][1]:.3f}]'
        out['alpha peak r'] = mean_sd(alpha, 2)
        for name in ('kurtosis', 'alpha_env_cv', 'alpha_sd_segments'):
            out[name] = mean_sd([b[name]['generated'] for b in beyond], 2) if beyond and all(name in b for b in beyond) else ''
        return out

    rows = [row(f'full model ({len(main_runs)} training seeds, mean ± SD)', [summary(r) for r in main_runs],
                [_interval(r) for r in main_runs], [_alpha_r(r) for r in main_runs], [_beyond(r) for r in main_runs])]
    first = main_runs[0]
    real = row('real: two halves of each recording', [summary(first, 'ceiling')], [_interval(first, 'ceiling')],
               [_alpha_r(first, 'ceiling')], [])
    for name, values in _beyond(first).items():   # the values of the real recordings themselves
        real[name] = f"{values['real']:.2f}"
    rows.append(real)
    return pd.DataFrame(rows)


def family_identification_table(main_runs):
    """Identification per feature space (mean over eye states)."""
    def per_family(run, section='generation'):
        fams = next(iter(run[section].values()))['families']
        return {f: float(np.mean([_value(cond['families'][f]['identification']) for cond in run[section].values()])) for f in fams}

    seeds = pd.DataFrame([per_family(r) for r in main_runs])
    rows = [{'source': f'full model ({len(main_runs)} training seeds)', **{f: mean_sd(seeds[f]) for f in seeds}}]
    rows.append({'source': 'real: two halves of each recording',
                 **{f: f'{v:.3f}' for f, v in per_family(main_runs[0], 'ceiling').items()}})
    return pd.DataFrame(rows)


def enrolment_summary_table(main_runs):
    """Held-out participants: identification of the second half of the recording."""
    def values(run):
        conds = [c['summary'] for c in run.get('enrolment', {}).values() if 'summary' in c]
        return {src: float(np.mean([c[src]['identification'] for c in conds])) for src in conds[0]} if conds else None

    main = [v for v in map(values, main_runs) if v]
    if not main:
        return pd.DataFrame()
    rows = [{'source': f'full model ({len(main)} training seeds)', 'enrolled': mean_sd([v['enrolled'] for v in main]),
             'without the participant (mean embedding)': mean_sd([v['mean_embedding'] for v in main])}]
    rows.append({'source': 'real: first half of the recording', 'enrolled': f"{main[0]['real_first_half']:.3f}",
                 'without the participant (mean embedding)': ''})
    return pd.DataFrame(rows)


def ablation_table(main_runs, ablations):
    seeds = pd.DataFrame([summary(r) for r in main_runs])
    rows = [{'model': f'full model ({len(main_runs)} training seeds)', **{m: mean_sd(seeds[m]) for m in METRICS},
             'outside seed range': ''}]
    rows.append({'model': 'range of the seeds', **{m: f'{seeds[m].min():.3f} to {seeds[m].max():.3f}' for m in METRICS},
                 'outside seed range': ''})
    for label, run in ablations.items():
        s = summary(run)
        flags = []
        for m in METRICS:
            if s[m] < seeds[m].min() or s[m] > seeds[m].max():
                worse = (s[m] > seeds[m].max()) == (m in LOWER_IS_BETTER)
                flags.append(f"{m} ({'worse' if worse else 'better'})")
        rows.append({'model': label, **{m: f'{s[m]:.3f}' for m in METRICS}, 'outside seed range': ', '.join(flags)})
    return pd.DataFrame(rows).fillna('')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', nargs='+', required=True, help='evaluation directories of the full model (training seeds)')
    parser.add_argument('--ablation', nargs='*', default=[], help='label=directory for each ablation')
    parser.add_argument('--out', default='logs/eval/tables')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    main_runs = [load(d) for d in args.main]
    ablations = {item.split('=', 1)[0]: load(item.split('=', 1)[1]) for item in args.ablation}
    tables = {'headline': ('The model and real data (four feature spaces, both eye states; single evaluations with the '
                           '95% interval over participants)', headline_table(main_runs)),
              'identification_by_family': ('Identification per feature space', family_identification_table(main_runs)),
              'enrolment_summary': ('Held-out participants: identification of the second half of the recording',
                                    enrolment_summary_table(main_runs)),
              'generation': ('Generated vs real recordings, training participants', generation_table(main_runs)),
              'region_fd': ('Fréchet distance by region and feature (Supplementary Table S1)', region_fd_table(main_runs)),
              'k_sensitivity': ('Precision and recall for different k', k_table(main_runs)),
              'alpha_joins_amplitude': ('Alpha peak, segment joins and kurtosis', other_table(main_runs)),
              'heldout': ('Generated population vs held-out participants', heldout_table(main_runs)),
              'enrolment': ('Embeddings fitted to the held-out participants', enrolment_table(main_runs)),
              'ablations': ('Ablations (mean over eye states and feature spaces)', ablation_table(main_runs, ablations))}
    with open(os.path.join(args.out, 'tables.md'), 'w') as md:
        md.write(f'Values are mean ± SD over {len(main_runs)} training seed(s); each seed is the mean of its generation seeds.\n')
        for name, (title, table) in tables.items():
            if table.empty:
                continue
            table.to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
            md.write(f'\n## {title}\n\n{table.to_markdown(index=False)}\n')
    print(f'wrote {args.out}/tables.md and one CSV per table')
    return tables


if __name__ == '__main__':
    main()
