"""Draw figures from the outputs of `evaluate.py`.

    python -m src.EEGModalNet.pipeline.make_figures logs/eval/main_s1_epoch_100 --out figures

Real data are black, generated data orange, data generated from the mean embedding green, references grey.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

from ...EEGModalNet.evaluation.features import BANDS  # noqa: E402
from .aggregate_eval import METRICS, load, summary  # noqa: E402

# one palette for all figures
REAL, GEN, MEAN_EMB = '#0b0b0b', '#eb6834', '#1baf7a'
REFERENCE, INK, INK_2, MUTED, GRID, AXIS = '#898781', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SEQUENTIAL = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('blue_red', ['#1c5cab', '#f0efec', '#d03b3b'])
CONDITIONS = {'EC': 'Eyes closed', 'EO': 'Eyes open'}
FAMILIES = {'band_power': 'Band power', 'coherence': 'Coherence', 'hjorth_shape': 'Hjorth', 'log_corr': 'Log-correlation'}
METRIC_LABELS = {'fd': 'Fréchet distance', 'precision': 'Precision', 'recall': 'Recall',
                 'rdm_rho': 'RDM correlation', 'identification': 'Identification'}
CHANNEL_ORDER = ['F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'O1', 'O2']
BAND_SYMBOLS = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.75,
    'grid.color': GRID, 'grid.linewidth': 0.75, 'grid.linestyle': '-', 'axes.axisbelow': True,
    'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.dpi': 300,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, f'{name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(out, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    return name


def legend(fig, entries, **kwargs):
    handles = [plt.Line2D([], [], color=c, lw=lw, marker=m, ms=5, ls=ls) for _, c, lw, m, ls in entries]
    fig.legend(handles, [e[0] for e in entries], loc='upper center', ncol=len(entries), handlelength=1.6,
               columnspacing=1.6, **kwargs)


def _value(node):
    return node['mean'] if isinstance(node, dict) and 'mean' in node else node


def fig_examples(arrays, channels, out):
    if 'EC/example_real' not in arrays:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=True)
    t = np.arange(384) / 128.0
    order = [channels.index(c) for c in CHANNEL_ORDER]
    for row, cond in enumerate(CONDITIONS):
        for col, (key, colour, label) in enumerate((('example_real', REAL, 'Real'), ('example_gen', GEN, 'Generated'))):
            seg = arrays[f'{cond}/{key}'][0][:384]
            ax = axes[row, col]
            for i, ch in enumerate(order):
                ax.plot(t, seg[:, ch] - 3.0 * i, color=colour, lw=0.75)
            ax.set_yticks([-3.0 * i for i in range(len(order))], CHANNEL_ORDER)
            ax.set_title(f'{label}, {CONDITIONS[cond].lower()}', loc='left')
            ax.tick_params(length=0)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle('One 3-s example segment of one participant (standardised units)', x=0.02, ha='left', fontsize=9)
    fig.tight_layout()
    return save(fig, out, 'fig_examples')


def fig_psd(arrays, channels, out):
    names = []
    for cond, title in CONDITIONS.items():
        freqs = arrays[f'{cond}/freqs']
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
        for ax, ch in zip(axes.ravel(), CHANNEL_ORDER):
            i = channels.index(ch)
            for key, colour in (('psd_real', REAL), ('psd_gen', GEN)):
                db = 10 * np.log10(arrays[f'{cond}/{key}'][:, :, i])
                mean, sd = db.mean(0), db.std(0)
                ax.fill_between(freqs, mean - sd, mean + sd, color=colour, alpha=0.10, lw=0)
                ax.plot(freqs, mean, color=colour)
            ax.set_title(ch, loc='left')
            ax.grid(True, axis='y')
            ax.set_xlim(1, 45)
        for ax in axes[-1]:
            ax.set_xlabel('Frequency (Hz)')
        for ax in axes[:, 0]:
            ax.set_ylabel('Power (dB)')
        fig.suptitle(f'{title}: power spectra, mean ± 1 SD over participants', x=0.02, ha='left', fontsize=9, y=1.0)
        legend(fig, [('Real', REAL, 1.5, None, '-'), ('Generated', GEN, 1.5, None, '-')], bbox_to_anchor=(0.78, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        names.append(save(fig, out, f'fig_psd_{cond}'))
    return names


def fig_psd_difference(arrays, channels, out):
    order = [channels.index(c) for c in CHANNEL_ORDER]
    diff = {c: (10 * np.log10(arrays[f'{c}/psd_gen']).mean(0) - 10 * np.log10(arrays[f'{c}/psd_real']).mean(0))[:, order].T
            for c in CONDITIONS}
    limit = max(np.abs(d).max() for d in diff.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (cond, title) in zip(axes, CONDITIONS.items()):
        freqs = arrays[f'{cond}/freqs']
        image = ax.imshow(diff[cond], aspect='auto', cmap=DIVERGING, vmin=-limit, vmax=limit,
                          extent=(freqs[0], freqs[-1], len(order) - 0.5, -0.5), interpolation='nearest')
        for name, (lo, hi) in BANDS.items():
            if lo > 1:
                ax.axvline(lo, color='white', lw=0.75)
            ax.text((lo + hi) / 2, -0.75, BAND_SYMBOLS[name], ha='center', va='bottom', fontsize=8, color=INK_2)
        ax.set_yticks(range(len(order)), CHANNEL_ORDER)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(title, loc='left', pad=14)
        ax.tick_params(length=0)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label('Generated − real (dB)')
    bar.outline.set_visible(False)
    return save(fig, out, 'fig_psd_difference')


def fig_coherence(arrays, channels, out):
    order = [channels.index(c) for c in CHANNEL_ORDER]
    n = len(channels)
    iu = np.triu_indices(n, k=1)

    def matrix(vector):
        m = np.zeros((n, n)); m[iu] = vector; m = m + m.T
        return m[np.ix_(order, order)]

    fig, axes = plt.subplots(2, len(BANDS), figsize=(7.2, 3.3), sharex=True, sharey=True)
    for row, (cond, title) in enumerate(CONDITIONS.items()):
        real = arrays[f'{cond}/coherence_real'].mean(0).reshape(len(BANDS), -1)
        gen = arrays[f'{cond}/coherence_gen'].mean(0).reshape(len(BANDS), -1)
        for col, band in enumerate(BANDS):
            r, g = matrix(real[col]), matrix(gen[col])
            both = np.tril(r, -1) + np.triu(g, 1)
            both[np.eye(n, dtype=bool)] = np.nan
            ax = axes[row, col]
            image = ax.imshow(both, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks(range(n), CHANNEL_ORDER, rotation=90)
            ax.set_yticks(range(n), CHANNEL_ORDER)
            ax.tick_params(length=0, labelsize=6)
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(band.capitalize())
        axes[row, 0].set_ylabel(title)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label('Coherence')
    bar.outline.set_visible(False)
    fig.suptitle('Group-mean coherence: lower triangle real, upper triangle generated', x=0.02, ha='left', fontsize=9)
    return save(fig, out, 'fig_coherence')


def fig_population(metrics, out):
    fig, axes = plt.subplots(2, len(METRICS), figsize=(7.2, 3.3), sharey=True, sharex='col')
    y = np.arange(len(FAMILIES))
    for row, (cond, title) in enumerate(CONDITIONS.items()):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            gen = [metrics['generation'][cond]['families'][f][metric] for f in FAMILIES]
            ceil = [metrics['ceiling'][cond]['families'][f][metric] for f in FAMILIES]
            ax.errorbar([g['mean'] for g in gen], y, xerr=[g['sd'] for g in gen], fmt='o', color=GEN, ms=5,
                        mec='white', mew=1, elinewidth=1, zorder=3)
            ax.plot(ceil, y, '|', color=REFERENCE, ms=10, mew=1.5, zorder=2)
            ax.grid(True, axis='x')
            ax.set_yticks(y, list(FAMILIES.values()))
            ax.tick_params(length=0)
            if metric != 'fd':
                ax.set_xlim(-0.05, 1.05)
                ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], loc='left')
        axes[row, 0].set_ylabel(title)
    axes[0, 0].set_ylim(len(FAMILIES) - 0.5, -0.5)
    legend(fig, [('Generated vs real (± SD over generation seeds)', GEN, 0, 'o', ''),
                 ('Real first half vs real second half', REFERENCE, 0, '|', '')], bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5)
    return save(fig, out, 'fig_population')


def fig_distances(arrays, metrics, out):
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(7.2, 3.4))
    for row, (cond, title) in enumerate(CONDITIONS.items()):
        for col, (fam, label) in enumerate(FAMILIES.items()):
            real, gen = arrays[f'{cond}/{fam}_real'], arrays[f'{cond}/{fam}_gen']
            mean, std = real.mean(0), real.std(0)
            d_real, d_gen = pdist((real - mean) / std), pdist((gen - mean) / std)
            bins = np.linspace(0, np.percentile(np.r_[d_real, d_gen], 99.5), 50)
            ax = axes[row, col]
            for d, colour in ((d_real, REAL), (d_gen, GEN)):
                ax.hist(d, bins=bins, density=True, histtype='step', color=colour, lw=1.5)
            rho = metrics['generation'][cond]['families'][fam]['rdm_rho']['mean']
            ax.set_title(f'{label}: ρ = {rho:.2f}' if row == 0 else f'ρ = {rho:.2f}', loc='left')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        axes[row, 0].set_ylabel(f'{title}\ndensity')
    fig.supxlabel('Distance between participants (standardised features)', fontsize=8, color=INK_2)
    legend(fig, [('Real', REAL, 1.5, None, '-'), ('Generated', GEN, 1.5, None, '-')], bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return save(fig, out, 'fig_distances')


def fig_alpha_peak(arrays, metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    bins = np.arange(8, 13.01, 0.25)
    for ax, (cond, title) in zip(axes, CONDITIONS.items()):
        for key, colour in (('iaf_real', REAL), ('iaf_gen', GEN)):
            values = arrays[f'{cond}/{key}']
            ax.hist(values[np.isfinite(values)], bins=bins, histtype='step', color=colour, lw=1.5)
        a = metrics['generation'][cond]['alpha_peak']
        ax.set_title(title, loc='left')
        ax.text(0.98, 0.95, f"peak found: {_value(a['valid_real']):.0%} of real,\n{_value(a['valid_generated']):.0%} of generated",
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK_2)
        ax.set_xlabel('Alpha peak frequency (Hz)')
        ax.grid(True, axis='y')
    axes[0].set_ylabel('Participants')
    legend(fig, [('Real', REAL, 1.5, None, '-'), ('Generated', GEN, 1.5, None, '-')], bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return save(fig, out, 'fig_alpha_peak')


def fig_k_sensitivity(metrics, out):
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(7.2, 3.3), sharex=True, sharey=True)
    cond = 'EC'
    for col, (fam, label) in enumerate(FAMILIES.items()):
        gen, ceil = metrics['generation'][cond]['families'][fam]['pr_by_k'], metrics['ceiling'][cond]['families'][fam]['pr_by_k']
        ks = sorted(gen, key=int)
        for row, name in enumerate(('Precision', 'Recall')):
            ax = axes[row, col]
            ax.plot(range(len(ks)), [ceil[k][row] for k in ks], color=REFERENCE, marker='o', ms=4, mec='white', mew=1)
            ax.plot(range(len(ks)), [gen[k][row]['mean'] for k in ks], color=GEN, marker='o', ms=4, mec='white', mew=1)
            ax.set_xticks(range(len(ks)), ks)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, axis='y')
            if row == 0:
                ax.set_title(label, loc='left')
            if col == 0:
                ax.set_ylabel(name)
    for ax in axes[-1]:
        ax.set_xlabel('k (nearest neighbours)')
    legend(fig, [('Generated vs real', GEN, 1.5, 'o', '-'), ('Real halves', REFERENCE, 1.5, 'o', '-')], bbox_to_anchor=(0.7, 1.05))
    fig.suptitle('Eyes closed: precision and recall for different neighbourhood sizes', x=0.02, ha='left', fontsize=9)
    fig.tight_layout()
    return save(fig, out, 'fig_k_sensitivity')


def fig_enrolment(metrics, out):
    if not metrics.get('enrolment'):
        return None
    sources = (('enrolled', 'Embedding fitted to the participant', GEN, 'o'),
               ('mean_embedding', 'Mean embedding', MEAN_EMB, 'o'),
               ('real_first_half', 'Real first half (reference)', REFERENCE, '|'))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    y = np.arange(len(FAMILIES))
    for ax, (cond, title) in zip(axes, CONDITIONS.items()):
        rows = metrics['enrolment'][cond]['families']
        for key, _, colour, marker in sources:
            ax.plot([rows[f][key]['identification'] for f in FAMILIES], y, marker, color=colour,
                    ms=10 if marker == '|' else 6, mew=1.5 if marker == '|' else 1, mec=colour if marker == '|' else 'white')
        chance = metrics['enrolment'][cond]['chance_identification']
        ax.axvline(chance, color=AXIS, lw=0.75)
        ax.text(chance + 0.015, -0.62, 'chance', color=MUTED, fontsize=7, va='center')
        ax.set_yticks(y, list(FAMILIES.values()))
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(title, loc='left')
        ax.grid(True, axis='x')
        ax.tick_params(length=0)
    axes[0].set_ylim(len(FAMILIES) - 0.5, -0.85)
    fig.supxlabel('Share of participants identified from the other half of their recording', fontsize=8, color=INK_2)
    legend(fig, [(label, colour, 0, marker, '') for _, label, colour, marker in sources], bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    return save(fig, out, 'fig_enrolment')


def fig_ablations(main_dirs, ablations, out):
    seeds = [summary(load(d)) for d in main_dirs]
    variants = {label: summary(load(d)) for label, d in ablations.items()}
    labels = ['Full model'] + list(variants)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.2, 0.5 + 0.28 * len(labels) + 0.6), sharey=True)
    y = np.arange(len(labels))
    for ax, metric in zip(axes, METRICS):
        values = [s[metric] for s in seeds]
        ax.axvspan(min(values), max(values), color=REFERENCE, alpha=0.18, lw=0)
        ax.plot([np.mean(values)] + [v[metric] for v in variants.values()], y, 'o', color=GEN, ms=6, mec='white', mew=1)
        ax.set_title(METRIC_LABELS[metric], loc='left')
        ax.grid(True, axis='x')
        ax.tick_params(length=0)
        ax.set_yticks(y, labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    fig.suptitle(f'Grey band: range of the {len(seeds)} training seeds of the full model', x=0.02, ha='left', fontsize=8, color=INK_2)
    fig.tight_layout()
    return save(fig, out, 'fig_ablations')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('evaluation', help='output directory of evaluate.py for the run shown in the figures')
    parser.add_argument('--main', nargs='*', default=[], help='evaluation directories of all training seeds (ablation figure)')
    parser.add_argument('--ablation', nargs='*', default=[], help='label=directory for each ablation')
    parser.add_argument('--out', default='figures')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.evaluation, 'metrics.json')) as f:
        metrics = json.load(f)
    arrays = np.load(os.path.join(args.evaluation, 'arrays.npz'), allow_pickle=True)
    channels = list(metrics['channels'])

    made = [fig_examples(arrays, channels, args.out), *fig_psd(arrays, channels, args.out),
            fig_psd_difference(arrays, channels, args.out), fig_coherence(arrays, channels, args.out),
            fig_population(metrics, args.out), fig_distances(arrays, metrics, args.out),
            fig_alpha_peak(arrays, metrics, args.out), fig_k_sensitivity(metrics, args.out),
            fig_enrolment(metrics, args.out)]
    if args.main and args.ablation:
        made.append(fig_ablations(args.main, dict(item.split('=', 1) for item in args.ablation), args.out))
    made = [m for m in made if m]
    print(f'wrote {len(made)} figures (PNG and PDF) to {args.out}: ' + ', '.join(made))
    return made


if __name__ == '__main__':
    main()
