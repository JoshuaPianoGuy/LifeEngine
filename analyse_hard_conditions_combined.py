"""
analyse_hard_conditions_combined.py
======================================================================
All THREE arms of the hard-environment trio on shared axes.

analyse_evolution_hard_runs.py and analyse_learning_hard_runs.py each draw ONE
arm, one figure set per arm, with the run as the only factor. This script draws
the three arms TOGETHER: every panel carries one line per condition, so the
factor being compared is the mechanism.

    run_evolution_condition_hard_w500_h128_array.slurm   GA only      100 runs
    run_learning_condition_hard_w500_h128_array.slurm    GA + in-life RL
    run_pure_rl_condition_hard_w500_h128_array.slurm     RL only, no GA

Arms are found by params.json, never by folder name. Pure RL is logged with
condition == "learning" (its job script passes --condition learning --mode
pure_rl), so MODE is what separates the two RL arms — matching on the condition
string alone would silently pool 200 runs into one curve.

WHAT IS DRAWN
-------------
  hard_conditions_combined.png
      Seven panels sharing one condition legend:
        avg fitness · total agents · converged-fitness strip
        genome variance · in-life weight change (mean absolute weight difference)
        genome weight magnitude (RMS) · Δ magnitude per generation
      The strip is one dot per run with the arm's mean and ±1 std, so the
      distribution behind each mean curve is visible in the same figure.

  hard_conditions_fitness_normalised.png
      AVERAGE fitness alone, rescaled to [0, 1] — see NORMALISATION. Three mean
      lines, no bands, no threshold: the spread is reported beside the plot.

  hard_conditions_summary.csv     converged value per arm per metric
  hard_conditions_normalised.csv  the rescaling constants + normalised endpoints

NORMALISATION — MIN-MAX, ONE RANGE FOR ALL THREE ARMS
------------------------------------------------------
Average fitness is rescaled by the SAME transform for every condition:

    z = (x - lo) / (hi - lo),  lo/hi = min/max over every generation of every
                               run of ALL THREE arms

so 0 is the worst value anything reached and 1 the best, and the arms stay
comparable: the gaps between them are preserved, only the unit changes.
Normalising each arm against its OWN range would map all three to the same
endpoints and delete the comparison the figure exists to make — which is why
that is not offered as an option.

The normalised panel draws the MEAN LINE ONLY — no band, and no replacement
line either. Three overlapping +/-1 std bands cover most of the plot and bury the
separation between the means, which is the one thing the panel exists to show.
The spread is not dropped, it is moved: mean +/- 1 std sits beside the plot per
condition, and hard_conditions_normalised.csv carries std, min, max and IQR in
both raw and normalised units. Since the transform is affine and shared, the
normalised spread is exactly the raw spread divided by (hi - lo), so nothing
about it flatters one arm over another. The raw combined figure keeps its bands
and its 4.375 line.

Usage
-----
  python analyse_hard_conditions_combined.py
  python analyse_hard_conditions_combined.py --band sem --smooth 21
  python analyse_hard_conditions_combined.py --env baseline    # once those exist
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# The single-arm script is the source of truth for discovery, loading and the
# band statistics; importing it keeps this figure set defined on exactly the same
# numbers as the per-arm ones rather than a second implementation of them.
import analyse_learning_hard_runs as la


# ── The three arms ────────────────────────────────────────────────────────────
# Colour triple is the validated one from analyse_learning_hard_runs' fitness
# panel (worst all-pairs dE 14.1 under deuteranopia). Line style is redundant
# with colour ON PURPOSE: three overlapping bands is exactly the case where a
# reader who cannot separate two hues has nothing else to go on.
ARMS = [
    {'key': 'evolution', 'label': 'Evolution (GA only)',
     'logs_root': 'logs/evolution', 'condition': 'evolution', 'mode': 'standard',
     'colour': '#D97706', 'ls': '--'},
    {'key': 'learning', 'label': 'Learning (GA + in-life RL)',
     'logs_root': 'logs/learning', 'condition': 'learning', 'mode': 'standard',
     'colour': '#2563EB', 'ls': '-'},
    {'key': 'pure_rl', 'label': 'Pure RL (no GA)',
     'logs_root': 'logs/learning', 'condition': 'learning', 'mode': 'pure_rl',
     'colour': '#DB2777', 'ls': '-.'},
]

# Panel order for the combined figure. 'strip' is not a curve metric — it is the
# per-run endpoint distribution, spliced into the grid so the spread behind the
# means is in the same figure rather than a separate one.
COMBINED_PANELS = ['avg_fitness', 'total_agents', 'strip', 'genome_variance',
                   'avg_learned_weight_diff', 'avg_network_weight_mag',
                   'inter_gen_weight_change']
FITNESS_METRICS = la.FITNESS_METRICS
# The one metric the normalised figure draws. See fig_normalised() for why the
# other two fitness metrics are deliberately not drawn beside it.
NORM_METRIC = 'avg_fitness'

# Metrics that cannot be negative, so a mean-minus-std band must not be drawn
# below 0. A variance or a population of -0.004 is not a wide error bar, it is a
# nonsense reading, and clipping it stops the panel implying the spread reaches
# somewhere it cannot. inter_gen_weight_change is deliberately NOT here: it is a
# first difference and genuinely goes negative.
NONNEGATIVE = {'avg_fitness', 'top20percent_fitness', 'best_fitness',
               'total_agents', 'peak_population', 'genome_variance',
               'avg_learned_weight_diff', 'avg_network_weight_mag',
               'avg_lifetime'}


def load_arm(arm, env_key, seeds, reps, max_gen, final_window):
    """Discover + load one arm. Returns the arm dict with table/curves attached,
    or None if nothing matched."""
    env = la.ENVIRONMENTS[env_key]
    want = {'condition': arm['condition'], 'mode': arm['mode'],
            'roaming': env['roaming'], 'drain': env['drain']}
    runs = la.discover(arm['logs_root'], want, seeds, reps)
    if not runs:
        print(f"  [warn] no runs for {arm['label']} under {arm['logs_root']}")
        return None
    table, curves = la.load_all(runs, max_gen, final_window)
    out = dict(arm)
    out['table'], out['curves'], out['n'] = table, curves, len(table)
    print(f"  {arm['label']:<28} {len(table):3d} runs, "
          f"{table['seed'].nunique()} seeds")
    return out


def draw_condition_metric(ax, arms, metric, band, smooth, normaliser=None,
                          show_band=True):
    """One panel: the same metric for every arm, optionally rescaled by
    `normaliser` (an (lo, hi) pair applied to every arm alike).

    show_band=False draws the mean line alone. Used by the normalised figure,
    where three overlapping ±1 std bands cover most of the plot area and hide the
    thing the panel is for — the separation between the mean curves. The spread
    is not dropped, it is reported as a number in the panel instead.
    """
    drew = False
    for arm in arms:
        series = la.unit_series(arm['curves'], metric, 'runs')
        if not series:
            continue
        if normaliser is not None:
            lo, hi = normaliser
            span = (hi - lo) or 1.0
            series = [(s - lo) / span for s in series]
        gens, centre, blo, bhi, _n = la.band_stats(series, band, smooth)
        if show_band:
            if metric in NONNEGATIVE:
                floor = 0.0 if normaliser is None else (0.0 - normaliser[0]) / \
                    ((normaliser[1] - normaliser[0]) or 1.0)
                blo = np.maximum(blo, floor)
            ax.fill_between(gens, blo, bhi, color=arm['colour'], alpha=0.16, lw=0)
        ax.plot(gens, centre, color=arm['colour'], lw=1.9, ls=arm['ls'],
                label=arm['label'], solid_capstyle='round')
        drew = True
    if not drew:
        ax.set_visible(False)
        return
    ax.set_xlabel('Generation')
    la.style_axes(ax)


def fig_combined(arms, band, smooth, final_window, out_png, rng, title):
    fig, axs = plt.subplots(2, 4, figsize=(21, 9))
    axs = axs.ravel()

    for ax, panel in zip(axs, COMBINED_PANELS):
        if panel == 'strip':
            groups = [(a['label'].split(' (')[0], a['table']['avg_fitness'].values)
                      for a in arms]
            la.strip(ax, groups, [a['colour'] for a in arms],
                     f'Average fitness (last {final_window} gens)', '',
                     rng, threshold=la.VIABILITY_THRESHOLD)
            ax.set_title(f'Converged fitness per run\none dot per run · '
                         f'crossbar = mean, whisker = ±1 std',
                         fontsize=10, color=la.INK)
            continue
        draw_condition_metric(ax, arms, panel, band, smooth)
        label, _fmt = la.METRICS[panel]
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10, color=la.INK)

    # The in-life-drift panel is the one that needs a word: it is not missing
    # for evolution, it is identically zero, and a flat line at 0 reads as an
    # absent series unless it is named.
    for ax, panel in zip(axs, COMBINED_PANELS):
        if panel == 'avg_learned_weight_diff':
            ax.annotate('evolution is identically 0 —\nno in-lifetime learning',
                        xy=(0.97, 0.06), xycoords='axes fraction', fontsize=7.5,
                        color=la.INK, ha='right', va='bottom')

    for ax in axs[len(COMBINED_PANELS):]:
        ax.set_visible(False)

    handles = [Line2D([], [], color=a['colour'], ls=a['ls'], lw=2.2,
                      label=f"{a['label']} — {a['n']} runs") for a in arms]
    handles.append(Line2D([], [], color=la.THRESHOLD_COLOUR, ls='--', lw=1.1,
                          label=f'replacement f = {la.VIABILITY_THRESHOLD:g}'))
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.985, 0.06),
               frameon=False, fontsize=10, labelcolor=la.INK)

    fig.suptitle(title, fontsize=14, color=la.INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def pooled_range(arms, metric):
    """(lo, hi) over every generation of every run of every arm."""
    lo, hi = np.inf, -np.inf
    for arm in arms:
        for df in arm['curves'].values():
            if metric not in df.columns:
                continue
            v = pd.to_numeric(df[metric], errors='coerce').to_numpy()
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = min(lo, v.min()), max(hi, v.max())
    return (float(lo), float(hi)) if np.isfinite(lo) else (0.0, 1.0)


def fig_normalised(arms, band, smooth, final_window, out_png, title,
                   metric=NORM_METRIC):
    """One panel: `metric` on [0, 1] under a single shared transform, three mean
    lines, and the spread reported beside the plot.

    ONE METRIC, AND WHY THE OTHER TWO ARE NOT HERE. Top-20% fitness carries
    essentially the same shape as the mean once the unit is removed, so a second
    panel of it is a duplicate that invites reading as independent evidence.
    Best fitness is the single luckiest agent of a run — a max over ~700-1300
    draws, which is an extreme-value statistic, not a measure of the policy the
    arm converged on. Its raw range also ran to 124 on one run, which squashed
    every arm into the bottom half of the axis. Both are still in
    hard_conditions_summary.csv in raw units for anyone who wants them.
    """
    lo, hi = pooled_range(arms, metric)
    span = (hi - lo) or 1.0

    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    draw_condition_metric(ax, arms, metric, band, smooth, normaliser=(lo, hi),
                          show_band=False)
    label, _fmt = la.METRICS[metric]
    ax.set_ylabel(f'{label} — normalised')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'rescaled by the pooled raw range [{lo:.2f}, {hi:.2f}] — '
                 f'one transform for all three arms',
                 fontsize=10, color=la.INK)

    endpoints, rows = [], []
    for arm in arms:
        v = pd.to_numeric(arm['table'][metric], errors='coerce').to_numpy()
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        z = (v - lo) / span
        rows.append((arm, z))
        endpoints.append({
            'metric': metric, 'condition': arm['key'], 'n_runs': len(z),
            'norm_lo': lo, 'norm_hi': hi,
            'raw_mean': float(v.mean()), 'raw_std': float(v.std(ddof=1)),
            'norm_mean': float(z.mean()), 'norm_std': float(z.std(ddof=1)),
            'norm_min': float(z.min()), 'norm_max': float(z.max()),
            'norm_iqr': float(np.percentile(z, 75) - np.percentile(z, 25)),
        })

    # Spread lives OUTSIDE the axes, one row per condition, so the plot area
    # holds nothing but the three lines it is meant to compare.
    handles = [Line2D([], [], color=a['colour'], ls=a['ls'], lw=2.4,
                      label=(f"{a['label']}\n"
                             f"{z.mean():.3f} ± {z.std(ddof=1):.3f}   "
                             f"(n = {len(z)})"))
               for a, z in rows]
    leg = fig.legend(handles=handles, loc='center left',
                     bbox_to_anchor=(0.78, 0.5), frameon=False, fontsize=9.5,
                     labelcolor=la.INK, labelspacing=1.5, handlelength=2.6,
                     title=f'converged mean ± 1 std\n(last {final_window} gens)')
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(la.INK)

    fig.suptitle(title, fontsize=13, color=la.INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.77, 0.94))
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')
    return pd.DataFrame(endpoints)


def summary_table(arms, final_window):
    rows = []
    for arm in arms:
        for metric, (label, _fmt) in la.METRICS.items():
            if metric not in arm['table'].columns:
                continue
            v = pd.to_numeric(arm['table'][metric], errors='coerce').to_numpy()
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            rows.append({'condition': arm['key'], 'label': arm['label'],
                         'n_runs': len(v), 'metric': metric, 'metric_label': label,
                         'mean': v.mean(), 'std': v.std(ddof=1),
                         'median': np.median(v),
                         'q25': np.percentile(v, 25), 'q75': np.percentile(v, 75),
                         'min': v.min(), 'max': v.max(),
                         'n_below_replacement': int((v < la.VIABILITY_THRESHOLD).sum())
                                                if metric in FITNESS_METRICS else ''})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description='Evolution, learning and pure RL on shared axes, plus the '
                    'normalised fitness comparison.')
    ap.add_argument('--env', choices=list(la.ENVIRONMENTS), default='hard')
    ap.add_argument('--out', default=None,
                    help='Default: output/hard_conditions_combined (per --env).')
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--reps', type=int, default=None,
                    help='Keep only the first N replicates of each seed.')
    ap.add_argument('--max-gen', type=int, default=None)
    ap.add_argument('--final-window', type=int, default=50)
    ap.add_argument('--band', choices=['std', 'sem', 'iqr'], default='std')
    ap.add_argument('--smooth', type=int, default=11)
    ap.add_argument('--jitter-seed', type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out or f'output/{args.env}_conditions_combined'
    os.makedirs(out_dir, exist_ok=True)
    seeds = set(args.seeds) if args.seeds else None

    print(f'Loading the three {args.env}-environment arms '
          f'({la.ENVIRONMENTS[args.env]["desc"]}) ...')
    arms = [a for a in (load_arm(arm, args.env, seeds, args.reps, args.max_gen,
                                 args.final_window) for arm in ARMS) if a]
    if not arms:
        raise SystemExit('no arms found — check --env and the logs tree')

    band_txt = la.band_label(args.band, 'runs')
    subtitle = (f'{la.ENVIRONMENTS[args.env]["desc"]} · '
                f'{sum(a["n"] for a in arms)} runs across {len(arms)} conditions · '
                f'band = {band_txt}')
    rng = np.random.default_rng(args.jitter_seed)

    fig_combined(arms, args.band, args.smooth, args.final_window,
                 os.path.join(out_dir, f'{args.env}_conditions_combined.png'),
                 rng, f'Evolution vs learning vs pure RL — {subtitle}')

    # Its own subtitle: this figure draws no band, so advertising one would be a
    # caption describing a different plot.
    norm_subtitle = (f'{la.ENVIRONMENTS[args.env]["desc"]} · '
                     f'{sum(a["n"] for a in arms)} runs across {len(arms)} '
                     f'conditions · mean line only, spread reported alongside')
    norm = fig_normalised(
        arms, args.band, args.smooth, args.final_window,
        os.path.join(out_dir, f'{args.env}_conditions_fitness_normalised.png'),
        f'Average fitness, min-max normalised to [0, 1] — {norm_subtitle}')

    summ = summary_table(arms, args.final_window)
    p = os.path.join(out_dir, f'{args.env}_conditions_summary.csv')
    summ.to_csv(p, index=False)
    print(f'  wrote {p}')
    p = os.path.join(out_dir, f'{args.env}_conditions_normalised.csv')
    norm.to_csv(p, index=False)
    print(f'  wrote {p}')

    print(f'\nConverged average fitness (last {args.final_window} generations):')
    for _, r in norm[norm['metric'] == 'avg_fitness'].iterrows():
        print(f"  {r['condition']:<10} raw {r['raw_mean']:6.3f} ± {r['raw_std']:.3f}"
              f"   normalised {r['norm_mean']:.3f} ± {r['norm_std']:.3f}"
              f"   (n={int(r['n_runs'])})")
    print(f'\nDone — {out_dir}')


if __name__ == '__main__':
    main()
