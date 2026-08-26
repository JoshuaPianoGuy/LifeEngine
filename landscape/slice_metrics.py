"""
landscape/slice_metrics.py
======================================================================
Characterise a landscape by MANY SLICES instead of one.

Every Stage-3/5/10 grid is a single 2D slice through a 7814-dimensional weight
space, anchored at one real genome and spanned by one pair of directions. Stage 9
computes FDC, dispersion, neutrality and autocorrelation on ONE such slice and
reports the number. That number has no error bar, and the slice was not chosen at
random — it was anchored on whichever run the pipeline happened to pick.

This script runs the same metrics over SEVERAL slices and reports the
DISTRIBUTION. A property that is real (the landscape is smooth / has global
structure / is largely neutral) survives re-slicing; a property that is an
artefact of one anchor does not. With n slices you can finally say "FDC = -0.86
+/- 0.13 over 2 slices" instead of "FDC = -0.762", and a baseline-vs-hard claim
becomes testable against slice-to-slice spread rather than against sampling noise
alone.

WHY THIS IS CHEAP, AND WHY IT IS NOT THE SAME PROBLEM AS DRAWING TRAJECTORIES
-----------------------------------------------------------------------------
Fitting many run TRAJECTORIES onto one shared plane fails hard: independent runs
are mutually orthogonal in weight space (see stage10_strata._pick_anchor), so a
2D plane holds about 3-4 trajectories before they render as stationary dots.
Landscape CHARACTERISATION has no such limit. Each cell here is just a scalar
f(theta) at a genome that was really probed, so a slice needs only ONE live
anchor genome to be a valid sample of the surface — it does not need to
"contain" any run's path at all. That decouples the two problems entirely: use
few trajectories per figure, and as many slices as you can afford for the
metrics.

The metrics themselves are pure reanalysis of an existing results.csv — NO new
simulation. Adding a slice costs a grid evaluation; re-reading every slice you
already have costs seconds.

UNITS: the autocorrelation length is reported in WEIGHT-SPACE units, not grid
cells. Slices differ in extent and resolution (stage3 hard spans 21.65 over 25
cells, stage10 hard spans 17.57 over 21), so a lag of "3 cells" means different
distances on different slices and the raw cell lag is not comparable. lambda_w =
lambda_cells * cell_spacing.

Usage
-----
  python landscape/slice_metrics.py \
      --slice hard:stage3_rerun6:landscape/out/stage3_hard_coarse \
      --slice hard:stage10_rerun5:landscape/out/stage10_hard_traj \
      --slice baseline:stage3:landscape/out/stage3_baseline_coarse \
      --slice baseline:stage10:landscape/out/stage10_baseline_traj \
      --out-dir landscape/out/slice_metrics

Each --slice is env:label:grid_dir. Repeat it as many times as you have grids.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ll_difficulty as ld
from stage9_difficulty import load_grid


# Metrics reported per slice: key -> (label, format)
#
# WHICH OF THESE SURVIVE THE MIN-MAX NORMALISATION, AND WHICH ONE DOES NOT
# ------------------------------------------------------------------------
# FDC is a correlation coefficient, dispersion is a ratio of distances, the
# quadratic R2 is a variance ratio and the autocorrelation length is computed on
# a linearly detrended series — every one of them is invariant under an affine
# rescaling of f, so normalising the fitness cannot move them and they are
# reported once.
#
# THE NEUTRAL FRACTION IS THE EXCEPTION and it is the reason --norm-csv exists.
# neutrality_summary() sets eps to 2% of the grid's OWN fitness range, so a
# slice spanning 4.9 food units calls a 0.10 step neutral while one spanning
# 22.4 needs 0.45 for the same label. Those two numbers are not the same
# measurement, and comparing them across slices — which is the entire point of
# this script — compares the ranges as much as the surfaces. neutral_frac_shared
# fixes eps at 2% of the SHARED normalisation range instead, so one step size
# means one thing everywhere, and it is the column to quote.
METRIC_SPEC = [
    ('fdc_best',        'FDC (best ref)',        '{:+.3f}'),
    ('fdc_top5',        'FDC (top-5% centroid)', '{:+.3f}'),
    ('lambda_w',        'Autocorr length (weight units)', '{:.2f}'),
    ('neutral_frac',    'Neutral fraction (eps=2% of OWN range)', '{:.3f}'),
    ('neutral_frac_shared', 'Neutral fraction (eps=2% of SHARED range)', '{:.3f}'),
    ('neutral_frac_noise', 'Neutral fraction (vs noise)', '{:.3f}'),
    ('plateau_largest', 'Largest neutral plateau (frac)', '{:.3f}'),
    ('dispersion_05',   'Dispersion (p=0.05)',   '{:.3f}'),
    ('viable_frac',     'Viable fraction',       '{:.3f}'),
    ('r2_quadratic',    'Quadratic R²',          '{:.3f}'),
    ('f_range',         'Fitness range on slice','{:.2f}'),
    ('f_range_norm',    'Fitness range (normalised)', '{:.3f}'),
]

# --slice's ENV field is free text, so it can carry the OUTCOME as well as the
# environment ('hard/failed' vs 'hard/succeeded'). That is usually what you want
# here: within the hard environment the failed and successful runs' slices by
# more than the two environments do, and pooling them into one 'hard' column
# reports a spread that is really a group difference.
ENV_COLOUR = {
    'hard': '#DC2626', 'baseline': '#2563EB',
    'hard/failed': '#DC2626', 'hard/succeeded': '#059669',
    'baseline/weakest': '#7C3AED', 'baseline/succeeded': '#2563EB',
}
FALLBACK_COLOURS = ['#DC2626', '#059669', '#2563EB', '#7C3AED', '#D97706']


def env_colour(env, envs):
    if env in ENV_COLOUR:
        return ENV_COLOUR[env]
    return FALLBACK_COLOURS[envs.index(env) % len(FALLBACK_COLOURS)]
INK = '#374151'
GRID = '#D1D5DB'


def grid_autocorr_length(f, spacing, kmax=None):
    """1/e autocorrelation length of the surface, in WEIGHT-SPACE units.

    The lattice analogue of the Stage-6 random-walk lambda: every row and every
    column is a straight walk across the slice, so their averaged autocorrelation
    is the isotropic decay. Each series is linearly detrended first (a global
    tilt is trend, not correlation, and would otherwise inflate every lag).

    Returns (lambda_in_weight_units, lambda_in_cells, rho_curve).
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    kmax = kmax or max(3, n // 2)
    curves = []
    for series in list(f) + list(f.T):
        s = series[np.isfinite(series)]
        if s.size < 4:
            continue
        resid, _slope = ld.linear_detrend(s)   # returns (residual, slope)
        curves.append(ld.autocorr(resid, min(kmax, s.size - 2)))
    if not curves:
        return float('nan'), float('nan'), np.array([])
    L = min(len(c) for c in curves)
    rho = np.nanmean(np.vstack([c[:L] for c in curves]), axis=0)
    lag = ld.crossing_lag(rho, level=1.0 / np.e)
    return (float(lag * spacing) if np.isfinite(lag) else float('nan'),
            float(lag), rho)


def read_norm_range(path):
    """(lo, hi) out of the normalisation.csv plot_slices.py wrote.

    Reusing that file rather than recomputing the range here is deliberate: the
    figures and the metrics then quote provably the same ruler, and the csv
    already records which slices and environments went into it.
    """
    lo = hi = None
    with open(path) as fh:
        header = fh.readline().strip().split(',')
        for line in fh:
            row = dict(zip(header, line.strip().split(',')))
            if row.get('metric') == 'f_evo':
                lo, hi = float(row['lo']), float(row['hi'])
                break
    if lo is None:
        raise SystemExit(f'no f_evo row in {path}')
    return lo, hi


def slice_metrics(grid_dir, k, norm_range=None):
    """Every metric for ONE slice, from its existing results.csv."""
    G = load_grid(grid_dir)
    f = G['f_evo']
    A, B = G['A'], G['B']
    spacing = float(abs(G['alphas'][1] - G['alphas'][0])) if len(G['alphas']) > 1 else float('nan')

    # fdc() returns (r, n_cells, reference_point); 'top5' is the stable target.
    fd = ld.fdc(A, B, f, reference='best')[0]
    fd5 = ld.fdc(A, B, f, reference='top5', top_frac=0.05)[0]
    lam_w, lam_c, _ = grid_autocorr_length(f, spacing)
    neu = ld.neutrality_summary(f, sem=G['paired_sem'])
    disp = ld.dispersion_profile(A, B, f)
    r2 = ld.meta_model_r2(A, B, f)
    finite = f[np.isfinite(f)]

    # eps on the shared ruler: the same absolute step size on every slice.
    span = (norm_range[1] - norm_range[0]) if norm_range else float('nan')
    if norm_range:
        neu_shared, _n = ld.neutral_fraction(f, 0.02 * span)
    else:
        neu_shared = float('nan')

    return {
        'grid_dir': grid_dir,
        'resolution': G['N'], 'repeats': G['R'],
        'norm_lo': norm_range[0] if norm_range else float('nan'),
        'norm_hi': norm_range[1] if norm_range else float('nan'),
        'neutral_frac_shared': float(neu_shared),
        'f_range_norm': (float(finite.max() - finite.min()) / span
                         if norm_range else float('nan')),
        'extent': float(G['alphas'][-1] - G['alphas'][0]),
        'spacing': spacing,
        'anchor': (G['meta'].get('plane_meta', {}) or {}).get('anchor_method', 'n/a'),
        'fdc_best': float(fd), 'fdc_top5': float(fd5),
        'lambda_w': lam_w, 'lambda_cells': lam_c,
        'neutral_frac': float(neu['neutral_frac']['0.02']),
        # Neutral EDGES cannot tell one huge plateau from many tiny ones, so the
        # largest connected component is reported beside the edge fraction.
        'plateau_largest': float(neu['plateaus']['largest_frac']),
        'neutral_frac_noise': float(neu['neutral_frac_noise']),
        'dispersion_05': float(disp['0.05']),
        'viable_frac': float(ld.viable_fraction(f, k=k)[0]),
        'r2_quadratic': float(r2['r2_quadratic']),
        'f_range': float(finite.max() - finite.min()) if finite.size else float('nan'),
    }


def report(rows):
    """Per-slice table, then the across-slice distribution that is the point."""
    envs = sorted({r['env'] for r in rows})

    print('\n=== per slice ===')
    print(f'  {"env":<9}{"label":<18}{"res":>5}{"R":>4}{"extent":>8}  anchor')
    for r in rows:
        print(f'  {r["env"]:<9}{r["label"]:<18}{r["resolution"]:>5}{r["repeats"]:>4}'
              f'{r["extent"]:>8.2f}  {str(r["anchor"])[:58]}')

    for key, label, fmt in METRIC_SPEC:
        print(f'\n  {label}')
        for env in envs:
            vals = np.array([r[key] for r in rows if r['env'] == env], dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            per = '  '.join(fmt.format(v) for v in vals)
            if vals.size > 1:
                spread = f'{fmt.format(vals.mean())} ± {fmt.format(vals.std(ddof=1)).lstrip("+")}'
                rel = abs(vals.std(ddof=1) / vals.mean()) if vals.mean() else float('nan')
                print(f'    {env:<9} {per:<30} -> {spread}   (spread/|mean| {rel:.0%})')
            else:
                print(f'    {env:<9} {per:<30} -> single slice, NO error bar')


def fig_slices(rows, out_png):
    """One panel per metric, one dot per slice, coloured by environment. The
    figure's job is to show whether the slices AGREE — a metric whose dots pile
    up is a property of the landscape, one whose dots scatter is a property of
    the slice."""
    keys = [k for k, _, _ in METRIC_SPEC]
    envs = sorted({r['env'] for r in rows})
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow))
    axs = np.atleast_1d(axs).ravel()

    for ax, (key, label, fmt) in zip(axs, METRIC_SPEC):
        for i, env in enumerate(envs):
            vals = np.array([r[key] for r in rows if r['env'] == env], dtype=float)
            labs = [r['label'] for r in rows if r['env'] == env]
            good = np.isfinite(vals)
            c = env_colour(env, envs)
            ax.scatter(np.full(good.sum(), i) + np.linspace(-.12, .12, good.sum()),
                       vals[good], s=52, color=c, alpha=.85, zorder=3,
                       edgecolors='white', linewidths=.8)
            if good.sum() > 1:
                m = vals[good].mean()
                ax.plot([i - .28, i + .28], [m, m], color=c, lw=2.2, zorder=4)
            for x, v, t in zip(np.linspace(-.12, .12, good.sum()), vals[good],
                               [l for l, g in zip(labs, good) if g]):
                ax.annotate(t, xy=(i + x, v), fontsize=6.5, color=INK,
                            ha='center', va='bottom', xytext=(0, 5),
                            textcoords='offset points')
        ax.set_xticks(range(len(envs)))
        # Broken at the '/' so 'hard/succeeded' does not run into its neighbour
        # once the ENV field carries the outcome group as well.
        ax.set_xticklabels([e.replace('/', '\n') for e in envs],
                           fontsize=8.5, color=INK)
        ax.set_xlim(-.6, len(envs) - .4)
        ax.set_title(label, fontsize=10, color=INK)
        ax.grid(True, axis='y', alpha=.25, lw=.7)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8)
    for ax in axs[len(METRIC_SPEC):]:
        ax.set_visible(False)

    fig.suptitle('Landscape metrics across independent slices — dots that pile up '
                 'are landscape properties, dots that scatter are slice artefacts',
                 fontsize=12, color=INK, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  wrote {out_png}')


def main():
    ap = argparse.ArgumentParser(
        description='Landscape difficulty metrics over several slices, reported '
                    'as a distribution rather than a single number.')
    ap.add_argument('--slice', action='append', required=True,
                    metavar='ENV:LABEL:GRID_DIR',
                    help='Repeatable. e.g. hard:stage3_rerun6:landscape/out/stage3_hard_coarse')
    ap.add_argument('--out-dir', default='landscape/out/slice_metrics')
    ap.add_argument('--k', type=float, default=1.0,
                    help='Viability threshold multiplier on VIABILITY_BASE. '
                         'Default 1.0 = the theoretical replacement 4.375, which '
                         'is what plot_slices.py draws — ld.K_DEFAULT (2.5) would '
                         'put this column at 10.94 and silently disagree with '
                         'every figure. --k-json overrides it with the Stage-9 '
                         'B1 measured crossing.')
    ap.add_argument('--k-json', default=None,
                    help='calibration.json from the Stage-9 B1 step.')
    ap.add_argument('--out-per-env', default=None, metavar='DIR',
                    help='Also write landscape_metrics.csv into DIR/<env>/ for '
                         'each environment, beside the figures those slices '
                         'produced. The ENV field is split on "/" so an '
                         'env/group label lands in the environment directory.')
    ap.add_argument('--norm-csv', default=None,
                    help='normalisation.csv from plot_slices.py. Fixes the '
                         'neutrality eps at 2%% of that SHARED fitness range so '
                         'the neutral fractions are comparable between slices — '
                         'see METRIC_SPEC. Without it that column is blank.')
    args = ap.parse_args()

    norm_range = read_norm_range(args.norm_csv) if args.norm_csv else None
    if norm_range:
        print(f'shared normalisation range [{norm_range[0]:.4f}, '
              f'{norm_range[1]:.4f}] -> neutrality eps = '
              f'{0.02 * (norm_range[1] - norm_range[0]):.4f} on every slice')

    k = args.k
    if args.k_json:
        cal = json.load(open(args.k_json))
        if cal.get('empirical_replacement_k') is not None:
            k = float(cal['empirical_replacement_k'])
    print(f'viability k = {k:.4f} (threshold f > {ld.viability_threshold(k):.3f})')

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for spec in args.slice:
        parts = spec.split(':', 2)
        if len(parts) != 3:
            raise SystemExit(f'--slice must be ENV:LABEL:GRID_DIR, got {spec!r}')
        env, label, gdir = parts
        print(f'  loading {env}/{label} from {gdir} ...')
        m = slice_metrics(gdir, k, norm_range)
        m.update({'env': env, 'label': label})
        rows.append(m)

    report(rows)

    csv_path = os.path.join(args.out_dir, 'slice_metrics.csv')
    keys = (['env', 'label', 'grid_dir', 'resolution', 'repeats', 'extent',
             'spacing', 'anchor'] + [k for k, _, _ in METRIC_SPEC]
            + ['lambda_cells', 'norm_lo', 'norm_hi'])
    with open(csv_path, 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(f'"{r.get(k, "")}"' if k in ('anchor', 'grid_dir')
                              else str(r.get(k, '')) for k in keys) + '\n')
    print(f'\n  wrote {csv_path}')

    # Per-environment copies, so each figure directory carries the numbers for
    # the slices it drew. The ENV field may be 'hard/failed'; the part before
    # the slash is the environment and the part after is the outcome group, and
    # both are kept as columns so the file stands on its own.
    if args.out_per_env:
        by_env = {}
        for r in rows:
            env = r['env'].split('/')[0]
            by_env.setdefault(env, []).append(r)
        for env, sub in sorted(by_env.items()):
            d = os.path.join(args.out_per_env, env)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, 'landscape_metrics.csv')
            with open(path, 'w') as fh:
                fh.write('environment,group,' + ','.join(keys[1:]) + '\n')
                for r in sub:
                    grp = r['env'].split('/', 1)[1] if '/' in r['env'] else ''
                    fh.write(f'{env},{grp},' + ','.join(
                        f'"{r.get(k, "")}"' if k in ('anchor', 'grid_dir')
                        else str(r.get(k, '')) for k in keys[1:]) + '\n')
            print(f'  wrote {path}   ({len(sub)} slices)')

    fig_slices(rows, os.path.join(args.out_dir, 'slice_metrics.png'))


if __name__ == '__main__':
    main()
