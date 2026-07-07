"""
analyse_predator_metrics.py
======================================================================
Population / fitness / lifetime view of the predator sweeps — the four
generations.csv signals you actually asked for, as count x drain HEATMAPS:

    total_agents      -- total agents that lived during the generation
    peak_population    -- peak simultaneous population (carrying capacity)
    avg_fitness        -- mean fitness
    avg_lifetime       -- mean lifespan (ticks)

Companion to analyse_predator_sweep.py, which layers in the organisms.csv
predation-death fraction. This script reads ONLY generations.csv, so it is fast
(no 100 MB+ organisms.csv scan) and is the right tool when you just want the
population/fitness/lifetime outcomes per predator parameter.

Like analyse_predator_sweep.py it handles BOTH sweeps via --predator-type:
    roaming -> count axis = roaming_predator_count  (patrol disabled that run)
    patrol  -> count axis = predators_per_patch     (roaming disabled that run)
and selects the experimental condition via --condition (default evolution, which
is where the predator sweeps live: logs/evolution/standard/auto-run).

Variable-length runs (HPC walltime cut some runs short)
-------------------------------------------------------
Each run is collapsed to the MEAN OF ITS LAST --final-window generations
(.tail(window)), which is relative to that run's own final generation — so a run
that stopped at gen 484 contributes gens 435..484 and a full run contributes
950..999. Nothing here assumes a fixed run length, so truncated runs are handled
without crashing. The one caveat is scientific, not mechanical: an endpoint from
a short run is not as converged as one from a full run. To compare fairly, pass
--max-gen N to clip EVERY run to the same horizon before taking the tail. The
per-cell gen-length range is printed and written to the summary CSV (gen_min /
gen_max) so you can see at a glance which cells were truncated.

Outputs (into --out, suffixed by predator type so both can share a folder):
    predator_metrics_<type>_heatmaps.png   -- 2x2 panel: one number per combo
    predator_metrics_<type>_curves.png     -- fitness + population OVER generations
                                              (row per count, col per metric, line
                                              per drain) so you can see crashes /
                                              recoveries the heatmaps can't show
    predator_metrics_<type>_summary.csv    -- per (count, drain) aggregated table

Usage
-----
    python analyse_predator_metrics.py --predator-type roaming --condition evolution
    python analyse_predator_metrics.py --predator-type patrol  --condition evolution
    # fair comparison across truncated runs — clip everyone to gen 480:
    python analyse_predator_metrics.py --predator-type roaming --max-gen 480
"""

import argparse
import glob
import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# (condition, mode) -> experimental-condition label (matches analyse_predator_sweep.py).
COND_LABELS = {
    ('evolution', 'standard'): 'evolution',
    ('learning',  'standard'): 'learning',
    ('learning',  'pure_rl'):  'pure_rl',
}

# Per predator-type: which params.json field is the count axis, which field must
# be zero to belong to THIS sweep, and the heatmap y-axis label.
PRED_AXIS = {
    'roaming': {'field': 'roaming_predator_count', 'other': 'predators_per_patch',
                'ylabel': 'Roaming predators'},
    'patrol':  {'field': 'predators_per_patch', 'other': 'roaming_predator_count',
                'ylabel': 'Predators per patch'},
}

# The four generations.csv signals -> (panel title, per-cell number format).
# All four are "more = the population is coping better", so all use the same
# green colormap where low = harder.
METRICS = {
    'total_agents':    ('Total agents (final gens)',    '{:.0f}'),
    'peak_population': ('Peak population (final gens)',  '{:.0f}'),
    'avg_fitness':     ('Mean fitness (final gens)',     '{:.2f}'),
    'avg_lifetime':    ('Mean lifespan (final gens)',    '{:.0f}'),
}

# Signals plotted OVER GENERATIONS (fitness + the population metrics) so you can
# see whether a parameter combo crashes, recovers, or drifts — the trajectory the
# single-number heatmaps can't show. column -> axis label.
CURVE_METRICS = {
    'avg_fitness':         'Mean fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'peak_population':     'Peak population',
    'total_agents':        'Total agents',
}

# Distinct colour per drain level (assigned in sorted-drain order).
DRAIN_COLOURS = ['#2563EB', '#059669', '#D97706', '#DC2626', '#7C3AED', '#0891B2']


def cond_label(params):
    return COND_LABELS.get((params.get('condition'), params.get('mode')),
                           f"{params.get('condition')}/{params.get('mode')}")


def discover_runs(logs_root, want_condition, predator_type):
    """Find run folders (recursively) matching want_condition AND predator_type.
    The count axis is the predator-type's field; the OTHER predator field must be
    zero so the roaming/patrol sweeps don't bleed into each other in a shared
    logs tree. Returns list of {count, drain, seed, dir, gen_csv}."""
    cfg = PRED_AXIS[predator_type]
    runs = []
    for params_path in glob.glob(os.path.join(logs_root, '**', 'params.json'),
                                 recursive=True):
        run_dir = os.path.dirname(params_path)
        gen_csv = os.path.join(run_dir, 'generations.csv')
        if not os.path.exists(gen_csv):
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
            if cond_label(p) != want_condition:
                continue
            count = int(p[cfg['field']])
            other = float(p[cfg['other']])
            drain = float(p['predator_drain'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        # Belongs to THIS sweep only if its own axis is active and the other
        # predator type is disabled.
        if count <= 0 or other != 0:
            continue
        runs.append({'count': count, 'drain': drain, 'seed': seed,
                     'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def gen_metrics(gen_csv, window, max_gen):
    """Mean of each metric over the last `window` recorded generations (<= max_gen).

    The tail is relative to the run's own final generation, so this is robust to
    walltime-truncated runs. Deduplicates on 'generation' (keep last) so a
    doubled/appended CSV doesn't double-count the tail."""
    df = pd.read_csv(gen_csv)
    df = df[df['generation'] <= max_gen].copy()
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    out = {'gen_min': int(df['generation'].min()) if len(df) else 0,
           'gen_max': int(df['generation'].max()) if len(df) else 0}
    for col in METRICS:
        if col in df.columns:
            tail = pd.to_numeric(df[col], errors='coerce').dropna().tail(window)
            out[col] = tail.mean() if len(tail) else np.nan
        else:
            out[col] = np.nan
    return out, df


def build_table(runs, window, max_gen):
    """Per-run final metrics -> table, plus each run's per-generation curve frame
    (generation-indexed CURVE_METRICS columns) keyed by (count, drain, seed)."""
    rows, curves = [], {}
    for r in runs:
        m, df = gen_metrics(r['gen_csv'], window, max_gen)
        row = {'count': r['count'], 'drain': r['drain'], 'seed': r['seed'],
               'gen_min': m['gen_min'], 'gen_max': m['gen_max']}
        row.update({col: m[col] for col in METRICS})
        rows.append(row)
        cols = [c for c in CURVE_METRICS if c in df.columns]
        curves[(r['count'], r['drain'], r['seed'])] = (
            df.set_index('generation')[cols].apply(pd.to_numeric, errors='coerce'))
        print(f"  count={r['count']:<3d} drain={r['drain']:<5g} seed={r['seed']:<5d} "
              f"tot={m['total_agents']:.0f} pop={m['peak_population']:.0f} "
              f"fit={m['avg_fitness']:.3f} life={m['avg_lifetime']:.0f} "
              f"(gen {m['gen_max']})")
    return pd.DataFrame(rows), curves


def pivot(table, value):
    """count (rows) x drain (cols) matrix of the seed-mean of `value`."""
    agg = table.groupby(['count', 'drain'])[value].mean().reset_index()
    mat = agg.pivot(index='count', columns='drain', values=value)
    mat = mat.sort_index(ascending=False)           # high count at top
    mat = mat.reindex(sorted(mat.columns), axis=1)   # low drain at left
    return mat


def heatmap(ax, mat, title, ylabel, fmt):
    """Render one count x drain heatmap with per-cell annotations. All four
    metrics share the YlGn 'more = coping better' scale (low = harder)."""
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap='YlGn', aspect='auto')
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f'{c:g}' for c in mat.columns])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f'{r:g}' for r in mat.index])
    ax.set_xlabel('Drain amount')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', color='gray', fontsize=8)
                continue
            rng = (vmax - vmin) or 1.0
            dark = (v - vmin) / rng > 0.6            # white text on dark cells
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=9,
                    color='white' if dark else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.text(0.0, 1.02, 'low = harder', transform=ax.transAxes,
            fontsize=8, color='#6B7280')


def plot_heatmaps(table, condition, predator_type, ylabel, out_dir):
    print("  Plotting population / fitness / lifetime heatmaps ...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, (title, fmt)) in zip(axs.flat, METRICS.items()):
        mat = pivot(table, col)
        if mat.isna().all().all():
            ax.axis('off')
            ax.set_title(f'{title}\n(no data)', fontsize=11, color='gray')
            continue
        heatmap(ax, mat, title, ylabel, fmt)
    plt.suptitle(f'{predator_type.capitalize()} predator sweep — {condition} '
                 f'(mean over {table["seed"].nunique()} seeds)',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, f'predator_metrics_{predator_type}_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(curves, condition, predator_type, ylabel, smooth, out_dir):
    """Over-generation trajectories as a small-multiples grid: one ROW per
    predator count, one COLUMN per curve metric (fitness + population). Within a
    cell, one line per drain level (mean across seeds, shaded +/-1 std). This is
    what the heatmaps can't show — whether a combo crashes early, recovers, or
    drifts. Seeds are aligned on the generation index and averaged NaN-skipping,
    so walltime-truncated runs just thin out the late-generation points."""
    print("  Plotting fitness / population time series ...")
    counts = sorted({c for (c, _d, _s) in curves})
    drains = sorted({d for (_c, d, _s) in curves})
    metrics = list(CURVE_METRICS)
    nrows, ncols = len(counts), len(metrics)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows),
                            sharex=True, squeeze=False)
    for ri, count in enumerate(counts):
        for ci, m in enumerate(metrics):
            ax = axs[ri][ci]
            for di, drain in enumerate(drains):
                colour = DRAIN_COLOURS[di % len(DRAIN_COLOURS)]
                series = [df[m] for (c, d, _s), df in curves.items()
                          if c == count and d == drain and m in df.columns]
                if not series:
                    continue
                wide = pd.concat(series, axis=1).sort_index()
                mean = wide.mean(axis=1)
                std = wide.std(axis=1).fillna(0)
                if smooth > 1:
                    mean = mean.rolling(smooth, min_periods=1, center=True).mean()
                    std = std.rolling(smooth, min_periods=1, center=True).mean()
                gens = mean.index.values
                ax.plot(gens, mean.values, color=colour, lw=1.8, label=f'drain {drain:g}')
                ax.fill_between(gens, mean.values - std.values, mean.values + std.values,
                                color=colour, alpha=0.12)
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(CURVE_METRICS[m], fontsize=12, fontweight='bold')
            if ri == nrows - 1:
                ax.set_xlabel('Generation')
            if ci == 0:
                ax.set_ylabel(f'{ylabel} = {count:g}', fontsize=10, fontweight='bold')
    axs[0][ncols - 1].legend(fontsize=8, title='drain', loc='best')
    plt.suptitle(f'{predator_type.capitalize()} predator sweep — trajectories over '
                 f'generations, {condition} (mean +/- std over seeds)',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, f'predator_metrics_{predator_type}_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description='Population/fitness/lifetime heatmaps for a predator sweep.')
    ap.add_argument('--logs-root', default='logs',
                    help='Root to search recursively for run folders.')
    ap.add_argument('--predator-type', default='roaming',
                    choices=['roaming', 'patrol'],
                    help='Which sweep to analyse: roaming (count axis = '
                         'roaming_predator_count) or patrol (predators_per_patch).')
    ap.add_argument('--condition', default='evolution',
                    choices=['evolution', 'learning', 'pure_rl'],
                    help='Which experimental condition (predator sweeps are evolution).')
    ap.add_argument('--final-window', type=int, default=50,
                    help='Average each metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Clip every run to this generation before taking the tail '
                         '(use for a fair comparison across walltime-truncated runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the time-series plot (1 = none).')
    ap.add_argument('--out', default='output/predator_metrics')
    args = ap.parse_args()

    ylabel = PRED_AXIS[args.predator_type]['ylabel']
    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering {args.predator_type} / {args.condition} runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root, args.condition, args.predator_type)
    if not runs:
        raise SystemExit(f"No {args.predator_type} {args.condition} runs with params.json + "
                         f"generations.csv under {args.logs_root}. Has the sweep finished?")
    print(f"Found {len(runs)} runs "
          f"({len({r['count'] for r in runs})} counts x "
          f"{len({r['drain'] for r in runs})} drains x "
          f"{len({r['seed'] for r in runs})} seeds). Reading signals ...")

    table, curves = build_table(runs, args.final_window, args.max_gen)

    # Surface run-length spread so truncated cells are visible, not silent.
    print(f"\nRun length (gen reached): min={int(table['gen_max'].min())} "
          f"max={int(table['gen_max'].max())} "
          f"median={int(table['gen_max'].median())}")
    truncated = table[table['gen_max'] < table['gen_max'].max()]
    if len(truncated):
        print(f"  {len(truncated)}/{len(table)} runs ended before the longest run "
              f"(gen {int(table['gen_max'].max())}). "
              f"Pass --max-gen {int(table['gen_max'].min())} to compare all on a common horizon.")

    agg = table.groupby(['count', 'drain']).agg(
        total_agents=('total_agents', 'mean'),
        peak_population=('peak_population', 'mean'),
        avg_fitness=('avg_fitness', 'mean'),
        avg_lifetime=('avg_lifetime', 'mean'),
        gen_min=('gen_max', 'min'),
        gen_max=('gen_max', 'max'),
        n_seeds=('seed', 'nunique'),
    ).reset_index().sort_values(['count', 'drain'])

    print("\n-- Aggregated (mean over seeds) --")
    print(agg.to_string(index=False))

    summary_path = os.path.join(args.out, f'predator_metrics_{args.predator_type}_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    plot_heatmaps(table, args.condition, args.predator_type, ylabel, args.out)
    plot_curves(curves, args.condition, args.predator_type, ylabel, args.smooth, args.out)
    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
