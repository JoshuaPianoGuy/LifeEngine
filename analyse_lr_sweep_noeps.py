"""
analyse_lr_sweep_noeps.py
======================================================================
Learning-rate sweep analysis for the NO-EPSILON family of the complex-
environment learning runs (roam60d5_noeps_lr<LR>_seed<SEED>/).

These are the pure on-policy REINFORCE runs (epsilon_enabled: false): no
epsilon-greedy exploration at all, so the ONLY factor is the learning rate.
The companion analyse_lr_sweep_epsilon.py covers the epsilon-greedy family,
which additionally sweeps epsilon_start x decay_shape — the two families are
kept in SEPARATE scripts / output folders on purpose.

Family layout (all runs share logs/learning/standard/auto-run/, split by
params.json -> epsilon_enabled):

    no-epsilon   learning_rate in {0.005, 0.01, 0.02, 0.03}
                 x 3 seeds (1, 42, 999)              -> 12 runs

params.json is authoritative for learning_rate / map_seed / epsilon_enabled
(folder names are only a hint). A run is DROPPED here if epsilon_enabled is
true — those belong to the epsilon script.

Metrics (all read from generations.csv, exactly as the other sweep scripts):
    avg_fitness              -- average fitness of the population
    top20percent_fitness     -- average fitness of the top 20%
    peak_population          -- peak organism count in a generation
    total_agents             -- total organisms alive in a generation
    avg_learned_weight_diff  -- mean absolute weight difference: mean |active - genome| (see calculateDrift)

Outputs (into --out), per metric:
    noeps_curves_<metric>.png   -- metric over generations, one line per LR
                                   (mean +/- std across the 3 seeds).
    noeps_final_<metric>.png    -- endpoint (mean of the last N generations) vs
                                   LR: each SEED's own last-N-gen mean as a
                                   point, PLUS the overall mean +/- std across
                                   the 3 seeds as a bold line. NOT written for
                                   Mean absolute weight difference (a diagnostic best read over time, not as
                                   a single endpoint).
Plus noeps_summary.csv: tidy (lr, metric, mean, std, sem, count) aggregated
across the 3 seeds so learning rate is the only factor.

Usage
-----
    python analyse_lr_sweep_noeps.py \
        --logs-root logs/learning/standard/auto-run \
        --final-window 20 \
        --out output/lr_sweep_noeps

    # restrict / reorder the metrics:
    python analyse_lr_sweep_noeps.py --metrics avg_fitness peak_population
"""

import argparse
import glob
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# Distinct colour per learning rate (assigned in sorted-LR order) — matches the
# other LR / epsilon sweep scripts so every figure in the set reads the same.
LR_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']
# Distinct colour per world seed (assigned in sorted-seed order) for the
# per-seed endpoint plots.
SEED_COLOURS = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED', '#0891B2']

# Metric column -> (axis/title label, accent colour, per-cell value format).
# Colours reuse the palette shared across the sweep scripts: fitness blue/green,
# population cyan/brown, mean absolute weight difference purple.
METRICS = {
    'avg_fitness':             ('Average fitness',           '#2563EB', '{:.3f}'),
    'top20percent_fitness':    ('Top-20% fitness',           '#059669', '{:.3f}'),
    'peak_population':          ('Peak population',           '#0891B2', '{:.0f}'),
    'total_agents':            ('Total agents',              '#92400E', '{:.0f}'),
    'avg_learned_weight_diff': ('Mean absolute weight difference', '#7C3AED', '{:.4g}'),
}
DEFAULT_METRICS = list(METRICS)

# Mean absolute weight difference gets NO endpoint plot: it is a learning DIAGNOSTIC whose value is in how it
# changes over generations, not in a single last-N-gen number (higher/lower isn't
# "better"). It still gets the over-generation curve.
NO_FINAL_METRICS = {'avg_learned_weight_diff'}


def discover_runs(logs_root):
    """Find every run folder under logs_root with params.json + generations.csv
    whose params.json says epsilon_enabled is FALSE. Returns a list of dicts:
    {lr, seed, dir, gen_csv}.

    params.json is authoritative (folder names are only a hint). Epsilon-ON runs
    are skipped here — they belong to analyse_lr_sweep_epsilon.py."""
    runs = []
    for params_path in glob.glob(os.path.join(logs_root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        gen_csv = os.path.join(run_dir, 'generations.csv')
        if not os.path.exists(gen_csv):
            print(f"  skip {os.path.basename(run_dir)} — no generations.csv")
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
            # epsilon_enabled defaults to True if an older log omitted it, so a
            # missing flag keeps the run OUT of the no-epsilon family.
            if bool(p.get('epsilon_enabled', True)):
                continue
            lr = float(p['learning_rate'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'lr': lr, 'seed': seed, 'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, dataframe).

    finals[m] = mean of metric m over the last `window` recorded generations
    (<= max_gen) — the "mean of the last N generations" endpoint for this run.
    Deduplicates on 'generation' (keep last) so a doubled/appended CSV neither
    double-counts the tail nor breaks the axis=1 concat in the curve plot."""
    df = pd.read_csv(gen_csv)
    df = df[df['generation'] <= max_gen].copy()
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    finals = {}
    for m in metrics:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors='coerce').dropna()
            tail = col.tail(window)
            finals[m] = tail.mean() if len(tail) else np.nan
        else:
            finals[m] = np.nan
    return finals, df


def build_table(runs, metrics, window, max_gen):
    """Per-run final metrics -> long dataframe (one row per run) + curve frames.
    curves is keyed by (lr, seed)."""
    rows, curves = [], {}
    for r in runs:
        finals, df = load_run(r['gen_csv'], metrics, window, max_gen)
        row = {'lr': r['lr'], 'seed': r['seed'],
               'n_gens': int(df['generation'].max()) if len(df) else 0}
        row.update(finals)
        rows.append(row)
        curves[(r['lr'], r['seed'])] = df
        vals = '  '.join(f"{m}={finals[m]:.4g}" if not np.isnan(finals[m]) else f"{m}=nan"
                         for m in metrics)
        print(f"  lr={r['lr']:<8g} seed={r['seed']:<5d} {vals}  ({row['n_gens']} gens)")
    return pd.DataFrame(rows), curves


def aggregate(table, metrics):
    """Aggregate each metric across seeds so LR is the only factor.
    Returns tidy long frame: lr, metric, mean, std, sem, count."""
    frames = []
    for m in metrics:
        g = table.groupby('lr')[m].agg(['mean', 'std', 'sem', 'count']).reset_index()
        g.insert(1, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'lr'])


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


def _set_lr_axis(ax, lrs):
    """Configure a learning-rate x-axis consistently across every summary plot:
    log scale (LRs span a wide range) with the actual LR values as ticks, shown
    as plain decimals (0.005, 0.01, ...) — NOT scientific notation. Minor ticks
    are removed so no stray '6x10^-3'-style labels appear between them."""
    ax.set_xscale('log')
    ax.set_xticks(lrs)
    ax.set_xticklabels([f'{lr:g}' for lr in lrs])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Learning rate (log scale)')


def plot_curves(curves, metric, smooth, out_dir):
    """Over-generation curve for one metric: one line per LR (mean across the 3
    seeds), shaded +/-1 std band. Shows convergence speed + stability."""
    label, _colour, _fmt = METRICS[metric]
    print(f"  Plotting {metric} curves ...")
    lrs = sorted({lr for (lr, _seed) in curves})
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    for i, lr in enumerate(lrs):
        colour = LR_COLOURS[i % len(LR_COLOURS)]
        series = []
        for (l, _seed), df in curves.items():
            if l != lr or metric not in df.columns:
                continue
            series.append(pd.to_numeric(df.set_index('generation')[metric],
                                        errors='coerce'))
        if not series:
            continue
        wide = pd.concat(series, axis=1).sort_index()
        mean = wide.mean(axis=1)
        std = wide.std(axis=1).fillna(0)
        if smooth > 1:
            mean = mean.rolling(smooth, min_periods=1, center=True).mean()
            std = std.rolling(smooth, min_periods=1, center=True).mean()
        gens = mean.index.values
        ax.plot(gens, mean.values, color=colour, lw=2, label=f'lr={lr:g}')
        ax.fill_between(gens, mean.values - std.values, mean.values + std.values,
                        color=colour, alpha=0.15)
        plotted = True
    ax.set_xlabel('Generation')
    ax.set_ylabel(label)
    ax.set_title(f'No-epsilon LR sweep — {label} over generations '
                 f'(mean ± std across seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=10, title='learning rate')
    plt.tight_layout()
    path = os.path.join(out_dir, f'noeps_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_final(table, agg, metric, out_dir, window):
    """Endpoint (mean of the last N generations) vs LR. Shows BOTH levels the
    other sweep scripts show: each SEED's own last-N-gen mean as a coloured
    point, plus the OVERALL mean +/- std across the 3 seeds as a bold black
    errorbar line. Log x-axis with plain-decimal LR ticks."""
    label, _colour, _fmt = METRICS[metric]
    print(f"  Plotting final {metric} vs learning rate ...")
    lrs = sorted(table['lr'].unique())
    sub = agg[agg['metric'] == metric].sort_values('lr')
    fig, ax = plt.subplots(figsize=(9, 6))

    # Overall mean across seeds (bold line), then each seed's own point on top.
    ax.errorbar(sub['lr'], sub['mean'], yerr=sub['std'].fillna(0), fmt='-o',
                color='#111827', ecolor='#6B7280', elinewidth=2, capsize=6,
                lw=2, markersize=9, zorder=3, label='mean ± std (seeds)')
    for i, (seed, s) in enumerate(table.groupby('seed')):
        colour = SEED_COLOURS[i % len(SEED_COLOURS)]
        ax.scatter(s['lr'], s[metric], s=55, color=colour, alpha=0.85,
                   zorder=4, label=f'seed {seed}')

    _set_lr_axis(ax, lrs)
    ax.set_ylabel(f'Final {label}\n(mean of last {window} gens)')
    ax.set_title(f'No-epsilon LR sweep — final {label} '
                 f'(per seed + mean across seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f'noeps_final_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='LR sweep — no-epsilon family.')
    ap.add_argument('--logs-root', default='logs/learning/standard/auto-run',
                    help='Folder containing the run dirs (both families live here).')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='generations.csv columns to plot (default: all five).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Average each metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curve plots (1 = no smoothing).')
    ap.add_argument('--out', default='output/lr_sweep_noeps')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering NO-EPSILON runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No epsilon_enabled=false runs with params.json + "
                         f"generations.csv under {args.logs_root}.")
    print(f"Found {len(runs)} no-epsilon runs. Reading {', '.join(args.metrics)} "
          f"(last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.metrics, args.final_window, args.max_gen)
    agg = aggregate(table, args.metrics)

    print("\n-- Aggregated (across seeds) --")
    for m in args.metrics:
        label = METRICS[m][0]
        sub = agg[agg['metric'] == m][['lr', 'mean', 'std', 'sem', 'count']]
        print(f"\n{label} ({m}):")
        print(sub.to_string(index=False,
              formatters={'lr': '{:g}'.format, 'mean': '{:.4g}'.format,
                          'std': '{:.4g}'.format, 'sem': '{:.4g}'.format}))

    summary_path = os.path.join(args.out, 'noeps_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    for m in args.metrics:
        plot_curves(curves, m, args.smooth, args.out)
        if m not in NO_FINAL_METRICS:
            plot_final(table, agg, m, args.out, args.final_window)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
