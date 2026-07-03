"""
analyse_lr_sweep.py
======================================================================
Aggregate + compare the learning-rate sweep produced by sweep_lr.sh.

The sweep runs 3 learning rates x 3 seeds = 9 jobs, each writing a run
folder (lr<LR>_seed<SEED>/) containing:
    params.json      -- authoritative learning_rate + map_seed for the run
    generations.csv  -- per-generation population metrics (see Logger.js)

This script discovers those folders, reads the final-fitness of each run
(mean of the last --final-window generations, to smooth run-to-run noise),
then aggregates across the 3 seeds so learning rate is the only factor:

    1. lr_sweep_final.png   -- final fitness vs learning rate, mean +/- spread
                               over the 3 seeds, with the individual seed
                               points overlaid. Answers "which LR is best?".
    2. lr_sweep_curves.png  -- learning curves: metric over generations, one
                               line per LR (mean across seeds, shaded +/-1 std).
    3. lr_sweep_summary.csv -- the aggregated table behind plot 1.

Usage
-----
    python analyse_lr_sweep.py \
        --logs-root logs/learning/standard/auto-run \
        --metric best_fitness \
        --final-window 20 \
        --out output/lr_sweep

`generations.csv` fitness columns you can pass to --metric:
    avg_fitness, top20percent_fitness, best_fitness
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
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# Distinct colour per learning rate (assigned in sorted-LR order).
LR_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']


def discover_runs(logs_root):
    """Find every run folder under logs_root that has both params.json and
    generations.csv, returning a list of dicts: {lr, seed, dir, gen_csv}.

    params.json is authoritative for lr/seed (folder names are only a hint),
    so the sweep still analyses correctly even if a folder was renamed."""
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
            lr = float(p['learning_rate'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'lr': lr, 'seed': seed, 'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def final_value(gen_csv, metric, window, max_gen):
    """Final metric for one run: mean of the metric over the last `window`
    recorded generations (<= max_gen). Returns (value, full_dataframe)."""
    df = pd.read_csv(gen_csv)
    if metric not in df.columns:
        raise KeyError(f"'{metric}' not in {gen_csv}. Have: {list(df.columns)}")
    df = df[df['generation'] <= max_gen].copy()
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.sort_values('generation')
    tail = df[metric].dropna().tail(window)
    return (tail.mean() if len(tail) else np.nan), df


def build_table(runs, metric, window, max_gen):
    """Per-run final metric -> long dataframe (one row per run)."""
    rows, curves = [], {}
    for r in runs:
        val, df = final_value(r['gen_csv'], metric, window, max_gen)
        rows.append({'lr': r['lr'], 'seed': r['seed'], 'final': val,
                     'n_gens': int(df['generation'].max()) if len(df) else 0})
        curves[(r['lr'], r['seed'])] = df
        print(f"  lr={r['lr']:<8g} seed={r['seed']:<5d} "
              f"{metric}={val:.4f}  ({rows[-1]['n_gens']} gens)")
    return pd.DataFrame(rows), curves


def aggregate(table):
    """Aggregate across seeds so LR is the only factor. mean/std/sem/n per LR."""
    g = table.groupby('lr')['final']
    agg = g.agg(['mean', 'std', 'sem', 'count']).reset_index()
    agg = agg.sort_values('lr').reset_index(drop=True)
    return agg


def plot_final_vs_lr(table, agg, metric, out_dir):
    """Final fitness vs learning rate: mean +/- std across seeds, seed points
    overlaid. Log x-axis (learning rates span decades)."""
    print("  Plotting final fitness vs learning rate ...")
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(agg['lr'], agg['mean'], yerr=agg['std'].fillna(0),
                fmt='-o', color='#111827', ecolor='#6B7280', elinewidth=2,
                capsize=6, lw=2, markersize=9, zorder=3, label='mean ± std (n seeds)')

    # Individual seed points (jittered slightly on log axis for visibility).
    for seed, sub in table.groupby('seed'):
        ax.scatter(sub['lr'], sub['final'], s=45, alpha=0.7, zorder=4,
                   label=f'seed {seed}')

    # Mark the best-mean LR.
    best = agg.loc[agg['mean'].idxmax()]
    ax.axvline(best['lr'], color='#059669', ls='--', alpha=0.6, zorder=1)
    ax.annotate(f"best: lr={best['lr']:g}\n{best['mean']:.3f}",
                xy=(best['lr'], best['mean']), xytext=(10, -30),
                textcoords='offset points', color='#059669', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Learning rate (log scale)')
    ax.set_ylabel(f'Final {metric}')
    ax.set_title(f'Learning-rate sweep: final {metric} vs learning rate',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(agg['lr'])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(curves, metric, smooth, out_dir):
    """Learning curves over generations: one line per LR (mean across seeds),
    shaded +/-1 std band. Shows convergence speed + stability, not just the end."""
    print("  Plotting learning curves ...")
    lrs = sorted({lr for (lr, _seed) in curves})
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lr in enumerate(lrs):
        colour = LR_COLOURS[i % len(LR_COLOURS)]
        # Stack this LR's seeds on a common generation index.
        series = []
        for (l, seed), df in curves.items():
            if l != lr:
                continue
            s = pd.to_numeric(df.set_index('generation')[metric], errors='coerce')
            series.append(s)
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

    ax.set_xlabel('Generation')
    ax.set_ylabel(metric)
    ax.set_title(f'Learning-rate sweep: {metric} over generations '
                 f'(mean ± std across seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, title='learning rate')
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='Aggregate the learning-rate sweep.')
    ap.add_argument('--logs-root', default='logs/learning/standard/auto-run',
                    help='Folder containing the lr<LR>_seed<SEED> run dirs.')
    ap.add_argument('--metric', default='best_fitness',
                    help='generations.csv column to compare '
                         '(avg_fitness | top20percent_fitness | best_fitness).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Average the metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the learning-curve plot.')
    ap.add_argument('--out', default='output/lr_sweep')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No runs with params.json + generations.csv under "
                         f"{args.logs_root}. Has the sweep finished?")
    print(f"Found {len(runs)} runs. Reading final {args.metric} "
          f"(last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.metric, args.final_window, args.max_gen)
    agg = aggregate(table)

    print("\n-- Aggregated (across seeds) --")
    print(agg.to_string(index=False,
          formatters={'lr': '{:g}'.format, 'mean': '{:.4f}'.format,
                      'std': '{:.4f}'.format, 'sem': '{:.4f}'.format}))
    best = agg.loc[agg['mean'].idxmax()]
    print(f"\nBest learning rate: {best['lr']:g} "
          f"(mean {args.metric} = {best['mean']:.4f} over {int(best['count'])} seeds)\n")

    summary_path = os.path.join(args.out, 'lr_sweep_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    plot_final_vs_lr(table, agg, args.metric, args.out)
    plot_curves(curves, args.metric, args.smooth, args.out)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
