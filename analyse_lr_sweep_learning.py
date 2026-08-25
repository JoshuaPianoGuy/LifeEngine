"""
analyse_lr_sweep_learning.py
======================================================================
Learning-DIAGNOSTIC companion to analyse_lr_sweep.py.

analyse_lr_sweep.py compares the learning-rate sweep on FITNESS ("which LR is
best?"). This script runs the SAME discovery over the SAME sweep runs, but looks
at the non-fitness signals that describe HOW the policies learn, per learning
rate:

    peak_population          -- carrying capacity reached (ecological outcome)
    total_agents             -- total agents alive (plotted alongside peak, as
                                the individual-run analyse_weights.py does)
    avg_learned_weight_diff  -- mean absolute weight difference: mean |active - genome| over the WHOLE
                                generation (every agent that lived, not just the
                                selected cohort), i.e. how much is learned WITHIN
                                a lifetime (see Logger._calcDrift)
    avg_network_weight_mag   -- RMS magnitude of the network weights, likewise
                                averaged over all agents (see Logger._calcRMSWeight)

Like the fitness script it uses ALL runs (3 LRs x 3 seeds), collapses each run
to a final value (mean of the last --final-window generations, to smooth
run-to-run noise), then aggregates across the seeds so learning rate is the only
factor.

Unlike the fitness script it does NOT pick a "best" LR: these are DESCRIPTIVE
diagnostics (higher isn't better), so it just shows the level + spread.

A duplicate-generation guard (drop_duplicates on 'generation') makes it robust
to a doubled/appended CSV, which would otherwise crash the curve plot.

Outputs (into --out):
    lr_sweep_learning_final.png    -- 1xN endpoint-vs-LR panels (mean +/- std over seeds)
    lr_sweep_learning_curves.png   -- Nx1 over-generation panels (one line per LR)
    lr_sweep_learning_summary.csv  -- tidy (lr, metric, mean, std, sem, count)

Usage
-----
    python analyse_lr_sweep_learning.py \
        --logs-root logs/learning/standard/auto-run \
        --final-window 20 \
        --out output/lr_sweep_learning

    # restrict to a subset / reorder the panels:
    python analyse_lr_sweep_learning.py --metrics avg_learned_weight_diff avg_network_weight_mag
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

# Distinct colour per learning rate (assigned in sorted-LR order) — matches
# analyse_lr_sweep.py so the two scripts' curve plots read the same.
LR_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

# Metric column -> (axis/panel label, accent colour, per-cell value format).
# Colours reuse the analyse_weights.py PALETTE so figures match: drift=#7C3AED,
# mag=#DB2777; population uses the peak-population cyan from analyse_weights.py.
METRICS = {
    'peak_population':         ('Peak population',            '#0891B2', '{:.0f}'),
    'total_agents':            ('Total agents',               '#92400E', '{:.0f}'),
    'avg_learned_weight_diff': ('Mean absolute weight difference',  '#7C3AED', '{:.4f}'),
    'avg_network_weight_mag':  ('RMS weight magnitude',       '#DB2777', '{:.4f}'),
}
DEFAULT_METRICS = list(METRICS)


def discover_runs(logs_root):
    """Find every run folder under logs_root that has both params.json and
    generations.csv, returning a list of dicts: {lr, seed, dir, gen_csv}.

    params.json is authoritative for lr/seed (folder names are only a hint)."""
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


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, dataframe).

    finals[m] = mean of metric m over the last `window` recorded generations.
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
    """Per-run final metrics -> long dataframe (one row per run) + curve frames."""
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


def plot_final(table, agg, metrics, out_dir):
    """Endpoint-vs-LR: one panel per metric, mean +/- std across seeds, with the
    individual seed points overlaid. Log x-axis (learning rates span decades)."""
    print("  Plotting final learning-diagnostics vs learning rate ...")
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5))
    axs = np.atleast_1d(axs)
    lrs = sorted(table['lr'].unique())
    for ax, m in zip(axs, metrics):
        label, colour, _fmt = METRICS[m]
        sub = agg[agg['metric'] == m].sort_values('lr')
        if sub['mean'].isna().all():
            ax.set_title(f'{label}\n(no data)', color='gray'); ax.axis('off'); continue
        ax.errorbar(sub['lr'], sub['mean'], yerr=sub['std'].fillna(0),
                    fmt='-o', color=colour, ecolor='#6B7280', elinewidth=2,
                    capsize=6, lw=2, markersize=9, zorder=3,
                    label='mean ± std (seeds)')
        for seed, s in table.groupby('seed'):
            ax.scatter(s['lr'], s[m], s=40, alpha=0.6, zorder=4, label=f'seed {seed}')
        ax.set_xscale('log')
        ax.set_xticks(lrs)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel('Learning rate (log scale)')
        ax.set_ylabel(f'Final {label}')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
    axs[0].legend(fontsize=8)
    plt.suptitle('LR sweep — learning diagnostics (final, mean ± std over seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_learning_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(curves, metrics, smooth, out_dir):
    """Over-generation panels (one per metric): one line per LR (mean across
    seeds), shaded +/-1 std band. Shows trajectory + stability, not just the end."""
    print("  Plotting learning-diagnostic curves ...")
    lrs = sorted({lr for (lr, _seed) in curves})
    n = len(metrics)
    fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, metrics):
        label, _colour, _fmt = METRICS[m]
        plotted = False
        for i, lr in enumerate(lrs):
            colour = LR_COLOURS[i % len(LR_COLOURS)]
            series = []
            for (l, _seed), df in curves.items():
                if l != lr or m not in df.columns:
                    continue
                s = pd.to_numeric(df.set_index('generation')[m], errors='coerce')
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
            plotted = True
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=9, title='learning rate')
    axs[-1].set_xlabel('Generation')
    plt.suptitle('LR sweep — learning diagnostics over generations '
                 '(mean ± std across seeds)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_learning_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='Learning-diagnostic view of the LR sweep.')
    ap.add_argument('--logs-root', default='logs/learning/standard/auto-run',
                    help='Folder containing the lr<LR>_seed<SEED> run dirs.')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='generations.csv columns to panel (default: all three).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Average each metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curve plot (1 = no smoothing).')
    ap.add_argument('--out', default='output/lr_sweep_learning')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No runs with params.json + generations.csv under "
                         f"{args.logs_root}. Has the sweep finished?")
    print(f"Found {len(runs)} runs. Reading {', '.join(args.metrics)} "
          f"(last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.metrics, args.final_window, args.max_gen)
    agg = aggregate(table, args.metrics)

    print("\n-- Aggregated (across seeds) --")
    for m in args.metrics:
        label = METRICS[m][0]
        sub = agg[agg['metric'] == m][['lr', 'mean', 'std', 'sem', 'count']]
        print(f"\n{label} ({m}):")
        # {:.4g} (sig-figs) not {:.4f}: mean absolute weight difference spans ~1e-5..1e-3, which fixed
        # 4-decimal formatting would flatten to 0.0000.
        print(sub.to_string(index=False,
              formatters={'lr': '{:g}'.format, 'mean': '{:.4g}'.format,
                          'std': '{:.4g}'.format, 'sem': '{:.4g}'.format}))

    summary_path = os.path.join(args.out, 'lr_sweep_learning_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    plot_final(table, agg, args.metrics, args.out)
    plot_curves(curves, args.metrics, args.smooth, args.out)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
