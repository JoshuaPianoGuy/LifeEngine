"""
analyse_lr_sweep_epsilon.py
======================================================================
Learning-rate sweep analysis for the EPSILON-GREEDY family of the complex-
environment learning runs (roam60d5_lr<LR>_eps<E>_<shape>_seed<SEED>/).

These are the runs that DO use epsilon-greedy exploration (epsilon_enabled:
true). The goal is still to tune the LEARNING RATE, but the family additionally
sweeps two exploration nuisance factors, so learning rate is compared WITHIN
each decay shape (averaging over epsilon_start). The companion
analyse_lr_sweep_noeps.py covers the pure-REINFORCE family — the two families
are kept in SEPARATE scripts / output folders on purpose.

Family layout (all runs share logs/learning/standard/auto-run/, split by
params.json -> epsilon_enabled):

    epsilon-ON   learning_rate      in {0.005, 0.01, 0.02, 0.03}
                 epsilon_start      in {0.2, 0.3, 0.5, 0.7}
                 epsilon_decay_shape in {linear, quadratic, sublinear}
                 x 3 seeds (1, 42, 999)             -> 144 runs

params.json is authoritative for every factor (folder names are only a hint).
A run is DROPPED here if epsilon_enabled is false — those belong to the
no-epsilon script.

Layout (per the requested split): every figure has one PANEL per decay shape,
and learning rate is the comparison inside each panel:

    [ linear ]  [ quadratic ]  [ sublinear ]

Within a panel, everything is averaged over the 4 epsilon_start values AND the
3 seeds, so a single line/point per learning rate answers "which LR is best
under this decay shape?". The std band / error bars are the spread across those
12 runs (4 eps_start x 3 seeds).

Metrics (all read from generations.csv, exactly as the other sweep scripts):
    avg_fitness              -- average fitness of the population
    top20percent_fitness     -- average fitness of the top 20%
    peak_population          -- peak organism count in a generation
    total_agents             -- total organisms alive in a generation
    avg_learned_weight_diff  -- mean absolute weight difference: mean |active - genome| (see calculateDrift)

Outputs (into --out), per metric:
    eps_curves_<metric>.png   -- metric over generations, panel per decay, one
                                 line per LR (mean +/- std across seeds AND
                                 epsilon_start).
    eps_final_<metric>.png    -- endpoint (mean of the last N generations) vs LR,
                                 panel per decay. Each SEED's own last-N-gen mean
                                 (averaged over its 4 epsilon_start runs) as a
                                 point, PLUS the overall mean +/- std across the
                                 3 seeds as a bold line. NOT written for mean absolute weight difference (a
                                 diagnostic best read over time, not as an
                                 endpoint).
Plus eps_summary.csv: tidy (decay, lr, metric, mean, std, sem, count).

Usage
-----
    python analyse_lr_sweep_epsilon.py \
        --logs-root logs/learning/standard/auto-run \
        --final-window 20 \
        --out output/lr_sweep_epsilon

    # restrict / reorder the metrics or decay panels:
    python analyse_lr_sweep_epsilon.py --metrics avg_fitness peak_population
    python analyse_lr_sweep_epsilon.py --panel-order sublinear linear quadratic
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
    whose params.json says epsilon_enabled is TRUE. Returns a list of dicts:
    {lr, decay, eps_start, seed, dir, gen_csv}.

    params.json is authoritative (folder names are only a hint). No-epsilon runs
    are skipped here — they belong to analyse_lr_sweep_noeps.py."""
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
            # epsilon_enabled defaults to True if an older log omitted it.
            if not bool(p.get('epsilon_enabled', True)):
                continue
            lr = float(p['learning_rate'])
            decay = str(p['epsilon_decay_shape'])
            eps_start = float(p['epsilon_start'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'lr': lr, 'decay': decay, 'eps_start': eps_start,
                     'seed': seed, 'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, dataframe).

    finals[m] = mean of metric m over the last `window` recorded generations
    (<= max_gen). Deduplicates on 'generation' (keep last) so a doubled/appended
    CSV neither double-counts the tail nor breaks the axis=1 concat later."""
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
    curves is keyed by (decay, lr, eps_start, seed)."""
    rows, curves = [], {}
    for r in runs:
        finals, df = load_run(r['gen_csv'], metrics, window, max_gen)
        row = {'decay': r['decay'], 'lr': r['lr'], 'eps_start': r['eps_start'],
               'seed': r['seed'],
               'n_gens': int(df['generation'].max()) if len(df) else 0}
        row.update(finals)
        rows.append(row)
        curves[(r['decay'], r['lr'], r['eps_start'], r['seed'])] = df
        vals = '  '.join(f"{m}={finals[m]:.4g}" if not np.isnan(finals[m]) else f"{m}=nan"
                         for m in metrics)
        print(f"  {r['decay']:<10} lr={r['lr']:<8g} eps={r['eps_start']:<4g} "
              f"seed={r['seed']:<5d} {vals}  ({row['n_gens']} gens)")
    return pd.DataFrame(rows), curves


def aggregate(table, metrics):
    """Aggregate each metric across seeds AND epsilon_start so, within a decay
    shape, learning rate is the only factor. Returns tidy long frame:
    decay, lr, metric, mean, std, sem, count."""
    frames = []
    for m in metrics:
        g = (table.groupby(['decay', 'lr'])[m]
                   .agg(['mean', 'std', 'sem', 'count']).reset_index())
        g.insert(2, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'decay', 'lr'])


def ordered_panels(table, panel_order):
    """Decay-shape panels present in the data, in panel_order (extras appended)."""
    present = list(dict.fromkeys(table['decay']))
    panels = [g for g in panel_order if g in present]
    panels += [g for g in present if g not in panels]
    return panels


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


def _set_lr_axis(ax, lrs):
    """Configure a learning-rate x-axis consistently across every panel/plot:
    log scale (LRs span a wide range) with the actual LR values as ticks, shown
    as plain decimals (0.005, 0.01, ...) — NOT scientific notation. Minor ticks
    are removed so no stray '6x10^-3'-style labels appear between them."""
    ax.set_xscale('log')
    ax.set_xticks(lrs)
    ax.set_xticklabels([f'{lr:g}' for lr in lrs])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Learning rate (log scale)')


def plot_curves(curves, metric, panels, smooth, out_dir):
    """Over-generation curves, one panel per decay shape. Within each panel: one
    line per LR (mean across seeds AND epsilon_start), shaded +/-1 std band.
    Panels share both axes so the decay shapes are directly comparable."""
    label, _colour, _fmt = METRICS[metric]
    lrs = sorted({lr for (_d, lr, _e, _s) in curves})
    print(f"  Plotting {metric} curves ...")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        plotted = False
        for i, lr in enumerate(lrs):
            colour = LR_COLOURS[i % len(LR_COLOURS)]
            series = []
            for (d, l, _e, _seed), df in curves.items():
                if d != panel or l != lr or metric not in df.columns:
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
        ax.set_title(f'{panel} decay', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=9, title='learning rate')
    axs[0].set_ylabel(label)
    plt.suptitle(f'Epsilon LR sweep — {label} over generations '
                 f'(mean ± std across seeds & epsilon start)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eps_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_final(table, metric, panels, out_dir, window):
    """Endpoint (mean of the last N generations) vs LR, one panel per decay
    shape. Within each panel, matching the other sweep scripts: each SEED's own
    last-N-gen mean — averaged over its 4 epsilon_start runs — as a coloured
    point, PLUS the overall mean +/- std across the 3 seeds as a bold black
    errorbar line. Panels share the y-axis; log x-axis with plain-decimal LR
    ticks."""
    label, _colour, _fmt = METRICS[metric]
    print(f"  Plotting final {metric} vs learning rate ...")
    lrs = sorted(table['lr'].unique())
    seeds = sorted(table['seed'].unique())
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.6 * n, 5), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        tsub = table[table['decay'] == panel]
        # Collapse the 4 epsilon_start runs of each (seed, lr) to one per-seed
        # mean, then take the mean +/- std of those across the 3 seeds.
        per_seed = tsub.groupby(['seed', 'lr'])[metric].mean().reset_index()
        overall = (per_seed.groupby('lr')[metric]
                           .agg(['mean', 'std']).reset_index().sort_values('lr'))
        ax.errorbar(overall['lr'], overall['mean'], yerr=overall['std'].fillna(0),
                    fmt='-o', color='#111827', ecolor='#6B7280', elinewidth=2,
                    capsize=6, lw=2, markersize=9, zorder=3,
                    label='mean ± std (seeds)')
        for i, seed in enumerate(seeds):
            colour = SEED_COLOURS[i % len(SEED_COLOURS)]
            s = per_seed[per_seed['seed'] == seed].sort_values('lr')
            if s.empty:
                continue
            ax.scatter(s['lr'], s[metric], s=55, color=colour, alpha=0.85,
                       zorder=4, label=f'seed {seed}')
        _set_lr_axis(ax, lrs)
        ax.set_title(f'{panel} decay', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
    axs[0].set_ylabel(f'Final {label}\n(mean of last {window} gens)')
    # One shared legend from the first panel (all panels have the same series).
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=8)
    plt.suptitle(f'Epsilon LR sweep — final {label} '
                 f'(per seed + mean across seeds, over epsilon start)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eps_final_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='LR sweep — epsilon-greedy family.')
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
    ap.add_argument('--panel-order', nargs='+',
                    default=['linear', 'quadratic', 'sublinear'],
                    help='Left-to-right order of the decay-shape panels.')
    ap.add_argument('--out', default='output/lr_sweep_epsilon')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering EPSILON runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No epsilon_enabled=true runs with params.json + "
                         f"generations.csv under {args.logs_root}.")
    print(f"Found {len(runs)} epsilon runs. Reading {', '.join(args.metrics)} "
          f"(last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.metrics, args.final_window, args.max_gen)
    agg = aggregate(table, args.metrics)
    panels = ordered_panels(table, args.panel_order)

    print("\n-- Aggregated (across seeds & epsilon start) --")
    for m in args.metrics:
        label = METRICS[m][0]
        sub = agg[agg['metric'] == m][['decay', 'lr', 'mean', 'std', 'sem', 'count']]
        print(f"\n{label} ({m}):")
        print(sub.to_string(index=False,
              formatters={'lr': '{:g}'.format, 'mean': '{:.4g}'.format,
                          'std': '{:.4g}'.format, 'sem': '{:.4g}'.format}))

    summary_path = os.path.join(args.out, 'eps_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    for m in args.metrics:
        plot_curves(curves, m, panels, args.smooth, args.out)
        if m not in NO_FINAL_METRICS:
            plot_final(table, m, panels, args.out, args.final_window)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
