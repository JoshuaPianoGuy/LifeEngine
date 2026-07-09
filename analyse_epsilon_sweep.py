"""
analyse_epsilon_sweep.py
======================================================================
Aggregate + compare the epsilon exploration experiment, the direct analogue of
analyse_lr_sweep.py / analyse_lr_sweep_learning.py for the epsilon+no-epsilon
runs (run_epsilon_sweep_array.slurm + run_no_epsilon_array.slurm).

The experiment has two families of runs, all in the SAME logs-root
(logs/learning/standard/auto-run/), distinguished by params.json:

    epsilon-ON  (epsilon_enabled: true)
        factors:  epsilon_start  in {0.2, 0.3, 0.5, 0.7}
                  epsilon_decay_shape in {linear, quadratic, sublinear}
                  x 3 seeds  (1, 42, 999)         -> 36 runs
    epsilon-OFF (epsilon_enabled: false)          -> 3 runs (seeds only)
        pure on-policy REINFORCE; epsilon pinned to 0 for the whole lifetime.

Like the LR scripts, each run is collapsed to a FINAL value = mean of the metric
over the last --final-window generations (default 20, smooths run-to-run noise),
then aggregated ACROSS SEEDS so the exploration setting is the only factor. The
std shown on the endpoint plots / summary is the spread across those seeds.

Layout (per the requested split by condition)
----------------------------------------------
Every figure has one PANEL per decay shape (epsilon-on) plus one panel for the
no-epsilon baseline:

    [ linear ] [ quadratic ] [ sublinear ] [ no-epsilon ]

This writes, per metric:

    epsilon_sweep_curves_<metric>.png   (every metric)
        metric over generations: one line per epsilon_start (mean across seeds,
        shaded +/-1 std). The no-epsilon panel is a single mean +/- std band.
    epsilon_sweep_final_<metric>.png    (fitness + population metrics only)
        endpoint (mean of last N gens) vs epsilon_start, mean +/- std across
        seeds with the individual seed points overlaid. The no-epsilon panel is
        the single-condition baseline (one point +/- std across seeds).

Metrics (all read from generations.csv, computed exactly as the LR scripts do).
"Fitness" is the species' fitness metric; "population"/"agents" is the organism
count — different measurements, so labelled separately:
    avg_fitness              -- average fitness of the population       (curves + endpoint)
    top20percent_fitness     -- average fitness of the top 20%          (curves + endpoint)
    peak_population          -- peak organism count in a generation     (curves + endpoint)
    total_agents             -- total organisms alive in a generation   (curves + endpoint)
    avg_learned_weight_diff  -- MAD: mean |active - genome|             (curves only)
    avg_network_weight_mag   -- RMS magnitude of the network weights    (curves only)
The MAD/RMS learning diagnostics get NO endpoint plot (higher/lower isn't
"better"). Plus a tidy epsilon_sweep_summary.csv behind the endpoint plots.

Usage
-----
    python analyse_epsilon_sweep.py \
        --logs-root logs/learning/standard/auto-run \
        --final-window 20 \
        --out output/epsilon_sweep

    # restrict / reorder the metrics:
    python analyse_epsilon_sweep.py --metrics avg_fitness top20percent_fitness
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

warnings.filterwarnings('ignore')

# Sentinel label + numeric level for the epsilon-OFF baseline. Grouping by a real
# number (0.0) keeps the no-epsilon rows out of NaN-group trouble; it is only ever
# drawn in its own panel, never mixed onto the epsilon_start axis.
NO_EPS = 'no-epsilon'
NO_EPS_LEVEL = 0.0
NO_EPS_COLOUR = '#111827'

# Distinct colour per epsilon_start (assigned in sorted order) — reuses the LR
# scripts' palette so the figures read the same.
EPS_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

# Metric column -> (axis/panel label, accent colour, per-cell value format).
# Colours reuse the LR-script palette: fitness blue/green, population cyan/brown,
# MAD purple, RMS pink — so the two experiments' figures match. "Fitness" is the
# species' fitness metric; "population"/"agents" is the organism count — kept
# separate in the labels because they measure different things.
METRICS = {
    'avg_fitness':             ('Average fitness',           '#2563EB', '{:.3f}'),
    'top20percent_fitness':    ('Top-20% fitness',           '#059669', '{:.3f}'),
    'peak_population':          ('Peak population',           '#0891B2', '{:.0f}'),
    'total_agents':            ('Total agents',              '#92400E', '{:.0f}'),
    'avg_learned_weight_diff': ('Learned weight diff (MAD)', '#7C3AED', '{:.4g}'),
    'avg_network_weight_mag':  ('RMS weight magnitude',      '#DB2777', '{:.4g}'),
}
DEFAULT_METRICS = list(METRICS)

# The endpoint (mean-of-last-N-gens) plots cover fitness + population only. MAD /
# RMS are learning diagnostics we only show as over-generation curves, so they
# get no endpoint plot (higher/lower isn't "better" for them anyway).
NO_FINAL_METRICS = {'avg_learned_weight_diff', 'avg_network_weight_mag'}


def discover_runs(logs_root):
    """Find every run folder under logs_root with params.json + generations.csv,
    splitting them by params.json into epsilon-on / epsilon-off.

    params.json is authoritative (folder names are only a hint): epsilon_enabled
    selects the family, and for epsilon-on the (epsilon_start, epsilon_decay_shape)
    pair is the factor. Returns a list of dicts:
        {group, level, seed, dir, gen_csv}
    where group is the decay shape ('linear'/'quadratic'/'sublinear') or the
    NO_EPS sentinel, and level is epsilon_start (NO_EPS_LEVEL for the baseline)."""
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
            seed = int(p['map_seed'])
            # epsilon_enabled defaults to True if an older log omitted it.
            enabled = bool(p.get('epsilon_enabled', True))
            if enabled:
                group = str(p['epsilon_decay_shape'])
                level = float(p['epsilon_start'])
            else:
                group, level = NO_EPS, NO_EPS_LEVEL
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'group': group, 'level': level, 'seed': seed,
                     'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, dataframe).

    finals[m] = mean of metric m over the last `window` recorded generations
    (<= max_gen). Deduplicates on 'generation' (keep last) so a doubled/appended
    CSV neither double-counts the tail nor breaks the axis=1 concat in the curve
    plot. Mirrors final_value() in analyse_lr_sweep.py exactly."""
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
        row = {'group': r['group'], 'level': r['level'], 'seed': r['seed'],
               'n_gens': int(df['generation'].max()) if len(df) else 0}
        row.update(finals)
        rows.append(row)
        curves[(r['group'], r['level'], r['seed'])] = df
        tag = (f"{r['group']:<10} eps_start={r['level']:<4g}"
               if r['group'] != NO_EPS else f"{NO_EPS:<10} (baseline) ")
        vals = '  '.join(f"{m}={finals[m]:.4g}" if not np.isnan(finals[m]) else f"{m}=nan"
                         for m in metrics)
        print(f"  {tag} seed={r['seed']:<5d} {vals}  ({row['n_gens']} gens)")
    return pd.DataFrame(rows), curves


def aggregate(table, metrics):
    """Aggregate each metric across seeds so the exploration setting is the only
    factor. Returns tidy long frame: group, level, metric, mean, std, sem, count."""
    frames = []
    for m in metrics:
        g = (table.groupby(['group', 'level'])[m]
                   .agg(['mean', 'std', 'sem', 'count']).reset_index())
        g.insert(2, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'group', 'level'])


def ordered_panels(table, panel_order):
    """Panels present in the data, decay shapes first (in panel_order), NO_EPS last."""
    present = set(table['group'].unique())
    panels = [g for g in panel_order if g in present]
    if NO_EPS in present:
        panels.append(NO_EPS)
    return panels


def level_colour_map(table):
    """Stable epsilon_start -> colour map (sorted), shared across every panel/plot."""
    starts = sorted(table.loc[table['group'] != NO_EPS, 'level'].unique())
    return {lvl: EPS_COLOURS[i % len(EPS_COLOURS)] for i, lvl in enumerate(starts)}


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


def plot_final(table, agg, metric, panels, out_dir, window):
    """Endpoint-vs-epsilon_start, one panel per decay shape + a no-epsilon panel.
    mean +/- std across seeds with the individual seed points overlaid. Panels
    share the y-axis so shapes are directly comparable. This is the "mean fitness
    of the last N generations, with std across seeds" plot."""
    label, colour, _fmt = METRICS[metric]
    print(f"  Plotting final {metric} vs epsilon_start ...")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        sub = agg[(agg['group'] == panel) & (agg['metric'] == metric)].sort_values('level')
        tsub = table[table['group'] == panel]
        if sub['mean'].isna().all():
            ax.set_title(f'{panel}\n(no data)', color='gray'); ax.axis('off'); continue

        if panel == NO_EPS:
            x = np.zeros(len(sub))
            ax.errorbar(x, sub['mean'], yerr=sub['std'].fillna(0), fmt='-o',
                        color=colour, ecolor='#6B7280', elinewidth=2, capsize=6,
                        lw=2, markersize=9, zorder=3, label='mean ± std (seeds)')
            for seed, s in tsub.groupby('seed'):
                ax.scatter(np.zeros(len(s)), s[metric], s=45, alpha=0.7,
                           zorder=4, label=f'seed {seed}')
            ax.set_xticks([0]); ax.set_xticklabels(['no epsilon'])
            ax.set_xlim(-0.6, 0.6)
            ax.set_xlabel('')
            title = 'no epsilon (baseline)'
        else:
            ax.errorbar(sub['level'], sub['mean'], yerr=sub['std'].fillna(0),
                        fmt='-o', color=colour, ecolor='#6B7280', elinewidth=2,
                        capsize=6, lw=2, markersize=9, zorder=3, label='mean ± std (seeds)')
            for seed, s in tsub.groupby('seed'):
                ax.scatter(s['level'], s[metric], s=45, alpha=0.7,
                           zorder=4, label=f'seed {seed}')
            starts = sorted(tsub['level'].unique())
            ax.set_xticks(starts)
            pad = (max(starts) - min(starts)) * 0.15 or 0.1
            ax.set_xlim(min(starts) - pad, max(starts) + pad)
            ax.set_xlabel('epsilon start')
            title = f'{panel} decay'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axs[0].set_ylabel(f'Final {label}\n(mean of last {window} gens)')
    # One shared legend from the first drawn panel.
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=8)
    plt.suptitle(f'Epsilon sweep — final {label} (mean ± std over seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'epsilon_sweep_final_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(curves, table, metric, panels, smooth, out_dir):
    """Over-generation curves, one panel per decay shape + a no-epsilon panel.
    Within each epsilon panel: one line per epsilon_start (mean across seeds,
    shaded +/-1 std). The no-epsilon panel is a single mean +/- std band. Panels
    share both axes so the shapes are directly comparable."""
    label, _colour, _fmt = METRICS[metric]
    lvl_colours = level_colour_map(table)
    print(f"  Plotting {metric} curves ...")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        levels = sorted({lvl for (g, lvl, _s) in curves if g == panel})
        plotted = False
        for lvl in levels:
            series = []
            for (g, l, _seed), df in curves.items():
                if g != panel or l != lvl or metric not in df.columns:
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
            if panel == NO_EPS:
                col, lbl = NO_EPS_COLOUR, 'no epsilon'
            else:
                col, lbl = lvl_colours.get(lvl, '#6B7280'), f'eps start={lvl:g}'
            gens = mean.index.values
            ax.plot(gens, mean.values, color=col, lw=2, label=lbl)
            ax.fill_between(gens, mean.values - std.values, mean.values + std.values,
                            color=col, alpha=0.15)
            plotted = True
        ax.set_title('no epsilon (baseline)' if panel == NO_EPS else f'{panel} decay',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=9, title=None if panel == NO_EPS else 'epsilon start')
    axs[0].set_ylabel(label)
    plt.suptitle(f'Epsilon sweep — {label} over generations (mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'epsilon_sweep_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='Aggregate the epsilon+no-epsilon sweep.')
    ap.add_argument('--logs-root', default='logs/learning/standard/auto-run',
                    help='Folder containing the eps<..>_<shape>_seed<..> and '
                         'noeps_seed<..> run dirs.')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='generations.csv columns to plot (default: all six).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Average each metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curve plots (1 = no smoothing).')
    ap.add_argument('--panel-order', nargs='+',
                    default=['linear', 'quadratic', 'sublinear'],
                    help='Left-to-right order of the decay-shape panels.')
    ap.add_argument('--out', default='output/epsilon_sweep')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No runs with params.json + generations.csv under "
                         f"{args.logs_root}. Has the sweep finished?")
    n_eps = sum(1 for r in runs if r['group'] != NO_EPS)
    n_off = sum(1 for r in runs if r['group'] == NO_EPS)
    print(f"Found {len(runs)} runs ({n_eps} epsilon-on, {n_off} no-epsilon). "
          f"Reading {', '.join(args.metrics)} (last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.metrics, args.final_window, args.max_gen)
    agg = aggregate(table, args.metrics)
    panels = ordered_panels(table, args.panel_order)

    print("\n-- Aggregated (across seeds) --")
    for m in args.metrics:
        label = METRICS[m][0]
        sub = agg[agg['metric'] == m][['group', 'level', 'mean', 'std', 'sem', 'count']]
        print(f"\n{label} ({m}):")
        print(sub.to_string(index=False,
              formatters={'level': '{:g}'.format, 'mean': '{:.4g}'.format,
                          'std': '{:.4g}'.format, 'sem': '{:.4g}'.format}))

    # Tidy summary CSV. Blank the level for the no-epsilon baseline (it has none).
    out_summary = agg.copy()
    out_summary.loc[out_summary['group'] == NO_EPS, 'level'] = np.nan
    out_summary = out_summary.rename(columns={'group': 'condition', 'level': 'epsilon_start'})
    summary_path = os.path.join(args.out, 'epsilon_sweep_summary.csv')
    out_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    for m in args.metrics:
        if m not in NO_FINAL_METRICS:
            plot_final(table, agg, m, panels, args.out, args.final_window)
        plot_curves(curves, table, m, panels, args.smooth, args.out)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
