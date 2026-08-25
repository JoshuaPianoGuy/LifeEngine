"""
analyse_lr_vs_evolution.py
======================================================================
FITNESS + POPULATION comparison of the learning-rate sweep against the
EVOLUTION baseline, for the h128 / 1000x1000 runs under logs/.

analyse_lr_sweep.py answers "which learning rate is best?" within the LEARNING
condition only. This script puts the LR sweep and the evolution (GA-only)
baseline on the SAME axes, for the two outcome families that matter:

    fitness     avg_fitness, top20percent_fitness, best_fitness
    population  peak_population, total_agents

so you can read off both "does in-lifetime learning beat pure evolution?" and
"at which learning rate?" from one figure.

Discovery + grouping
--------------------
Every run folder under --logs-root with params.json + generations.csv is read;
params.json is authoritative (folder names are only a hint). Runs are split by
ENVIRONMENT, which is derived from the predator parameters rather than the
folder prefix:

    baseline    roaming_predator_count == 0            (logs/.../base_w1k_*)
    hard        roaming_predator_count  > 0            (logs/.../roam240d5_w1k_*)

and within an environment by GROUP:

    lr=<LR>     condition == learning   (one group per learning rate)
    evolution   condition == evolution  (GA only; its params.json still carries a
                                         learning_rate, which is unused, so the
                                         condition — not the LR — defines the group)

Each environment gets its own figures, since evolution and learning are only
comparable within the same predator regime. In the current logs/ tree evolution
was only run in the HARD environment, so the baseline figures show the LR sweep
alone; that is reported, not silently hidden.

Averaging across runs (--agg, default two-stage)
-----------------------------------------------
The sweep has TWO kinds of repeat: independent map seeds (map_seed) and
replicates of the same seed (the _r1.._r5 suffix). Averaging all 15 runs of a
group in one pool would let a seed with more replicates dominate and would blur
"variation between maps" with "variation between repeats of one map". So by
default this script aggregates in two stages:

    1. replicates -> seed curve   (mean over the _rN runs of that map_seed)
    2. seed curves -> group curve (mean +/- std across map_seeds)

The shaded band and the error bars are therefore the spread ACROSS MAP SEEDS,
which is what the write-up claims. --agg pooled skips stage 1 and treats every
run as an independent sample (band = spread across all runs) if you want that
instead. Note the two conditions do not share a seed set (learning: 1/42/999;
evolution: 1/42/123/456/999) — the per-group seed count is printed and stored in
the summary CSVs.

A duplicate-generation guard (drop_duplicates on 'generation') makes it robust
to a doubled/appended CSV, which would otherwise break the axis=1 concat.

Which groups are plotted (--only-learning / --lr)
------------------------------------------------
By default every group of an environment goes on one figure. Two selectors carve
out the sub-comparisons:

    --only-learning     drop the evolution baseline -> the LR sweep on its own.
    --lr 0.02           keep one learning rate -> a two-condition head-to-head
                        against evolution.

With fewer than two learning rates there is no LR axis left to plot against, so
the endpoint figure switches from "metric vs LR (log x)" to one bar per
condition. The curve figure is the same either way. The selectors are recorded
in the output basename (e.g. hard_lr0.02_final.png), so a sweep and a
head-to-head can share one --out without overwriting each other.

Outputs (into --out, one set per environment)
---------------------------------------------
    <tag>_curves.png        -- one panel per metric over generations, one line
                               per group (mean +/- std across seeds).
    <tag>_final.png         -- endpoint per metric: either learning against LR
                               (log x) with evolution as a horizontal reference
                               band, or one bar per condition for a head-to-head.
                               Individual seeds overlaid in both.
    <tag>_curves.csv        -- tidy (env, group, generation, metric, mean, std, n_seeds).
    <tag>_finals.csv        -- tidy (env, group, lr, metric, mean, std, sem, n_seeds).

where <tag> is the environment plus any selector (hard, hard_lr0.02,
baseline_learning_only, hard_common_seeds, ...).

Usage
-----
    # everything on one figure per environment:
    python analyse_lr_vs_evolution.py --logs-root logs --out output/lr_vs_evolution

    # the LR sweeps on their own, one figure per environment:
    python analyse_lr_vs_evolution.py --only-learning --out output/lr_sweep_only

    # best LR head-to-head against the GA baseline in the hard environment:
    python analyse_lr_vs_evolution.py --env hard --lr 0.02 \
        --out output/lr0.02_vs_evolution

    # paired comparison: only the map seeds every group was run on (1/42/999):
    python analyse_lr_vs_evolution.py --env hard --common-seeds \
        --out output/lr_vs_evolution_common
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

# Metric column -> (axis/panel label, family). Family drives the row layout of
# the figures: fitness panels on top, population panels below.
METRICS = {
    'avg_fitness':           ('Average fitness',      'fitness'),
    'top20percent_fitness':  ('Top-20% fitness',      'fitness'),
    'best_fitness':          ('Best fitness',         'fitness'),
    'peak_population':       ('Peak population',      'population'),
    'total_agents':          ('Total agents',         'population'),
}
DEFAULT_METRICS = ['avg_fitness', 'top20percent_fitness', 'peak_population', 'total_agents']

# Distinct colour per learning rate (assigned in sorted-LR order) — matches
# analyse_lr_sweep.py so the LR lines read the same across scripts. Evolution
# gets its own near-black so the baseline never looks like "another LR".
LR_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']
EVOLUTION_COLOUR = '#111827'
EVOLUTION_GROUP = 'evolution'

ENV_LABELS = {
    'baseline': 'Baseline environment (no predators)',
    'hard': 'Hard environment (roaming predators)',
}


def discover_runs(logs_root, hidden_size, grid_cols):
    """Find every run folder under logs_root (searched recursively) that has both
    params.json and generations.csv.

    Returns a list of dicts: {env, group, lr, seed, replicate, dir, gen_csv}.
    Runs whose hidden_size / grid_cols do not match the requested configuration
    are skipped, so an older sweep sitting in the same tree cannot contaminate
    the averages."""
    runs, skipped = [], {'no_csv': 0, 'bad_params': 0, 'wrong_config': 0, 'other_cond': 0}
    pattern = os.path.join(logs_root, '**', 'params.json')
    for params_path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(params_path)
        gen_csv = os.path.join(run_dir, 'generations.csv')
        if not os.path.exists(gen_csv):
            skipped['no_csv'] += 1
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
            condition = str(p['condition'])
            seed = int(p['map_seed'])
            lr = float(p['learning_rate'])
            roaming = int(p.get('roaming_predator_count', 0))
            hs = int(p['hidden_size'])
            cols = int(p['grid_cols'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            skipped['bad_params'] += 1
            continue

        if (hidden_size is not None and hs != hidden_size) or \
           (grid_cols is not None and cols != grid_cols):
            skipped['wrong_config'] += 1
            continue

        if condition == 'learning':
            group = f'lr={lr:g}'
        elif condition == 'evolution':
            group = EVOLUTION_GROUP
            lr = np.nan          # the GA baseline has no learning rate
        else:
            skipped['other_cond'] += 1
            continue

        # Replicate index from the trailing _rN of the folder name; runs without
        # one are treated as replicate 1 (a single unreplicated run).
        name = os.path.basename(run_dir)
        replicate = 1
        if '_r' in name and name.rsplit('_r', 1)[1].isdigit():
            replicate = int(name.rsplit('_r', 1)[1])

        runs.append({'env': 'baseline' if roaming == 0 else 'hard',
                     'group': group, 'lr': lr, 'seed': seed, 'replicate': replicate,
                     'dir': run_dir, 'gen_csv': gen_csv, 'name': name})

    for reason, n in skipped.items():
        if n:
            print(f"  ({n} run(s) skipped: {reason})")
    return runs


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, curve_frame).

    finals[m] = mean of metric m over the last `window` recorded generations.
    curve_frame is indexed by generation and holds the requested metrics.
    Deduplicates on 'generation' (keep last) so a doubled/appended CSV neither
    double-counts the tail nor misaligns the across-run concat."""
    df = pd.read_csv(gen_csv)
    df = df[df['generation'] <= max_gen].copy()
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    df = df.set_index('generation')

    curve = pd.DataFrame(index=df.index)
    finals = {}
    for m in metrics:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors='coerce')
            curve[m] = col
            tail = col.dropna().tail(window)
            finals[m] = tail.mean() if len(tail) else np.nan
        else:
            curve[m] = np.nan
            finals[m] = np.nan
    return finals, curve


def build_tables(runs, metrics, window, max_gen):
    """Read every run once. Returns (finals_table, curves) where finals_table has
    one row per run and curves maps run index -> per-generation frame."""
    rows, curves = [], {}
    for i, r in enumerate(runs):
        finals, curve = load_run(r['gen_csv'], metrics, window, max_gen)
        row = {k: r[k] for k in ('env', 'group', 'lr', 'seed', 'replicate', 'name')}
        row['n_gens'] = int(curve.index.max()) if len(curve) else 0
        row.update(finals)
        rows.append(row)
        curves[i] = curve
    return pd.DataFrame(rows), curves


def seed_curves(runs, curves, indices, metric, pooled):
    """Collapse a group's runs to one column per SEED (mean over that seed's
    replicates), or one column per RUN when pooled=True.

    Returns a wide frame: index = generation, columns = seed (or run name)."""
    per_run = {}
    for i in indices:
        col = curves[i][metric]
        key = runs[i]['name'] if pooled else runs[i]['seed']
        per_run.setdefault(key, []).append(col)
    out = {}
    for key, cols in per_run.items():
        wide = pd.concat(cols, axis=1).sort_index()
        out[key] = wide.mean(axis=1)          # stage 1: replicates -> seed
    return pd.DataFrame(out).sort_index()


def aggregate_curves(runs, curves, table, metrics, pooled):
    """Per environment/group/metric: mean +/- std over seed curves (stage 2).

    Returns tidy long frame: env, group, generation, metric, mean, std, n_seeds."""
    frames = []
    for (env, group), sub in table.groupby(['env', 'group']):
        for m in metrics:
            wide = seed_curves(runs, curves, list(sub.index), m, pooled)
            if wide.empty or wide.isna().all().all():
                continue
            frames.append(pd.DataFrame({
                'env': env, 'group': group, 'generation': wide.index,
                'metric': m, 'mean': wide.mean(axis=1).values,
                'std': wide.std(axis=1).fillna(0).values,
                'n_seeds': wide.notna().sum(axis=1).values,
            }))
    if not frames:
        return pd.DataFrame(columns=['env', 'group', 'generation', 'metric',
                                     'mean', 'std', 'n_seeds'])
    return pd.concat(frames, ignore_index=True)


def aggregate_finals(table, metrics, pooled):
    """Per environment/group/metric endpoint: collapse replicates into a seed
    value (unless pooled), then mean/std/sem across seeds."""
    keys = ['env', 'group', 'lr', 'seed'] if not pooled else ['env', 'group', 'lr', 'name']
    # dropna=False on lr: the evolution group's lr is NaN and must survive groupby.
    per_seed = table.groupby(keys, dropna=False)[metrics].mean().reset_index()
    frames = []
    for m in metrics:
        g = (per_seed.groupby(['env', 'group', 'lr'], dropna=False)[m]
             .agg(['mean', 'std', 'sem', 'count']).reset_index()
             .rename(columns={'count': 'n_seeds'}))
        g.insert(3, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True), per_seed


def restrict_to_common_seeds(table):
    """Drop, per environment, every run whose map_seed is not present in ALL of
    that environment's groups.

    The conditions were not run on the same seed set (learning: 1/42/999;
    evolution: 1/42/123/456/999), so by default a group difference partly
    reflects a map difference. This makes the comparison paired — every group is
    averaged over exactly the same maps — at the cost of the extra evolution
    seeds."""
    keep = []
    for env, sub in table.groupby('env'):
        per_group = [set(g['seed']) for _, g in sub.groupby('group')]
        common = set.intersection(*per_group) if per_group else set()
        dropped = sorted(set(sub['seed']) - common)
        if dropped:
            print(f"  {env}: restricting to seeds {sorted(common)} "
                  f"(dropped {dropped} — not run in every group)")
        keep.append(sub[sub['seed'].isin(common)])
    return pd.concat(keep).sort_index()


def group_order(groups):
    """LR groups in ascending LR, evolution last (it is the reference line)."""
    lrs = sorted(g for g in groups if g != EVOLUTION_GROUP)
    ordered = sorted(lrs, key=lambda g: float(g.split('=')[1]))
    if EVOLUTION_GROUP in groups:
        ordered.append(EVOLUTION_GROUP)
    return ordered


def group_style(group, lr_index):
    """Colour/linestyle per group: LRs use the shared palette, evolution is a
    near-black dashed reference so it never reads as 'another learning rate'."""
    if group == EVOLUTION_GROUP:
        return EVOLUTION_COLOUR, '--', 'evolution (GA only)'
    return LR_COLOURS[lr_index % len(LR_COLOURS)], '-', group


def panel_grid(metrics):
    """Lay panels out as fitness row(s) then population row(s)."""
    fam = {}
    for m in metrics:
        fam.setdefault(METRICS[m][1], []).append(m)
    rows = [fam[f] for f in ('fitness', 'population') if f in fam]
    ncols = max(len(r) for r in rows)
    return rows, ncols


def plot_curves(agg, env, metrics, smooth, out_dir, tag, subtitle):
    """Over-generation panels: one line per group (mean across seeds), shaded
    +/-1 std band. Shows trajectory and stability, not just the endpoint."""
    sub_env = agg[agg['env'] == env]
    groups = group_order(sub_env['group'].unique())
    rows, ncols = panel_grid(metrics)

    fig, axs = plt.subplots(len(rows), ncols, figsize=(7 * ncols, 4.6 * len(rows)),
                            squeeze=False)
    lr_i = 0
    handles = {}
    for group in groups:
        colour, ls, label = group_style(group, lr_i)
        if group != EVOLUTION_GROUP:
            lr_i += 1
        for ri, row_metrics in enumerate(rows):
            for ci in range(ncols):
                ax = axs[ri][ci]
                if ci >= len(row_metrics):
                    ax.axis('off')
                    continue
                m = row_metrics[ci]
                s = sub_env[(sub_env['group'] == group) & (sub_env['metric'] == m)]
                if s.empty:
                    continue
                s = s.sort_values('generation')
                mean = s['mean'].to_numpy(dtype=float)
                std = s['std'].to_numpy(dtype=float)
                if smooth > 1:
                    mean = pd.Series(mean).rolling(smooth, min_periods=1, center=True).mean().to_numpy()
                    std = pd.Series(std).rolling(smooth, min_periods=1, center=True).mean().to_numpy()
                gens = s['generation'].to_numpy()
                line, = ax.plot(gens, mean, color=colour, ls=ls, lw=2, label=label)
                ax.fill_between(gens, mean - std, mean + std, color=colour, alpha=0.15)
                handles[label] = line

    for ri, row_metrics in enumerate(rows):
        for ci in range(ncols):
            if ci >= len(row_metrics):
                continue
            ax = axs[ri][ci]
            m = row_metrics[ci]
            ax.set_title(METRICS[m][0], fontsize=12, fontweight='bold')
            ax.set_ylabel(METRICS[m][0])
            ax.set_xlabel('Generation')
            ax.grid(True, alpha=0.3)
    if handles:
        axs[0][0].legend(handles.values(), handles.keys(), fontsize=9, title='condition')

    plt.suptitle(f'{ENV_LABELS.get(env, env)} — {subtitle}\n'
                 f'fitness & population (mean ± std across map seeds)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{tag}_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_finals(finals, per_seed, env, metrics, window, out_dir, tag, subtitle):
    """Endpoint per metric, in whichever of the two forms fits the selection:

    a sweep (2+ learning rates) is drawn against LR on a log x-axis, with
    evolution as a horizontal reference band; a head-to-head (one learning rate,
    or learning vs evolution only) has no meaningful LR axis, so it is drawn as
    one bar per condition instead. Both show mean ± std across map seeds with
    the per-seed values overlaid."""
    f_env = finals[finals['env'] == env]
    lrs = sorted(f_env[f_env['group'] != EVOLUTION_GROUP]['lr'].dropna().unique())
    if len(lrs) < 2:
        return plot_finals_headtohead(finals, per_seed, env, metrics, window,
                                      out_dir, tag, subtitle)
    return plot_finals_sweep(finals, per_seed, env, metrics, window,
                             out_dir, tag, subtitle)


def plot_finals_sweep(finals, per_seed, env, metrics, window, out_dir, tag, subtitle):
    """Endpoint per metric: learning against LR (log x, mean ± std across seeds,
    seed points overlaid); evolution as a horizontal reference band, since the
    GA baseline has no learning rate to place it at."""
    f_env = finals[finals['env'] == env]
    s_env = per_seed[per_seed['env'] == env]
    lr_rows = f_env[f_env['group'] != EVOLUTION_GROUP]
    lrs = sorted(lr_rows['lr'].dropna().unique())
    rows, ncols = panel_grid(metrics)

    fig, axs = plt.subplots(len(rows), ncols, figsize=(6.5 * ncols, 4.8 * len(rows)),
                            squeeze=False)
    for ri, row_metrics in enumerate(rows):
        for ci in range(ncols):
            ax = axs[ri][ci]
            if ci >= len(row_metrics):
                ax.axis('off')
                continue
            m = row_metrics[ci]
            label = METRICS[m][0]

            lr_sub = lr_rows[lr_rows['metric'] == m].sort_values('lr')
            if not lr_sub.empty:
                ax.errorbar(lr_sub['lr'], lr_sub['mean'], yerr=lr_sub['std'].fillna(0),
                            fmt='-o', color='#2563EB', ecolor='#6B7280', elinewidth=2,
                            capsize=6, lw=2, markersize=9, zorder=3,
                            label='learning (mean ± std)')
                pts = s_env[s_env['group'] != EVOLUTION_GROUP]
                ax.scatter(pts['lr'], pts[m], s=32, color='#2563EB', alpha=0.45,
                           zorder=4, label='per-seed')

            evo = f_env[(f_env['group'] == EVOLUTION_GROUP) & (f_env['metric'] == m)]
            if not evo.empty and lrs:
                mu = float(evo['mean'].iloc[0])
                sd = float(evo['std'].fillna(0).iloc[0])
                ax.axhline(mu, color=EVOLUTION_COLOUR, ls='--', lw=2,
                           label='evolution (GA only)', zorder=2)
                ax.axhspan(mu - sd, mu + sd, color=EVOLUTION_COLOUR, alpha=0.10, zorder=1)

            if lrs:
                ax.set_xscale('log')
                ax.set_xticks(lrs)
                ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel('Learning rate (log scale)')
            ax.set_ylabel(f'Final {label}')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
    axs[0][0].legend(fontsize=8)

    plt.suptitle(f'{ENV_LABELS.get(env, env)} — {subtitle}\n'
                 f'final fitness & population (last {window} generations, '
                 f'mean ± std across map seeds)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{tag}_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_finals_headtohead(finals, per_seed, env, metrics, window, out_dir, tag, subtitle):
    """Endpoint per metric as one mean ± std marker per condition (spread across
    map seeds), with the per-seed values overlaid and the mean ± std annotated.

    Used when there is no LR axis to plot against — i.e. a single learning rate
    compared with the evolution baseline. Drawn as points rather than bars: the
    conditions differ by a few percent on a scale that does not start at zero, so
    bars would either lie about the baseline or flatten the difference."""
    f_env = finals[finals['env'] == env]
    s_env = per_seed[per_seed['env'] == env]
    groups = group_order(f_env['group'].unique())
    rows, ncols = panel_grid(metrics)

    fig, axs = plt.subplots(len(rows), ncols, figsize=(5.5 * ncols, 5.0 * len(rows)),
                            squeeze=False)
    for ri, row_metrics in enumerate(rows):
        for ci in range(ncols):
            ax = axs[ri][ci]
            if ci >= len(row_metrics):
                ax.axis('off')
                continue
            m = row_metrics[ci]
            label = METRICS[m][0]
            xs, labels = np.arange(len(groups)), []
            lr_i = 0
            for x, group in zip(xs, groups):
                colour, _ls, glabel = group_style(group, lr_i)
                if group != EVOLUTION_GROUP:
                    lr_i += 1
                labels.append(glabel)
                row = f_env[(f_env['group'] == group) & (f_env['metric'] == m)]
                if row.empty:
                    continue
                mu = float(row['mean'].iloc[0])
                sd = float(row['std'].fillna(0).iloc[0])
                ax.errorbar(x, mu, yerr=sd, fmt='o', color=colour, ecolor=colour,
                            elinewidth=2.5, capsize=10, markersize=13, zorder=4,
                            label='mean ± std' if x == 0 else None)
                pts = s_env[s_env['group'] == group][m].dropna()
                # Deterministic spread so overlapping seed points stay readable.
                jitter = np.linspace(-0.10, 0.10, len(pts)) if len(pts) > 1 else np.zeros(len(pts))
                ax.scatter(x + 0.17 + jitter, pts, s=42, color=colour, alpha=0.45,
                           zorder=3, label='per-seed' if x == 0 else None)
                ax.annotate(f'{mu:.4g} ± {sd:.3g}', (x, mu), textcoords='offset points',
                            xytext=(-14, 0), ha='right', va='center',
                            fontsize=9, zorder=6)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_xlim(-0.6, len(groups) - 0.4)
            ax.set_ylabel(f'Final {label}')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
    axs[0][0].legend(fontsize=8)

    plt.suptitle(f'{ENV_LABELS.get(env, env)} — {subtitle}\n'
                 f'final fitness & population (last {window} generations, '
                 f'mean ± std across map seeds)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{tag}_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description='Fitness + population: LR sweep vs the evolution baseline.')
    ap.add_argument('--logs-root', default='logs',
                    help='Root searched recursively for run folders (default: logs).')
    ap.add_argument('--env', default='all', choices=['all', 'baseline', 'hard'],
                    help='Which environment(s) to plot (default: all found).')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=list(METRICS),
                    help=f'generations.csv columns to panel (default: {" ".join(DEFAULT_METRICS)}).')
    ap.add_argument('--hidden-size', type=int, default=128,
                    help='Only include runs with this hidden size (0 = any).')
    ap.add_argument('--grid-cols', type=int, default=1000,
                    help='Only include runs with this world width (0 = any).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Average each metric over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curve plot (1 = no smoothing).')
    ap.add_argument('--lr', type=float, default=None,
                    help='Keep only this learning rate (e.g. --lr 0.02). With the '
                         'evolution baseline this gives a two-condition head-to-head, '
                         'drawn as bars rather than against an LR axis.')
    ap.add_argument('--only-learning', action='store_true',
                    help='Drop the evolution baseline: the LR sweep on its own.')
    ap.add_argument('--common-seeds', action='store_true',
                    help='Use only the map seeds present in EVERY group of an '
                         'environment, so learning and evolution are compared on '
                         'the same maps (learning ran 3 seeds, evolution 5).')
    ap.add_argument('--agg', default='two-stage', choices=['two-stage', 'pooled'],
                    help='two-stage: replicates -> seed -> across-seed spread (default). '
                         'pooled: every run is an independent sample.')
    ap.add_argument('--out', default='output/lr_vs_evolution')
    args = ap.parse_args()
    pooled = args.agg == 'pooled'

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root,
                         args.hidden_size or None, args.grid_cols or None)
    if not runs:
        raise SystemExit(f"No learning/evolution runs with params.json + "
                         f"generations.csv under {args.logs_root}.")

    print(f"Found {len(runs)} runs. Reading {', '.join(args.metrics)} "
          f"(final = last {args.final_window} gens) ...")
    table, curves = build_tables(runs, args.metrics, args.final_window, args.max_gen)

    if args.only_learning:
        table = table[table['group'] != EVOLUTION_GROUP]
    if args.lr is not None:
        # Keep the requested LR plus the evolution baseline (which has lr=NaN and
        # is only present if --only-learning did not already drop it).
        keep_lr = np.isclose(table['lr'], args.lr) & (table['group'] != EVOLUTION_GROUP)
        if not keep_lr.any():
            raise SystemExit(f"No learning runs with --lr {args.lr:g}. Available: "
                             f"{sorted({f'{v:g}' for v in table['lr'].dropna()})}")
        table = table[keep_lr | (table['group'] == EVOLUTION_GROUP)]
    if table.empty:
        raise SystemExit("No runs left after --lr / --only-learning filtering.")

    # Figure/CSV basename, so a sweep and a head-to-head written into the same
    # --out never overwrite each other.
    tag_bits = []
    if args.lr is not None:
        tag_bits.append(f'lr{args.lr:g}')
    if args.only_learning:
        tag_bits.append('learning_only')
    if args.common_seeds:
        tag_bits.append('common_seeds')
    tag_suffix = ('_' + '_'.join(tag_bits)) if tag_bits else ''

    if args.only_learning:
        subtitle = f'learning rate {args.lr:g}' if args.lr is not None else 'learning-rate sweep'
    elif args.lr is not None:
        subtitle = f'learning (lr={args.lr:g}) vs evolution'
    else:
        subtitle = 'learning-rate sweep vs evolution'

    if args.common_seeds:
        table = restrict_to_common_seeds(table)

    print("\n-- Runs per environment / group --")
    counts = (table.groupby(['env', 'group'])
              .agg(runs=('name', 'size'), seeds=('seed', 'nunique'),
                   gens=('n_gens', 'max')).reset_index())
    print(counts.to_string(index=False))

    envs = sorted(table['env'].unique()) if args.env == 'all' else [args.env]
    envs = [e for e in envs if e in set(table['env'])]
    if not envs:
        raise SystemExit(f"No runs for --env {args.env}.")

    agg = aggregate_curves(runs, curves, table, args.metrics, pooled)
    finals, per_seed = aggregate_finals(table, args.metrics, pooled)

    for env in envs:
        print(f"\n=== {ENV_LABELS.get(env, env)} ===")
        f_env = finals[finals['env'] == env]
        if EVOLUTION_GROUP not in set(f_env['group']) and not args.only_learning:
            print("  note: no evolution runs in this environment — "
                  "the LR sweep is shown without the GA baseline.")
        for m in args.metrics:
            sub = f_env[f_env['metric'] == m][['group', 'mean', 'std', 'sem', 'n_seeds']]
            print(f"\n{METRICS[m][0]} ({m}), final {args.final_window} gens:")
            print(sub.to_string(index=False,
                  formatters={'mean': '{:.4g}'.format, 'std': '{:.4g}'.format,
                              'sem': '{:.4g}'.format}))

        tag = f'{env}{tag_suffix}'
        curves_csv = os.path.join(args.out, f'{tag}_curves.csv')
        finals_csv = os.path.join(args.out, f'{tag}_finals.csv')
        agg[agg['env'] == env].to_csv(curves_csv, index=False)
        f_env.to_csv(finals_csv, index=False)
        print(f"\n  Saved: {curves_csv}\n  Saved: {finals_csv}")

        plot_curves(agg, env, args.metrics, args.smooth, args.out, tag, subtitle)
        plot_finals(finals, per_seed, env, args.metrics, args.final_window,
                    args.out, tag, subtitle)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
