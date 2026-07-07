"""
analyse_lr_sweep_behaviour.py
======================================================================
Behaviour-DIAGNOSTIC companion to analyse_lr_sweep_learning.py.

analyse_lr_sweep.py picks the best LR on FITNESS; analyse_lr_sweep_learning.py
describes HOW the policies learn (weight drift, magnitude, population). This
script runs the SAME discovery over the SAME sweep runs, but reads the
per-organism BEHAVIOUR recorded in organisms.csv, to see how learning rate
changes what the organisms actually DO:

    what they eat     -- food_low / food_medium / food_prestige / food_default
                         (counts of each food tier consumed per lifetime)
    where they hide   -- cave_entries (+ day/night split)
    how they die      -- death_cause fractions: starved / drained / lifespan /
                         survived
    how far they roam -- cells_visited (distinct cells visited = distance proxy)
    predator contact  -- predator_touches (distinct attachment episodes)

Like the learning script it uses ALL runs (LRs x seeds), collapses each run to a
final value (organisms recorded over the last --final-window generations, so we
describe the CONVERGED behaviour, not the early exploration), then aggregates
across seeds so learning rate is the only factor.

organisms.csv logs the top-ranked organisms per generation, so these are the
behaviours of the FITTEST organisms at each LR — i.e. what the successful
strategy looks like, not the population average.

These are DESCRIPTIVE diagnostics (higher isn't "better"); the script shows the
level + spread and the compositional make-up, it does NOT pick a winner.

Outputs (into --out):
    lr_sweep_behaviour_final.png    -- 1xN endpoint-vs-LR panels (mean +/- std over seeds)
    lr_sweep_behaviour_death.png    -- stacked death-cause composition per LR
    lr_sweep_behaviour_food.png     -- stacked food-tier composition per LR
    lr_sweep_behaviour_curves.png   -- Nx1 over-generation panels (one line per LR)
    lr_sweep_behaviour_summary.csv  -- tidy (lr, metric, mean, std, sem, count)

Usage
-----
    python analyse_lr_sweep_behaviour.py \
        --logs-root logs/learning/standard/auto-run \
        --final-window 20 \
        --out output/lr_sweep_behaviour

    # restrict / reorder the scalar panels:
    python analyse_lr_sweep_behaviour.py --metrics cells_visited cave_entries
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
# analyse_lr_sweep.py / analyse_lr_sweep_learning.py so the curve plots read the same.
LR_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

# ── Scalar per-organism behaviours ────────────────────────────────────────────
# Column -> (axis/panel label, accent colour, per-cell value format). These are
# averaged over the organisms in the final window, per run, then over seeds.
METRICS = {
    'cells_visited':    ('Distance travelled (cells visited)', '#0891B2', '{:.0f}'),
    'cave_entries':     ('Cave entries',                       '#7C3AED', '{:.2f}'),
    'predator_touches': ('Predator touches',                   '#DC2626', '{:.2f}'),
    'food_total':       ('Food eaten (total)',                 '#059669', '{:.1f}'),
}
DEFAULT_METRICS = list(METRICS)

# ── Compositional behaviours ──────────────────────────────────────────────────
# death_cause categories (fixed order + colour) for the stacked-composition plot.
DEATH_CAUSES = {
    'starved':  '#D97706',   # ran out of energy
    'drained':  '#DC2626',   # killed by a predator
    'lifespan': '#2563EB',   # hit the max-lifetime cap
    'survived': '#059669',   # still alive at generation end
}
# food tiers, mapped from the organisms.csv columns.
FOOD_TIERS = {
    'food_default':  ('default',  '#9CA3AF'),
    'food_low':      ('low',      '#60A5FA'),
    'food_medium':   ('medium',   '#F59E0B'),
    'food_prestige': ('prestige', '#DB2777'),
}
FOOD_COLS = list(FOOD_TIERS)

# All organisms.csv columns we read (kept minimal — organisms.csv can be large).
USE_COLS = ['generation', 'death_cause', 'cells_visited', 'cave_entries',
            'cave_entries_day', 'cave_entries_night', 'predator_touches'] + FOOD_COLS


def discover_runs(logs_root):
    """Find every run folder under logs_root with params.json + organisms.csv,
    returning a list of dicts: {lr, seed, dir, org_csv}.

    params.json is authoritative for lr/seed (folder names are only a hint)."""
    runs = []
    for params_path in glob.glob(os.path.join(logs_root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        org_csv = os.path.join(run_dir, 'organisms.csv')
        if not os.path.exists(org_csv):
            print(f"  skip {os.path.basename(run_dir)} — no organisms.csv")
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
            lr = float(p['learning_rate'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'lr': lr, 'seed': seed, 'dir': run_dir, 'org_csv': org_csv})
    return runs


def load_run(org_csv, window, max_gen):
    """Read one run's organisms.csv once; return (finals, per_gen_curve).

    finals: dict of behavioural summaries over the last `window` generations —
        scalar means (cells_visited, cave_entries, predator_touches, food_total),
        food_<tier>_frac (share of consumed food in each tier), and
        death_<cause>_frac (share of organisms with each death cause).
    per_gen_curve: DataFrame indexed by generation with the per-generation MEAN
        of each scalar metric, for the over-generation curve plot.

    The window is the last `window` recorded generations (organisms.csv logs the
    fittest organisms per generation), so finals describe the CONVERGED behaviour.
    Missing columns degrade to NaN/0 rather than crashing on older logs."""
    # Only load columns that exist, so a slightly older organisms.csv still reads.
    available = pd.read_csv(org_csv, nrows=0).columns
    cols = [c for c in USE_COLS if c in available]
    df = pd.read_csv(org_csv, usecols=cols)
    for c in USE_COLS:
        if c not in df.columns:
            df[c] = np.nan if c not in ('death_cause',) else 'unknown'

    df = df[df['generation'] <= max_gen].copy()
    if not len(df):
        return _empty_finals(), pd.DataFrame()

    # food_total = sum of all tiers eaten this lifetime.
    df['food_total'] = df[FOOD_COLS].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    # Per-generation curve: mean of each scalar across that generation's organisms.
    curve = (df.groupby('generation')[list(METRICS)]
               .mean().sort_index())

    # Restrict to the final window for the endpoint summaries.
    cutoff = df['generation'].max() - window
    win = df[df['generation'] > cutoff]
    if not len(win):
        win = df

    finals = {}
    for m in METRICS:
        finals[m] = pd.to_numeric(win[m], errors='coerce').mean()

    # Cave day/night split (extra scalars kept for the summary CSV).
    finals['cave_entries_day']   = pd.to_numeric(win['cave_entries_day'], errors='coerce').mean()
    finals['cave_entries_night'] = pd.to_numeric(win['cave_entries_night'], errors='coerce').mean()

    # Food composition: share of total consumed food in each tier.
    food_sums = {c: pd.to_numeric(win[c], errors='coerce').sum() for c in FOOD_COLS}
    grand = sum(v for v in food_sums.values()) or np.nan
    for c in FOOD_COLS:
        finals[f'{c}_frac'] = food_sums[c] / grand

    # Death-cause composition: share of organisms with each cause.
    dc = win['death_cause'].astype(str).value_counts(normalize=True)
    for cause in DEATH_CAUSES:
        finals[f'death_{cause}_frac'] = float(dc.get(cause, 0.0))

    return finals, curve


def _empty_finals():
    finals = {m: np.nan for m in METRICS}
    finals['cave_entries_day'] = np.nan
    finals['cave_entries_night'] = np.nan
    for c in FOOD_COLS:
        finals[f'{c}_frac'] = np.nan
    for cause in DEATH_CAUSES:
        finals[f'death_{cause}_frac'] = np.nan
    return finals


def build_table(runs, window, max_gen):
    """Per-run behavioural summaries -> long dataframe + per-run curve frames."""
    rows, curves = [], {}
    for r in runs:
        finals, curve = load_run(r['org_csv'], window, max_gen)
        row = {'lr': r['lr'], 'seed': r['seed'],
               'n_gens': int(curve.index.max()) if len(curve) else 0}
        row.update(finals)
        rows.append(row)
        curves[(r['lr'], r['seed'])] = curve
        vals = '  '.join(
            f"{m}={finals[m]:.3g}" if not np.isnan(finals[m]) else f"{m}=nan"
            for m in METRICS)
        print(f"  lr={r['lr']:<8g} seed={r['seed']:<5d} {vals}  ({row['n_gens']} gens)")
    return pd.DataFrame(rows), curves


def aggregate(table, cols):
    """Aggregate each column across seeds so LR is the only factor.
    Returns tidy long frame: lr, metric, mean, std, sem, count."""
    frames = []
    for m in cols:
        g = table.groupby('lr')[m].agg(['mean', 'std', 'sem', 'count']).reset_index()
        g.insert(1, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'lr'])


def plot_final(table, agg, metrics, out_dir):
    """Endpoint-vs-LR: one panel per scalar behaviour, mean +/- std across seeds,
    with the individual seed points overlaid. Log x-axis (LRs span decades)."""
    print("  Plotting final behaviour vs learning rate ...")
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
    plt.suptitle('LR sweep — behaviour (final, mean ± std over seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_behaviour_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_composition(table, kind, out_dir):
    """Stacked-bar composition per LR, averaged over seeds. kind='death' uses the
    death_<cause>_frac columns; kind='food' uses the food_<tier>_frac columns."""
    if kind == 'death':
        spec = [(f'death_{c}_frac', c, col) for c, col in DEATH_CAUSES.items()]
        title = 'Death-cause composition vs learning rate'
        ylabel = 'Fraction of organisms'
        fname = 'lr_sweep_behaviour_death.png'
    else:
        spec = [(f'{c}_frac', FOOD_TIERS[c][0], FOOD_TIERS[c][1]) for c in FOOD_COLS]
        title = 'Food-tier composition vs learning rate'
        ylabel = 'Fraction of food eaten'
        fname = 'lr_sweep_behaviour_food.png'

    print(f"  Plotting {kind} composition ...")
    means = table.groupby('lr')[[s[0] for s in spec]].mean()
    lrs = sorted(means.index)
    x = np.arange(len(lrs))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(lrs) + 3), 5))
    bottom = np.zeros(len(lrs))
    for col, lbl, colour in spec:
        vals = means.loc[lrs, col].fillna(0).values
        ax.bar(x, vals, bottom=bottom, color=colour, label=lbl, edgecolor='white', width=0.6)
        # Annotate non-trivial segments.
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax.text(xi, b + v / 2, f'{v:.0%}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:g}' for lr in lrs])
    ax.set_xlabel('Learning rate')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, title=kind, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(curves, metrics, smooth, out_dir):
    """Over-generation panels (one per scalar behaviour): one line per LR (mean
    across seeds), shaded +/-1 std band. Shows how behaviour develops, not just
    the converged end."""
    print("  Plotting behaviour curves ...")
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
                series.append(pd.to_numeric(df[m], errors='coerce'))
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
    plt.suptitle('LR sweep — behaviour over generations (mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, 'lr_sweep_behaviour_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='Behaviour-diagnostic view of the LR sweep.')
    ap.add_argument('--logs-root', default='logs/learning/standard/auto-run',
                    help='Folder containing the lr<LR>_seed<SEED> run dirs.')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='organisms.csv scalar behaviours to panel (default: all).')
    ap.add_argument('--final-window', type=int, default=20,
                    help='Summarise organisms over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (align uneven-length runs).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curve plot (1 = no smoothing).')
    ap.add_argument('--out', default='output/lr_sweep_behaviour')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No runs with params.json + organisms.csv under "
                         f"{args.logs_root}. Has the sweep finished?")
    print(f"Found {len(runs)} runs. Reading behaviour "
          f"(last {args.final_window} gens) ...")

    table, curves = build_table(runs, args.final_window, args.max_gen)

    # Columns to aggregate for the summary CSV: scalars + composition fractions.
    frac_cols = ([f'{c}_frac' for c in FOOD_COLS]
                 + [f'death_{c}_frac' for c in DEATH_CAUSES])
    scalar_cols = list(METRICS) + ['cave_entries_day', 'cave_entries_night']
    agg = aggregate(table, scalar_cols + frac_cols)

    print("\n-- Aggregated (across seeds) --")
    for m in args.metrics:
        label = METRICS[m][0]
        sub = agg[agg['metric'] == m][['lr', 'mean', 'std', 'sem', 'count']]
        print(f"\n{label} ({m}):")
        print(sub.to_string(index=False,
              formatters={'lr': '{:g}'.format, 'mean': '{:.4g}'.format,
                          'std': '{:.4g}'.format, 'sem': '{:.4g}'.format}))

    summary_path = os.path.join(args.out, 'lr_sweep_behaviour_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    plot_final(table, agg, args.metrics, args.out)
    plot_composition(table, 'death', args.out)
    plot_composition(table, 'food', args.out)
    plot_curves(curves, args.metrics, args.smooth, args.out)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
