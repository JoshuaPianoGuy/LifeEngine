"""
analyse_evolution_sweep.py
======================================================================
Aggregate + compare the EVOLUTION disaster-tuning experiment — the two disaster
sweeps run against the pure natural-selection (evolution) condition:

    run_disaster_sweep_array.slurm           (SNAP-BACK: one fixed cull per strike)
        disaster_fraction        in {0.2, 0.3, 0.4, 0.5, 0.6}
        disaster_recovery_rate   = 0            (snaps straight back)
        x 3 map seeds (1, 42, 999)                        -> 15 runs

    run_disaster_recovery_sweep_array.slurm  (GRADUAL: cull tapers back over gens)
        disaster_fraction        in {0.3, 0.4, 0.5, 0.6}
        disaster_recovery_rate   in {0.05, 0.10}   (fraction shed per generation)
        x 3 map seeds (1, 42, 999)                        -> 24 runs

Both families live in the SAME logs-root (logs/evolution/standard/auto-run/) and
are told apart by params.json (disaster_recovery_rate == 0 => snap-back). The
question is how the CULLING SEVERITY (disaster_fraction) and the RECOVERY RATE
affect fitness + population, and — the headline — how well the species RECOVERS
between strikes. So every metric is drawn over generations, and the disaster
strikes are marked so a dip/recovery can be read straight off the curve.

Panels are the RECOVERY condition (the taper policy); lines within a panel are
the disaster_fraction (culling severity):

    [ snap-back ] [ recovery 0.05 ] [ recovery 0.10 ]

REPLICATES (10-runs-per-seed variants): a (recovery, fraction, seed) cell may be
run several independent times (e.g. run_disaster_sweep_roaming_10rep_array.slurm
does 10 replicates/seed). All runs sharing a (recovery, fraction, seed) — whether
from the original 1-run sweep or a 10-rep campaign — are treated as REPLICATES of
that cell and averaged FIRST into one per-seed value/curve. Only THEN are seeds
aggregated (mean +/- std across seeds). So the band is always the between-seed
spread, never inflated by replicate noise, and a single run per cell reduces to
the original behaviour exactly. (Runs are pooled by params.json alone, so point
this at a logs-root holding ONE environment's disaster runs — e.g. the roaming
sweep — not baseline + roaming mixed.)

Each line is the mean across the 3 map seeds (shaded +/-1 std). Because the
strike PRNG is shared but the taper length (fraction / rate) shifts each strike's
cooldown, strike timing drifts a few generations across fractions/seeds — so the
disaster markers are the CLUSTERED median onset across every run in the panel,
drawn as a faint vertical line + a small triangle along the top axis. They show
"a wave of strikes happened around here", not a single exact generation.

This writes, into --out (default output/evolution_sweep):

    evolution_sweep_curves_<metric>.png     (every metric)
        metric over generations, one panel per recovery condition, one line per
        disaster_fraction (mean across seeds, +/-1 std band), disaster markers.
    evolution_sweep_summary.png             (fitness + population metrics)
        2x2 endpoint panels: converged value (mean of the last --final-window
        generations) vs disaster_fraction, one series per recovery condition,
        mean +/- std across seeds with the seed points overlaid. The compact
        "for each parameter, the averaged value" tuning view.
    evolution_sweep_summary.csv             tidy table behind the summary plot.

Metrics (all read straight from generations.csv). "Fitness" is the species'
fitness metric; "population"/"agents" is the organism count — different
measurements, so labelled separately:
    avg_fitness            -- average fitness of the population
    top20percent_fitness   -- average fitness of the top 20%
    peak_population         -- peak organism count in a generation
    total_agents           -- total organisms alive in a generation

Usage
-----
    python analyse_disaster_sweep_evolution.py \
        --logs-root logs/evolution/standard/auto-run \
        --final-window 50 \
        --out output/evolution_sweep

    # restrict / reorder the metrics, or drop the disaster markers:
    python analyse_disaster_sweep_evolution.py --metrics avg_fitness peak_population
    python analyse_disaster_sweep_evolution.py --no-disaster-markers
"""

import argparse
import glob
import json
import os
import re
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

warnings.filterwarnings('ignore')

# Recovery condition -> panel. Snap-back (recovery_rate 0) is its own sentinel so
# it sorts first and never gets mixed onto a numeric recovery-rate axis.
SNAP = 'snap-back'
PANEL_ORDER = [SNAP, '0.05', '0.10']

# Distinct colour per disaster_fraction (assigned in sorted order), shared across
# every panel/plot. Reuses the sweep-script palette so the figures read the same.
FRACTION_COLOURS = ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

# Colour for the disaster markers (neutral so it never fights the data lines).
DISASTER_COLOUR = '#6B7280'

# Metric column -> (axis/panel label, per-cell value format). Fitness blue/green,
# population cyan/brown — kept separate in the labels because fitness and organism
# count measure different things.
METRICS = {
    'avg_fitness':          ('Average fitness',  '{:.3f}'),
    'top20percent_fitness': ('Top-20% fitness',  '{:.3f}'),
    'peak_population':       ('Peak population',  '{:.0f}'),
    'total_agents':         ('Total agents',     '{:.0f}'),
}
DEFAULT_METRICS = list(METRICS)


def recovery_label(rate):
    """Map disaster_recovery_rate -> panel key. 0 (or ~0) is the snap-back family;
    anything positive is a gradual taper labelled by its rate."""
    if rate is None or float(rate) == 0.0:
        return SNAP
    return f'{float(rate):.2f}'


def discover_runs(logs_root):
    """Find every run under logs_root with params.json + generations.csv that is an
    EVOLUTION disaster run (disaster_enabled true). params.json is authoritative
    (folder names are only a hint): the panel is disaster_recovery_rate, the line
    is disaster_fraction, aggregated over map_seed (and, first, over any replicate
    runs sharing a (panel, fraction, seed) cell). Returns list of dicts:
        {panel, fraction, seed, dir, gen_csv}."""
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
            if not bool(p.get('disaster_enabled', False)):
                continue  # not a disaster run — skip quietly
            seed = int(p['map_seed'])
            fraction = float(p['disaster_fraction'])
            panel = recovery_label(p.get('disaster_recovery_rate', 0))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        runs.append({'panel': panel, 'fraction': fraction, 'seed': seed,
                     'dir': run_dir, 'gen_csv': gen_csv})
    return runs


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once; return (finals, dataframe).

    finals[m] = mean of metric m over the last `window` recorded generations
    (<= max_gen), the run's converged value. Deduplicates on 'generation'
    (keep last) so a doubled/appended CSV neither double-counts the tail nor
    breaks the axis=1 concat in the curve plot."""
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


def disaster_onsets(df):
    """Generations on which a disaster STRIKE begins, read from disaster_cull_frac.
    A strike is the leading edge of a nonzero run of the column (frac>0 while the
    previous generation was 0), so a multi-generation taper counts once, at its
    onset. Returns a sorted list of generation numbers."""
    if 'disaster_cull_frac' not in df.columns:
        return []
    d = df.sort_values('generation')
    frac = pd.to_numeric(d['disaster_cull_frac'], errors='coerce').fillna(0).values
    gens = d['generation'].values
    onsets = []
    prev = 0.0
    for g, f in zip(gens, frac):
        if f > 0 and prev == 0:
            onsets.append(int(g))
        prev = f
    return onsets


def cluster_onsets(onsets, gap):
    """Collapse a pooled list of strike-onset generations (from several runs whose
    timing drifts a few gens) into one representative generation per WAVE: sort,
    start a new cluster whenever the gap to the previous onset exceeds `gap`, and
    take the median of each cluster. Returns a sorted list of ints."""
    if not onsets:
        return []
    xs = sorted(onsets)
    clusters, cur = [], [xs[0]]
    for g in xs[1:]:
        if g - cur[-1] > gap:
            clusters.append(cur)
            cur = [g]
        else:
            cur.append(g)
    clusters.append(cur)
    return [int(np.median(c)) for c in clusters]


def average_replicate_curves(dfs, metrics):
    """Average several replicate generations.csv frames (same cell) into one
    per-generation frame: the mean of each metric across replicates, aligned on
    generation. Replicates share terrain + disaster timing, so their generations
    line up; any per-run tail difference degrades to a NaN-skipping mean. Returns
    a DataFrame with a 'generation' column + one column per present metric."""
    per = [df.drop_duplicates('generation', keep='last').set_index('generation')
           for df in dfs if len(df)]
    if not per:
        return pd.DataFrame(columns=['generation'] + list(metrics))
    cols = {}
    for m in metrics:
        series = [d[m] for d in per if m in d.columns]
        if series:
            cols[m] = pd.concat(series, axis=1).mean(axis=1)
    out = pd.DataFrame(cols).sort_index()
    out.index.name = 'generation'
    return out.reset_index()


def build_table(runs, metrics, window, max_gen):
    """Per-CELL final metrics -> long dataframe (one row per (panel, fraction,
    seed)) + replicate-averaged curve frames + pooled strike onsets.

    Runs sharing a (panel, fraction, seed) are REPLICATES of one cell: their final
    metrics are averaged into the per-seed value, their curves into one per-seed
    curve, and their strike onsets pooled. With one run per cell this is a no-op
    (identical to the original per-run behaviour); with N replicates the row is the
    seed's replicate-mean, so the later across-seed aggregation never sees the
    replicate noise. n_reps records how many runs backed each cell."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r['panel'], r['fraction'], r['seed'])].append(r)

    rows, curves, onsets = [], {}, {}
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1], k[2])):
        panel, fraction, seed = key
        grp = groups[key]
        finals_list, dfs, pooled_onsets = [], [], []
        for r in grp:
            finals, df = load_run(r['gen_csv'], metrics, window, max_gen)
            finals_list.append(finals)
            dfs.append(df)
            pooled_onsets.extend(disaster_onsets(df))
        # Replicate-mean of the per-run finals (NaN-skipping); replicate-mean curve.
        finals_mean = {m: np.nanmean([f[m] for f in finals_list]) for m in metrics}
        avg_df = average_replicate_curves(dfs, metrics)
        n_gens = int(avg_df['generation'].max()) if len(avg_df) else 0

        row = {'panel': panel, 'fraction': fraction, 'seed': seed,
               'n_reps': len(grp), 'n_gens': n_gens}
        row.update(finals_mean)
        rows.append(row)
        curves[key] = avg_df
        onsets[key] = pooled_onsets
        vals = '  '.join(f"{m}={finals_mean[m]:.4g}" if not np.isnan(finals_mean[m]) else f"{m}=nan"
                         for m in metrics)
        reps_note = f"{len(grp)} reps" if len(grp) > 1 else "1 run"
        print(f"  {panel:<10} frac={fraction:<4g} seed={seed:<5d} "
              f"{vals}  ({reps_note}, {n_gens} gens)")
    return pd.DataFrame(rows), curves, onsets


def aggregate(table, metrics):
    """Aggregate each metric across seeds so (recovery, fraction) is the only
    factor. `table` has one row per (panel, fraction, seed) — already the
    replicate-mean — so mean/std/sem/count here are strictly across SEEDS (count =
    number of seeds, not runs). Returns tidy long frame: panel, fraction, metric,
    mean, std, sem, count."""
    frames = []
    for m in metrics:
        g = (table.groupby(['panel', 'fraction'])[m]
                   .agg(['mean', 'std', 'sem', 'count']).reset_index())
        g.insert(2, 'metric', m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'panel', 'fraction'])


def ordered_panels(table):
    """Recovery panels present in the data, snap-back first then ascending rate."""
    present = set(table['panel'].unique())
    return [p for p in PANEL_ORDER if p in present]


def fraction_colour_map(table):
    """Stable disaster_fraction -> colour map (sorted), shared across every panel."""
    fracs = sorted(table['fraction'].unique())
    return {f: FRACTION_COLOURS[i % len(FRACTION_COLOURS)] for i, f in enumerate(fracs)}


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


def panel_title(panel):
    return 'snap-back (no recovery)' if panel == SNAP else f'gradual recovery (rate {panel})'


def recovery_short(panel):
    """Compact legend label that keeps the taper rate distinct (0.05 vs 0.10)."""
    return 'snap-back' if panel == SNAP else f'recovery {panel}'


def draw_disaster_markers(ax, gens):
    """Faint vertical line per strike-wave + a small triangle along the top axis,
    so a dip/recovery in the data can be lined up against a strike. x is in data
    coords, the triangle's y is pinned to the top of the axes."""
    if not gens:
        return
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for g in gens:
        ax.axvline(g, color=DISASTER_COLOUR, lw=0.8, alpha=0.22, zorder=1)
    ax.plot(gens, [1.0] * len(gens), marker='v', linestyle='none',
            color=DISASTER_COLOUR, markersize=6, alpha=0.9, clip_on=False,
            transform=trans, zorder=5)


def plot_curves(curves, onsets, table, metric, panels, smooth, gap,
                show_disasters, out_dir):
    """Over-generation curves, one panel per recovery condition. Within each panel:
    one line per disaster_fraction (mean across seeds, shaded +/-1 std), with the
    strike waves marked. Panels share both axes so severities are comparable."""
    label, _fmt = METRICS[metric]
    frac_colours = fraction_colour_map(table)
    print(f"  Plotting {metric} curves ...")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        fracs = sorted({f for (p, f, _s) in curves if p == panel})
        panel_onsets = []
        for frac in fracs:
            series = []
            for (p, f, _seed), df in curves.items():
                if p != panel or f != frac or metric not in df.columns:
                    continue
                series.append(pd.to_numeric(df.set_index('generation')[metric],
                                            errors='coerce'))
                panel_onsets.extend(onsets.get((p, f, _seed), []))
            if not series:
                continue
            wide = pd.concat(series, axis=1).sort_index()
            mean = wide.mean(axis=1)
            std = wide.std(axis=1).fillna(0)
            if smooth > 1:
                mean = mean.rolling(smooth, min_periods=1, center=True).mean()
                std = std.rolling(smooth, min_periods=1, center=True).mean()
            col = frac_colours.get(frac, '#6B7280')
            gens = mean.index.values
            ax.plot(gens, mean.values, color=col, lw=1.8, label=f'cull {frac:g}', zorder=3)
            ax.fill_between(gens, mean.values - std.values, mean.values + std.values,
                            color=col, alpha=0.13, zorder=2)
        if show_disasters:
            draw_disaster_markers(ax, cluster_onsets(panel_onsets, gap))
        ax.set_title(panel_title(panel), fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        handles, _lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, title='disaster cull frac', loc='best')
    axs[0].set_ylabel(label)
    marker_note = '  (▼ = natural-disaster strike wave)' if show_disasters else ''
    plt.suptitle(f'Evolution disaster sweep — {label} over generations '
                 f'(mean ± std across seeds){marker_note}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'evolution_sweep_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(table, agg, metrics, panels, out_dir, window):
    """2x2 endpoint panels (one per metric): converged value (mean of the last N
    gens) vs disaster_fraction, one series per recovery condition, mean +/- std
    across seeds with the seed points overlaid. The compact 'averaged value for
    each parameter' tuning view."""
    print("  Plotting summary (endpoint vs disaster fraction) ...")
    metrics = [m for m in metrics if m in METRICS]
    rows = (len(metrics) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(13, 5.2 * rows), squeeze=False)
    axs = axs.ravel()
    # One colour per recovery panel so the series read consistently.
    panel_colours = {SNAP: '#111827', '0.05': '#2563EB', '0.10': '#DC2626'}
    for ax, metric in zip(axs, metrics):
        label, _fmt = METRICS[metric]
        for panel in panels:
            sub = agg[(agg['panel'] == panel) & (agg['metric'] == metric)].sort_values('fraction')
            if sub.empty or sub['mean'].isna().all():
                continue
            col = panel_colours.get(panel, '#6B7280')
            ax.errorbar(sub['fraction'], sub['mean'], yerr=sub['std'].fillna(0),
                        fmt='-o', color=col, ecolor=col, elinewidth=1.5, capsize=5,
                        lw=2, markersize=8, zorder=3,
                        label=recovery_short(panel))
            tsub = table[table['panel'] == panel]
            ax.scatter(tsub['fraction'], tsub[metric], s=28, color=col,
                       alpha=0.35, zorder=2)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('disaster cull fraction')
        ax.set_ylabel(f'{label}\n(mean of last {window} gens)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, title='recovery')
    for ax in axs[len(metrics):]:
        ax.axis('off')
    plt.suptitle('Evolution disaster sweep — converged metrics vs culling severity '
                 '(mean ± std across seeds)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'evolution_sweep_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-root', default='logs/evolution/standard/auto-run',
                    help='root holding the evo_cull*/ run folders')
    ap.add_argument('--out', default='output/evolution_sweep', help='output directory')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    help='subset/order of metrics to plot')
    ap.add_argument('--final-window', type=int, default=50,
                    help='endpoint = mean of the last N generations (default 50)')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='ignore generations beyond this (default: all)')
    ap.add_argument('--smooth', type=int, default=5,
                    help='rolling-mean window for the curves (1 = raw, default 5)')
    ap.add_argument('--disaster-gap', type=int, default=40,
                    help='onsets within this many gens are one strike wave (default 40)')
    ap.add_argument('--no-disaster-markers', action='store_true',
                    help='omit the natural-disaster strike markers')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics = [m for m in args.metrics if m in METRICS]
    if not metrics:
        raise SystemExit(f"No known metrics in {args.metrics}; choose from {list(METRICS)}")

    print(f"Discovering evolution disaster runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root)
    if not runs:
        raise SystemExit(f"No evolution disaster runs found under {args.logs_root}")
    print(f"Found {len(runs)} runs. Loading ...")

    table, curves, onsets = build_table(runs, metrics, args.final_window, args.max_gen)
    panels = ordered_panels(table)
    print(f"Recovery panels: {panels}")
    print(f"Disaster fractions: {sorted(table['fraction'].unique())}")

    agg = aggregate(table, metrics)
    csv_path = os.path.join(args.out, 'evolution_sweep_summary.csv')
    agg.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    show = not args.no_disaster_markers
    for m in metrics:
        plot_curves(curves, onsets, table, m, panels, args.smooth,
                    args.disaster_gap, show, args.out)
    plot_summary(table, agg, metrics, panels, args.out, args.final_window)
    print("Done.")


if __name__ == '__main__':
    main()
