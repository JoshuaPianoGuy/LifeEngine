"""
analyse_disaster_replicates.py
======================================================================
Plot the AVERAGE FITNESS ACROSS RUNS for the natural-disaster tuning sweep.

Companion to analyse_disaster_sweep_evolution.py. That script answers "how does
the tuning behave across MAP SEEDS" — it averages the replicates of a cell away
first, so its band is strictly the between-seed spread. This script answers the
other half: "how does the tuning behave across RUNS" — every independent run is
its own unit, so the band is the run-to-run stochasticity of the disaster
response, and the individual run curves can be drawn behind the mean.

The runs it reads are the disaster-tuning campaigns, told apart by params.json
(never by folder name):

    disaster_fraction        culling severity   -> one line per value
    disaster_recovery_rate   taper policy       -> one PANEL per value
                             0 => snap-back (population snaps straight back)
    map_seed                 terrain seed       -> pooled

REPLICATE COUNT IS FREE. A (recovery, fraction, seed) cell may hold 1, 3, 5 or
10 independent runs — whatever the campaign that produced it did (the folder
suffix _r<N> / _rep<N> is read only to order them). The script reports the census
it found, and `--reps N` truncates every cell to its first N runs, so a 10-runs-
per-seed campaign can be re-analysed as if it were a 3- or 5-run one and the
figures compared like for like. `--reps` with no replicate suffixes present is
still well defined: runs are ordered by folder name.

Written, into --out (default output/disaster_replicates):

    disaster_replicates_curves_<metric>.png
        metric over generations, one panel per recovery condition, one line per
        disaster_fraction (mean across runs, +/-1 std band), strike waves marked.
    disaster_replicates_spread_<metric>.png
        the same data unpooled: a (recovery x fraction) grid, every individual
        run drawn faint behind its mean. This is where run-to-run noise is read.
    disaster_replicates_summary.png / .csv
        converged value (mean of the last --final-window generations) vs
        disaster_fraction, one series per recovery condition, mean +/- std with
        the individual runs overlaid. The compact tuning view.
    disaster_replicates_runs.csv
        the per-run endpoint table behind everything above (one row per run).

Metrics are read straight from generations.csv. "Fitness" is the species' fitness
metric; "population"/"agents" is the organism count — different measurements, so
labelled separately:
    avg_fitness            -- average fitness of the population   (default)
    top20percent_fitness   -- average fitness of the top 20%
    best_fitness           -- best fitness in the generation
    peak_population        -- peak organism count in a generation
    total_agents           -- total organisms alive in a generation

Because the strike PRNG is shared but the taper length (fraction / rate) shifts
each strike's cooldown, strike timing drifts a few generations across
fractions/seeds — so the disaster markers are the CLUSTERED median onset across
every run in the panel, drawn as a faint vertical line + a small triangle along
the top axis. They mean "a wave of strikes happened around here", not one exact
generation.

Usage
-----
    python analyse_disaster_replicates.py --logs-root logs/evolution

    # re-analyse a 10-runs-per-seed campaign as a 3- and a 5-run one:
    python analyse_disaster_replicates.py --reps 3 --out output/disaster_reps3
    python analyse_disaster_replicates.py --reps 5 --out output/disaster_reps5

    # more metrics; band = between-seed spread instead of between-run:
    python analyse_disaster_replicates.py --metrics avg_fitness peak_population
    python analyse_disaster_replicates.py --aggregate seeds
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
# it sorts first and never lands on a numeric recovery-rate axis.
SNAP = 'snap-back'
PANEL_ORDER = [SNAP, '0.05', '0.10']

# Distinct colour per disaster_fraction, assigned in sorted (ascending severity)
# order and shared across every panel/plot. Same hues as the sibling sweep
# scripts, ORDERED so no two adjacent severities collide: the amber/red pair is
# split apart (they sit at dE 14.4 for normal vision, below the readable floor),
# which puts every adjacent pair above the normal-vision and colour-vision-
# deficiency separation floors.
FRACTION_COLOURS = ['#2563EB', '#D97706', '#059669', '#DC2626', '#7C3AED', '#0891B2']

# One colour per recovery panel for the summary plot (first three of the same
# validated order, so the two figures never disagree about a hue).
PANEL_COLOURS = {SNAP: '#2563EB', '0.05': '#D97706', '0.10': '#059669'}

# Neutral, so the strike markers never fight the data lines.
DISASTER_COLOUR = '#6B7280'

# Recessive ink for axes/labels — text never wears a series colour.
INK = '#374151'

# Metric column -> (axis/panel label, per-cell value format).
METRICS = {
    'avg_fitness':          ('Average fitness', '{:.3f}'),
    'top20percent_fitness': ('Top-20% fitness', '{:.3f}'),
    'best_fitness':         ('Best fitness',    '{:.3f}'),
    'peak_population':      ('Peak population', '{:.0f}'),
    'total_agents':         ('Total agents',    '{:.0f}'),
}
DEFAULT_METRICS = ['avg_fitness']


def recovery_label(rate):
    """disaster_recovery_rate -> panel key. 0 (or ~0) is the snap-back family;
    anything positive is a gradual taper labelled by its rate."""
    if rate is None or float(rate) == 0.0:
        return SNAP
    return f'{float(rate):.2f}'


def replicate_index(run_dir):
    """Replicate number parsed off the folder name (…_r7 / …_rep7). Used only to
    ORDER the runs of a cell so `--reps N` takes a stable "first N". Folders with
    no replicate suffix sort first (index 0) and then by name."""
    m = re.search(r'_r(?:ep)?(\d+)$', os.path.basename(run_dir))
    return int(m.group(1)) if m else 0


def discover_runs(logs_root, seeds=None):
    """Every run under logs_root with params.json + generations.csv that is a
    disaster run (disaster_enabled true). params.json is authoritative — folder
    names are only a hint — so point this at a logs-root holding ONE
    environment's disaster runs (e.g. the roaming sweep), not baseline + roaming
    mixed. Returns a list of dicts: {panel, fraction, seed, rep, name, gen_csv}."""
    runs = []
    for params_path in sorted(glob.glob(os.path.join(logs_root, '**', 'params.json'),
                                        recursive=True)):
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
        if seeds and seed not in seeds:
            continue
        runs.append({'panel': panel, 'fraction': fraction, 'seed': seed,
                     'rep': replicate_index(run_dir),
                     'name': os.path.basename(run_dir), 'gen_csv': gen_csv})
    return runs


def limit_replicates(runs, n_reps):
    """Keep only the first `n_reps` runs of every (panel, fraction, seed) cell,
    ordered by replicate index then folder name. This is what lets a 10-runs-per-
    seed campaign be re-read as a 3- or 5-run one. n_reps None/0 keeps everything."""
    if not n_reps:
        return runs
    cells = defaultdict(list)
    for r in runs:
        cells[(r['panel'], r['fraction'], r['seed'])].append(r)
    kept = []
    for grp in cells.values():
        grp.sort(key=lambda r: (r['rep'], r['name']))
        kept.extend(grp[:n_reps])
    return kept


def replicate_census(runs):
    """{runs-per-cell: how many cells have that many} — printed so an uneven or
    truncated campaign is visible before any figure is drawn."""
    counts = defaultdict(int)
    for r in runs:
        counts[(r['panel'], r['fraction'], r['seed'])] += 1
    census = defaultdict(int)
    for n in counts.values():
        census[n] += 1
    return dict(sorted(census.items())), len(counts)


def load_run(gen_csv, metrics, window, max_gen):
    """Read one run's generations.csv once -> (finals, curve frame).

    finals[m] is the mean of metric m over the last `window` recorded generations
    (<= max_gen) — the run's converged value. Rows are deduplicated on
    'generation' (keep last) because Logger auto-save APPENDS, so a resumed or
    re-run folder can hold doubled rows that would otherwise double-count the
    tail and break the concat in the curve plots."""
    keep = ['generation', 'disaster_cull_frac'] + list(metrics)
    df = pd.read_csv(gen_csv)
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df['generation'] <= max_gen]
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    finals = {}
    for m in metrics:
        if m in df.columns:
            tail = pd.to_numeric(df[m], errors='coerce').dropna().tail(window)
            finals[m] = tail.mean() if len(tail) else np.nan
        else:
            finals[m] = np.nan
    return finals, df


def disaster_onsets(df):
    """Generations on which a disaster STRIKE begins, read from disaster_cull_frac:
    the leading edge of a nonzero run of the column (frac>0 while the previous
    generation was 0), so a multi-generation taper counts once, at its onset."""
    if 'disaster_cull_frac' not in df.columns:
        return []
    frac = pd.to_numeric(df['disaster_cull_frac'], errors='coerce').fillna(0).values
    gens = df['generation'].values
    onsets, prev = [], 0.0
    for g, f in zip(gens, frac):
        if f > 0 and prev == 0:
            onsets.append(int(g))
        prev = f
    return onsets


def cluster_onsets(onsets, gap):
    """Collapse pooled strike onsets (from runs whose timing drifts a few gens)
    into one representative generation per WAVE: sort, start a new cluster when
    the gap to the previous onset exceeds `gap`, take each cluster's median."""
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


def load_all(runs, metrics, window, max_gen):
    """Load every run once -> (per-run endpoint table, {run key: metric series},
    {run key: strike onsets}). The run key is (panel, fraction, seed, rep, name)
    so a run is addressable and nothing is silently merged."""
    rows, curves, onsets = [], {}, {}
    for r in sorted(runs, key=lambda r: (str(r['panel']), r['fraction'], r['seed'], r['rep'])):
        finals, df = load_run(r['gen_csv'], metrics, window, max_gen)
        key = (r['panel'], r['fraction'], r['seed'], r['rep'], r['name'])
        curves[key] = {m: pd.to_numeric(df.set_index('generation')[m], errors='coerce')
                       for m in metrics if m in df.columns}
        onsets[key] = disaster_onsets(df)
        row = {'panel': r['panel'], 'fraction': r['fraction'], 'seed': r['seed'],
               'rep': r['rep'], 'run': r['name'],
               'n_gens': int(df['generation'].max()) if len(df) else 0}
        row.update(finals)
        rows.append(row)
    return pd.DataFrame(rows), curves, onsets


def unit_curves(curves, metric, mode):
    """Regroup the per-run curves into the UNITS the band is measured over.

    mode 'runs'  -> one unit per run; the +/-1 std band is the run-to-run spread.
    mode 'seeds' -> replicates are averaged into one curve per seed first, so the
                    band is the between-seed spread (matches
                    analyse_disaster_sweep_evolution.py).

    Returns {(panel, fraction): [series, ...]}."""
    by_cell = defaultdict(list)
    for (panel, frac, seed, _rep, _name), per_metric in curves.items():
        if metric in per_metric:
            by_cell[(panel, frac, seed)].append(per_metric[metric])
    out = defaultdict(list)
    for (panel, frac, seed), series in by_cell.items():
        if mode == 'seeds':
            out[(panel, frac)].append(pd.concat(series, axis=1).mean(axis=1))
        else:
            out[(panel, frac)].extend(series)
    return out


def unit_table(table, metrics, mode):
    """The endpoint table at the aggregation unit: unchanged for 'runs', or
    replicate-averaged to one row per (panel, fraction, seed) for 'seeds'."""
    if mode == 'runs':
        return table
    return (table.groupby(['panel', 'fraction', 'seed'], as_index=False)[list(metrics)]
                 .mean())


def aggregate(units, metrics, mode):
    """Aggregate each metric over the units so (recovery, fraction) is the only
    factor left. Returns a tidy frame: panel, fraction, metric, mean, std, sem,
    count (count = number of units, i.e. runs or seeds depending on `mode`)."""
    frames = []
    for m in metrics:
        g = (units.groupby(['panel', 'fraction'])[m]
                  .agg(['mean', 'std', 'sem', 'count']).reset_index())
        g.insert(2, 'metric', m)
        g['unit'] = mode
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'panel', 'fraction'])


def ordered_panels(table):
    """Recovery panels present, snap-back first then ascending taper rate."""
    present = set(table['panel'].unique())
    known = [p for p in PANEL_ORDER if p in present]
    return known + sorted(present - set(known))


def fraction_colour_map(table):
    """Stable disaster_fraction -> colour (ascending severity), shared everywhere."""
    fracs = sorted(table['fraction'].unique())
    return {f: FRACTION_COLOURS[i % len(FRACTION_COLOURS)] for i, f in enumerate(fracs)}


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


def panel_title(panel):
    return 'snap-back (no recovery)' if panel == SNAP else f'gradual recovery (rate {panel})'


def recovery_short(panel):
    return 'snap-back' if panel == SNAP else f'recovery {panel}'


def band_label(mode):
    return 'runs' if mode == 'runs' else 'seeds'


def style_axes(ax):
    """Recessive grid + axes so the data lines carry the figure."""
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#D1D5DB')
    ax.tick_params(colors=INK, labelsize=9)


def draw_disaster_markers(ax, gens):
    """A faint vertical line per strike-wave plus a small triangle along the top
    axis, so a dip/recovery in the data can be lined up against a strike. x is in
    data coords; the triangle's y is pinned to the top of the axes."""
    if not gens:
        return
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for g in gens:
        ax.axvline(g, color=DISASTER_COLOUR, lw=0.8, alpha=0.22, zorder=1)
    ax.plot(gens, [1.0] * len(gens), marker='v', linestyle='none',
            color=DISASTER_COLOUR, markersize=6, alpha=0.9, clip_on=False,
            transform=trans, zorder=5)


def mean_band(series, smooth):
    """Mean +/- 1 std over a list of generation-indexed series, aligned on
    generation and optionally smoothed. Returns (generations, mean, std)."""
    wide = pd.concat(series, axis=1).sort_index()
    mean = wide.mean(axis=1)
    std = wide.std(axis=1).fillna(0)
    if smooth > 1:
        mean = mean.rolling(smooth, min_periods=1, center=True).mean()
        std = std.rolling(smooth, min_periods=1, center=True).mean()
    return mean.index.values, mean.values, std.values


def plot_curves(curves, onsets, table, metric, panels, mode, smooth, gap,
                show_disasters, out_dir):
    """Over-generation curves, one panel per recovery condition, one line per
    disaster_fraction (mean over units, shaded +/-1 std), strike waves marked.
    Panels share both axes so severities are comparable across policies."""
    label, _fmt = METRICS[metric]
    frac_colours = fraction_colour_map(table)
    cells = unit_curves(curves, metric, mode)
    print(f"  Plotting {metric} curves ...")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, panel in zip(axs, panels):
        fracs = sorted({f for (p, f) in cells if p == panel})
        panel_onsets = []
        for frac in fracs:
            series = cells[(panel, frac)]
            if not series:
                continue
            gens, mean, std = mean_band(series, smooth)
            col = frac_colours.get(frac, DISASTER_COLOUR)
            ax.fill_between(gens, mean - std, mean + std, color=col, alpha=0.13,
                            lw=0, zorder=2)
            ax.plot(gens, mean, color=col, lw=2, solid_capstyle='round',
                    label=f'cull {frac:g}', zorder=3)
            for key, run_onsets in onsets.items():
                if key[0] == panel and key[1] == frac:
                    panel_onsets.extend(run_onsets)
        if show_disasters:
            draw_disaster_markers(ax, cluster_onsets(panel_onsets, gap))
        ax.set_title(panel_title(panel), fontsize=12, fontweight='bold', color=INK)
        ax.set_xlabel('Generation', color=INK)
        style_axes(ax)
        if ax.get_legend_handles_labels()[0]:
            # lower right: the strike triangles live along the TOP axis, and these
            # curves rise, so the bottom-right corner is the reliably empty one.
            legend = ax.legend(fontsize=9, title='disaster cull frac',
                               loc='lower right', framealpha=0.85, edgecolor='none')
            legend.get_frame().set_facecolor('white')
    axs[0].set_ylabel(label, color=INK)
    marker_note = '   ▼ = natural-disaster strike wave' if show_disasters else ''
    plt.suptitle(f'{label} over generations — mean ± 1 std across '
                 f'{band_label(mode)}{marker_note}',
                 fontsize=13, fontweight='bold', y=1.02, color=INK)
    plt.tight_layout()
    path = os.path.join(out_dir, f'disaster_replicates_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_spread(curves, table, metric, panels, smooth, out_dir):
    """The same curves UNPOOLED: a (recovery x fraction) grid with every
    individual run drawn faint behind the run-mean. Reading run-to-run
    stochasticity off the band alone hides whether the spread is a few outliers
    or a genuinely wide distribution — this panel shows which."""
    label, _fmt = METRICS[metric]
    frac_colours = fraction_colour_map(table)
    fracs = sorted(table['fraction'].unique())
    print(f"  Plotting {metric} per-run spread ...")
    fig, axs = plt.subplots(len(panels), len(fracs),
                            figsize=(3.4 * len(fracs), 2.9 * len(panels)),
                            sharex=True, sharey=True, squeeze=False)
    per_cell = defaultdict(list)
    for (panel, frac, _seed, _rep, _name), per_metric in curves.items():
        if metric in per_metric:
            per_cell[(panel, frac)].append(per_metric[metric])
    for row, panel in enumerate(panels):
        for col, frac in enumerate(fracs):
            ax = axs[row][col]
            series = per_cell.get((panel, frac), [])
            style_axes(ax)
            if not series:
                ax.text(0.5, 0.5, 'no runs', ha='center', va='center',
                        transform=ax.transAxes, color=DISASTER_COLOUR, fontsize=9)
                continue
            colour = frac_colours.get(frac, DISASTER_COLOUR)
            for s in series:
                y = s.rolling(smooth, min_periods=1, center=True).mean() if smooth > 1 else s
                ax.plot(y.index.values, y.values, color=colour, lw=0.7, alpha=0.28, zorder=2)
            gens, mean, _std = mean_band(series, smooth)
            ax.plot(gens, mean, color=colour, lw=2, solid_capstyle='round', zorder=3)
            ax.set_title(f'cull {frac:g}  ({len(series)} runs)', fontsize=10,
                         fontweight='bold', color=INK)
            if col == 0:
                ax.set_ylabel(f'{recovery_short(panel)}\n{label}', fontsize=9, color=INK)
            if row == len(panels) - 1:
                ax.set_xlabel('Generation', fontsize=9, color=INK)
    plt.suptitle(f'{label} per individual run — thin = one run, '
                 f'thick = mean across runs',
                 fontsize=13, fontweight='bold', y=1.005, color=INK)
    plt.tight_layout()
    path = os.path.join(out_dir, f'disaster_replicates_spread_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(table, agg, metrics, panels, mode, out_dir, window):
    """Endpoint panels (one per metric): converged value (mean of the last N
    generations) vs disaster_fraction, one series per recovery condition, mean
    +/- 1 std over units with the individual RUNS overlaid. The compact "for each
    parameter, the averaged value" tuning view."""
    print("  Plotting summary (endpoint vs disaster fraction) ...")
    metrics = [m for m in metrics if m in METRICS]
    cols = min(2, len(metrics))
    rows = (len(metrics) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6.6 * cols, 5.2 * rows), squeeze=False)
    axs = axs.ravel()
    for ax, metric in zip(axs, metrics):
        label, _fmt = METRICS[metric]
        for panel in panels:
            sub = agg[(agg['panel'] == panel) & (agg['metric'] == metric)].sort_values('fraction')
            if sub.empty or sub['mean'].isna().all():
                continue
            col = PANEL_COLOURS.get(panel, DISASTER_COLOUR)
            tsub = table[table['panel'] == panel]
            # jitter the run points so replicates of one fraction do not stack
            jitter = (np.random.default_rng(0).random(len(tsub)) - 0.5) * 0.012
            ax.scatter(tsub['fraction'] + jitter, tsub[metric], s=18, color=col,
                       alpha=0.30, lw=0, zorder=2)
            ax.errorbar(sub['fraction'], sub['mean'], yerr=sub['std'].fillna(0),
                        fmt='-o', color=col, ecolor=col, elinewidth=1.5, capsize=5,
                        lw=2, markersize=8, markeredgecolor='white',
                        markeredgewidth=1.2, zorder=3, label=recovery_short(panel))
        ax.set_title(label, fontsize=12, fontweight='bold', color=INK)
        ax.set_xlabel('disaster cull fraction', color=INK)
        ax.set_ylabel(f'{label}\n(mean of last {window} generations)', color=INK)
        style_axes(ax)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    plt.suptitle(f'Converged value vs culling severity — mean ± 1 std across '
                 f'{band_label(mode)}\ndots = individual runs',
                 fontsize=13, fontweight='bold', y=1.01, color=INK)
    plt.tight_layout()
    # One figure-level legend under the panels: the run dots cover every corner of
    # every panel, so an in-axes legend always sits on top of data.
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        # No legend title — the labels already name the recovery policy, and a
        # title stacks a second row straight onto the x-axis label.
        fig.legend(handles, labels, fontsize=9, frameon=False, loc='lower center',
                   ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    path = os.path.join(out_dir, 'disaster_replicates_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-root', default='logs/evolution',
                    help='root holding the disaster run folders (searched recursively)')
    ap.add_argument('--out', default='output/disaster_replicates', help='output directory')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    help=f'metrics to plot (default avg_fitness); choose from {list(METRICS)}')
    ap.add_argument('--reps', type=int, default=0,
                    help='use only the first N runs of each (recovery, fraction, seed) '
                         'cell — e.g. 3 or 5 to re-read a 10-per-seed campaign '
                         '(default 0 = every run found)')
    ap.add_argument('--seeds', type=int, nargs='+',
                    help='restrict to these map seeds (default: all found)')
    ap.add_argument('--aggregate', choices=['runs', 'seeds'], default='runs',
                    help="what the mean/band is taken over: 'runs' (default, every "
                         "run is a unit) or 'seeds' (replicates averaged first)")
    ap.add_argument('--final-window', type=int, default=50,
                    help='endpoint = mean of the last N generations (default 50)')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='ignore generations beyond this (default: all)')
    ap.add_argument('--smooth', type=int, default=5,
                    help='rolling-mean window for the curves (1 = raw, default 5)')
    ap.add_argument('--disaster-gap', type=int, default=40,
                    help='onsets within this many generations are one strike wave (default 40)')
    ap.add_argument('--no-disaster-markers', action='store_true',
                    help='omit the natural-disaster strike markers')
    ap.add_argument('--no-spread-plot', action='store_true',
                    help='skip the per-run spread grid')
    args = ap.parse_args()

    metrics = [m for m in args.metrics if m in METRICS]
    if not metrics:
        raise SystemExit(f"No known metrics in {args.metrics}; choose from {list(METRICS)}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Discovering natural-disaster runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root, set(args.seeds) if args.seeds else None)
    if not runs:
        raise SystemExit(f"No disaster runs found under {args.logs_root}")
    census, n_cells = replicate_census(runs)
    print(f"Found {len(runs)} runs over {n_cells} (recovery, fraction, seed) cells; "
          f"runs per cell: {census}")
    if args.reps:
        runs = limit_replicates(runs, args.reps)
        census, n_cells = replicate_census(runs)
        print(f"--reps {args.reps}: kept {len(runs)} runs; runs per cell: {census}")

    print("Loading ...")
    table, curves, onsets = load_all(runs, metrics, args.final_window, args.max_gen)
    panels = ordered_panels(table)
    fracs = sorted(table['fraction'].unique())
    print(f"Recovery panels: {panels}")
    print(f"Disaster fractions: {fracs}")
    print(f"Map seeds: {sorted(table['seed'].unique())}")

    runs_csv = os.path.join(args.out, 'disaster_replicates_runs.csv')
    table.to_csv(runs_csv, index=False)
    print(f"Saved: {runs_csv}")

    units = unit_table(table, metrics, args.aggregate)
    agg = aggregate(units, metrics, args.aggregate)
    summary_csv = os.path.join(args.out, 'disaster_replicates_summary.csv')
    agg.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    for m in metrics:
        head = METRICS[m][0]
        print(f"\n{head} — endpoint (mean of last {args.final_window} gens), "
              f"mean ± std across {band_label(args.aggregate)}:")
        for panel in panels:
            sub = agg[(agg['panel'] == panel) & (agg['metric'] == m)].sort_values('fraction')
            cells = '  '.join(f"cull {r.fraction:g}: {r['mean']:.3f}±{(r['std'] or 0):.3f}"
                              f" (n={int(r['count'])})" for _i, r in sub.iterrows())
            print(f"  {recovery_short(panel):<14} {cells}")
    print()

    show = not args.no_disaster_markers
    for m in metrics:
        plot_curves(curves, onsets, table, m, panels, args.aggregate, args.smooth,
                    args.disaster_gap, show, args.out)
        if not args.no_spread_plot:
            plot_spread(curves, table, m, panels, args.smooth, args.out)
    plot_summary(table, agg, metrics, panels, args.aggregate, args.out, args.final_window)
    print("Done.")


if __name__ == '__main__':
    main()
