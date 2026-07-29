"""
analyse_food_shuffle.py
======================================================================
Aggregate + plot the FOOD-SHUFFLE (non-stationary payoff) experiment.

The shuffle world periodically re-permutes which food TIER pays which ENERGY
VALUE (src/Environments/FoodShuffle.js): every `food_shuffle_period` TICKS the
multiset {0.5, 1.0, 2.0} is re-assigned across low/medium/prestige food. The
spatial layout never moves — only the payoff a colour delivers. A genome that
hard-codes "always chase prestige" is punished after each shuffle, so the shorter
the period the more the environment rewards WITHIN-LIFETIME learning over a fixed
evolved policy.

The sweep run by run_shuffle_condition_sweep_array.slurm is:

    food_shuffle_period  in {300, 1000, 3000, 10000, 100000} ticks
    condition            evolution / learning / pure RL
    x 5 map seeds (1, 42, 123, 456, 999) x 3 replicates    -> 15 runs per cell

With ticks_per_gen = 10000 the period spans two regimes either side of a
generation: P=300 reshuffles ~33x per generation (no fixed policy can hold),
P=100000 reshuffles once per 10 generations (a fixed policy is fine most of the
time). Panel titles print both the raw period and that per-generation rate.

LAYOUT — the question the figures answer is "at a given shuffle rate, how do the
three conditions compare?", so by default each PANEL is one food_shuffle_period
and the three CONDITIONS are drawn together inside it, over generations. Pass
`--facet condition` for the transpose (one panel per condition, one line per
period) when the question is instead "how does one condition degrade as the world
speeds up?".

Every independent run is one UNIT: the line is the mean across all runs of a
(condition, period) cell and the band is +/-1 std across those runs, so the
spread shown is genuine run-to-run variability (seeds and replicates pooled).
`--aggregate seeds` averages the replicates of a seed together first, making the
band the between-seed spread instead. Replicate count is free — 1, 3, 5 or 10
runs per (condition, period, seed) all work, and `--reps N` truncates each cell
to its first N so an uneven campaign can be levelled off.

METRICS
-------
From generations.csv (population level, one row per generation):
    avg_fitness            -- average fitness of the population
    top20percent_fitness   -- average fitness of the top 20%
    total_agents           -- total organisms alive in the generation
    peak_population        -- peak organism count in the generation

From organisms.csv (per-organism rows, averaged to a per-generation
PER-ORGANISM mean so a bigger population does not inflate the number) — this is
the "are they still finding food?" signal, which is what separates a population
that has adapted to the shuffle from one that is merely surviving:
    food_items_eaten       -- food items eaten per organism per lifetime
                              (food_low + food_medium + food_prestige + food_default)
    food_prestige / food_medium / food_low / food_default
                           -- the same count split BY TIER. The shuffle permutes
                              which tier pays what, so a population that re-targets
                              its diet after a shuffle moves these curves even when
                              the total items eaten holds flat. Note the tier NAME
                              is fixed (it is the colour the agent perceives) while
                              its payoff moves, so these are "how much of each
                              colour was eaten", not "how much high-value food".
    food_value_per_item    -- food_score / food_items_eaten: the average payoff
                              per item eaten. THIS is the metric that separates
                              "still finding food" from "still finding GOOD food"
                              — an agent that keeps eating at the same rate but
                              chases a tier the shuffle has devalued holds its
                              item count while this value falls toward the 0.5/1.0
                              tiers and away from 2.0.
    food_score             -- cumulative_food_score: the ENERGY those items paid.
                              NOTE this run family scores fitness AS the cumulative
                              food score, so this curve tracks avg_fitness almost
                              exactly. It is kept as the cross-check that the
                              fitness signal really is food, not as new evidence.

organisms.csv is ~150 MB per run, so it is read with usecols + a groupby only
(~0.4 s/run). Pass --no-food to skip it entirely when only the population
metrics are wanted.

OUTPUTS (into --out, default output/food_shuffle)
    food_shuffle_overview.png            every metric x every panel in one grid
    food_shuffle_curves_<metric>.png     one figure per metric, full size
    food_shuffle_summary.png / .csv      converged value vs shuffle period (log x)
    food_shuffle_runs.csv                per-run endpoint table behind it all

Usage
-----
    python analyse_food_shuffle.py --logs-root logs

    # population metrics only (skips the organisms.csv pass):
    python analyse_food_shuffle.py --no-food

    # transpose: one panel per condition, one line per shuffle period
    python analyse_food_shuffle.py --facet condition

    # band = between-seed spread instead of between-run:
    python analyse_food_shuffle.py --aggregate seeds
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

warnings.filterwarnings('ignore')

# Condition label, keyed by (params.condition, params.mode) because "learning"
# and "pure RL" share a condition string and differ only by mode.
CONDITIONS = {
    ('evolution', 'standard'): 'evolution',
    ('learning', 'standard'):  'learning',
    ('learning', 'pure_rl'):   'pure RL',
}
CONDITION_ORDER = ['evolution', 'learning', 'pure RL']

# Colour per CONDITION — the default series, so these are the hues that carry
# most figures. Taken from the sibling analysis scripts in an order whose every
# adjacent pair clears the normal-vision and colour-vision-deficiency separation
# floors (the amber/red pair of the original palette sits at dE 14.4 for normal
# vision, below the readable floor, so red is not used alongside amber here).
CONDITION_COLOURS = {'evolution': '#2563EB', 'learning': '#D97706', 'pure RL': '#059669'}

# Secondary encoding for the condition series. Evolution and learning track each
# other almost exactly on most metrics, so whichever is drawn second would hide
# the other under a solid line of equal width. Dashing learning keeps both
# readable where they coincide, and keeps identity off colour alone for a
# colour-vision-deficient or greyscale-printed reader.
CONDITION_STYLES = {'evolution': '-', 'learning': (0, (6, 3)), 'pure RL': '-'}

# Colour per food_shuffle_period, assigned in sorted (ascending period) order.
# Used when --facet condition makes the period the series instead.
PERIOD_COLOURS = ['#2563EB', '#D97706', '#059669', '#DC2626', '#7C3AED', '#0891B2']

# Recessive ink for axes/labels — text never wears a series colour.
INK = '#374151'
MUTED = '#6B7280'

# The per-organism columns summed into "items eaten". food_default is the
# unshuffled fallback tier; it is counted because the question is whether they
# are eating AT ALL, not which tier they picked.
FOOD_COUNT_COLS = ['food_low', 'food_medium', 'food_prestige', 'food_default']
ORGANISM_COLS = ['generation', 'cumulative_food_score'] + FOOD_COUNT_COLS

# Metric -> (axis label, source, value format). 'gen' = generations.csv,
# 'org' = derived from organisms.csv.
METRICS = {
    'avg_fitness':          ('Average fitness',            'gen', '{:.3f}'),
    'top20percent_fitness': ('Top-20% fitness',            'gen', '{:.3f}'),
    'total_agents':         ('Total agents',               'gen', '{:.0f}'),
    'peak_population':      ('Peak agents',                'gen', '{:.0f}'),
    'food_items_eaten':     ('Food items eaten\nper organism', 'org', '{:.2f}'),
    'food_prestige':        ('Prestige food eaten\nper organism', 'org', '{:.2f}'),
    'food_medium':          ('Medium food eaten\nper organism', 'org', '{:.2f}'),
    'food_low':             ('Low food eaten\nper organism', 'org', '{:.2f}'),
    'food_default':         ('Default food eaten\nper organism', 'org', '{:.2f}'),
    'food_value_per_item':  ('Food value\nper item eaten', 'org', '{:.3f}'),
    'food_score':           ('Food score\nper organism',   'org', '{:.2f}'),
}
DEFAULT_METRICS = list(METRICS)


def replicate_index(run_dir):
    """Replicate number parsed off the folder name (…_rep3 / …_r3). Used only to
    ORDER a cell's runs so `--reps N` takes a stable "first N"; folders with no
    replicate suffix sort first."""
    m = re.search(r'_r(?:ep)?(\d+)$', os.path.basename(run_dir))
    return int(m.group(1)) if m else 0


def shuffle_rate_label(period, ticks_per_gen):
    """Period in ticks -> how often that is in GENERATIONS, which is the scale the
    x-axis is in. Above one shuffle per generation the rate is the readable
    number; below it, the gap between shuffles is."""
    if not period or not ticks_per_gen:
        return ''
    rate = ticks_per_gen / float(period)
    if rate >= 1:
        return f'{rate:.0f}/gen' if rate >= 2 else '1/gen'
    return f'1 per {1 / rate:.0f} gens'


def discover_runs(logs_root, seeds=None, conditions=None):
    """Every run under logs_root (searched recursively) with params.json +
    generations.csv and a POSITIVE food_shuffle_period. params.json is
    authoritative — folder names are only read for the replicate index — so this
    picks the shuffle runs out of a logs tree that also holds the disaster and
    baseline campaigns. Returns dicts: {condition, period, seed, rep,
    ticks_per_gen, name, dir, gen_csv, org_csv}."""
    runs = []
    for params_path in sorted(glob.glob(os.path.join(logs_root, '**', 'params.json'),
                                        recursive=True)):
        run_dir = os.path.dirname(params_path)
        gen_csv = os.path.join(run_dir, 'generations.csv')
        if not os.path.exists(gen_csv):
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
            period = float(p.get('food_shuffle_period', 0) or 0)
            if period <= 0:
                continue  # not a shuffle run (baseline / disaster / roam) — skip
            condition = CONDITIONS.get((p.get('condition'), p.get('mode')))
            if condition is None:
                print(f"  skip {os.path.basename(run_dir)} — unknown condition/mode "
                      f"({p.get('condition')}/{p.get('mode')})")
                continue
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"  skip {os.path.basename(run_dir)} — bad params.json ({e})")
            continue
        if seeds and seed not in seeds:
            continue
        if conditions and condition not in conditions:
            continue
        org_csv = os.path.join(run_dir, 'organisms.csv')
        runs.append({'condition': condition, 'period': period, 'seed': seed,
                     'rep': replicate_index(run_dir),
                     'ticks_per_gen': float(p.get('ticks_per_gen', 0) or 0),
                     'name': os.path.basename(run_dir), 'dir': run_dir,
                     'gen_csv': gen_csv,
                     'org_csv': org_csv if os.path.exists(org_csv) else None})
    return runs


def limit_replicates(runs, n_reps):
    """Keep only the first `n_reps` runs of each (condition, period, seed) cell,
    ordered by replicate index then folder name. n_reps None/0 keeps everything."""
    if not n_reps:
        return runs
    cells = defaultdict(list)
    for r in runs:
        cells[(r['condition'], r['period'], r['seed'])].append(r)
    kept = []
    for grp in cells.values():
        grp.sort(key=lambda r: (r['rep'], r['name']))
        kept.extend(grp[:n_reps])
    return kept


def replicate_census(runs):
    """{runs-per-cell: how many cells have that count} — printed so an uneven or
    still-running campaign is visible before any figure is drawn."""
    counts = defaultdict(int)
    for r in runs:
        counts[(r['condition'], r['period'], r['seed'])] += 1
    census = defaultdict(int)
    for n in counts.values():
        census[n] += 1
    return dict(sorted(census.items())), len(counts)


def load_generations(gen_csv, metrics, max_gen):
    """Population metrics from one run's generations.csv, indexed by generation.
    Rows are deduplicated on 'generation' (keep last) because Logger auto-save
    APPENDS — a resumed or re-run folder can hold doubled rows that would
    otherwise double-count the tail and break the concat in the plots."""
    want = [m for m in metrics if METRICS[m][1] == 'gen']
    df = pd.read_csv(gen_csv)
    keep = ['generation'] + [c for c in want if c in df.columns]
    df = df[keep].copy()
    df = df[df['generation'] <= max_gen]
    return df.drop_duplicates('generation', keep='last').set_index('generation').sort_index()


def load_food(org_csv, max_gen):
    """Per-generation, PER-ORGANISM food metrics from one run's organisms.csv.

    organisms.csv is one row per organism per generation (~1.4M rows, ~150 MB), so
    only the six needed columns are parsed and the frame is collapsed by a single
    groupby('generation').mean() — the MEAN over the generation's organisms, not
    the sum, so a generation with more agents does not read as "more food found".

    Returns a frame indexed by generation with food_items_eaten, food_score and
    their ratio food_value_per_item. The ratio is taken between the two
    generation MEANS rather than per organism, so an organism that died before
    eating anything cannot divide by zero."""
    df = pd.read_csv(org_csv, usecols=lambda c: c in ORGANISM_COLS)
    missing = [c for c in ORGANISM_COLS if c not in df.columns]
    if missing:
        return None
    df = df[df['generation'] <= max_gen]
    tiers = df[FOOD_COUNT_COLS].apply(pd.to_numeric, errors='coerce')
    per_org = pd.DataFrame({
        'generation': df['generation'],
        'food_items_eaten': tiers.sum(axis=1),
        'food_score': pd.to_numeric(df['cumulative_food_score'], errors='coerce'),
        # Each tier kept alongside the total: the shuffle permutes which TIER pays
        # what, so the per-tier counts are where a diet shift shows up even when
        # the total items eaten holds flat.
        **{c: tiers[c] for c in FOOD_COUNT_COLS},
    })
    out = per_org.groupby('generation').mean().sort_index()
    out['food_value_per_item'] = (out['food_score'] /
                                  out['food_items_eaten'].replace(0, np.nan))
    return out


def load_all(runs, metrics, window, max_gen, with_food):
    """Load every run once -> (per-run endpoint table, {run key: {metric: series}}).

    The endpoint is the mean of the last `window` recorded generations — the run's
    converged value. The run key is (condition, period, seed, rep, name) so every
    run stays individually addressable and nothing is silently merged."""
    need_food = with_food and any(METRICS[m][1] == 'org' for m in metrics)
    rows, curves = [], {}
    total = len(runs)
    for i, r in enumerate(sorted(runs, key=lambda r: (r['condition'], r['period'],
                                                      r['seed'], r['rep'])), 1):
        frames = [load_generations(r['gen_csv'], metrics, max_gen)]
        if need_food and r['org_csv']:
            food = load_food(r['org_csv'], max_gen)
            if food is not None:
                frames.append(food)
        df = pd.concat(frames, axis=1)

        key = (r['condition'], r['period'], r['seed'], r['rep'], r['name'])
        curves[key] = {m: pd.to_numeric(df[m], errors='coerce')
                       for m in metrics if m in df.columns}
        row = {'condition': r['condition'], 'period': r['period'], 'seed': r['seed'],
               'rep': r['rep'], 'run': r['name'],
               'n_gens': int(df.index.max()) if len(df) else 0}
        for m in metrics:
            tail = curves[key][m].dropna().tail(window) if m in curves[key] else pd.Series(dtype=float)
            row[m] = tail.mean() if len(tail) else np.nan
        rows.append(row)
        if i % 25 == 0 or i == total:
            print(f"  loaded {i}/{total} runs")
    return pd.DataFrame(rows), curves


def unit_curves(curves, metric, mode):
    """Regroup per-run curves into the UNITS the band is measured over.

    mode 'runs'  -> one unit per run; the band is the run-to-run spread.
    mode 'seeds' -> replicates averaged into one curve per seed first, so the
                    band is the between-seed spread.
    Returns {(condition, period): [series, ...]}."""
    by_cell = defaultdict(list)
    for (condition, period, seed, _rep, _name), per_metric in curves.items():
        if metric in per_metric:
            by_cell[(condition, period, seed)].append(per_metric[metric])
    out = defaultdict(list)
    for (condition, period, _seed), series in by_cell.items():
        if mode == 'seeds':
            out[(condition, period)].append(pd.concat(series, axis=1).mean(axis=1))
        else:
            out[(condition, period)].extend(series)
    return out


def unit_table(table, metrics, mode):
    """The endpoint table at the aggregation unit: unchanged for 'runs', or
    replicate-averaged to one row per (condition, period, seed) for 'seeds'."""
    if mode == 'runs':
        return table
    return (table.groupby(['condition', 'period', 'seed'], as_index=False)[list(metrics)]
                 .mean())


def aggregate(units, metrics, mode):
    """Aggregate each metric over the units so (condition, period) is the only
    factor left. Returns tidy: condition, period, metric, mean, std, sem, count
    (count = units, i.e. runs or seeds depending on `mode`)."""
    frames = []
    for m in metrics:
        g = (units.groupby(['condition', 'period'])[m]
                  .agg(['mean', 'std', 'sem', 'count']).reset_index())
        g.insert(2, 'metric', m)
        g['unit'] = mode
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(['metric', 'condition', 'period'])


def ordered_conditions(table):
    """Conditions present, in the canonical evolution -> learning -> pure RL order."""
    present = set(table['condition'].unique())
    known = [c for c in CONDITION_ORDER if c in present]
    return known + sorted(present - set(known))


class Facets:
    """How the (condition, period) grid is split into panels and lines.

    `by='period'` (default) puts one PANEL per food_shuffle_period and draws the
    three CONDITIONS together inside it — the layout for "at this shuffle rate,
    how do the conditions compare?". `by='condition'` is the transpose. Every
    plotting function reads panels/series/colour/title from here, so the two
    layouts share one code path and cannot drift apart."""

    def __init__(self, by, table, ticks_per_gen):
        self.by = by
        self.ticks_per_gen = ticks_per_gen
        self.conditions = ordered_conditions(table)
        self.periods = sorted(table['period'].unique())
        if by == 'period':
            self.panels, self.series = self.periods, self.conditions
            self.colours = dict(CONDITION_COLOURS)
            self.styles = dict(CONDITION_STYLES)
        else:
            self.panels, self.series = self.conditions, self.periods
            self.colours = {p: PERIOD_COLOURS[i % len(PERIOD_COLOURS)]
                            for i, p in enumerate(self.periods)}
            # Five well-separated hues that do not systematically overlap — no
            # dashing needed, and dashing five lines would only add clutter.
            self.styles = {p: '-' for p in self.periods}

    def cell(self, panel, series):
        """(panel, series) -> the (condition, period) key `unit_curves` is keyed by."""
        return (series, panel) if self.by == 'period' else (panel, series)

    def panel_title(self, panel):
        if self.by == 'period':
            rate = shuffle_rate_label(panel, self.ticks_per_gen)
            return f'P={panel:g} ticks  ({rate})' if rate else f'P={panel:g} ticks'
        return panel

    def series_label(self, series):
        if self.by == 'period':
            return series
        rate = shuffle_rate_label(series, self.ticks_per_gen)
        return f'P={series:g}  ({rate})' if rate else f'P={series:g}'

    def colour(self, series):
        return self.colours.get(series, MUTED)

    def style(self, series):
        return self.styles.get(series, '-')


def _slug(metric):
    return re.sub(r'[^0-9a-zA-Z]+', '_', metric).strip('_')


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


def draw_panel(ax, cells, facets, panel, smooth):
    """One panel: a mean line per series with its +/-1 std band. Every band is
    laid down before any line so a wide band never buries a neighbouring mean."""
    drawn = []
    for s in facets.series:
        series = cells.get(facets.cell(panel, s))
        if not series:
            continue
        gens, mean, std = mean_band(series, smooth)
        col = facets.colour(s)
        ax.fill_between(gens, mean - std, mean + std, color=col, alpha=0.13,
                        lw=0, zorder=2)
        drawn.append((gens, mean, col, facets.style(s), facets.series_label(s)))
    for gens, mean, col, ls, label in drawn:
        ax.plot(gens, mean, color=col, lw=2, linestyle=ls, solid_capstyle='round',
                dash_capstyle='round', label=label, zorder=3)
    style_axes(ax)


def figure_legend(fig, ax, ncol, y):
    """One legend for the whole figure, under the panels: every panel carries the
    same series, so a per-axes legend would just repeat itself on top of the data.
    No legend title — the labels already name the factor, and a title stacks a
    second row straight onto the panels' x-axis label."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=10, frameon=False, loc='lower center',
                   ncol=ncol, bbox_to_anchor=(0.5, y))


def plot_overview(curves, metrics, facets, mode, smooth, out_dir):
    """Every metric x every panel in one grid: rows are metrics, columns are the
    facet (shuffle period by default), lines are the series (the three conditions
    by default). Rows share a y-axis so a panel can be compared straight across;
    columns share the generation axis."""
    print("  Plotting overview grid ...")
    n_rows, n_cols = len(metrics), len(facets.panels)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(4.4 * n_cols, 2.9 * n_rows),
                            sharex=True, sharey='row', squeeze=False)
    for row, metric in enumerate(metrics):
        cells = unit_curves(curves, metric, mode)
        label = METRICS[metric][0]
        for col, panel in enumerate(facets.panels):
            ax = axs[row][col]
            draw_panel(ax, cells, facets, panel, smooth)
            if row == 0:
                ax.set_title(facets.panel_title(panel), fontsize=12,
                             fontweight='bold', color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, color=INK)
            if row == n_rows - 1:
                ax.set_xlabel('Generation', fontsize=10, color=INK)
    plt.suptitle(f'Food-shuffle environment — mean ± 1 std across {band_label(mode)}',
                 fontsize=15, fontweight='bold', y=1.005, color=INK)
    plt.tight_layout()
    figure_legend(fig, axs[0][0], len(facets.series), y=-0.015)
    path = os.path.join(out_dir, 'food_shuffle_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_metric(curves, metric, facets, mode, smooth, out_dir):
    """One metric at full size: a panel per facet value, a line per series, mean
    across units with the +/-1 std band. Panels share both axes so they are
    directly comparable."""
    label, _src, _fmt = METRICS[metric]
    print(f"  Plotting {metric} ...")
    cells = unit_curves(curves, metric, mode)
    n = len(facets.panels)
    fig, axs = plt.subplots(1, n, figsize=(4.8 * n, 4.6), sharex=True, sharey=True,
                            squeeze=False)
    axs = axs[0]
    for ax, panel in zip(axs, facets.panels):
        draw_panel(ax, cells, facets, panel, smooth)
        ax.set_title(facets.panel_title(panel), fontsize=12, fontweight='bold',
                     color=INK)
        ax.set_xlabel('Generation', color=INK)
    axs[0].set_ylabel(label.replace('\n', ' '), color=INK)
    plt.suptitle(f'{label.replace(chr(10), " ")} over generations — '
                 f'mean ± 1 std across {band_label(mode)}',
                 fontsize=13, fontweight='bold', y=1.03, color=INK)
    plt.tight_layout()
    figure_legend(fig, axs[0], len(facets.series), y=-0.11)
    path = os.path.join(out_dir, f'food_shuffle_curves_{_slug(metric)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(table, agg, metrics, conditions, mode, out_dir, window):
    """Endpoint panels (one per metric): converged value (mean of the last N
    generations) vs food_shuffle_period on a log x-axis, one series per condition,
    mean +/- 1 std over units with the individual runs overlaid. Independent of
    --facet: this view always compares the three conditions across the periods,
    because that is the endpoint question."""
    print("  Plotting summary (endpoint vs shuffle period) ...")
    cols = min(3, len(metrics))
    rows = (len(metrics) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.4 * rows), squeeze=False)
    axs = axs.ravel()
    rng = np.random.default_rng(0)
    for ax, metric in zip(axs, metrics):
        label = METRICS[metric][0]
        for condition in conditions:
            sub = agg[(agg['condition'] == condition) &
                      (agg['metric'] == metric)].sort_values('period')
            if sub.empty or sub['mean'].isna().all():
                continue
            col = CONDITION_COLOURS.get(condition, MUTED)
            tsub = table[table['condition'] == condition]
            # multiplicative jitter: the x-axis is log, so a fixed offset would
            # smear the small periods and do nothing at the large ones.
            jitter = np.exp((rng.random(len(tsub)) - 0.5) * 0.10)
            ax.scatter(tsub['period'] * jitter, tsub[metric], s=16, color=col,
                       alpha=0.28, lw=0, zorder=2)
            ax.errorbar(sub['period'], sub['mean'], yerr=sub['std'].fillna(0),
                        fmt='-o', color=col, ecolor=col, elinewidth=1.5, capsize=4,
                        lw=2, markersize=7, markeredgecolor='white',
                        markeredgewidth=1.2, zorder=3, label=condition)
        ax.set_xscale('log')
        ax.set_title(label.replace('\n', ' '), fontsize=12, fontweight='bold', color=INK)
        ax.set_xlabel('food-shuffle period (ticks, log scale)', color=INK)
        ax.set_ylabel(f'mean of last {window} generations', fontsize=9, color=INK)
        style_axes(ax)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    plt.suptitle(f'Food-shuffle environment — converged metrics vs shuffle period\n'
                 f'mean ± 1 std across {band_label(mode)}, dots = individual runs',
                 fontsize=13, fontweight='bold', y=1.02, color=INK)
    plt.tight_layout()
    figure_legend(fig, axs[0], len(conditions), y=-0.04)
    path = os.path.join(out_dir, 'food_shuffle_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-root', default='logs',
                    help='root searched recursively for shuffle runs (default logs)')
    ap.add_argument('--out', default='output/food_shuffle', help='output directory')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    help=f'subset/order of metrics (default all); choose from {list(METRICS)}')
    ap.add_argument('--facet', choices=['period', 'condition'], default='period',
                    help="what a PANEL is: 'period' (default — one panel per "
                         "food_shuffle_period, the three conditions drawn together) "
                         "or 'condition' (one panel per condition, one line per period)")
    ap.add_argument('--no-food', action='store_true',
                    help='skip the organisms.csv pass (drops the food metrics; much faster)')
    ap.add_argument('--conditions', nargs='+', choices=CONDITION_ORDER,
                    help='restrict to these conditions (default: all found)')
    ap.add_argument('--seeds', type=int, nargs='+',
                    help='restrict to these map seeds (default: all found)')
    ap.add_argument('--reps', type=int, default=0,
                    help='use only the first N runs of each (condition, period, seed) '
                         'cell (default 0 = every run found)')
    ap.add_argument('--aggregate', choices=['runs', 'seeds'], default='runs',
                    help="what the mean/band is taken over: 'runs' (default, every "
                         "run is a unit) or 'seeds' (replicates averaged first)")
    ap.add_argument('--final-window', type=int, default=50,
                    help='endpoint = mean of the last N generations (default 50)')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='ignore generations beyond this (default: all)')
    ap.add_argument('--smooth', type=int, default=5,
                    help='rolling-mean window for the curves (1 = raw, default 5)')
    args = ap.parse_args()

    metrics = [m for m in args.metrics if m in METRICS]
    if args.no_food:
        metrics = [m for m in metrics if METRICS[m][1] != 'org']
    if not metrics:
        raise SystemExit(f"No known metrics in {args.metrics}; choose from {list(METRICS)}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Discovering food-shuffle runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root,
                         set(args.seeds) if args.seeds else None,
                         set(args.conditions) if args.conditions else None)
    if not runs:
        raise SystemExit(f"No food-shuffle runs (food_shuffle_period > 0) under {args.logs_root}")
    census, n_cells = replicate_census(runs)
    print(f"Found {len(runs)} runs over {n_cells} (condition, period, seed) cells; "
          f"runs per cell: {census}")
    if args.reps:
        runs = limit_replicates(runs, args.reps)
        census, n_cells = replicate_census(runs)
        print(f"--reps {args.reps}: kept {len(runs)} runs; runs per cell: {census}")

    ticks_per_gen = next((r['ticks_per_gen'] for r in runs if r['ticks_per_gen']), 0)
    missing_org = [r['name'] for r in runs if r['org_csv'] is None]
    if missing_org and not args.no_food:
        print(f"  note: {len(missing_org)} run(s) have no organisms.csv — "
              f"they contribute no food metrics")

    print(f"Loading {len(runs)} runs "
          f"({'generations.csv only' if args.no_food else 'generations.csv + organisms.csv'}) ...")
    table, curves = load_all(runs, metrics, args.final_window, args.max_gen,
                             not args.no_food)
    facets = Facets(args.facet, table, ticks_per_gen)
    print(f"Conditions: {facets.conditions}")
    print(f"Shuffle periods: "
          f"{[f'{p:g} ({shuffle_rate_label(p, ticks_per_gen)})' for p in facets.periods]}")
    print(f"Map seeds: {sorted(table['seed'].unique())}")
    print(f"Panels: {args.facet}")

    runs_csv = os.path.join(args.out, 'food_shuffle_runs.csv')
    table.to_csv(runs_csv, index=False)
    print(f"Saved: {runs_csv}")

    units = unit_table(table, metrics, args.aggregate)
    agg = aggregate(units, metrics, args.aggregate)
    summary_csv = os.path.join(args.out, 'food_shuffle_summary.csv')
    agg.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Printed condition-major: one block per metric, one line per condition, the
    # periods across it — the same reading order as the default panels.
    for m in metrics:
        label, _src, fmt = METRICS[m]
        print(f"\n{label.replace(chr(10), ' ')} — endpoint (mean of last "
              f"{args.final_window} gens), mean ± std across {band_label(args.aggregate)}:")
        for condition in facets.conditions:
            sub = agg[(agg['condition'] == condition) &
                      (agg['metric'] == m)].sort_values('period')
            cells = '  '.join(f"P={r.period:g}: {fmt.format(r['mean'])}"
                              f"±{fmt.format(r['std'] or 0)} (n={int(r['count'])})"
                              for _i, r in sub.iterrows())
            print(f"  {condition:<11} {cells}")
    print()

    plot_overview(curves, metrics, facets, args.aggregate, args.smooth, args.out)
    for m in metrics:
        plot_metric(curves, m, facets, args.aggregate, args.smooth, args.out)
    plot_summary(table, agg, metrics, facets.conditions, args.aggregate, args.out,
                 args.final_window)
    print("Done.")


if __name__ == '__main__':
    main()
