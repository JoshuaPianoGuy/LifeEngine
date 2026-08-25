"""
analyse_evolution_hard_runs.py
======================================================================
The EVOLUTION campaign on the HARD environment, read as one population of
independent runs (the 100-run set: 10 map seeds x 10 replicates).

Every other script in this family compares something — conditions, learning
rates, disaster tunings. This one compares nothing. It asks the plain question
"what does evolution do on the hard map, and how much do runs disagree?", so the
only factor is the RUN itself and every figure is a mean over runs with the
run-to-run spread drawn around it.

Runs are found by params.json (never by folder name):

    condition                 == evolution
    roaming_predator_count    == --predators (60)      -> the HARD environment
    predator_drain            == --drain (5)
    map_seed                  pooled (10 seeds x 10 replicates = 100 runs)

REPLICATE COUNT IS FREE. The seeds only fix the terrain; agent spawning,
mutation and predator wandering are unseeded, so the 10 runs of a seed are
independent samples. `--reps N` truncates every seed to its first N runs (folder
suffix _r<N> orders them), so the same campaign can be re-analysed as a 3- or
5-run one and the figures compared like for like.

Written, into --out (default output/evolution_hard_runs):

    evolution_hard_fitness.png
        avg / top-20% / best fitness over generations, one panel each, mean over
        runs with a +/-1 std band and every run drawn faint behind it.
    evolution_hard_population.png
        total agents and peak population over generations, same treatment.
    evolution_hard_genome_variance.png
        genome_variance over generations -- how much genetic diversity the
        population is still carrying as it converges.
    evolution_hard_final_fitness_strip.png
        one dot per run for the converged fitness (mean of the last
        --final-window generations): each fitness metric pooled over all runs on
        its own axis, then average fitness split by map seed so terrain effects
        can be told apart from run-to-run noise.
    evolution_hard_overview.png
        the four-panel headline figure (fitness, population, genome variance,
        final-fitness strip).
    evolution_hard_weight_space.png
        the genome's weight SCALE: avg_network_weight_mag (an RMS — the level)
        beside inter_gen_weight_change (its per-generation rate of change). See
        the WEIGHT_METRICS comment below for why the two are always drawn
        together. Pairs with learning_hard_weight_space.png from
        analyse_learning_hard_runs.py, which adds the in-life mean absolute weight difference panel that has
        no meaning here. Suppress with --no-weight-space.
    evolution_hard_runs.csv
        per-run endpoint table -- one row per run, the data behind the strip.
    evolution_hard_curves.csv
        the mean curves themselves (generation x metric: mean, std, sem, n).

All metrics are read straight from generations.csv. "Fitness" is the species'
fitness metric; "population"/"agents" is the organism count -- different
measurements, so never plotted on a shared axis.

Usage
-----
    python analyse_evolution_hard_runs.py

    # band = between-seed spread instead of between-run:
    python analyse_evolution_hard_runs.py --aggregate seeds

    # median/IQR instead of mean/std; the literal last generation as "final":
    python analyse_evolution_hard_runs.py --band iqr --final-window 1

    # re-analyse the 10-per-seed campaign as a 3-per-seed one:
    python analyse_evolution_hard_runs.py --reps 3 --out output/evo_hard_reps3
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

# ── Environment presets ───────────────────────────────────────────────────────
# `roaming` filters roaming_predator_count; `drain` filters predator_drain
# (None = don't filter, used for baseline where drain is an irrelevant default).
ENVIRONMENTS = {
    'baseline': {'roaming': 0.0,  'drain': None, 'desc': 'predator-free baseline'},
    'hard':     {'roaming': 60.0, 'drain': 5.0,  'desc': '60 roaming predators, drain 5'},
}

# ── Colours ───────────────────────────────────────────────────────────────────
# Every set that shares a panel was checked with the palette validator for
# colour-vision separation, so no two lines a reader must tell apart collide:
#   fitness    #2563EB/#D97706/#DB2777 -- worst all-pairs dE 14.1 (deutan)
#   population #0891B2/#DC2626         -- worst all-pairs dE 19.5 (deutan)
# genome variance is a lone series, so it only has to clear the surface.
# NOTE if you re-colour: blue+purple and green+pink are both invisible pairs
# under deuteranopia, and amber+red falls below the normal-vision floor.
METRIC_COLOURS = {
    'avg_fitness':          '#2563EB',   # blue
    'top20percent_fitness': '#D97706',   # amber
    'best_fitness':         '#DB2777',   # pink
    'total_agents':         '#0891B2',   # cyan
    'peak_population':      '#DC2626',   # red
    'genome_variance':      '#7C3AED',   # purple
    # Same hues as analyse_learning_hard_runs.py, deliberately: a metric wears
    # one colour across both arms so the figures can be read side by side.
    'avg_network_weight_mag':  '#0F766E',   # deep teal
    'inter_gen_weight_change': '#475569',   # slate
}

# Recessive ink for axes/labels — text never wears a series colour.
INK = '#374151'
GRID = '#D1D5DB'

# ── Viability threshold ───────────────────────────────────────────────────────
# The replacement point: below it a lineage shrinks, above it grows.
# AdvancedOrganism fires a reproduction ATTEMPT at 3.5 food gained since the last
# one, and the counter resets whether or not placement succeeded; attempts land
# with REPRODUCTION_SUCCESS_RATE = 0.8. So the food needed to actually replace
# itself is 3.5 / 0.8 = 4.375.
#
# SCALE: 4.375 lives on the REPLAY scale — a real reproducing generation over 5
# maps — which is exactly what avg_fitness in generations.csv measures, so it is
# directly comparable to the values plotted here. It is NOT the number to use on
# a monomorphic landscape probe (fixed cohort, 1 map, reproduction off); that
# scale needs the k conversion from the Stage-9 calibration (threshold 4.798 at
# the measured k = 1.097). Do not carry this constant into landscape/ code.
#
# It is a LOWER bound: the gate is isClear() && canAddOrganism() && rand() < 0.8,
# and the first two also fail in a crowded world, so real replacement costs more.
VIABILITY_THRESHOLD = 3.5 / 0.8   # 4.375
THRESHOLD_COLOUR = '#6B7280'
# Neutral for the per-seed strip: seeds are told apart by POSITION, not hue (10
# seeds is past the point where categorical colour stays readable).
NEUTRAL = '#6B7280'

# Metric column -> (axis label, per-value format).
METRICS = {
    'avg_fitness':          ('Average fitness',      '{:.3f}'),
    'top20percent_fitness': ('Top-20% fitness',      '{:.3f}'),
    'best_fitness':         ('Best fitness',         '{:.3f}'),
    'total_agents':         ('Total agents',         '{:.1f}'),
    'peak_population':      ('Peak population',      '{:.1f}'),
    'genome_variance':      ('Genome variance',      '{:.5f}'),
    'avg_lifetime':         ('Average lifetime',     '{:.1f}'),
    'avg_energy_end':       ('Average end energy',   '{:.3f}'),
    # Labels identical to the learning script so a side-by-side pair of figures
    # shares its axis text as well as its colours.
    'avg_network_weight_mag':  ('Genome weight magnitude (RMS)', '{:.4f}'),
    'inter_gen_weight_change': ('Δ magnitude per generation', '{:.5f}'),
}
FITNESS_METRICS = ['avg_fitness', 'top20percent_fitness', 'best_fitness']
POP_METRICS = ['total_agents', 'peak_population']
VARIANCE_METRIC = 'genome_variance'
# The genome's weight SCALE, as a level and as that level's rate of change — and
# the order matters, because the second is the derivative of the first and is
# easy to misread alone:
#   avg_network_weight_mag   RMS of the genome weights, ||w||/sqrt(N): the typical
#                            size of one heritable weight. Starts at the Xavier
#                            init value (~0.106 at h128) and is capped at 1.0 by
#                            the mutation clamp in GAManager._mutate.
#   inter_gen_weight_change  the FIRST DIFFERENCE of the line above — a RATE, not
#                            a magnitude. It decaying to 0 means the weight scale
#                            has PLATEAUED, NOT that the weights vanished.
#                            Cumulatively summing it reproduces the level exactly,
#                            which is why it is drawn beside the level, never
#                            alone.
# avg_learned_weight_diff is deliberately absent: it is the mean absolute weight difference between an agent's
# active and genome weights, and _calcDrift returns 0 whenever the two are the
# same object — which is exactly the evolution condition. It is identically 0.0
# across all 100 runs and every generation, so it would plot as a flat zero line.
# The learning arm has it as a third panel in its own weight-space figure.
WEIGHT_METRICS = ['avg_network_weight_mag', 'inter_gen_weight_change']
# Everything carried in the per-run endpoint table / curves CSV.
TABLE_METRICS = FITNESS_METRICS + POP_METRICS + [VARIANCE_METRIC,
                                                 'avg_lifetime', 'avg_energy_end'] + WEIGHT_METRICS


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def replicate_index(run_dir):
    """Replicate number parsed off the folder name (…_r7 / …_rep7). Used only to
    ORDER the runs of a seed so `--reps N` takes a stable "first N". Folders with
    no replicate suffix sort first (index 0) and then by name."""
    m = re.search(r'_r(?:ep)?(\d+)$', os.path.basename(run_dir))
    return int(m.group(1)) if m else 0


def discover(logs_root, want, seeds=None, reps=None):
    """Every run under logs_root with params.json + generations.csv that matches
    the condition/environment filters. params.json is authoritative — folder
    names are only a hint. Returns a list of dicts sorted by (seed, rep):
    {seed, rep, name, dir, gen_csv}.

    `want` keys: condition (exact), roaming (tol 1e-9), drain (tol 1e-9, skipped
    if None). `seeds` keeps only those map seeds; `reps` keeps only the first N
    runs of each seed."""
    runs = []
    for params_path in glob.glob(os.path.join(logs_root, '**', 'params.json'),
                                 recursive=True):
        run_dir = os.path.dirname(params_path)
        gen_csv = os.path.join(run_dir, 'generations.csv')
        if not os.path.exists(gen_csv):
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
        except (ValueError, OSError):
            continue
        if str(p.get('condition')) != want['condition']:
            continue
        if abs(_num(p, 'roaming_predator_count') - want['roaming']) > 1e-9:
            continue
        if want.get('drain') is not None and abs(_num(p, 'predator_drain') - want['drain']) > 1e-9:
            continue
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        if seeds is not None and seed not in seeds:
            continue
        runs.append({'seed': seed, 'rep': replicate_index(run_dir),
                     'name': os.path.basename(run_dir), 'dir': run_dir,
                     'gen_csv': gen_csv})

    runs.sort(key=lambda r: (r['seed'], r['rep'], r['name']))
    if reps is not None:
        kept, seen = [], {}
        for r in runs:
            n = seen.get(r['seed'], 0)
            if n < reps:
                kept.append(r)
                seen[r['seed']] = n + 1
        runs = kept
    return runs


def load_run(gen_csv, max_gen, final_window):
    """One run's generations.csv -> (per-generation frame indexed by generation,
    {metric: converged value}). The converged value is the mean of the last
    `final_window` generations (window 1 = the literal last generation), which
    rides out the single-generation noise in the fitness metrics."""
    df = pd.read_csv(gen_csv)
    df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
    df = df.dropna(subset=['generation'])
    if max_gen:
        df = df[df['generation'] <= max_gen]
    df = df.sort_values('generation').set_index('generation')

    present = [m for m in TABLE_METRICS if m in df.columns]
    for m in present:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    tail = df.tail(final_window)
    finals = {m: float(tail[m].mean()) for m in present}
    return df[present], finals


def load_all(runs, max_gen, final_window):
    """Load every run once -> (per-run endpoint table, {run key: frame}). The run
    key is (seed, rep, name) so a run stays addressable and nothing is silently
    merged."""
    rows, curves = [], {}
    for r in runs:
        df, finals = load_run(r['gen_csv'], max_gen, final_window)
        key = (r['seed'], r['rep'], r['name'])
        curves[key] = df
        row = {'seed': r['seed'], 'rep': r['rep'], 'run': r['name'],
               'n_gens': int(df.index.max()) if len(df) else 0}
        row.update(finals)
        rows.append(row)
    return pd.DataFrame(rows), curves


def unit_series(curves, metric, mode):
    """The per-run curves regrouped into the UNITS the band is measured over.

    mode 'runs'  -> one unit per run; the band is the run-to-run spread.
    mode 'seeds' -> replicates are averaged into one curve per seed first, so the
                    band is the between-seed (terrain) spread.

    Returns a list of generation-indexed Series."""
    by_seed = {}
    for (seed, _rep, _name), df in curves.items():
        if metric in df.columns:
            by_seed.setdefault(seed, []).append(df[metric])
    if mode == 'runs':
        return [s for series in by_seed.values() for s in series]
    return [pd.concat(series, axis=1).mean(axis=1) for series in by_seed.values()]


def band_stats(series, band, smooth):
    """Centre line + band over a list of generation-indexed series, aligned on
    generation and optionally smoothed. band 'std'/'sem' centre on the MEAN,
    'iqr' centres on the MEDIAN with the 25-75% range. Returns
    (generations, centre, lo, hi, n)."""
    wide = pd.concat(series, axis=1).sort_index()
    n = int(wide.notna().sum(axis=1).max())
    if band == 'iqr':
        centre, lo, hi = wide.median(axis=1), wide.quantile(0.25, axis=1), wide.quantile(0.75, axis=1)
    else:
        centre = wide.mean(axis=1)
        spread = wide.std(axis=1).fillna(0)
        if band == 'sem':
            spread = spread / max(np.sqrt(n), 1.0)
        lo, hi = centre - spread, centre + spread
    if smooth > 1:
        roll = lambda s: s.rolling(smooth, min_periods=1, center=True).mean()
        centre, lo, hi = roll(centre), roll(lo), roll(hi)
    return wide.index.values, centre.values, lo.values, hi.values, n


def band_label(band, mode):
    """What the shaded region actually is, spelled out for the legend."""
    unit = 'runs' if mode == 'runs' else 'seeds'
    if band == 'iqr':
        return f'IQR across {unit}'
    if band == 'sem':
        return f'±1 s.e.m. across {unit}'
    return f'±1 std across {unit}'


def centre_label(band):
    return 'Median' if band == 'iqr' else 'Mean'


def style_axes(ax):
    """Recessive grid + axes so the data lines carry the figure."""
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def draw_metric(ax, curves, metric, mode, band, smooth, show_runs, legend=True):
    """One metric over generations: every run faint behind the centre line and
    its band. Returns the unit count, or None if the metric is missing."""
    series = unit_series(curves, metric, mode)
    if not series:
        return None
    colour = METRIC_COLOURS.get(metric, INK)
    label, _fmt = METRICS.get(metric, (metric, '{:.3f}'))

    if show_runs:
        raw = unit_series(curves, metric, 'runs')
        # Alpha scaled to the pile depth so 10 runs and 100 runs both read as a
        # cloud rather than a blob or a stray line.
        alpha = float(np.clip(3.0 / max(len(raw), 1), 0.03, 0.35))
        for s in raw:
            # Smoothed with the same window as the mean: a single run swings
            # hard generation to generation, and 100 raw runs overplot into a
            # vertical hash that hides the very band they are drawn behind.
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
            ax.plot(s.index.values, s.values, color=colour, lw=0.5, alpha=alpha,
                    zorder=1)

    x, centre, lo, hi, n = band_stats(series, band, smooth)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.22, lw=0, zorder=2,
                    label=band_label(band, mode))
    ax.plot(x, centre, color=colour, lw=2.0, zorder=3,
            label=f'{centre_label(band)} of {n} {"runs" if mode == "runs" else "seeds"}')
    ax.set_xlabel('Generation')
    ax.set_ylabel(label)
    style_axes(ax)
    if legend:
        leg = ax.legend(frameon=False, fontsize=8.5, loc='lower right')
        for t in leg.get_texts():
            t.set_color(INK)
    return n


def plot_curve_figure(curves, metrics, mode, band, smooth, show_runs, title,
                      out_path):
    """A row of over-generation panels, one per metric. Panels share the x axis
    only — the metrics are different measurements, so each keeps its own y."""
    metrics = [m for m in metrics if any(m in df.columns for df in curves.values())]
    if not metrics:
        print(f'  (skipped {os.path.basename(out_path)}: no data)')
        return
    fig, axs = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.6),
                            sharex=True)
    axs = np.atleast_1d(axs)
    for ax, metric in zip(axs, metrics):
        draw_metric(ax, curves, metric, mode, band, smooth, show_runs)
        ax.set_title(METRICS.get(metric, (metric, ''))[0], color=INK, fontsize=11)
    fig.suptitle(title, color=INK, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def strip(ax, groups, colours, ylabel, xlabel, rng, point_size=16,
          threshold=None):
    """A jittered strip of one dot per run, one column per group, with the
    group's mean drawn as a wide crossbar and +/-1 std as a thin whisker.

    `groups` is a list of (column label, values array). The dots ARE the data —
    no box, no violin, nothing smoothed between them.

    `threshold` draws the replacement line (see VIABILITY_THRESHOLD). It is only
    meaningful on a fitness axis — never pass it for a population or variance
    panel, where the value has no interpretation."""
    for i, (name, vals) in enumerate(groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        colour = colours[i % len(colours)]
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=point_size,
                   color=colour, alpha=0.55, linewidths=0.6, edgecolors='white',
                   zorder=3)
        m = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ax.plot([i - 0.30, i + 0.30], [m, m], color=colour, lw=2.4, zorder=4,
                solid_capstyle='round')
        ax.plot([i, i], [m - sd, m + sd], color=colour, lw=1.2, alpha=0.9, zorder=4)
        # The summary is spelled out in ink, never left to the dot colour.
        ax.annotate(f'{m:.2f}', xy=(i + 0.33, m), fontsize=8, color=INK,
                    va='center', ha='left', zorder=5)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(groups) - 0.15)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    style_axes(ax)
    ax.grid(axis='x', visible=False)

    if threshold is not None:
        # Drawn UNDER the dots and outside the autoscale: if every run sits far
        # above replacement the line should not rescale the axis and squash the
        # spread that the figure exists to show.
        lo, hi = ax.get_ylim()
        ax.axhline(threshold, color=THRESHOLD_COLOUR, lw=1.1, ls='--',
                   alpha=0.9, zorder=1)
        if lo <= threshold <= hi:
            ax.annotate(f'{threshold:.3f}',
                        xy=(ax.get_xlim()[0], threshold), fontsize=7.5,
                        color=THRESHOLD_COLOUR, va='bottom', ha='left',
                        xytext=(3, 2), textcoords='offset points', zorder=2)
        else:
            # Off-scale: say so in the corner rather than silently dropping it.
            where = 'below' if threshold < lo else 'above'
            ax.annotate(f'{threshold:.3f} is {where} this axis',
                        xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7.5,
                        color=THRESHOLD_COLOUR, va='bottom', ha='left')
        ax.set_ylim(lo, hi)



def seed_groups(table, metric):
    """(label, values) columns for the by-seed strip: every run pooled first,
    then one column per map seed."""
    seeds = sorted(table['seed'].unique())
    groups = [(f'all\n({len(table)} runs)', table[metric].values)]
    groups += [(str(s), table.loc[table['seed'] == s, metric].values) for s in seeds]
    return groups


def seed_colours(n_seeds):
    """Pooled column wears the metric's hue; the seeds are neutral — they are
    told apart by POSITION, since 10 categories is well past the point where
    categorical colour stays separable."""
    return [METRIC_COLOURS['avg_fitness']] + [NEUTRAL] * n_seeds


def plot_strip_figure(table, final_window, out_path, seed=0):
    """Converged fitness, one dot per run.

    Top row: each fitness metric pooled over every run, on its OWN axis — avg,
    top-20% and best fitness differ by an order of magnitude, so a shared scale
    would flatten the average-fitness spread into a line.
    Bottom row: average fitness split by map seed, so terrain effects can be
    told apart from run-to-run noise."""
    rng = np.random.default_rng(seed)
    have = [m for m in FITNESS_METRICS if m in table.columns]
    if not have:
        print(f'  (skipped {os.path.basename(out_path)}: no fitness columns)')
        return
    n_seeds = table['seed'].nunique()

    fig = plt.figure(figsize=(4.4 * len(have), 8.4))
    gs = fig.add_gridspec(2, len(have), height_ratios=[1.0, 1.05], hspace=0.34)
    for i, metric in enumerate(have):
        ax = fig.add_subplot(gs[0, i])
        strip(ax, [('all runs', table[metric].values)], [METRIC_COLOURS[metric]],
              METRICS[metric][0], '', rng, point_size=20,
              threshold=VIABILITY_THRESHOLD)
        ax.set_title(METRICS[metric][0], color=INK, fontsize=11)

    ax = fig.add_subplot(gs[1, :])
    strip(ax, seed_groups(table, 'avg_fitness'), seed_colours(n_seeds),
          'Average fitness', 'Map seed', rng, threshold=VIABILITY_THRESHOLD)
    ax.set_title('Average fitness by map seed — terrain effect vs run-to-run noise',
                 color=INK, fontsize=11)

    window = ('last generation' if final_window == 1
              else f'mean of last {final_window} generations')
    fig.suptitle(f'Final fitness per run ({window}) — one dot per run, '
                 f'bar = mean, whisker = ±1 std',
                 color=INK, fontsize=12.5, y=0.985)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_overview(curves, table, mode, band, smooth, show_runs, final_window,
                  title, out_path, seed=0):
    """The headline figure: mean fitness, mean population, genome variance and
    the final-fitness strip in one 2x2."""
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.0))
    draw_metric(axs[0, 0], curves, 'avg_fitness', mode, band, smooth, show_runs)
    axs[0, 0].set_title('Average fitness', color=INK, fontsize=11)
    draw_metric(axs[0, 1], curves, 'total_agents', mode, band, smooth, show_runs)
    axs[0, 1].set_title('Population (total agents)', color=INK, fontsize=11)
    draw_metric(axs[1, 0], curves, VARIANCE_METRIC, mode, band, smooth, show_runs)
    axs[1, 0].set_title('Genome variance', color=INK, fontsize=11)

    strip(axs[1, 1], seed_groups(table, 'avg_fitness'),
          seed_colours(table['seed'].nunique()),
          'Average fitness', 'Map seed', rng, threshold=VIABILITY_THRESHOLD)
    window = ('last generation' if final_window == 1
              else f'mean of last {final_window} gens')
    axs[1, 1].set_title(f'Final average fitness per run ({window})',
                        color=INK, fontsize=11)

    fig.suptitle(title, color=INK, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def curves_table(curves, mode, band):
    """The mean curves as a tidy frame: generation, metric, centre, lo, hi, std,
    sem, n — so the figures can be redrawn or re-tested without re-reading 100
    runs. UNSMOOTHED, whatever --smooth the figures were drawn with: smoothing
    is a legibility choice for the plot, not a property of the data."""
    frames = []
    for metric in TABLE_METRICS:
        series = unit_series(curves, metric, mode)
        if not series:
            continue
        wide = pd.concat(series, axis=1).sort_index()
        n = wide.notna().sum(axis=1)
        x, centre, lo, hi, _ = band_stats(series, band, 1)
        frames.append(pd.DataFrame({
            'generation': x, 'metric': metric, 'centre': centre,
            'lo': lo, 'hi': hi,
            'mean': wide.mean(axis=1).values, 'std': wide.std(axis=1).values,
            'sem': wide.sem(axis=1).values, 'n': n.values,
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['band'] = band
    out['unit'] = mode
    return out


def report(table, final_window):
    """Console census: what was found, and where the runs landed."""
    print(f'\nRuns: {len(table)} across {table["seed"].nunique()} map seeds')
    per_seed = table.groupby('seed').size()
    print('  replicates per seed: ' +
          ', '.join(f'{s}:{n}' for s, n in per_seed.items()))
    gens = table['n_gens']
    print(f'  generations per run: min {gens.min()}, max {gens.max()}')
    window = 'last generation' if final_window == 1 else f'last {final_window} generations'
    print(f'\nConverged values ({window}) across runs.')
    # Median and quartiles alongside the mean, because a mean +/- std hides a
    # SPLIT distribution: if some runs stall low while the rest climb, the mean
    # sits between two clusters where no run actually is, and median >> mean is
    # the tell.
    print(f'  {"metric":<22} {"mean":>10} {"std":>10} {"median":>10} '
          f'{"q25":>10} {"q75":>10} {"min":>10} {"max":>10}')
    for m in TABLE_METRICS:
        if m not in table.columns:
            continue
        label, fmt = METRICS.get(m, (m, '{:.3f}'))
        v = table[m].dropna()
        if not len(v):
            continue
        cells = [v.mean(), v.std(), v.median(), v.quantile(0.25),
                 v.quantile(0.75), v.min(), v.max()]
        print(f'  {label:<22} ' + ' '.join(f'{fmt.format(c):>10}' for c in cells))


def main():
    ap = argparse.ArgumentParser(
        description='Mean curves and per-run endpoints for the evolution '
                    'campaign on the hard (roaming-predator) environment.')
    ap.add_argument('--logs-root', default='logs/evolution',
                    help='Root to search recursively for run folders. '
                         'Default: logs/evolution.')
    ap.add_argument('--out', default=None,
                    help='Output directory. Default: output/evolution_<env>_runs.')
    ap.add_argument('--env', choices=list(ENVIRONMENTS), default='hard',
                    help='Environment: hard (60 roaming predators, drain 5) or '
                         'baseline (no predators). Default: hard.')
    ap.add_argument('--condition', default='evolution',
                    help='params.json condition to match. Default: evolution.')
    ap.add_argument('--predators', type=float, default=60.0,
                    help='roaming_predator_count for the HARD environment '
                         '(default 60; ignored for baseline).')
    ap.add_argument('--drain', type=float, default=5.0,
                    help='predator_drain for the HARD environment (default 5; '
                         'ignored for baseline).')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Only these map seeds (default: all found).')
    ap.add_argument('--reps', type=int, default=None,
                    help='Keep only the first N replicates of each seed.')
    ap.add_argument('--max-gen', type=int, default=None,
                    help='Truncate every run at this generation (fair '
                         'head-to-head when run lengths differ).')
    ap.add_argument('--final-window', type=int, default=50,
                    help='"Final" value = mean of the last N generations '
                         '(default 50; use 1 for the literal last generation).')
    ap.add_argument('--aggregate', choices=['runs', 'seeds'], default='runs',
                    help='Unit the band is measured over: runs (run-to-run '
                         'noise, default) or seeds (between-terrain spread).')
    ap.add_argument('--band', choices=['std', 'sem', 'iqr'], default='std',
                    help='Shaded region: +/-1 std (default), +/-1 s.e.m., or '
                         'median with the 25-75%% range.')
    ap.add_argument('--smooth', type=int, default=11,
                    help='Rolling-mean window over generations for the drawn '
                         'curves, mean and individual runs alike (default 11; '
                         '1 = raw, which at 1000 generations is legible for the '
                         'mean but not for 100 overplotted runs). Cosmetic '
                         'only — the CSVs and the endpoint table are unsmoothed.')
    ap.add_argument('--no-runs', action='store_true',
                    help='Hide the faint individual run curves behind the mean.')
    ap.add_argument('--no-weight-space', action='store_true',
                    help='Skip the genome weight-scale figure.')
    ap.add_argument('--jitter-seed', type=int, default=0,
                    help='RNG seed for the strip-plot jitter (default 0, so the '
                         'figure is reproducible).')
    args = ap.parse_args()

    env = ENVIRONMENTS[args.env]
    want = {'condition': args.condition}
    if args.env == 'hard':
        want['roaming'], want['drain'] = args.predators, args.drain
    else:
        want['roaming'], want['drain'] = env['roaming'], env['drain']

    out_dir = args.out or f'output/evolution_{args.env}_runs'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Searching {args.logs_root} for condition={args.condition!r} '
          f'in the {args.env} environment ({env["desc"]}) ...')
    runs = discover(args.logs_root, want, set(args.seeds) if args.seeds else None,
                    args.reps)
    if not runs:
        print('No matching runs found. Check --logs-root / --env / --condition.')
        return

    print(f'Loading {len(runs)} runs ...')
    table, curves = load_all(runs, args.max_gen, args.final_window)
    report(table, args.final_window)

    runs_csv = os.path.join(out_dir, f'evolution_{args.env}_runs.csv')
    table.to_csv(runs_csv, index=False)
    print(f'\n  wrote {runs_csv}')

    tidy = curves_table(curves, args.aggregate, args.band)
    curves_csv = os.path.join(out_dir, f'evolution_{args.env}_curves.csv')
    tidy.to_csv(curves_csv, index=False)
    print(f'  wrote {curves_csv}')

    unit = 'runs' if args.aggregate == 'runs' else 'map seeds'
    n_units = len(table) if args.aggregate == 'runs' else table['seed'].nunique()
    subtitle = (f'{args.condition} · {env["desc"]} · {len(table)} runs '
                f'({table["seed"].nunique()} seeds) · band = {band_label(args.band, args.aggregate)}')
    show_runs = not args.no_runs

    plot_curve_figure(curves, FITNESS_METRICS, args.aggregate, args.band,
                      args.smooth, show_runs,
                      f'Fitness over generations — {subtitle}',
                      os.path.join(out_dir, f'evolution_{args.env}_fitness.png'))
    plot_curve_figure(curves, POP_METRICS, args.aggregate, args.band,
                      args.smooth, show_runs,
                      f'Population over generations — {subtitle}',
                      os.path.join(out_dir, f'evolution_{args.env}_population.png'))
    plot_curve_figure(curves, [VARIANCE_METRIC], args.aggregate, args.band,
                      args.smooth, show_runs,
                      f'Genome variance over generations — {subtitle}',
                      os.path.join(out_dir, f'evolution_{args.env}_genome_variance.png'))
    plot_strip_figure(table, args.final_window,
                      os.path.join(out_dir, f'evolution_{args.env}_final_fitness_strip.png'),
                      args.jitter_seed)
    plot_overview(curves, table, args.aggregate, args.band, args.smooth,
                  show_runs, args.final_window,
                  f'Evolution on the {args.env} environment — {subtitle}',
                  os.path.join(out_dir, f'evolution_{args.env}_overview.png'),
                  args.jitter_seed)
    if not args.no_weight_space:
        plot_curve_figure(curves, WEIGHT_METRICS, args.aggregate, args.band,
                          args.smooth, show_runs,
                          f'Genome weight scale: level and its rate of change '
                          f'— {subtitle}',
                          os.path.join(out_dir, f'evolution_{args.env}_weight_space.png'))

    print(f'\nDone — {n_units} {unit} aggregated into {out_dir}')


if __name__ == '__main__':
    main()
