"""
analyse_conditions_by_environment.py
======================================================================
The THREE conditions in BOTH environments, as individual per-metric figures.

Every other comparison script in this repo draws one multi-panel sheet per
campaign. This one is built for a thesis figure directory instead: one PNG per
metric per environment, so a figure can be dropped into the write-up without
cropping a panel out of a grid, plus a `compare/` set that puts the two
environments side by side for the same metric.

    run_evolution_condition_*_array.slurm    GA only          -> evolution
    run_learning_condition_*_array.slurm     GA + in-life RL  -> learning
    run_pure_rl_condition_*_array.slurm      RL only, no GA   -> pure_rl

Six cells (3 conditions x 2 environments), 100 runs each, discovered from
params.json and never from folder names. Pure RL is logged with
condition == "learning" (its job script passes --condition learning
--mode pure_rl), so MODE is what separates the two RL arms; matching on the
condition string alone would silently pool 200 runs into one curve. Random-floor
runs masquerade as evolution runs and are excluded by discover()'s default.

FIGURE STYLE — matched to output/hard_conditions_combined
----------------------------------------------------------
Every curve figure carries: a smoothed mean line per condition, a shaded +/-1
std band behind it, and the mean +/- 1 std reported in a legend block OUTSIDE the
axes, one row per condition with its n. The plot area then holds nothing but the
lines it exists to compare, and the number a reader would quote is on the figure
rather than in a separate table.

WHAT THE +/- IN THE LEGEND IS (and is not)
-------------------------------------------
It is the spread ACROSS RUNS, n = 100 per cell — not a spread over generations.
Each run is first reduced to ONE number (its converged value: the mean over the
last --final-window generations, default 50), and the legend reports the mean
and standard deviation of those 100 per-run numbers. `n = 100` in the legend is
the run count, which is the giveaway.

The legend therefore says only "converged mean +/- 1 std across runs". What
"converged" means is defined once in the subtitle, with the other method notes,
rather than restated in the legend where it reads as a qualifier on the +/-.

Pooling every generation of every run into one standard deviation instead would
be a worse statistic, not a broader one: it mixes the transient climb with the
converged plateau, so it is dominated by how far the arm rose from generation 1
rather than by how much runs disagree at the end. The transient IS visible — it
is the shaded band, which is the same across-run spread computed separately at
every generation. Converged spread in the legend, per-generation spread in the
band: the two answer different questions and neither replaces the other.

The 50-generation window only defines each run's converged value; it rides out
single-generation noise without touching what the +/- is measured over. Widen or
narrow it with --final-window.

Curves are SMOOTHED by default (--smooth 15, a centred rolling mean over
generations). Per-generation fitness and population are noisy enough that an
unsmoothed mean line reads as a fuzzy band rather than a line, which then
collides visually with the actual std band and makes the spread unreadable.
Smoothing is applied identically to the centre and to both band edges, so the
band still shows the run-to-run spread and not the smoother's residual. Pass
--smooth 1 for the raw per-generation values.

WHAT IS WRITTEN
---------------
  <out>/baseline/ and <out>/hard/   one PNG per metric, three condition lines
      fitness_avg_normalised.png        \
      fitness_top20_normalised.png       | min-max rescaled, see NORMALISATION
      fitness_best_normalised.png       /
      fitness_avg_raw.png               raw units + the viability threshold
      strip_final_fitness_raw.png       one dot per run, converged fitness
      strip_final_fitness_normalised.png
      weight_difference.png             avg_learned_weight_diff (0 in evolution)
      weight_magnitude.png              avg_network_weight_mag (RMS)
      genome_variance.png               GA diversity
      population_total_agents.png
      population_peak.png
      behaviour_*.png                   from organisms.csv, see BEHAVIOUR

  <out>/compare/   the same metric, baseline LEFT and hard RIGHT, shared y-axis
      compare_<metric>.png

  <out>/fitness_summary.csv one row per environment x condition x fitness
                            metric: n, mean, std, sem, 95% CI, median, quartiles,
                            IQR, min, max — each also in normalised units — and
                            how many runs converged below the viability
                            threshold. This is the table a thesis fitness table
                            is built from.
  <out>/runs.csv            one row per RUN — every converged value behind the
                            aggregates, so any mean or std above can be
                            recomputed and an outlier traced to its seed.
  <out>/summary.csv         the same aggregate for ALL 19 metrics (a lookup
                            table rather than a presentation one).
  <out>/time_to_threshold.csv
                            per run: the generation at which it first held
                            viable fitness for --cross-sustain consecutive
                            generations, right-censored if it never did.
  <out>/time_to_threshold_km.csv / _cox.csv / _logrank.csv
                            Kaplan-Meier medians, the seed-clustered Cox model
                            and the omnibus log-rank test.
  <out>/variance_components.csv
                            sd between seeds (terrain) and sd within seed
                            (run-to-run chance) per cell, with the ICC they
                            produce — the table to read ICC beside.
  <out>/paired_by_seed.csv  condition-vs-condition paired on the ten shared map
                            seeds: mean difference, 95% CI, Cohen's dz, paired
                            t, Wilcoxon and Holm-adjusted p.
  <out>/bootstrap_ci.csv    seed-clustered bootstrap CIs for the mean AND the
                            median (10 000 resamples of whole seeds).
  <out>/collapse_rate.csv   share of runs below the viability threshold, with
                            Wilson and seed-clustered bootstrap intervals.
  <out>/normalisation.csv   the rescaling constants, per fitness metric

  <out>/.cache_values/<env>__<condition>/
                            every number the figures and the tests above are
                            built from, stored per cell so they survive the logs
                            being transferred away. See THE VALUE CACHE.

THE VALUE CACHE — ONE ENVIRONMENT AT A TIME
--------------------------------------------
The two log roots are tens of GB each and do not fit on one disk together, so
the campaign is analysed in two passes. Every pass writes .cache_values for the
cells it loaded, and any cell with no logs on disk is drawn and tested from that
cache instead of being dropped:

    # 1. predator logs on disk, baseline not yet transferred
    python analyse_conditions_by_environment.py

    # 2. swap the logs: remove logs_hard/, put the baseline under logs/
    python analyse_conditions_by_environment.py

Pass 2 loads baseline from its logs and hard from .cache_values, and produces the
same figures, the same normalisation and the same tables as a pass with both log
roots present. Verified by running it both ways: every figure is pixel-identical
and every table agrees to float round-trip (< 5e-13).

The cache holds per-generation mean/std/median/quartiles/n for every metric (so
--band and --aggregate still work, and --smooth is still applied at draw time),
the per-run converged table every significance test consumes, the min/max extent
that keeps both environments on ONE normalised axis, and the time-to-viability
event table. What it cannot do is change --final-window, --max-gen, --reps or
--seeds after the fact: those choose what went into the stored numbers, so
meta.json stamps them and a mismatched cache is refused out loud rather than
mixed in. --no-cache turns the whole mechanism off.

NORMALISATION — ONE RANGE PER METRIC, SHARED BY EVERY CONDITION *AND* BOTH
ENVIRONMENTS
--------------------------------------------------------------------------
Fitness is rescaled by an affine map

    z = (x - lo) / (hi - lo)

where lo/hi are the min/max of that metric over EVERY generation of EVERY run of
ALL THREE conditions in BOTH environments (--norm-scope global, the default).

The scope is the whole decision. Normalising each environment against its own
range would map both to the same endpoints and delete the very comparison the
side-by-side figures exist to make: a hard-environment arm that converges lower
than every baseline arm would still touch 1.0 on its own panel. A global range
keeps the two environments on one ruler, so the vertical gap between the left
and right panels is real. The cost is that neither environment necessarily spans
the full [0, 1] — which is the honest picture, not a defect.

`--norm-scope env` is offered for the case where only within-environment shape
matters, and it is deliberately NOT the default. Whichever is used, the constants
land in normalisation.csv and are stamped on the figure subtitle, because a
normalised axis with an unstated range is not interpretable.

Each fitness METRIC gets its own lo/hi. best_fitness runs well above avg_fitness,
so one shared range across metrics would compress avg_fitness into the bottom of
the axis. Per-metric ranges keep every panel occupying [0, 1] while staying
comparable across the conditions and environments that panel compares.

Only fitness is normalised. Population, genome variance, the weight metrics and
the behaviours are left in their own units — they are not being compared against
a different-scaled twin, and rescaling them would only hide their magnitudes.

BEHAVIOUR — READ FROM organisms.csv, AND CACHED
------------------------------------------------
generations.csv carries no behavioural columns. Cells explored, predator
encounters, cave use, diet composition and death cause are per-ORGANISM fields in
organisms.csv, which runs to ~120 MB per run (~60 GB across the 600 runs). Each
run is therefore parsed ONCE with usecols, reduced to a per-generation mean, and
cached under <out>/.cache_behaviour/. The first pass costs a few minutes; every
later pass is instant. Delete the cache directory to force a re-read.

Two death-cause FRACTIONS are derived per generation (share of that generation's
organisms killed by predation / by starvation) so the composition fits the same
one-PNG-per-metric layout as everything else rather than needing stacked areas.

Usage
-----
    python analyse_conditions_by_environment.py
    python analyse_conditions_by_environment.py --smooth 1        # raw, unsmoothed
    python analyse_conditions_by_environment.py --band sem
    python analyse_conditions_by_environment.py --no-behaviour    # skip organisms.csv
    python analyse_conditions_by_environment.py --behaviour-reps 3
    python analyse_conditions_by_environment.py --norm-scope env
"""

import argparse
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# The single-arm script is the source of truth for discovery, loading and the
# band statistics; importing it keeps these figures defined on exactly the same
# numbers as the per-arm ones rather than a second implementation of them.
import analyse_learning_hard_runs as la


# ── The three conditions ──────────────────────────────────────────────────────
# Colour triple is the validated one from analyse_hard_conditions_combined
# (worst all-pairs dE 14.1 under deuteranopia). Line style is redundant with
# colour ON PURPOSE: three overlapping bands is exactly the case where a reader
# who cannot separate two hues has nothing else to go on.
CONDITIONS = [
    {'key': 'evolution', 'label': 'Evolution (GA only)',
     'condition': 'evolution', 'mode': 'standard',
     'colour': '#D97706', 'ls': '--'},
    {'key': 'learning', 'label': 'Learning (GA + in-life RL)',
     'condition': 'learning', 'mode': 'standard',
     'colour': '#2563EB', 'ls': '-'},
    {'key': 'pure_rl', 'label': 'Pure RL (no GA)',
     'condition': 'learning', 'mode': 'pure_rl',
     'colour': '#DB2777', 'ls': '-.'},
]

# `key` is an IDENTIFIER — it names the output folder, the discover() filter and
# the environment column of every CSV, so it stays 'hard' and existing tables
# and paths keep working. `label`/`short` are what appears on figures.
ENVIRONMENTS = [
    {'key': 'baseline', 'root': 'logs',
     'label': 'Baseline (predator-free)', 'short': 'baseline'},
    {'key': 'hard', 'root': 'logs_hard',
     'label': 'Predator environment (60 roaming, drain 5)',
     'short': 'predator environment'},
]

FITNESS_METRICS = ['avg_fitness', 'top20percent_fitness', 'best_fitness']

# (metric, file stem, normalise?) — read from generations.csv.
GEN_FIGURES = [
    ('avg_fitness',             'fitness_avg_normalised',   True),
    ('top20percent_fitness',    'fitness_top20_normalised', True),
    ('best_fitness',            'fitness_best_normalised',  True),
    ('avg_fitness',             'fitness_avg_raw',          False),
    ('avg_learned_weight_diff', 'weight_difference',        False),
    ('avg_network_weight_mag',  'weight_magnitude',         False),
    ('genome_variance',         'genome_variance',          False),
    ('total_agents',            'population_total_agents',  False),
    ('peak_population',         'population_peak',          False),
    ('avg_lifetime',            'lifetime',                 False),
]

# ── Behaviour, from organisms.csv ─────────────────────────────────────────────
# Per-organism fields averaged over each generation's organisms. predator_touches
# and drained_ticks are identically 0 in the baseline environment (no predators),
# which is the correct reading and worth seeing beside the hard panel rather than
# suppressing.
BEHAVIOUR_METRICS = {
    'cells_visited':      'Cells explored (mean per organism)',
    'predator_touches':   'Predator encounters (mean per organism)',
    'drained_ticks':      'Ticks drained (mean per organism)',
    'cave_entries':       'Cave entries (mean per organism)',
    'cave_entries_night': 'Cave entries at night (mean per organism)',
    'food_prestige':      'Prestige food eaten (mean per organism)',
    'food_low':           'Low-tier food eaten (mean per organism)',
    'energy_at_death':    'Energy at death (mean per organism)',
    'frac_death_drained': 'Fraction of deaths by predation',
    'frac_death_starved': 'Fraction of deaths by starvation',
}
BEHAVIOUR_ORDER = list(BEHAVIOUR_METRICS)
# Columns pulled out of organisms.csv. Anything absent in an older run is simply
# missing from that run's cache and drops out of the mean, rather than failing.
BEHAVIOUR_RAW = ['generation', 'death_cause', 'cells_visited', 'predator_touches',
                 'drained_ticks', 'cave_entries', 'cave_entries_day',
                 'cave_entries_night', 'energy_at_death', 'lifetime',
                 'food_low', 'food_medium', 'food_prestige', 'food_default']

INK = la.INK
GRID = la.GRID
VIABILITY = la.VIABILITY_THRESHOLD   # 4.375

# Output formats and resolution, set once in main() from --formats / --dpi.
#
# PNG IS ALREADY LOSSLESS — it uses DEFLATE, not JPEG's quantisation, so nothing
# is thrown away at any dpi. What limits a PNG is RESOLUTION: at 160 dpi a 11in
# figure is 1760px, which softens when scaled up in a document. Two fixes, and
# they are different things:
#   - raise --dpi (300 is the print standard; the default here)
#   - emit a VECTOR format as well, which has no resolution at all and stays
#     sharp at any zoom. PDF is what a LaTeX thesis wants (\includegraphics
#     takes it directly); SVG suits the web.
# Vector output is small here because these are line plots; the strips carry a
# few hundred scatter points, which is nothing.
OUTPUT = {'formats': ('png', 'pdf'), 'dpi': 300}


# ── Loading: generations.csv ──────────────────────────────────────────────────

def usable(runs, skipped):
    """Drop runs whose generations.csv is empty or unparseable.

    A run that is mid-transfer, or whose logger never flushed, leaves a 0-byte
    generations.csv beside a populated organisms.csv. la.load_all lets pandas
    raise on it, which aborts the whole campaign over one bad file — wrong
    behaviour for a script that gets re-run as data lands. Skipped runs are
    collected and reported rather than silently dropped, because a quietly
    missing run is a quietly wrong n.
    """
    keep = []
    for r in runs:
        try:
            if os.path.getsize(r['gen_csv']) == 0:
                raise ValueError('empty file')
            pd.read_csv(r['gen_csv'], nrows=1)
        except (OSError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError) as exc:
            skipped.append((r['name'], type(exc).__name__))
            continue
        keep.append(r)
    return keep


def load_cells(envs, conditions, max_gen, final_window, seeds, reps,
               cache_root=None, args=None):
    """Discover and load every (environment, condition) cell exactly once.

    Loading is done up front for ALL cells because the global normalisation range
    cannot be known until every run has been read — a two-pass structure is
    unavoidable. Cells with no runs are omitted, not faked.

    A cell with no runs on disk falls back to `cache_root`, so an environment
    whose logs have been transferred away still appears in every figure and
    table built from converged values. The fallback is one-directional: runs on
    disk always win, and a cache that is present but does not match the current
    options is reported rather than used. See the value-cache section.
    """
    cells, skipped = {}, []
    for env in envs:
        spec = la.ENVIRONMENTS[env['key']]
        for cond in conditions:
            want = {'condition': cond['condition'], 'mode': cond['mode'],
                    'roaming': spec['roaming'], 'drain': spec['drain']}
            runs = la.discover(env['root'], want, seeds=seeds, reps=reps)
            found = len(runs)
            runs = usable(runs, skipped)
            if not runs:
                cached, why = ((None, None) if not cache_root else
                               read_cache(cache_root, env['key'], cond['key'],
                                          args, seeds))
                if cached is not None:
                    cells[(env['key'], cond['key'])] = cached
                    print(f'  [cache] {env["key"]:<9} {cond["key"]:<10} '
                          f'{cached["n"]:3d} runs from stored values '
                          f'(no logs under {env["root"]})')
                    continue
                note = f'  [cache rejected: {why}]' if why else ''
                print(f'  [skip] {env["key"]:<9} {cond["key"]:<10} no usable runs under {env["root"]}{note}')
                continue
            table, curves = la.load_all(runs, max_gen, final_window)
            cells[(env['key'], cond['key'])] = {'table': table, 'curves': curves,
                                                'runs': runs, 'n': len(runs)}
            lost = found - len(runs)
            note = f'  ({lost} unreadable, skipped)' if lost else ''
            print(f'  [load] {env["key"]:<9} {cond["key"]:<10} {len(runs):3d} runs{note}')
    if skipped:
        print(f'\n  WARNING: {len(skipped)} run(s) skipped — generations.csv empty or unparseable:')
        for name, why in skipped:
            print(f'    {name}  ({why})')
    return cells


# ── Loading: organisms.csv behaviour, with a cache ────────────────────────────

def behaviour_cache_path(cache_dir, run_dir, org_csv):
    """Cache key = run path + the source file's size and mtime.

    Keying on the stat rather than the path alone means a re-transferred or
    re-run organisms.csv invalidates its own cache entry automatically; a stale
    aggregate is worse than no aggregate.
    """
    st = os.stat(org_csv)
    key = f'{os.path.abspath(run_dir)}|{st.st_size}|{int(st.st_mtime)}'
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + '.csv')


def behaviour_for_run(run, cache_dir, max_gen):
    """Per-generation behaviour means for one run, from cache when possible.

    Returns a generation-indexed frame, or None when the run has no readable
    organisms.csv (which is not fatal — that run simply does not contribute).
    """
    org_csv = os.path.join(run['dir'], 'organisms.csv')
    if not os.path.exists(org_csv) or os.path.getsize(org_csv) == 0:
        return None
    cpath = behaviour_cache_path(cache_dir, run['dir'], org_csv)
    if os.path.exists(cpath):
        try:
            return pd.read_csv(cpath).set_index('generation')
        except (ValueError, OSError, pd.errors.EmptyDataError):
            pass   # fall through and rebuild

    try:
        df = pd.read_csv(org_csv, usecols=lambda c: c in BEHAVIOUR_RAW,
                         low_memory=False)
    except (ValueError, OSError, pd.errors.ParserError,
            pd.errors.EmptyDataError):
        return None
    if 'generation' not in df.columns or df.empty:
        return None
    df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
    df = df.dropna(subset=['generation'])
    if max_gen:
        df = df[df['generation'] <= max_gen]
    if df.empty:
        return None

    # Death cause -> two per-generation fractions, so composition fits the same
    # one-metric-per-figure layout as every scalar.
    if 'death_cause' in df.columns:
        df['frac_death_drained'] = (df['death_cause'] == 'drained').astype(float)
        df['frac_death_starved'] = (df['death_cause'] == 'starved').astype(float)

    numeric = [c for c in df.columns
               if c not in ('generation', 'death_cause')]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = df.groupby('generation')[numeric].mean()

    os.makedirs(cache_dir, exist_ok=True)
    try:
        out.to_csv(cpath)
    except OSError:
        pass
    return out


def load_behaviour(cells, cache_dir, max_gen, behaviour_reps):
    """Attach {'behaviour': {run key: frame}} to every cell.

    behaviour_reps caps how many replicates per seed are parsed — the first pass
    over 600 runs is a few minutes of I/O, so a smaller cap makes a quick look
    cheap without discarding the cache built for the runs it did read.
    """
    total = 0
    for (env_key, cond_key), cell in sorted(cells.items()):
        if cell.get('cached'):
            n_beh = len(cell.get('behaviour_table', ()))
            print(f'  [behaviour] {env_key:<9} {cond_key:<10} cached cell — '
                  f'{n_beh} run(s) of stored values, no organisms.csv read')
            continue
        runs = cell['runs']
        if behaviour_reps is not None:
            seen, capped = {}, []
            for r in runs:
                c = seen.get(r['seed'], 0)
                if c < behaviour_reps:
                    capped.append(r)
                    seen[r['seed']] = c + 1
            runs = capped
        frames, missing = {}, 0
        for i, r in enumerate(runs, 1):
            b = behaviour_for_run(r, cache_dir, max_gen)
            if b is None:
                missing += 1
                continue
            frames[(r['seed'], r['rep'], r['name'])] = b
            if i % 25 == 0 or i == len(runs):
                print(f'\r  [behaviour] {env_key:<9} {cond_key:<10} '
                      f'{i}/{len(runs)} runs', end='', flush=True)
        print()
        if missing:
            print(f'    ({missing} run(s) had no readable organisms.csv)')
        cell['behaviour'] = frames
        total += len(frames)
    return total


# ── Value cache: figures and tests after the logs are gone ───────────────
#
# The two environments' logs are tens of GB each and do not sit on the same disk
# at the same time, so a campaign gets analysed in two passes: one environment
# now, the other after its logs are transferred in. Everything downstream of
# loading reads a cell in one of four ways, and all four reduce to something
# small enough to keep beside the figures:
#
#   curves.csv     per generation x metric x aggregation unit: n, mean, std,
#                  median, q25, q75. That is every band --band can draw ('std'
#                  and 'sem' from mean/std/n, 'iqr' from the quartiles) over
#                  both --aggregate units, stored UNSMOOTHED. Smoothing is a
#                  rolling mean applied at draw time; caching the smoothed line
#                  would freeze --smooth at whatever it was when the logs were
#                  last present.
#   table.csv      one row per run, the converged value of every metric. This is
#                  what every significance test actually consumes — the paired
#                  seed tests, the clustered bootstrap, the collapse rates and
#                  the variance components all start from this table and never
#                  touch a per-generation curve.
#   extent.csv     per metric, the min and max over every generation of every
#                  run. It is the only thing normalisation_ranges() reads the
#                  curves for, so a cached cell still joins the GLOBAL min-max
#                  range and both environments stay on one normalised axis
#                  whether or not their logs were present at the same time.
#   survival.csv   one row per run: the time-to-viability event and its
#                  censoring flag, at the --cross-sustain it was built with.
#   behaviour_table.csv
#                  one row per run, the converged value of each organisms.csv
#                  metric. converged() aggregates these from the per-run frames,
#                  which the cache does not keep.
#
# WHAT THE CACHE DELIBERATELY WILL NOT DO
# ---------------------------------------
# Every stored statistic is conditional on the options that chose which runs and
# which generations went into it: --final-window, --max-gen, --reps, --seeds.
# meta.json stamps all four and read_cache() REFUSES a cell whose stamp differs
# from the current invocation, printing what disagreed, rather than quietly
# pooling two definitions of "converged" into one figure. --cross-sustain is
# stamped the same way and a mismatch drops the survival rows alone, since it
# affects nothing else. Changing any of those options needs the logs back.
#
# A cell whose logs ARE present is always recomputed and never read from cache,
# so a stale cache can only ever be the fallback, never an override.

CACHE_MODES = ('runs', 'seeds')
# The loading options every cached statistic is conditional on.
CACHE_STAMP = ('final_window', 'max_gen', 'reps', 'seeds')


def cache_stamp(args, seeds):
    return {'final_window': args.final_window, 'max_gen': args.max_gen,
            'reps': args.reps, 'seeds': (sorted(seeds) if seeds else None)}


def cache_cell_dir(cache_root, env_key, cond_key):
    return os.path.join(cache_root, f'{env_key}__{cond_key}')


def metric_sources(figures):
    """The (metric, source) pairs to cache, de-duplicated.

    GEN_FIGURES lists avg_fitness twice (normalised and raw); the numbers behind
    the two are identical, so caching it once is enough.
    """
    seen, out = set(), []
    for metric, _stem, _norm, source in figures:
        if (metric, source) not in seen:
            seen.add((metric, source))
            out.append((metric, source))
    return out


def curve_stats(series):
    """Per-generation statistics for one metric, before smoothing.

    Everything la.band_stats() needs: the mean and std that 'std'/'sem' centre
    on, the n that turns one into the other, and the quartiles 'iqr' uses
    instead. n is stored per generation rather than as band_stats's single
    maximum so a cached curve still records where runs ended.
    """
    wide = pd.concat(series, axis=1).sort_index()
    return pd.DataFrame({
        'generation': wide.index.to_numpy(dtype=float),
        'n': wide.notna().sum(axis=1).to_numpy(dtype=int),
        'mean': wide.mean(axis=1).to_numpy(dtype=float),
        'std': wide.std(axis=1).to_numpy(dtype=float),
        'median': wide.median(axis=1).to_numpy(dtype=float),
        'q25': wide.quantile(0.25, axis=1).to_numpy(dtype=float),
        'q75': wide.quantile(0.75, axis=1).to_numpy(dtype=float),
    })


def band_from_cache(df, band, smooth):
    """Cached statistics -> la.band_stats()'s (gens, centre, lo, hi, n).

    Mirrors band_stats line for line, including the max-over-generations n that
    'sem' divides by, so a cached panel and a live one are the same figure.
    """
    gens = df['generation'].to_numpy(dtype=float)
    n = int(df['n'].max()) if len(df) else 0
    if band == 'iqr':
        centre, lo, hi = df['median'], df['q25'], df['q75']
    else:
        centre = df['mean']
        spread = df['std'].fillna(0)
        if band == 'sem':
            spread = spread / max(np.sqrt(n), 1.0)
        lo, hi = centre - spread, centre + spread
    if smooth > 1:
        roll = lambda s: s.rolling(smooth, min_periods=1, center=True).mean()
        centre, lo, hi = roll(centre), roll(lo), roll(hi)
    return (gens, centre.to_numpy(dtype=float), lo.to_numpy(dtype=float),
            hi.to_numpy(dtype=float), n)


def band_for(cell, metric, mode, band, smooth, source='gen'):
    """Centre line + band for one condition, from the logs or from the cache.

    Live curves take precedence: a cell whose logs are present is recomputed, so
    the cache can only ever fill a gap and never overrule the data.
    """
    series = series_for(cell, metric, mode, source)
    if series:
        return la.band_stats(series, band, smooth)
    df = cell.get('cache_curves', {}).get((metric, mode, source))
    if df is None or df.empty:
        return None
    return band_from_cache(df, band, smooth)


def write_cache(cells, cache_root, figures, args, seeds):
    """Store every loaded cell's statistics under cache_root.

    Only cells that were loaded FROM the logs are written; a cell that was itself
    read from the cache is skipped, so re-running with one environment missing
    cannot round-trip and degrade the stored numbers.
    """
    os.makedirs(cache_root, exist_ok=True)
    pairs = metric_sources(figures)
    written = 0
    for (env_key, cond_key), cell in sorted(cells.items()):
        if cell.get('cached'):
            print(f'  [cache] {env_key:<9} {cond_key:<10} already cached, left as is')
            continue
        d = cache_cell_dir(cache_root, env_key, cond_key)
        os.makedirs(d, exist_ok=True)

        blocks = []
        for metric, source in pairs:
            for mode in CACHE_MODES:
                series = series_for(cell, metric, mode, source)
                if not series:
                    continue
                st = curve_stats(series)
                st.insert(0, 'source', source)
                st.insert(0, 'mode', mode)
                st.insert(0, 'metric', metric)
                blocks.append(st)
        if blocks:
            pd.concat(blocks, ignore_index=True).to_csv(
                os.path.join(d, 'curves.csv'), index=False)

        cell['table'].to_csv(os.path.join(d, 'table.csv'), index=False)

        beh_rows = []
        for (sd, rep, name), df in sorted(cell.get('behaviour', {}).items()):
            row = {'seed': int(sd), 'rep': int(rep), 'run': name}
            tail = df.tail(args.final_window)
            for m in BEHAVIOUR_ORDER:
                if m in df.columns:
                    row[m] = float(tail[m].mean())
            beh_rows.append(row)
        if beh_rows:
            pd.DataFrame(beh_rows).to_csv(
                os.path.join(d, 'behaviour_table.csv'), index=False)

        ext = []
        for metric, source in pairs:
            curves = cell['curves'] if source == 'gen' else cell.get('behaviour', {})
            vals = [df[metric].to_numpy(dtype=float)
                    for df in curves.values() if metric in df.columns]
            vals = [v[np.isfinite(v)] for v in vals]
            vals = [v for v in vals if v.size]
            if not vals:
                continue
            allv = np.concatenate(vals)
            ext.append({'metric': metric, 'source': source,
                        'lo': float(allv.min()), 'hi': float(allv.max())})
        if ext:
            pd.DataFrame(ext).to_csv(os.path.join(d, 'extent.csv'), index=False)

        meta = {'environment': env_key, 'condition': cond_key,
                'n': int(cell['n']),
                'n_behaviour': len(cell.get('behaviour', {})),
                'runs': [r['name'] for r in cell['runs']],
                'cross_sustain': None,
                'written': dt.datetime.now().isoformat(timespec='seconds')}
        meta.update(cache_stamp(args, seeds))
        with open(os.path.join(d, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        written += 1
        print(f'  [cache] {env_key:<9} {cond_key:<10} {cell["n"]:3d} runs -> {d}')
    return written


def write_cache_survival(cells, cache_root, tt, args):
    """Add the time-to-viability event table to each freshly cached cell.

    Separate from write_cache() because the event table depends on
    --cross-sustain, which nothing else in the cache does, and because the cache
    is written before the figures so that a crash in plotting cannot cost the
    stored numbers.
    """
    if tt is None or not len(tt):
        return 0
    n = 0
    for (env_key, cond_key), cell in sorted(cells.items()):
        if cell.get('cached'):
            continue
        d = cache_cell_dir(cache_root, env_key, cond_key)
        if not os.path.isdir(d):
            continue
        sub = tt[(tt['environment'] == env_key) & (tt['condition'] == cond_key)]
        if not len(sub):
            continue
        sub.to_csv(os.path.join(d, 'survival.csv'), index=False)
        mpath = os.path.join(d, 'meta.json')
        try:
            with open(mpath) as f:
                meta = json.load(f)
        except (ValueError, OSError):
            meta = {}
        meta['cross_sustain'] = args.cross_sustain
        with open(mpath, 'w') as f:
            json.dump(meta, f, indent=2)
        n += 1
    if n:
        print(f'    cached the event table for {n} cell(s) '
              f'(sustain = {args.cross_sustain})')
    return n


def read_cache(cache_root, env_key, cond_key, args, seeds):
    """Rebuild one cell from stored statistics.

    Returns (cell, None) on success and (None, reason) otherwise, so the caller
    can say WHY a cache did not stand in — a silently ignored cache and an
    absent one look identical in a figure and differ completely in what they
    mean.
    """
    d = cache_cell_dir(cache_root, env_key, cond_key)
    mpath = os.path.join(d, 'meta.json')
    if not os.path.exists(mpath):
        return None, None
    try:
        with open(mpath) as f:
            meta = json.load(f)
    except (ValueError, OSError) as exc:
        return None, f'meta.json unreadable ({type(exc).__name__})'

    want = cache_stamp(args, seeds)
    bad = [k for k in CACHE_STAMP if meta.get(k) != want[k]]
    if bad:
        detail = ', '.join(f'{k}: cached {meta.get(k)!r} vs requested {want[k]!r}'
                           for k in bad)
        return None, f'stamp mismatch ({detail})'

    cell = {'table': pd.DataFrame(), 'curves': {}, 'behaviour': {},
            'runs': [], 'n': int(meta.get('n', 0)), 'cached': True,
            'cache_curves': {}, 'extent': {}, 'meta': meta}

    tpath = os.path.join(d, 'table.csv')
    if os.path.exists(tpath):
        cell['table'] = pd.read_csv(tpath)
    if cell['table'].empty:
        return None, 'table.csv missing or empty'
    cell['n'] = int(len(cell['table']))

    cpath = os.path.join(d, 'curves.csv')
    if os.path.exists(cpath):
        cur = pd.read_csv(cpath)
        for (metric, mode, source), g in cur.groupby(['metric', 'mode', 'source']):
            cell['cache_curves'][(str(metric), str(mode), str(source))] = (
                g.sort_values('generation').reset_index(drop=True))

    epath = os.path.join(d, 'extent.csv')
    if os.path.exists(epath):
        for _i, r in pd.read_csv(epath).iterrows():
            cell['extent'][(str(r['metric']), str(r['source']))] = (
                float(r['lo']), float(r['hi']))

    bpath = os.path.join(d, 'behaviour_table.csv')
    if os.path.exists(bpath):
        cell['behaviour_table'] = pd.read_csv(bpath)

    spath = os.path.join(d, 'survival.csv')
    if os.path.exists(spath):
        if meta.get('cross_sustain') == args.cross_sustain:
            cell['survival'] = pd.read_csv(spath)
        else:
            cell['survival_rejected'] = (
                f'cached at sustain {meta.get("cross_sustain")}, '
                f'requested {args.cross_sustain}')
    return cell, None


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalisation_ranges(cells, metrics, scope):
    """lo/hi per fitness metric, over every generation of every run in scope.

    Cached cells contribute through their stored extent, so the range is the
    same whether both environments' logs were on disk together or one was read
    back from the cache — which is the whole point of storing the extent.
    """
    out = {}
    env_keys = sorted({e for e, _c in cells})
    groups = ([('*', list(cells))] if scope == 'global'
              else [(e, [k for k in cells if k[0] == e]) for e in env_keys])
    for gkey, keys in groups:
        for m in metrics:
            vals = []
            for k in keys:
                cell = cells[k]
                for df in cell['curves'].values():
                    if m in df.columns:
                        v = df[m].to_numpy(dtype=float)
                        vals.append(v[np.isfinite(v)])
                # A cached cell kept only the min and max of that metric over
                # every generation of every run, which is all this needs: the
                # min-max of two endpoints IS the min-max of what produced them,
                # so a cached environment joins the pooled range exactly.
                if not cell['curves']:
                    ext = cell.get('extent', {}).get((m, 'gen'))
                    if ext is not None:
                        vals.append(np.asarray(ext, dtype=float))
            if not vals:
                continue
            allv = np.concatenate(vals)
            if allv.size:
                out[(gkey, m)] = (float(allv.min()), float(allv.max()))
    return out


def get_range(ranges, env_key, metric, scope):
    return ranges.get(('*' if scope == 'global' else env_key, metric))


def rescale(values, rng):
    if rng is None:
        return values
    lo, hi = rng
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return values
    return (np.asarray(values, dtype=float) - lo) / span


# ── Series assembly ───────────────────────────────────────────────────────────

def series_for(cell, metric, mode, source):
    """The per-run curves for one metric, regrouped into band units.

    source 'gen'       -> generations.csv (population-level, per generation)
    source 'behaviour' -> organisms.csv aggregate (per-organism means)
    Both are generation-indexed, so band_stats treats them identically.
    """
    curves = cell['curves'] if source == 'gen' else cell.get('behaviour', {})
    if source == 'gen':
        return la.unit_series(curves, metric, mode)
    by_seed = {}
    for (seed, _rep, _name), df in curves.items():
        if metric in df.columns:
            by_seed.setdefault(seed, []).append(df[metric])
    if mode == 'runs':
        return [s for ss in by_seed.values() for s in ss]
    return [pd.concat(ss, axis=1).mean(axis=1) for ss in by_seed.values()]


def converged(cell, metric, source, final_window, rng=None):
    """Converged value per run: the mean of the last `final_window` generations.

    For generations.csv metrics this is already in the endpoint table. Behaviour
    metrics are aggregated here from their own frames so both sources report the
    same statistic over the same window.
    """
    if source == 'gen':
        t = cell['table']
        if metric not in t.columns:
            return np.array([])
        v = t[metric].to_numpy(dtype=float)
    else:
        frames = cell.get('behaviour', {})
        if frames:
            vals = []
            for df in frames.values():
                if metric in df.columns:
                    tail = df[metric].tail(final_window)
                    if len(tail):
                        vals.append(float(tail.mean()))
        else:
            # Cached cell: the same per-run number, already aggregated over the
            # same window when the organisms.csv files were last readable.
            bt = cell.get('behaviour_table')
            vals = ([] if bt is None or metric not in bt.columns
                    else bt[metric].tolist())
        v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    return rescale(v, rng) if rng is not None else v


# ── Error analysis ────────────────────────────────────────────────────────────

def cluster_stats(values, seeds, conf=0.95):
    """Seed-clustered uncertainty for a mean over runs.

    THE 100 RUNS OF A CELL ARE NOT 100 INDEPENDENT SAMPLES. Ten map seeds x ten
    replicates share terrain within a seed, so replicates of one seed are
    positively correlated and the usual sd/sqrt(n) understates the error of the
    mean. Measured here, the intra-class correlation runs 0.01-0.49 depending on
    the cell, giving design effects of 1.1-5.4 and effective sample sizes as low
    as ~18 rather than 100.

    So two error bars are reported and clearly named:
      *_naive    sd/sqrt(n) with n = runs. Correct only if runs are independent,
                 which they are not. Kept because it is what a reader would
                 compute themselves, and the gap to the clustered figure is
                 itself worth seeing.
      *_cluster  the seed treated as the unit of replication: each seed is
                 reduced to its own mean, and the interval is built from the
                 spread of those k means with a t_{k-1} critical value (k = 10,
                 so t = 2.26, not 1.96 — with ten clusters the normal
                 approximation is itself optimistic).

    Quote the clustered interval. Returns a dict; ICC/design effect are reported
    alongside so the inflation is auditable rather than asserted.
    """
    from scipy import stats
    v = np.asarray(values, dtype=float)
    sd_arr = np.asarray(seeds)
    out = {'n_runs': int(v.size), 'n_seeds': 0, 'icc': np.nan,
           'design_effect': np.nan, 'n_effective': np.nan,
           'sem_naive': np.nan, 'sem_cluster': np.nan,
           'ci95_lo_naive': np.nan, 'ci95_hi_naive': np.nan,
           'ci95_lo_cluster': np.nan, 'ci95_hi_cluster': np.nan}
    if v.size == 0:
        return out
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    out['sem_naive'] = sd / np.sqrt(v.size)
    out['ci95_lo_naive'] = mean - 1.96 * out['sem_naive']
    out['ci95_hi_naive'] = mean + 1.96 * out['sem_naive']

    uniq = np.unique(sd_arr)
    k = uniq.size
    out['n_seeds'] = int(k)
    if k < 2:
        return out
    means = np.array([v[sd_arr == u].mean() for u in uniq])
    counts = np.array([(sd_arr == u).sum() for u in uniq], dtype=float)
    m = v.size / k                      # average cluster size

    # One-way random-effects ANOVA -> ICC, then Kish's design effect.
    grand = v.mean()
    msb = float((counts * (means - grand) ** 2).sum() / (k - 1))
    within = sum(float(((v[sd_arr == u] - v[sd_arr == u].mean()) ** 2).sum())
                 for u in uniq)
    msw = within / (v.size - k) if v.size > k else np.nan
    if np.isfinite(msw):
        # sd_seed_means is the OBSERVED spread of the k seed means. It is not
        # the between-seed component: each seed mean is itself an average of m
        # noisy runs, so its expectation is var_between + var_within/m —
        # inflated. sd_between subtracts that inflation, which is exactly what
        # (MSB - MSW)/m does. Using the observed spread directly would put
        # baseline/evolution's ICC at 0.518 instead of 0.493.
        var_between = max((msb - msw) / m, 0.0)
        denom = var_between + msw
        icc = var_between / denom if denom > 0 else 0.0
        out['icc'] = float(icc)
        out['design_effect'] = float(1 + (m - 1) * icc)
        out['n_effective'] = float(v.size / out['design_effect'])
        out['reps_per_seed'] = float(m)
        out['sd_seed_means'] = float(means.std(ddof=1))     # observed, inflated
        out['sd_between_seeds'] = float(np.sqrt(var_between))  # terrain component
        out['sd_within_seed'] = float(np.sqrt(msw))            # run-to-run, pooled
        out['ms_between'] = float(msb)
        out['ms_within'] = float(msw)

    sem_c = float(means.std(ddof=1) / np.sqrt(k))
    tcrit = float(stats.t.ppf(0.5 + conf / 2, k - 1))
    out['sem_cluster'] = sem_c
    out['ci95_lo_cluster'] = mean - tcrit * sem_c
    out['ci95_hi_cluster'] = mean + tcrit * sem_c
    out['t_crit'] = tcrit
    return out


def seed_means(cell, metric):
    """{seed: mean over that seed's replicates}. The seed is the unit."""
    t = cell['table']
    if metric not in t.columns:
        return {}
    return {int(sd): float(g[metric].mean())
            for sd, g in t.groupby('seed') if np.isfinite(g[metric].mean())}


def paired_by_seed(cells, envs, conditions, metric='avg_fitness'):
    """Condition-vs-condition, PAIRED on the map seed.

    The two arms were run on the SAME ten maps, so an unpaired test on 100 vs
    100 throws that away and then pays for it twice: it ignores the pairing that
    removes terrain variance, and it assumes an independence the runs do not
    have. Pairing the ten seed means tests the same hypothesis on 9 df with the
    terrain differenced out.

    Reports the parametric paired t AND the Wilcoxon signed-rank, because with
    k = 10 the t-test's normality assumption is doing real work and the rank
    test does not need it. `wins` counts the seeds favouring condition A, which
    is the assumption-free statement of the same thing.

    p-values are Holm-adjusted across every pair tested here — six comparisons
    on one dataset, so the uncorrected p is optimistic.
    """
    from scipy import stats
    rows = []
    for env in envs:
        for i, a in enumerate(conditions):
            for b in conditions[i + 1:]:
                ca = cells.get((env['key'], a['key']))
                cb = cells.get((env['key'], b['key']))
                if ca is None or cb is None:
                    continue
                ma, mb = seed_means(ca, metric), seed_means(cb, metric)
                shared = sorted(set(ma) & set(mb))
                if len(shared) < 3:
                    continue
                da = np.array([ma[s] for s in shared])
                db = np.array([mb[s] for s in shared])
                diff = da - db
                k = len(shared)
                sem = diff.std(ddof=1) / np.sqrt(k)
                tcrit = float(stats.t.ppf(0.975, k - 1))
                tstat, tp = stats.ttest_rel(da, db)
                try:
                    _w, wp = stats.wilcoxon(da, db)
                except ValueError:
                    wp = np.nan
                # Cohen's dz — the paired effect size, mean difference over the
                # sd of the differences. Scale-free, so it is comparable between
                # the two environments whose fitness ranges differ.
                dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
                rows.append({
                    'environment': env['key'], 'condition_a': a['key'],
                    'condition_b': b['key'], 'n_seeds': k,
                    'mean_a': da.mean(), 'mean_b': db.mean(),
                    'mean_diff': diff.mean(), 'sd_diff': diff.std(ddof=1),
                    'sem_diff': sem,
                    'ci95_lo': diff.mean() - tcrit * sem,
                    'ci95_hi': diff.mean() + tcrit * sem,
                    'cohens_dz': dz,
                    't_stat': float(tstat), 'p_paired_t': float(tp),
                    'p_wilcoxon': float(wp),
                    'wins_a': int((diff > 0).sum()),
                    'wins_b': int((diff < 0).sum()),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Holm step-down on the paired-t p-values.
    order = np.argsort(df['p_paired_t'].to_numpy())
    m = len(df)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * df['p_paired_t'].iloc[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    df['p_holm'] = adj
    return df


def cluster_bootstrap(cells, envs, conditions, metric='avg_fitness',
                      n_boot=10000, seed=20260824):
    """Percentile CIs for the mean AND the median, resampling SEEDS.

    Two problems the t-interval cannot handle at once, both present here:
    hard/evolution is bimodal (so a normal-theory interval is on a distribution
    with a hole in the middle), and the runs are clustered. Resampling whole
    SEEDS with replacement — all ten replicates travel together — preserves the
    within-seed correlation instead of assuming it away, and makes no
    distributional assumption at all.

    The median is bootstrapped too, since for a bimodal arm it is the statistic
    worth quoting and it has no closed-form standard error.
    """
    rng_ = np.random.default_rng(seed)
    rows = []
    for env in envs:
        for cond in conditions:
            cell = cells.get((env['key'], cond['key']))
            if cell is None:
                continue
            t = cell['table']
            if metric not in t.columns:
                continue
            by_seed = [g[metric].to_numpy(dtype=float)
                       for _sd, g in t.groupby('seed')]
            by_seed = [v[np.isfinite(v)] for v in by_seed]
            by_seed = [v for v in by_seed if v.size]
            k = len(by_seed)
            if k < 3:
                continue
            obs = np.concatenate(by_seed)
            means = np.empty(n_boot)
            meds = np.empty(n_boot)
            for b in range(n_boot):
                pick = rng_.integers(0, k, k)
                draw = np.concatenate([by_seed[j] for j in pick])
                means[b] = draw.mean()
                meds[b] = np.median(draw)
            rows.append({
                'environment': env['key'], 'condition': cond['key'],
                'n_seeds': k, 'n_runs': int(obs.size), 'n_boot': n_boot,
                'mean': float(obs.mean()),
                'mean_ci95_lo': float(np.percentile(means, 2.5)),
                'mean_ci95_hi': float(np.percentile(means, 97.5)),
                'mean_boot_se': float(means.std(ddof=1)),
                'median': float(np.median(obs)),
                'median_ci95_lo': float(np.percentile(meds, 2.5)),
                'median_ci95_hi': float(np.percentile(meds, 97.5)),
                'median_boot_se': float(meds.std(ddof=1)),
            })
    return pd.DataFrame(rows)


def _wilson(x, n, z=1.96):
    """Wilson score interval for a proportion.

    Not the Wald interval: at x = 0 (five of the six cells here) Wald gives the
    degenerate [0, 0], which would report "no run can ever collapse" from
    evidence that only says "none did in 100 runs". Wilson stays inside [0, 1]
    and keeps a sensible upper bound at zero counts.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def collapse_rates(cells, envs, conditions, metric='avg_fitness',
                   n_boot=10000, seed=20260824):
    """Share of runs converging below the viability threshold, with intervals.

    Folding collapse into a mean fitness reports a lineage that half-failed;
    what actually happened is that a definite subset of runs failed outright.
    That is a PROPORTION and deserves its own interval.

    Two are given. The Wilson interval treats the 100 runs as independent, so it
    is too narrow for the same reason the naive sem is. The seed-clustered
    bootstrap resamples whole seeds and is the one to quote — collapse turns out
    to be concentrated in particular seeds, so the clustering matters more here
    than anywhere else.
    """
    rng_ = np.random.default_rng(seed)
    rows = []
    for env in envs:
        for cond in conditions:
            cell = cells.get((env['key'], cond['key']))
            if cell is None:
                continue
            t = cell['table']
            if metric not in t.columns:
                continue
            by_seed, per_seed_counts = [], {}
            for sd, g in t.groupby('seed'):
                v = g[metric].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                by_seed.append(v)
                per_seed_counts[int(sd)] = int((v < VIABILITY).sum())
            if not by_seed:
                continue
            obs = np.concatenate(by_seed)
            x, nn = int((obs < VIABILITY).sum()), int(obs.size)
            wlo, whi = _wilson(x, nn)
            k = len(by_seed)
            props = np.empty(n_boot)
            for b in range(n_boot):
                pick = rng_.integers(0, k, k)
                draw = np.concatenate([by_seed[j] for j in pick])
                props[b] = (draw < VIABILITY).mean()
            rows.append({
                'environment': env['key'], 'condition': cond['key'],
                'n_runs': nn, 'n_seeds': k, 'n_collapsed': x,
                'rate': x / nn,
                'wilson_lo': wlo, 'wilson_hi': whi,
                'boot_lo': float(np.percentile(props, 2.5)),
                'boot_hi': float(np.percentile(props, 97.5)),
                'seeds_with_any_collapse': int(sum(c > 0 for c in per_seed_counts.values())),
                'per_seed_counts': ';'.join(f'{sd}:{c}' for sd, c
                                            in sorted(per_seed_counts.items()) if c),
            })
    return pd.DataFrame(rows)


# ── Time to viability (survival analysis) ─────────────────────────────────────

def time_to_threshold(cells, envs, conditions, sustain=10, metric='avg_fitness'):
    """First generation at which a run reaches — and HOLDS — viable fitness.

    The Baldwin effect is a claim about ACCELERATION, so the quantity it
    predicts is a time, not an endpoint. This builds the event table for it.

    WHY "HOLDS", AND NOT THE FIRST CROSSING. Measured on this data, the plain
    first crossing of f = 4.375 is useless: every one of the 600 runs crosses it,
    so there is no censoring at all and the statistic cannot distinguish a run
    that reached viability from one that brushed past it on the way to collapse.
    Hard/evolution's median first crossing is generation 91 — yet ten of those
    runs are below the line at the end. The event is therefore defined as the
    first generation starting `sustain` CONSECUTIVE generations at or above the
    threshold. Runs that never manage it are right-censored at their last
    generation, which is what a survival model needs.

    sustain = 10 is the default because it is where censoring first becomes
    informative (10 hard/evolution and 2 hard/pure_rl runs censored — exactly the
    runs that converge below the line) and because the answer barely moves after
    it: medians at sustain = 10, 25 and 50 agree to within one generation in
    five of the six cells. Sweep it with --cross-sustain.
    """
    rows = []
    for env in envs:
        for cond in conditions:
            cell = cells.get((env['key'], cond['key']))
            if cell is None:
                continue
            if not cell['curves']:
                # No curves to scan; the event table was stored instead. It is
                # accepted only when it was built at this --cross-sustain, which
                # read_cache() has already checked.
                sv = cell.get('survival')
                if sv is not None and len(sv):
                    rows.extend(sv.to_dict('records'))
                elif cell.get('survival_rejected'):
                    print(f'    [skip] {env["key"]:<9} {cond["key"]:<10} '
                          f'cached event table unusable: '
                          f'{cell["survival_rejected"]}')
                continue
            for (seed, rep, name), df in cell['curves'].items():
                if metric not in df.columns:
                    continue
                f = df[metric].to_numpy(dtype=float)
                g = df.index.to_numpy(dtype=float)
                ok = np.isfinite(f)
                f, g = f[ok], g[ok]
                if f.size == 0:
                    continue
                above = (f >= VIABILITY).astype(int)
                hit = None
                if above.size >= sustain:
                    run = np.convolve(above, np.ones(sustain, dtype=int), 'valid')
                    if (run == sustain).any():
                        hit = int(np.argmax(run == sustain))
                rows.append({
                    'environment': env['key'], 'condition': cond['key'],
                    'seed': int(seed), 'rep': int(rep), 'run': name,
                    'time': float(g[hit]) if hit is not None else float(g[-1]),
                    'event': 1 if hit is not None else 0,
                    'last_gen': float(g[-1]),
                    'first_crossing': (float(g[np.argmax(above)])
                                       if above.any() else np.nan),
                })
    return pd.DataFrame(rows)


def km_curves(tt, envs, conditions):
    """Kaplan-Meier estimate per cell, plus median and restricted mean.

    KM is what handles the censored runs correctly: a run that never reaches
    viability still contributes everything it tells us — that it had not done so
    by generation 1000 — instead of being dropped (which would bias the median
    down) or scored as 1000 (which would invent a time it never had).
    """
    from lifelines import KaplanMeierFitter
    fits, rows = {}, []
    for env in envs:
        for cond in conditions:
            sub = tt[(tt['environment'] == env['key'])
                     & (tt['condition'] == cond['key'])]
            if sub.empty:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub['time'], sub['event'], label=cond['label'])
            fits[(env['key'], cond['key'])] = kmf
            ev = int(sub['event'].sum())
            rows.append({
                'environment': env['key'], 'condition': cond['key'],
                'n_runs': len(sub), 'n_reached': ev,
                'n_censored': len(sub) - ev,
                'median_gen': float(kmf.median_survival_time_),
                'q25_gen': float(kmf.percentile(0.75)),
                'q75_gen': float(kmf.percentile(0.25)),
                'reached_by_100': float(1 - kmf.predict(100)),
                'reached_by_250': float(1 - kmf.predict(250)),
                'reached_by_500': float(1 - kmf.predict(500)),
            })
    return fits, pd.DataFrame(rows)


def cox_models(tt, envs, conditions, ref='evolution'):
    """Cox proportional-hazards per environment, SEs clustered by map seed.

    The hazard here is the instantaneous rate of REACHING viability, so a hazard
    ratio above 1 means faster — which is the Baldwin claim stated in the form
    the model tests. Evolution is the reference level, so HR is read as "this
    condition reaches viability HR times as fast as the GA alone".

    Clustering on `seed` is the same correction as the clustered confidence
    intervals elsewhere: the ten replicates of a map are not independent, and an
    unclustered Cox model would report standard errors that are too small. (A
    seed frailty term would be the alternative; lifelines exposes the robust
    sandwich estimator instead, which answers the same question without having
    to assume a frailty distribution.)

    The proportional-hazards assumption is checked and its p-value reported —
    if it fails, the HR is a time-averaged summary rather than a constant
    effect, and the KM curves are the honest presentation.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    rows = []
    for env in envs:
        sub = tt[tt['environment'] == env['key']].copy()
        if sub.empty or sub['condition'].nunique() < 2:
            continue
        levels = [c['key'] for c in conditions if c['key'] in set(sub['condition'])]
        if ref not in levels:
            continue
        design = pd.DataFrame({'time': sub['time'].to_numpy(),
                               'event': sub['event'].to_numpy(),
                               'seed': sub['seed'].to_numpy()})
        for lv in levels:
            if lv != ref:
                design[f'cond_{lv}'] = (sub['condition'] == lv).astype(float).to_numpy()
        cph = CoxPHFitter()
        try:
            cph.fit(design, duration_col='time', event_col='event',
                    cluster_col='seed')
        except Exception as exc:                      # noqa: BLE001
            print(f'    (Cox failed for {env["key"]}: {exc})')
            continue
        try:
            ph = proportional_hazard_test(cph, design, time_transform='rank')
            ph_p = float(np.nanmin(ph.p_value))
        except Exception:                             # noqa: BLE001
            ph_p = np.nan
        summ = cph.summary
        for name, r in summ.iterrows():
            rows.append({
                'environment': env['key'],
                'term': str(name).replace('cond_', ''),
                'reference': ref,
                'coef': float(r['coef']),
                'hazard_ratio': float(r['exp(coef)']),
                'hr_ci95_lo': float(r['exp(coef) lower 95%']),
                'hr_ci95_hi': float(r['exp(coef) upper 95%']),
                'se_clustered': float(r['se(coef)']),
                'z': float(r['z']), 'p': float(r['p']),
                'n_runs': int(len(design)),
                'n_events': int(design['event'].sum()),
                'n_seeds': int(design['seed'].nunique()),
                'concordance': float(cph.concordance_index_),
                'ph_assumption_min_p': ph_p,
            })
    return pd.DataFrame(rows)


def logrank_by_env(tt, envs):
    """Multivariate log-rank across the three conditions, per environment.

    An omnibus test of "do these curves differ at all", reported beside the Cox
    model because it assumes nothing about proportional hazards.
    """
    from lifelines.statistics import multivariate_logrank_test
    rows = []
    for env in envs:
        sub = tt[tt['environment'] == env['key']]
        if sub.empty or sub['condition'].nunique() < 2:
            continue
        res = multivariate_logrank_test(sub['time'], sub['condition'], sub['event'])
        rows.append({'environment': env['key'], 'n_runs': len(sub),
                     'test_statistic': float(res.test_statistic),
                     'p': float(res.p_value)})
    return pd.DataFrame(rows)


# ── Drawing ───────────────────────────────────────────────────────────────────

def style_axes(ax):
    la.style_axes(ax)


def draw_conditions(ax, cells, env_key, conditions, metric, mode, band, smooth,
                    source='gen', rng=None, show_threshold=False):
    """One axis: a smoothed mean line + shaded band per condition.

    Returns the rows needed for the side legend: [(condition, converged values)].
    `rng` non-None rescales the centre and both band edges by the same affine
    map, so the normalised spread is exactly the raw spread divided by
    (hi - lo) and nothing about it flatters one arm over another.
    """
    rows = []
    for cond in conditions:
        cell = cells.get((env_key, cond['key']))
        if cell is None:
            continue
        stats = band_for(cell, metric, mode, band, smooth, source)
        if stats is None:
            continue
        gens, centre, lo, hi, n = stats
        if rng is not None:
            centre, lo, hi = rescale(centre, rng), rescale(lo, rng), rescale(hi, rng)
        if metric in getattr(la, 'NONNEGATIVE', set()) or metric.startswith('frac_'):
            lo = np.maximum(lo, 0.0)
        ax.fill_between(gens, lo, hi, color=cond['colour'], alpha=0.15, lw=0,
                        zorder=2)
        # Solid lines UNDER dashed ones. Where two arms converge to the same
        # value — baseline evolution and learning are within 0.013 of each other —
        # whichever is drawn last hides the other completely. Dashes drawn on top
        # let the solid line show through the gaps, so an overlap reads as an
        # overlap instead of as a missing condition.
        ax.plot(gens, centre, color=cond['colour'], ls=cond['ls'], lw=2.1,
                zorder=3 if cond['ls'] == '-' else 4)
        rows.append(cond)
    if not rows:
        return []

    if show_threshold:
        y = VIABILITY if rng is None else rescale([VIABILITY], rng)[0]
        ax.axhline(y, color=la.THRESHOLD_COLOUR, lw=1.0, ls=':', zorder=1)
        ax.annotate('viability threshold', xy=(0.985, y),
                    xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                    fontsize=7.5, color=la.THRESHOLD_COLOUR)

    ax.set_xlabel('Generation')
    style_axes(ax)
    return rows


def legend_strip(fig, handles, title, ncol=None, y=0.005, fontsize=9.0):
    """A legend UNDER the axes, one column per entry.

    WHY NOT DOWN THE RIGHT-HAND SIDE, WHICH IS WHERE THIS USED TO LIVE
    ------------------------------------------------------------------
    A legend column costs its width across the FULL height of the plate, and
    three short blocks of text do not fill that height — on the two-panel
    comparison it reserved 19.5% of the width and left most of that column
    blank, on the single-panel figures 23.5%. In a thesis the figure is scaled
    to the text width, so that blank column is paid for by shrinking the axes:
    the curves lose a quarter of their width to whitespace that carries nothing.

    Underneath, the same text runs ACROSS the width in as many columns as there
    are conditions, so it occupies a band roughly two text-lines deep and the
    axes get the whole width back. It is also the correct reading order — the
    numbers are read after the curves, not beside them.

    `y` is a figure-fraction anchor slightly below 0, so the strip hangs off the
    bottom and save()'s bbox_inches='tight' grows the canvas to include it. That
    keeps tight_layout's rect free to give the axes the full width, which is the
    entire point.
    """
    leg = fig.legend(handles=handles, loc='upper center',
                     bbox_to_anchor=(0.5, y), ncol=(ncol or len(handles)),
                     frameon=False, fontsize=fontsize, labelcolor=INK,
                     handlelength=2.4, handletextpad=0.8, columnspacing=3.0,
                     borderaxespad=0.0, title=title)
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(INK)
    return leg


def side_legend(fig, entries, final_window, band, anchor=None):
    """Converged mean ± 1 std per condition, in a strip beneath the axes.

    The plot area still holds nothing but the lines it exists to compare, and
    the number a reader would quote is still on the figure rather than only in
    summary.csv — see legend_strip for where it moved to and why.
    """
    handles = []
    for cond, v in entries:
        if v.size:
            sd = v.std(ddof=1) if v.size > 1 else 0.0
            txt = f'{cond["label"]}\n{v.mean():.4g} ± {sd:.3g}   (n = {v.size})'
        else:
            txt = f'{cond["label"]}\n(no converged value)'
        handles.append(Line2D([], [], color=cond['colour'], ls=cond['ls'],
                              lw=2.4, label=txt))
    spread = 'IQR' if band == 'iqr' else '1 std'
    return legend_strip(fig, handles,
                        f'converged mean ± {spread} across runs')


def bar_key(ax):
    """Neutral-ink key for the two bar styles, inside the axes.

    The condition colours already carry the condition; this key carries only
    what solid vs dashed means, so it does not compete with them.
    """
    handles = [Line2D([], [], color=la.NEUTRAL, lw=2.4, label='mean'),
               Line2D([], [], color=la.NEUTRAL, lw=2.0, ls=(0, (4, 2.5)),
                      label='median')]
    leg = ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower right',
                    handlelength=2.4, borderpad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def strip_columns(ax, groups, colours, show_threshold=True, rng=None,
                  point_size=16, jitter=0.17, seed0=12345):
    """Jittered dots + a solid MEAN bar and a dashed MEDIAN bar, per column.

    groups: [(label, values ndarray)]. One drawing routine for both the
    by-condition and the by-seed strips, so the two figures cannot drift apart
    in what a bar means.
    """
    drawn = []
    for i, ((label, v), colour) in enumerate(zip(groups, colours)):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        # Deterministic jitter: the same run lands in the same place on every
        # redraw, so two versions of a figure can be compared dot for dot.
        rs = np.random.default_rng(seed0 + i)
        ax.scatter(i + rs.uniform(-jitter, jitter, v.size), v, s=point_size,
                   color=colour, alpha=0.55, lw=0.4, edgecolors='white',
                   zorder=3)
        ax.plot([i - 0.30, i + 0.30], [v.mean()] * 2, color=colour, lw=2.4,
                zorder=5, solid_capstyle='butt')
        ax.plot([i - 0.30, i + 0.30], [np.median(v)] * 2, color=colour, lw=2.0,
                ls=(0, (4, 2.5)), zorder=6, dash_capstyle='butt')
        if v.size > 1:
            sd = v.std(ddof=1)
            ax.plot([i, i], [v.mean() - sd, v.mean() + sd], color=colour,
                    lw=1.2, zorder=2, alpha=0.8)
        drawn.append((label, v))
    if not drawn:
        return []
    if show_threshold:
        y = VIABILITY if rng is None else rescale([VIABILITY], rng)[0]
        ax.axhline(y, color=la.THRESHOLD_COLOUR, lw=1.0, ls=':', zorder=1)
        ax.annotate('viability threshold', xy=(0.985, y),
                    xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                    fontsize=7.5, color=la.THRESHOLD_COLOUR)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    style_axes(ax)
    return drawn


def seed_groups(cell, metric, source, final_window, rng):
    """Pooled column first, then one column per map seed.

    The pooled column is the number every table quotes; the per-seed columns are
    what says whether that number is a property of the algorithm or of the ten
    maps it happened to be measured on.
    """
    table = cell['table']
    if metric not in table.columns:
        return [], []
    v_all = converged(cell, metric, source, final_window, rng)
    groups = [(f'all\n({v_all.size} runs)', v_all)]
    for sd in sorted(table['seed'].unique()):
        vals = table.loc[table['seed'] == sd, metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if rng is not None:
            vals = rescale(vals, rng)
        groups.append((str(int(sd)), vals))
    return groups, [g[1] for g in groups]


def strip_conditions(ax, cells, env_key, conditions, metric, source='gen',
                     final_window=50, rng=None, show_threshold=True):
    """Converged value, one dot per run, one column per condition.

    The mean curve hides the distribution behind it; in the hard environment the
    evolution arm is bimodal (a collapsed cluster and a successful one) and a
    mean line alone would report a value almost no run actually took.
    """
    present, groups, colours = [], [], []
    for cond in conditions:
        cell = cells.get((env_key, cond['key']))
        if cell is None:
            continue
        v = converged(cell, metric, source, final_window, rng)
        if not v.size:
            continue
        present.append(cond)
        groups.append((cond['label'].split(' (')[0], v))
        colours.append(cond['colour'])
    if not groups:
        return []
    strip_columns(ax, groups, colours, show_threshold=show_threshold, rng=rng)
    return list(zip(present, [g[1] for g in groups]))


def metric_label(metric, source):
    if source == 'behaviour':
        return BEHAVIOUR_METRICS.get(metric, metric)
    return la.METRICS.get(metric, (metric, '{:.3f}'))[0]


def axis_label(metric, source, normalised, rng):
    base = metric_label(metric, source)
    if not normalised:
        return base
    return f'{base} — normalised'


def save(fig, path, rect=None, dpi=None):
    """Write one figure in every configured format. `path` names the .png."""
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    stem = os.path.splitext(path)[0]
    written = []
    for fmt in OUTPUT['formats']:
        out = f'{stem}.{fmt}'
        fig.savefig(out, dpi=(dpi or OUTPUT['dpi']), bbox_inches='tight',
                    facecolor='white', format=fmt)
        written.append(fmt)
    plt.close(fig)
    print(f'    wrote {stem}.{{{",".join(written)}}}')


def fig_seed_strip(cells, env, cond, ranges, scope, args, out_path,
                   metric='avg_fitness'):
    """One condition, one environment: pooled column then one column per seed.

    The single most useful diagnostic for the error analysis. If the per-seed
    columns sit at visibly different heights, the terrain is a real factor and
    the ten replicates of a seed are NOT ten independent samples — which is
    exactly what the clustered confidence interval in fitness_summary.csv
    accounts for and the naive one does not.
    """
    cell = cells.get((env['key'], cond['key']))
    if cell is None:
        return False
    rng = get_range(ranges, env['key'], metric, scope)
    groups, _ = seed_groups(cell, metric, 'gen', args.final_window, rng)
    if len(groups) < 2:
        return False

    fig, ax = plt.subplots(figsize=(1.05 * len(groups) + 2.6, 4.9))
    # Pooled column wears the condition's hue; seeds are NEUTRAL — ten
    # categories is well past where categorical colour stays separable, so they
    # are told apart by position.
    colours = [cond['colour']] + [la.NEUTRAL] * (len(groups) - 1)
    strip_columns(ax, groups, colours, rng=rng, point_size=18)
    ax.set_ylabel(metric_label(metric, 'gen')
                  + (' — normalised' if rng else ''))
    ax.set_xlabel('Map seed')

    st = cluster_stats(cell['table'][metric].to_numpy(dtype=float),
                       cell['table']['seed'].to_numpy())
    # ICC only on the figure. The design effect and effective n are derived
    # from it and stay in fitness_summary.csv rather than crowding the title.
    ax.set_title(f'solid bar = mean · dashed bar = median · whisker = ±1 std'
                 f'   |   ICC {st["icc"]:.2f}', fontsize=9, color=INK)
    fig.suptitle(f'{metric_label(metric, "gen")} by map seed — '
                 f'{cond["label"]} · {env["label"]}',
                 fontsize=12, color=INK, y=0.995)
    save(fig, out_path, rect=(0, 0, 1, 0.93))
    return True


def fig_seed_strip_grid(cells, envs, conditions, ranges, scope, args, out_path,
                        metric='avg_fitness'):
    """Every environment x condition on one sheet, rows = environment.

    Rows share a y-axis so a seed that is hard in one condition can be seen to
    be hard in the others — the signature of a terrain effect rather than an
    algorithmic one.
    """
    nrow, ncol = len(envs), len(conditions)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow),
                            sharey='row', squeeze=False)
    drew = False
    for r, env in enumerate(envs):
        for c, cond in enumerate(conditions):
            ax = axs[r][c]
            cell = cells.get((env['key'], cond['key']))
            if cell is None:
                ax.axis('off')
                continue
            rng = get_range(ranges, env['key'], metric, scope)
            groups, _ = seed_groups(cell, metric, 'gen', args.final_window, rng)
            if len(groups) < 2:
                ax.axis('off')
                continue
            colours = [cond['colour']] + [la.NEUTRAL] * (len(groups) - 1)
            # Threshold on EVERY panel, not just the last of a row. Rows share
            # a y-axis so the line sits at the same height across the row
            # regardless, but a reader comparing one panel against its
            # neighbour should not have to carry the level across from the far
            # right — and a panel whose points all sit above it says something
            # only if the line is in the panel.
            strip_columns(ax, groups, colours, rng=rng, point_size=11,
                          show_threshold=True)
            ax.tick_params(axis='x', labelsize=7)
            st = cluster_stats(cell['table'][metric].to_numpy(dtype=float),
                               cell['table']['seed'].to_numpy())
            ax.set_title(f'{cond["label"].split(" (")[0]} — {env["short"]}'
                         f'   (ICC {st["icc"]:.2f})', fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel(metric_label(metric, 'gen')
                              + (' — normalised' if rng else ''))
            if r == nrow - 1:
                ax.set_xlabel('Map seed')
            drew = True
    if not drew:
        plt.close(fig)
        return False
    fig.suptitle(f'{metric_label(metric, "gen")} by map seed — every condition '
                 f'and environment  ·  solid = mean, dashed = median, '
                 f'whisker = ±1 std', fontsize=12.5, color=INK, y=1.0)
    save(fig, out_path, rect=(0, 0, 1, 0.96))
    return True


def draw_km(ax, tt, env, conditions, sustain, show_ci=True):
    """Cumulative incidence: the share of runs that have reached viability by g.

    Plotted as 1 - S(g) rather than S(g) because the event here is an
    achievement, not a failure — a curve that climbs to 1 reads directly as
    "this fraction has got there", where a falling survival curve would have to
    be mentally inverted. The step shape is the Kaplan-Meier estimator, so
    censored runs are handled properly; the shaded band is its 95% CI.
    """
    from lifelines import KaplanMeierFitter
    drawn = []
    for cond in conditions:
        sub = tt[(tt['environment'] == env['key'])
                 & (tt['condition'] == cond['key'])]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['time'], sub['event'])
        cd = kmf.cumulative_density_
        x = cd.index.to_numpy(dtype=float)
        y = cd.iloc[:, 0].to_numpy(dtype=float)
        if show_ci:
            ci = kmf.confidence_interval_cumulative_density_
            ax.fill_between(ci.index.to_numpy(dtype=float),
                            ci.iloc[:, 0].to_numpy(dtype=float),
                            ci.iloc[:, 1].to_numpy(dtype=float),
                            color=cond['colour'], alpha=0.13, lw=0, step='post',
                            zorder=2)
        ax.step(x, y, where='post', color=cond['colour'], ls=cond['ls'], lw=2.1,
                zorder=3 if cond['ls'] == '-' else 4)
        drawn.append((cond, sub, kmf))
    if not drawn:
        return []
    ax.set_xlabel('Generation')
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax)
    return drawn


def km_legend(fig, drawn, sustain, anchor=None):
    handles = []
    for cond, sub, kmf in drawn:
        med = kmf.median_survival_time_
        cens = int((sub['event'] == 0).sum())
        med_txt = 'not reached' if not np.isfinite(med) else f'{med:.0f}'
        tail = f' · {cens} censored' if cens else ''
        handles.append(Line2D([], [], color=cond['colour'], ls=cond['ls'], lw=2.4,
                              label=f'{cond["label"]}\nmedian gen {med_txt}'
                                    f'  (n = {len(sub)}{tail})'))
    return legend_strip(fig, handles,
                        'median generation reaching sustained viability')


def _thr_text(thr_norm):
    """The event threshold in both unit systems.

    The KM y-axis is a PROPORTION OF RUNS, not fitness, so the threshold cannot
    live on it — it belongs in the event definition. Quoting it in normalised
    units as well keeps it tied to the fitness figures, where 0.464 is the
    height the viability line sits at.
    """
    if thr_norm is None or not np.isfinite(thr_norm):
        return f'f ≥ {VIABILITY:.3f}'
    return f'f ≥ {VIABILITY:.3f}  ({thr_norm:.3f} normalised)'


def fig_km(tt, env, conditions, sustain, out_path, thr_norm=None):
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    drawn = draw_km(ax, tt, env, conditions, sustain)
    if not drawn:
        plt.close(fig)
        return False
    ax.set_ylabel('Fraction of runs having reached viability\n'
                  '(share of runs, not a fitness)')
    ax.set_title(f'Kaplan–Meier estimate · event = first generation with '
                 f'{sustain} consecutive generations at {_thr_text(thr_norm)} · '
                 f'shaded 95% CI', fontsize=9.5, color=INK)
    km_legend(fig, drawn, sustain)
    fig.suptitle(f'Time to viability — {env["label"]}', fontsize=12.5,
                 color=INK, y=0.995)
    save(fig, out_path, rect=(0, 0, 1, 0.93))
    return True


def fig_km_compare(tt, envs, conditions, sustain, out_path, thr_norm=None):
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=True)
    per_env, any_drawn = [], False
    for ax, env in zip(axs, envs):
        drawn = draw_km(ax, tt, env, conditions, sustain)
        ax.set_title(env['label'], fontsize=10.5, color=INK)
        per_env.append((env, drawn))
        any_drawn = any_drawn or bool(drawn)
    if not any_drawn:
        plt.close(fig)
        return False
    axs[0].set_ylabel('Fraction of runs having reached viability\n'
                      '(share of runs, not a fitness)')

    handles = []
    for cond in conditions:
        parts = []
        for env, drawn in per_env:
            for c, sub, kmf in drawn:
                if c['key'] != cond['key']:
                    continue
                med = kmf.median_survival_time_
                med_txt = 'not reached' if not np.isfinite(med) else f'{med:.0f}'
                cens = int((sub['event'] == 0).sum())
                tail = f', {cens} censored' if cens else ''
                parts.append(f'{env["short"]}: gen {med_txt} (n = {len(sub)}{tail})')
        if parts:
            handles.append(Line2D([], [], color=cond['colour'], ls=cond['ls'],
                                  lw=2.4,
                                  label=cond['label'] + '\n' + '\n'.join(parts)))
    leg = fig.legend(handles=handles, loc='center left',
                     bbox_to_anchor=(0.815, 0.5), frameon=False, fontsize=9,
                     labelcolor=INK, labelspacing=1.6, handlelength=2.6,
                     title='median generation reaching\nsustained viability')
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(INK)
    fig.suptitle(f'Time to viability — {envs[0]["short"]} vs {envs[1]["short"]}',
                 fontsize=12.5, color=INK, y=1.005)
    fig.text(0.40, 0.945,
             f'Kaplan–Meier · event = {sustain} consecutive generations at '
             f'{_thr_text(thr_norm)} · shaded 95% CI · shared y-axis, '
             f'INDEPENDENT x (the two environments differ ~10x in timescale)',
             ha='center', fontsize=8.5, color=INK, alpha=0.78)
    save(fig, out_path, rect=(0, 0, 1, 0.925))
    return True


# ── Per-environment figures ───────────────────────────────────────────────────

def per_environment(cells, env, conditions, ranges, scope, args, out_root, figures):
    env_key = env['key']
    out_dir = os.path.join(out_root, env_key)
    os.makedirs(out_dir, exist_ok=True)
    n_total = sum(c['n'] for k, c in cells.items() if k[0] == env_key)
    print(f'  {env["label"]}  ({n_total} runs) -> {out_dir}')

    for metric, stem, normalise, source in figures:
        rng = get_range(ranges, env_key, metric, scope) if normalise else None
        # 8.6in of plate, all of it axes. Was 11.0in with the right-hand 23.5%
        # reserved for the legend, i.e. the same 8.4in of curve on a plate a
        # quarter wider — see legend_strip.
        fig, ax = plt.subplots(figsize=(8.6, 4.5))
        drawn = draw_conditions(ax, cells, env_key, conditions, metric,
                                args.aggregate, args.band, args.smooth,
                                source=source, rng=rng,
                                show_threshold=(metric in FITNESS_METRICS))
        if not drawn:
            plt.close(fig)
            continue
        ax.set_ylabel(axis_label(metric, source, normalise, rng))
        # Every method note lives here, on one line, so the legend can carry
        # nothing but the numbers. "converged = last N gens" defines the value
        # the legend reports; it is NOT what the ± is measured over.
        note = (f'{la.centre_label(args.band)} over runs · shaded '
                f'{la.band_label(args.band, args.aggregate)}'
                + (f' · smoothed {args.smooth} gens' if args.smooth > 1 else '')
                + f' · converged = last {args.final_window} gens')
        if normalise:
            ax.set_ylim(-0.02, 1.02)
            note += (f'\nrescaled by the {scope} raw range '
                     f'[{rng[0]:.2f}, {rng[1]:.2f}] — one transform for every '
                     f'condition and environment')
        ax.set_title(note, fontsize=9.5, color=INK)

        entries = [(c, converged(cells[(env_key, c['key'])], metric, source,
                                 args.final_window, rng))
                   for c in drawn]
        side_legend(fig, entries, args.final_window, args.band)
        fig.suptitle(f'{metric_label(metric, source)} — {env["label"]}  ·  '
                     f'{n_total} runs across {len(drawn)} conditions',
                     fontsize=12.5, color=INK, y=1.0)
        # 0.945, not 0.93: the axes title already carries the method note, so
        # the only thing this band still holds is the suptitle. Raising it
        # further is a false economy — savefig(bbox_inches='tight') just grows
        # the canvas to fit whatever spills over the top.
        save(fig, os.path.join(out_dir, f'{stem}.png'), rect=(0, 0, 1, 0.945))

    # Converged-fitness strips, raw and normalised.
    for normalise, stem in ((False, 'strip_final_fitness_raw'),
                            (True, 'strip_final_fitness_normalised')):
        rng = get_range(ranges, env_key, 'avg_fitness', scope) if normalise else None
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        entries = strip_conditions(ax, cells, env_key, conditions, 'avg_fitness',
                                   final_window=args.final_window, rng=rng)
        if not entries:
            plt.close(fig)
            continue
        ax.set_ylabel(axis_label('avg_fitness', 'gen', normalise, rng))
        ax.set_title(f'one dot per run · solid bar = mean ±1 std · dashed bar = '
                     f'median · converged = last {args.final_window} gens',
                     fontsize=9.5, color=INK)
        bar_key(ax)
        fig.suptitle(f'Converged average fitness — {env["label"]}',
                     fontsize=12.5, color=INK, y=0.995)
        if normalise:
            ax.set_ylim(-0.02, 1.02)
        save(fig, os.path.join(out_dir, f'{stem}.png'), rect=(0, 0, 1, 0.93))

    # Per-seed strips: one per condition, plus all three on one sheet.
    for cond in conditions:
        fig_seed_strip(cells, env, cond, ranges, scope, args,
                       os.path.join(out_dir, f'strip_by_seed_{cond["key"]}.png'))
    fig_seed_strip_grid(cells, [env], conditions, ranges, scope, args,
                        os.path.join(out_dir, 'strip_by_seed_all_conditions.png'))


# ── Side-by-side comparison figures ───────────────────────────────────────────

def comparisons(cells, envs, conditions, ranges, scope, args, out_root, figures):
    """Same metric, baseline LEFT and hard RIGHT, on a SHARED y-axis.

    The shared axis is the point: two panels auto-scaled independently would
    place a converged hard arm and a converged baseline arm at the same height
    on the page while their values differ by a factor of two.
    """
    out_dir = os.path.join(out_root, 'compare')
    os.makedirs(out_dir, exist_ok=True)
    print(f'  side-by-side -> {out_dir}')

    for metric, stem, normalise, source in figures:
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)
        drawn_any, per_env = False, []
        for ax, env in zip(axs, envs):
            rng = get_range(ranges, env['key'], metric, scope) if normalise else None
            drawn = draw_conditions(ax, cells, env['key'], conditions, metric,
                                    args.aggregate, args.band, args.smooth,
                                    source=source, rng=rng,
                                    show_threshold=(metric in FITNESS_METRICS))
            ax.set_title(env['label'], fontsize=10.5, color=INK)
            per_env.append((env, drawn, rng))
            drawn_any = drawn_any or bool(drawn)
        if not drawn_any:
            plt.close(fig)
            continue

        rng0 = get_range(ranges, envs[0]['key'], metric, scope) if normalise else None
        axs[0].set_ylabel(axis_label(metric, source, normalise, rng0))
        if normalise:
            axs[0].set_ylim(-0.02, 1.02)

        # The legend carries BOTH environments' converged values per condition,
        # since that side-by-side difference is what the figure is for.
        handles = []
        for cond in conditions:
            parts = []
            for env, drawn, rng in per_env:
                if cond not in drawn:
                    continue
                v = converged(cells[(env['key'], cond['key'])], metric, source,
                              args.final_window, rng)
                if v.size:
                    sd = v.std(ddof=1) if v.size > 1 else 0.0
                    parts.append(f'{env["short"]}: {v.mean():.4g} ± {sd:.3g} '
                                 f'(n = {v.size})')
            if parts:
                handles.append(Line2D([], [], color=cond['colour'], ls=cond['ls'],
                                      lw=2.4,
                                      label=cond['label'] + '\n' + '\n'.join(parts)))
        spread = 'IQR' if args.band == 'iqr' else '1 std'
        legend_strip(fig, handles, f'converged mean ± {spread} across runs')

        note = (f'{la.centre_label(args.band)} over runs · shaded '
                f'{la.band_label(args.band, args.aggregate)} · shared y-axis'
                + (f' · smoothed {args.smooth} gens' if args.smooth > 1 else '')
                + f' · converged = last {args.final_window} gens')
        if normalise:
            note += f' · one min–max range for both panels ({scope} scope)'
        fig.suptitle(f'{metric_label(metric, source)} — '
                     f'{envs[0]["short"]} vs {envs[1]["short"]}',
                     fontsize=12.5, color=INK, y=1.01)
        fig.text(0.5, 0.955, note, ha='center', fontsize=8.5, color=INK,
                 alpha=0.78)
        save(fig, os.path.join(out_dir, f'compare_{stem}.png'),
             rect=(0, 0, 1, 0.94))

    # Strip comparison.
    for normalise, stem in ((False, 'strip_final_fitness_raw'),
                            (True, 'strip_final_fitness_normalised')):
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.9), sharey=True)
        drew = []
        for ax, env in zip(axs, envs):
            rng = get_range(ranges, env['key'], 'avg_fitness', scope) if normalise else None
            drew.append(bool(strip_conditions(ax, cells, env['key'], conditions,
                                              'avg_fitness',
                                              final_window=args.final_window,
                                              rng=rng)))
            ax.set_title(env['label'], fontsize=10.5, color=INK)
        if not any(drew):
            plt.close(fig)
            continue
        rng0 = get_range(ranges, envs[0]['key'], 'avg_fitness', scope) if normalise else None
        axs[0].set_ylabel(axis_label('avg_fitness', 'gen', normalise, rng0))
        if normalise:
            axs[0].set_ylim(-0.02, 1.02)
        fig.suptitle(f'Converged average fitness — {envs[0]["short"]} vs '
                     f'{envs[1]["short"]}', fontsize=12.5, color=INK, y=1.005)
        fig.text(0.5, 0.945,
                 f'one dot per run · solid bar = mean ±1 std · dashed bar = median'
                 f' · converged = last {args.final_window} gens · shared y-axis',
                 ha='center', fontsize=8.5, color=INK, alpha=0.78)
        bar_key(axs[0])
        save(fig, os.path.join(out_dir, f'compare_{stem}.png'),
             rect=(0, 0, 1, 0.925))

    fig_seed_strip_grid(cells, envs, conditions, ranges, scope, args,
                        os.path.join(out_dir, 'compare_strip_by_seed.png'))


# ── Tables ────────────────────────────────────────────────────────────────────

def write_variance_components(cells, envs, conditions, out_root, ranges=None,
                              scope='global', metric='avg_fitness'):
    """The two standard deviations ICC is built from, per cell.

    Reported beside ICC because ICC is a RATIO and cannot distinguish "terrain
    matters a lot" from "runs are quiet". Baseline evolution and baseline
    pure_rl have effectively the same terrain effect (sd_between 0.0396 vs
    0.0136) yet ICCs of 0.49 and 0.03, because the run-to-run noise differs
    threefold. The absolute columns say which of the two moved.
    """
    rows = []
    for env in envs:
        for cond in conditions:
            cell = cells.get((env['key'], cond['key']))
            if cell is None or metric not in cell['table'].columns:
                continue
            t = cell['table']
            st = cluster_stats(t[metric].to_numpy(dtype=float),
                               t['seed'].to_numpy())
            if not np.isfinite(st.get('icc', np.nan)):
                continue
            total = float(t[metric].std(ddof=1))
            sb = st.get('sd_between_seeds', np.nan)
            sw = st.get('sd_within_seed', np.nan)
            # Normalised sigmas: a SPREAD carries no offset, so the min-max map
            # reduces to a division by the span — the `lo` cancels in any
            # difference. Expressed this way each sigma reads as "this fraction
            # of the full observed fitness range", which is comparable between
            # environments whose raw fitness scales differ.
            #
            # ICC IS UNCHANGED BY THIS. It is a ratio of two variances that are
            # both divided by span^2, so the span cancels exactly — the
            # normalised table and the raw table carry the identical ICC column,
            # which is the check that the normalisation was affine.
            rng = (get_range(ranges, env['key'], metric, scope)
                   if ranges is not None else None)
            span = (rng[1] - rng[0]) if rng else np.nan
            rows.append({
                'environment': env['key'], 'condition': cond['key'],
                'metric': metric,
                'n_runs': st['n_runs'], 'n_seeds': st['n_seeds'],
                'reps_per_seed': st.get('reps_per_seed', np.nan),
                'sd_total': total,
                'sd_between_seeds': sb,
                'sd_within_seed': sw,
                'sd_seed_means_observed': st.get('sd_seed_means', np.nan),
                'var_between_seeds': sb ** 2,
                'var_within_seed': sw ** 2,
                'norm_span': float(span) if rng else np.nan,
                'sd_total_normalised': (total / span) if rng else np.nan,
                'sd_between_normalised': (sb / span) if rng else np.nan,
                'sd_within_normalised': (sw / span) if rng else np.nan,
                'icc': st['icc'],
                'design_effect': st['design_effect'],
                'n_effective': st['n_effective'],
            })
    df = pd.DataFrame(rows)
    p = os.path.join(out_root, 'variance_components.csv')
    df.to_csv(p, index=False)
    print(f'    wrote {p}   ({len(df)} rows)')
    write_variance_latex(df, envs, out_root)
    return df


ENV_TEX = {'baseline': 'Baseline', 'hard': 'Predator'}
COND_TEX = {'evolution': 'Evolution', 'learning': 'Learning',
            'pure_rl': 'Pure RL'}


def write_variance_latex(df, envs, out_root, normalised=True):
    """Emit the variance-component table as LaTeX, raw and normalised.

    Generated from the same frame the CSV is written from, so the thesis table
    cannot drift away from the numbers behind it — which is the failure mode
    this function exists to prevent.
    """
    if df.empty:
        return
    order = {e['key']: i for i, e in enumerate(envs)}
    df = df.sort_values(['environment', 'condition'],
                        key=lambda c: c.map(order) if c.name == 'environment' else c)

    for norm in (False, True):
        if norm and not np.isfinite(df['sd_between_normalised']).any():
            continue
        cols = (['sd_total_normalised', 'sd_between_normalised',
                 'sd_within_normalised'] if norm
                else ['sd_total', 'sd_between_seeds', 'sd_within_seed'])
        unit = ('as a fraction of the pooled fitness range'
                if norm else 'in raw fitness units')
        lines = [r'\begin{table}[t]', r'  \centering',
                 r'  \caption{Variance components of converged average fitness, '
                 + unit + r' (mean of the last 50 generations per run; 10 map '
                 r'seeds $\times$ 10 replicates per cell). '
                 r'$\sigma_{\text{between}}$ is the standard deviation '
                 r'attributable to map identity and $\sigma_{\text{within}}$ '
                 r'that of replicate runs on a fixed map, both estimated by '
                 r'one-way random-effects ANOVA. The ICC is their variance '
                 r'ratio and is therefore scale-free: it is identical in both '
                 r'tables.}',
                 r'  \label{tab:icc' + ('-norm' if norm else '') + r'}',
                 r'  \small', r'  \setlength{\tabcolsep}{6pt}',
                 r'  \begin{tabular}{@{}llrrrr@{}}', r'    \toprule',
                 r'    Environment & Condition & $\sigma_{\text{total}}$ & '
                 r'$\sigma_{\text{between}}$ & $\sigma_{\text{within}}$ & ICC \\',
                 r'    \midrule']
        last = None
        for _, r in df.iterrows():
            if last is not None and r['environment'] != last:
                lines.append(r'    \midrule')
            last = r['environment']
            lines.append(
                f'    {ENV_TEX.get(r["environment"], r["environment"]):<9} & '
                f'{COND_TEX.get(r["condition"], r["condition"]):<9} & '
                f'{r[cols[0]]:.4f} & {r[cols[1]]:.4f} & {r[cols[2]]:.4f} & '
                f'{r["icc"]:.2f} ' + r'\\')
        lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
        name = ('variance_components_normalised.tex' if norm
                else 'variance_components.tex')
        path = os.path.join(out_root, name)
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f'    wrote {path}')


def write_fitness_summary(cells, ranges, scope, out_root, final_window):
    """One row per (environment, condition): the headline fitness table.

    summary.csv carries all 19 metrics and is a lookup table; this is the one a
    thesis table is built from, so it is fitness only and adds the statistics a
    reader of a mean +/- std would ask for next:

      sem, ci95_lo/hi   the precision of the MEAN, which std alone does not give
                        (std stays flat as n grows, sem shrinks as 1/sqrt(n))
      median, iqr       robust counterparts, because the hard evolution arm is
                        bimodal and its mean sits in the empty gap between the
                        two clusters
      n_below_threshold how many runs converged under the viability threshold,
                        the number the mean most actively hides
    """
    rows = []
    for (env_key, cond_key), cell in sorted(cells.items()):
        for metric in FITNESS_METRICS:
            v = converged(cell, metric, 'gen', final_window)
            if not v.size:
                continue
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            st = cluster_stats(cell['table'][metric].to_numpy(dtype=float),
                               cell['table']['seed'].to_numpy())
            sem = st['sem_naive']
            med = float(np.median(v))
            q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
            rng = get_range(ranges, env_key, metric, scope)
            # The transform is affine and increasing, so a normalised ORDER
            # statistic is exactly the order statistic of the normalised values
            # — normalising the median is the same as taking the median of the
            # normalised runs, with no approximation. Spreads (std, IQR) divide
            # by the span only; no offset, since a difference cancels the `lo`.
            span = (rng[1] - rng[0]) if rng else np.nan
            nrm = (lambda x: float(rescale([x], rng)[0])) if rng else (lambda x: np.nan)
            rows.append({
                'environment': env_key, 'condition': cond_key, 'metric': metric,
                'n_runs': int(v.size),
                'n_seeds': st['n_seeds'],
                'mean': float(v.mean()), 'std': sd,
                # Two error bars, deliberately both present — see cluster_stats.
                # QUOTE THE CLUSTERED ONE.
                'sem_naive': float(sem),
                'ci95_lo_naive': st['ci95_lo_naive'],
                'ci95_hi_naive': st['ci95_hi_naive'],
                'sem_cluster': st['sem_cluster'],
                'ci95_lo_cluster': st['ci95_lo_cluster'],
                'ci95_hi_cluster': st['ci95_hi_cluster'],
                'icc': st['icc'], 'design_effect': st['design_effect'],
                'n_effective': st['n_effective'],
                'min': float(v.min()), 'max': float(v.max()),
                'median': med,
                'q25': float(q25), 'q75': float(q75), 'iqr': float(q75 - q25),
                'mean_normalised': nrm(v.mean()),
                'median_normalised': nrm(med),
                'std_normalised': (sd / span) if rng else np.nan,
                'sem_naive_normalised': (sem / span) if rng else np.nan,
                'sem_cluster_normalised': ((st['sem_cluster'] / span)
                                           if rng else np.nan),
                'q25_normalised': nrm(q25), 'q75_normalised': nrm(q75),
                'iqr_normalised': ((q75 - q25) / span) if rng else np.nan,
                'min_normalised': nrm(v.min()), 'max_normalised': nrm(v.max()),
                'n_below_threshold': int((v < VIABILITY).sum()),
                'frac_below_threshold': float((v < VIABILITY).mean()),
            })
    df = pd.DataFrame(rows)
    p = os.path.join(out_root, 'fitness_summary.csv')
    df.to_csv(p, index=False)
    print(f'    wrote {p}   ({len(df)} rows)')
    return df


def write_runs_table(cells, ranges, scope, out_root, final_window):
    """One row per RUN — the values every aggregate above is computed from.

    Written because an aggregate nobody can audit is an aggregate nobody should
    quote: with this, any mean or std in the other tables can be recomputed, and
    an individual outlier run can be traced back to its seed and folder.
    """
    frames = []
    for (env_key, cond_key), cell in sorted(cells.items()):
        t = cell['table'].copy()
        # Per-run normalised fitness, so every normalised aggregate in
        # fitness_summary.csv (mean, median, quartiles) can be recomputed from
        # this file alone rather than taken on trust.
        for metric in FITNESS_METRICS:
            rng = get_range(ranges, env_key, metric, scope)
            if rng and metric in t.columns:
                t[f'{metric}_normalised'] = rescale(
                    t[metric].to_numpy(dtype=float), rng)
        t.insert(0, 'environment', env_key)
        t.insert(1, 'condition', cond_key)
        frames.append(t)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    p = os.path.join(out_root, 'runs.csv')
    df.to_csv(p, index=False)
    print(f'    wrote {p}   ({len(df)} runs, converged = last {final_window} gens)')
    return df


def write_tables(cells, ranges, scope, out_root, final_window, figures):
    rows = []
    seen = set()
    for (env_key, cond_key), cell in sorted(cells.items()):
        for metric, _stem, _n, source in figures:
            if (env_key, cond_key, metric) in seen:
                continue
            v = converged(cell, metric, source, final_window)
            if not v.size:
                continue
            seen.add((env_key, cond_key, metric))
            rng = get_range(ranges, env_key, metric, scope)
            row = {'environment': env_key, 'condition': cond_key,
                   'metric': metric, 'source': source, 'n_runs': v.size,
                   'mean': v.mean(),
                   'std': v.std(ddof=1) if v.size > 1 else 0.0,
                   'min': v.min(), 'max': v.max(),
                   'q25': np.percentile(v, 25), 'median': np.median(v),
                   'q75': np.percentile(v, 75)}
            if rng and metric in FITNESS_METRICS:
                row['mean_normalised'] = float(rescale([v.mean()], rng)[0])
                row['median_normalised'] = float(rescale([np.median(v)], rng)[0])
                row['std_normalised'] = row['std'] / (rng[1] - rng[0])
            rows.append(row)
    summary = pd.DataFrame(rows)
    p = os.path.join(out_root, 'summary.csv')
    summary.to_csv(p, index=False)
    print(f'    wrote {p}   ({len(summary)} rows, converged = last {final_window} gens)')

    nrows = [{'scope': scope, 'group': g, 'metric': m, 'lo': lo, 'hi': hi,
              'span': hi - lo,
              'viability_4.375_normalised': (VIABILITY - lo) / (hi - lo) if hi > lo else np.nan}
             for (g, m), (lo, hi) in sorted(ranges.items())]
    p = os.path.join(out_root, 'normalisation.csv')
    pd.DataFrame(nrows).to_csv(p, index=False)
    print(f'    wrote {p}')
    return summary


def report(fitness, final_window):
    """The headline table, printed. One block per fitness metric."""
    if fitness.empty:
        return
    for metric in FITNESS_METRICS:
        sub = fitness[fitness['metric'] == metric]
        if sub.empty:
            continue
        print(f'\n=== converged {la.METRICS.get(metric, (metric,))[0].lower()} '
              f'(mean of last {final_window} generations per run) ===')
        print(f'  {"environment":<12}{"condition":<11}{"n":>4}{"mean":>9}{"± std":>9}'
              f'{"sem":>8}{"semC":>8}{"95% CI (clustered)":>22}'
              f'{"ICC":>7}{"eff n":>7}{"median":>9}{"nMed":>7}{"<thr":>6}')
        for _, r in sub.iterrows():
            ci = f'[{r["ci95_lo_cluster"]:.3f}, {r["ci95_hi_cluster"]:.3f}]'
            print(f'  {r["environment"]:<12}{r["condition"]:<11}{int(r["n_runs"]):>4}'
                  f'{r["mean"]:>9.3f}{r["std"]:>9.3f}{r["sem_naive"]:>8.3f}'
                  f'{r["sem_cluster"]:>8.3f}{ci:>22}'
                  f'{r["icc"]:>7.2f}{r["n_effective"]:>7.0f}'
                  f'{r["median"]:>9.3f}{r["median_normalised"]:>7.3f}'
                  f'{int(r["n_below_threshold"]):>6}')
        print(f'  (sem = naive sd/sqrt(n); semC = seed-clustered, and the CI is '
              f'built from it with t_(k-1); ICC/eff n show how far the 100 runs '
              f'fall short of 100\n   independent samples. nMed = median on the '
              f'[0,1] scale; <thr = runs below the viability threshold, '
              f'f = {VIABILITY:.3f})')


# ── CLI ───────────────────────────────────────────────────────────────────────

def report_survival(km_tab, cox, lr, sustain):
    """Time-to-viability tables, printed."""
    if not km_tab.empty:
        print(f'\n=== time to viability (event = {sustain} consecutive '
              f'generations at f >= {VIABILITY:.3f}) ===')
        print(f'  {"env":<10}{"condition":<11}{"n":>4}{"reached":>9}{"cens":>6}'
              f'{"median":>8}{"IQR":>16}{"by g100":>9}{"by g250":>9}{"by g500":>9}')
        for _, r in km_tab.iterrows():
            med = ('n/r' if not np.isfinite(r['median_gen'])
                   else f'{r["median_gen"]:.0f}')
            iqr = f'[{r["q25_gen"]:.0f}, {r["q75_gen"]:.0f}]'
            print(f'  {r["environment"]:<10}{r["condition"]:<11}'
                  f'{int(r["n_runs"]):>4}{int(r["n_reached"]):>9}'
                  f'{int(r["n_censored"]):>6}{med:>8}{iqr:>16}'
                  f'{r["reached_by_100"]:>9.2f}{r["reached_by_250"]:>9.2f}'
                  f'{r["reached_by_500"]:>9.2f}')
        print('  (median = Kaplan-Meier median generation; by gN = estimated '
              'fraction reached by generation N)')

    if not cox.empty:
        print('\n=== Cox proportional hazards (SEs clustered by map seed) ===')
        print(f'  {"env":<10}{"term vs evolution":<20}{"HR":>7}{"95% CI":>18}'
              f'{"p":>10}{"C-index":>9}{"PH p":>8}')
        for _, r in cox.iterrows():
            ci = f'[{r["hr_ci95_lo"]:.2f}, {r["hr_ci95_hi"]:.2f}]'
            print(f'  {r["environment"]:<10}{r["term"]:<20}'
                  f'{r["hazard_ratio"]:>7.2f}{ci:>18}{r["p"]:>10.2g}'
                  f'{r["concordance"]:>9.3f}{r["ph_assumption_min_p"]:>8.2g}')
        print('  (HR > 1 = reaches viability FASTER than the GA-only arm. '
              'PH p < 0.05 means the\n   hazard ratio is a time-average, not a '
              'constant effect — read the KM curves.)')

    if not lr.empty:
        print('\n=== log-rank across the three conditions ===')
        for _, r in lr.iterrows():
            print(f'  {r["environment"]:<10}chi2 = {r["test_statistic"]:8.1f}'
                  f'   p = {r["p"]:.3g}   (n = {int(r["n_runs"])})')


def report_variance(df):
    """Where the variance in converged fitness sits: terrain vs run-to-run."""
    if df.empty:
        return
    print('\n=== variance components of converged average fitness ===')
    print(f'  {"env":<10}{"condition":<11}{"---- raw fitness units ----":>31}'
          f'{"--- normalised [0,1] ---":>28}{"ICC":>7}')
    print(f'  {"":<21}{"total":>9}{"between":>11}{"within":>11}'
          f'{"total":>9}{"between":>10}{"within":>9}{"":>7}')
    for _, r in df.iterrows():
        print(f'  {r["environment"]:<10}{r["condition"]:<11}{r["sd_total"]:>9.4f}'
              f'{r["sd_between_seeds"]:>11.4f}{r["sd_within_seed"]:>11.4f}'
              f'{r["sd_total_normalised"]:>9.4f}{r["sd_between_normalised"]:>10.4f}'
              f'{r["sd_within_normalised"]:>9.4f}{r["icc"]:>7.2f}')
    print('  (sd between = terrain, one number per map; sd within = run-to-run '
          'chance on a fixed map.\n   ICC = var_between / (var_between + '
          'var_within) — a SHARE, so read it beside the absolute sds.)')


def report_stats(paired, boot, coll):
    """The three error-analysis tables, printed."""
    if not paired.empty:
        print('\n=== paired by map seed (k = 10 pairs, terrain differenced out) ===')
        print(f'  {"env":<10}{"comparison":<24}{"Δmean":>8}'
              f'{"95% CI":>20}{"dz":>7}{"p(t)":>10}{"p(Holm)":>10}'
              f'{"p(Wilcox)":>11}{"wins":>8}')
        for _, r in paired.iterrows():
            ci = f'[{r["ci95_lo"]:+.3f}, {r["ci95_hi"]:+.3f}]'
            comp = f'{r["condition_a"]} − {r["condition_b"]}'
            print(f'  {r["environment"]:<10}{comp:<24}{r["mean_diff"]:>+8.3f}'
                  f'{ci:>20}{r["cohens_dz"]:>7.2f}{r["p_paired_t"]:>10.2g}'
                  f'{r["p_holm"]:>10.2g}{r["p_wilcoxon"]:>11.2g}'
                  + f'{int(r["wins_a"])}/{int(r["n_seeds"])}'.rjust(8))
        print('  (Δ > 0 means the FIRST condition is fitter; wins = seeds favouring it)')

    if not boot.empty:
        print('\n=== seed-clustered bootstrap (whole seeds resampled) ===')
        print(f'  {"env":<10}{"condition":<11}{"mean":>8}{"mean 95% CI":>20}'
              f'{"median":>9}{"median 95% CI":>20}')
        for _, r in boot.iterrows():
            mci = f'[{r["mean_ci95_lo"]:.3f}, {r["mean_ci95_hi"]:.3f}]'
            dci = f'[{r["median_ci95_lo"]:.3f}, {r["median_ci95_hi"]:.3f}]'
            print(f'  {r["environment"]:<10}{r["condition"]:<11}{r["mean"]:>8.3f}'
                  f'{mci:>20}{r["median"]:>9.3f}{dci:>20}')

    if not coll.empty:
        print('\n=== collapse rate (runs below the viability threshold) ===')
        print(f'  {"env":<10}{"condition":<11}{"x/n":>8}{"rate":>7}'
              f'{"Wilson 95%":>18}{"clustered boot 95%":>22}{"seeds hit":>11}')
        for _, r in coll.iterrows():
            w = f'[{r["wilson_lo"]:.3f}, {r["wilson_hi"]:.3f}]'
            b = f'[{r["boot_lo"]:.3f}, {r["boot_hi"]:.3f}]'
            xn = f'{int(r["n_collapsed"])}/{int(r["n_runs"])}'
            print(f'  {r["environment"]:<10}{r["condition"]:<11}{xn:>8}'
                  f'{r["rate"]:>7.2f}{w:>18}{b:>22}'
                  + f'{int(r["seeds_with_any_collapse"])}/{int(r["n_seeds"])}'.rjust(11))
        print('  (quote the clustered bootstrap: collapse concentrates in particular seeds)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='output/conditions_by_environment')
    ap.add_argument('--band', choices=['std', 'sem', 'iqr'], default='std',
                    help='what the shaded region shows (default std)')
    ap.add_argument('--aggregate', choices=['runs', 'seeds'], default='runs',
                    help='band over runs, or over per-seed means (default runs)')
    ap.add_argument('--smooth', type=int, default=15,
                    help='centred rolling mean in generations (default 15; '
                         'pass 1 for raw per-generation values)')
    ap.add_argument('--final-window', type=int, default=50,
                    help='generations averaged for the converged value (default 50)')
    ap.add_argument('--max-gen', type=int, default=None,
                    help='truncate every run at this generation')
    ap.add_argument('--reps', type=int, default=None,
                    help='keep only the first N replicates of each seed')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated map seeds to keep (default all)')
    ap.add_argument('--norm-scope', choices=['global', 'env'], default='global',
                    help='min-max range over both environments (default) or per environment')
    ap.add_argument('--no-compare', action='store_true',
                    help='skip the side-by-side figures')
    ap.add_argument('--no-behaviour', action='store_true',
                    help='skip organisms.csv entirely (generations.csv figures only)')
    ap.add_argument('--behaviour-reps', type=int, default=None,
                    help='parse only the first N replicates per seed for behaviour')
    ap.add_argument('--no-stats', action='store_true',
                    help='skip the paired tests, bootstrap and collapse rates')
    ap.add_argument('--no-survival', action='store_true',
                    help='skip the time-to-viability survival analysis')
    ap.add_argument('--cross-sustain', type=int, default=10,
                    help='consecutive generations at or above the viability '
                         'threshold that count as reaching it (default 10)')
    ap.add_argument('--cache-dir', default=None,
                    help='where per-cell statistics are stored and read back '
                         '(default <out>/.cache_values). A cell with no logs on '
                         'disk is drawn and tested from this instead, so one '
                         'environment can be analysed after its logs are gone')
    ap.add_argument('--no-cache', action='store_true',
                    help='neither write nor read the value cache')
    ap.add_argument('--n-boot', type=int, default=10000,
                    help='bootstrap resamples (default 10000)')
    ap.add_argument('--formats', default='png,pdf',
                    help='comma-separated output formats (default png,pdf; '
                         'pdf/svg are vector and stay sharp at any zoom)')
    ap.add_argument('--dpi', type=int, default=300,
                    help='raster resolution for png/jpg (default 300)')
    args = ap.parse_args()

    fmts = tuple(f.strip().lower() for f in args.formats.split(',') if f.strip())
    if not fmts:
        raise SystemExit('--formats needs at least one format')
    OUTPUT['formats'] = fmts
    OUTPUT['dpi'] = args.dpi
    # Embed TrueType rather than the default Type-3 subsets: Type 3 is rejected
    # by some thesis-submission checkers and does not copy/paste as text.
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    seeds = ({int(s) for s in args.seeds.split(',') if s.strip()}
             if args.seeds else None)
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    cache_root = (None if args.no_cache else
                  (args.cache_dir or os.path.join(out_root, '.cache_values')))

    print('Discovering runs (params.json is authoritative) ...')
    cells = load_cells(ENVIRONMENTS, CONDITIONS, args.max_gen, args.final_window,
                       seeds, args.reps, cache_root=cache_root, args=args)
    if not cells:
        raise SystemExit('No runs matched, and no usable cache. '
                         'Check the log roots, the filters and --cache-dir.')

    figures = [(m, s, n, 'gen') for m, s, n in GEN_FIGURES]
    if not args.no_behaviour:
        print('\nBehaviour (organisms.csv; cached under .cache_behaviour/) ...')
        n_beh = load_behaviour(cells, os.path.join(out_root, '.cache_behaviour'),
                               args.max_gen, args.behaviour_reps)
        print(f'  {n_beh} run(s) contributed behaviour')
        figures += [(m, f'behaviour_{m}', False, 'behaviour')
                    for m in BEHAVIOUR_ORDER]

    ranges = normalisation_ranges(cells, FITNESS_METRICS, args.norm_scope)
    print(f'\nNormalisation ({args.norm_scope} scope):')
    for (g, m), (lo, hi) in sorted(ranges.items()):
        print(f'  {m:<24} {g:<9} lo={lo:8.4f}  hi={hi:8.4f}  span={hi - lo:8.4f}')

    if cache_root:
        print(f'\nValue cache -> {cache_root}')
        write_cache(cells, cache_root, figures, args, seeds)

    print('\nPer-environment figures:')
    for env in ENVIRONMENTS:
        if any(k[0] == env['key'] for k in cells):
            per_environment(cells, env, CONDITIONS, ranges, args.norm_scope,
                            args, out_root, figures)

    if not args.no_compare and len({k[0] for k in cells}) > 1:
        print('\nComparison figures:')
        comparisons(cells, ENVIRONMENTS, CONDITIONS, ranges, args.norm_scope,
                    args, out_root, figures)

    print('\nTables:')
    write_tables(cells, ranges, args.norm_scope, out_root,
                 args.final_window, figures)
    fitness = write_fitness_summary(cells, ranges, args.norm_scope, out_root,
                                    args.final_window)
    write_runs_table(cells, ranges, args.norm_scope, out_root,
                     args.final_window)
    variance = write_variance_components(cells, ENVIRONMENTS, CONDITIONS,
                                         out_root, ranges, args.norm_scope)
    report(fitness, args.final_window)
    report_variance(variance)

    if not args.no_survival:
        print('\nTime to viability (survival analysis):')
        tt = time_to_threshold(cells, ENVIRONMENTS, CONDITIONS,
                               sustain=args.cross_sustain)
        p_tt = os.path.join(out_root, 'time_to_threshold.csv')
        tt.to_csv(p_tt, index=False)
        print(f'    wrote {p_tt}   ({len(tt)} runs)')
        if cache_root:
            write_cache_survival(cells, cache_root, tt, args)
        _fits, km_tab = km_curves(tt, ENVIRONMENTS, CONDITIONS)
        cox = cox_models(tt, ENVIRONMENTS, CONDITIONS)
        lr = logrank_by_env(tt, ENVIRONMENTS)
        for df_, name in ((km_tab, 'time_to_threshold_km.csv'),
                          (cox, 'time_to_threshold_cox.csv'),
                          (lr, 'time_to_threshold_logrank.csv')):
            path = os.path.join(out_root, name)
            df_.to_csv(path, index=False)
            print(f'    wrote {path}   ({len(df_)} rows)')
        for env in ENVIRONMENTS:
            if any(k[0] == env['key'] for k in cells):
                rngv = get_range(ranges, env['key'], 'avg_fitness',
                                 args.norm_scope)
                tn = (rescale([VIABILITY], rngv)[0] if rngv else None)
                fig_km(tt, env, CONDITIONS, args.cross_sustain,
                       os.path.join(out_root, env['key'],
                                    'time_to_viability_km.png'), thr_norm=tn)
        if len({k[0] for k in cells}) > 1:
            rngv = get_range(ranges, ENVIRONMENTS[0]['key'], 'avg_fitness',
                             args.norm_scope)
            tn = (rescale([VIABILITY], rngv)[0] if rngv else None)
            fig_km_compare(tt, ENVIRONMENTS, CONDITIONS, args.cross_sustain,
                           os.path.join(out_root, 'compare',
                                        'compare_time_to_viability_km.png'),
                           thr_norm=tn)
        report_survival(km_tab, cox, lr, args.cross_sustain)

    if not args.no_stats:
        print('\nError analysis (seed = unit of replication):')
        paired = paired_by_seed(cells, ENVIRONMENTS, CONDITIONS)
        boot = cluster_bootstrap(cells, ENVIRONMENTS, CONDITIONS,
                                 n_boot=args.n_boot)
        coll = collapse_rates(cells, ENVIRONMENTS, CONDITIONS,
                              n_boot=args.n_boot)
        for df, name in ((paired, 'paired_by_seed.csv'),
                         (boot, 'bootstrap_ci.csv'),
                         (coll, 'collapse_rate.csv')):
            path = os.path.join(out_root, name)
            df.to_csv(path, index=False)
            print(f'    wrote {path}   ({len(df)} rows)')
        report_stats(paired, boot, coll)
    print(f'\nDone -> {out_root}')


if __name__ == '__main__':
    main()
