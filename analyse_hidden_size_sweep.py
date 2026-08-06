"""
analyse_hidden_size_sweep.py
======================================================================
Effect of NETWORK CAPACITY (NNBrain hidden-layer width) on fitness, population,
lifetime/mortality and behaviour, compared ACROSS ALL THREE CONDITIONS, from the
matched trio produced by

    run_hidden_size_sweep_evolution_array.slurm   evolution/standard  (GA only)
    run_hidden_size_sweep_learning_array.slurm    learning/standard   (GA + RL)
    run_hidden_size_sweep_pure_rl_array.slurm     learning/pure_rl    (RL only)

NNBrain is a single-hidden-layer policy network, 54 -> HIDDEN -> 6, so the width
sets the size of the flat genome every organism carries:

    GENOME_SIZE = 61*H + 6      H =  32 ->  1958 weights  (half capacity)
                                H =  64 ->  3910 weights  (the default control)
                                H = 128 ->  7814 weights  (double capacity)

What the width MEANS differs per condition, which is exactly why the three are
plotted side by side rather than in three separate reports:

    evolution   the genome is the only thing that changes across generations, so
                the width IS the dimensionality of the GA's search space —
                doubling it doubles what mutation has to get right at a fixed
                mut_prob / mut_sigma and a fixed 1000-generation budget. The arm
                where more capacity is most likely to HURT.
    learning    GA + within-life RL: extra capacity is searched by both channels.
    pure_rl     no GA at all; capacity is only ever moved by RL, and inheritance
                across a collapse runs through the 100-slot dead-weight buffer.

FIGURE STRUCTURE (the same in every figure this script writes)
-------------------------------------------------------------
    one COLUMN per condition   (evolution | learning | pure RL)
    one ROW per metric
    one LINE per hidden size   (32 / 64 / 128), mean +/- std band

so each panel answers "what did width do HERE" and each row answers "did it do
the same thing in the other two mechanisms". Panels in a row share a y-axis by
default (--free-y turns that off) — a width effect that looks large on a free
axis is often small next to the between-condition gap, and the shared row makes
that visible instead of hiding it.

ENVIRONMENTS ARE NOT MIXED. baseline and hard ran at different learning rates
(0.01 vs 0.02 for the two RL conditions), so folding them onto one axis would
confound width with lr. Each environment gets its own output directory, as in
analyse_buffer_sweep.py / analyse_trace_decay_sweep.py.

AGGREGATION
-----------
Every run is one unit: the 15 runs of a (condition, environment, width) cell —
3 seeds x 5 replicates — are averaged into one curve, and the band is the
standard deviation across those 15 runs. Seeds are NOT a separate level of
aggregation, so the band carries both map-to-map variation and run-to-run
algorithmic chance (only the map TERRAIN is seeded; founder weights are an
unseeded Xavier draw and RL action sampling is unseeded too). The width
comparison below tests arms at that same run level (n = 15 vs 15).

Outputs (into --out / output/hidden_size_sweep_<env>):
    hid_fitness_population.png -- avg / top-20% / best fitness, peak population,
                                  total agents.            <- the headline figure
    hid_lifetime.png           -- mean/median/p90 lifetime and the death-cause
                                  split (survived / drained / starved).
    hid_behaviour.png          -- per-organism scalar behaviours (cells explored,
                                  cave entries, predator encounters, ticks
                                  drained, food eaten).
    hid_behaviour_food.png     -- diet composition, one row per food tier.
    hid_cave_day_night.png     -- cave entries day vs night + the night share.
    hid_diagnostics.png        -- MAD, genomic variance, weight magnitude and
                                  inter-generation weight change. Capacity moves
                                  these mechanically (more weights = more to
                                  move), so read them as mechanism, not outcome.
    hid_vs_width.png           -- final-window value of the key metrics plotted
                                  AGAINST width, one line per condition. The
                                  at-a-glance "which width won, and did the three
                                  mechanisms agree?" summary.
    hid_consistency.png        -- RUN CONSISTENCY: how many of each arm's 15 runs
                                  ended in each fitness region, plus the runs
                                  themselves against the same band edges. See
                                  below.
    hid_curves_summary.csv     -- tidy (condition, hidden_size, generation,
                                  metric, mean, std).
    hid_final_window.csv       -- per-run final-window values behind hid_vs_width.
    hid_width_comparison.csv   -- run-level test of each width against the 64
                                  control, within each condition (difference,
                                  % change, Mann-Whitney p, Cliff's delta).
    hid_run_bands.csv          -- per run: final-window fitness, its region, its
                                  late growth and whether it was still climbing.
    hid_band_counts.csv        -- runs per (condition, width, region).

RUN CONSISTENCY
---------------
An arm's mean curve hides the thing the hard environment actually does: its runs
do not scatter around a mean, they split into a group that takes off and a group
that plateaus low and never recovers. An arm whose mean is 12% higher because
10/15 runs took off instead of 4/15 is a CONSISTENCY result, not a
performance-per-run one, so the two are reported separately.

The regions are cut from the POOLED final-window fitness of the environment (all
conditions and widths together), so the same edges apply to every arm and the
counts are comparable across columns. Two schemes, chosen by the data:
    takeoff   one dominant gap in the sorted values — wider than --gap-frac of
              the range AND 3x the next-largest gap — so the runs genuinely fall
              into two separated groups. Split there. (The hard environment: the
              gap is ~1.4 fitness wide.)
    quartile  no such gap; the runs are one continuous spread, so a two-way split
              would invent a boundary. Fall back to the pooled quartiles. (The
              baseline environment.)
Each run is also marked "still climbing" or "plateaued" by comparing its final
window against the equally sized window before it (--plateau-tol), which
separates a run that had converged from one the generation budget cut short.

Reading 270 organisms.csv files (~100 MB each) is the slow part, so each run's
per-generation curve is cached under --cache-dir (keyed on the CSV's size+mtime)
and the first pass is built in parallel (--workers). Re-plotting with a different
--smooth or --final-frac is then near-instant.

Usage
-----
    python analyse_hidden_size_sweep.py                    # both envs
    python analyse_hidden_size_sweep.py --env hard
    python analyse_hidden_size_sweep.py --conditions evolution learning
    python analyse_hidden_size_sweep.py --seeds 1 42
    python analyse_hidden_size_sweep.py --free-y --smooth 10
    python analyse_hidden_size_sweep.py --no-cache         # force re-read
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

try:
    from scipy import stats as _scipy_stats
except ImportError:                                    # tests degrade to n/a
    _scipy_stats = None

# ── The three conditions, in plot order ───────────────────────────────────────
# Each is a (condition, mode) pair in params.json living under its own log root.
# 'learning'/'pure_rl' share the condition field, so mode is what separates them.
CONDITIONS = {
    'evolution': {
        'root':  'logs/evolution/standard/auto-run',
        'condition': 'evolution', 'mode': 'standard',
        'label': 'Evolution  (GA only, no RL)',
    },
    'learning': {
        'root':  'logs/learning/standard/auto-run',
        'condition': 'learning', 'mode': 'standard',
        'label': 'Learning  (GA + RL)',
    },
    'pure_rl': {
        'root':  'logs/learning/pure_rl/auto-run',
        'condition': 'learning', 'mode': 'pure_rl',
        'label': 'Pure RL  (no GA, buffer 100)',
    },
}
COND_ORDER = ['evolution', 'learning', 'pure_rl']

# ── Environment presets ───────────────────────────────────────────────────────
# lr differs per environment for the two RL conditions (the evolution arm runs no
# RL at all, so its recorded lr is inert and is not matched on).
ENVIRONMENTS = {
    'baseline': {'roaming': 0.0,  'drain': None, 'desc': 'predator-free baseline'},
    'hard':     {'roaming': 60.0, 'drain': 5.0,
                 'desc': '60 roaming predators, drain 5'},
}

# The default width every earlier run in this codebase used; the sweep's matched
# control and the reference arm the width comparison is taken against.
_DEFAULT_HIDDEN = 64

# ── Generation-level outcome metrics (from generations.csv) ───────────────────
GEN_METRICS = {
    'avg_fitness':          'Mean average fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'best_fitness':         'Best fitness',
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
}

# ── Lifetime / mortality (avg_lifetime from generations.csv, rest derived from
# organisms.csv) ──────────────────────────────────────────────────────────────
# Mean, median and p90 are kept apart because capacity can lengthen the tail
# without moving the middle: a wider net that saves the best organisms and no one
# else shows up in p90 first. The death-cause shares say WHY the lifetimes moved
# — starvation (a foraging failure) and drained (a predator failure) are
# different verdicts on the same drop.
LIFE_METRICS = {
    'avg_lifetime':    'Mean lifetime (ticks)',
    'lifetime_median': 'Median lifetime (ticks)',
    'lifetime_p90':    'p90 lifetime (ticks)',
    'survived_frac':   'Survived to window close',
    'drained_frac':    'Killed by predators',
    'starved_frac':    'Starved',
}

# ── Scalar per-organism behaviours (from organisms.csv) ───────────────────────
BEHAV_METRICS = {
    'cells_visited':    'Cells explored',
    'cave_entries':     'Cave entries',
    'predator_touches': 'Predator encounters',
    'drained_ticks':    'Ticks drained by predators',
    'food_total':       'Food eaten (total)',
}

# ── Learning diagnostics (generations.csv) ────────────────────────────────────
# Width moves these mechanically — MAD and the weight magnitude are computed over
# 61H+6 weights — so they are mechanism, not outcome. Included because they are
# the only view of WHERE the capacity went: a 128 arm whose MAD stays flat never
# used the extra weights.
DIAG_METRICS = {
    'avg_learned_weight_diff': 'Learned weight diff (MAD)',
    'genome_variance':         'Genomic variance',
    'avg_network_weight_mag':  'Network weight magnitude (RMS)',
    'inter_gen_weight_change': 'Inter-generation weight change',
}
# Span decades within a run — drawn on a log y-axis, else the early transient
# flattens the rest of the curve into the axis.
LOG_METRICS = {'avg_learned_weight_diff', 'genome_variance'}

GEN_COLS = list(GEN_METRICS) + ['avg_lifetime'] + list(DIAG_METRICS)

# ── Compositional behaviours ──────────────────────────────────────────────────
FOOD_TIERS = {
    'food_default':  'default',
    'food_low':      'low',
    'food_medium':   'medium',
    'food_prestige': 'prestige',
}
FOOD_COLS = list(FOOD_TIERS)

DEATH_CAUSES = ['survived', 'drained', 'starved']

SCALAR_COLS = ['cells_visited', 'cave_entries', 'predator_touches',
               'drained_ticks', 'lifetime', 'energy_at_death',
               'cave_entries_day', 'cave_entries_night', 'food_total']

USE_COLS = (['generation', 'cells_visited', 'predator_touches', 'drained_ticks',
             'cave_entries', 'cave_entries_day', 'cave_entries_night', 'lifetime',
             'energy_at_death', 'death_cause'] + FOOD_COLS)

# Metrics for the final-window "vs width" summary. Deliberately mixes the four
# families the sweep is judged on: outcome, lifetime, diagnostic, behaviour.
VS_WIDTH = [
    ('avg_fitness',             'Mean average fitness'),
    ('top20percent_fitness',    'Top-20% fitness'),
    ('peak_population',         'Peak population'),
    ('total_agents',            'Total agents'),
    ('avg_lifetime',            'Mean lifetime (ticks)'),
    ('lifetime_p90',            'p90 lifetime (ticks)'),
    ('survived_frac',           'Survived to window close'),
    ('avg_learned_weight_diff', 'Learned weight diff (MAD)'),
    ('genome_variance',         'Genomic variance'),
    ('cells_visited',           'Cells explored / organism'),
    ('cave_entries',            'Cave entries / organism'),
    ('food_total',              'Food eaten / organism'),
]

# Sweep-run gate + width source: only folders carrying an _h<N>_ tag are this
# sweep's runs. The tag is cross-checked against params.json below — a clone that
# predates --hidden-size silently runs every arm at 64, which would look like a
# null result instead of a mistake.
_H_RE = re.compile(r'_h(\d+)_')
_REP_RE = re.compile(r'_r(\d+)$')

# One colour per width: red 32, green 64 (the control), blue 128. A 4+-value
# sweep falls back to a sequential ramp in run_env(), since width is an ordered
# axis and three arbitrary hues stop working once there are more of them.
_WIDTH_COLOURS_3 = ['#D62728', '#2CA02C', '#1F77B4']
# Conditions deliberately AVOID red/green/blue: they key the vs-width and
# consistency figures, and reusing the width hues there would make the same
# colour mean two different things across the output set.
_COND_COLOURS = {'evolution': '#7C3AED', 'learning': '#0D9488', 'pure_rl': '#B45309'}

# Bands for the run-consistency view, low fitness -> high.
_BAND_COLOURS_2 = ['#B91C1C', '#15803D']
_BAND_COLOURS_4 = ['#B91C1C', '#F59E0B', '#65A30D', '#15803D']

# Set per-env in run_env(); used by the plot titles / filenames.
_ENV_DESC = ''
_SUFFIX = ''
_BAND_DESC = ''
_HID_COLOURS = {}   # hidden size -> colour, filled per run.


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def _hid_label(h):
    """Consistent display label for a width, flagging the control arm and showing
    the genome size it implies (61H + 6), which is the quantity that actually
    changes for the GA."""
    tag = '  (default)' if h == _DEFAULT_HIDDEN else ''
    return f'hidden {h}  ({61 * h + 6} weights){tag}'


def discover(cond_key, env_key, roaming, drain, seed_filter, root=None):
    """Find one condition's hidden-size sweep run folders for one environment.

    A run qualifies when it has params.json + organisms.csv, an _h<N>_ tag in its
    folder name, and its params match the condition/mode and the environment's
    predator settings. Returns {hidden, seed, rep, name, dir, org_csv} dicts.

    A run whose _h<N>_ tag disagrees with its recorded hidden_size is DROPPED
    with a warning: that is the signature of a clone that ignored --hidden-size,
    and folding it into the arm named by its folder would quietly average the
    control into the treatment."""
    spec = CONDITIONS[cond_key]
    root = root or spec['root']
    runs, mismatched = [], []
    for params_path in glob.glob(os.path.join(root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        org_csv = os.path.join(run_dir, 'organisms.csv')
        if not os.path.exists(org_csv):
            continue
        name = os.path.basename(run_dir)
        m = _H_RE.search(name)
        if not m:      # no _h tag => not a sweep run
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
        except (ValueError, OSError):
            continue
        if str(p.get('condition')) != spec['condition']:
            continue
        if str(p.get('mode')) != spec['mode']:
            continue
        if abs(_num(p, 'roaming_predator_count') - roaming) > 1e-9:
            continue
        if drain is not None and abs(_num(p, 'predator_drain') - drain) > 1e-9:
            continue
        hid = int(m.group(1))
        recorded = _num(p, 'hidden_size')
        if np.isfinite(recorded) and int(recorded) != hid:
            mismatched.append((name, hid, int(recorded)))
            continue
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        rm = _REP_RE.search(name)
        rep = int(rm.group(1)) if rm else 1
        runs.append({'hidden': hid, 'seed': seed, 'rep': rep, 'name': name,
                     'dir': run_dir, 'org_csv': org_csv,
                     'gen_csv': os.path.join(run_dir, 'generations.csv')})
    for name, tagged, recorded in mismatched:
        print(f"  WARNING: {name} is tagged h{tagged} but ran with "
              f"hidden_size={recorded} — dropped (stale clone?).")
    return sorted(runs, key=lambda r: (r['hidden'], r['seed'], r['rep']))


def _cache_path(cache_dir, run_name, org_csv):
    """Cache key = run name + the organisms.csv size/mtime, so an updated log
    invalidates its own entry without touching the rest."""
    st = os.stat(org_csv)
    return os.path.join(cache_dir, f'{run_name}__{st.st_size}_{int(st.st_mtime)}.csv')


def load_run(org_csv, run_name, cache_dir):
    """Read one run's organisms.csv; return a per-generation DataFrame (indexed by
    generation) with, for each generation: the mean of every scalar behaviour, the
    lifetime median and 90th percentile, the death-cause shares, the
    food_<tier>_frac diet shares, and night_share (the day-vs-night cave
    preference). Missing columns degrade to NaN rather than crashing on older
    logs. Cached per run when cache_dir is not None."""
    cache_file = _cache_path(cache_dir, run_name, org_csv) if cache_dir else None
    if cache_file and os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file, index_col='generation')
        except (ValueError, OSError):
            pass   # unreadable cache entry: fall through and re-read the log

    available = pd.read_csv(org_csv, nrows=0).columns
    cols = [c for c in USE_COLS if c in available]
    df = pd.read_csv(org_csv, usecols=cols)
    for c in USE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if not len(df):
        return pd.DataFrame()

    for c in SCALAR_COLS + FOOD_COLS:
        if c != 'food_total':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['food_total'] = df[FOOD_COLS].sum(axis=1)

    grouped = df.groupby('generation')
    scal = grouped[SCALAR_COLS].mean()

    # Lifetime spread. Mean alone hides a capacity effect that only reaches the
    # top of the distribution, which is the shape the GA cares about — it selects
    # on the top, not the average.
    life = grouped['lifetime'].quantile([0.5, 0.9]).unstack()
    life.columns = ['lifetime_median', 'lifetime_p90']

    # Death-cause shares, as a fraction of the generation's organisms. Reindexed
    # over the known causes so a generation (or an environment) in which nobody
    # was drained yields 0.0 rather than a missing column.
    cause = df['death_cause'].astype(str)
    dummies = pd.DataFrame({f'{c}_frac': (cause == c).astype(float)
                            for c in DEATH_CAUSES})
    dummies['generation'] = df['generation'].values
    causes = dummies.groupby('generation').mean()

    fsum = grouped[FOOD_COLS].sum()
    ffrac = fsum.div(fsum.sum(axis=1).replace(0, np.nan), axis=0)
    ffrac.columns = [f'{c}_frac' for c in FOOD_COLS]

    # Night share from the generation's TOTAL entries, not the mean of per-organism
    # ratios: organisms that never entered a cave have no ratio, and one that
    # entered once shouldn't outvote one that entered fifty times.
    csum = grouped[['cave_entries_day', 'cave_entries_night']].sum()
    total = (csum['cave_entries_day'] + csum['cave_entries_night']).replace(0, np.nan)
    night_share = (csum['cave_entries_night'] / total).rename('night_share')

    out = pd.concat([scal, life, causes, ffrac, night_share], axis=1).sort_index()
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_csv(cache_file, index_label='generation')
    return out


def _cache_one(task):
    """Pool worker: build (and write) one run's cache entry. Returns (name, ok).
    Only the cache file matters — the parent re-reads it — so nothing large is
    pickled back across the process boundary."""
    org_csv, run_name, cache_dir = task
    try:
        load_run(org_csv, run_name, cache_dir)
        return run_name, True
    except Exception as exc:                              # noqa: BLE001
        return run_name, f'{type(exc).__name__}: {exc}'


def build_cache(runs, cache_dir, workers):
    """Populate the per-run cache in parallel for every run that has no entry yet.

    Each organisms.csv is ~100 MB and there are up to 270 of them, so the first
    pass is the expensive one; subsequent runs of this script hit the cache and
    skip straight to plotting. No-op when caching is off (the sequential path in
    aggregate_condition() then reads each log directly)."""
    if not cache_dir or workers <= 1:
        return
    todo = [(r['org_csv'], r['name'], cache_dir) for r in runs
            if not os.path.exists(_cache_path(cache_dir, r['name'], r['org_csv']))]
    if not todo:
        return
    os.makedirs(cache_dir, exist_ok=True)
    from multiprocessing import Pool
    print(f"  Building cache for {len(todo)} run(s) on {workers} workers "
          f"(~100 MB each, first pass only) ...")
    done = 0
    with Pool(workers) as pool:
        for name, ok in pool.imap_unordered(_cache_one, todo):
            done += 1
            if ok is not True:
                print(f"    WARNING: {name} failed to cache — {ok}")
            elif done % 10 == 0 or done == len(todo):
                print(f"    cached {done}/{len(todo)}")


def load_gen_metrics(gen_csv, max_gen):
    """Read the outcome + diagnostic metrics (GEN_COLS) from a run's
    generations.csv, indexed by generation. Deduplicates on generation (keep
    last). None if unavailable."""
    if not os.path.exists(gen_csv):
        return None
    head = pd.read_csv(gen_csv, nrows=0).columns
    if 'generation' not in head:
        return None
    have = [c for c in GEN_COLS if c in head]
    if not have:
        return None
    df = pd.read_csv(gen_csv, usecols=['generation'] + have)
    df = df[df['generation'] <= max_gen]
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    for c in have:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('generation')[have]


def stack_mean_std(unit_curves):
    """Aggregate an arm's per-unit curves -> (mean, std) DataFrames indexed by
    generation. unit_curves: list of (key, DataFrame)."""
    stacked = pd.concat({key: curve for key, curve in unit_curves},
                        names=['unit', 'generation'])
    return (stacked.groupby(level='generation').mean(),
            stacked.groupby(level='generation').std())


def make_units(run_frames):
    """Turn an arm's per-run curves into the units the mean +/- std is taken over:
    every run is one unit, pooled across seeds AND replicates.

    run_frames: list of (seed, rep, DataFrame). Returns list of (key, DataFrame)
    with scalar string keys — stack_mean_std builds a (unit, generation)
    MultiIndex, so a tuple key would add a level and collide with the frames' own
    'generation' index name."""
    return [(f'seed{seed}_r{rep}', df)
            for seed, rep, df in sorted(run_frames, key=lambda t: (t[0], t[1]))]


def aggregate_condition(runs, cache_dir, max_gen):
    """Load + aggregate one condition's runs into
    (stats, per_unit) = ({hidden: (mean_df, std_df)}, {hidden: [(unit, df)]}).

    The organisms-derived curve and the generations.csv outcomes/diagnostics are
    joined per RUN, before aggregating: both come from the same run, so the join
    on generation is exact and every downstream unit carries all families at
    once."""
    by_hid = {}
    for r in runs:
        by_hid.setdefault(r['hidden'], []).append(r)

    stats, per_unit = {}, {}
    for hid in sorted(by_hid):
        run_frames = []
        for r in by_hid[hid]:
            curve = load_run(r['org_csv'], r['name'], cache_dir)
            if not len(curve):
                print(f"    skip h{hid} seed {r['seed']} r{r['rep']} — no rows")
                continue
            curve = curve[curve.index <= max_gen]
            gm = load_gen_metrics(r['gen_csv'], max_gen)
            if gm is not None and len(gm):
                curve = curve.join(gm, how='outer')
            run_frames.append((r['seed'], r['rep'], curve))
        if not run_frames:
            continue
        units = make_units(run_frames)
        mean_df, std_df = stack_mean_std(units)
        stats[hid] = (mean_df, std_df)
        per_unit[hid] = units
        print(f"    hidden {hid:<4d} averaged over {len(units)} runs, "
              f"{int(mean_df.index.max())} gens")
    return stats, per_unit


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw(ax, cond_stats, col, smooth, logy=False):
    """One line per hidden size (mean +/- std band) for column `col` in one
    condition's panel. Returns True if anything was drawn.
    cond_stats: {hidden: (mean_df, std_df)}, ascending."""
    drawn = False
    for hid, (mean_df, std_df) in sorted(cond_stats.items()):
        if col not in mean_df.columns:
            continue
        s = mean_df[col].dropna()
        if not len(s):
            continue
        sd = std_df[col].reindex(s.index).fillna(0) if col in std_df.columns else \
            pd.Series(0.0, index=s.index)
        if smooth > 1:
            s = s.rolling(smooth, min_periods=1, center=True).mean()
            sd = sd.rolling(smooth, min_periods=1, center=True).mean()
        lo, hi = s.values - sd.values, s.values + sd.values
        if logy:
            # A log axis cannot show a band that reaches <= 0; clip it to a decade
            # below the smallest positive mean rather than dropping the band.
            pos = s.values[s.values > 0]
            floor = (pos.min() * 0.1) if len(pos) else 1e-12
            lo = np.maximum(lo, floor)
        colour = _HID_COLOURS[hid]
        ax.plot(s.index, s.values, color=colour, lw=2, label=_hid_label(hid))
        ax.fill_between(s.index, lo, hi, color=colour, alpha=0.12)
        drawn = True
    return drawn


def _has_signal(all_stats, col):
    """True if any condition/arm's mean curve for `col` is non-empty and not
    identically zero. Drops the rows that are structurally flat rather than
    uninformative — predator_touches / drained_ticks / drained_frac in the
    predator-free baseline, where an all-zero row would waste a band of the
    figure."""
    for cond_stats in all_stats.values():
        for mean_df, _sd in cond_stats.values():
            if col not in mean_df.columns:
                continue
            s = mean_df[col].dropna()
            if len(s) and (s.abs() > 0).any():
                return True
    return False


def _share_row_y(axs_row, log_row):
    """Give every panel in a row the same y-limits, so the width effect in one
    condition is read against the same scale as the others. Log rows are shared
    too — the decades being comparable is the whole point — and the scale is
    re-applied afterwards because set_ylim on a log axis can reset it."""
    lims = [ax.get_ylim() for ax in axs_row if ax.has_data()]
    if len(lims) < 2:
        return
    lo = min(l for l, _h in lims)
    hi = max(h for _l, h in lims)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    for ax in axs_row:
        ax.set_ylim(lo, hi)
        if log_row:
            ax.set_yscale('log')


def _grid_plot(all_stats, conditions, metrics, titles, smooth, out_dir, fname,
               suptitle, share_y=True, ylim_bottom_zero=False,
               log_cols=frozenset()):
    """THE shared figure layout for this script: one column per condition, one row
    per metric, one line per hidden size.

    all_stats: {cond_key: {hidden: (mean_df, std_df)}}. Rows with no signal in any
    condition are dropped before laying the figure out."""
    metrics = [m for m in metrics if _has_signal(all_stats, m)]
    if not metrics:
        print(f"  (no columns for {fname} — skipping)")
        return
    nrows, ncols = len(metrics), len(conditions)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.3 * nrows),
                            sharex=True, squeeze=False)

    for r, m in enumerate(metrics):
        logy = m in log_cols
        for c, ck in enumerate(conditions):
            ax = axs[r][c]
            _draw(ax, all_stats.get(ck, {}), m, smooth, logy=logy)
            ax.grid(True, alpha=0.3)
            ax.margins(x=0)
            if logy:
                ax.set_yscale('log')
            elif ylim_bottom_zero:
                ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(CONDITIONS[ck]['label'], fontsize=12,
                             fontweight='bold', pad=10)
            if r == nrows - 1:
                ax.set_xlabel('Generation')
            if c == 0:
                ax.set_ylabel(titles[m] + (' (log)' if logy else ''),
                              fontsize=11, fontweight='bold')
        if share_y:
            _share_row_y(axs[r], logy)

    handles, labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   fontsize=10, frameon=False,
                   title='NNBrain hidden width (54 -> H -> 6)')
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    # Leave room for the figure-level legend strip below the axes and the
    # two-line suptitle above them; rect is in figure fractions, so it has to
    # scale with the row count or a tall figure wastes half its height.
    pad = 0.42 / nrows
    plt.tight_layout(rect=(0, min(0.10, pad), 1, 1 - min(0.06, pad * 0.9)))
    path = os.path.join(out_dir, f'{fname}{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _title(head, tail):
    return f'{head} — network-capacity sweep\n({tail}; {_ENV_DESC}; mean ± std {_BAND_DESC})'


def plot_fitness_population(all_stats, conditions, smooth, out_dir, share_y):
    """The headline figure: fitness + population over generations, one column per
    condition, one line per width."""
    print("  Plotting fitness + population ...")
    _grid_plot(all_stats, conditions, list(GEN_METRICS), GEN_METRICS, smooth,
               out_dir, 'hid_fitness_population',
               _title('Fitness & population over generations',
                      'per-generation population outcomes'),
               share_y=share_y)


def plot_lifetime(all_stats, conditions, smooth, out_dir, share_y):
    """Lifetime distribution + why organisms died, same layout."""
    print("  Plotting lifetime + mortality ...")
    _grid_plot(all_stats, conditions, list(LIFE_METRICS), LIFE_METRICS, smooth,
               out_dir, 'hid_lifetime',
               _title('Lifetime & mortality over generations',
                      'mean/median/p90 lifetime and the death-cause split'),
               share_y=share_y)
    # NOT ylim_bottom_zero: lifetimes sit around 2000 ticks and the death-cause
    # shares around 0.95, so anchoring either at zero flattens every curve — and
    # the width differences are exactly what would be flattened away.


def plot_behaviour(all_stats, conditions, smooth, out_dir, share_y):
    """Per-organism scalar behaviours, same layout."""
    print("  Plotting behaviour curves ...")
    _grid_plot(all_stats, conditions, list(BEHAV_METRICS), BEHAV_METRICS, smooth,
               out_dir, 'hid_behaviour',
               _title('Behaviour over generations', 'population average per organism'),
               share_y=share_y)


def plot_food(all_stats, conditions, smooth, out_dir, share_y):
    """Diet composition, one row per food tier, same layout."""
    print("  Plotting food-tier composition ...")
    titles = {f'{c}_frac': f'{FOOD_TIERS[c]} food (share of diet)' for c in FOOD_COLS}
    _grid_plot(all_stats, conditions, list(titles), titles, smooth, out_dir,
               'hid_behaviour_food',
               _title('Food-tier composition over generations',
                      'share of the generation\'s diet in each tier'),
               share_y=share_y, ylim_bottom_zero=True)


def plot_cave_day_night(all_stats, conditions, smooth, out_dir, share_y):
    """Cave entries day vs night plus the night share, same layout. The 0.5 line
    on the share row marks no preference."""
    print("  Plotting cave entries by time of day ...")
    titles = {'cave_entries_day':   'Cave entries — day',
              'cave_entries_night': 'Cave entries — night',
              'night_share':        'Night share'}
    metrics = [m for m in titles if _has_signal(all_stats, m)]
    if not metrics:
        print("  (no cave day/night columns — skipping)")
        return
    nrows, ncols = len(metrics), len(conditions)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.3 * nrows),
                            sharex=True, squeeze=False)
    for r, m in enumerate(metrics):
        for c, ck in enumerate(conditions):
            ax = axs[r][c]
            _draw(ax, all_stats.get(ck, {}), m, smooth)
            if m == 'night_share':
                ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.margins(x=0)
            if r == 0:
                ax.set_title(CONDITIONS[ck]['label'], fontsize=12,
                             fontweight='bold', pad=10)
            if r == nrows - 1:
                ax.set_xlabel('Generation')
            if c == 0:
                ax.set_ylabel(titles[m], fontsize=11, fontweight='bold')
        if share_y:
            _share_row_y(axs[r], False)
    # Day and night rows also share a y-axis WITH EACH OTHER, so "more at night"
    # is readable off the panels rather than off the axis labels.
    day_night = [r for r, m in enumerate(metrics) if m != 'night_share']
    if len(day_night) == 2:
        flat = [axs[r][c] for r in day_night for c in range(ncols)]
        top = max((ax.get_ylim()[1] for ax in flat if ax.has_data()), default=1)
        for ax in flat:
            ax.set_ylim(0, top)

    handles, labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   fontsize=10, frameon=False,
                   title='NNBrain hidden width (54 -> H -> 6)')
    plt.suptitle(_title('Cave entries by time of day',
                        'population average per organism; 0.5 = no preference'),
                 fontsize=14, fontweight='bold')
    pad = 0.42 / nrows
    plt.tight_layout(rect=(0, min(0.10, pad), 1, 1 - min(0.06, pad * 0.9)))
    path = os.path.join(out_dir, f'hid_cave_day_night{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_diagnostics(all_stats, conditions, smooth, out_dir, share_y):
    """MAD / genomic variance / weight magnitude / inter-gen change, same layout.
    These move mechanically with the number of weights, so they say where the
    capacity went, not whether it helped."""
    print("  Plotting learning diagnostics ...")
    _grid_plot(all_stats, conditions, list(DIAG_METRICS), DIAG_METRICS, smooth,
               out_dir, 'hid_diagnostics',
               _title('Learning diagnostics over generations',
                      'MAD = within-life weight change; genomic variance = '
                      'population diversity in weight space'),
               share_y=share_y, log_cols=LOG_METRICS)


# ── Final-window summary ──────────────────────────────────────────────────────

def final_value(df, col, final_frac):
    """Mean of `col` over the last `final_frac` of a unit's generations."""
    if col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if not len(s):
        return np.nan
    cutoff = s.index.max() - final_frac * (s.index.max() - s.index.min())
    window = s[s.index >= cutoff]
    return window.mean() if len(window) else s.iloc[-1]


def final_window_table(all_units, final_frac):
    """Per-run final-window value of every VS_WIDTH metric, for every condition.
    all_units: {cond_key: {hidden: [(unit, df)]}}. Returns a tidy DataFrame
    (condition, hidden_size, unit, metric, value)."""
    rows = []
    for ck, per_hid in all_units.items():
        for hid, units in per_hid.items():
            for unit, df in units:
                for col, _t in VS_WIDTH:
                    rows.append({'condition': ck, 'hidden_size': hid,
                                 'unit': str(unit), 'metric': col,
                                 'value': final_value(df, col, final_frac)})
    return pd.DataFrame(rows)


def plot_vs_width(final_tbl, conditions, widths, final_frac, out_dir):
    """Final-window value of the key metrics plotted AGAINST width, one line per
    condition — the at-a-glance 'which width won, and did the three mechanisms
    agree?' summary.

    Widths go on an evenly spaced categorical axis: 32/64/128 is geometric, and a
    linear numeric axis would squash the two narrow arms together and imply an
    interpolation between them that was never run."""
    print("  Plotting final-window metrics vs hidden width ...")
    xs = np.arange(len(widths))
    ncols = 3
    nrows = (len(VS_WIDTH) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.0 * nrows),
                            squeeze=False)
    axs = axs.ravel()
    # Nudge each condition sideways so two mechanisms that landed on the same
    # value stay separately visible instead of the last-drawn one hiding the rest
    # (evolution and learning sit on top of each other in most panels).
    dodge = {ck: (i - (len(conditions) - 1) / 2) * 0.055
             for i, ck in enumerate(conditions)}
    for ax, (col, title) in zip(axs, VS_WIDTH):
        any_data = False
        for ck in conditions:
            off = dodge[ck]
            means, stds, px = [], [], []
            for i, h in enumerate(widths):
                vals = final_tbl[(final_tbl['condition'] == ck) &
                                 (final_tbl['hidden_size'] == h) &
                                 (final_tbl['metric'] == col)]['value'].dropna()
                if not len(vals):
                    continue
                px.append(xs[i] + off)
                means.append(vals.mean())
                stds.append(vals.std(ddof=0))
            if not px:
                continue
            any_data = True
            colour = _COND_COLOURS.get(ck, '#111827')
            ax.errorbar(px, means, yerr=stds, marker='o', lw=2, capsize=4,
                        color=colour, ecolor=colour, alpha=0.9, zorder=2,
                        label=CONDITIONS[ck]['label'].split('  ')[0])
            # Individual runs behind the summary line, so a mean driven by one
            # outlier run is visible rather than hidden inside the error bar.
            for i, h in enumerate(widths):
                vals = final_tbl[(final_tbl['condition'] == ck) &
                                 (final_tbl['hidden_size'] == h) &
                                 (final_tbl['metric'] == col)]['value'].dropna()
                if len(vals):
                    ax.scatter(np.full(len(vals), xs[i] + off), vals.values, s=14,
                               color=colour, alpha=0.35, zorder=1)
        if not any_data:
            ax.set_title(f'{title}\n(no data)', color='gray')
            ax.axis('off')
            continue
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('NNBrain hidden width')
        ax.set_ylabel(title)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(h) for h in widths])
        ax.set_xlim(-0.4, len(widths) - 0.6)
        ax.grid(True, alpha=0.3)
    for ax in axs[len(VS_WIDTH):]:
        ax.axis('off')
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   fontsize=11, frameon=False, title='condition')
    plt.suptitle('Final-window outcome vs network capacity — '
                 f'{_ENV_DESC}\n(mean ± std across all runs of the arm, over the '
                 f'last {int(final_frac * 100)}% of generations; '
                 'dots = individual runs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.99))
    path = os.path.join(out_dir, f'hid_vs_width{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Run-consistency: which fitness region did each individual run land in? ────
# The arm means above hide the thing the hard environment actually does: runs do
# not spread evenly around a mean, they split into a group that takes off and a
# group that plateaus low and never recovers. An arm whose mean is 12% higher
# because 10/15 runs took off instead of 4/15 is a CONSISTENCY result, not a
# performance-per-run one, and the two need separating.

def growth_table(all_units, col, final_frac):
    """Per-run endpoint + late trajectory for `col`.

    Returns a tidy DataFrame (condition, hidden_size, unit, final, previous,
    late_growth), where `final` is the mean over the last `final_frac` of the
    run's generations, `previous` the mean over the equally sized window
    immediately before it, and late_growth their difference. That difference is
    what separates a run sitting on a plateau from one still climbing at the
    generation budget's end — two runs can share an endpoint and mean completely
    different things about whether the budget was the binding constraint."""
    rows = []
    for ck, per_hid in all_units.items():
        for hid, units in per_hid.items():
            for unit, df in units:
                if col not in df.columns:
                    continue
                s = pd.to_numeric(df[col], errors='coerce').dropna()
                if not len(s):
                    continue
                lo, hi = s.index.min(), s.index.max()
                span = hi - lo
                cut1 = hi - final_frac * span              # start of final window
                cut0 = hi - 2 * final_frac * span          # start of the one before
                final = s[s.index >= cut1]
                prev = s[(s.index >= cut0) & (s.index < cut1)]
                rows.append({
                    'condition': ck, 'hidden_size': hid, 'unit': str(unit),
                    'final': final.mean() if len(final) else s.iloc[-1],
                    'previous': prev.mean() if len(prev) else np.nan,
                })
    out = pd.DataFrame(rows)
    if len(out):
        out['late_growth'] = out['final'] - out['previous']
    return out


def make_bands(values, gap_frac):
    """Choose the fitness regions the runs are counted into, from the POOLED
    distribution of one environment (all conditions and widths together) so the
    same edges apply to every arm and the counts are directly comparable.

    Two schemes, picked by the data:
      'takeoff'  the sorted values contain one dominant gap — wider than
                 `gap_frac` of the full range and wider than 3x the next-largest
                 gap — i.e. the runs genuinely fall into two separated groups.
                 Split there: 'plateaued low' vs 'took off'. This is the hard
                 environment, where the gap is ~1.4 fitness wide.
      'quartile' no such gap: the runs are one continuous spread, so imposing a
                 two-way split would invent a boundary. Fall back to the pooled
                 quartiles, which still show whether an arm's runs cluster high
                 or scatter across the range. This is the baseline environment.

    Returns (scheme, edges, labels): `edges` are the len(labels)-1 interior
    boundaries, ascending."""
    v = np.sort(np.asarray(values, float))
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return 'quartile', [], ['all runs']
    gaps = np.diff(v)
    rng = v[-1] - v[0]
    i = int(np.argmax(gaps))
    biggest = gaps[i]
    runner_up = np.sort(gaps)[-2] if len(gaps) > 1 else 0.0
    if rng > 0 and biggest >= gap_frac * rng and biggest >= 3 * runner_up:
        thr = 0.5 * (v[i] + v[i + 1])
        return ('takeoff', [thr],
                [f'plateaued low (≤ {thr:.2f})', f'took off (> {thr:.2f})'])
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return ('quartile', list(q),
            [f'Q1  ≤ {q[0]:.2f}', f'Q2  {q[0]:.2f}–{q[1]:.2f}',
             f'Q3  {q[1]:.2f}–{q[2]:.2f}', f'Q4  > {q[2]:.2f}'])


def assign_bands(growth, conditions, widths, edges, labels, plateau_tol):
    """Label every run with its band and whether it was still climbing at the end.

    'still climbing' = the final window beat the one before it by more than
    `plateau_tol` (a FRACTION of the run's own final value, so the test means the
    same thing to a run at 3.5 and one at 7.2). Everything else is flat or
    falling, i.e. the run had converged on whatever level it reached."""
    g = growth.copy()
    g['band_index'] = np.digitize(g['final'].values, edges)
    g['band'] = [labels[min(i, len(labels) - 1)] for i in g['band_index']]
    tol = plateau_tol * g['final'].abs()
    g['trajectory'] = np.where(g['late_growth'] > tol, 'still climbing', 'plateaued')
    g['trajectory'] = np.where(g['late_growth'].isna(), 'unknown', g['trajectory'])
    order = {ck: i for i, ck in enumerate(conditions)}
    return g.sort_values(['condition', 'hidden_size', 'final'],
                         key=lambda s: s.map(order) if s.name == 'condition' else s)


def band_counts(bands, conditions, widths, labels):
    """Runs per (condition, width, band) as a tidy table, zeros included so an arm
    with no runs in a band is stated rather than missing."""
    rows = []
    for ck in conditions:
        for h in widths:
            sub = bands[(bands['condition'] == ck) & (bands['hidden_size'] == h)]
            n = len(sub)
            for lab in labels:
                k = int((sub['band'] == lab).sum())
                rows.append({'condition': ck, 'hidden_size': h, 'band': lab,
                             'n_runs': k, 'arm_total': n,
                             'share': (k / n if n else np.nan)})
            for traj in ('still climbing', 'plateaued'):
                rows.append({'condition': ck, 'hidden_size': h,
                             'band': f'[trajectory] {traj}',
                             'n_runs': int((sub['trajectory'] == traj).sum()),
                             'arm_total': n,
                             'share': ((sub['trajectory'] == traj).sum() / n
                                       if n else np.nan)})
    return pd.DataFrame(rows)


def plot_consistency(bands, counts, conditions, widths, labels, scheme, edges,
                     band_metric, band_title, final_frac, out_dir):
    """The consistency figure: one column per condition, one bar group per width.

    Row 1 stacks each arm's 15 runs by the region its final fitness landed in —
    the answer to 'how many runs of this width failed to get going?'.
    Row 2 shows the runs themselves against the same band edges, so the stack is
    auditable and a near-boundary run is visible as such rather than silently
    counted one side of a line."""
    print("  Plotting run-consistency bands ...")
    colours = (_BAND_COLOURS_2 if len(labels) == 2 else
               _BAND_COLOURS_4 if len(labels) == 4 else
               [plt.get_cmap('viridis')(i / max(len(labels) - 1, 1))
                for i in range(len(labels))])
    xs = np.arange(len(widths))
    ncols = len(conditions)
    fig, axs = plt.subplots(2, ncols, figsize=(6.4 * ncols, 9.0), squeeze=False)

    for c, ck in enumerate(conditions):
        ax = axs[0][c]
        bottom = np.zeros(len(widths))
        for li, lab in enumerate(labels):
            vals = np.array([
                counts[(counts['condition'] == ck) & (counts['hidden_size'] == h) &
                       (counts['band'] == lab)]['n_runs'].sum()
                for h in widths], float)
            ax.bar(xs, vals, 0.62, bottom=bottom, color=colours[li],
                   edgecolor='white', linewidth=1.2, label=lab)
            for x, v, b in zip(xs, vals, bottom):
                if v > 0:
                    ax.text(x, b + v / 2, f'{int(v)}', ha='center', va='center',
                            color='white', fontsize=11, fontweight='bold')
            bottom += vals
        ax.set_xticks(xs)
        ax.set_xticklabels([f'h{h}' for h in widths])
        ax.set_ylim(0, max(bottom.max(), 1) * 1.08)
        ax.set_title(CONDITIONS[ck]['label'], fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, axis='y', alpha=0.3)
        if c == 0:
            ax.set_ylabel(f'Runs in each region  (n per arm = {int(bottom.max())})',
                          fontsize=11, fontweight='bold')

        # Row 2: the runs behind the stack. Jittered so 15 runs at nearly the same
        # value stay countable, and marked by trajectory so a run that plateaued
        # low is distinguishable from one that was still climbing when the
        # generation budget ran out. Dots are coloured by BAND, not by width —
        # width is already the x-axis, and reusing the width hues here would put
        # a red h32 dot next to a red "plateaued low" bar meaning something else.
        ax = axs[1][c]
        for e in edges:
            ax.axhline(e, color='#6B7280', ls='--', lw=1)
        rng = np.random.default_rng(0)
        band_colour = dict(zip(labels, colours))
        for i, h in enumerate(widths):
            sub = bands[(bands['condition'] == ck) & (bands['hidden_size'] == h)]
            if not len(sub):
                continue
            climbing = (sub['trajectory'] == 'still climbing').values
            cols = np.array([band_colour.get(b, '#6B7280') for b in sub['band']])
            jit = rng.uniform(-0.16, 0.16, len(sub))
            ax.scatter(xs[i] + jit[~climbing], sub['final'].values[~climbing],
                       s=42, color=cols[~climbing], alpha=0.85, zorder=3)
            ax.scatter(xs[i] + jit[climbing], sub['final'].values[climbing],
                       s=52, facecolors='none', edgecolors=cols[climbing],
                       linewidths=1.8, zorder=3)
            # Arm median, on a white underlay: without it the bar disappears into
            # a tight cluster of same-coloured dots, which is exactly the case
            # (an arm whose runs all landed together) where it matters most.
            med = sub['final'].median()
            ax.plot([xs[i] - 0.3, xs[i] + 0.3], [med] * 2,
                    color='white', lw=5, solid_capstyle='butt', zorder=4)
            ax.plot([xs[i] - 0.3, xs[i] + 0.3], [med] * 2,
                    color='#111827', lw=2.2, solid_capstyle='butt', zorder=5)
        if c == 0:
            # Grey stand-ins: the real markers take their colour from the band, so
            # a legend built from them would imply the marker shape and the colour
            # encode the same thing.
            ax.scatter([], [], s=42, color='#6B7280', label='plateaued')
            ax.scatter([], [], s=52, facecolors='none', edgecolors='#6B7280',
                       linewidths=1.8, label='still climbing')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'h{h}' for h in widths])
        ax.set_xlim(-0.5, len(widths) - 0.5)
        ax.set_xlabel('NNBrain hidden width')
        ax.grid(True, axis='y', alpha=0.3)
        if c == 0:
            ax.set_ylabel(f'{band_title}\n(final-window, per run)',
                          fontsize=11, fontweight='bold')
    # Row 2 shares one y-axis across conditions: the band edges are global, so
    # per-column autoscaling would draw the same threshold at three heights.
    _share_row_y(axs[1], False)

    h1, l1 = axs[0][0].get_legend_handles_labels()
    h2, l2 = axs[1][0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=len(l1) + len(l2),
               fontsize=10, frameon=False,
               title=f'fitness region ({scheme} split of the pooled runs)  |  '
                     'marker = end-of-run trajectory')
    plt.suptitle(
        f'Run consistency across network capacity — {_ENV_DESC}\n'
        f'(where each individual run\'s {band_title.lower()} ended up, over the '
        f'last {int(final_frac * 100)}% of generations; bands from the pooled '
        f'{len(bands)} runs of this environment; black bar = arm median)',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.09, 1, 0.98))
    path = os.path.join(out_dir, f'hid_consistency{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_band_table(counts, conditions, widths, labels, scheme, band_title):
    """Console version of the consistency counts — the headline answer to 'how
    many runs of each width got going at all?'."""
    print(f"\n  Run consistency: {band_title} regions "
          f"({scheme} split of this environment's pooled runs)")
    rows = labels + ['[trajectory] still climbing', '[trajectory] plateaued']
    for ck in conditions:
        print(f"\n    {CONDITIONS[ck]['label']}")
        header = '      {:<34}'.format('region') + ''.join(
            f'{"h" + str(h):>14}' for h in widths)
        print(header)
        print('      ' + '-' * (len(header) - 6))
        for lab in rows:
            cells = []
            for h in widths:
                sel = counts[(counts['condition'] == ck) &
                             (counts['hidden_size'] == h) &
                             (counts['band'] == lab)]
                if not len(sel):
                    cells.append('n/a'.rjust(14))
                    continue
                n = int(sel['n_runs'].iloc[0])
                tot = int(sel['arm_total'].iloc[0])
                cells.append(f'{n}/{tot}'.rjust(14))
            print('      {:<34}'.format(lab[:34]) + ''.join(cells))


def _cliffs_delta(a, b):
    """Cliff's delta for b vs a: P(b > a) - P(b < a), in [-1, 1]. Non-parametric
    effect size — reported alongside p because 15-vs-15 runs can be significantly
    different and practically identical, or vice versa."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    diff = b[:, None] - a[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size


def width_comparison(final_tbl, conditions, widths, final_frac):
    """Each non-default width against the 64 control, WITHIN each condition, over
    the same run-level units the curves are averaged over (n = seeds x replicates
    per arm). Returns a tidy DataFrame."""
    base = _DEFAULT_HIDDEN if _DEFAULT_HIDDEN in widths else widths[0]
    rows = []
    for ck in conditions:
        for h in [w for w in widths if w != base]:
            for col, title in VS_WIDTH:
                sel = ((final_tbl['condition'] == ck) & (final_tbl['metric'] == col))
                av = final_tbl[sel & (final_tbl['hidden_size'] == base)]['value'].dropna()
                cv = final_tbl[sel & (final_tbl['hidden_size'] == h)]['value'].dropna()
                if not len(av) or not len(cv):
                    continue
                if _scipy_stats is not None and len(av) > 1 and len(cv) > 1:
                    try:
                        p = _scipy_stats.mannwhitneyu(cv, av, alternative='two-sided').pvalue
                    except ValueError:      # all values identical
                        p = np.nan
                else:
                    p = np.nan
                rows.append({
                    'condition': ck, 'metric': col, 'label': title,
                    'baseline_hidden': base, 'hidden_size': h,
                    'mean_baseline': av.mean(), 'mean_arm': cv.mean(),
                    'std_baseline': av.std(ddof=0), 'std_arm': cv.std(ddof=0),
                    'diff': cv.mean() - av.mean(),
                    'pct_change': (100.0 * (cv.mean() - av.mean()) / av.mean()
                                   if av.mean() else np.nan),
                    'mannwhitney_p': p,
                    'cliffs_delta': _cliffs_delta(av.values, cv.values),
                    'n_baseline': len(av), 'n_arm': len(cv),
                    'final_frac': final_frac,
                })
    return pd.DataFrame(rows)


def write_summary(all_stats, final_tbl, cmp_tbl, bands, counts, out_dir):
    """Tidy (condition, hidden_size, generation, metric, mean, std) across each
    arm's runs, plus the per-run final-window table behind hid_vs_width and the
    width comparison."""
    frames = []
    for ck, cond_stats in all_stats.items():
        for hid, (mean_df, std_df) in cond_stats.items():
            m_long = mean_df.reset_index().melt(id_vars='generation',
                                                var_name='metric', value_name='mean')
            s_long = std_df.reset_index().melt(id_vars='generation',
                                               var_name='metric', value_name='std')
            merged = m_long.merge(s_long, on=['generation', 'metric'])
            merged.insert(0, 'hidden_size', hid)
            merged.insert(0, 'condition', ck)
            frames.append(merged)
    if frames:
        path = os.path.join(out_dir, f'hid_curves_summary{_SUFFIX}.csv')
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"  Saved: {path}")

    path = os.path.join(out_dir, f'hid_final_window{_SUFFIX}.csv')
    final_tbl.to_csv(path, index=False)
    print(f"  Saved: {path}")

    if len(cmp_tbl):
        path = os.path.join(out_dir, f'hid_width_comparison{_SUFFIX}.csv')
        cmp_tbl.to_csv(path, index=False)
        print(f"  Saved: {path}")

    if len(bands):
        path = os.path.join(out_dir, f'hid_run_bands{_SUFFIX}.csv')
        bands.to_csv(path, index=False)
        print(f"  Saved: {path}")
        path = os.path.join(out_dir, f'hid_band_counts{_SUFFIX}.csv')
        counts.to_csv(path, index=False)
        print(f"  Saved: {path}")


def print_final_table(final_tbl, conditions, widths):
    """Console summary: final-window mean +/- std per (condition, width) for each
    key metric, so the headline numbers are readable without opening the PNGs."""
    for ck in conditions:
        print(f"\n  Final-window summary — {CONDITIONS[ck]['label']} "
              f"(mean ± std across all runs of the arm):")
        header = '    {:<30}'.format('metric') + ''.join(
            f'{"h" + str(h):>22}' for h in widths)
        print(header)
        print('    ' + '-' * (len(header) - 4))
        for col, title in VS_WIDTH:
            cells = []
            for h in widths:
                vals = final_tbl[(final_tbl['condition'] == ck) &
                                 (final_tbl['hidden_size'] == h) &
                                 (final_tbl['metric'] == col)]['value'].dropna()
                cells.append('n/a'.rjust(22) if not len(vals) else
                             f'{vals.mean():.4g} ± {vals.std(ddof=0):.2g}'.rjust(22))
            print('    {:<30}'.format(title[:30]) + ''.join(cells))


def print_comparison(cmp_tbl):
    """Console version of the width-vs-control comparison (run level)."""
    if not len(cmp_tbl):
        return
    for (ck, h), sub in cmp_tbl.groupby(['condition', 'hidden_size']):
        base = int(sub['baseline_hidden'].iloc[0])
        print(f"\n  {CONDITIONS[ck]['label']}: hidden {int(h)} vs hidden {base} "
              f"(n={int(sub['n_arm'].iloc[0])} vs {int(sub['n_baseline'].iloc[0])} runs):")
        print('    {:<30}{:>12}{:>10}{:>10}{:>9}'.format(
            'metric', 'diff', '%', 'p(MWU)', 'delta'))
        print('    ' + '-' * 71)
        for _, r in sub.iterrows():
            p = 'n/a' if not np.isfinite(r['mannwhitney_p']) else f"{r['mannwhitney_p']:.3f}"
            star = ' *' if np.isfinite(r['mannwhitney_p']) and r['mannwhitney_p'] < 0.05 else ''
            print('    {:<30}{:>12.4g}{:>10.1f}{:>10}{:>9.2f}{}'.format(
                r['label'][:30], r['diff'], r['pct_change'], p,
                r['cliffs_delta'], star))
        print(f"    * p < 0.05 (Mann-Whitney, uncorrected across the {len(sub)} "
              "metrics above — treat as descriptive).")


def run_env(env_key, args):
    """Discover + aggregate + plot one environment's capacity sweep, across every
    requested condition."""
    env = ENVIRONMENTS[env_key]
    roaming, drain = ((args.predators, args.drain) if env_key == 'hard'
                      else (env['roaming'], env['drain']))

    out_dir = args.out or f'output/hidden_size_sweep_{env_key}'
    os.makedirs(out_dir, exist_ok=True)

    global _ENV_DESC, _SUFFIX, _BAND_DESC, _HID_COLOURS
    _ENV_DESC = (env['desc'] if env_key == 'baseline' else
                 f'{int(roaming)} roaming predators, drain {int(drain)}')
    seed_filter = set(args.seeds) if args.seeds else None
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''
    _BAND_DESC = 'across all runs of the arm, seeds × replicates pooled'

    print(f"\n=== {env_key.upper()} environment ({_ENV_DESC}) ===")

    found = {}
    for ck in args.conditions:
        root = args.roots.get(ck) if args.roots else None
        runs = discover(ck, env_key, roaming, drain, seed_filter, root)
        if not runs:
            print(f"  WARNING: no {ck} runs found under "
                  f"{root or CONDITIONS[ck]['root']} for env={env_key} — "
                  "that column will be missing.")
            continue
        widths = sorted({r['hidden'] for r in runs})
        print(f"  {CONDITIONS[ck]['label']}: {len(runs)} runs, widths {widths}")
        found[ck] = runs
    if not found:
        print("  WARNING: nothing to analyse for this environment. Skipping.")
        return

    conditions = [ck for ck in COND_ORDER if ck in found]
    widths = sorted({r['hidden'] for runs in found.values() for r in runs})

    if len(widths) <= len(_WIDTH_COLOURS_3):
        _HID_COLOURS = {h: _WIDTH_COLOURS_3[i] for i, h in enumerate(widths)}
    else:   # a 4+-value sweep is an ordered axis: sequential ramp instead
        cmap = plt.get_cmap('viridis')
        n = max(len(widths) - 1, 1)
        _HID_COLOURS = {h: cmap(0.08 + 0.84 * i / n) for i, h in enumerate(widths)}

    cache_dir = None if args.no_cache else args.cache_dir
    build_cache([r for runs in found.values() for r in runs], cache_dir, args.workers)

    all_stats, all_units = {}, {}
    for ck in conditions:
        print(f"  Loading {CONDITIONS[ck]['label']} ...")
        stats, per_unit = aggregate_condition(found[ck], cache_dir, args.max_gen)
        if stats:
            all_stats[ck] = stats
            all_units[ck] = per_unit
    if not all_stats:
        print("  WARNING: nothing loaded for this env. Skipping.")
        return
    conditions = [ck for ck in conditions if ck in all_stats]

    print("Plotting ...")
    share_y = not args.free_y
    final_tbl = final_window_table(all_units, args.final_frac)
    cmp_tbl = (width_comparison(final_tbl, conditions, widths, args.final_frac)
               if len(widths) > 1 else pd.DataFrame())

    plot_fitness_population(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_lifetime(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_behaviour(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_food(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_cave_day_night(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_diagnostics(all_stats, conditions, args.smooth, out_dir, share_y)
    plot_vs_width(final_tbl, conditions, widths, args.final_frac, out_dir)

    # Run-consistency: the same 15 runs per arm, counted by the fitness region
    # they ended in rather than averaged into one curve.
    band_title = (GEN_METRICS.get(args.band_metric) or
                  LIFE_METRICS.get(args.band_metric) or
                  BEHAV_METRICS.get(args.band_metric) or args.band_metric)
    growth = growth_table(all_units, args.band_metric, args.final_frac)
    bands, counts = pd.DataFrame(), pd.DataFrame()
    if len(growth):
        scheme, edges, labels = make_bands(growth['final'], args.gap_frac)
        bands = assign_bands(growth, conditions, widths, edges, labels,
                             args.plateau_tol)
        bands.insert(0, 'band_scheme', scheme)
        counts = band_counts(bands, conditions, widths, labels)
        plot_consistency(bands, counts, conditions, widths, labels, scheme, edges,
                         args.band_metric, band_title, args.final_frac, out_dir)
    else:
        print(f"  (no {args.band_metric} data — skipping the consistency view)")

    write_summary(all_stats, final_tbl, cmp_tbl, bands, counts, out_dir)
    print_final_table(final_tbl, conditions, widths)
    if len(counts):
        print_band_table(counts, conditions, widths, labels, scheme, band_title)
    print_comparison(cmp_tbl)
    print(f"\nDone ({env_key}). Outputs in {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser(
        description='Effect of NNBrain hidden width (network capacity) on '
                    'fitness, population, lifetime and behaviour, compared '
                    'across the evolution / learning / pure-RL conditions.')
    ap.add_argument('--env', choices=['baseline', 'hard', 'both'], default='both',
                    help='Environment(s) to analyse (default: both).')
    ap.add_argument('--conditions', nargs='+', choices=COND_ORDER,
                    default=COND_ORDER,
                    help='Conditions to include as columns (default: all three).')
    ap.add_argument('--root', nargs='+', default=None, metavar='COND=DIR',
                    help='Override a condition\'s log root, e.g. '
                         'pure_rl=logs/learning/pure_rl/auto-run. Repeatable.')
    ap.add_argument('--predators', type=float, default=60,
                    help='roaming_predator_count for the HARD env (default 60).')
    ap.add_argument('--drain', type=float, default=5,
                    help='predator_drain for the HARD env (default 5).')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this.')
    ap.add_argument('--smooth', type=int, default=10,
                    help='Rolling-mean window for the curves/bands (1 = none). '
                         'Default 10: the death-cause and survival curves carry a '
                         'strong ~5-generation oscillation (maps_per_gen = 5) that '
                         'buries the width lines at smaller windows.')
    ap.add_argument('--final-frac', type=float, default=0.2,
                    help='Fraction of the final generations averaged for the '
                         'vs-width summary plot (default 0.2 = last 20%%).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered.')
    ap.add_argument('--band-metric', default='avg_fitness',
                    help='Metric whose final-window value the run-consistency '
                         'regions are cut on (default avg_fitness).')
    ap.add_argument('--gap-frac', type=float, default=0.15,
                    help='A gap in the pooled run distribution wider than this '
                         'fraction of its range (and 3x the next-largest gap) is '
                         'treated as a genuine two-group split; otherwise the '
                         'consistency bands fall back to pooled quartiles.')
    ap.add_argument('--plateau-tol', type=float, default=0.02,
                    help='A run counts as "still climbing" when its final window '
                         'beats the preceding one by more than this fraction of '
                         'its own final value (default 0.02 = 2%%).')
    ap.add_argument('--free-y', action='store_true',
                    help='Do NOT share a y-axis across the conditions in a row. '
                         'Maximises within-panel detail at the cost of making the '
                         'columns visually incomparable.')
    ap.add_argument('--cache-dir', default='output/.cache_hidden_curves',
                    help='Where per-run organisms.csv summaries are cached.')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-read every organisms.csv instead of using the cache.')
    ap.add_argument('--workers', type=int, default=6,
                    help='Parallel workers for the first (cache-building) pass '
                         '(default 6; 1 = sequential).')
    ap.add_argument('-o', '--out', default=None,
                    help='Output directory (default: output/hidden_size_sweep_<env>). '
                         'With --env both this is ignored (each env needs its own).')
    args = ap.parse_args()

    args.roots = {}
    for spec in (args.root or []):
        if '=' not in spec:
            ap.error(f'--root expects COND=DIR, got {spec!r}')
        ck, path = spec.split('=', 1)
        if ck not in CONDITIONS:
            ap.error(f'--root: unknown condition {ck!r}')
        args.roots[ck] = path

    envs = ['baseline', 'hard'] if args.env == 'both' else [args.env]
    if args.env == 'both' and args.out:
        print("note: --out ignored with --env both (each env writes its own dir).")
        args.out = None
    for env_key in envs:
        run_env(env_key, args)
    print()


if __name__ == '__main__':
    main()
