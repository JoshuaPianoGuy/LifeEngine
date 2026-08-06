"""
analyse_buffer_sweep.py
======================================================================
Effect of the collapse-respawn BUFFER SIZE on fitness, population, the learning
diagnostics (MAD + genomic variance) AND behaviour, from the pure-RL buffer
sweep produced by run_pure_rl_collapse_buffer_array.slurm.

That job sweeps PureRLManager's rolling dead-weight buffer — the pool of
active_weight snapshots taken from recently dead organisms
(PureRLManager._bufferDeadWeights) that a collapse is refilled from
(_respawnFromBuffer). With no GA in the pure_rl mode, that buffer is the ONLY
mechanism carrying learned strategies across a collapse, so its size is pure
RL's entire inheritance channel:

    collapse_buffer : 25 (default / matched control), 100
    environments    : baseline (predator-free, lr 0.01)
                      hard     (60 roaming predators, drain 5, lr 0.02)
    seeds           : 1, 42, 999   (SAME 3 maps as the LR / trace sweeps)
    replicates      : 1..5         (identical terrain, different RL chance)
    2 buffers x 3 seeds x 5 replicates = 30 runs per environment.

25 is the value every earlier pure-RL run used (the old hard-coded constant), so
that arm doubles as the baseline the sweep is measured against.

The hypothesis the plots are built to answer: if pure RL's collapse-and-restart
churn is a DIVERSITY problem, buffer 100 should reduce collapse depth/frequency
and raise the level it recovers to; if it is a RECENCY problem, 100 should be
WORSE, because it reseeds from weights that have already been superseded. The
two diagnostics separate those readings —

    avg_learned_weight_diff (MAD)  mean |active - genome| over the ranked
                                   population: how far within-life RL has moved
                                   each brain from the weights it was born with.
                                   In pure RL the genome is resynced at each
                                   window, so MAD reads as "learning done since
                                   the last respawn" — a recency problem shows up
                                   as a LOWER MAD (fresh spawns re-learning from
                                   staler starting points).
    genome_variance                spread of the population in weight space, i.e.
                                   exactly the diversity the bigger buffer is
                                   supposed to buy. A diversity win is a HIGHER
                                   genome_variance that survives the collapses.

plus the mechanism itself, recovered from events.csv (the per-window
"topping up N from buffer (survivors=S, buffer_size=B)" lines): how often the
population collapses, how deep, and how full the buffer actually gets.

This is the buffer analogue of analyse_trace_decay_sweep.py: same organisms.csv /
generations.csv schema and the same behaviour + outcome plots, but the SERIES is
the collapse buffer size within the pure-RL condition, and it adds the learning
diagnostics and the collapse/top-up panels.

AGGREGATION
-----------
Every run of an arm is one unit: the 15 runs of a (buffer, environment) cell —
3 seeds x 5 replicates — are averaged into a single curve per arm, and the band
is the standard deviation across those 15 runs. Seeds are NOT a separate level
of aggregation, so the band carries both map-to-map variation and the
run-to-run algorithmic chance (only the map TERRAIN is seeded; founder weights
are an unseeded Xavier draw and RL action sampling is unseeded too). Pure RL is
the noisiest condition in this codebase, so expect a wide band — that spread IS
the measurement, and the arm comparison below tests the two arms at the same
run level (n = 15 vs 15).

Runs in BOTH environments by default (--env both), each into its own output dir
so baseline and hard never mix.

Outputs (into --out / output/buffer_sweep_<env>):
    buf_fitness_population.png  -- avg/top-20%/best fitness, peak population,
                                   total agents, avg lifetime over generations;
                                   one line per buffer size.
    buf_learning_diagnostics.png-- MAD, genomic variance, network weight
                                   magnitude and inter-generation weight change.
                                   THE requested diagnostics view; MAD and
                                   genome_variance are drawn on a log y-axis
                                   because they span decades over a run.
    buf_collapse.png            -- the buffer mechanism from events.csv:
                                   survivors at window close, organisms respawned
                                   from the buffer, rolling collapse rate
                                   (windows with 0 survivors) and buffer fill.
    buf_behaviour_curves.png    -- per-organism scalar behaviours over generations
                                   (cells explored, cave entries, predator
                                   encounters, food eaten, lifetime, ...).
    buf_cave_day_night.png      -- cave entries split day vs night + the night
                                   share over time.
    buf_behaviour_food.png      -- food-tier diet composition (one panel/tier).
    buf_vs_buffer.png           -- final-window (last --final-frac of generations)
                                   value of the key fitness / diagnostic /
                                   behaviour metrics plotted AGAINST buffer size,
                                   mean +/- std across runs. The at-a-glance
                                   "which buffer won?" summary.
    buf_behaviour_summary.csv   -- tidy (collapse_buffer, generation, metric,
                                   mean, std).
    buf_final_window.csv        -- per-run final-window values behind
                                   buf_vs_buffer.
    buf_arm_comparison.csv      -- run-level arm-vs-arm test per metric
                                   (difference, % change, Mann-Whitney p,
                                   Cliff's delta).

Reading 30 organisms.csv files per environment (~96 MB each) is the slow part, so
each run's per-generation curve is cached under --cache-dir (keyed on the CSV's
size+mtime); re-plotting with a different --smooth or --final-frac is then
near-instant.

Usage
-----
    python analyse_buffer_sweep.py                      # both envs
    python analyse_buffer_sweep.py --env baseline
    python analyse_buffer_sweep.py --env hard --seeds 1 42
    python analyse_buffer_sweep.py --smooth 10
    python analyse_buffer_sweep.py --no-cache           # force re-read
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

# ── Environment presets (pure-RL sweep; lr differs per env) ───────────────────
ENVIRONMENTS = {
    'baseline': {'roaming': 0.0,  'drain': None, 'lr': 0.01,
                 'desc': 'predator-free baseline'},
    'hard':     {'roaming': 60.0, 'drain': 5.0,  'lr': 0.02,
                 'desc': '60 roaming predators, drain 5'},
}

# ── Scalar per-organism behaviours (curve panels) ─────────────────────────────
METRICS = {
    'cells_visited':    'Cells explored',
    'cave_entries':     'Cave entries',
    'predator_touches': 'Predator encounters',
    'drained_ticks':    'Ticks drained by predators',
    'food_total':       'Food eaten (total)',
    'lifetime':         'Lifetime (ticks survived)',
}

# ── Compositional behaviours ──────────────────────────────────────────────────
FOOD_TIERS = {
    'food_default':  ('default',  '#9CA3AF'),
    'food_low':      ('low',      '#60A5FA'),
    'food_medium':   ('medium',   '#F59E0B'),
    'food_prestige': ('prestige', '#DB2777'),
}
FOOD_COLS = list(FOOD_TIERS)

SCALAR_EXTRA = ['cave_entries_day', 'cave_entries_night', 'energy_at_death']
SCALAR_COLS = list(METRICS) + SCALAR_EXTRA

USE_COLS = (['generation', 'cells_visited', 'predator_touches', 'drained_ticks',
             'cave_entries', 'cave_entries_day', 'cave_entries_night', 'lifetime',
             'energy_at_death'] + FOOD_COLS)

# ── Generation-level outcome metrics (from generations.csv) ───────────────────
GEN_METRICS = {
    'avg_fitness':          'Mean average fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'best_fitness':         'Best fitness',
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
    'avg_lifetime':         'Mean lifetime (ticks)',
}

# ── Learning diagnostics (also generations.csv) ───────────────────────────────
# MAD and genome_variance are the two the buffer is expected to move; the weight
# magnitude and inter-generation change are supporting context (a buffer that
# reseeds from stale vectors shows up as a smaller inter-gen change).
DIAG_METRICS = {
    'avg_learned_weight_diff': 'Learned weight diff (MAD)',
    'genome_variance':         'Genomic variance',
    'avg_network_weight_mag':  'Network weight magnitude (RMS)',
    'inter_gen_weight_change': 'Inter-generation weight change',
}

# Diagnostics that span decades within a run — drawn on a log y-axis, else the
# early transient flattens the whole rest of the curve into the axis.
LOG_METRICS = {'avg_learned_weight_diff', 'genome_variance'}

GEN_COLS = list(GEN_METRICS) + list(DIAG_METRICS)

# ── Buffer-mechanism metrics (parsed from events.csv) ─────────────────────────
# Per window: how many organisms survived to the window close, how many had to be
# respawned from the buffer, whether the population fully collapsed, how full the
# buffer was at that moment, and how many were randomly culled back to the cap.
EVENT_METRICS = {
    'survivors':     'Survivors at window close',
    'respawned':     'Respawned from buffer',
    'collapse':      'Collapse rate (windows with 0 survivors)',
    'buffer_fill':   'Buffer occupancy at top-up',
    'cap_removed':   'Randomly culled over cap',
}

# Metrics for the final-window "vs buffer" summary plot. Deliberately mixes the
# three families the sweep is judged on: outcome, diagnostic, behaviour.
VS_BUFFER = [
    ('avg_fitness',             'Mean average fitness'),
    ('top20percent_fitness',    'Top-20% fitness'),
    ('peak_population',         'Peak population'),
    ('avg_lifetime',            'Mean lifetime (ticks)'),
    ('avg_learned_weight_diff', 'Learned weight diff (MAD)'),
    ('genome_variance',         'Genomic variance'),
    ('collapse',                'Collapse rate'),
    ('respawned',               'Respawned from buffer / window'),
    ('cells_visited',           'Cells explored / organism'),
    ('cave_entries',            'Cave entries / organism'),
    ('night_share',             'Night share of cave entries'),
    ('food_total',              'Food eaten / organism'),
]

# Sweep-run gate + buffer source: only folders carrying a _buf<N>_ tag are this
# sweep's runs. The tag is cross-checked against params.json below — a clone that
# predates --collapse-buffer silently runs every arm at 25, which would look like
# a null result instead of a mistake.
_BUF_RE = re.compile(r'_buf(\d+)_')
_REP_RE = re.compile(r'_r(\d+)$')

# events.csv lines the buffer mechanism is recovered from (PureRLManager.js).
_TOPUP_RE = re.compile(
    r'Window (\d+) \| topping up (\d+) from buffer '
    r'\(survivors=(\d+), buffer_size=(\d+)\)')
_CAP_RE = re.compile(
    r'Window (\d+) \| random cap: removed (\d+) organisms randomly')

# Categorical colours for up to 4 arms (control first); more arms fall back to a
# sequential ramp, since a 4+-value sweep is an ordered axis again.
_ARM_COLOURS = ['#2563EB', '#F97316', '#059669', '#DB2777']

# Set per-env in run_env(); used by the plot titles / filenames.
_ENV_DESC = ''
_SUFFIX = ''
_BAND_DESC = ''
_BUF_COLOURS = {}   # buffer size -> colour, filled per run.
_DEFAULT_BUFFER = 25


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def _buf_label(n):
    """Consistent display label for a buffer size, flagging the default arm."""
    return f'buffer {n}' + ('  (default)' if n == _DEFAULT_BUFFER else '')


def discover(root, want):
    """Find buffer-sweep run folders under `root` (params.json + organisms.csv)
    matching every filter in `want`, returning
    {buffer, seed, rep, name, dir, org_csv, events_csv} dicts.

    `want` keys: condition (req), mode, epsilon_enabled, learning_rate, roaming,
    drain (skipped if None) — same semantics as analyse_trace_decay_sweep.py.

    A run whose _buf<N>_ tag disagrees with its recorded collapse_buffer_size is
    DROPPED with a warning: that is the signature of a clone that ignored
    --collapse-buffer, and folding it into the arm named by its folder would
    quietly average the control into the treatment."""
    runs, mismatched = [], []
    for params_path in glob.glob(os.path.join(root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        org_csv = os.path.join(run_dir, 'organisms.csv')
        if not os.path.exists(org_csv):
            continue
        name = os.path.basename(run_dir)
        m = _BUF_RE.search(name)
        if not m:      # no _buf tag => not a sweep run
            continue
        try:
            with open(params_path) as f:
                p = json.load(f)
        except (ValueError, OSError):
            continue
        if str(p.get('condition')) != want['condition']:
            continue
        if 'mode' in want and str(p.get('mode')) != want['mode']:
            continue
        if 'epsilon_enabled' in want and bool(p.get('epsilon_enabled', True)) != want['epsilon_enabled']:
            continue
        if 'learning_rate' in want and abs(_num(p, 'learning_rate') - want['learning_rate']) > 1e-9:
            continue
        # Width guard: the buffer sweep ran entirely at the default hidden_size 64,
        # and the later network-size sweep reuses this condition + lr + predator
        # combination at 32 and 128. Its runs are named _cb<N>_ rather than
        # _buf<N>_ so they don't reach here, but pinning the width means they
        # cannot be folded into a buffer arm even if a run is renamed.
        if 'hidden_size' in want and _num(p, 'hidden_size') != want['hidden_size']:
            continue
        if abs(_num(p, 'roaming_predator_count') - want['roaming']) > 1e-9:
            continue
        if want.get('drain') is not None and abs(_num(p, 'predator_drain') - want['drain']) > 1e-9:
            continue
        buf = int(m.group(1))
        recorded = _num(p, 'collapse_buffer_size')
        if np.isfinite(recorded) and int(recorded) != buf:
            mismatched.append((name, buf, int(recorded)))
            continue
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        rm = _REP_RE.search(name)
        rep = int(rm.group(1)) if rm else 1
        runs.append({'buffer': buf, 'seed': seed, 'rep': rep, 'name': name,
                     'dir': run_dir, 'org_csv': org_csv,
                     'events_csv': os.path.join(run_dir, 'events.csv'),
                     'pop_size': _num(p, 'population_size')})
    for name, tagged, recorded in mismatched:
        print(f"  WARNING: {name} is tagged buf{tagged} but ran with "
              f"collapse_buffer_size={recorded} — dropped (stale clone?).")
    return sorted(runs, key=lambda r: (r['buffer'], r['seed'], r['rep']))


def _cache_path(cache_dir, run_name, org_csv):
    """Cache key = run name + the organisms.csv size/mtime, so an updated log
    invalidates its own entry without touching the rest."""
    st = os.stat(org_csv)
    return os.path.join(cache_dir, f'{run_name}__{st.st_size}_{int(st.st_mtime)}.csv')


def load_run(org_csv, run_name, cache_dir):
    """Read one run's organisms.csv; return a per-generation DataFrame (indexed by
    generation) with the mean of each scalar behaviour across that generation's
    organisms, the food_<tier>_frac diet shares, and night_share (the day-vs-night
    cave preference). Missing columns degrade to NaN rather than crashing on
    older logs. Cached per run when cache_dir is not None."""
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

    scal = df.groupby('generation')[SCALAR_COLS].mean()

    fsum = df.groupby('generation')[FOOD_COLS].sum()
    ffrac = fsum.div(fsum.sum(axis=1).replace(0, np.nan), axis=0)
    ffrac.columns = [f'{c}_frac' for c in FOOD_COLS]

    # Night share from the generation's TOTAL entries, not the mean of per-organism
    # ratios: organisms that never entered a cave have no ratio, and one that
    # entered once shouldn't outvote one that entered fifty times.
    csum = df.groupby('generation')[['cave_entries_day', 'cave_entries_night']].sum()
    total = (csum['cave_entries_day'] + csum['cave_entries_night']).replace(0, np.nan)
    night_share = (csum['cave_entries_night'] / total).rename('night_share')

    out = pd.concat([scal, ffrac, night_share], axis=1).sort_index()
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_csv(cache_file, index_label='generation')
    return out


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


def load_events(events_csv, max_gen, pop_size):
    """Recover the buffer mechanism per window from events.csv, indexed by
    generation: survivors, respawned (the top-up deficit), collapse (1 when the
    population hit 0 survivors), buffer_fill and cap_removed.

    PureRLManager only logs a top-up when the deficit is non-zero, so a window
    with NO top-up line is a full house: respawned 0, survivors = population_size.
    That absence is information, and treating it as missing would bias the
    collapse rate upward on exactly the arms that collapse least. Windows are
    therefore filled in from the 'Window N closed' roster, which every window
    emits. Returns None if the file is unusable."""
    if not os.path.exists(events_csv) or not np.isfinite(pop_size):
        return None
    pop = int(pop_size)
    topups, caps, seen = {}, {}, set()
    closed_re = re.compile(r'Window (\d+) closed')
    try:
        with open(events_csv, errors='replace') as f:
            for line in f:
                if 'Window' not in line:
                    continue
                m = _TOPUP_RE.search(line)
                if m:
                    gen = int(m.group(1))
                    topups[gen] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                    seen.add(gen)
                    continue
                m = _CAP_RE.search(line)
                if m:
                    caps[int(m.group(1))] = int(m.group(2))
                    continue
                m = closed_re.search(line)
                if m:
                    seen.add(int(m.group(1)))
    except OSError:
        return None
    if not seen:
        return None

    gens = sorted(g for g in seen if g <= max_gen)
    rows = []
    for g in gens:
        deficit, survivors, fill = topups.get(g, (0, pop, np.nan))
        rows.append({'generation': g,
                     'survivors': float(survivors),
                     'respawned': float(deficit),
                     'collapse': 1.0 if survivors == 0 else 0.0,
                     'buffer_fill': float(fill),
                     'cap_removed': float(caps.get(g, 0))})
    return pd.DataFrame(rows).set_index('generation')


def stack_mean_std(unit_curves):
    """Aggregate an arm's per-unit curves -> (mean, std) DataFrames indexed by
    generation. unit_curves: list of (key, DataFrame)."""
    stacked = pd.concat({key: curve for key, curve in unit_curves},
                        names=['unit', 'generation'])
    return (stacked.groupby(level='generation').mean(),
            stacked.groupby(level='generation').std())


def make_units(run_frames):
    """Turn an arm's per-run curves into the units the mean +/- std is taken over:
    every run is one unit, pooled across seeds AND replicates, so an arm's line is
    the average over all its runs in that environment and the band is their
    standard deviation.

    run_frames: list of (seed, rep, DataFrame). Returns list of (key, DataFrame)
    with scalar string keys — stack_mean_std builds a (unit, generation)
    MultiIndex, so a tuple key would add a level and collide with the frames' own
    'generation' index name."""
    return [(f'seed{seed}_r{rep}', df)
            for seed, rep, df in sorted(run_frames, key=lambda t: (t[0], t[1]))]


def _draw(ax, buf_stats, col, smooth, logy=False):
    """One line per buffer size (mean +/- std band) for column `col`. Returns True
    if anything was drawn. buf_stats: buffer -> (mean_df, std_df), ascending."""
    drawn = False
    for buf, (mean_df, std_df) in buf_stats.items():
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
        colour = _BUF_COLOURS[buf]
        ax.plot(s.index, s.values, color=colour, lw=2, label=_buf_label(buf))
        ax.fill_between(s.index, lo, hi, color=colour, alpha=0.12)
        drawn = True
    return drawn


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, title='collapse buffer', ncol=1)


def _has_signal(buf_stats, col):
    """True if any arm's mean curve for `col` is non-empty and not identically
    zero. Drops the panels that are structurally flat rather than uninformative —
    predator_touches / drained_ticks in the predator-free baseline, where an
    all-zero axis would waste a sixth of the figure."""
    for mean_df, _sd in buf_stats.values():
        if col not in mean_df.columns:
            continue
        s = mean_df[col].dropna()
        if len(s) and (s.abs() > 0).any():
            return True
    return False


def _grid_plot(buf_stats, metrics, titles, smooth, out_dir, fname, suptitle,
               ncols=2, ylim_bottom_zero=False, log_cols=frozenset()):
    """Shared panel-grid helper: one panel per metric, one line per buffer size."""
    metrics = [m for m in metrics if _has_signal(buf_stats, m)]
    if not metrics:
        print(f"  (no columns for {fname} — skipping)")
        return
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows),
                            sharex=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, m in zip(axs, metrics):
        logy = m in log_cols
        _draw(ax, buf_stats, m, smooth, logy=logy)
        ax.set_title(titles[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(titles[m] + (' (log)' if logy else ''))
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        if logy:
            ax.set_yscale('log')
        elif ylim_bottom_zero:
            ax.set_ylim(bottom=0)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    # Label the BOTTOM-MOST USED axis of each column, not the last row: with
    # sharex, a column whose final row was blanked above would otherwise keep its
    # tick labels hidden and end up with a bare, unreadable x-axis.
    for c in range(ncols):
        used = [i for i in range(c, len(metrics), ncols)]
        if not used:
            continue
        ax = axs[used[-1]]
        ax.set_xlabel('Generation')
        ax.tick_params(labelbottom=True)
    _legend(axs[0])
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{fname}{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gen_metrics(buf_stats, smooth, out_dir):
    """Fitness + population outcome metrics over generations; one line per arm."""
    print("  Plotting fitness + population vs buffer size ...")
    _grid_plot(buf_stats, list(GEN_METRICS), GEN_METRICS, smooth, out_dir,
               'buf_fitness_population',
               'Fitness & population over generations — collapse-buffer sweep\n'
               f'({_ENV_DESC}; pure-RL condition; mean ± std {_BAND_DESC})',
               ncols=3)


def plot_diagnostics(buf_stats, smooth, out_dir):
    """MAD + genomic variance (+ weight magnitude / inter-gen change) over
    generations; one line per arm. MAD is the within-life learning signal, so a
    buffer that reseeds from stale weights shows up here before it shows up in
    fitness; genome_variance is the population diversity the bigger buffer is
    meant to buy. Both use a log y-axis — over 1000 generations they cross
    decades, and a linear axis would show only the early transient."""
    print("  Plotting learning diagnostics (MAD, genomic variance) ...")
    _grid_plot(buf_stats, list(DIAG_METRICS), DIAG_METRICS, smooth, out_dir,
               'buf_learning_diagnostics',
               'Learning diagnostics over generations — collapse-buffer sweep\n'
               f'(MAD = within-life weight change; genomic variance = population '
               f'diversity in weight space; {_ENV_DESC}; mean ± std {_BAND_DESC})',
               log_cols=LOG_METRICS)


def plot_collapse(buf_stats, smooth, out_dir):
    """The buffer mechanism itself, from events.csv. Panel 1-2 are the size of
    each window's respawn (survivors / respawned), panel 3 the collapse rate
    (smoothed, so it reads as 'fraction of recent windows that wiped out') and
    panel 4 how full the buffer actually was when drawn from — a 100-slot buffer
    that never fills past 40 is not really a 100 arm."""
    print("  Plotting collapse / buffer usage ...")
    _grid_plot(buf_stats, list(EVENT_METRICS), EVENT_METRICS, smooth, out_dir,
               'buf_collapse',
               'Collapse & buffer usage over generations — collapse-buffer sweep\n'
               f'(per window, from events.csv; {_ENV_DESC}; mean ± std {_BAND_DESC})',
               ylim_bottom_zero=True)


def plot_behaviour_curves(buf_stats, smooth, out_dir):
    """Per-organism scalar behaviours over generations; one line per arm."""
    print("  Plotting behaviour curves vs buffer size ...")
    _grid_plot(buf_stats, list(METRICS), METRICS, smooth, out_dir,
               'buf_behaviour_curves',
               'Behaviour over generations — collapse-buffer sweep\n'
               f'({_ENV_DESC}; population average; mean ± std {_BAND_DESC})')


def plot_cave_day_night(buf_stats, smooth, out_dir):
    """Cave entries split day vs night + the night share over time; one line per
    arm. Panels 1-2 share a y-axis so day and night are directly comparable;
    panel 3 is the preference itself, with run-to-run spread as a band."""
    print("  Plotting cave entries by time of day vs buffer size ...")
    day_col, night_col = 'cave_entries_day', 'cave_entries_night'
    if not any(day_col in md.columns and night_col in md.columns
               for md, _sd in buf_stats.values()):
        print("  (no cave_entries_day / cave_entries_night columns — skipping)")
        return
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    axs = np.atleast_1d(axs)

    for ax, (col, panel_title) in zip(axs[:2],
                                      [(day_col, 'Cave entries — daytime'),
                                       (night_col, 'Cave entries — night-time')]):
        _draw(ax, buf_stats, col, smooth)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cave entries per organism')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    top = max((ax.get_ylim()[1] for ax in axs[:2]), default=1)
    for ax in axs[:2]:
        ax.set_ylim(0, top)

    # Panel 3: night share = night / (day + night), computed per run from that
    # generation's total entries. Isolates the day-vs-night PREFERENCE from the
    # (large) variation in total cave use; 0.5 = no preference.
    ax = axs[2]
    _draw(ax, buf_stats, 'night_share', smooth)
    ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
    ax.text(0.01, 0.5, ' no preference (0.5)', color='#6B7280', fontsize=8,
            va='bottom', ha='left', transform=ax.get_yaxis_transform())
    ax.set_title('Night share of cave entries (preference over time)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('night / (day + night)')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    _legend(axs[0])
    plt.suptitle('Cave entries by time of day — collapse-buffer sweep\n'
                 f'({_ENV_DESC}; population average; mean ± std {_BAND_DESC})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'buf_cave_day_night{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_food_lines(buf_stats, smooth, out_dir):
    """Food-tier diet composition over generations as line panels (one per tier);
    one line per arm. Shows whether a wider respawn pool shifts diet up-tier."""
    print("  Plotting food-tier composition vs buffer size ...")
    titles = {f'{c}_frac': f'{FOOD_TIERS[c][0]} food' for c in FOOD_COLS}
    _grid_plot(buf_stats, list(titles), titles, smooth, out_dir,
               'buf_behaviour_food',
               'Food-tier composition over generations — collapse-buffer sweep\n'
               f'(share of diet in each tier, {_ENV_DESC}; mean ± std {_BAND_DESC})',
               ylim_bottom_zero=True)


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


def final_window_table(per_unit, buffers, final_frac):
    """Per-run final-window value of every VS_BUFFER metric.
    per_unit: buffer -> list of (run key, per-gen DF). Returns a tidy DataFrame
    (collapse_buffer, unit, metric, value), one row per run x metric."""
    rows = []
    for b in buffers:
        for unit, df in per_unit.get(b, []):
            for col, _title in VS_BUFFER:
                rows.append({'collapse_buffer': b, 'unit': str(unit),
                             'metric': col, 'value': final_value(df, col, final_frac)})
    return pd.DataFrame(rows)


def plot_vs_buffer(final_tbl, buffers, final_frac, out_dir):
    """Final-window value of the key metrics plotted AGAINST buffer size, mean +/-
    std across the arm's runs. The at-a-glance 'which buffer won?' summary.

    Buffer sizes go on an evenly spaced categorical axis: with only two arms a
    numeric axis says nothing a categorical one doesn't, and it keeps the figure
    correct if a later run sweeps 25/50/100/200 (geometric spacing)."""
    print("  Plotting final-window metrics vs buffer size ...")
    xs = np.arange(len(buffers))
    labels = [str(b) for b in buffers]

    ncols = 3
    nrows = (len(VS_BUFFER) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (col, title) in zip(axs, VS_BUFFER):
        means, stds, px = [], [], []
        for i, b in enumerate(buffers):
            vals = final_tbl[(final_tbl['collapse_buffer'] == b) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            if not len(vals):
                continue
            px.append(xs[i])
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
        if not px:
            ax.set_title(f'{title}\n(no data)', color='gray'); ax.axis('off'); continue
        ax.errorbar(px, means, yerr=stds, marker='o', lw=2, capsize=4,
                    color='#111827', ecolor='#6B7280', zorder=2)
        # Individual runs behind the summary line, so a mean driven by one
        # outlier run is visible rather than hidden inside the error bar.
        for i, b in enumerate(buffers):
            vals = final_tbl[(final_tbl['collapse_buffer'] == b) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            if len(vals):
                ax.scatter(np.full(len(vals), xs[i]), vals.values, s=22,
                           color=_BUF_COLOURS[b], alpha=0.6, zorder=3)
        if col == 'night_share':
            ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('collapse buffer size')
        ax.set_ylabel(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(buffers) - 0.5)
        ax.grid(True, alpha=0.3)
    for ax in axs[len(VS_BUFFER):]:
        ax.axis('off')
    plt.suptitle('Final-window outcome vs collapse buffer size — '
                 f'{_ENV_DESC}\n(mean ± std {_BAND_DESC}, over the last '
                 f'{int(final_frac * 100)}% of generations; dots = individual runs)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'buf_vs_buffer{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _cliffs_delta(a, b):
    """Cliff's delta for b vs a: P(b > a) - P(b < a), in [-1, 1]. Non-parametric
    effect size — reported alongside p because 15-vs-15 pure-RL runs can be
    significantly different and practically identical, or vice versa."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    diff = b[:, None] - a[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size


def arm_comparison(final_tbl, buffers, final_frac):
    """Arm-vs-arm comparison of the final-window metrics, over the same run-level
    units the curves are averaged over (n = seeds x replicates per arm).

    Compares every non-default arm against the default (25); returns a tidy
    DataFrame, empty when there is only one arm. final_tbl: tidy
    (collapse_buffer, unit, metric, value)."""
    base = _DEFAULT_BUFFER if _DEFAULT_BUFFER in buffers else buffers[0]
    others = [b for b in buffers if b != base]
    rows = []
    for b in others:
        for col, title in VS_BUFFER:
            av = final_tbl[(final_tbl['collapse_buffer'] == base) &
                           (final_tbl['metric'] == col)]['value'].dropna()
            cv = final_tbl[(final_tbl['collapse_buffer'] == b) &
                           (final_tbl['metric'] == col)]['value'].dropna()
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
                'metric': col, 'label': title,
                'baseline_buffer': base, 'buffer': b,
                f'mean_buf{base}': av.mean(), f'mean_buf{b}': cv.mean(),
                f'std_buf{base}': av.std(ddof=0), f'std_buf{b}': cv.std(ddof=0),
                'diff': cv.mean() - av.mean(),
                'pct_change': (100.0 * (cv.mean() - av.mean()) / av.mean()
                               if av.mean() else np.nan),
                'mannwhitney_p': p,
                'cliffs_delta': _cliffs_delta(av.values, cv.values),
                'n_baseline': len(av), 'n_buffer': len(cv),
                'final_frac': final_frac,
            })
    return pd.DataFrame(rows)


def write_summary(buf_stats, final_tbl, arm_tbl, out_dir):
    """Tidy (collapse_buffer, generation, metric, mean, std) across the arm's
    runs, plus the per-run final-window table behind buf_vs_buffer and the arm
    comparison."""
    frames = []
    for buf, (mean_df, std_df) in buf_stats.items():
        m_long = mean_df.reset_index().melt(id_vars='generation',
                                            var_name='metric', value_name='mean')
        s_long = std_df.reset_index().melt(id_vars='generation',
                                           var_name='metric', value_name='std')
        merged = m_long.merge(s_long, on=['generation', 'metric'])
        merged.insert(0, 'collapse_buffer', buf)
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f'buf_behaviour_summary{_SUFFIX}.csv')
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")

    path = os.path.join(out_dir, f'buf_final_window{_SUFFIX}.csv')
    final_tbl.to_csv(path, index=False)
    print(f"  Saved: {path}")

    if len(arm_tbl):
        path = os.path.join(out_dir, f'buf_arm_comparison{_SUFFIX}.csv')
        arm_tbl.to_csv(path, index=False)
        print(f"  Saved: {path}")


def print_final_table(final_tbl, buffers):
    """Console summary: final-window mean +/- std per arm for each key metric, so
    the headline numbers are readable without opening the PNGs."""
    print("\n  Final-window summary (mean ± std across all runs of the arm):")
    header = '    {:<30}'.format('metric') + ''.join(f'{b:>20d}' for b in buffers)
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for col, title in VS_BUFFER:
        cells = []
        for b in buffers:
            vals = final_tbl[(final_tbl['collapse_buffer'] == b) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            cells.append('n/a'.rjust(20) if not len(vals) else
                         f'{vals.mean():.4g} ± {vals.std(ddof=0):.2g}'.rjust(20))
        print('    {:<30}'.format(title[:30]) + ''.join(cells))


def print_arm_table(arm_tbl):
    """Console version of the arm-vs-arm comparison (run level)."""
    if not len(arm_tbl):
        return
    for buf, sub in arm_tbl.groupby('buffer'):
        base = int(sub['baseline_buffer'].iloc[0])
        print(f"\n  Arm comparison: buffer {int(buf)} vs buffer {base} "
              f"(n={int(sub['n_buffer'].iloc[0])} vs "
              f"{int(sub['n_baseline'].iloc[0])} runs):")
        print('    {:<30}{:>12}{:>10}{:>10}{:>9}'.format(
            'metric', 'diff', '%', 'p(MWU)', 'delta'))
        print('    ' + '-' * 71)
        for _, r in sub.iterrows():
            p = 'n/a' if not np.isfinite(r['mannwhitney_p']) else f"{r['mannwhitney_p']:.3f}"
            star = ' *' if np.isfinite(r['mannwhitney_p']) and r['mannwhitney_p'] < 0.05 else ''
            print('    {:<30}{:>12.4g}{:>10.1f}{:>10}{:>9.2f}{}'.format(
                r['label'][:30], r['diff'], r['pct_change'], p,
                r['cliffs_delta'], star))
        print("    * p < 0.05 (Mann-Whitney, uncorrected across the "
              f"{len(sub)} metrics above — treat as descriptive).")


def run_env(env_key, args):
    """Discover + aggregate + plot one environment's collapse-buffer sweep."""
    env = ENVIRONMENTS[env_key]
    lr = args.learning_lr if args.learning_lr is not None else env['lr']
    if env_key == 'hard':
        roaming, drain = args.predators, args.drain
    else:
        roaming, drain = env['roaming'], env['drain']

    out_dir = args.out or f'output/buffer_sweep_{env_key}'
    os.makedirs(out_dir, exist_ok=True)

    global _ENV_DESC, _SUFFIX, _BAND_DESC, _BUF_COLOURS
    _ENV_DESC = env['desc'] if env_key == 'baseline' else \
        f'{int(roaming)} roaming predators, drain {int(drain)}'
    seed_filter = set(args.seeds) if args.seeds else None
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''
    _BAND_DESC = 'across all runs of the arm (seeds × replicates pooled)'

    want = {'condition': 'learning', 'mode': 'pure_rl', 'epsilon_enabled': False,
            'learning_rate': lr, 'roaming': roaming, 'drain': drain,
            'hidden_size': float(args.hidden_size)}

    print(f"\n=== {env_key.upper()} environment ({_ENV_DESC}, lr {lr:g}) ===")
    print(f"Discovering collapse-buffer sweep runs under {args.root} ...")
    runs = discover(args.root, want)
    if seed_filter is not None:
        runs = [r for r in runs if r['seed'] in seed_filter]
    if not runs:
        print(f"  WARNING: no matching runs (env={env_key}, lr={lr}, roaming={roaming}"
              f"{'' if drain is None else f', drain={drain}'}). Skipping this env.")
        return

    by_buf = {}
    for r in runs:
        by_buf.setdefault(r['buffer'], []).append(r)
    buffers = sorted(by_buf)
    print(f"  Found {len(runs)} runs across buffers {buffers} "
          f"({', '.join(f'{b}: {len(by_buf[b])} runs' for b in buffers)}). "
          f"Reading logs ...")

    if len(buffers) <= len(_ARM_COLOURS):
        _BUF_COLOURS = {b: _ARM_COLOURS[i] for i, b in enumerate(buffers)}
    else:   # a 5+-value sweep is an ordered axis: sequential ramp instead
        cmap = plt.get_cmap('viridis')
        ncol = max(len(buffers) - 1, 1)
        _BUF_COLOURS = {b: cmap(0.08 + 0.84 * i / ncol) for i, b in enumerate(buffers)}

    cache_dir = None if args.no_cache else args.cache_dir

    buf_stats = {}   # buffer -> (mean_df, std_df) across runs
    per_unit = {}    # buffer -> list of (run key, per-gen DataFrame)
    for b in buffers:
        run_frames = []      # (seed, rep, joined per-generation curve)
        for r in by_buf[b]:
            curve = load_run(r['org_csv'], r['name'], cache_dir)
            if not len(curve):
                print(f"    skip buffer {b} seed {r['seed']} r{r['rep']} — no rows")
                continue
            curve = curve[curve.index <= args.max_gen]
            # Join the generation-level outcomes + diagnostics and the events-derived
            # buffer mechanism onto the same generation index, per run, BEFORE
            # aggregating: every source comes from the same run, so the join is
            # exact and the units downstream carry all three families at once.
            gm = load_gen_metrics(os.path.join(r['dir'], 'generations.csv'), args.max_gen)
            if gm is not None and len(gm):
                curve = curve.join(gm, how='outer')
            ev = load_events(r['events_csv'], args.max_gen, r['pop_size'])
            if ev is not None and len(ev):
                curve = curve.join(ev, how='outer')
            elif not args.quiet:
                print(f"    note: no usable events.csv for {r['name']} — "
                      f"collapse panels will miss this run")
            run_frames.append((r['seed'], r['rep'], curve))
        if not run_frames:
            continue

        units = make_units(run_frames)
        mean_df, std_df = stack_mean_std(units)
        buf_stats[b] = (mean_df, std_df)
        per_unit[b] = units
        print(f"    buffer {b:<5d} averaged over {len(units)} runs, "
              f"{int(mean_df.index.max())} gens")

    if not buf_stats:
        print("  WARNING: nothing loaded for this env. Skipping.")
        return

    print("Plotting ...")
    buffers = sorted(buf_stats)
    final_tbl = final_window_table(per_unit, buffers, args.final_frac)
    arm_tbl = arm_comparison(final_tbl, buffers, args.final_frac) \
        if len(buffers) > 1 else pd.DataFrame()

    plot_gen_metrics(buf_stats, args.smooth, out_dir)
    plot_diagnostics(buf_stats, args.smooth, out_dir)
    plot_collapse(buf_stats, args.smooth, out_dir)
    plot_behaviour_curves(buf_stats, args.smooth, out_dir)
    plot_cave_day_night(buf_stats, args.smooth, out_dir)
    plot_food_lines(buf_stats, args.smooth, out_dir)
    plot_vs_buffer(final_tbl, buffers, args.final_frac, out_dir)
    write_summary(buf_stats, final_tbl, arm_tbl, out_dir)
    print_final_table(final_tbl, buffers)
    print_arm_table(arm_tbl)
    print(f"\nDone ({env_key}). Outputs in {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser(
        description='Effect of the pure-RL collapse-respawn buffer size on '
                    'fitness, population, the learning diagnostics (MAD and '
                    'genomic variance) and behaviour, baseline + hard '
                    'environments.')
    ap.add_argument('--env', choices=['baseline', 'hard', 'both'], default='both',
                    help='Environment(s) to analyse (default: both).')
    ap.add_argument('--root', default='logs/learning/pure_rl/auto-run',
                    help='Folder holding the buffer-sweep run dirs (pure-RL mode; '
                         'default logs/learning/pure_rl/auto-run).')
    ap.add_argument('--learning-lr', type=float, default=None,
                    help='learning_rate selecting the runs. Default: env optimum '
                         '(0.01 baseline / 0.02 hard).')
    ap.add_argument('--hidden-size', type=int, default=64,
                    help='NNBrain hidden width selecting the runs (default 64, '
                         'the width the buffer sweep ran at). Keeps the later '
                         'network-size sweep out of these arms.')
    ap.add_argument('--predators', type=float, default=60,
                    help='roaming_predator_count for the HARD env (default 60).')
    ap.add_argument('--drain', type=float, default=5,
                    help='predator_drain for the HARD env (default 5).')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this.')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curves/bands (1 = none).')
    ap.add_argument('--final-frac', type=float, default=0.2,
                    help='Fraction of the final generations averaged for the '
                         'vs-buffer summary plot (default 0.2 = last 20%%).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered.')
    ap.add_argument('--cache-dir', default='output/.cache_buffer_curves',
                    help='Where per-run organisms.csv summaries are cached.')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-read every organisms.csv instead of using the cache.')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress the per-run notes about missing events.csv.')
    ap.add_argument('-o', '--out', default=None,
                    help='Output directory (default: output/buffer_sweep_<env>). '
                         'With --env both this is ignored (each env needs its own).')
    args = ap.parse_args()

    envs = ['baseline', 'hard'] if args.env == 'both' else [args.env]
    if args.env == 'both' and args.out:
        print("note: --out ignored with --env both (each env writes its own dir).")
        args.out = None
    for env_key in envs:
        run_env(env_key, args)
    print()


if __name__ == '__main__':
    main()
