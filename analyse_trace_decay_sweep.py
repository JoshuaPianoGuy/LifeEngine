"""
analyse_trace_decay_sweep.py
======================================================================
Effect of the eligibility-trace decay gamma on fitness, population AND
behaviour (cave usage by time of day, exploration, diet), from the trace-decay
sweep runs produced by run_trace_decay_sweep_baseline_array.slurm and
run_trace_decay_sweep_hard_array.slurm.

Those jobs sweep the NNBrain eligibility-trace decay — each tick

    trace <- gamma * trace + grad log pi(a)

so the credit horizon is ~1/(1-gamma) ticks — across 4 values, each replicated
over 3 map seeds x 3 independent replicates, for the learning condition
(GA + on-policy REINFORCE, no epsilon):

    trace_decay : 0.90 (~10 ticks), 0.95 (~20), 0.98 (~50), 0.99 (~100)
    seeds       : 1, 42, 999      (SAME 3 maps as the LR / epsilon sweeps)
    replicates  : 1..3            (identical terrain, different GA/RL chance)
    4 gammas x 3 seeds x 3 replicates = 36 runs per environment.

0.90 is the value every earlier experiment used (the old hard-coded constant),
so that arm doubles as the baseline the rest of the sweep is measured against.

The question is whether spreading credit further back in a lifetime changes
WHAT organisms learn to do — in particular cave usage, which is the clearest
delayed-reward behaviour in the model: sheltering costs foraging time now and
pays off later, so a longer trace is exactly the thing that could make it
learnable. Hence the day/night cave panels and the night-share-over-time curve.

This is the trace-decay analogue of analyse_explore_bonus_behaviour.py: same
organisms.csv / generations.csv schema and the same behaviour + outcome plots,
but the SERIES is the trace_decay value within the single learning condition.
Because the sweep axis is an ordered scalar, the series use a sequential colour
ramp (dark = 0.90, bright = 0.99) rather than categorical colours.

AGGREGATION (this is what makes the bands mean something)
---------------------------------------------------------
Only the map TERRAIN is seeded; GA mutation and RL action sampling are unseeded,
so the 3 replicates of a (gamma, seed) cell differ purely by algorithmic chance.
Following the sweep's own design note, replicates are POOLED FIRST: the 3 runs
of a cell are averaged into one curve per (gamma, seed), and the plotted band is
then the spread across the 3 seeds — i.e. map-to-map variation, with run-to-run
noise already averaged down. `--band run` instead treats all 9 runs of a gamma
as independent units, giving the wider "seed + chance" spread.

Runs in BOTH environments by default (--env both), each into its own output dir
so baseline and hard never mix:
    baseline   predator-free (roaming 0);              RL lr 0.01
    hard       60 roaming predators, drain 5;          RL lr 0.02
The two sweeps differ ONLY in those settings, so the pair answers "does the best
gamma depend on the environment?".

Outputs (into --out / output/trace_decay_<env>):
    td_fitness_population.png   -- avg/top-20%/best fitness, peak population,
                                   total agents, avg lifetime over generations;
                                   one line per gamma.
    td_behaviour_curves.png     -- per-organism scalar behaviours over generations
                                   (cells explored, cave entries, predator
                                   encounters, food eaten, lifetime, …).
    td_cave_day_night.png       -- cave entries split day vs night + the night
                                   share over time. THE cave-usage view: panel 3
                                   is the day-vs-night PREFERENCE as it evolves.
    td_behaviour_food.png       -- food-tier diet composition (one panel/tier).
    td_vs_trace.png             -- final-window (last --final-frac of generations)
                                   value of the key fitness + cave metrics plotted
                                   AGAINST gamma, mean ± std across units. The
                                   at-a-glance "which gamma won?" summary.
    td_behaviour_summary.csv    -- tidy (trace_decay, generation, metric, mean, std).
    td_final_window.csv         -- per-unit final-window values behind td_vs_trace.

Reading 36 organisms.csv files per environment is the slow part, so each run's
per-generation curve is cached under --cache-dir (keyed on the CSV's size+mtime);
re-plotting with a different --smooth or --final-frac is then near-instant.

Usage
-----
    python analyse_trace_decay_sweep.py                     # both envs
    python analyse_trace_decay_sweep.py --env baseline
    python analyse_trace_decay_sweep.py --env hard --seeds 1 42
    python analyse_trace_decay_sweep.py --band run --smooth 10
    python analyse_trace_decay_sweep.py --no-cache          # force re-read
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

# ── Environment presets (learning-condition sweep; lr differs per env) ────────
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
GEN_COLS = list(GEN_METRICS)

# Metrics for the final-window "vs gamma" summary plot.
VS_TRACE = [
    ('avg_fitness',          'Mean average fitness'),
    ('top20percent_fitness', 'Top-20% fitness'),
    ('peak_population',      'Peak population'),
    ('cave_entries',         'Cave entries / organism'),
    ('night_share',          'Night share of cave entries'),
    ('cells_visited',        'Cells explored / organism'),
]

# Sweep-run gate + gamma source: only folders carrying a _trace<val>_ tag are
# this sweep's runs. Keying gamma off params.json instead would sweep in every
# OTHER learning run that matches the same env filter, since they all carry the
# default trace_decay 0.9 and would pile into the 0.90 arm.
_TRACE_RE = re.compile(r'_trace([0-9.]+)_')
_REP_RE = re.compile(r'_r(\d+)$')

# Set per-env in run_env(); used by the plot titles / filenames.
_ENV_DESC = ''
_SUFFIX = ''
_BAND_DESC = ''
_TD_COLOURS = {}   # gamma -> colour (sequential ramp), filled per run.


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def _td_label(gamma):
    """Consistent display label for a gamma, with its ~1/(1-gamma) tick horizon."""
    return f'γ {gamma:g}  (~{int(round(1.0 / (1.0 - gamma)))} ticks)'


def discover(root, want):
    """Find trace-decay sweep run folders under `root` (params.json +
    organisms.csv) matching every filter in `want`, returning
    {gamma, seed, rep, name, dir, org_csv} dicts.

    `want` keys: condition (req), mode, epsilon_enabled, learning_rate, roaming,
    drain (skipped if None) — same semantics as analyse_explore_bonus_behaviour.py."""
    runs = []
    for params_path in glob.glob(os.path.join(root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        org_csv = os.path.join(run_dir, 'organisms.csv')
        if not os.path.exists(org_csv):
            continue
        name = os.path.basename(run_dir)
        m = _TRACE_RE.search(name)
        if not m:      # no _trace tag => not a sweep run
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
        if abs(_num(p, 'roaming_predator_count') - want['roaming']) > 1e-9:
            continue
        if want.get('drain') is not None and abs(_num(p, 'predator_drain') - want['drain']) > 1e-9:
            continue
        gamma = float(m.group(1))
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        rm = _REP_RE.search(name)
        rep = int(rm.group(1)) if rm else 1
        runs.append({'gamma': gamma, 'seed': seed, 'rep': rep, 'name': name,
                     'dir': run_dir, 'org_csv': org_csv})
    return sorted(runs, key=lambda r: (r['gamma'], r['seed'], r['rep']))


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
    """Read the outcome metrics (GEN_COLS) from a run's generations.csv, indexed
    by generation. Deduplicates on generation (keep last). None if unavailable."""
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
    """Aggregate a gamma's per-unit curves -> (mean, std) DataFrames indexed by
    generation. unit_curves: list of (key, DataFrame)."""
    stacked = pd.concat({key: curve for key, curve in unit_curves},
                        names=['unit', 'generation'])
    return (stacked.groupby(level='generation').mean(),
            stacked.groupby(level='generation').std())


def make_units(run_frames, band):
    """Turn a gamma's per-run curves into the units the mean ± std is taken over.

    band='seed' (default): pool the replicates of each (gamma, seed) cell into
    one curve, so the band is map-to-map variation with run-to-run chance already
    averaged down — the comparison the sweep was designed for.
    band='run': every run is its own unit, so the band also carries the
    algorithmic noise between replicates of the same map.

    run_frames: list of (seed, rep, DataFrame). Returns list of (key, DataFrame)
    with scalar string keys — concat below builds a (unit, generation) MultiIndex,
    so a tuple key would add a level and collide with the frames' own
    'generation' index name."""
    if band == 'run':
        return [(f'seed{seed}_r{rep}', df)
                for seed, rep, df in sorted(run_frames, key=lambda t: (t[0], t[1]))]
    by_seed = {}
    for seed, _rep, df in run_frames:
        by_seed.setdefault(seed, []).append(df)
    units = []
    for seed in sorted(by_seed):
        frames = by_seed[seed]
        pooled = (frames[0] if len(frames) == 1 else
                  pd.concat(frames).groupby(level=0).mean())
        units.append((f'seed{seed}', pooled))
    return units


def _draw(ax, td_stats, col, smooth):
    """One line per gamma (mean ± std band) for column `col`. Returns True if
    anything was drawn. td_stats: gamma -> (mean_df, std_df), ascending gamma."""
    drawn = False
    for gamma, (mean_df, std_df) in td_stats.items():
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
        colour = _TD_COLOURS[gamma]
        ax.plot(s.index, s.values, color=colour, lw=2, label=_td_label(gamma))
        ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                        color=colour, alpha=0.12)
        drawn = True
    return drawn


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, title='trace decay', ncol=1)


def _has_signal(td_stats, col):
    """True if any gamma's mean curve for `col` is non-empty and not identically
    zero. Drops the panels that are structurally flat rather than uninformative —
    predator_touches / drained_ticks in the predator-free baseline, where an
    all-zero axis would waste a sixth of the figure."""
    for mean_df, _sd in td_stats.values():
        if col not in mean_df.columns:
            continue
        s = mean_df[col].dropna()
        if len(s) and (s.abs() > 0).any():
            return True
    return False


def _grid_plot(td_stats, metrics, titles, smooth, out_dir, fname, suptitle,
               ncols=2, ylim_bottom_zero=False):
    """Shared panel-grid helper: one panel per metric, one line per gamma."""
    metrics = [m for m in metrics if _has_signal(td_stats, m)]
    if not metrics:
        print(f"  (no columns for {fname} — skipping)")
        return
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows),
                            sharex=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, m in zip(axs, metrics):
        _draw(ax, td_stats, m, smooth)
        ax.set_title(titles[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(titles[m])
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        if ylim_bottom_zero:
            ax.set_ylim(bottom=0)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    for ax in axs[max(0, len(metrics) - ncols):len(metrics)]:
        ax.set_xlabel('Generation')
    _legend(axs[0])
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{fname}{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gen_metrics(td_stats, smooth, out_dir):
    """Fitness + population outcome metrics over generations; one line per gamma."""
    print("  Plotting fitness + population vs trace decay ...")
    _grid_plot(td_stats, list(GEN_METRICS), GEN_METRICS, smooth, out_dir,
               'td_fitness_population',
               'Fitness & population over generations — trace-decay sweep\n'
               f'({_ENV_DESC}; learning condition; mean ± std {_BAND_DESC})',
               ncols=3)


def plot_behaviour_curves(td_stats, smooth, out_dir):
    """Per-organism scalar behaviours over generations; one line per gamma."""
    print("  Plotting behaviour curves vs trace decay ...")
    _grid_plot(td_stats, list(METRICS), METRICS, smooth, out_dir,
               'td_behaviour_curves',
               'Behaviour over generations — trace-decay sweep\n'
               f'({_ENV_DESC}; population average; mean ± std {_BAND_DESC})')


def plot_cave_day_night(td_stats, smooth, out_dir):
    """Cave entries split day vs night + the night share over time; one line per
    gamma. THE view for 'does a longer credit horizon change when organisms
    shelter?'. Panels 1-2 share a y-axis so day and night are directly comparable;
    panel 3 is the preference itself, with run-to-run spread as a band."""
    print("  Plotting cave entries by time of day vs trace decay ...")
    day_col, night_col = 'cave_entries_day', 'cave_entries_night'
    if not any(day_col in md.columns and night_col in md.columns
               for md, _sd in td_stats.values()):
        print("  (no cave_entries_day / cave_entries_night columns — skipping)")
        return
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    axs = np.atleast_1d(axs)

    for ax, (col, panel_title) in zip(axs[:2],
                                      [(day_col, 'Cave entries — daytime'),
                                       (night_col, 'Cave entries — night-time')]):
        _draw(ax, td_stats, col, smooth)
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
    _draw(ax, td_stats, 'night_share', smooth)
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
    plt.suptitle('Cave entries by time of day — trace-decay sweep\n'
                 f'({_ENV_DESC}; population average; mean ± std {_BAND_DESC})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'td_cave_day_night{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_food_lines(td_stats, smooth, out_dir):
    """Food-tier diet composition over generations as line panels (one per tier);
    one line per gamma. Shows whether a longer trace shifts diet up-tier."""
    print("  Plotting food-tier composition vs trace decay ...")
    titles = {f'{c}_frac': f'{FOOD_TIERS[c][0]} food' for c in FOOD_COLS}
    _grid_plot(td_stats, list(titles), titles, smooth, out_dir,
               'td_behaviour_food',
               'Food-tier composition over generations — trace-decay sweep\n'
               f'(share of diet in each tier, {_ENV_DESC}; mean ± std {_BAND_DESC})',
               ylim_bottom_zero=True)


def final_window_table(per_unit, gammas, final_frac):
    """Per-unit final-window value of every VS_TRACE metric.
    per_unit: gamma -> list of (unit key, combined per-gen DF). Returns a tidy
    DataFrame (trace_decay, unit, metric, value)."""
    def final_value(df, col):
        if col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if not len(s):
            return np.nan
        cutoff = s.index.max() - final_frac * (s.index.max() - s.index.min())
        window = s[s.index >= cutoff]
        return window.mean() if len(window) else s.iloc[-1]

    rows = []
    for g in gammas:
        for unit, df in per_unit.get(g, []):
            for col, _title in VS_TRACE:
                rows.append({'trace_decay': g, 'unit': str(unit),
                             'metric': col, 'value': final_value(df, col)})
    return pd.DataFrame(rows)


def plot_vs_trace(final_tbl, gammas, final_frac, out_dir):
    """Final-window value of the key metrics plotted AGAINST gamma, mean ± std
    across units. The at-a-glance 'which credit horizon won?' summary.

    The gammas are plotted on an evenly spaced categorical axis: 0.90/0.95/0.98/
    0.99 are geometrically spaced in credit horizon (~10/20/50/100 ticks), so a
    linear gamma axis would bunch the two most interesting arms on top of each
    other."""
    print("  Plotting final-window metrics vs trace decay ...")
    xs = np.arange(len(gammas))
    labels = [f'{g:g}\n~{int(round(1.0 / (1.0 - g)))} ticks' for g in gammas]

    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    axs = axs.ravel()
    for ax, (col, title) in zip(axs, VS_TRACE):
        means, stds, px = [], [], []
        for i, g in enumerate(gammas):
            vals = final_tbl[(final_tbl['trace_decay'] == g) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            if not len(vals):
                continue
            px.append(xs[i])
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
        if not px:
            ax.set_title(f'{title}\n(no data)', color='gray'); ax.axis('off'); continue
        ax.errorbar(px, means, yerr=stds, marker='o', lw=2, capsize=4,
                    color='#2563EB', ecolor='#6B7280')
        # Individual units behind the summary line, so a mean driven by one
        # outlier unit is visible rather than hidden inside the error bar.
        for i, g in enumerate(gammas):
            vals = final_tbl[(final_tbl['trace_decay'] == g) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            if len(vals):
                ax.scatter(np.full(len(vals), xs[i]), vals.values, s=18,
                           color=_TD_COLOURS[g], alpha=0.55, zorder=3)
        if col == 'night_share':
            ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('trace decay γ (credit horizon)')
        ax.set_ylabel(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
    for ax in axs[len(VS_TRACE):]:
        ax.axis('off')
    plt.suptitle('Final-window outcome vs trace decay — '
                 f'{_ENV_DESC}\n(mean ± std {_BAND_DESC}, over the last '
                 f'{int(final_frac * 100)}% of generations; dots = individual units)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'td_vs_trace{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def write_summary(td_stats, final_tbl, out_dir):
    """Tidy (trace_decay, generation, metric, mean, std) across units, plus the
    per-unit final-window table behind td_vs_trace."""
    frames = []
    for gamma, (mean_df, std_df) in td_stats.items():
        m_long = mean_df.reset_index().melt(id_vars='generation',
                                            var_name='metric', value_name='mean')
        s_long = std_df.reset_index().melt(id_vars='generation',
                                           var_name='metric', value_name='std')
        merged = m_long.merge(s_long, on=['generation', 'metric'])
        merged.insert(0, 'trace_decay', gamma)
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f'td_behaviour_summary{_SUFFIX}.csv')
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")

    path = os.path.join(out_dir, f'td_final_window{_SUFFIX}.csv')
    final_tbl.to_csv(path, index=False)
    print(f"  Saved: {path}")


def print_final_table(final_tbl, gammas):
    """Console summary: final-window mean ± std per gamma for each key metric,
    so the headline numbers are readable without opening the PNGs."""
    print("\n  Final-window summary (mean ± std across units):")
    header = '    {:<28}'.format('metric') + ''.join(f'{g:>18g}' for g in gammas)
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for col, title in VS_TRACE:
        cells = []
        for g in gammas:
            vals = final_tbl[(final_tbl['trace_decay'] == g) &
                             (final_tbl['metric'] == col)]['value'].dropna()
            cells.append('n/a'.rjust(18) if not len(vals) else
                         f'{vals.mean():.3g} ± {vals.std(ddof=0):.2g}'.rjust(18))
        print('    {:<28}'.format(title[:28]) + ''.join(cells))


def run_env(env_key, args):
    """Discover + aggregate + plot one environment's trace-decay sweep."""
    env = ENVIRONMENTS[env_key]
    lr = args.learning_lr if args.learning_lr is not None else env['lr']
    if env_key == 'hard':
        roaming, drain = args.predators, args.drain
    else:
        roaming, drain = env['roaming'], env['drain']

    out_dir = args.out or f'output/trace_decay_{env_key}'
    os.makedirs(out_dir, exist_ok=True)

    global _ENV_DESC, _SUFFIX, _BAND_DESC, _TD_COLOURS
    _ENV_DESC = env['desc'] if env_key == 'baseline' else \
        f'{int(roaming)} roaming predators, drain {int(drain)}'
    seed_filter = set(args.seeds) if args.seeds else None
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''
    _BAND_DESC = ('across seeds (replicates pooled first)' if args.band == 'seed'
                  else 'across all runs (seeds x replicates)')

    want = {'condition': 'learning', 'mode': 'standard', 'epsilon_enabled': False,
            'learning_rate': lr, 'roaming': roaming, 'drain': drain}

    print(f"\n=== {env_key.upper()} environment ({_ENV_DESC}, lr {lr:g}) ===")
    print(f"Discovering trace-decay sweep runs under {args.root} ...")
    runs = discover(args.root, want)
    if seed_filter is not None:
        runs = [r for r in runs if r['seed'] in seed_filter]
    if not runs:
        print(f"  WARNING: no matching runs (env={env_key}, lr={lr}, roaming={roaming}"
              f"{'' if drain is None else f', drain={drain}'}). Skipping this env.")
        return

    by_gamma = {}
    for r in runs:
        by_gamma.setdefault(r['gamma'], []).append(r)
    gammas = sorted(by_gamma)
    print(f"  Found {len(runs)} runs across γ {[f'{g:g}' for g in gammas]} "
          f"({', '.join(f'{g:g}: {len(by_gamma[g])} runs' for g in gammas)}). "
          f"Reading logs ...")

    # Sequential colour ramp over the ordered gammas (dark 0.90 -> bright 0.99).
    cmap = plt.get_cmap('viridis')
    ncol = max(len(gammas) - 1, 1)
    _TD_COLOURS = {g: cmap(0.08 + 0.84 * i / ncol) for i, g in enumerate(gammas)}

    cache_dir = None if args.no_cache else args.cache_dir

    td_stats = {}   # gamma -> (mean_df, std_df) across units
    per_unit = {}   # gamma -> list of (unit key, combined per-gen DataFrame)
    for g in gammas:
        run_frames = []      # (seed, rep, behaviour curve)
        gen_frames = []      # (seed, rep, generations.csv metrics)
        for r in by_gamma[g]:
            curve = load_run(r['org_csv'], r['name'], cache_dir)
            if not len(curve):
                print(f"    skip γ {g:g} seed {r['seed']} r{r['rep']} — no rows")
                continue
            curve = curve[curve.index <= args.max_gen]
            gm = load_gen_metrics(os.path.join(r['dir'], 'generations.csv'), args.max_gen)
            run_frames.append((r['seed'], r['rep'], curve))
            if gm is not None and len(gm):
                gen_frames.append((r['seed'], r['rep'], gm))
        if not run_frames:
            continue

        units = make_units(run_frames, args.band)
        mean_df, std_df = stack_mean_std(units)

        # Fold the generation-level outcomes in, aggregated over the same units.
        combined = {key: df for key, df in units}
        if gen_frames:
            gunits = make_units(gen_frames, args.band)
            gmean, gstd = stack_mean_std(gunits)
            for c in gmean.columns:
                mean_df[c] = gmean[c].reindex(mean_df.index)
                std_df[c] = gstd[c].reindex(std_df.index)
            for key, gdf in gunits:
                if key in combined:
                    combined[key] = combined[key].join(gdf, how='outer')
        td_stats[g] = (mean_df, std_df)
        per_unit[g] = sorted(combined.items(), key=lambda kv: str(kv[0]))
        print(f"    γ {g:<5g} {len(run_frames)} runs -> {len(units)} units, "
              f"{int(mean_df.index.max())} gens")

    if not td_stats:
        print("  WARNING: nothing loaded for this env. Skipping.")
        return

    print("Plotting ...")
    final_tbl = final_window_table(per_unit, gammas, args.final_frac)
    plot_gen_metrics(td_stats, args.smooth, out_dir)
    plot_behaviour_curves(td_stats, args.smooth, out_dir)
    plot_cave_day_night(td_stats, args.smooth, out_dir)
    plot_food_lines(td_stats, args.smooth, out_dir)
    plot_vs_trace(final_tbl, gammas, args.final_frac, out_dir)
    write_summary(td_stats, final_tbl, out_dir)
    print_final_table(final_tbl, gammas)
    print(f"Done ({env_key}). Outputs in {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser(
        description='Effect of the eligibility-trace decay gamma on fitness, '
                    'population and behaviour (cave usage by time of day, diet), '
                    'baseline + hard environments.')
    ap.add_argument('--env', choices=['baseline', 'hard', 'both'], default='both',
                    help='Environment(s) to analyse (default: both).')
    ap.add_argument('--root', default='logs/learning/standard/auto-run',
                    help='Folder holding the trace-decay sweep run dirs '
                         '(learning condition; default logs/learning/standard/auto-run).')
    ap.add_argument('--learning-lr', type=float, default=None,
                    help='learning_rate selecting the runs. Default: env optimum '
                         '(0.01 baseline / 0.02 hard).')
    ap.add_argument('--predators', type=float, default=60,
                    help='roaming_predator_count for the HARD env (default 60).')
    ap.add_argument('--drain', type=float, default=5,
                    help='predator_drain for the HARD env (default 5).')
    ap.add_argument('--band', choices=['seed', 'run'], default='seed',
                    help="What the mean ± std is taken over: 'seed' (default) "
                         'pools each cell\'s replicates first, so the band is '
                         "map-to-map variation; 'run' treats every run as its own "
                         'unit, widening the band with run-to-run chance.')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this.')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curves/bands (1 = none).')
    ap.add_argument('--final-frac', type=float, default=0.2,
                    help='Fraction of the final generations averaged for the '
                         'vs-trace summary plot (default 0.2 = last 20%%).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered.')
    ap.add_argument('--cache-dir', default='output/.cache_trace_curves',
                    help='Where per-run organisms.csv summaries are cached.')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-read every organisms.csv instead of using the cache.')
    ap.add_argument('-o', '--out', default=None,
                    help='Output directory (default: output/trace_decay_<env>). '
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
