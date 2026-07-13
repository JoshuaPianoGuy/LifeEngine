"""
analyse_conditions_comparison.py
======================================================================
Behaviour-over-time comparison of ALL THREE experimental conditions
(EVOLUTION, LEARNING, PURE-RL) in EITHER the predator-free BASELINE
environment or the roaming-predator HARD environment.

This is the three-condition generalisation of analyse_condition_behaviour.py
(which only compared learning vs evolution in the roaming environment). It does
the same thing — population-averaged per-organism behaviours and generation-level
outcomes, averaged across seeds (mean ± std) — but for:

    evolution   GA only, no in-lifetime learning        (condition=evolution)
    learning    GA + in-lifetime REINFORCE              (condition=learning, mode=standard)
    pure RL     REINFORCE only, no GA                    (condition=learning, mode=pure_rl)

in one of two environments, chosen with --env:

    baseline    predator-free  (roaming_predator_count == 0);       RL LR default 0.01
    hard        roaming predators (count == --predators, drain == --drain);  RL LR default 0.02

Each --env produces its OWN set of graphs in its OWN output directory
(default output/condition_comparison_<env>), so baseline and hard never mix.

The three conditions live in different log trees but share the organisms.csv /
generations.csv schema, so each is discovered separately (by params.json) and
tagged with a condition label:

    evolution   --evolution-root (default logs/evolution/standard/auto-run),
                condition=evolution
    learning    --learning-root  (default logs/learning/standard/auto-run),
                condition=learning, mode=standard, epsilon disabled,
                learning_rate == --learning-lr
    pure RL     --pure-rl-root   (default logs/learning/pure_rl/auto-run),
                condition=learning, mode=pure_rl, epsilon disabled,
                learning_rate == --learning-lr

All three are additionally filtered to the chosen environment (roaming==0 for
baseline; roaming==--predators and drain==--drain for hard). Each condition is
aggregated across its seeds (mean ± std), so the condition is the only factor.

organisms.csv logs EVERY organism that lived each generation (its row count per
generation matches total_agents in generations.csv), so each per-generation
value here is the true POPULATION average across all organisms alive that
generation — not just the fittest. These are per-organism, per-lifetime
quantities averaged over the population, then over seeds; the shaded band is the
spread across seeds, not within the population. Conditions can differ in run
length (e.g. a still-running condition may only reach generation ~700); each line
is drawn over its own available range (pass --max-gen N for a strict head-to-head).

MAD (avg_learned_weight_diff — the within-life weight change, the learning
signal) is reported for BOTH RL-using conditions (learning and pure RL) and is
~0 for evolution (active === genome), in the learning-diagnostics plot.

Behaviours read from organisms.csv:
    cells_visited     -- distinct cells visited (exploration / distance proxy)
    predator_touches  -- distinct predator-contact episodes (encounters; ~0 baseline)
    drained_ticks     -- ticks spent being drained by a predator (~0 baseline)
    cave_entries      -- cave entries (+ day/night split in the summary CSV)
    food_total        -- food eaten this lifetime (sum of all tiers)
    lifetime          -- ticks survived
    death_cause       -- how they died: starved / drained / survived  (composition)
    food_<tier>       -- default/low/medium/prestige  (composition)

Outputs (into --out, default output/condition_comparison_<env>):
    condition_behaviour_curves.png       -- one panel per scalar behaviour, one
                                            line per condition (mean ± std across
                                            seeds) over generations.
    condition_lifetime.png               -- POPULATION-average lifetime per
                                            generation (generations.csv avg_lifetime).
    condition_cave_day_night.png         -- cave entries split into daytime vs
                                            night-time panels over generations,
                                            one line per condition (does sheltering
                                            shift to night, when caves are free?).
    condition_fitness_population.png     -- generations.csv outcome metrics
                                            (avg_fitness, top-20% fitness,
                                            peak_population, total_agents).
    condition_learning_diagnostics.png   -- MAD (avg_learned_weight_diff) and
                                            genome_variance over generations, one
                                            line per condition (MAD ~0 evolution,
                                            non-zero learning & pure RL).
    condition_behaviour_death.png        -- per-condition stacked-area death-cause
                                            composition over generations.
    condition_behaviour_food.png         -- food-tier diet composition as LINE
                                            panels (one per tier), one line/cond.
    condition_seed<seed>.png             -- per-seed detail sheet (one per seed):
                                            all three conditions on the SAME map.
    condition_behaviour_summary.csv      -- tidy (condition, generation, metric,
                                            mean, std) across seeds.

Usage
-----
    python analyse_conditions_comparison.py --env hard
    python analyse_conditions_comparison.py --env baseline
    python analyse_conditions_comparison.py --env hard --max-gen 700   # fair head-to-head
    python analyse_conditions_comparison.py --env baseline --seeds 1 42
    python analyse_conditions_comparison.py --env hard --conditions evolution learning
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

warnings.filterwarnings('ignore')

# ── Condition identities ──────────────────────────────────────────────────────
# Internal keys (order = plot order) and their display colours. The display
# labels are built per-run in main() so the RL learning rate can be appended.
COND_ORDER = ['evolution', 'learning', 'pure_rl']
COND_COLOURS_KEY = {
    'evolution': '#DC2626',   # red
    'learning':  '#2563EB',   # blue
    'pure_rl':   '#059669',   # green
}
# Filled in main(): display label -> colour, keyed by the resolved label so the
# plotting helpers (which see labels, not keys) can look colours up.
COND_COLOURS = {}

# Environment presets. `roaming` is the roaming_predator_count filter; `drain` is
# the predator_drain filter (None = don't filter on drain, used for baseline
# where drain is an irrelevant default). `lr` is the default RL learning rate for
# the two RL-using conditions in that environment.
ENVIRONMENTS = {
    'baseline': {'roaming': 0.0,  'drain': None, 'lr': 0.01,
                 'desc': 'predator-free baseline'},
    'hard':     {'roaming': 60.0, 'drain': 5.0,  'lr': 0.02,
                 'desc': '60 roaming predators, drain 5'},
}

# ── Scalar per-organism behaviours (curve panels) ─────────────────────────────
METRICS = {
    'cells_visited':    'Cells explored',
    'predator_touches': 'Predator encounters',
    'drained_ticks':    'Ticks drained by predators',
    'cave_entries':     'Cave entries',
    'food_total':       'Food eaten (total)',
    'lifetime':         'Lifetime (ticks survived)',
}
DEFAULT_METRICS = list(METRICS)

# Extra scalars kept in the summary CSV but not given their own curve panel.
SCALAR_EXTRA = ['cave_entries_day', 'cave_entries_night', 'energy_at_death']

# ── Compositional behaviours ──────────────────────────────────────────────────
DEATH_CAUSES = {
    'starved':  '#D97706',   # ran out of energy
    'drained':  '#DC2626',   # killed by a predator
    'lifespan': '#2563EB',   # hit the max-lifetime cap
    'survived': '#059669',   # still alive at generation end
}
FOOD_TIERS = {
    'food_default':  ('default',  '#9CA3AF'),
    'food_low':      ('low',      '#60A5FA'),
    'food_medium':   ('medium',   '#F59E0B'),
    'food_prestige': ('prestige', '#DB2777'),
}
FOOD_COLS = list(FOOD_TIERS)

SCALAR_COLS = list(METRICS) + SCALAR_EXTRA

USE_COLS = (['generation', 'death_cause', 'cells_visited', 'predator_touches',
             'drained_ticks', 'cave_entries', 'cave_entries_day',
             'cave_entries_night', 'lifetime', 'energy_at_death'] + FOOD_COLS)

# Environment description used in plot titles (set in main from --env).
_ENV_DESC = ''
# Seed-subset tag appended to the aggregated output filenames (set in main).
_SUFFIX = ''


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def discover(root, want):
    """Find run folders under `root` (params.json + organisms.csv) whose
    params.json matches every filter in `want`, returning {seed, dir, org_csv}
    dicts. params.json is authoritative (folder names are only a hint).

    `want` keys:
        condition        required exact match
        mode             optional exact match ('standard' / 'pure_rl')
        epsilon_enabled  optional exact bool match
        learning_rate    optional float match (tol 1e-9)
        roaming          roaming_predator_count match (tol 1e-9)
        drain            predator_drain match (tol 1e-9); skipped if None"""
    runs = []
    for params_path in glob.glob(os.path.join(root, '*', 'params.json')):
        run_dir = os.path.dirname(params_path)
        org_csv = os.path.join(run_dir, 'organisms.csv')
        if not os.path.exists(org_csv):
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
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        runs.append({'seed': seed, 'dir': run_dir, 'org_csv': org_csv})
    return sorted(runs, key=lambda r: r['seed'])


def load_run(org_csv, max_gen):
    """Read one run's organisms.csv once; return a per-generation DataFrame
    (indexed by generation) holding, for each generation:
        - the mean of each scalar behaviour across that generation's organisms,
        - death_<cause>_frac : share of organisms with each death cause,
        - food_<tier>_frac   : share of consumed food in each tier.
    Missing columns degrade to NaN/0 rather than crashing on older logs."""
    available = pd.read_csv(org_csv, nrows=0).columns
    cols = [c for c in USE_COLS if c in available]
    df = pd.read_csv(org_csv, usecols=cols)
    for c in USE_COLS:
        if c not in df.columns:
            df[c] = 'unknown' if c == 'death_cause' else np.nan

    df = df[df['generation'] <= max_gen].copy()
    if not len(df):
        return pd.DataFrame()

    for c in SCALAR_COLS + FOOD_COLS:
        if c != 'food_total':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['food_total'] = df[FOOD_COLS].sum(axis=1)

    scal = df.groupby('generation')[SCALAR_COLS].mean()

    df['death_cause'] = df['death_cause'].astype(str)
    dc = (df.groupby('generation')['death_cause']
            .value_counts(normalize=True).unstack(fill_value=0.0))
    dc = dc.reindex(columns=list(DEATH_CAUSES), fill_value=0.0)
    dc.columns = [f'death_{c}_frac' for c in DEATH_CAUSES]

    fsum = df.groupby('generation')[FOOD_COLS].sum()
    ffrac = fsum.div(fsum.sum(axis=1).replace(0, np.nan), axis=0)
    ffrac.columns = [f'{c}_frac' for c in FOOD_COLS]

    return pd.concat([scal, dc, ffrac], axis=1).sort_index()


def aggregate_condition(run_curves):
    """Aggregate a condition's per-run curves across seeds. Returns (mean, std)
    DataFrames indexed by generation (mean/std across seeds per column)."""
    stacked = pd.concat({r['seed']: curve for r, curve in run_curves},
                        names=['seed', 'generation'])
    mean = stacked.groupby(level='generation').mean()
    std = stacked.groupby(level='generation').std()
    return mean, std


# ── Generation-level metrics (from generations.csv, not organisms.csv) ────────
GEN_METRICS = {
    'avg_fitness':          'Average fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
}
# MAD = avg_learned_weight_diff (within-life weight change; non-zero for learning
# AND pure RL, ~0 for evolution). genome_variance = GA population diversity.
DIAG_METRICS = {
    'avg_learned_weight_diff': 'Learned weight diff (MAD)',
    'genome_variance':         'Genome variance',
}
GEN_COLS = list(GEN_METRICS) + ['avg_lifetime'] + list(DIAG_METRICS)


def load_gen_metrics(gen_csv, max_gen):
    """Read generation-level metrics (GEN_COLS) from a run's generations.csv,
    which are already POPULATION-level. Returns a DataFrame indexed by generation
    (only the present columns), or None if unavailable. Deduplicates on generation
    (keep last) to be robust to a doubled/appended CSV."""
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


def plot_curves(cond_stats, metrics, smooth, out_dir):
    """One panel per scalar behaviour; within each, one line per condition (mean
    across seeds, shaded ±1 std) over generations."""
    print("  Plotting behaviour curves over generations ...")
    n = len(metrics)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, metrics):
        for label, (mean_df, std_df) in cond_stats.items():
            if m not in mean_df.columns:
                continue
            s = mean_df[m].dropna()
            sd = std_df[m].reindex(s.index).fillna(0)
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
                sd = sd.rolling(smooth, min_periods=1, center=True).mean()
            colour = COND_COLOURS.get(label, '#6B7280')
            ax.plot(s.index, s.values, color=colour, lw=2, label=label)
            ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                            color=colour, alpha=0.15)
        ax.set_ylabel(METRICS[m])
        ax.set_title(METRICS[m], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    axs[0].legend(fontsize=10, title='condition')
    axs[-1].set_xlabel('Generation')
    plt.suptitle('Behaviour over generations — three conditions '
                 f'({_ENV_DESC}; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_behaviour_curves{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _line_over_gens(ax, cond_gen, col, smooth):
    """Draw one line per condition (mean ± std across seeds) for column `col`.
    Returns True if anything was drawn."""
    drawn = False
    for label, (mean_df, std_df) in cond_gen.items():
        if col not in mean_df.columns:
            continue
        s = mean_df[col].dropna()
        if not len(s):
            continue
        sd = std_df[col].reindex(s.index).fillna(0)
        if smooth > 1:
            s = s.rolling(smooth, min_periods=1, center=True).mean()
            sd = sd.rolling(smooth, min_periods=1, center=True).mean()
        colour = COND_COLOURS.get(label, '#6B7280')
        ax.plot(s.index, s.values, color=colour, lw=2, label=label)
        ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                        color=colour, alpha=0.15)
        drawn = True
    return drawn


def plot_lifetime(cond_gen, smooth, out_dir):
    """POPULATION-average lifetime per generation (generations.csv avg_lifetime),
    one line per condition (mean ± std across seeds)."""
    print("  Plotting average lifetime per generation ...")
    fig, ax = plt.subplots(figsize=(12, 6))
    if not _line_over_gens(ax, cond_gen, 'avg_lifetime', smooth):
        print("  (no generations.csv avg_lifetime found — skipping lifetime plot)")
        plt.close()
        return
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average lifetime (ticks)')
    ax.set_title('Average lifetime per generation — three conditions\n'
                 f'(population average, {_ENV_DESC}; mean ± std across seeds)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    ax.legend(fontsize=10, title='condition')
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_lifetime{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gen_metrics(cond_gen, smooth, out_dir):
    """Fitness + population outcome metrics from generations.csv over generations:
    one panel per metric, one line per condition (mean ± std across seeds)."""
    print("  Plotting fitness + population metrics per generation ...")
    metrics = [m for m in GEN_METRICS
               if any(m in md.columns for md, _sd in cond_gen.values())]
    if not cond_gen or not metrics:
        print("  (no generations.csv fitness/population columns — skipping)")
        return
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()
    for ax, m in zip(axs, metrics):
        _line_over_gens(ax, cond_gen, m, smooth)
        ax.set_title(GEN_METRICS[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(GEN_METRICS[m])
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    for ax in axs[len(metrics):]:      # hide any unused panels
        ax.axis('off')
    for ax in axs[max(0, len(metrics) - 2):len(metrics)]:
        ax.set_xlabel('Generation')
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=10, title='condition')
    plt.suptitle('Fitness & population over generations — three conditions\n'
                 f'({_ENV_DESC}; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_fitness_population{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_learning_diagnostics(cond_gen, smooth, out_dir):
    """MAD + genome variance over generations (from generations.csv), one panel
    each, one line per condition (mean ± std across seeds). MAD (avg_learned_
    weight_diff) is the within-life learning signal — non-zero for BOTH RL-using
    conditions (learning and pure RL), ~0 for evolution (active === genome);
    genome_variance is the GA's population diversity (~0 / absent for pure RL,
    which has no GA). y-axes auto-scaled per panel (MAD is tiny vs variance)."""
    print("  Plotting learning diagnostics (MAD, genome variance) ...")
    metrics = [m for m in DIAG_METRICS
               if any(m in md.columns for md, _sd in cond_gen.values())]
    if not cond_gen or not metrics:
        print("  (no generations.csv MAD / genome_variance columns — skipping)")
        return
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 5))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, metrics):
        _line_over_gens(ax, cond_gen, m, smooth)
        ax.set_title(DIAG_METRICS[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(DIAG_METRICS[m])
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=10, title='condition')
    plt.suptitle('Learning diagnostics over generations — three conditions\n'
                 '(MAD = within-life weight change, for learning & pure RL; '
                 f'genome variance = GA diversity; {_ENV_DESC}; mean ± std across seeds)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_learning_diagnostics{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cave_day_night(cond_stats, smooth, out_dir):
    """Cave entries split by time of day over generations: two panels (daytime
    entries, night-time entries), one line per condition (mean ± std across
    seeds). Reads the cave_entries_day / cave_entries_night per-organism columns
    (population-averaged per generation, then over seeds). Shows whether
    organisms learn to shelter in caves at night (cheap: ×0 decay) rather than by
    day (costly: ×2 decay). Skipped if the day/night columns aren't logged."""
    print("  Plotting cave entries by time of day over generations ...")
    day_col, night_col = 'cave_entries_day', 'cave_entries_night'
    if not any(day_col in md.columns and night_col in md.columns
               for md, _sd in cond_stats.values()):
        print("  (no cave_entries_day / cave_entries_night columns — skipping)")
        return
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    axs = np.atleast_1d(axs)

    # Panels 1-2: absolute day / night entries per organism (mean ± std).
    for ax, (col, panel_title) in zip(axs[:2],
                                      [(day_col, 'Cave entries — daytime'),
                                       (night_col, 'Cave entries — night-time')]):
        for label, (mean_df, std_df) in cond_stats.items():
            if col not in mean_df.columns:
                continue
            s = mean_df[col].dropna()
            sd = std_df[col].reindex(s.index).fillna(0)
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
                sd = sd.rolling(smooth, min_periods=1, center=True).mean()
            colour = COND_COLOURS.get(label, '#6B7280')
            ax.plot(s.index, s.values, color=colour, lw=2, label=label)
            ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                            color=colour, alpha=0.15)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cave entries per organism')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    # Share a y-scale across the two absolute panels so the split is comparable.
    top = max((ax.get_ylim()[1] for ax in axs[:2]), default=1)
    for ax in axs[:2]:
        ax.set_ylim(0, top)

    # Panel 3: night SHARE of cave entries = night / (day + night). This isolates
    # the day-vs-night preference from the (large) variation in total cave use, so
    # the small-but-consistent lean the absolute panels hide becomes legible. It's
    # the ratio of the population-mean curves (no band — a band would need the
    # per-seed fractions, which aren't carried here); 0.5 = no preference.
    ax = axs[2]
    for label, (mean_df, _std_df) in cond_stats.items():
        if day_col not in mean_df.columns or night_col not in mean_df.columns:
            continue
        day = mean_df[day_col]
        night = mean_df[night_col]
        share = (night / (day + night)).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(share):
            continue
        if smooth > 1:
            share = share.rolling(smooth, min_periods=1, center=True).mean()
        ax.plot(share.index, share.values, color=COND_COLOURS.get(label, '#6B7280'),
                lw=2, label=label)
    ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
    ax.text(0.01, 0.5, ' no preference (0.5)', color='#6B7280', fontsize=8,
            va='bottom', ha='left', transform=ax.get_yaxis_transform())
    ax.set_title('Night share of cave entries', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('night / (day + night)')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=10, title='condition')
    plt.suptitle('Cave entries by time of day over generations — three conditions\n'
                 f'({_ENV_DESC}; population average; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_cave_day_night{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_composition(cond_means, kind, smooth, out_dir):
    """Per-condition stacked-area composition over generations. kind='death' uses
    the death_<cause>_frac columns; kind='food' uses food_<tier>_frac. One panel
    per condition (compositions can't be overlaid), shared y-axis."""
    if kind == 'death':
        spec = [(f'death_{c}_frac', c, col) for c, col in DEATH_CAUSES.items()]
        title = 'How organisms died over generations'
        ylabel = 'Fraction of organisms'
        fname = f'condition_behaviour_death{_SUFFIX}.png'
        legend_title = 'death cause'
    else:
        spec = [(f'{c}_frac', FOOD_TIERS[c][0], FOOD_TIERS[c][1]) for c in FOOD_COLS]
        title = 'Food-tier composition over generations'
        ylabel = 'Fraction of food eaten'
        fname = f'condition_behaviour_food{_SUFFIX}.png'
        legend_title = 'food tier'

    print(f"  Plotting {kind} composition over generations ...")
    labels = list(cond_means)
    fig, axs = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 5),
                            sharey=True)
    axs = np.atleast_1d(axs)
    for ax, label in zip(axs, labels):
        mean_df = cond_means[label]
        present = [(c, l, col) for c, l, col in spec if c in mean_df.columns]
        if not present:
            ax.set_title(f'{label}\n(no data)', color='gray'); ax.axis('off'); continue
        series = []
        for c, _l, _col in present:
            s = mean_df[c].fillna(0)
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
            series.append(s)
        gens = mean_df.index.values
        ax.stackplot(gens, *[s.values for s in series],
                     colors=[col for _c, _l, col in present],
                     labels=[l for _c, l, _col in present])
        ax.set_xlabel('Generation')
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.margins(x=0)
    axs[0].set_ylabel(ylabel)
    axs[-1].legend(fontsize=9, title=legend_title, loc='upper left',
                   bbox_to_anchor=(1.01, 1))
    plt.suptitle(f'{title} — {_ENV_DESC} (mean across seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_food_lines(cond_stats, smooth, out_dir):
    """Food-tier diet composition over generations as small-multiple LINE panels
    (one per tier) instead of a stacked area — far easier to read across hundreds
    of generations. Each panel shows the fraction of eaten food in that tier, one
    line per condition (mean ± std across seeds); y-axis auto-scaled per tier."""
    print("  Plotting food-tier composition (line panels) ...")
    tiers = [(f'{c}_frac', FOOD_TIERS[c][0], FOOD_TIERS[c][1]) for c in FOOD_COLS]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()
    for ax, (col, tier_label, _colour) in zip(axs, tiers):
        for label, (mean_df, std_df) in cond_stats.items():
            if col not in mean_df.columns:
                continue
            s = mean_df[col].dropna()
            sd = std_df[col].reindex(s.index).fillna(0)
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
                sd = sd.rolling(smooth, min_periods=1, center=True).mean()
            colour = COND_COLOURS.get(label, '#6B7280')
            ax.plot(s.index, s.values, color=colour, lw=2, label=label)
            ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                            color=colour, alpha=0.15)
        ax.set_title(f'{tier_label} food', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of food eaten')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axs[2].set_xlabel('Generation')
    axs[3].set_xlabel('Generation')
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=10, title='condition')
    plt.suptitle('Food-tier composition over generations — three conditions\n'
                 f'(share of diet in each tier, {_ENV_DESC}; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_behaviour_food{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_per_seed(seed, cond_data, smooth, out_dir):
    """Single-seed detail sheet: all conditions ON THE SAME MAP (seed), every
    behaviour + outcome metric in ONE png. Single seed => one run per condition,
    so these are the raw per-generation lines (no across-seed band). cond_data
    maps condition label -> a per-generation DataFrame (organisms behaviours
    joined with generations.csv metrics)."""
    labels = {**METRICS, **GEN_METRICS, **DIAG_METRICS}
    order = list(METRICS) + list(GEN_METRICS) + list(DIAG_METRICS)
    metrics = [m for m in order if any(m in df.columns for df in cond_data.values())]
    if not metrics:
        return
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.3 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, m in zip(axs, metrics):
        for label, df in cond_data.items():
            if m not in df.columns:
                continue
            s = pd.to_numeric(df[m], errors='coerce').dropna()
            if not len(s):
                continue
            if smooth > 1:
                s = s.rolling(smooth, min_periods=1, center=True).mean()
            ax.plot(s.index, s.values, color=COND_COLOURS.get(label, '#6B7280'),
                    lw=1.8, label=label)
        ax.set_title(labels[m], fontsize=11, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    for ax in axs[len(metrics):]:      # hide unused panels
        ax.axis('off')
    handles, lbls = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, lbls, fontsize=9, title='condition')
    plt.suptitle(f'Seed {seed} — three conditions (same map)\n'
                 f'behaviour & outcomes per generation ({_ENV_DESC}; single seed)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_seed{seed}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def write_summary(cond_stats, out_dir):
    """Tidy (condition, generation, metric, mean, std) across seeds for every
    scalar + composition fraction."""
    frames = []
    for label, (mean_df, std_df) in cond_stats.items():
        m_long = mean_df.reset_index().melt(id_vars='generation',
                                            var_name='metric', value_name='mean')
        s_long = std_df.reset_index().melt(id_vars='generation',
                                           var_name='metric', value_name='std')
        merged = m_long.merge(s_long, on=['generation', 'metric'])
        merged.insert(0, 'condition', label)
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f'condition_behaviour_summary{_SUFFIX}.csv')
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description='Behaviour over generations — evolution vs learning vs pure RL, '
                    'in the baseline or hard environment.')
    ap.add_argument('--env', choices=list(ENVIRONMENTS), default='hard',
                    help='Environment to analyse: baseline (no predators) or hard '
                         '(60 roaming predators, drain 5). Default: hard.')
    ap.add_argument('--conditions', nargs='+', default=COND_ORDER, choices=COND_ORDER,
                    help='Which of the three conditions to include (default: all three).')
    ap.add_argument('--evolution-root', default='logs/evolution/standard/auto-run',
                    help='Folder holding the evolution-condition run dirs.')
    ap.add_argument('--learning-root', default='logs/learning/standard/auto-run',
                    help='Folder holding the learning-condition run dirs.')
    ap.add_argument('--pure-rl-root', default='logs/learning/pure_rl/auto-run',
                    help='Folder holding the pure-RL-condition run dirs.')
    ap.add_argument('--learning-lr', type=float, default=None,
                    help='learning_rate that selects the learning + pure-RL runs. '
                         'Default: the environment optimum (0.01 baseline / 0.02 hard).')
    ap.add_argument('--predators', type=float, default=60,
                    help='roaming_predator_count for the HARD environment (default 60; '
                         'ignored for baseline, which requires roaming==0).')
    ap.add_argument('--drain', type=float, default=5,
                    help='predator_drain for the HARD environment (default 5; '
                         'ignored for baseline).')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='organisms.csv scalar behaviours to panel (default: all six).')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (e.g. 700 for a fair '
                         'head-to-head when the conditions differ in length).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curves/areas (1 = none).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered. '
                         'Aggregated PNGs/CSV get a _seed<..> suffix.')
    ap.add_argument('-o', '--out', default=None,
                    help='Directory for the PNGs + summary CSV '
                         '(default: output/condition_comparison_<env>).')
    args = ap.parse_args()

    env = ENVIRONMENTS[args.env]
    lr = args.learning_lr if args.learning_lr is not None else env['lr']
    # For the hard env, honour --predators/--drain overrides; baseline is fixed at
    # roaming==0 with no drain filter.
    if args.env == 'hard':
        roaming, drain = args.predators, args.drain
    else:
        roaming, drain = env['roaming'], env['drain']

    out_dir = args.out or f'output/condition_comparison_{args.env}'
    os.makedirs(out_dir, exist_ok=True)

    global _ENV_DESC, _SUFFIX
    _ENV_DESC = env['desc'] if args.env == 'baseline' else \
        f'{int(roaming)} roaming predators, drain {int(drain)}'

    seed_filter = set(args.seeds) if args.seeds else None
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''

    # Build the requested condition families with resolved display labels + colours.
    base_want = {'roaming': roaming, 'drain': drain}
    lr_tag = f'lr {lr:g}'
    family_specs = {
        'evolution': ('evolution', args.evolution_root,
                      {**base_want, 'condition': 'evolution'}),
        'learning':  (f'learning ({lr_tag})', args.learning_root,
                      {**base_want, 'condition': 'learning', 'mode': 'standard',
                       'epsilon_enabled': False, 'learning_rate': lr}),
        'pure_rl':   (f'pure RL ({lr_tag})', args.pure_rl_root,
                      {**base_want, 'condition': 'learning', 'mode': 'pure_rl',
                       'epsilon_enabled': False, 'learning_rate': lr}),
    }
    families = []
    for key in args.conditions:
        label, root, want = family_specs[key]
        COND_COLOURS[label] = COND_COLOURS_KEY[key]
        families.append((label, root, want))

    cond_stats = {}   # label -> (mean_df, std_df)  organisms.csv behaviours
    cond_means = {}   # label -> mean_df (for composition plots)
    cond_gen = {}     # label -> (mean_df, std_df)  generations.csv outcome metrics
    per_seed = {}     # seed -> {label -> combined per-generation DataFrame}
    for label, root, want in families:
        print(f"\nDiscovering {label} runs under {root} ...")
        runs = discover(root, want)
        if seed_filter is not None:
            runs = [r for r in runs if r['seed'] in seed_filter]
            missing = seed_filter - {r['seed'] for r in runs}
            if missing:
                print(f"  note: requested seeds not found for {label}: {sorted(missing)}")
        if not runs:
            print(f"  WARNING: no matching runs for {label} "
                  f"(env={args.env}, roaming={roaming}"
                  f"{'' if drain is None else f', drain={drain}'}"
                  f"{'' if seed_filter is None else f', seeds={sorted(seed_filter)}'}). Skipping.")
            continue
        print(f"  Found {len(runs)} runs: seeds {[r['seed'] for r in runs]}. Reading organisms.csv ...")
        run_curves, gen_runs = [], []
        for r in runs:
            curve = load_run(r['org_csv'], args.max_gen)
            if not len(curve):
                print(f"    skip seed {r['seed']} — no rows"); continue
            run_curves.append((r, curve))
            gm = load_gen_metrics(os.path.join(r['dir'], 'generations.csv'), args.max_gen)
            if gm is not None and len(gm):
                gen_runs.append((r, gm))
            combined = curve if (gm is None or not len(gm)) else curve.join(gm, how='outer')
            per_seed.setdefault(r['seed'], {})[label] = combined
            print(f"    seed {r['seed']:<5d} {int(curve.index.max())} gens")
        if not run_curves:
            continue
        mean_df, std_df = aggregate_condition(run_curves)
        if gen_runs:
            gen_mean, gen_std = aggregate_condition(gen_runs)
            cond_gen[label] = (gen_mean, gen_std)
            for c in gen_mean.columns:
                out_name = 'avg_lifetime_pop' if c == 'avg_lifetime' else c
                mean_df[out_name] = gen_mean[c].reindex(mean_df.index)
                std_df[out_name] = gen_std[c].reindex(std_df.index)
        cond_stats[label] = (mean_df, std_df)
        cond_means[label] = mean_df

    if not cond_stats:
        raise SystemExit("No runs discovered for any condition — check the "
                         "--env / --*-root / --learning-lr / --predators / --drain filters.")

    print("\nPlotting ...")
    plot_curves(cond_stats, args.metrics, args.smooth, out_dir)
    plot_gen_metrics(cond_gen, args.smooth, out_dir)
    plot_learning_diagnostics(cond_gen, args.smooth, out_dir)
    plot_lifetime(cond_gen, args.smooth, out_dir)
    plot_cave_day_night(cond_stats, args.smooth, out_dir)
    plot_composition(cond_means, 'death', args.smooth, out_dir)
    plot_food_lines(cond_stats, args.smooth, out_dir)
    for seed in sorted(per_seed):
        plot_per_seed(seed, per_seed[seed], args.smooth, out_dir)
    write_summary(cond_stats, out_dir)

    print(f"\nDone. Outputs in {os.path.abspath(out_dir)}\n")


if __name__ == '__main__':
    main()
