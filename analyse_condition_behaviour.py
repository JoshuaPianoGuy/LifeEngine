"""
analyse_condition_behaviour.py
======================================================================
Behaviour-over-time comparison of the LEARNING and EVOLUTION conditions in the
roaming-predator environment (60 predators, drain 5).

Where analyse_lr_sweep_behaviour.py compares behaviour ACROSS learning rates
(one condition, many LRs), this script compares the two experimental CONDITIONS
against each other and tracks how their behaviour develops PER GENERATION:

    learning   no-epsilon, learning_rate 0.02  (GA + REINFORCE)
    evolution  GA only, no in-lifetime learning

Both families live in different log trees but share the same organisms.csv
schema, so they are discovered separately (by params.json) and tagged with a
condition label:

    learning   --learning-root  (default logs/learning/standard/auto-run),
               filtered to condition=learning, epsilon disabled,
               learning_rate == --learning-lr (0.02)
    evolution  --evolution-root (default logs/evolution/standard/auto-run),
               filtered to condition=evolution
Both are additionally filtered to the roaming-predator environment
(roaming_predator_count == --predators, predator_drain == --drain), so only the
matched 60-predator/drain-5 runs are compared. Each condition is aggregated
across its seeds (mean ± std) so the condition is the only factor.

organisms.csv logs EVERY organism that lived each generation (its row count per
generation matches total_agents in generations.csv), so each per-generation
value here is the true POPULATION average across all organisms alive that
generation — not just the fittest. Note these are per-organism, per-lifetime
quantities (e.g. predator_touches is the encounters ONE organism had over its
life), averaged over the population, then over seeds; the shaded band is the
spread across seeds, not within the population. The two conditions can have
different run lengths (e.g. learning 500 gens, evolution 1000); each line is
drawn over its own available range (pass --max-gen 500 for a strict head-to-head).

Behaviours read from organisms.csv:
    cells_visited     -- distinct cells visited (exploration / distance proxy)
    predator_touches  -- distinct predator-contact episodes (encounters)
    drained_ticks     -- ticks spent being drained by a predator (contact time)
    cave_entries      -- cave entries (+ day/night split in the summary CSV)
    food_total        -- food eaten this lifetime (sum of all tiers)
    lifetime          -- ticks survived
    death_cause       -- how they died: starved / drained / survived  (composition)
    food_<tier>       -- default/low/medium/prestige  (composition)

Outputs (into --out):
    condition_behaviour_curves.png       -- one panel per scalar behaviour, one
                                            line per condition (mean ± std across
                                            seeds) over generations.
    condition_lifetime.png               -- dedicated plot of POPULATION-average
                                            lifetime per generation (generations.csv
                                            avg_lifetime), one line per condition.
    condition_fitness_population.png     -- generations.csv outcome metrics over
                                            generations (avg_fitness, top-20%
                                            fitness, peak_population, total_agents),
                                            one line per condition. Less
                                            behavioural, useful head-to-head context.
    condition_learning_diagnostics.png   -- MAD (avg_learned_weight_diff, the
                                            within-life learning signal; ~0 for
                                            evolution) and genome_variance (GA
                                            diversity) over generations, one line
                                            per condition.
    condition_behaviour_death.png        -- per-condition stacked-area death-cause
                                            composition over generations.
    condition_behaviour_food.png         -- food-tier diet composition over
                                            generations as one LINE panel per tier
                                            (default/low/medium/prestige), one line
                                            per condition — clearer than a stacked
                                            area across hundreds of generations.
    condition_seed<seed>.png             -- per-seed detail sheet (one per seed):
                                            learning vs evolution on the SAME map,
                                            every behaviour + outcome metric in one
                                            png (raw per-generation lines, single
                                            seed so no across-seed band).
    condition_behaviour_summary.csv      -- tidy (condition, generation, metric,
                                            mean, std) across seeds for every
                                            scalar + composition fraction.

Usage
-----
    python analyse_condition_behaviour.py
    python analyse_condition_behaviour.py --max-gen 500        # fair head-to-head
    python analyse_condition_behaviour.py --metrics cells_visited predator_touches
    python analyse_condition_behaviour.py --seeds 1 42         # subset of seeds
                                                               #  -> *_seed1_42.png / .csv
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

# One colour per condition — lines/areas are coloured by condition, not metric.
LEARNING_LABEL  = 'learning (no-eps, lr 0.02)'
EVOLUTION_LABEL = 'evolution'
COND_COLOURS = {LEARNING_LABEL: '#2563EB', EVOLUTION_LABEL: '#DC2626'}

# ── Scalar per-organism behaviours (curve panels) ─────────────────────────────
# Column -> axis/panel label. Averaged over the organisms in each generation,
# per run, then over seeds.
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
# death_cause categories (fixed order + colour). Only starved/drained/survived
# appear in the current logs; lifespan is kept for forward-compatibility (0 if
# absent).
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

# Scalars computed per generation (panels + extras). food_total is derived below.
SCALAR_COLS = list(METRICS) + SCALAR_EXTRA

# All organisms.csv columns we read (kept minimal — organisms.csv is large).
USE_COLS = (['generation', 'death_cause', 'cells_visited', 'predator_touches',
             'drained_ticks', 'cave_entries', 'cave_entries_day',
             'cave_entries_night', 'lifetime', 'energy_at_death'] + FOOD_COLS)


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


# Seed-subset tag appended to the aggregated output filenames (set in main from
# --seeds; '' when all seeds are used). Keeps a restricted-seed run's PNGs/CSV
# from overwriting the all-seeds ones. The per-seed sheets already carry the seed
# in their name, so they are not suffixed.
_SUFFIX = ''


def discover(root, want):
    """Find run folders under `root` (params.json + organisms.csv) whose
    params.json matches every filter in `want`, returning {seed, org_csv} dicts.

    params.json is authoritative (folder names are only a hint). `want` keys:
        condition       required exact match
        epsilon_enabled optional exact bool match
        learning_rate   optional float match (tol 1e-9)
        roaming/drain    predator-environment match (tol 1e-9)"""
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
        if 'epsilon_enabled' in want and bool(p.get('epsilon_enabled', True)) != want['epsilon_enabled']:
            continue
        if 'learning_rate' in want and abs(_num(p, 'learning_rate') - want['learning_rate']) > 1e-9:
            continue
        if abs(_num(p, 'roaming_predator_count') - want['roaming']) > 1e-9:
            continue
        if abs(_num(p, 'predator_drain') - want['drain']) > 1e-9:
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

    # Scalar per-generation means.
    scal = df.groupby('generation')[SCALAR_COLS].mean()

    # Death-cause fractions per generation.
    df['death_cause'] = df['death_cause'].astype(str)
    dc = (df.groupby('generation')['death_cause']
            .value_counts(normalize=True).unstack(fill_value=0.0))
    dc = dc.reindex(columns=list(DEATH_CAUSES), fill_value=0.0)
    dc.columns = [f'death_{c}_frac' for c in DEATH_CAUSES]

    # Food-tier fractions per generation (share of total food eaten that gen).
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
# Outcome metrics — less behavioural, but useful head-to-head context. These are
# already population-level in generations.csv, so no per-organism averaging.
GEN_METRICS = {
    'avg_fitness':          'Average fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
}
# Learning diagnostics (generations.csv). MAD = mean |active - genome| weight,
# i.e. how much is learned WITHIN a life — non-zero only where RL runs (the
# learning condition; ~0 for evolution). genome_variance = GA population
# diversity (mean per-weight variance across genomes).
DIAG_METRICS = {
    'avg_learned_weight_diff': 'Learned weight diff (MAD)',
    'genome_variance':         'Genome variance',
}
# avg_lifetime + the diagnostics get their own plots but are read here too.
GEN_COLS = list(GEN_METRICS) + ['avg_lifetime'] + list(DIAG_METRICS)


def load_gen_metrics(gen_csv, max_gen):
    """Read the generation-level metrics (GEN_COLS) from a run's generations.csv,
    which are already POPULATION-level (e.g. avg_lifetime is averaged over the
    whole population, unlike the organisms.csv 'lifetime'). Returns a DataFrame
    indexed by generation (only the columns present), or None if unavailable.
    Deduplicates on generation (keep last) to be robust to a doubled/appended CSV."""
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
    plt.suptitle('Behaviour over generations — learning vs evolution '
                 '(60 predators, drain 5; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_behaviour_curves{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _line_over_gens(ax, cond_gen, col, smooth):
    """Draw one line per condition (mean ± std across seeds) for column `col` of
    the per-condition (mean_df, std_df) gen-stats. Returns True if anything drawn."""
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
    """Dedicated plot: POPULATION-average lifetime per generation (from
    generations.csv avg_lifetime), one line per condition (mean ± std across
    seeds) over generations."""
    print("  Plotting average lifetime per generation ...")
    fig, ax = plt.subplots(figsize=(12, 6))
    if not _line_over_gens(ax, cond_gen, 'avg_lifetime', smooth):
        print("  (no generations.csv avg_lifetime found — skipping lifetime plot)")
        plt.close()
        return
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average lifetime (ticks)')
    ax.set_title('Average lifetime per generation — learning vs evolution\n'
                 '(population average, 60 predators, drain 5; mean ± std across seeds)',
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
    one panel per metric (avg_fitness, top20percent_fitness, peak_population,
    total_agents), one line per condition (mean ± std across seeds). Less
    behavioural than the organisms.csv panels, but useful head-to-head context."""
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
    plt.suptitle('Fitness & population over generations — learning vs evolution\n'
                 '(60 predators drain 5; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_fitness_population{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_learning_diagnostics(cond_gen, smooth, out_dir):
    """MAD + genome variance over generations (from generations.csv), one panel
    each, one line per condition (mean ± std across seeds). MAD (avg_learned_
    weight_diff) is the within-life learning signal — non-zero for the learning
    condition, ~0 for evolution (active === genome); genome_variance is the GA's
    population diversity. y-axes auto-scaled per panel (MAD is tiny vs variance)."""
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
    plt.suptitle('Learning diagnostics over generations — learning vs evolution\n'
                 '(MAD = within-life weight change; genome variance = GA diversity; '
                 'mean ± std across seeds)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_learning_diagnostics{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_composition(cond_means, kind, smooth, out_dir):
    """Per-condition stacked-area composition over generations. kind='death' uses
    the death_<cause>_frac columns; kind='food' uses the food_<tier>_frac columns.
    One panel per condition (compositions can't be overlaid), shared y-axis."""
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
    fig, axs = plt.subplots(1, len(labels), figsize=(7.5 * len(labels), 5),
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
    plt.suptitle(f'{title} — 60 predators, drain 5 (mean across seeds)',
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
    line per condition (mean ± std across seeds); the y-axis is auto-scaled per
    tier so small tiers (e.g. prestige) stay legible."""
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
    # Bottom-row x-labels (top-row ticks are hidden by sharex).
    axs[2].set_xlabel('Generation')
    axs[3].set_xlabel('Generation')
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, fontsize=10, title='condition')
    plt.suptitle('Food-tier composition over generations — learning vs evolution\n'
                 '(share of diet in each tier, 60 predators drain 5; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'condition_behaviour_food{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_per_seed(seed, cond_data, smooth, out_dir):
    """Single-seed detail sheet: learning vs evolution ON THE SAME MAP (seed),
    every behaviour + outcome metric in ONE png. Because it is a single seed there
    is one run per condition, so these are the raw per-generation lines (no
    across-seed averaging / band). cond_data maps condition label -> a per-
    generation DataFrame (organisms behaviours joined with generations.csv metrics)."""
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
    plt.suptitle(f'Seed {seed} — learning vs evolution (same map)\n'
                 f'behaviour & outcomes per generation (60 predators drain 5; single seed)',
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
    ap = argparse.ArgumentParser(description='Behaviour over generations — learning vs evolution.')
    ap.add_argument('--learning-root', default='logs/learning/standard/auto-run',
                    help='Folder holding the learning-condition run dirs.')
    ap.add_argument('--evolution-root', default='logs/evolution/standard/auto-run',
                    help='Folder holding the evolution-condition run dirs.')
    ap.add_argument('--learning-lr', type=float, default=0.02,
                    help='learning_rate that selects the learning runs (default 0.02).')
    ap.add_argument('--predators', type=float, default=60,
                    help='roaming_predator_count both conditions must match (default 60).')
    ap.add_argument('--drain', type=float, default=5,
                    help='predator_drain both conditions must match (default 5).')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=DEFAULT_METRICS,
                    help='organisms.csv scalar behaviours to panel (default: all six).')
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (e.g. 500 for a fair '
                         'head-to-head when the conditions differ in length).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curves/areas (1 = none).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds (e.g. --seeds 1 42 to compare '
                         'with less between-seed variation). Default: all discovered '
                         'seeds. Aggregated PNGs/CSV get a _seed<..> suffix (e.g. '
                         '_seed1_42); per-seed sheets are limited to the chosen seeds.')
    ap.add_argument('-o', '--out', default='output/condition_behaviour',
                    help='Directory for the PNGs + summary CSV (default: output/condition_behaviour).')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Seed filter + filename suffix. When --seeds is given, aggregated outputs are
    # tagged (e.g. condition_behaviour_curves_seed1_42.png) so they don't clobber
    # the all-seeds versions.
    seed_filter = set(args.seeds) if args.seeds else None
    global _SUFFIX
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''

    families = [
        (LEARNING_LABEL, args.learning_root,
         {'condition': 'learning', 'epsilon_enabled': False,
          'learning_rate': args.learning_lr, 'roaming': args.predators, 'drain': args.drain}),
        (EVOLUTION_LABEL, args.evolution_root,
         {'condition': 'evolution', 'roaming': args.predators, 'drain': args.drain}),
    ]

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
                  f"(predators={args.predators}, drain={args.drain}"
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
            # Per-seed detail: organisms behaviours + generations.csv metrics on
            # one generation index, kept per (seed, condition) for the seed plots.
            combined = curve if (gm is None or not len(gm)) else curve.join(gm, how='outer')
            per_seed.setdefault(r['seed'], {})[label] = combined
            print(f"    seed {r['seed']:<5d} {int(curve.index.max())} gens")
        if not run_curves:
            continue
        mean_df, std_df = aggregate_condition(run_curves)
        # Generation-level outcome metrics (generations.csv) — dedicated plots +
        # folded into the summary CSV. avg_lifetime is renamed avg_lifetime_pop
        # there to flag it as the population average (vs the organisms 'lifetime').
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
        raise SystemExit("No runs discovered for either condition — check the "
                         "--learning-root / --evolution-root / --predators / --drain filters.")

    print("\nPlotting ...")
    plot_curves(cond_stats, args.metrics, args.smooth, args.out)
    plot_gen_metrics(cond_gen, args.smooth, args.out)
    plot_learning_diagnostics(cond_gen, args.smooth, args.out)
    plot_lifetime(cond_gen, args.smooth, args.out)
    plot_composition(cond_means, 'death', args.smooth, args.out)
    plot_food_lines(cond_stats, args.smooth, args.out)
    for seed in sorted(per_seed):
        plot_per_seed(seed, per_seed[seed], args.smooth, args.out)
    write_summary(cond_stats, args.out)

    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
