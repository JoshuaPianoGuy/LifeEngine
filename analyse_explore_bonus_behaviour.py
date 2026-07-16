"""
analyse_explore_bonus_behaviour.py
======================================================================
Effect of the REINFORCE exploration-bonus reward on fitness, population AND
behaviour (cave usage, diet), from the explore-bonus sweep runs produced by
run_explore_bonus_sweep_array.slurm.

That job sweeps the per-tick exploration bonus — the intrinsic reward added the
first time an organism steps onto a new grid cell — across 6 values, each
replicated over the 5 map seeds, for the learning condition (GA + on-policy
REINFORCE, no epsilon):

    explore_bonus : 0, 0.02, 0.04, 0.08, 0.12, 0.15   (0.15 = old default)
    seeds         : 1, 42, 999, 123, 456
    6 bonuses x 5 seeds = 30 runs per environment.

The point of the sweep is to see whether shaping the reward this way changes
CAVE USAGE and FITNESS: a bigger novelty bonus should push organisms to roam and
explore more (visit more cells, maybe shelter in caves less / differently),
which may help or hurt fitness.

This is the explore-bonus analogue of analyse_conditions_comparison.py: same
organisms.csv / generations.csv schema and the same behaviour + outcome plots,
but the SERIES is the explore_bonus value (within the single learning
condition), not the condition. Each bonus value is aggregated across its seeds
(mean ± std), so the bonus is the only factor and seed/map noise averages out.
Because the sweep is an ordered scalar, the series use a sequential colour ramp
(dark = 0, bright = 0.15) rather than categorical colours.

Runs in BOTH environments by default (--env both), each into its own output dir
so baseline and hard never mix:
    baseline   predator-free (roaming==0);           RL lr 0.01
    hard       60 roaming predators, drain 5;         RL lr 0.02

Behaviours read from organisms.csv (population average per generation, then over
seeds) and outcomes from generations.csv — identical columns to the condition
comparison. See that script's header for the full column list.

Outputs (into --out / output/explore_bonus_<env>):
    eb_fitness_population.png   -- avg_fitness, top-20% fitness, peak_population,
                                   total_agents over generations; one line/bonus.
    eb_behaviour_curves.png     -- per-organism scalar behaviours over generations
                                   (cells explored, cave entries, food eaten,
                                   lifetime, …); one line per bonus.
    eb_cave_day_night.png       -- cave entries split day vs night + the night
                                   share; one line per bonus. THE cave-usage view.
    eb_behaviour_food.png       -- food-tier diet composition (line per tier
                                   panel); one line per bonus.
    eb_vs_bonus.png             -- final-window (last --final-frac of generations)
                                   value of the key fitness + cave metrics plotted
                                   AGAINST explore_bonus (x-axis), mean ± std
                                   across seeds. The at-a-glance "did the reward
                                   move the needle?" summary.
    eb_behaviour_summary.csv    -- tidy (bonus, generation, metric, mean, std).

Usage
-----
    python analyse_explore_bonus_behaviour.py                 # both envs
    python analyse_explore_bonus_behaviour.py --env baseline
    python analyse_explore_bonus_behaviour.py --env hard --seeds 1 42
    python analyse_explore_bonus_behaviour.py --env both --max-gen 1000
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
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
}
GEN_COLS = list(GEN_METRICS)

# Metrics for the final-window "vs bonus" summary plot: (source column, title,
# where the column lives — 'scalar' behaviours vs 'gen' outcomes vs 'night_share').
VS_BONUS = [
    ('avg_fitness',          'Mean average fitness',       'gen'),
    ('top20percent_fitness', 'Top-20% fitness',            'gen'),
    ('peak_population',      'Peak population',             'gen'),
    ('cave_entries',         'Cave entries / organism',     'scalar'),
    ('night_share',          'Night share of cave entries', 'night_share'),
    ('food_prestige_frac',   'Prestige-food share of diet', 'scalar'),
]

_EB_RE = re.compile(r'_eb([0-9.]+)_')

# Set per-env in run_env(); used by the plot titles / filenames.
_ENV_DESC = ''
_SUFFIX = ''
_EB_COLOURS = {}   # bonus value -> colour (sequential ramp), filled per run.


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def _eb_label(bonus):
    """Consistent display label + summary key for a bonus value."""
    return f'eb {bonus:g}'


def discover(root, want):
    """Find explore-bonus sweep run folders under `root` (params.json +
    organisms.csv) matching every filter in `want`, returning
    {bonus, seed, dir, org_csv} dicts. The bonus is the sweep axis and is read
    from the folder-name _eb<val>_ tag.

    Requiring that _eb tag is load-bearing: the same param filter (learning,
    lr 0.02, roaming 60, drain 5, epsilon off) ALSO matches the condition-
    consistency replicate runs (…_seedS_repN), which carry the DEFAULT
    explore_bonus=0.15 in params.json — so keying the bonus off params.json would
    dump all 25 of them into the 0.15 sweep point. Only folders bearing the sweep's
    own _eb tag are its runs, so that tag is both the gate and the bonus source.

    `want` keys: condition (req), mode, epsilon_enabled, learning_rate, roaming,
    drain (skipped if None) — same semantics as analyse_conditions_comparison.py."""
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
        m = _EB_RE.search(os.path.basename(run_dir))
        if not m:      # no _eb tag => not a sweep run (e.g. a consistency replicate)
            continue
        bonus = float(m.group(1))
        try:
            seed = int(p['map_seed'])
        except (KeyError, ValueError, TypeError):
            continue
        runs.append({'bonus': bonus, 'seed': seed, 'dir': run_dir, 'org_csv': org_csv})
    return sorted(runs, key=lambda r: (r['bonus'], r['seed']))


def load_run(org_csv, max_gen):
    """Read one run's organisms.csv; return a per-generation DataFrame (indexed by
    generation) with the mean of each scalar behaviour across that generation's
    organisms plus food_<tier>_frac diet shares. Missing columns degrade to
    NaN/0 rather than crashing on older logs."""
    available = pd.read_csv(org_csv, nrows=0).columns
    cols = [c for c in USE_COLS if c in available]
    df = pd.read_csv(org_csv, usecols=cols)
    for c in USE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[df['generation'] <= max_gen].copy()
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

    return pd.concat([scal, ffrac], axis=1).sort_index()


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


def aggregate(run_curves):
    """Aggregate a bonus value's per-seed curves across seeds -> (mean, std)
    DataFrames indexed by generation."""
    stacked = pd.concat({seed: curve for seed, curve in run_curves},
                        names=['seed', 'generation'])
    return (stacked.groupby(level='generation').mean(),
            stacked.groupby(level='generation').std())


def _draw(ax, eb_stats, col, smooth):
    """One line per bonus (mean ± std band) for column `col`. Returns True if
    anything was drawn. eb_stats: bonus -> (mean_df, std_df), ascending bonus."""
    drawn = False
    for bonus, (mean_df, std_df) in eb_stats.items():
        if col not in mean_df.columns:
            continue
        s = mean_df[col].dropna()
        if not len(s):
            continue
        sd = std_df[col].reindex(s.index).fillna(0)
        if smooth > 1:
            s = s.rolling(smooth, min_periods=1, center=True).mean()
            sd = sd.rolling(smooth, min_periods=1, center=True).mean()
        colour = _EB_COLOURS[bonus]
        ax.plot(s.index, s.values, color=colour, lw=2, label=_eb_label(bonus))
        ax.fill_between(s.index, s.values - sd.values, s.values + sd.values,
                        color=colour, alpha=0.12)
        drawn = True
    return drawn


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, title='explore bonus', ncol=1)


def plot_gen_metrics(eb_stats, smooth, out_dir):
    """Fitness + population outcome metrics over generations; one line per bonus."""
    print("  Plotting fitness + population vs explore bonus ...")
    metrics = [m for m in GEN_METRICS
               if any(m in md.columns for md, _sd in eb_stats.values())]
    if not metrics:
        print("  (no generations.csv fitness/population columns — skipping)")
        return
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()
    for ax, m in zip(axs, metrics):
        _draw(ax, eb_stats, m, smooth)
        ax.set_title(GEN_METRICS[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(GEN_METRICS[m])
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    for ax in axs[max(0, len(metrics) - 2):len(metrics)]:
        ax.set_xlabel('Generation')
    _legend(axs[0])
    plt.suptitle('Fitness & population over generations — exploration-bonus sweep\n'
                 f'({_ENV_DESC}; learning condition; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eb_fitness_population{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_behaviour_curves(eb_stats, smooth, out_dir):
    """Per-organism scalar behaviours over generations; one line per bonus."""
    print("  Plotting behaviour curves vs explore bonus ...")
    metrics = [m for m in METRICS
               if any(m in md.columns for md, _sd in eb_stats.values())]
    if not metrics:
        return
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, m in zip(axs, metrics):
        _draw(ax, eb_stats, m, smooth)
        ax.set_title(METRICS[m], fontsize=12, fontweight='bold')
        ax.set_ylabel(METRICS[m])
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    for ax in axs[len(metrics):]:
        ax.axis('off')
    for ax in axs[max(0, len(metrics) - ncols):len(metrics)]:
        ax.set_xlabel('Generation')
    _legend(axs[0])
    plt.suptitle('Behaviour over generations — exploration-bonus sweep\n'
                 f'({_ENV_DESC}; population average; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eb_behaviour_curves{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cave_day_night(eb_stats, smooth, out_dir):
    """Cave entries split day vs night + the night share; one line per bonus.
    THE view for 'does a bigger novelty bonus change cave usage?'. Reads the
    cave_entries_day / cave_entries_night per-organism columns."""
    print("  Plotting cave entries by time of day vs explore bonus ...")
    day_col, night_col = 'cave_entries_day', 'cave_entries_night'
    if not any(day_col in md.columns and night_col in md.columns
               for md, _sd in eb_stats.values()):
        print("  (no cave_entries_day / cave_entries_night columns — skipping)")
        return
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    axs = np.atleast_1d(axs)

    for ax, (col, panel_title) in zip(axs[:2],
                                      [(day_col, 'Cave entries — daytime'),
                                       (night_col, 'Cave entries — night-time')]):
        _draw(ax, eb_stats, col, smooth)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cave entries per organism')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    top = max((ax.get_ylim()[1] for ax in axs[:2]), default=1)
    for ax in axs[:2]:
        ax.set_ylim(0, top)

    # Panel 3: night share = night / (day + night); isolates the day-vs-night
    # preference from the (large) variation in total cave use. Ratio of the
    # seed-mean curves (no band); 0.5 = no preference.
    ax = axs[2]
    for bonus, (mean_df, _std_df) in eb_stats.items():
        if day_col not in mean_df.columns or night_col not in mean_df.columns:
            continue
        share = (mean_df[night_col] / (mean_df[day_col] + mean_df[night_col]))
        share = share.replace([np.inf, -np.inf], np.nan).dropna()
        if not len(share):
            continue
        if smooth > 1:
            share = share.rolling(smooth, min_periods=1, center=True).mean()
        ax.plot(share.index, share.values, color=_EB_COLOURS[bonus], lw=2,
                label=_eb_label(bonus))
    ax.axhline(0.5, color='#6B7280', ls='--', lw=1)
    ax.text(0.01, 0.5, ' no preference (0.5)', color='#6B7280', fontsize=8,
            va='bottom', ha='left', transform=ax.get_yaxis_transform())
    ax.set_title('Night share of cave entries', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('night / (day + night)')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    _legend(axs[0])
    plt.suptitle('Cave entries by time of day — exploration-bonus sweep\n'
                 f'({_ENV_DESC}; population average; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eb_cave_day_night{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_food_lines(eb_stats, smooth, out_dir):
    """Food-tier diet composition over generations as line panels (one per tier);
    one line per bonus. Shows whether a bigger bonus shifts diet up-tier."""
    print("  Plotting food-tier composition vs explore bonus ...")
    tiers = [(f'{c}_frac', FOOD_TIERS[c][0]) for c in FOOD_COLS]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()
    for ax, (col, tier_label) in zip(axs, tiers):
        _draw(ax, eb_stats, col, smooth)
        ax.set_title(f'{tier_label} food', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of food eaten')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axs[2].set_xlabel('Generation')
    axs[3].set_xlabel('Generation')
    _legend(axs[0])
    plt.suptitle('Food-tier composition over generations — exploration-bonus sweep\n'
                 f'(share of diet in each tier, {_ENV_DESC}; mean ± std across seeds)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eb_behaviour_food{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_vs_bonus(per_seed, bonuses, final_frac, out_dir):
    """Final-window value of the key metrics plotted AGAINST explore_bonus (x),
    mean ± std across seeds. The at-a-glance 'did the reward change fitness /
    cave usage?' summary. per_seed: bonus -> list of (seed, combined per-gen DF)
    where the combined DF holds scalar behaviours + food fracs + gen outcomes."""
    print("  Plotting final-window metrics vs explore bonus ...")

    def final_value(df, col):
        """Mean of `col` over the final `final_frac` of that run's generations."""
        if col == 'night_share':
            if 'cave_entries_day' not in df or 'cave_entries_night' not in df:
                return np.nan
            s = (df['cave_entries_night'] /
                 (df['cave_entries_day'] + df['cave_entries_night']))
        elif col not in df.columns:
            return np.nan
        else:
            s = df[col]
        s = pd.to_numeric(s, errors='coerce').dropna()
        if not len(s):
            return np.nan
        cutoff = s.index.max() - final_frac * (s.index.max() - s.index.min())
        window = s[s.index >= cutoff]
        return window.mean() if len(window) else s.iloc[-1]

    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    axs = axs.ravel()
    for ax, (col, title, _src) in zip(axs, VS_BONUS):
        xs, means, stds = [], [], []
        for b in bonuses:
            vals = [final_value(df, col) for _seed, df in per_seed.get(b, [])]
            vals = [v for v in vals if v == v]   # drop NaN
            if not vals:
                continue
            xs.append(b)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        if not xs:
            ax.set_title(f'{title}\n(no data)', color='gray'); ax.axis('off'); continue
        ax.errorbar(xs, means, yerr=stds, marker='o', lw=2, capsize=4,
                    color='#2563EB', ecolor='#6B7280')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('explore bonus')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    for ax in axs[len(VS_BONUS):]:
        ax.axis('off')
    plt.suptitle('Final-window outcome vs exploration bonus — '
                 f'{_ENV_DESC}\n(mean ± std across seeds over the last '
                 f'{int(final_frac * 100)}% of generations)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'eb_vs_bonus{_SUFFIX}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def write_summary(eb_stats, out_dir):
    """Tidy (bonus, generation, metric, mean, std) across seeds."""
    frames = []
    for bonus, (mean_df, std_df) in eb_stats.items():
        m_long = mean_df.reset_index().melt(id_vars='generation',
                                            var_name='metric', value_name='mean')
        s_long = std_df.reset_index().melt(id_vars='generation',
                                           var_name='metric', value_name='std')
        merged = m_long.merge(s_long, on=['generation', 'metric'])
        merged.insert(0, 'explore_bonus', bonus)
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f'eb_behaviour_summary{_SUFFIX}.csv')
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")


def run_env(env_key, args):
    """Discover + aggregate + plot one environment's explore-bonus sweep."""
    env = ENVIRONMENTS[env_key]
    lr = args.learning_lr if args.learning_lr is not None else env['lr']
    if env_key == 'hard':
        roaming = args.predators
        drain = args.drain
    else:
        roaming, drain = env['roaming'], env['drain']

    out_dir = args.out or f'output/explore_bonus_{env_key}'
    os.makedirs(out_dir, exist_ok=True)

    global _ENV_DESC, _SUFFIX, _EB_COLOURS
    _ENV_DESC = env['desc'] if env_key == 'baseline' else \
        f'{int(roaming)} roaming predators, drain {int(drain)}'
    seed_filter = set(args.seeds) if args.seeds else None
    _SUFFIX = ('_seed' + '_'.join(str(s) for s in sorted(seed_filter))) if seed_filter else ''

    want = {'condition': 'learning', 'mode': 'standard', 'epsilon_enabled': False,
            'learning_rate': lr, 'roaming': roaming, 'drain': drain}

    print(f"\n=== {env_key.upper()} environment ({_ENV_DESC}, lr {lr:g}) ===")
    print(f"Discovering explore-bonus sweep runs under {args.root} ...")
    runs = discover(args.root, want)
    if seed_filter is not None:
        runs = [r for r in runs if r['seed'] in seed_filter]
    if not runs:
        print(f"  WARNING: no matching runs (env={env_key}, lr={lr}, roaming={roaming}"
              f"{'' if drain is None else f', drain={drain}'}). Skipping this env.")
        return

    # Group by bonus value.
    by_bonus = {}
    for r in runs:
        by_bonus.setdefault(r['bonus'], []).append(r)
    bonuses = sorted(by_bonus)
    print(f"  Found {len(runs)} runs across bonuses {[f'{b:g}' for b in bonuses]} "
          f"({', '.join(f'{b:g}: {len(by_bonus[b])} seeds' for b in bonuses)}). "
          f"Reading logs ...")

    # Sequential colour ramp over the ordered bonus values (dark 0 -> bright max).
    cmap = plt.get_cmap('viridis')
    ncol = max(len(bonuses) - 1, 1)
    _EB_COLOURS = {b: cmap(0.08 + 0.84 * i / ncol) for i, b in enumerate(bonuses)}

    eb_stats = {}   # bonus -> (mean_df, std_df) across seeds
    per_seed = {}   # bonus -> list of (seed, combined per-gen DataFrame)
    for b in bonuses:
        run_curves = []
        for r in by_bonus[b]:
            curve = load_run(r['org_csv'], args.max_gen)
            if not len(curve):
                print(f"    skip bonus {b:g} seed {r['seed']} — no rows"); continue
            gm = load_gen_metrics(os.path.join(r['dir'], 'generations.csv'), args.max_gen)
            combined = curve if (gm is None or not len(gm)) else curve.join(gm, how='outer')
            run_curves.append((r['seed'], curve))
            per_seed.setdefault(b, []).append((r['seed'], combined))
        if not run_curves:
            continue
        mean_df, std_df = aggregate(run_curves)
        # Fold the generation-level outcomes into the aggregated frames too.
        gen_runs = [(s, df[[c for c in GEN_COLS if c in df.columns]])
                    for s, df in per_seed[b]
                    if any(c in df.columns for c in GEN_COLS)]
        if gen_runs:
            gmean, gstd = aggregate(gen_runs)
            for c in gmean.columns:
                mean_df[c] = gmean[c].reindex(mean_df.index)
                std_df[c] = gstd[c].reindex(std_df.index)
        eb_stats[b] = (mean_df, std_df)
        print(f"    bonus {b:<5g} {len(run_curves)} seeds, {int(mean_df.index.max())} gens")

    if not eb_stats:
        print("  WARNING: nothing loaded for this env. Skipping.")
        return

    print("Plotting ...")
    plot_gen_metrics(eb_stats, args.smooth, out_dir)
    plot_behaviour_curves(eb_stats, args.smooth, out_dir)
    plot_cave_day_night(eb_stats, args.smooth, out_dir)
    plot_food_lines(eb_stats, args.smooth, out_dir)
    plot_vs_bonus(per_seed, bonuses, args.final_frac, out_dir)
    write_summary(eb_stats, out_dir)
    print(f"Done ({env_key}). Outputs in {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser(
        description='Effect of the REINFORCE exploration bonus on fitness, '
                    'population and behaviour (cave usage, diet), baseline + hard.')
    ap.add_argument('--env', choices=['baseline', 'hard', 'both'], default='both',
                    help='Environment(s) to analyse (default: both).')
    ap.add_argument('--root', default='logs/learning/standard/auto-run',
                    help='Folder holding the explore-bonus sweep run dirs '
                         '(learning condition; default logs/learning/standard/auto-run).')
    ap.add_argument('--learning-lr', type=float, default=None,
                    help='learning_rate selecting the runs. Default: env optimum '
                         '(0.01 baseline / 0.02 hard).')
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
                         'vs-bonus summary plot (default 0.2 = last 20%%).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered.')
    ap.add_argument('-o', '--out', default=None,
                    help='Output directory (default: output/explore_bonus_<env>). '
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
