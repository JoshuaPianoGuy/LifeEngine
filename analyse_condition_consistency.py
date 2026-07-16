"""
analyse_condition_consistency.py
======================================================================
Per-seed STOCHASTICITY of the three experimental conditions (EVOLUTION,
LEARNING, PURE-RL), from the condition-consistency runs produced by
run_condition_consistency_array.slurm.

That job runs, for each condition, EVERY map seed FIVE independent times
(5 replicates per (condition, seed) cell):

    3 conditions x 5 seeds x 5 replicates = 75 runs.

Only the map TERRAIN is seeded (--map-seed); GA mutation, RL exploration and
predator wandering are UNSEEDED, so the 5 replicates of a (condition, seed)
cell share identical terrain but differ purely by algorithmic/predator chance.
Their spread is therefore the run-to-run NOISE of that condition on that map.

This script reads generations.csv from every replicate and produces, for each
of the three fitness/population outcome metrics:

    avg_fitness            Mean average fitness
    top20percent_fitness   Top-20% fitness
    peak_population        Peak population
    total_agents           Total agents

TWO kinds of graph:

  * PER-SEED (one figure per seed): all three conditions on the same map, each
    drawn as the mean across its 5 replicates with a ±1 std band. The band here
    is the REPLICATE spread on that seed -> the algorithmic stochasticity of the
    condition on that terrain. The tighter the band, the more CONSISTENT the
    condition on that map.

  * OVERALL (one figure): each condition's per-seed mean curves averaged across
    the 5 seeds, with a ±1 std band that is the spread ACROSS SEEDS -> the
    map/terrain sensitivity of the condition (the between-seed variability that
    the per-seed figures hold fixed).

This mirrors analyse_conditions_comparison.py (same discovery-by-params.json,
env presets, colours, smoothing) but its repeat unit is the REPLICATE within a
seed, not the seed within a condition — the consistency runs put 5 same-seed
replicate folders (…_seedS_rep1..5) in each condition tree, which the comparison
script would collide on (it keys aggregation by seed).

Discovery matches the same three condition families by params.json, then splits
each family's runs by (map_seed, replicate). The replicate index is taken from
the folder-name _rep<N> suffix (params.json carries no replicate field).

Outputs (into --out, default output/condition_consistency_<env>):
    consistency_seed<seed>.png     -- per-seed: 4 metric panels, one line per
                                      condition (mean ± std across replicates).
    consistency_overall.png        -- across seeds: 4 metric panels, one line per
                                      condition (mean ± std across seeds).
    consistency_summary.csv        -- tidy (scope, condition, seed, generation,
                                      metric, mean, std); scope is 'seed<S>' for
                                      the per-seed replicate stats or 'overall'
                                      for the across-seed stats.

Usage
-----
    python analyse_condition_consistency.py --env hard
    python analyse_condition_consistency.py --env baseline
    python analyse_condition_consistency.py --env hard --seeds 1 42
    python analyse_condition_consistency.py --env hard --max-gen 1000
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

# ── Condition identities (order = plot order + display colours) ───────────────
COND_ORDER = ['evolution', 'learning', 'pure_rl']
COND_COLOURS_KEY = {
    'evolution': '#DC2626',   # red
    'learning':  '#2563EB',   # blue
    'pure_rl':   '#059669',   # green
}
# Filled in main(): display label -> colour, keyed by the resolved label.
COND_COLOURS = {}

# Environment presets (same as analyse_conditions_comparison.py). `roaming` is the
# roaming_predator_count filter; `drain` is the predator_drain filter (None = don't
# filter). `lr` is the RL learning rate for the two RL-using conditions.
ENVIRONMENTS = {
    'baseline': {'roaming': 0.0,  'drain': None, 'lr': 0.01,
                 'desc': 'predator-free baseline'},
    'hard':     {'roaming': 60.0, 'drain': 5.0,  'lr': 0.02,
                 'desc': '60 roaming predators, drain 5'},
}

# ── Outcome metrics (from generations.csv, already population-level) ───────────
GEN_METRICS = {
    'avg_fitness':          'Mean average fitness',
    'top20percent_fitness': 'Top-20% fitness',
    'peak_population':      'Peak population',
    'total_agents':         'Total agents',
}
GEN_COLS = list(GEN_METRICS)

_REP_RE = re.compile(r'_rep(\d+)$')

# Environment description used in plot titles / seed-subset filename tag (main()).
_ENV_DESC = ''
_SUFFIX = ''


def _num(p, key):
    """Read a numeric params.json field, tolerating strings; NaN if absent/bad."""
    try:
        return float(p[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def discover(root, want, require_rep=True):
    """Find replicate run folders under `root` (params.json + generations.csv)
    whose params.json matches every filter in `want`, returning
    {seed, rep, dir, gen_csv} dicts. params.json is authoritative for the run
    identity; the replicate index comes from the folder-name _rep<N> suffix
    (params.json carries no replicate field).

    The consistency job is the only thing that writes _rep<N> folders, so by
    default (require_rep=True) folders WITHOUT that suffix are skipped. This is
    load-bearing: the same param filter (e.g. learning, lr 0.02, roaming 60,
    drain 5) also matches unrelated runs living in the same tree — notably the
    explore-bonus sweep's _eb<X> folders — and those must not be mistaken for
    replicates. Pass require_rep=False (‑‑include-non-rep) to treat a suffix-less
    folder as replicate 1 (e.g. for a tree that predates the _rep convention).

    `want` keys (same semantics as analyse_conditions_comparison.py):
        condition        required exact match
        mode             optional exact match ('standard' / 'pure_rl')
        epsilon_enabled  optional exact bool match
        learning_rate    optional float match (tol 1e-9)
        roaming          roaming_predator_count match (tol 1e-9)
        drain            predator_drain match (tol 1e-9); skipped if None"""
    runs = []
    for params_path in glob.glob(os.path.join(root, '*', 'params.json')):
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
        m = _REP_RE.search(os.path.basename(run_dir))
        if m is None and require_rep:
            continue
        rep = int(m.group(1)) if m else 1
        runs.append({'seed': seed, 'rep': rep, 'dir': run_dir, 'gen_csv': gen_csv})
    return sorted(runs, key=lambda r: (r['seed'], r['rep']))


def load_gen_metrics(gen_csv, max_gen):
    """Read the outcome metrics (GEN_COLS) from a run's generations.csv, indexed
    by generation (only the present columns), or None if unavailable. Deduplicates
    on generation (keep last) to be robust to a doubled/appended CSV."""
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


def aggregate(curves, key_name):
    """Aggregate a list of (key, per-generation DataFrame) into (mean, std)
    DataFrames indexed by generation (mean/std across the keyed repeats)."""
    stacked = pd.concat({k: c for k, c in curves}, names=[key_name, 'generation'])
    mean = stacked.groupby(level='generation').mean()
    std = stacked.groupby(level='generation').std()
    return mean, std


def _draw(ax, cond_stats, col, smooth):
    """Draw one line per condition (mean ± std band) for column `col` on `ax`.
    cond_stats maps condition label -> (mean_df, std_df). Returns True if drawn."""
    drawn = False
    for label, (mean_df, std_df) in cond_stats.items():
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


def plot_metrics(cond_stats, smooth, out_dir, fname, suptitle):
    """Generic 4-panel (one per GEN_METRICS metric) figure: one line per
    condition (mean ± std band). Returns the saved path, or None if nothing
    was drawable."""
    metrics = [m for m in GEN_METRICS
               if any(m in md.columns for md, _sd in cond_stats.values())]
    if not cond_stats or not metrics:
        return None
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()
    for ax, m in zip(axs, metrics):
        _draw(ax, cond_stats, m, smooth)
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
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def write_summary(per_seed_stats, overall_stats, out_dir):
    """Tidy (scope, condition, seed, generation, metric, mean, std). scope is
    'seed<S>' for per-seed replicate stats, 'overall' for across-seed stats."""
    frames = []

    def _tidy(mean_df, std_df, scope, label, seed):
        m_long = mean_df.reset_index().melt(id_vars='generation',
                                            var_name='metric', value_name='mean')
        s_long = std_df.reset_index().melt(id_vars='generation',
                                           var_name='metric', value_name='std')
        merged = m_long.merge(s_long, on=['generation', 'metric'])
        merged.insert(0, 'seed', seed)
        merged.insert(0, 'condition', label)
        merged.insert(0, 'scope', scope)
        return merged

    for seed, cond_stats in per_seed_stats.items():
        for label, (mean_df, std_df) in cond_stats.items():
            frames.append(_tidy(mean_df, std_df, f'seed{seed}', label, seed))
    for label, (mean_df, std_df) in overall_stats.items():
        frames.append(_tidy(mean_df, std_df, 'overall', label, 'all'))

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f'consistency_summary{_SUFFIX}.csv')
    out.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(
        description='Per-seed stochasticity of the three conditions from the '
                    'condition-consistency replicate runs.')
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
    ap.add_argument('--max-gen', type=int, default=10**9,
                    help='Ignore generations beyond this (for a strict head-to-head '
                         'when conditions differ in length).')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Rolling-mean window for the curves/bands (1 = none).')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Restrict to these world seeds. Default: all discovered. '
                         'Output filenames get a _seed<..> suffix.')
    ap.add_argument('--include-non-rep', action='store_true',
                    help='Also include matching run folders that lack a _rep<N> '
                         'suffix (treated as replicate 1). Off by default so that '
                         'unrelated same-param runs sharing the tree — e.g. the '
                         'explore-bonus _eb<X> sweep — are NOT counted as replicates.')
    ap.add_argument('-o', '--out', default=None,
                    help='Directory for the PNGs + summary CSV '
                         '(default: output/condition_consistency_<env>).')
    args = ap.parse_args()

    env = ENVIRONMENTS[args.env]
    lr = args.learning_lr if args.learning_lr is not None else env['lr']
    if args.env == 'hard':
        roaming, drain = args.predators, args.drain
    else:
        roaming, drain = env['roaming'], env['drain']

    out_dir = args.out or f'output/condition_consistency_{args.env}'
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

    # per_seed_stats: seed -> {label -> (mean_df, std_df)}   band = replicate spread
    # overall_stats:  label -> (mean_df, std_df)             band = across-seed spread
    per_seed_stats = {}
    overall_stats = {}

    for label, root, want in families:
        print(f"\nDiscovering {label} runs under {root} ...")
        runs = discover(root, want, require_rep=not args.include_non_rep)
        if seed_filter is not None:
            runs = [r for r in runs if r['seed'] in seed_filter]
            missing = seed_filter - {r['seed'] for r in runs}
            if missing:
                print(f"  note: requested seeds not found for {label}: {sorted(missing)}")
        if not runs:
            print(f"  WARNING: no matching runs for {label} "
                  f"(env={args.env}, roaming={roaming}"
                  f"{'' if drain is None else f', drain={drain}'}). Skipping.")
            continue

        # Group replicates by seed.
        by_seed = {}
        for r in runs:
            by_seed.setdefault(r['seed'], []).append(r)
        seeds_found = sorted(by_seed)
        print(f"  Found {len(runs)} replicate runs across seeds {seeds_found} "
              f"({', '.join(f'seed {s}: {len(by_seed[s])} reps' for s in seeds_found)}). "
              f"Reading generations.csv ...")

        seed_means = []   # (seed, per-seed mean curve) -> feeds the OVERALL aggregate
        for seed in seeds_found:
            rep_curves = []
            for r in by_seed[seed]:
                gm = load_gen_metrics(r['gen_csv'], args.max_gen)
                if gm is None or not len(gm):
                    print(f"    skip seed {seed} rep {r['rep']} — no rows"); continue
                rep_curves.append((r['rep'], gm))
            if not rep_curves:
                continue
            mean_df, std_df = aggregate(rep_curves, 'rep')
            per_seed_stats.setdefault(seed, {})[label] = (mean_df, std_df)
            seed_means.append((seed, mean_df))
            print(f"    seed {seed:<5d} {len(rep_curves)} reps, "
                  f"{int(mean_df.index.max())} gens")

        if seed_means:
            # Across-seed aggregate of the per-seed mean curves (band = seed spread).
            overall_stats[label] = aggregate(seed_means, 'seed')

    if not per_seed_stats:
        raise SystemExit("No runs discovered for any condition — check the "
                         "--env / --*-root / --learning-lr / --predators / --drain filters.")

    print("\nPlotting ...")
    # Per-seed figures: replicate spread of each condition on that map.
    for seed in sorted(per_seed_stats):
        path = plot_metrics(
            per_seed_stats[seed], args.smooth, out_dir,
            f'consistency_seed{seed}.png',
            f'Seed {seed} — fitness & population per generation (three conditions)\n'
            f'({_ENV_DESC}; mean ± std across 5 replicates — replicate spread on this map)')
        if path:
            print(f"  Saved: {path}")

    # Overall figure: across-seed spread of each condition.
    path = plot_metrics(
        overall_stats, args.smooth, out_dir,
        f'consistency_overall{_SUFFIX}.png',
        'Fitness & population per generation — three conditions, all seeds\n'
        f'({_ENV_DESC}; mean ± std across seeds — between-seed / map spread)')
    if path:
        print(f"  Saved: {path}")

    summ = write_summary(per_seed_stats, overall_stats, out_dir)
    if summ:
        print(f"  Saved: {summ}")

    print(f"\nDone. Outputs in {os.path.abspath(out_dir)}\n")


if __name__ == '__main__':
    main()
