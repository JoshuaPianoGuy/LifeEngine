"""
analyse_predator_sweep.py
======================================================================
Difficulty analysis for the predator sweeps — handles BOTH the roaming sweep
(run_predator_sweep_array.slurm) and the patrol sweep (run_patrol_sweep_array.slurm)
via --predator-type.

Each sweep runs a grid of predator COUNT x DRAIN amount x seed, for one condition
at a time (evolution first). This script discovers those runs, reads each run's
difficulty signals, averages over the seeds, and renders count x drain HEATMAPS so
you can read off where the predator dynamics become too hard (population collapses
/ fitness craters / heavy predation) or too easy (predators barely dent survival).

--predator-type selects which sweep to analyse:
    roaming -> count axis = roaming_predator_count  (patrol disabled that run)
    patrol  -> count axis = predators_per_patch     (roaming disabled that run)
Both sweeps write into the same logs/<cond>/<mode>/auto-run tree, so the selector
also filters to just that sweep's runs (the OTHER predator field is 0 there).

Discovery uses params.json (authoritative), which records:
    roaming_predator_count, predators_per_patch, predator_drain,
    map_seed, condition, mode
so runs are grouped correctly regardless of folder name. The three experimental
conditions are distinguished by (condition, mode):
    evolution = (evolution, standard)
    learning  = (learning,  standard)
    pure_rl   = (learning,  pure_rl)

Signals (averaged over the last --final-window generations, then over seeds):
    from generations.csv : avg_fitness, avg_lifetime, peak_population
    from organisms.csv   : drained-death fraction (direct predation mortality)
                           [skip with --no-death-cause for speed]

Outputs (into --out, suffixed by predator type so both can share a folder):
    predator_sweep_<type>_heatmaps.png   -- 2x2 panel of the signals above
    predator_sweep_<type>_summary.csv    -- per (count, drain) aggregated table

Usage
-----
    python analyse_predator_sweep.py --predator-type roaming --condition evolution
    python analyse_predator_sweep.py --predator-type patrol  --condition evolution
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

# (condition, mode) -> experimental-condition label.
COND_LABELS = {
    ('evolution', 'standard'): 'evolution',
    ('learning',  'standard'): 'learning',
    ('learning',  'pure_rl'):  'pure_rl',
}

# Per predator-type: which params.json field is the count axis, which field must
# be zero to belong to THIS sweep, and the heatmap y-axis label.
PRED_AXIS = {
    'roaming': {'field': 'roaming_predator_count', 'other': 'predators_per_patch',
                'ylabel': 'Roaming predators'},
    'patrol':  {'field': 'predators_per_patch', 'other': 'roaming_predator_count',
                'ylabel': 'Predators per patch'},
}


def cond_label(params):
    return COND_LABELS.get((params.get('condition'), params.get('mode')),
                           f"{params.get('condition')}/{params.get('mode')}")


def discover_runs(logs_root, want_condition, predator_type):
    """Find run folders (recursively) matching want_condition AND predator_type.
    The count axis is the predator-type's field; the OTHER predator field must be
    zero so the two sweeps don't bleed into each other in a shared logs tree.
    Returns list of {count, drain, seed, dir, gen_csv, org_csv}."""
    cfg = PRED_AXIS[predator_type]
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
            if cond_label(p) != want_condition:
                continue
            count = int(p[cfg['field']])
            other = float(p[cfg['other']])
            drain = float(p['predator_drain'])
            seed = int(p['map_seed'])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        # Belongs to THIS sweep only if its own axis is active and the other
        # predator type is disabled.
        if count <= 0 or other != 0:
            continue
        org_csv = os.path.join(run_dir, 'organisms.csv')
        runs.append({'count': count, 'drain': drain, 'seed': seed, 'dir': run_dir,
                     'gen_csv': gen_csv,
                     'org_csv': org_csv if os.path.exists(org_csv) else None})
    return runs


def gen_metrics(gen_csv, window, max_gen):
    """Mean of each difficulty signal over the last `window` generations."""
    df = pd.read_csv(gen_csv)
    df = df[df['generation'] <= max_gen].copy()
    df = df.sort_values('generation')
    out = {'n_gens': int(df['generation'].max()) if len(df) else 0}
    for col in ('avg_fitness', 'avg_lifetime', 'peak_population'):
        if col in df.columns:
            tail = pd.to_numeric(df[col], errors='coerce').dropna().tail(window)
            out[col] = tail.mean() if len(tail) else np.nan
        else:
            out[col] = np.nan
    return out


def drained_fraction(org_csv, window, max_gen):
    """Fraction of deaths caused by predators over the last `window` gens.
    Reads only the two needed columns to stay light on large files."""
    if org_csv is None:
        return np.nan
    try:
        df = pd.read_csv(org_csv, usecols=['generation', 'death_cause'])
    except (ValueError, KeyError):
        return np.nan
    df = df[df['generation'] <= max_gen]
    if not len(df):
        return np.nan
    cutoff = df['generation'].max() - window
    df = df[df['generation'] > cutoff]
    if not len(df):
        return np.nan
    return (df['death_cause'] == 'drained').mean()


def build_table(runs, window, max_gen, with_death):
    rows = []
    for r in runs:
        m = gen_metrics(r['gen_csv'], window, max_gen)
        drained = drained_fraction(r['org_csv'], window, max_gen) if with_death else np.nan
        rows.append({'count': r['count'], 'drain': r['drain'], 'seed': r['seed'],
                     'avg_fitness': m['avg_fitness'], 'avg_lifetime': m['avg_lifetime'],
                     'peak_population': m['peak_population'], 'drained_frac': drained,
                     'n_gens': m['n_gens']})
        print(f"  count={r['count']:<3d} drain={r['drain']:<5g} seed={r['seed']:<5d} "
              f"fit={m['avg_fitness']:.3f} life={m['avg_lifetime']:.0f} "
              f"pop={m['peak_population']:.0f} drained={drained if not np.isnan(drained) else float('nan'):.2f} "
              f"({m['n_gens']} gens)")
    return pd.DataFrame(rows)


def pivot(table, value):
    """count (rows) x drain (cols) matrix of the seed-mean of `value`."""
    agg = table.groupby(['count', 'drain'])[value].mean().reset_index()
    mat = agg.pivot(index='count', columns='drain', values=value)
    mat = mat.sort_index(ascending=False)          # high count at top
    mat = mat.reindex(sorted(mat.columns), axis=1)  # low drain at left
    return mat


def heatmap(ax, mat, title, cmap, harder_when_low, ylabel, fmt='{:.2f}'):
    """Render one count x drain heatmap with per-cell annotations."""
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f'{c:g}' for c in mat.columns])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f'{r:g}' for r in mat.index])
    ax.set_xlabel('Drain amount')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', color='gray', fontsize=8)
                continue
            # White text on dark cells for legibility.
            rng = (vmax - vmin) or 1.0
            dark = (v - vmin) / rng > 0.6
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=9,
                    color='white' if dark else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    corner = 'low = harder' if harder_when_low else 'high = harder'
    ax.text(0.0, 1.02, corner, transform=ax.transAxes, fontsize=8, color='#6B7280')


def plot_heatmaps(table, condition, predator_type, ylabel, out_dir):
    print("  Plotting difficulty heatmaps ...")
    panels = [
        ('avg_fitness',     'Mean fitness (final gens)',      'YlGn',  True),
        ('avg_lifetime',    'Mean lifespan (final gens)',     'YlGn',  True),
        ('peak_population', 'Peak population (final gens)',    'YlGn',  True),
        ('drained_frac',    'Predation death fraction',       'Reds',  False),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, title, cmap, harder_low) in zip(axs.flat, panels):
        mat = pivot(table, col)
        if mat.isna().all().all():
            ax.axis('off')
            ax.set_title(f'{title}\n(no data)', fontsize=11, color='gray')
            continue
        fmt = '{:.2f}' if col in ('avg_fitness', 'drained_frac') else '{:.0f}'
        heatmap(ax, mat, title, cmap, harder_low, ylabel, fmt)
    plt.suptitle(f'{predator_type.capitalize()} predator difficulty sweep — {condition} '
                 f'(mean over {table["seed"].nunique()} seeds)',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(out_dir, f'predator_sweep_{predator_type}_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description='Predator difficulty-sweep analysis.')
    ap.add_argument('--logs-root', default='logs',
                    help='Root to search recursively for run folders.')
    ap.add_argument('--predator-type', default='roaming',
                    choices=['roaming', 'patrol'],
                    help='Which sweep to analyse: roaming (count axis = '
                         'roaming_predator_count) or patrol (predators_per_patch).')
    ap.add_argument('--condition', default='evolution',
                    choices=['evolution', 'learning', 'pure_rl'],
                    help='Which experimental condition to analyse.')
    ap.add_argument('--final-window', type=int, default=50,
                    help='Average signals over the last N generations per run.')
    ap.add_argument('--max-gen', type=int, default=10**9)
    ap.add_argument('--no-death-cause', action='store_true',
                    help='Skip reading organisms.csv (faster; drops predation heatmap).')
    ap.add_argument('--out', default='output/predator_sweep')
    args = ap.parse_args()

    ylabel = PRED_AXIS[args.predator_type]['ylabel']
    os.makedirs(args.out, exist_ok=True)
    print(f"Discovering {args.predator_type} / {args.condition} runs under {args.logs_root} ...")
    runs = discover_runs(args.logs_root, args.condition, args.predator_type)
    if not runs:
        raise SystemExit(f"No {args.predator_type} {args.condition} runs with params.json + "
                         f"generations.csv under {args.logs_root}. Has the sweep finished?")
    print(f"Found {len(runs)} runs "
          f"({len({r['count'] for r in runs})} counts x "
          f"{len({r['drain'] for r in runs})} drains x "
          f"{len({r['seed'] for r in runs})} seeds). Reading signals ...")

    table = build_table(runs, args.final_window, args.max_gen,
                        with_death=not args.no_death_cause)

    agg = table.groupby(['count', 'drain']).agg(
        avg_fitness=('avg_fitness', 'mean'),
        avg_lifetime=('avg_lifetime', 'mean'),
        peak_population=('peak_population', 'mean'),
        drained_frac=('drained_frac', 'mean'),
        n_seeds=('seed', 'nunique'),
    ).reset_index().sort_values(['count', 'drain'])

    print("\n-- Aggregated (mean over seeds) --")
    print(agg.to_string(index=False))

    # Flag the extremes so the too-hard / too-easy ends are explicit.
    hardest = agg.loc[agg['avg_fitness'].idxmin()]
    easiest = agg.loc[agg['avg_fitness'].idxmax()]
    print(f"\nHardest cell  (lowest fitness): count={hardest['count']:g} "
          f"drain={hardest['drain']:g}  fitness={hardest['avg_fitness']:.3f}")
    print(f"Easiest cell (highest fitness): count={easiest['count']:g} "
          f"drain={easiest['drain']:g}  fitness={easiest['avg_fitness']:.3f}\n")

    summary_path = os.path.join(args.out, f'predator_sweep_{args.predator_type}_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    plot_heatmaps(table, args.condition, args.predator_type, ylabel, args.out)
    print(f"\nDone. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
