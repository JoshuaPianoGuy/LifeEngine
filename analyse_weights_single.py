"""
analyse_weights_single.py
======================================================================
Per-condition summary plotting suite for LifeEngine.
Processes a single simulation run at a time with descriptive, interpretable axes.

Generates:
  1. execution_summary.png     -- 7-panel isolated diagnostics grid (always)
  2. organism_fitness.png      -- per-organism fitness distribution over generations
                                  (requires --organisms)
  3. organism_food_types.png   -- mean food-type breakdown per generation
                                  (requires --organisms)

PCA and t-SNE plots have been removed -- they depend on w1_weights snapshots
which are no longer logged. Re-add if weight snapshots are re-enabled.

Usage
-----
    # Generation summary only
    python analyse_weights_single.py \
        --generations path/to/generations.csv \
        --label "GA + RL Learning Condition" \
        --max-gen 1000 \
        --out output/plots

    # With per-organism plots
    python analyse_weights_single.py \
        --generations path/to/generations.csv \
        --organisms path/to/organisms.csv \
        --label "GA + RL Learning Condition" \
        --max-gen 1000 \
        --out output/plots
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── 1. Historical Summary Plot ───────────────────────────────────────────────

def _clean_generations(g):
    """
    Strip artifacts that make the line plots backtrack and spike:
      * periodic world-snapshot dumps logged with generation == 0
        (they carry reset values like total_agents=100 and a non-standard
         generation_ticks, and corrupt every panel when plotted on the same axes)
      * duplicate generation rows (keep the last occurrence)
      * unsorted rows (plot() connects in file order, not by x)
    """
    g = g[g['generation'] > 0]
    g = g.drop_duplicates(subset='generation', keep='last')
    g = g.sort_values('generation').reset_index(drop=True)
    return g


def _series(g, col, smooth):
    """Return the column, rolling-mean smoothed when smooth > 0, else raw."""
    s = g[col]
    if smooth and smooth > 0:
        return s.rolling(window=smooth, min_periods=1).mean()
    return s


def plot_exact_summary_panel(gen_path, label, max_gen, out_dir='.', smooth=0):
    print("  Generating 7-panel line metrics canvas ...")
    g = pd.read_csv(gen_path)
    g = g[g['generation'] <= max_gen]
    g = _clean_generations(g)
    if smooth and smooth > 0:
        print(f"  Applying rolling-mean smoothing (window={smooth}).")

    fig, axs = plt.subplots(3, 3, figsize=(18, 15))

    # Panel 1: Fitness Performance
    axs[0, 0].plot(g['generation'], _series(g, 'best_fitness', smooth), color='#1D4ED8', linestyle='-', lw=1.2, alpha=0.4, label='Best Fitness')
    axs[0, 0].plot(g['generation'], _series(g, 'top20percent_fitness', smooth), color='#1D4ED8', linestyle='--', lw=1.8, label='Top 20% Fittest')
    axs[0, 0].plot(g['generation'], _series(g, 'avg_fitness', smooth), color='#1D4ED8', linestyle=':', lw=2.2, label='Average Fitness')
    axs[0, 0].set_title('Fitness Performance History', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Absolute Fitness')
    axs[0, 0].legend(loc='upper left', fontsize=9)
    axs[0, 0].grid(True, alpha=0.3)

    # Panel 2: Population Growth Dynamics
    axs[0, 1].plot(g['generation'], _series(g, 'total_agents', smooth), color='#2563EB', linestyle='-', lw=2, label='Total Agent Count')
    axs[0, 1].plot(g['generation'], _series(g, 'peak_population', smooth), color='#DB2777', linestyle='--', lw=2, label='Peak Population')
    axs[0, 1].set_title('Population Growth Dynamics', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Organism Counts')
    axs[0, 1].legend(loc='upper left', fontsize=9)
    axs[0, 1].grid(True, alpha=0.3)

    # Panel 3: Average Lifespan
    axs[0, 2].plot(g['generation'], _series(g, 'avg_lifetime', smooth), color='#D97706', lw=2)
    axs[0, 2].set_title('Average Lifespan per Generation', fontsize=12, fontweight='bold')
    axs[0, 2].set_ylabel('Ticks Survived')
    axs[0, 2].grid(True, alpha=0.3)

    # Panel 4: Within-Lifetime Weight Drift (Plasticity)
    if 'avg_learned_weight_diff' in g.columns and g['avg_learned_weight_diff'].max() > 0:
        axs[1, 0].plot(g['generation'], _series(g, 'avg_learned_weight_diff', smooth), color='#059669', lw=2)
        axs[1, 0].set_title('Within-Lifetime Weight Drift (Plasticity)', fontsize=12, fontweight='bold')
        axs[1, 0].set_ylabel('Mean Absolute Shift (deltaW_life)')
    else:
        axs[1, 0].text(0.5, 0.5, 'Plasticity Inactive\n(Pure Genetic Control)',
                       ha='center', va='center', fontsize=11, color='gray', transform=axs[1, 0].transAxes)
        axs[1, 0].set_title('Within-Lifetime Weight Drift', fontsize=12, fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3)

    # Panel 5: Baseline Network Weight Magnitude
    axs[1, 1].plot(g['generation'], _series(g, 'avg_network_weight_mag', smooth), color='#7C3AED', lw=2)
    axs[1, 1].set_title('Average Network Weight Magnitude', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Structural Scale Values')
    axs[1, 1].grid(True, alpha=0.3)

    # Panel 6: Inter-generational Structural Weight Step (Velocity)
    axs[1, 2].plot(g['generation'], _series(g, 'inter_gen_weight_change', smooth), color='#EA580C', lw=2)
    axs[1, 2].set_title('Inter-Generational Weight Step (Velocity)', fontsize=12, fontweight='bold')
    axs[1, 2].set_ylabel('Evolutionary Step Delta (deltaW_gen)')
    axs[1, 2].grid(True, alpha=0.3)

    # Panel 7: Structural Genomic Variance
    axs[2, 0].plot(g['generation'], _series(g, 'genome_variance', smooth), color='#DC2626', lw=2)
    axs[2, 0].set_title('Genomic Convergence (Structural Variance)', fontsize=12, fontweight='bold')
    axs[2, 0].set_ylabel('Variance')
    axs[2, 0].set_xlabel('Generation')
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].axis('off')
    axs[2, 2].axis('off')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 2].set_xlabel('Generation')

    plt.suptitle(f"Historical Run Execution Diagnostics: {label}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    safe_label = label.lower().replace(' ', '_').replace('/', '_')
    out_path = os.path.join(out_dir, f'execution_summary_{safe_label}.png')
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── 2. Per-organism Plots ────────────────────────────────────────────────────

def load_organisms(path, max_gen):
    print(f"  Loading organisms dataset from {path} ...")
    df = pd.read_csv(path)
    df = df[df['generation'] <= max_gen]
    # Coerce numeric columns that may have been read as strings
    for col in ['fitness', 'lifetime', 'food_low', 'food_medium', 'food_prestige', 'food_default',
                'learned_weight_diff', 'cells_visited']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  Loaded {len(df)} organism rows across {df['generation'].nunique()} generations.")
    return df


def plot_organism_fitness(df, label, out_dir='.'):
    """
    Two panels:
      Left:  per-generation mean and max individual fitness (scatter + line)
      Right: fitness distribution across all organisms as a histogram, coloured
             by generation era (early / mid / late thirds of the run)
    """
    print("  Generating per-organism fitness plots ...")

    gens = sorted(df['generation'].unique())
    gen_mean = df.groupby('generation')['fitness'].mean()
    gen_max  = df.groupby('generation')['fitness'].max()

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: mean and max fitness per generation
    axs[0].plot(gens, gen_mean.loc[gens], color='#2563EB', lw=2, label='Mean fitness')
    axs[0].plot(gens, gen_max.loc[gens],  color='#DB2777', lw=1.5, linestyle='--', label='Max fitness')
    axs[0].set_title('Per-Generation Individual Fitness', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness (cumulative food score)')
    axs[0].legend(fontsize=9)
    axs[0].grid(True, alpha=0.3)

    # Panel 2: fitness histogram split into three eras
    n_gens = len(gens)
    era_boundaries = [gens[0], gens[n_gens // 3], gens[2 * n_gens // 3], gens[-1]]
    era_labels = [
        f'Early  (gen {era_boundaries[0]}–{era_boundaries[1]})',
        f'Mid    (gen {era_boundaries[1]}–{era_boundaries[2]})',
        f'Late   (gen {era_boundaries[2]}–{era_boundaries[3]})',
    ]
    era_colours = ['#93C5FD', '#3B82F6', '#1E3A8A']

    for i, (lo, hi, colour, lbl) in enumerate(zip(
            era_boundaries, era_boundaries[1:], era_colours, era_labels)):
        subset = df[(df['generation'] >= lo) & (df['generation'] <= hi)]['fitness'].dropna()
        if len(subset) == 0:
            continue
        axs[1].hist(subset, bins=40, color=colour, alpha=0.6, label=lbl, density=True)

    axs[1].set_title('Fitness Distribution by Era', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Fitness (cumulative food score)')
    axs[1].set_ylabel('Density')
    axs[1].legend(fontsize=9)
    axs[1].grid(True, alpha=0.3)

    plt.suptitle(f"Per-Organism Fitness Analysis: {label}", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    safe_label = label.lower().replace(' ', '_').replace('/', '_')
    out_path = os.path.join(out_dir, f'organism_fitness_{safe_label}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_organism_food_types(df, label, out_dir='.'):
    """
    Two panels:
      Left:  stacked area chart of mean food counts per type across generations
      Right: proportional stacked bar chart (food-type share) sampled at
             evenly spaced generation checkpoints so the bars are readable
             even over long runs
    """
    print("  Generating food-type breakdown plots ...")

    food_cols = {
        'food_prestige': ('Prestige food', '#7C3AED'),
        'food_medium':   ('Medium food',   '#059669'),
        'food_low':      ('Low food',      '#D97706'),
        'food_default':  ('Default food',  '#94A3B8'),
    }
    # Only keep columns that actually exist in this CSV
    present = {k: v for k, v in food_cols.items() if k in df.columns}
    if not present:
        print("  No food-type columns found in organisms.csv — skipping food plot.")
        return

    gens = sorted(df['generation'].unique())
    agg = df.groupby('generation')[[*present.keys()]].mean()

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: stacked area — absolute mean counts
    bottoms = np.zeros(len(gens))
    for col, (lbl, colour) in present.items():
        vals = agg.loc[gens, col].fillna(0).values
        axs[0].fill_between(gens, bottoms, bottoms + vals, alpha=0.75, color=colour, label=lbl)
        bottoms += vals

    axs[0].set_title('Mean Food Items Eaten per Type', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Mean count per organism')
    axs[0].legend(fontsize=9, loc='upper left')
    axs[0].grid(True, alpha=0.2)

    # Panel 2: proportional stacked bar at sampled checkpoints
    # Pick up to 20 evenly spaced generations so bars don't overlap
    n_bars = min(20, len(gens))
    indices = np.round(np.linspace(0, len(gens) - 1, n_bars)).astype(int)
    sample_gens = [gens[i] for i in indices]

    bar_width = max(1, (sample_gens[-1] - sample_gens[0]) / (n_bars * 1.5)) if n_bars > 1 else 1
    bottoms = np.zeros(n_bars)
    for col, (lbl, colour) in present.items():
        vals = agg.loc[sample_gens, col].fillna(0).values
        totals = agg.loc[sample_gens].fillna(0).sum(axis=1).values
        # Avoid division by zero for generations where no food was eaten
        props = np.where(totals > 0, vals / totals, 0)
        axs[1].bar(sample_gens, props, width=bar_width, bottom=bottoms,
                   color=colour, alpha=0.85, label=lbl)
        bottoms += props

    axs[1].set_title('Food-Type Proportion at Generation Checkpoints', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Proportion of food eaten')
    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=9, loc='upper left')
    axs[1].grid(True, alpha=0.2, axis='y')

    plt.suptitle(f"Food-Type Breakdown: {label}", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    safe_label = label.lower().replace(' ', '_').replace('/', '_')
    out_path = os.path.join(out_dir, f'organism_food_types_{safe_label}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main Execution ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Descriptive Single Run Summary Suite')
    parser.add_argument('--generations', required=True,
                        help='Path to generations.csv')
    parser.add_argument('--organisms', default=None,
                        help='Path to organisms.csv (optional). '
                             'If provided, generates organism_fitness and '
                             'organism_food_types plots in addition to the '
                             'generation summary.')
    parser.add_argument('--label', default='Simulation Run',
                        help='Human-readable condition label used in plot titles '
                             'and output filenames.')
    parser.add_argument('--out', default='.',
                        help='Output directory for all generated plots.')
    parser.add_argument('--max-gen', type=int, default=3000,
                        help='Discard rows beyond this generation (default 3000).')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Rolling-mean window (in generations) for the summary '
                             'panel lines. 0 (default) plots raw values; >0 smooths '
                             'noisy metrics for readability.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n-- Processing Diagnostic Extraction: {args.label} --")
    plot_exact_summary_panel(args.generations, args.label,
                             max_gen=args.max_gen, out_dir=args.out,
                             smooth=args.smooth)

    if args.organisms:
        org_df = load_organisms(args.organisms, max_gen=args.max_gen)
        plot_organism_fitness(org_df, args.label, out_dir=args.out)
        plot_organism_food_types(org_df, args.label, out_dir=args.out)

    print(f"\n-- Complete. Graphics saved to: {os.path.abspath(args.out)}\n")

if __name__ == '__main__':
    main()