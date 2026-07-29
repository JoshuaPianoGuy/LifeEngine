"""
analyse_behaviour.py
======================================================================
Per-organism behavioural analysis for LifeEngine predator experiments.
Plots how organism behaviour (predator avoidance, cave use, exploration,
foraging) changes over generations, and correlates behaviour with survival.

Reads organisms.csv with columns:
  generation, rank, map_seed, condition, fitness, lifetime,
  energy_at_death, death_cause, energy_at_tick_1000, cumulative_food_score,
  food_low, food_medium, food_prestige, food_default,
  cells_visited, predator_touches, drained_ticks,
  cave_entries, cave_entries_day, cave_entries_night,
  learned_weight_diff, network_weight_magnitude

Generates:
  1. behaviour_over_time.png   -- per-generation means of behavioural metrics
  2. survival_behaviour.png    -- behaviour broken down by death cause
  3. death_cause_over_time.png -- stacked proportion of death causes per generation

Usage
-----
    python analyse_behaviour.py \
        --organisms path/to/organisms.csv \
        --label "GA + RL 15 Predators Drain 5" \
        --max-gen 1000 \
        --smooth 10 \
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

# Consistent colours for death causes across all plots
DEATH_COLOURS = {
    'survived': '#059669',  # green — made it to end of life
    'starved':  '#D97706',  # amber — ran out of energy
    'drained':  '#DC2626',  # red   — killed by predator
}


def load(path, max_gen):
    print(f"  Loading {path} ...")
    df = pd.read_csv(path)
    df = df[df['generation'] <= max_gen].copy()
    numeric = ['fitness', 'lifetime', 'cells_visited', 'predator_touches',
               'drained_ticks', 'cave_entries', 'cave_entries_day',
               'cave_entries_night', 'food_low', 'food_medium',
               'food_prestige', 'food_default', 'learned_weight_diff',
               'network_weight_magnitude']
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f"  {len(df)} rows, generations {df['generation'].min()}-{df['generation'].max()}, "
          f"conditions: {list(df['condition'].unique())}")
    return df


def _smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_behaviour_over_time(df, label, smooth, out_dir):
    """Six-panel grid of behavioural metric means per generation."""
    print("  Plotting behaviour over time ...")
    g = df.groupby('generation')

    gens = sorted(df['generation'].unique())
    panels = [
        ('predator_touches', 'Mean Predator Contacts',      'Contacts per organism',     '#DC2626'),
        ('drained_ticks',    'Mean Ticks Drained',          'Ticks drained per organism','#B91C1C'),
        ('cave_entries',     'Mean Cave Entries',           'Entries per organism',      '#2563EB'),
        ('cells_visited',    'Mean Exploration (cells)',    'Unique cells visited',      '#7C3AED'),
        ('lifetime',         'Mean Lifespan',               'Ticks survived',            '#D97706'),
        ('fitness',          'Mean Fitness',                'Cumulative food score',     '#059669'),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (col, title, ylabel, colour) in zip(axs.flat, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f'{col}\nnot in data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        mean_series = g[col].mean()
        ax.plot(gens, _smooth(mean_series.loc[gens], smooth), color=colour, lw=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Behavioural Metrics Over Time: {label}",
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    safe = label.lower().replace(' ', '_').replace('/', '_')
    path = os.path.join(out_dir, f'behaviour_over_time_{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_survival_behaviour(df, label, out_dir):
    """
    How does behaviour differ between organisms that survived, starved, or
    were drained by predators? This is the core causal-chain plot: it shows
    whether avoidance behaviours (cave use, low contact) correlate with
    surviving rather than being drained.
    """
    print("  Plotting behaviour by death cause ...")
    if 'death_cause' not in df.columns:
        print("  No death_cause column — skipping survival plot.")
        return

    causes = [c for c in ['survived', 'starved', 'drained']
              if c in df['death_cause'].unique()]
    metrics = [
        ('predator_touches', 'Predator Contacts'),
        ('cave_entries',     'Cave Entries'),
        ('drained_ticks',    'Ticks Drained'),
        ('cells_visited',    'Cells Visited'),
        ('fitness',          'Fitness'),
        ('lifetime',         'Lifespan'),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (col, title) in zip(axs.flat, metrics):
        if col not in df.columns:
            ax.axis('off')
            continue
        means = [df.loc[df['death_cause'] == c, col].mean() for c in causes]
        sems  = [df.loc[df['death_cause'] == c, col].sem()  for c in causes]
        colours = [DEATH_COLOURS.get(c, '#888888') for c in causes]
        ax.bar(causes, means, yerr=sems, color=colours, alpha=0.85, capsize=4)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {title.lower()}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Behaviour by Death Cause: {label}",
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    safe = label.lower().replace(' ', '_').replace('/', '_')
    path = os.path.join(out_dir, f'survival_behaviour_{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_death_cause_over_time(df, label, smooth, out_dir):
    """Stacked proportion of death causes per generation — shows whether the
    population shifts from being drained toward surviving as avoidance evolves."""
    print("  Plotting death cause composition over time ...")
    if 'death_cause' not in df.columns:
        print("  No death_cause column — skipping.")
        return

    gens = sorted(df['generation'].unique())
    causes = [c for c in ['drained', 'starved', 'survived']
              if c in df['death_cause'].unique()]

    # Proportion of each cause per generation
    counts = df.groupby(['generation', 'death_cause']).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(gens))
    for c in causes:
        if c not in props.columns:
            continue
        vals = _smooth(props[c].reindex(gens).fillna(0), smooth).values
        ax.fill_between(gens, bottom, bottom + vals, color=DEATH_COLOURS.get(c, '#888'),
                        alpha=0.8, label=c)
        bottom += vals

    ax.set_title(f'Death Cause Composition Over Time: {label}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Proportion of deaths')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    safe = label.lower().replace(' ', '_').replace('/', '_')
    path = os.path.join(out_dir, f'death_cause_over_time_{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_summary(df):
    """Print the key behavioural correlations to stdout for quick inspection."""
    if 'death_cause' not in df.columns:
        return
    print("\n  ── Behavioural summary by death cause ──")
    cols = [c for c in ['predator_touches', 'drained_ticks', 'cave_entries',
                        'cells_visited', 'fitness', 'lifetime'] if c in df.columns]
    summary = df.groupby('death_cause')[cols].mean().round(2)
    print(summary.to_string())
    print()


def main():
    ap = argparse.ArgumentParser(description='Per-organism behavioural analysis')
    ap.add_argument('--organisms', required=True)
    ap.add_argument('--label', default='Run')
    ap.add_argument('--out', default='.')
    ap.add_argument('--max-gen', type=int, default=10000)
    ap.add_argument('--smooth', type=int, default=10,
                    help='Rolling-mean window for time-series plots (1 = no smoothing)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.organisms, args.max_gen)

    plot_behaviour_over_time(df, args.label, args.smooth, args.out)
    plot_survival_behaviour(df, args.label, args.out)
    plot_death_cause_over_time(df, args.label, args.smooth, args.out)
    print_summary(df)

    print(f"  Done. Plots in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()