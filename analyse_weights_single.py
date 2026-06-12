"""
analyse_weights_single.py
======================================================================
Per-condition comprehensive plotting suite for LifeEngine.
Processes a single simulation run at a time with descriptive, interpretable axes.

Generates:
  1. execution_summary.png — 7-panel isolated diagnostics grid
  2. pca_trajectory.png     — Macro path through weight space with loading drivers
  3. tsne_clades.png        — Interval-sampled density clades with topological labels

Usage
-----
    python analyse_weights_single.py \
        --organisms path/to/organisms.csv \
        --generations path/to/generations.csv \
        --label "GA + RL Learning Condition" \
        --tsne-step 200 \
        --max-gen 3000 \
        --out output/plots
"""

import argparse
import base64
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ── Weight Decoding Helpers ──────────────────────────────────────────────────

def _is_base64(s):
    if not s or ' ' in s: return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception: return False

def decode_w1(field):
    if not isinstance(field, str) or field.strip() in ('', 'nan'): return None
    field = field.strip()
    if _is_base64(field):
        raw = base64.b64decode(field)
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 127.0
    try: return np.array(field.split(), dtype=np.float32)
    except ValueError: return None

def load_organisms(path, max_gen):
    print(f"  Parsing organisms dataset from {path} ...")
    df = pd.read_csv(path)
    df = df[df['generation'] <= max_gen]
    df['w1_arr'] = df['w1_weights'].apply(decode_w1)
    df = df[df['w1_arr'].apply(lambda x: x is not None and len(x) > 0)]
    return df

def generation_centroids(df):
    gens = sorted(df['generation'].unique())
    mats = [np.stack(df.loc[df['generation'] == g, 'w1_arr'].values).mean(axis=0) for g in gens]
    return np.array(gens), np.stack(mats)

# ── 1. Historical Summary Plot ───────────────────────────────────────────────

def plot_exact_summary_panel(gen_path, label, max_gen, out_dir='.'):
    print("  Generating 7-panel line metrics canvas ...")
    g = pd.read_csv(gen_path)
    g = g[g['generation'] <= max_gen]
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    
    # Panel 1: Fitness Performance
    axs[0, 0].plot(g['generation'], g['best_fitness'], color='#1D4ED8', linestyle='-', lw=1.2, alpha=0.4, label='Best Fitness')
    axs[0, 0].plot(g['generation'], g['top20percent_fitness'], color='#1D4ED8', linestyle='--', lw=1.8, label='Top 20% Fittest')
    axs[0, 0].plot(g['generation'], g['avg_fitness'], color='#1D4ED8', linestyle=':', lw=2.2, label='Average Fitness')
    axs[0, 0].set_title('Fitness Performance History', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Absolute Fitness')
    axs[0, 0].legend(loc='upper left', fontsize=9)
    axs[0, 0].grid(True, alpha=0.3)

    # Panel 2: Population Growth Dynamics
    axs[0, 1].plot(g['generation'], g['total_agents'], color='#2563EB', linestyle='-', lw=2, label='Total Agent Count')
    axs[0, 1].plot(g['generation'], g['peak_population'], color='#DB2777', linestyle='--', lw=2, label='Peak Population')
    axs[0, 1].set_title('Population Growth Dynamics', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Organism Counts')
    axs[0, 1].legend(loc='upper left', fontsize=9)
    axs[0, 1].grid(True, alpha=0.3)

    # Panel 3: Average Lifespan
    axs[0, 2].plot(g['generation'], g['avg_lifetime'], color='#D97706', lw=2)
    axs[0, 2].set_title('Average Lifespan per Generation', fontsize=12, fontweight='bold')
    axs[0, 2].set_ylabel('Ticks Survived')
    axs[0, 2].grid(True, alpha=0.3)

    # Panel 4: Within-Lifetime Weight Drift (Plasticity)
    if 'avg_learned_weight_diff' in g.columns and g['avg_learned_weight_diff'].max() > 0:
        axs[1, 0].plot(g['generation'], g['avg_learned_weight_diff'], color='#059669', lw=2)
        axs[1, 0].set_title('Within-Lifetime Weight Drift (Plasticity)', fontsize=12, fontweight='bold')
        axs[1, 0].set_ylabel('Mean Absolute Shift (ΔW_life)')
    else:
        axs[1, 0].text(0.5, 0.5, 'Plasticity Inactive\n(Pure Genetic Control)', 
                       ha='center', va='center', fontsize=11, color='gray', transform=axs[1, 0].transAxes)
        axs[1, 0].set_title('Within-Lifetime Weight Drift', fontsize=12, fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3)

    # Panel 5: Baseline Network Weight Magnitude
    axs[1, 1].plot(g['generation'], g['avg_network_weight_mag'], color='#7C3AED', lw=2)
    axs[1, 1].set_title('Average Network Weight Magnitude', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Structural Scale Values')
    axs[1, 1].grid(True, alpha=0.3)

    # Panel 6: Inter-generational Structural Weight Step (Velocity)
    axs[1, 2].plot(g['generation'], g['inter_gen_weight_change'], color='#EA580C', lw=2)
    axs[1, 2].set_title('Inter-Generational Weight Step (Velocity)', fontsize=12, fontweight='bold')
    axs[1, 2].set_ylabel('Evolutionary Step Delta (ΔW_gen)')
    axs[1, 2].grid(True, alpha=0.3)

    # Panel 7: Structural Genomic Variance
    axs[2, 0].plot(g['generation'], g['genome_variance'], color='#DC2626', lw=2)
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
    fig.savefig(os.path.join(out_dir, 'execution_summary_pure_rl_500_gen.png'), dpi=150)
    plt.close()

# ── 2. Interpretable PCA Trajectory with Feature Drivers ─────────────────────

def plot_pca_trajectory(df, label, smooth=10, out_dir='.'):
    print("  Calculating continuous PCA centroid path with loading weights ...")
    gens, centroids = generation_centroids(df)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(centroids)
    var = pca.explained_variance_ratio_ * 100
    
    # Extract loading vectors (eigenvectors) to see which neural weights drive the axes
    loadings = pca.components_
    n_weights = loadings.shape[1]
    k_drivers = min(3, n_weights)
    
    # Sort weight indices by absolute influence on PC1 and PC2
    top_pc1_indices = np.argsort(np.abs(loadings[0]))[-k_drivers:][::-1]
    top_pc2_indices = np.argsort(np.abs(loadings[1]))[-k_drivers:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    n_points = len(coords)
    
    colours = cm.plasma(np.linspace(0.1, 0.95, max(n_points - 1, 1)))
    segments = [[coords[i], coords[i+1]] for i in range(n_points - 1)]
    lc = LineCollection(segments, colors=colours, linewidths=2.5, zorder=2)
    ax.add_collection(lc)

    step = max(1, smooth)
    for i in range(0, n_points - step, step):
        ax.annotate('', xy=coords[i+step], xytext=coords[i],
                    arrowprops=dict(arrowstyle='->', color=colours[i], lw=1.5), zorder=3)
        
    ax.scatter(*coords[0], s=140, color=colours[0], edgecolors='black', zorder=5, label=f'Start (Gen {gens[0]})')
    ax.scatter(*coords[-1], s=180, color=colours[-1], edgecolors='black', zorder=5, marker='*', label=f'End (Gen {gens[-1]})')

    # Highly descriptive structural axis labeling
    ax.set_xlabel(
        f'PC1 ({var[0]:.1f}% Variance Explained)\n'
        f'➔ Direction of Maximum Population-Wide Weight Divergence Across the Run', 
        fontsize=11, fontweight='semibold', labelpad=10
    )
    ax.set_ylabel(
        f'PC2 ({var[1]:.1f}% Variance Explained)\n'
        f'➔ Secondary Orthogonal Axis of Independent Synaptic Variation', 
        fontsize=11, fontweight='semibold', labelpad=10
    )
    ax.set_title(f'Weight Space PCA Centroid Path\nCondition: {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Format structural metadata text box describing component meaning
    driver_box_text = "Component Structural Interpretation\n" + "─"*36 + "\n"
    driver_box_text += "PC1 Primary Synaptic Drivers:\n"
    for idx in top_pc1_indices:
        driver_box_text += f"  • Connection W[{idx}]: Loading = {loadings[0, idx]:+.3f}\n"
    driver_box_text += "\nPC2 Primary Synaptic Drivers:\n"
    for idx in top_pc2_indices:
        driver_box_text += f"  • Connection W[{idx}]: Loading = {loadings[1, idx]:+.3f}\n"
    driver_box_text += "\n" + "─"*36 + "\n"
    driver_box_text += "Interpretation:\nOrganisms moving down these\nvectors are selectively scaling\nthe connection IDs noted above."

    # Affix metadata box cleanly onto the side margins
    ax.text(
        1.02, 0.5, driver_box_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95)
    )
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_trajectory_pure_rl_500_gen.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ── 3. Interpretable Step t-SNE Density Clades ───────────────────────────────

def plot_tsne_clades(df, label, step, max_gen, top_k, out_dir='.'):
    all_gens = sorted(df['generation'].unique())
    epochs = [g for g in all_gens if g % step == 0 or g == 1 or g == max_gen]
    epochs = sorted(list(set(epochs)))
    
    print(f"  Projecting t-SNE structural clades for target eras: {epochs} ...")
    
    samples = []
    for e in epochs:
        sub = df[df['generation'] == e].sort_values('rank').head(top_k)
        for _, row in sub.iterrows():
            samples.append({'epoch': e, 'vec': row['w1_arr']})
            
    if len(samples) < 10:
        print("  Skipping t-SNE plot: Insufficient generation matches found."); return

    vecs = np.stack([s['vec'] for s in samples])
    perp = min(40, max(5, len(vecs) // 6))
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    coords = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = cm.get_cmap('viridis')
    norm_vals = np.linspace(0.0, 0.9, len(epochs))
    color_map = {epochs[idx]: cmap(val) for idx, val in enumerate(norm_vals)}

    for i, item in enumerate(samples):
        ax.scatter(coords[i, 0], coords[i, 1], c=[color_map[item['epoch']]], s=45, alpha=0.75, edgecolors='none')

    legend_elements = []
    for ep in epochs:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color_map[ep], markersize=9, label=f'Gen {ep}'))

    ax.legend(handles=legend_elements, loc='best', title="Evolutionary Eras", fontsize=9, title_fontsize=10)
    ax.set_title(f'Population t-SNE Clades over Time\nCondition: {label}', fontsize=13, fontweight='bold')
    
    # Highly descriptive topology axis labeling (explaining arbitrary spatial units)
    ax.set_xlabel(
        "t-SNE Axis 1 (Relative Spatial Topology Units)\n"
        "➔ Spatial closeness maps high-dimensional similarity. Clustered islands share identical neural strategies.", 
        fontsize=10.5, fontweight='semibold', labelpad=10
    )
    ax.set_ylabel(
        "t-SNE Axis 2 (Relative Spatial Topology Units)\n"
        "➔ Distances indicate topological divergence. Isolated islands represent unique functional network clades.", 
        fontsize=10.5, fontweight='semibold', labelpad=10
    )
    
    # Hide standard numeric tick values since coordinates are purely relational
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tsne_clades_pure_rl_500_gen.png'), dpi=150)
    plt.close()

# ── Main Execution ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Descriptive Single Run Analysis Suite')
    parser.add_argument('--organisms', required=True, help='Path to organisms.csv')
    parser.add_argument('--generations', required=True, help='Path to generations.csv')
    parser.add_argument('--label', default='Simulation Run', help='Plot title flag')
    parser.add_argument('--out', default='.', help='Output directory workspace')
    parser.add_argument('--tsne-step', type=int, default=200, help='Generations interval snapshot step')
    parser.add_argument('--max-gen', type=int, default=3000, help='Hard limit generation clamp threshold')
    parser.add_argument('--tsne-top-k', type=int, default=30, help='Top fittest entities to parse per slice group')
    parser.add_argument('--pca-smooth', type=int, default=10, help='Arrow gap step size for PCA plot path')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n── Processing Diagnostic Extraction: {args.label} ──")
    plot_exact_summary_panel(args.generations, args.label, max_gen=args.max_gen, out_dir=args.out)
    
    org_df = load_organisms(args.organisms, max_gen=args.max_gen)
    plot_pca_trajectory(org_df, args.label, smooth=args.pca_smooth, out_dir=args.out)
    plot_tsne_clades(org_df, args.label, step=args.tsne_step, max_gen=args.max_gen, top_k=args.tsne_top_k, out_dir=args.out)

    print(f"\n── Complete. Graphics successfully saved to: {os.path.abspath(args.out)}\n")

if __name__ == '__main__':
    main()