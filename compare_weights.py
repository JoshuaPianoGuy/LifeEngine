"""
compare_weights.py  —  LifeEngine Baldwin Effect Comparative Analysis
======================================================================
Joint-fits PCA and t-SNE across two distinct simulation conditions (e.g. GA vs GA+RL) 
to ensure spatial coordinates are strictly comparable. Eliminates independent-run 
artifacts and utilizes Epoch-sampling for t-SNE density.

Usage
-----
    python compare_weights.py \
        --org1 path/to/ga/organisms.csv \
        --gen1 path/to/ga/generations.csv \
        --label1 "Pure GA" \
        --org2 path/to/rl/organisms.csv \
        --gen2 path/to/rl/generations.csv \
        --label2 "GA + RL" \
        --out figures/comparison

Optional flags
--------------
    --epochs       Three generations to sample for t-SNE clades (default: 1 350 700)
    --tsne-top-k   Number of top organisms to sample per epoch for density (default: 50)
    --pca-smooth   Arrow spacing on PCA trajectory (default: 10)
"""

import argparse, base64, os, sys, warnings
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

# ── Weight decoding (from original) ──────────────────────────────────────────

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

def load_organisms(path):
    print(f"  Loading {path} ...")
    df = pd.read_csv(path)
    df['w1_arr'] = df['w1_weights'].apply(decode_w1)
    df = df[df['w1_arr'].apply(lambda x: x is not None and len(x) > 0)]
    return df

def generation_centroids(df):
    gens = sorted(df['generation'].unique())
    mats = [np.stack(df.loc[df['generation'] == g, 'w1_arr'].values).mean(axis=0) for g in gens]
    return np.array(gens), np.stack(mats)

# ── 1. Joint PCA Trajectory ──────────────────────────────────────────────────

def plot_joint_pca(df1, label1, df2, label2, smooth=10, out_dir='.'):
    print("  Fitting Joint PCA ...")
    g1, c1 = generation_centroids(df1)
    g2, c2 = generation_centroids(df2)
    
    # Fit ONE model to the combined centroid space
    combined_centroids = np.vstack([c1, c2])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined_centroids)
    
    # Transform both datasets into the shared space
    coords1 = pca.transform(c1)
    coords2 = pca.transform(c2)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    
    def plot_traj(coords, gens, cmap_name, marker, label):
        n = len(coords)
        colours = cm.get_cmap(cmap_name)(np.linspace(0.3, 0.9, max(n - 1, 1)))
        segments = [[coords[i], coords[i+1]] for i in range(n - 1)]
        lc = LineCollection(segments, colors=colours, linewidths=2, zorder=2, alpha=0.8)
        ax.add_collection(lc)

        step = max(1, smooth)
        for i in range(0, n - step, step):
            ax.annotate('', xy=coords[i+step], xytext=coords[i],
                        arrowprops=dict(arrowstyle='->', color=colours[i], lw=1.5), zorder=3)
        
        # Start and End markers
        ax.scatter(*coords[0], s=150, color=colours[0], edgecolors='black', zorder=5, marker=marker, label=f'{label} (Gen {gens[0]})')
        ax.scatter(*coords[-1], s=150, color=colours[-1], edgecolors='black', zorder=5, marker='*', label=f'{label} (Gen {gens[-1]})')

    plot_traj(coords1, g1, 'Blues', 'o', label1)
    plot_traj(coords2, g2, 'Oranges', '^', label2)

    ax.set_xlabel(f'Shared PC1 ({var[0]:.1f}% var)', fontsize=12)
    ax.set_ylabel(f'Shared PC2 ({var[1]:.1f}% var)', fontsize=12)
    ax.set_title(f'Joint PCA Trajectory: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'joint_pca_trajectory.png'), dpi=150)
    plt.close()

# ── 2. Epoch-Based Joint t-SNE ───────────────────────────────────────────────

def plot_joint_tsne(df1, label1, df2, label2, epochs, top_k, out_dir='.'):
    print(f"  Fitting Joint t-SNE across Epochs {epochs} ...")
    
    # Extract dense populations for specific epochs
    def get_epoch_samples(df, condition):
        samples = []
        for e in epochs:
            sub = df[df['generation'] == e].sort_values('rank').head(top_k)
            for _, row in sub.iterrows():
                samples.append({'epoch': e, 'condition': condition, 'vec': row['w1_arr']})
        return samples

    data1 = get_epoch_samples(df1, label1)
    data2 = get_epoch_samples(df2, label2)
    combined = data1 + data2
    
    if len(combined) < 20:
        print("  Skipping t-SNE: Not enough organisms found for chosen epochs."); return

    vecs = np.stack([r['vec'] for r in combined])
    
    # perplexity scales with sample size, but caps out to prevent blurring
    perp = min(40, max(5, len(vecs) // 6))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    coords = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotting styles: Epoch = Color intensity, Condition = Marker Shape
    colors = {epochs[0]: '#94A3B8', epochs[1]: '#3B82F6', epochs[2]: '#1D4ED8'} # Slate -> Blue -> Deep Blue
    colors2 = {epochs[0]: '#FDBA74', epochs[1]: '#F97316', epochs[2]: '#9A3412'} # Peach -> Orange -> Rust

    for i, item in enumerate(combined):
        if item['condition'] == label1:
            ax.scatter(coords[i, 0], coords[i, 1], c=colors[item['epoch']], marker='o', s=40, alpha=0.7, edgecolors='none')
        else:
            ax.scatter(coords[i, 0], coords[i, 1], c=colors2[item['epoch']], marker='^', s=40, alpha=0.7, edgecolors='none')

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label=label1),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label=label2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[epochs[0]], markersize=8, label=f'Epoch {epochs[0]}'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[epochs[1]], markersize=8, label=f'Epoch {epochs[1]}'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[epochs[2]], markersize=8, label=f'Epoch {epochs[2]}')
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_title(f'Joint t-SNE Clades (Top {top_k} organisms per Epoch)', fontsize=14, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]) # Hide meaningless axes
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'joint_tsne_clades.png'), dpi=150)
    plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Joint comparative analysis for LifeEngine')
    parser.add_argument('--org1', required=True)
    parser.add_argument('--gen1', required=True)
    parser.add_argument('--label1', default='Condition 1')
    parser.add_argument('--org2', required=True)
    parser.add_argument('--gen2', required=True)
    parser.add_argument('--label2', default='Condition 2')
    parser.add_argument('--out', default='figures/comparison')
    parser.add_argument('--epochs', nargs=3, type=int, default=[1, 350, 700])
    parser.add_argument('--tsne-top-k', type=int, default=50)
    parser.add_argument('--pca-smooth', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n── Loading Data ──────────────────────────────────────────────────")
    org_df1 = load_organisms(args.org1)
    org_df2 = load_organisms(args.org2)

    print("\n── Generating Joint Figures ──────────────────────────────────────")
    plot_joint_pca(org_df1, args.label1, org_df2, args.label2, smooth=args.pca_smooth, out_dir=args.out)
    plot_joint_tsne(org_df1, args.label1, org_df2, args.label2, epochs=args.epochs, top_k=args.tsne_top_k, out_dir=args.out)

    print(f"\n── Done. Comparison figures saved in: {os.path.abspath(args.out)}\n")

if __name__ == '__main__':
    main()