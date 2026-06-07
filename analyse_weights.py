"""
analyse_weights.py
==================
Per-condition neural weight analysis for the LifeEngine Baldwin Effect study.

Produces four figures per condition run:
  1. PCA trajectory  — centroid path through weight space over generations
  2. t-SNE snapshot  — weight clusters sampled every N generations
  3. Genome drift    — cosine similarity between genome (w1) and active weights
                       over generations (RL condition only; shows Baldwin encoding)
  4. Summary panel   — early-life energy, fitness, genomic variance, weight magnitude

Usage
-----
Run on a single condition at a time:

    python analyse_weights.py --organisms organisms.csv --generations generations.csv

Optional flags:
    --label       "RL+GA"          # plot title / filename prefix  (default: inferred)
    --out         ./figures        # output directory               (default: .)
    --tsne-every  50               # sample every N generations     (default: 50)
    --pca-smooth  10               # rolling window for PCA arrows  (default: 10)
    --top-k       5                # only use top-k organisms/gen   (default: all)

Weight encoding
---------------
Handles both formats automatically:
  • Old format : space-separated float strings  (pre-binary-encoding runs)
  • New format : Base64-encoded binary
      w1_weights   → Int8 quantised, decode as np.int8 / 127
      active_weights → Float32, decode as np.float32
"""

import argparse
import base64
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')


# ── Colour palette (neutral, works for both conditions) ──────────────────────
PALETTE = {
    'traj':    '#2563EB',   # blue  – PCA trajectory
    'arrow':   '#1D4ED8',
    'tsne_lo': '#93C5FD',   # light blue – early generations
    'tsne_hi': '#1E3A8A',   # dark  blue – late  generations
    'drift':   '#7C3AED',   # purple – cosine similarity
    'energy':  '#059669',   # green  – early-life energy
    'fitness': '#DC2626',   # red    – fitness
    'var':     '#D97706',   # amber  – genomic variance
    'mag':     '#DB2777',   # pink   – weight magnitude
}


# ══════════════════════════════════════════════════════════════════════════════
# Weight decoding
# ══════════════════════════════════════════════════════════════════════════════

def _is_base64(s: str) -> bool:
    """Heuristic: base64 strings have no spaces and use only [A-Za-z0-9+/=]."""
    if not s or ' ' in s:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def decode_w1(field: str) -> np.ndarray | None:
    """
    Decode a w1_weights field to a float32 numpy array.
    Handles:
      • space-separated text floats  (old format)
      • Base64 Int8                  (new format: divide by 127)
    """
    if not isinstance(field, str) or field.strip() in ('', 'nan'):
        return None
    field = field.strip()
    if _is_base64(field):
        raw = base64.b64decode(field)
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 127.0
    else:
        try:
            return np.array(field.split(), dtype=np.float32)
        except ValueError:
            return None


def decode_active(field: str) -> np.ndarray | None:
    """
    Decode an active_weights field to a float32 numpy array.
    Handles:
      • space-separated text floats  (old format)
      • Base64 Float32               (new format)
    """
    if not isinstance(field, str) or field.strip() in ('', 'nan'):
        return None
    field = field.strip()
    if _is_base64(field):
        raw = base64.b64decode(field)
        return np.frombuffer(raw, dtype=np.float32).copy()
    else:
        try:
            return np.array(field.split(), dtype=np.float32)
        except ValueError:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_organisms(path: str, top_k: int | None = None) -> pd.DataFrame:
    """
    Load organisms.csv and decode weight fields into numpy arrays.
    Keeps only the top-k ranked organisms per generation if top_k is set.
    Drops rows where w1_weights could not be decoded.
    """
    print(f"  Loading {path} …")
    df = pd.read_csv(path)
    required = {'generation', 'rank', 'w1_weights'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: organisms.csv is missing columns: {missing}")

    if top_k is not None:
        df = df[df['rank'] <= top_k].copy()

    print(f"  Decoding w1_weights …")
    df['w1_arr']     = df['w1_weights'].apply(decode_w1)
    df['active_arr'] = df['active_weights'].apply(decode_active) \
                       if 'active_weights' in df.columns else None

    n_before = len(df)
    df = df.dropna(subset=['w1_arr'])
    # dropna doesn't catch object columns with None — filter explicitly
    df = df[df['w1_arr'].apply(lambda x: x is not None and len(x) > 0)]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Warning: dropped {n_dropped} rows with undecodable w1_weights")

    # Verify all w1 vectors have the same length
    lengths = df['w1_arr'].apply(len).unique()
    if len(lengths) > 1:
        most_common = df['w1_arr'].apply(len).mode()[0]
        df = df[df['w1_arr'].apply(len) == most_common]
        print(f"  Warning: mixed w1 lengths {lengths}; kept length={most_common}")

    print(f"  Loaded {len(df)} organism rows across "
          f"{df['generation'].nunique()} generations")
    return df


def load_generations(path: str) -> pd.DataFrame:
    print(f"  Loading {path} …")
    df = pd.read_csv(path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Per-generation weight matrix helpers
# ══════════════════════════════════════════════════════════════════════════════

def generation_centroids(df: pd.DataFrame, col: str = 'w1_arr') -> tuple:
    """
    Returns (generations_sorted, centroid_matrix).
    centroid_matrix[i] is the mean weight vector for generation generations[i].
    """
    gens = sorted(df['generation'].unique())
    centroids = []
    for g in gens:
        vecs = np.stack(df.loc[df['generation'] == g, col].values)
        centroids.append(vecs.mean(axis=0))
    return np.array(gens), np.stack(centroids)


def cosine_similarity_per_gen(df: pd.DataFrame) -> tuple:
    """
    For each organism with both w1_arr and active_arr, compute cosine similarity.
    Returns (generations, mean_cosine_similarity_per_generation).
    Only meaningful for RL condition where active != genome.
    """
    sub = df[df['active_arr'].apply(
        lambda x: x is not None and len(x) > 0)].copy()
    if len(sub) == 0:
        return None, None

    def cos_sim(row):
        w1  = row['w1_arr']
        act = row['active_arr']
        # active_weights is the full genome; w1 is just the first W1_SIZE weights
        # compare against the corresponding slice of active
        n = min(len(w1), len(act))
        a, b = w1[:n], act[:n]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return np.nan
        return float(np.dot(a, b) / denom)

    sub['cos_sim'] = sub.apply(cos_sim, axis=1)
    result = sub.groupby('generation')['cos_sim'].mean()
    return result.index.values, result.values


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — PCA Trajectory
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_trajectory(df: pd.DataFrame, label: str,
                        smooth: int = 10, out_dir: str = '.'):
    print("  Plotting PCA trajectory …")
    gens, centroids = generation_centroids(df, 'w1_arr')

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(centroids)   # (n_gens, 2)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    # Colour segments by generation (dark = early, light = late)
    n = len(coords)
    colours = cm.viridis(np.linspace(0.05, 0.95, n - 1))
    segments = [[coords[i], coords[i+1]] for i in range(n - 1)]
    lc = LineCollection(segments, colors=colours, linewidths=1.4, zorder=2)
    ax.add_collection(lc)

    # Arrow every ~smooth generations to show direction
    step = max(1, smooth)
    for i in range(0, n - step, step):
        dx = coords[i+step, 0] - coords[i, 0]
        dy = coords[i+step, 1] - coords[i, 1]
        ax.annotate('', xy=coords[i+step], xytext=coords[i],
                    arrowprops=dict(arrowstyle='->', color=PALETTE['arrow'],
                                    lw=1.2, mutation_scale=12),
                    zorder=3)

    # Start / end markers
    ax.scatter(*coords[0],  s=120, color='#16A34A', zorder=5,
               label=f'Gen {gens[0]}',  marker='o')
    ax.scatter(*coords[-1], s=120, color='#DC2626',  zorder=5,
               label=f'Gen {gens[-1]}', marker='s')

    # Colourbar
    sm = cm.ScalarMappable(cmap='viridis',
                            norm=plt.Normalize(gens[0], gens[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=10)

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% var)', fontsize=11)
    ax.set_title(f'PCA Trajectory of W1 Genome Weights\n{label}', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal', 'datalim')
    fig.tight_layout()

    fname = os.path.join(out_dir, f'{_slug(label)}_pca_trajectory.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — t-SNE snapshot
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne(df: pd.DataFrame, label: str,
              every_n: int = 50, out_dir: str = '.'):
    print(f"  Plotting t-SNE (every {every_n} gens) …")
    all_gens = sorted(df['generation'].unique())

    # Sample one representative organism per generation (rank-1) at every_n intervals
    sampled_gens = [g for g in all_gens if g % every_n == 0 or g == all_gens[-1]]
    if len(sampled_gens) < 2:
        sampled_gens = all_gens  # too few gens — use all

    rows = []
    for g in sampled_gens:
        sub = df[df['generation'] == g].sort_values('rank')
        if len(sub) == 0:
            continue
        # Use rank-1 organism (best of that generation)
        rows.append({'generation': g, 'vec': sub.iloc[0]['w1_arr']})

    if len(rows) < 5:
        print("  Skipping t-SNE: too few sample points")
        return

    vecs = np.stack([r['vec'] for r in rows])
    gen_labels = np.array([r['generation'] for r in rows])

    perplexity = min(30, max(5, len(rows) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000, init='pca')
    coords = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=gen_labels, cmap='viridis', s=60,
                    alpha=0.85, edgecolors='none', zorder=2)

    # Connect points in generation order to show trajectory
    order = np.argsort(gen_labels)
    ax.plot(coords[order, 0], coords[order, 1],
            color='#94A3B8', lw=0.6, alpha=0.5, zorder=1)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=10)

    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title(
        f't-SNE of Rank-1 W1 Genome Weights (sampled every {every_n} gens)\n{label}',
        fontsize=13)
    fig.tight_layout()

    fname = os.path.join(out_dir, f'{_slug(label)}_tsne.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Genome–Active cosine drift  (RL condition)
# ══════════════════════════════════════════════════════════════════════════════

def plot_genome_drift(df: pd.DataFrame, label: str, out_dir: str = '.'):
    print("  Plotting genome–active cosine similarity …")
    gens, sims = cosine_similarity_per_gen(df)
    if gens is None:
        print("  Skipping genome drift plot: no active_weights data")
        return

    # Check whether active ≠ genome (RL condition)
    # In GA-only condition active_weights IS genome_weights; cosine sim = 1.0 always
    if np.nanstd(sims) < 1e-4:
        print("  Skipping genome drift: active_weights identical to w1 "
              "(GA-only condition — no within-lifetime learning)")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, sims, color=PALETTE['drift'], lw=1.2, alpha=0.7, label='per gen')
    # Smoothed trend
    w = max(1, len(gens) // 40)
    smoothed = np.convolve(sims, np.ones(w)/w, mode='valid')
    offset = (len(sims) - len(smoothed)) // 2
    ax.plot(gens[offset:offset+len(smoothed)], smoothed,
            color=PALETTE['drift'], lw=2.5, label=f'{w}-gen rolling mean')

    ax.axhline(1.0, color='#6B7280', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Cosine Similarity (genome W1 vs active W1)', fontsize=10)
    ax.set_title(
        'Baldwin Encoding Signal: W1 Genome → Active Weight Alignment\n'
        f'{label}  |  Rising trend = genome converging toward learned solution',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=max(0, np.nanmin(sims) - 0.05))
    fig.tight_layout()

    fname = os.path.join(out_dir, f'{_slug(label)}_genome_drift.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Summary panel (from generations.csv)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(gen_df: pd.DataFrame, label: str, out_dir: str = '.'):
    print("  Plotting summary panel …")
    g = gen_df.sort_values('generation')
    gens = g['generation'].values

    has_early_energy = ('avg_energy_at_tick_1000' in g.columns and
                        g['avg_energy_at_tick_1000'].replace(0, np.nan).notna().any())

    n_rows = 3 if has_early_energy else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    def smooth(arr, w=20):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    ax_idx = 0

    # ── Fitness: best, top-20%, average ──────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(gens, g['best_fitness'], color='#DC2626', lw=0.8, alpha=0.4)
    ax.plot(gens, smooth(g['best_fitness'].values), color='#DC2626', lw=2,
            label='Best')
    ax.plot(gens, g['top20percent_fitness'], color='#7C3AED', lw=0.8, alpha=0.4)
    ax.plot(gens, smooth(g['top20percent_fitness'].values), color='#7C3AED', lw=2,
            label='Top 20%')
    ax.plot(gens, g['avg_fitness'], color='#D97706', lw=0.8, alpha=0.4)
    ax.plot(gens, smooth(g['avg_fitness'].values), color='#D97706', lw=2,
            label='Average')
    ax.set_title('Fitness Progression', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness Score')
    ax.legend(fontsize=8)

    # ── Genomic variance ─────────────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(gens, g['genome_variance'], color=PALETTE['var'], lw=0.8, alpha=0.4)
    ax.plot(gens, smooth(g['genome_variance'].values), color=PALETTE['var'],
            lw=2.5, label='Genome Variance')
    ax.set_title('Genomic Variance', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean Per-Position Variance')
    ax.legend(fontsize=8)

    # ── Weight magnitude & inter-gen change ──────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    ax2 = ax.twinx()
    ax.plot(gens, g['avg_network_weight_mag'], color=PALETTE['mag'],
            lw=1.5, label='Weight Magnitude')
    ax2.plot(gens, g['inter_gen_weight_change'], color='#94A3B8',
             lw=0.8, alpha=0.6, label='Inter-Gen Change')
    ax2.axhline(0, color='#94A3B8', lw=0.5, ls='--')
    ax.set_title('Weight Magnitude & Inter-Gen Change', fontsize=11)
    ax.set_xlabel('Generation')
    ax.set_ylabel('RMS Weight Magnitude', color=PALETTE['mag'])
    ax2.set_ylabel('Inter-Gen ΔMagnitude', color='#94A3B8')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    # ── Population ───────────────────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(gens, g['total_agents'], color='#92400E', lw=0.8, alpha=0.4)
    ax.plot(gens, smooth(g['total_agents'].values), color='#92400E',
            lw=2, label='Total Agents')
    ax.plot(gens, g['peak_population'], color='#0891B2', lw=0.8,
            alpha=0.4, ls='--')
    ax.plot(gens, smooth(g['peak_population'].values), color='#0891B2',
            lw=2, ls='--', label='Peak Pop')
    ax.set_title('Population Dynamics', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Agent Count')
    ax.legend(fontsize=8)

    # ── Early-life energy (Baldwin signal) ───────────────────────────────────
    if has_early_energy and ax_idx < len(axes):
        ax = axes[ax_idx]; ax_idx += 1
        early = g['avg_energy_at_tick_1000'].replace(0, np.nan)
        ax.plot(gens, early, color=PALETTE['energy'], lw=0.8, alpha=0.4)
        ax.plot(gens, smooth(early.ffill().values),
                color=PALETTE['energy'], lw=2.5,
                label='Avg Energy @ tick 2000')
        ax.set_title('Early-Life Energy (Baldwin Encoding Signal)', fontsize=11)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Energy at Tick 2000')
        ax.legend(fontsize=8)
        note = ('Rising trend = organisms eating effectively early in life\n'
                'without needing to learn it → behavioural encoding')
        ax.text(0.98, 0.05, note, transform=ax.transAxes,
                fontsize=7.5, ha='right', va='bottom', color='#374151',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    # ── Learned weight diff (RL condition) ───────────────────────────────────
    if ax_idx < len(axes):
        ax = axes[ax_idx]; ax_idx += 1
        if 'avg_learned_weight_diff' in g.columns:
            diff = g['avg_learned_weight_diff']
            if diff.abs().max() > 1e-6:
                ax.plot(gens, diff, color='#2563EB', lw=0.8, alpha=0.4)
                ax.plot(gens, smooth(diff.values), color='#2563EB',
                        lw=2.5, label='Learned Weight Diff')
                ax.set_title('Within-Lifetime RL Weight Drift', fontsize=11)
                ax.set_xlabel('Generation')
                ax.set_ylabel('Mean |active − genome|')
                ax.legend(fontsize=8)
                note = 'Non-zero only in RL condition'
                ax.text(0.98, 0.05, note, transform=ax.transAxes,
                        fontsize=7.5, ha='right', va='bottom', color='#374151',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6',
                                  ec='none'))
            else:
                ax.text(0.5, 0.5,
                        'No within-lifetime learning\n(GA-only condition)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='#6B7280')
                ax.set_title('Within-Lifetime RL Weight Drift', fontsize=11)

    # Hide any unused axes
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Summary — {label}', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    fname = os.path.join(out_dir, f'{_slug(label)}_summary.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — PCA variance-explained over generations
# (How quickly the population compresses onto a low-dimensional manifold)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_variance_over_time(df: pd.DataFrame, label: str,
                                 n_components: int = 5, out_dir: str = '.'):
    print("  Plotting PCA variance-explained over time …")
    gens = sorted(df['generation'].unique())
    results = []

    for g in gens:
        vecs = np.stack(df.loc[df['generation'] == g, 'w1_arr'].values)
        if len(vecs) < n_components + 1:
            continue
        pca = PCA(n_components=min(n_components, len(vecs) - 1), random_state=42)
        pca.fit(vecs)
        results.append({
            'gen': g,
            'var_total': pca.explained_variance_ratio_.sum(),
            **{f'pc{i+1}': pca.explained_variance_ratio_[i]
               for i in range(len(pca.explained_variance_ratio_))}
        })

    if not results:
        print("  Skipping: insufficient data per generation")
        return

    res = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 4))

    colours = cm.cool(np.linspace(0.1, 0.9, n_components))
    for i in range(n_components):
        col = f'pc{i+1}'
        if col not in res:
            break
        ax.plot(res['gen'], res[col], color=colours[i], lw=1.5,
                label=f'PC{i+1}')

    ax.plot(res['gen'], res['var_total'], color='#1F2937', lw=2.5,
            ls='--', label=f'Total (PC1–{n_components})')

    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel(f'Variance Explained (top {n_components} PCs)', fontsize=11)
    ax.set_title(
        f'Population Compression: PCA Variance Explained Over Generations\n'
        f'{label}  |  Faster drop = population converging onto fewer dimensions',
        fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    fname = os.path.join(out_dir, f'{_slug(label)}_pca_variance.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def _slug(s: str) -> str:
    return s.lower().replace(' ', '_').replace('+', '_').replace('/', '_')


def infer_label(gen_df: pd.DataFrame) -> str:
    if 'condition' in gen_df.columns:
        cond = gen_df['condition'].iloc[0]
        if cond == 'learning':
            return 'RL+GA'
        elif cond in ('natural_selection', 'evolution'):
            return 'GA only'
    return 'condition'


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Per-condition neural weight analysis for LifeEngine Baldwin Effect study')
    parser.add_argument('--organisms',    required=True,
                        help='Path to organisms.csv')
    parser.add_argument('--generations',  required=True,
                        help='Path to generations.csv')
    parser.add_argument('--label',        default=None,
                        help='Condition label for plot titles (inferred if omitted)')
    parser.add_argument('--out',          default='.',
                        help='Output directory for figures')
    parser.add_argument('--tsne-every',   type=int, default=50,
                        help='Sample rank-1 organism every N generations for t-SNE')
    parser.add_argument('--pca-smooth',   type=int, default=10,
                        help='Arrow spacing (generations) on PCA trajectory plot')
    parser.add_argument('--top-k',        type=int, default=None,
                        help='Restrict to top-k ranked organisms per generation')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────────────")
    org_df = load_organisms(args.organisms, top_k=args.top_k)
    gen_df = load_generations(args.generations)

    label = args.label or infer_label(gen_df)
    print(f"  Condition label: {label}")

    print("\n── Generating figures ───────────────────────────────────────────")
    plot_pca_trajectory(org_df, label,
                        smooth=args.pca_smooth, out_dir=args.out)
    plot_tsne(org_df, label,
              every_n=args.tsne_every, out_dir=args.out)
    plot_genome_drift(org_df, label, out_dir=args.out)
    plot_pca_variance_over_time(org_df, label, out_dir=args.out)
    plot_summary(gen_df, label, out_dir=args.out)

    print(f"\n── Done. Figures written to: {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()