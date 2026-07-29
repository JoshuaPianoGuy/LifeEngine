"""
analyse_weights.py  —  LifeEngine Baldwin Effect analysis
==========================================================
Per-condition neural weight analysis. Run once per condition; overlay the
resulting figures manually when the second condition is complete.

Figures produced
----------------
  1. pca_trajectory     — centroid path through W1 weight space, coloured by gen
  2. tsne               — all top-20% organisms from sampled generations clustered
  3. weight_drift       — MEAN ABSOLUTE DIFFERENCE between genome W1 and active W1
                          per generation (the real Baldwin encoding signal, not cosine)
  4. pca_variance       — rolling-smoothed variance explained by top-5 PCs over time
  5. summary            — fitness / genomic variance / weight magnitude / population
                          from generations.csv, with early-life energy if non-zero

Usage
-----
    python analyse_weights.py \
        --organisms  path/to/organisms.csv \
        --generations path/to/generations.csv \
        --label "RL+GA" \
        --out figures/rl

Optional flags
--------------
    --tsne-every  N    sample every N generations for t-SNE  (default 25)
    --tsne-orgs   K    use top-K organisms per sampled gen   (default 5)
    --pca-smooth  N    arrow spacing on PCA trajectory       (default 10)
    --top-k       K    restrict to top-K ranks overall       (default: all)

Weight encoding (auto-detected)
--------------------------------
  Old format  : space-separated float strings
  New format  : w1_weights = Base64 Int8 (/127), active_weights = Base64 Float32
"""

import argparse, base64, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

PALETTE = {
    'traj':   '#2563EB',
    'arrow':  '#1D4ED8',
    'drift':  '#7C3AED',
    'energy': '#059669',
    'var':    '#D97706',
    'mag':    '#DB2777',
}


# ── Weight decoding ───────────────────────────────────────────────────────────

def _is_base64(s):
    if not s or ' ' in s:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def decode_w1(field):
    if not isinstance(field, str) or field.strip() in ('', 'nan'):
        return None
    field = field.strip()
    if _is_base64(field):
        raw = base64.b64decode(field)
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 127.0
    try:
        return np.array(field.split(), dtype=np.float32)
    except ValueError:
        return None

def decode_active(field):
    if not isinstance(field, str) or field.strip() in ('', 'nan'):
        return None
    field = field.strip()
    if _is_base64(field):
        raw = base64.b64decode(field)
        return np.frombuffer(raw, dtype=np.float32).copy()
    try:
        return np.array(field.split(), dtype=np.float32)
    except ValueError:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_organisms(path, top_k=None):
    print(f"  Loading {path} ...")
    df = pd.read_csv(path)
    missing = {'generation', 'rank', 'w1_weights'} - set(df.columns)
    if missing:
        sys.exit(f"ERROR: organisms.csv missing columns: {missing}")
    if top_k is not None:
        df = df[df['rank'] <= top_k].copy()

    print(f"  Decoding w1_weights ...")
    df['w1_arr'] = df['w1_weights'].apply(decode_w1)

    has_active = 'active_weights' in df.columns
    if has_active:
        print(f"  Decoding active_weights ...")
        df['active_arr'] = df['active_weights'].apply(decode_active)
    else:
        df['active_arr'] = None

    n_before = len(df)
    df = df[df['w1_arr'].apply(lambda x: x is not None and len(x) > 0)]
    if n_before - len(df):
        print(f"  Warning: dropped {n_before - len(df)} rows with undecodable w1")

    lengths = df['w1_arr'].apply(len).unique()
    if len(lengths) > 1:
        mc = df['w1_arr'].apply(len).mode()[0]
        df = df[df['w1_arr'].apply(len) == mc]
        print(f"  Warning: mixed w1 lengths {lengths}; kept {mc}")

    print(f"  Loaded {len(df)} rows across {df['generation'].nunique()} generations")
    return df

def load_generations(path):
    print(f"  Loading {path} ...")
    return pd.read_csv(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def generation_centroids(df, col='w1_arr'):
    gens = sorted(df['generation'].unique())
    mats = []
    for g in gens:
        vecs = np.stack(df.loc[df['generation'] == g, col].values)
        mats.append(vecs.mean(axis=0))
    return np.array(gens), np.stack(mats)

def rolling(arr, w):
    if len(arr) < w:
        return arr, np.arange(len(arr))
    kernel = np.ones(w) / w
    out = np.convolve(arr, kernel, mode='valid')
    idx = np.arange(len(arr) - w + 1) + w // 2
    return out, idx

def _slug(s):
    return s.lower().replace(' ', '_').replace('+', '_').replace('/', '_')

def infer_label(gen_df):
    if 'condition' in gen_df.columns:
        c = gen_df['condition'].iloc[0]
        if c == 'learning':   return 'RL+GA'
        if c in ('natural_selection', 'evolution'): return 'GA only'
    return 'condition'


# ── Figure 1: PCA trajectory ──────────────────────────────────────────────────

def plot_pca_trajectory(df, label, smooth=10, out_dir='.', pca_ref=None):
    """
    Returns (pca, gens, coords) so the caller can pass pca_ref to the second
    condition and project both into the same space for cross-condition comparison.
    """
    print("  Plotting PCA trajectory ...")
    gens, centroids = generation_centroids(df, 'w1_arr')
    if len(gens) < 3:
        print("  Skipping: need ≥3 generations"); return None, None, None

    if pca_ref is not None:
        pca    = pca_ref
        coords = pca.transform(centroids)
    else:
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(centroids)
    var = pca.explained_variance_ratio_ * 100

    # Arc-length speed: Euclidean distance the centroid moves each generation
    steps    = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(steps)])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [1.4, 1]})

    # Left — trajectory map
    ax = axes[0]
    n = len(coords)
    colours = cm.viridis(np.linspace(0.05, 0.95, max(n - 1, 1)))
    segments = [[coords[i], coords[i+1]] for i in range(n - 1)]
    lc = LineCollection(segments, colors=colours, linewidths=1.4, zorder=2)
    ax.add_collection(lc)

    step = max(1, smooth)
    for i in range(0, n - step, step):
        ax.annotate('', xy=coords[i+step], xytext=coords[i],
                    arrowprops=dict(arrowstyle='->', color=PALETTE['arrow'],
                                   lw=1.2, mutation_scale=12), zorder=3)

    ax.scatter(*coords[0],  s=120, color='#16A34A', zorder=5,
               label=f'Gen {gens[0]}',  marker='o')
    ax.scatter(*coords[-1], s=120, color='#DC2626',  zorder=5,
               label=f'Gen {gens[-1]}', marker='s')

    sm = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(gens[0], gens[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=10)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% var)', fontsize=11)
    ax.set_title(f'PCA Trajectory of W1 Genome Weights\n{label}', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal', 'datalim')

    # Right — arc-length speed + cumulative distance
    ax2  = axes[1]
    ax2b = ax2.twinx()

    w = max(5, len(steps) // 30)
    spd_sm, spd_idx = rolling(steps, w)
    ax2.plot(gens[1:][spd_idx], spd_sm,
             color=PALETTE['traj'], lw=2, label=f'Speed ({w}-gen mean)')
    ax2.fill_between(gens[1:][spd_idx], 0, spd_sm,
                     color=PALETTE['traj'], alpha=0.15)
    ax2b.plot(gens, cum_dist, color='#94A3B8', lw=1.5, ls='--',
              label='Cumulative distance')

    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Arc-length per generation (PCA units)', color=PALETTE['traj'],
                   fontsize=10)
    ax2b.set_ylabel('Cumulative distance', color='#94A3B8', fontsize=10)
    ax2.set_title('Trajectory Speed Through Weight Space\n'
                  '(use pca_ref= to compare on same axes)', fontsize=11)
    lines  = [plt.Line2D([0],[0], color=PALETTE['traj'], lw=2),
              plt.Line2D([0],[0], color='#94A3B8', lw=1.5, ls='--')]
    labels = [f'Speed ({w}-gen mean)', 'Cumulative distance']
    ax2.legend(lines, labels, fontsize=9)
    note = (f'Total distance gen {gens[0]}→{gens[-1]}: {cum_dist[-1]:.2f} PCA units\n'
            f'Mean speed: {steps.mean():.3f}  |  Max: {steps.max():.3f}')
    ax2.text(0.98, 0.97, note, transform=ax2.transAxes, fontsize=8,
             ha='right', va='top', color='#374151',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    fig.tight_layout()
    fname = os.path.join(out_dir, f'{_slug(label)}_pca_trajectory.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")
    return pca, gens, coords


# ── Figure 2: t-SNE ───────────────────────────────────────────────────────────
# Uses ALL top-k organisms from sampled generations so t-SNE has enough points
# to form meaningful clusters (not just 15 dots from rank-1 only).

def plot_tsne(df, label, every_n=25, top_orgs=5, out_dir='.'):
    print(f"  Plotting t-SNE (every {every_n} gens, top {top_orgs} orgs/gen) ...")
    all_gens = sorted(df['generation'].unique())
    sampled = [g for g in all_gens if g % every_n == 0 or g == all_gens[-1]]
    if len(sampled) < 3:
        sampled = all_gens

    rows = []
    for g in sampled:
        sub = df[df['generation'] == g].sort_values('rank').head(top_orgs)
        for _, row in sub.iterrows():
            rows.append({'generation': g, 'vec': row['w1_arr']})

    n_pts = len(rows)
    if n_pts < 10:
        print(f"  Skipping t-SNE: only {n_pts} points, need ≥10"); return

    vecs      = np.stack([r['vec']       for r in rows])
    gen_labels = np.array([r['generation'] for r in rows])

    perplexity = min(30, max(5, n_pts // 5))
    print(f"  Running t-SNE on {n_pts} points (perplexity={perplexity}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000, init='pca')
    coords = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=gen_labels,
                    cmap='viridis', s=35, alpha=0.75, edgecolors='none', zorder=2)

    # Connect generation centroids to show directional trajectory
    unique_gens = sorted(set(gen_labels))
    cx = [coords[gen_labels == g, 0].mean() for g in unique_gens]
    cy = [coords[gen_labels == g, 1].mean() for g in unique_gens]
    ax.plot(cx, cy, color='#94A3B8', lw=1.0, alpha=0.6, zorder=1)
    ax.scatter(cx, cy, color='white', s=25, zorder=3, edgecolors='#64748B', lw=0.8)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=10)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title(
        f't-SNE: W1 Genome Weights — top {top_orgs} orgs every {every_n} gens\n{label}',
        fontsize=12)
    fig.tight_layout()
    fname = os.path.join(out_dir, f'{_slug(label)}_tsne.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Figure 3: Weight drift (mean absolute difference, not cosine) ─────────────
# Cosine similarity is insensitive to small magnitude changes in high-dimensional
# space and will always be ~1.0 for REINFORCE-scale updates (~0.009 mean diff
# across 1472 weights).  Mean absolute difference directly tracks how far RL
# moves the weights from the inherited starting point each generation, and whether
# that gap narrows over generations (encoding signal).

def plot_weight_drift(df, gen_df, label, out_dir='.'):
    print("  Plotting weight drift (genome vs active) ...")

    # Check active_weights are present and non-trivially different from w1
    has_active = df['active_arr'].apply(
        lambda x: x is not None and len(x) > 0).any()
    if not has_active:
        print("  Skipping: no active_weights data"); return

    # ── Per-organism MAD: genome W1 vs corresponding active slice ───────────
    # Use the full active_weights length aligned to w1 for comparability with
    # avg_learned_weight_diff (which covers all 1702 weights, not just W1).
    # We align to the W1 size because w1_weights logs only the first layer.
    def mad(row):
        w1  = row['w1_arr']
        act = row['active_arr']
        if act is None or len(act) == 0:
            return np.nan
        # Compare the W1 positions in genome vs active — same slice, consistent length
        n = len(w1)
        if len(act) < n:
            return np.nan
        return float(np.mean(np.abs(w1 - act[:n])))

    df = df.copy()
    df['mad'] = df.apply(mad, axis=1)

    # Check if RL is actually doing anything (GA-only: active == genome => MAD=0)
    overall_mad = df['mad'].mean()
    if overall_mad < 1e-5:
        print("  Skipping weight drift: active_weights identical to genome "
              "(GA-only condition)")
        return

    per_gen = df.groupby('generation')['mad'].mean()
    gens    = per_gen.index.values
    diffs   = per_gen.values

    # Rolling mean
    w = max(5, len(gens) // 40)
    smoothed, s_idx = rolling(diffs, w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: raw MAD per generation
    ax = axes[0]
    ax.plot(gens, diffs, color=PALETTE['drift'], lw=0.8, alpha=0.5,
            label='per gen mean')
    ax.plot(gens[s_idx], smoothed, color=PALETTE['drift'], lw=2.5,
            label=f'{w}-gen rolling mean')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Mean |active W1 − genome W1|', fontsize=10)
    ax.set_title('Within-Lifetime RL Drift: How Far RL Moves From Genome', fontsize=11)
    ax.legend(fontsize=9)
    note = ('Each generation: mean over top-20% organisms.\n'
            'Measures how much RL changes inherited weights within one lifetime.')
    ax.text(0.98, 0.05, note, transform=ax.transAxes, fontsize=7.5,
            ha='right', va='bottom', color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    # Right: overlay MAD with avg_learned_weight_diff from generations.csv
    # (the scalar already logged) for cross-validation
    ax2 = axes[1]
    ax2.plot(gens, diffs, color=PALETTE['drift'], lw=0.8, alpha=0.4,
             label='MAD (from organisms.csv)')
    ax2.plot(gens[s_idx], smoothed, color=PALETTE['drift'], lw=2,
             label=f'MAD {w}-gen mean')

    if 'avg_learned_weight_diff' in gen_df.columns:
        gd = gen_df.sort_values('generation')
        ld = gd['avg_learned_weight_diff'].values
        if np.abs(ld).max() > 1e-6:
            ax2.plot(gd['generation'].values, ld, color='#F59E0B',
                     lw=1.5, alpha=0.7, ls='--',
                     label='avg_learned_weight_diff (generations.csv)')

    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Weight Difference', fontsize=10)
    ax2.set_title('Cross-validation: organisms.csv MAD vs generations.csv scalar',
                  fontsize=11)
    ax2.legend(fontsize=8)
    note2 = ('Both lines measuring the same thing from different sources.\n'
             'Agreement validates the logging pipeline.')
    ax2.text(0.98, 0.05, note2, transform=ax2.transAxes, fontsize=7.5,
             ha='right', va='bottom', color='#374151',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    fig.suptitle(f'Baldwin Encoding Signal — {label}', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(out_dir, f'{_slug(label)}_weight_drift.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Figure 4: PCA variance over time (rolling-smoothed) ──────────────────────

def plot_pca_variance(df, label, n_components=5, out_dir='.'):
    print("  Plotting PCA variance over time ...")
    gens = sorted(df['generation'].unique())
    results = []
    for g in gens:
        vecs = np.stack(df.loc[df['generation'] == g, 'w1_arr'].values)
        k = min(n_components, len(vecs) - 1)
        if k < 1:
            continue
        pca = PCA(n_components=k, random_state=42)
        pca.fit(vecs)
        entry = {'gen': g, 'total': pca.explained_variance_ratio_.sum()}
        for i, v in enumerate(pca.explained_variance_ratio_):
            entry[f'pc{i+1}'] = v
        results.append(entry)

    if not results:
        print("  Skipping: insufficient data"); return

    res  = pd.DataFrame(results)
    gval = res['gen'].values
    w    = max(10, len(gval) // 25)     # smooth window: ~4% of run length

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: individual PCs (smoothed)
    ax = axes[0]
    colours = cm.cool(np.linspace(0.1, 0.9, n_components))
    for i in range(n_components):
        col = f'pc{i+1}'
        if col not in res: break
        sm, si = rolling(res[col].values, w)
        ax.plot(gval[si], sm, color=colours[i], lw=1.8, label=f'PC{i+1}')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel(f'Variance Explained', fontsize=11)
    ax.set_title(f'Per-PC Variance Explained ({w}-gen rolling mean)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, None)

    # Right: total (PC1-5), raw + smoothed
    ax = axes[1]
    ax.plot(gval, res['total'].values, color='#94A3B8', lw=0.6, alpha=0.5)
    sm, si = rolling(res['total'].values, w)
    ax.plot(gval[si], sm, color='#1F2937', lw=2.5,
            label=f'Total PC1–{n_components} ({w}-gen mean)')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel(f'Total Variance (PC1–{n_components})', fontsize=11)
    ax.set_title('Population Compression Over Time', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    note = ('Falling total = population converging onto fewer dimensions.\n'
            'Rising = population diversifying. Noisy because ~20-30 orgs/gen.')
    ax.text(0.98, 0.05, note, transform=ax.transAxes, fontsize=7.5,
            ha='right', va='bottom', color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    fig.suptitle(f'PCA Variance Explained — {label}', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(out_dir, f'{_slug(label)}_pca_variance.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Figure 5: Summary panel ───────────────────────────────────────────────────

def plot_summary(gen_df, label, out_dir='.'):
    print("  Plotting summary panel ...")
    g    = gen_df.sort_values('generation')
    gens = g['generation'].values

    has_early = ('avg_energy_at_tick_1000' in g.columns and
                 g['avg_energy_at_tick_1000'].replace(0, np.nan).notna().any())
    has_rl    = ('avg_learned_weight_diff' in g.columns and
                 g['avg_learned_weight_diff'].abs().max() > 1e-6)

    n_extra = sum([has_early, has_rl])
    n_rows  = 2 + (1 if n_extra else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    def sm(arr, w=20):
        if len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode='same')

    idx = 0

    # Fitness
    ax = axes[idx]; idx += 1
    ax.plot(gens, g['best_fitness'],         color='#DC2626', lw=0.7, alpha=0.3)
    ax.plot(gens, sm(g['best_fitness'].values),        color='#DC2626', lw=2, label='Best')
    ax.plot(gens, g['top20percent_fitness'], color='#7C3AED', lw=0.7, alpha=0.3)
    ax.plot(gens, sm(g['top20percent_fitness'].values),color='#7C3AED', lw=2, label='Top 20%')
    ax.plot(gens, g['avg_fitness'],          color='#D97706', lw=0.7, alpha=0.3)
    ax.plot(gens, sm(g['avg_fitness'].values),         color='#D97706', lw=2, label='Average')
    ax.set_title('Fitness Progression', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness Score')
    ax.legend(fontsize=8)

    # Genomic variance
    ax = axes[idx]; idx += 1
    ax.plot(gens, g['genome_variance'], color=PALETTE['var'], lw=0.7, alpha=0.3)
    ax.plot(gens, sm(g['genome_variance'].values), color=PALETTE['var'], lw=2.5)
    ax.set_title('Genomic Variance', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean Per-Position Variance')

    # Weight magnitude + inter-gen change
    ax = axes[idx]; idx += 1
    ax2 = ax.twinx()
    ax.plot(gens, g['avg_network_weight_mag'], color=PALETTE['mag'], lw=1.8)
    ax2.plot(gens, g['inter_gen_weight_change'], color='#94A3B8', lw=0.7, alpha=0.6)
    ax2.axhline(0, color='#94A3B8', lw=0.5, ls='--')
    ax.set_title('Weight Magnitude & Inter-Gen Change', fontsize=11)
    ax.set_xlabel('Generation')
    ax.set_ylabel('RMS Weight Magnitude', color=PALETTE['mag'])
    ax2.set_ylabel('Inter-Gen ΔMagnitude', color='#94A3B8')
    lines  = [plt.Line2D([0],[0],color=PALETTE['mag'],lw=2),
              plt.Line2D([0],[0],color='#94A3B8',lw=1.5)]
    labels = ['Weight Magnitude', 'Inter-Gen Change']
    ax.legend(lines, labels, fontsize=8)

    # Population
    ax = axes[idx]; idx += 1
    ax.plot(gens, g['total_agents'],   color='#92400E', lw=0.7, alpha=0.3)
    ax.plot(gens, sm(g['total_agents'].values),   color='#92400E', lw=2, label='Total')
    ax.plot(gens, g['peak_population'],color='#0891B2', lw=0.7, alpha=0.3, ls='--')
    ax.plot(gens, sm(g['peak_population'].values),color='#0891B2', lw=2, ls='--', label='Peak')
    ax.set_title('Population Dynamics', fontsize=11)
    ax.set_xlabel('Generation'); ax.set_ylabel('Agent Count')
    ax.legend(fontsize=8)

    # Early-life energy (Baldwin signal)
    if has_early and idx < len(axes):
        ax = axes[idx]; idx += 1
        early = g['avg_energy_at_tick_1000'].replace(0, np.nan)
        ax.plot(gens, early, color=PALETTE['energy'], lw=0.7, alpha=0.3)
        ax.plot(gens, sm(early.ffill().values), color=PALETTE['energy'], lw=2.5)
        ax.set_title('Early-Life Energy @ Tick 2000 (Baldwin Signal)', fontsize=11)
        ax.set_xlabel('Generation'); ax.set_ylabel('Mean Energy')
        ax.text(0.98, 0.05,
                'Rising = organisms eating effectively early without needing to learn it',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))

    # RL weight drift
    if has_rl and idx < len(axes):
        ax = axes[idx]; idx += 1
        ld = g['avg_learned_weight_diff'].values
        ax.plot(gens, ld, color=PALETTE['drift'], lw=0.7, alpha=0.3)
        ax.plot(gens, sm(ld), color=PALETTE['drift'], lw=2.5)
        ax.set_title('Within-Lifetime RL Weight Drift', fontsize=11)
        ax.set_xlabel('Generation'); ax.set_ylabel('Mean |active − genome|')
        ax.text(0.98, 0.05, 'Non-zero only in RL condition',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F3F4F6', ec='none'))
    elif not has_rl and idx < len(axes):
        ax = axes[idx]; idx += 1
        ax.text(0.5, 0.5, 'No within-lifetime learning\n(GA-only condition)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#6B7280')
        ax.set_title('Within-Lifetime RL Weight Drift', fontsize=11)

    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Summary — {label}', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fname = os.path.join(out_dir, f'{_slug(label)}_summary.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LifeEngine per-condition neural weight analysis')
    parser.add_argument('--organisms',   required=True)
    parser.add_argument('--generations', required=True)
    parser.add_argument('--label',       default=None)
    parser.add_argument('--out',         default='.')
    parser.add_argument('--tsne-every',  type=int, default=25)
    parser.add_argument('--tsne-orgs',   type=int, default=5)
    parser.add_argument('--pca-smooth',  type=int, default=10)
    parser.add_argument('--top-k',       type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────────────")
    org_df = load_organisms(args.organisms, top_k=args.top_k)
    gen_df = load_generations(args.generations)
    label  = args.label or infer_label(gen_df)
    print(f"  Label: {label}")

    print("\n── Generating figures ───────────────────────────────────────────")
    pca, pca_gens, pca_coords = plot_pca_trajectory(org_df, label, smooth=args.pca_smooth, out_dir=args.out)
    plot_tsne(org_df, label, every_n=args.tsne_every,
              top_orgs=args.tsne_orgs, out_dir=args.out)
    plot_weight_drift(org_df, gen_df, label, out_dir=args.out)
    plot_pca_variance(org_df, label, out_dir=args.out)
    plot_summary(gen_df, label, out_dir=args.out)

    print(f"\n── Done. Figures in: {os.path.abspath(args.out)}\n")

if __name__ == '__main__':
    main()