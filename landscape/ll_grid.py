"""
landscape/ll_grid.py
======================================================================
Shared plane + grid machinery for Stages 3 (PCA plane), 4 (random
filter-normalised planes), 5 (grid evaluation) and 7 (plots).

A PLANE is (anchor theta0, unit directions u, v, and per-axis extents). The grid
samples f(theta0 + alpha*u + beta*v) over an (alpha, beta) mesh, evaluated on
BOTH RL passes at IDENTICAL map indices (paired) so that map-to-map variance —
the largest noise source — cancels in the f_learn - f_evo difference.

Grid genomes are clipped to [-1,1] (the simulator clips weights, so unclipped
corners would be genomes the system cannot produce — their fitness is
meaningless). Far corners being clipped is noted in plot captions.
"""

import json
import os

import numpy as np

import ll_common as ll


# ── plane save/load ───────────────────────────────────────────────────────────
def save_plane(path, plane):
    obj = {
        'anchor_b64': ll.encode_b64(plane['anchor']),
        'u_b64': ll.encode_b64(plane['u']),
        'v_b64': ll.encode_b64(plane['v']),
        'alpha_extent': float(plane['alpha_extent']),
        'beta_extent': float(plane['beta_extent']),
        'meta': plane.get('meta', {}),
    }
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_plane(path):
    o = json.load(open(path))
    return {
        'anchor': ll.decode_b64(o['anchor_b64']),
        'u': ll.decode_b64(o['u_b64']),
        'v': ll.decode_b64(o['v_b64']),
        'alpha_extent': o['alpha_extent'],
        'beta_extent': o['beta_extent'],
        'meta': o.get('meta', {}),
    }


# ── grid job generation (Stage 5) ─────────────────────────────────────────────
def build_grid_jobs(plane, resolution, R, map_start=0, rl_passes=(False, True),
                    clip=True):
    """Return (genomes, jobs, axes) for a plane grid.

    resolution: int N -> N x N grid.
    R: repeats per point; both RL passes use the SAME R map indices (pairing).
    map_start: first map index (repeats use map_start .. map_start+R-1).
    """
    a_ext = plane['alpha_extent']; b_ext = plane['beta_extent']
    alphas = np.linspace(-a_ext, a_ext, resolution)
    betas = np.linspace(-b_ext, b_ext, resolution)
    anchor, u, v = plane['anchor'], plane['u'], plane['v']

    genomes, jobs = {}, []
    maps = [map_start + r for r in range(R)]
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta = anchor + a * u + b * v
            if clip:
                theta = np.clip(theta, -1.0, 1.0)
            key = f'g{i}_{j}'
            genomes[key] = theta
            for r, m in enumerate(maps):
                for rl in rl_passes:
                    tag = 'on' if rl else 'off'
                    jobs.append({'id': f'p{i}_{j}_r{r}_{tag}', 'genome': key,
                                 'map_index': int(m), 'rl': bool(rl)})
    axes = {'alphas': alphas.tolist(), 'betas': betas.tolist(),
            'resolution': resolution, 'R': R, 'maps': maps,
            'rl_passes': list(rl_passes)}
    return genomes, jobs, axes


# ── grid reconstruction (Stage 7) ─────────────────────────────────────────────
def reconstruct_grid(results_df, axes, rl):
    """Return (mean_grid, std_grid) shaped (resolution, resolution) [i=alpha, j=beta].
    Averages the R repeats per point; std is across the R run means (the honest
    error, since repeats are independent runs)."""
    N = axes['resolution']
    tag = 'on' if rl else 'off'
    mean_g = np.full((N, N), np.nan)
    std_g = np.full((N, N), np.nan)
    df = results_df.copy()
    # id: p{i}_{j}_r{r}_{tag}
    ex = df['id'].str.extract(r'p(\d+)_(\d+)_r(\d+)_(on|off)')
    df['i'] = ex[0].astype('Int64'); df['j'] = ex[1].astype('Int64'); df['tag'] = ex[3]
    df = df[df['tag'] == tag]
    for (i, j), sub in df.groupby(['i', 'j']):
        vals = sub['mean_fitness'].to_numpy()
        mean_g[int(i), int(j)] = vals.mean()
        std_g[int(i), int(j)] = vals.std(ddof=1) if vals.size > 1 else 0.0
    return mean_g, std_g


def count_peaks(grid):
    """Count strict 2D local maxima (8-neighbour). NaNs ignored. A rough basin
    counter for the 'peak count per panel' scalar."""
    g = grid
    N, M = g.shape
    peaks = 0
    for i in range(N):
        for j in range(M):
            c = g[i, j]
            if not np.isfinite(c):
                continue
            is_peak = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M and np.isfinite(g[ni, nj]):
                        if g[ni, nj] >= c:
                            is_peak = False
                            break
                if not is_peak:
                    break
            if is_peak:
                peaks += 1
    return peaks


def write_grid_jobs(out_dir, plane, resolution, R, map_start=0,
                    rl_passes=(False, True), config_overrides=None, extra_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    genomes, jobs, axes = build_grid_jobs(plane, resolution, R, map_start, rl_passes)
    ll.write_jobs(os.path.join(out_dir, 'jobs.json'), genomes, jobs, config_overrides)
    meta = {'axes': axes, 'alpha_extent': plane['alpha_extent'],
            'beta_extent': plane['beta_extent'], 'plane_meta': plane.get('meta', {}),
            'config_overrides': config_overrides}
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
    save_plane(os.path.join(out_dir, 'plane.json'), plane)
    print(f'[grid] {len(jobs)} jobs ({resolution}x{resolution} x R={R} x {len(rl_passes)} passes) -> {out_dir}/jobs.json')
    return axes
