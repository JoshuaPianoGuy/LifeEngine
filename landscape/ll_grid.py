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


def clipping_diagnostics(plane, axes, clip=1.0):
    """How far the [-1,1] weight clamp pushed each grid genome off its nominal
    plane position.

    build_grid_jobs() clips every grid genome because the simulator clips
    weights — an unclipped genome is one the system cannot produce, so its
    fitness would be meaningless. The cost is that in the outer regions the cell
    labelled (alpha, beta) is no longer anchor + alpha*u + beta*v, so the
    coordinate system degrades toward the edges. This measures the damage, which
    is typically far larger than the usual "corners are clipped" caption implies
    (~half the cells at coarse extents).

    Note the displacement is mostly PERPENDICULAR to the plane: clipping moves
    the genome off the slice rather than sliding it along the axes, so the
    (alpha, beta) label stays roughly honest while "this point lies in the
    plane" does not.

    Returns dict of (N,N) arrays: clip_fraction (share of the 3910 weights
    clamped) and displacement (L2 from the nominal point).
    """
    anchor, u, v = plane['anchor'], plane['u'], plane['v']
    alphas = np.array(axes['alphas']); betas = np.array(axes['betas'])
    N = axes['resolution']
    frac = np.zeros((N, N)); disp = np.zeros((N, N))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            ideal = anchor + a * u + b * v
            frac[i, j] = float(np.mean(np.abs(ideal) > clip))
            disp[i, j] = float(np.linalg.norm(np.clip(ideal, -clip, clip) - ideal))
    return {'clip_fraction': frac, 'displacement': disp}


def reconstruct_cube(results_df, axes, rl):
    """Return the per-repeat cube shaped (resolution, resolution, R).

    reconstruct_grid() collapses the R repeats immediately; the paired
    difference statistics need them intact. Repeat index r corresponds to
    map index axes['maps'][r] in BOTH RL passes, so cube_on - cube_off is a
    genuine paired difference (same terrain on both sides).
    """
    N = axes['resolution']
    R = axes['R']
    tag = 'on' if rl else 'off'
    cube = np.full((N, N, R), np.nan)
    df = results_df.copy()
    ex = df['id'].str.extract(r'p(\d+)_(\d+)_r(\d+)_(on|off)')
    df['i'] = ex[0].astype('Int64'); df['j'] = ex[1].astype('Int64')
    df['r'] = ex[2].astype('Int64'); df['tag'] = ex[3]
    df = df[df['tag'] == tag]
    cube[df['i'].to_numpy(dtype=int),
         df['j'].to_numpy(dtype=int),
         df['r'].to_numpy(dtype=int)] = df['mean_fitness'].to_numpy()
    return cube


def map_offset_decomposition(cube):
    """Split the within-cell repeat variance into the part explained by map
    IDENTITY (common-mode across the whole grid, since every cell uses the same
    map set) and the part that actually varies per genome.

    The σ panels plot std across repeats, but the repeats are different MAPS —
    stage 2 measured σ_vary ≈ 6-8 vs σ_fixed ≈ 0.1-1.0, so that std is
    dominated by terrain, not by anything spatial. This quantifies it.

    Returns dict: map_offsets (R,), explained_fraction, sigma_within (N,N).
    """
    R = cube.shape[2]
    dev = cube - np.nanmean(cube, axis=2, keepdims=True)
    map_off = np.nanmean(dev, axis=(0, 1))
    n_cells = np.isfinite(cube[:, :, 0]).sum()
    ss_map = float((map_off ** 2).sum() * n_cells)
    ss_tot = float(np.nansum(dev ** 2))
    resid = dev - map_off[None, None, :]
    sigma_within = np.sqrt(np.nansum(resid ** 2, axis=2) / max(R - 1, 1))
    return {
        'map_offsets': map_off,
        'explained_fraction': (ss_map / ss_tot) if ss_tot > 0 else float('nan'),
        'sigma_within': sigma_within,
    }


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
