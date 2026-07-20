"""
landscape/stage7_plots.py
======================================================================
Stage 7 — plot a grid out-dir (works for Stage-3 PCA and Stage-4 random planes)
and report the per-panel scalars.

Panels (raw units):
    f_evo (RL off)   f_learn (RL on)   f_learn - f_evo (the halo/difference)
    sigma_evo        sigma_learn       sigma_evo - sigma_learn
Trajectories (evo/learn centroid paths) are overlaid on f_evo/f_learn for PCA
planes. f_evo and f_learn share one colour scale (a compared pair must — else
autoscale stretches a smooth surface to look as structured as a rugged one).

The Hinton & Nowlan prediction is a HALO on the difference panel: a positive
ring around the peak, ~0 at the summit (a good genome has nothing left to learn)
and ~0 far out (a hopeless genome can't learn its way in). A uniform lift is NOT
landscape smoothing — check the shape, not just the sign.

Three normalisations, all emitted:
  raw               — absolute fitness (m=1 contour lives here, in raw units)
  fitness-normalised— f(alpha,beta)/f(theta0), comparable across environments
  axis-normalised   — alpha,beta divided by ||theta0|| (relative displacement)

Scalars -> scalars.json: peak count per panel, mean sigma per pass, EVR (PCA),
f(theta0). Pass --lambda-json <stage6 summary> to annotate lambda.

Usage
-----
  python landscape/stage7_plots.py --out-dir <grid dir> [--m1-level 4.375]
        [--lambda-json <stage6/lambda_summary.json>] [--pair-vmax auto]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ll_common as ll
import ll_grid


def _center_value(grid):
    """f(theta0): the (0,0) grid point if the grid has an odd resolution (center
    == anchor), else the mean of the 4 central cells."""
    N = grid.shape[0]
    if N % 2 == 1:
        return grid[N // 2, N // 2]
    c = N // 2
    return np.nanmean(grid[c - 1:c + 1, c - 1:c + 1])


def _imshow(ax, grid, extent, cmap, vmin=None, vmax=None, title=''):
    # grid is [i=alpha, j=beta]; transpose so alpha=x, beta=y and origin lower.
    im = ax.imshow(grid.T, origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('α'); ax.set_ylabel('β')
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--m1-level', type=float, default=None,
                    help='draw the m=1 population growth/collapse contour at this raw fitness (e.g. 4.375)')
    ap.add_argument('--lambda-json', default=None)
    ap.add_argument('--normalize-axes', action='store_true',
                    help='divide α,β by ||θ0|| for relative-displacement axes')
    args = ap.parse_args()

    out_dir = args.out_dir
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    plane = ll_grid.load_plane(os.path.join(out_dir, 'plane.json'))
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    axes = meta['axes']
    N = axes['resolution']

    f_off, s_off = ll_grid.reconstruct_grid(res, axes, rl=False)
    f_on, s_on = ll_grid.reconstruct_grid(res, axes, rl=True)

    theta_norm = float(np.linalg.norm(plane['anchor']))
    a_ext = plane['alpha_extent']; b_ext = plane['beta_extent']
    if args.normalize_axes:
        extent = [-a_ext / theta_norm, a_ext / theta_norm, -b_ext / theta_norm, b_ext / theta_norm]
        axlabel = 'α/‖θ₀‖'
    else:
        extent = [-a_ext, a_ext, -b_ext, b_ext]
        axlabel = 'α'

    f0_off = _center_value(f_off)
    f0_on = _center_value(f_on)

    diff = f_on - f_off
    sdiff = s_off - s_on

    # ── scalars ───────────────────────────────────────────────────────────────
    scalars = {
        'peaks_f_evo': ll_grid.count_peaks(f_off),
        'peaks_f_learn': ll_grid.count_peaks(f_on),
        'mean_sigma_evo': float(np.nanmean(s_off)),
        'mean_sigma_learn': float(np.nanmean(s_on)),
        'f_theta0_evo': float(f0_off),
        'f_theta0_learn': float(f0_on),
        'f_evo_range': [float(np.nanmin(f_off)), float(np.nanmax(f_off))],
        'f_learn_range': [float(np.nanmin(f_on)), float(np.nanmax(f_on))],
    }
    if meta.get('plane_meta', {}).get('kind') == 'pca':
        scalars['explained_variance_ratio'] = meta['plane_meta'].get('explained_variance_ratio')
    lam_txt = ''
    if args.lambda_json and os.path.exists(args.lambda_json):
        lam = json.load(open(args.lambda_json))
        scalars['lambda'] = lam.get('lambda'); scalars['rho1'] = lam.get('rho1')
        lam_txt = f"   λ={lam.get('lambda'):.2f} (ρ1={lam.get('rho1'):.3f})"
    json.dump(scalars, open(os.path.join(out_dir, 'scalars.json'), 'w'), indent=2)

    # shared colour scale for the compared pair f_evo / f_learn
    pair_min = np.nanmin([np.nanmin(f_off), np.nanmin(f_on)])
    pair_max = np.nanmax([np.nanmax(f_off), np.nanmax(f_on)])

    pm = meta.get('plane_meta', {})
    evo_traj = np.array(pm.get('evo_traj', [])) if pm else np.array([])
    lrn_traj = np.array(pm.get('lrn_traj', [])) if pm else np.array([])
    tscale = theta_norm if args.normalize_axes else 1.0

    def overlay_traj(ax):
        if evo_traj.size:
            ax.plot(evo_traj[:, 0] / tscale, evo_traj[:, 1] / tscale, '-', color='k', lw=1.3, alpha=0.9)
            ax.plot(evo_traj[0, 0] / tscale, evo_traj[0, 1] / tscale, 'o', color='w', mec='k', ms=5)
            ax.plot(evo_traj[-1, 0] / tscale, evo_traj[-1, 1] / tscale, 's', color='k', ms=5, label='evo')
        if lrn_traj.size:
            ax.plot(lrn_traj[:, 0] / tscale, lrn_traj[:, 1] / tscale, '-', color='magenta', lw=1.3, alpha=0.9)
            ax.plot(lrn_traj[-1, 0] / tscale, lrn_traj[-1, 1] / tscale, 's', color='magenta', ms=5, label='learn')

    # ── figure: raw ───────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    e2 = [extent[0], extent[1], extent[2], extent[3]]

    im = _imshow(axs[0, 0], f_off, e2, 'viridis', pair_min, pair_max,
                 f'f_evo (RL off)   peaks={scalars["peaks_f_evo"]}   f(θ₀)={f0_off:.2f}')
    overlay_traj(axs[0, 0])
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046)
    if args.m1_level is not None:
        alphas = np.linspace(extent[0], extent[1], N); betas = np.linspace(extent[2], extent[3], N)
        AA, BB = np.meshgrid(alphas, betas)
        axs[0, 0].contour(AA, BB, f_off.T, levels=[args.m1_level], colors='red', linewidths=1.5)

    im = _imshow(axs[0, 1], f_on, e2, 'viridis', pair_min, pair_max,
                 f'f_learn (RL on)   peaks={scalars["peaks_f_learn"]}   f(θ₀)={f0_on:.2f}')
    overlay_traj(axs[0, 1]); axs[0, 1].legend(fontsize=7, loc='upper right')
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046)

    dmax = np.nanmax(np.abs(diff)) or 1.0
    im = _imshow(axs[0, 2], diff, e2, 'RdBu_r', -dmax, dmax,
                 'f_learn − f_evo  (Hinton–Nowlan halo: +ring, ~0 at summit)')
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046)

    smax = np.nanmax([np.nanmax(s_off), np.nanmax(s_on)]) or 1.0
    im = _imshow(axs[1, 0], s_off, e2, 'magma', 0, smax,
                 f'σ_evo   mean={scalars["mean_sigma_evo"]:.3f}')
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046)
    im = _imshow(axs[1, 1], s_on, e2, 'magma', 0, smax,
                 f'σ_learn   mean={scalars["mean_sigma_learn"]:.3f}')
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046)
    sdm = np.nanmax(np.abs(sdiff)) or 1.0
    im = _imshow(axs[1, 2], sdiff, e2, 'PuOr', -sdm, sdm, 'σ_evo − σ_learn')
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)

    for ax in axs.flat:
        ax.set_xlabel(axlabel); ax.set_ylabel(axlabel.replace('α', 'β'))
        # Pin to the grid extent so the trajectory overlay can't expand the axes
        # past the heatmap and leave white margins.
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    kind = pm.get('kind', 'grid'); grid = meta.get('grid', '')
    cfg = meta.get('config_overrides')
    fig.suptitle(f'Fitness landscape — {kind} {grid}   (corners clipped to [−1,1]){lam_txt}\n'
                 f'{os.path.basename(os.path.normpath(out_dir))}   env_overrides={cfg}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    raw_path = os.path.join(out_dir, 'landscape_raw.png')
    fig.savefig(raw_path, dpi=130)
    plt.close(fig)

    # ── figure: fitness-normalised f/f(θ0) for the pair (shared scale) ────────
    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4.6))
    fn_off = f_off / f0_off if f0_off else f_off
    fn_on = f_on / f0_off if f0_off else f_on   # normalise BOTH by the SAME f0 (RL-off anchor)
    nmin = np.nanmin([np.nanmin(fn_off), np.nanmin(fn_on)])
    nmax = np.nanmax([np.nanmax(fn_off), np.nanmax(fn_on)])
    im = _imshow(axs2[0], fn_off, e2, 'viridis', nmin, nmax, 'f_evo / f(θ₀)')
    overlay_traj(axs2[0]); fig2.colorbar(im, ax=axs2[0], fraction=0.046)
    im = _imshow(axs2[1], fn_on, e2, 'viridis', nmin, nmax, 'f_learn / f(θ₀)')
    overlay_traj(axs2[1]); fig2.colorbar(im, ax=axs2[1], fraction=0.046)
    for ax in axs2:
        ax.set_xlabel(axlabel); ax.set_ylabel(axlabel.replace('α', 'β'))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig2.suptitle(f'Fitness-normalised (θ₀ = RL-off anchor)   {os.path.basename(os.path.normpath(out_dir))}')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    norm_path = os.path.join(out_dir, 'landscape_normalised.png')
    fig2.savefig(norm_path, dpi=130)
    plt.close(fig2)

    print(f'[stage7] scalars: {json.dumps(scalars, indent=2)}')
    print(f'[stage7] -> {raw_path}\n           {norm_path}')


if __name__ == '__main__':
    main()
