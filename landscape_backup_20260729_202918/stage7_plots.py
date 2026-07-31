"""
landscape/stage7_plots.py
======================================================================
Stage 7 — plot a grid out-dir (works for Stage-3 PCA and Stage-4 random planes)
and report the per-panel scalars.

Panels (raw units):
    f(theta) | RL off   f(theta) | RL on    Delta = RL on - RL off (the raw lift)
    Delta - mean Delta  sigma | RL off      sigma | RL on

WHAT THE TWO SURFACE PANELS ARE (the single most-misread thing here):
BOTH evaluate the SAME 625 synthetic grid genomes theta_ij = anchor + a*u + b*v
on the SAME map set. The ONLY difference is whether REINFORCE may update
active_weights during the 2000 ticks. Neither panel evaluates a logged centroid,
and neither panel is "the evolution run's landscape" or "the learning run's
landscape" — the grid is shared and knows nothing about either run.

The overlaid paths are a SEPARATE object: two independent 1000-generation
training runs (one evolution-condition, one learning-condition), projected into
the same plane. They are real logged genomes, not grid points, and they appear
IDENTICALLY on both panels. Panel identity = the probe's RL flag; marker
identity = which training run produced the genome. The two axes are unrelated.

The two surface panels share one colour scale (a compared pair must — else
autoscale stretches a smooth surface to look as structured as a rugged one),
which is also why a ~2%-of-range lift makes them look identical. That is
expected; the lift lives in the Delta panels.

READ THE PANELS LIKE THIS
-------------------------
1. The Hinton & Nowlan prediction is a HALO: a positive ring around the peak,
   ~0 at the summit (a good genome has nothing left to learn) and ~0 far out (a
   hopeless genome can't learn its way in). A UNIFORM LIFT IS NOT LANDSCAPE
   SMOOTHING — learning that adds a constant everywhere leaves the shape
   evolution sees unchanged and cannot guide it. So the raw-lift panel answers
   only "does learning help, and where does it hurt"; the SHAPE panel
   (lift - mean lift) is the one carrying the Hinton-Nowlan claim, and
   `landscape_halo.png` is the actual test.

2. sigma is NOT evaluation noise at a point. The R repeats are R different
   MAPS, so sigma is that genome's TERRAIN SENSITIVITY. Stage 2 measured
   sigma_vary ~ 6-8 vs sigma_fixed ~ 0.1-1.0, and the map-offset decomposition
   printed here typically attributes 80-96% of it to map identity alone —
   which is common-mode across every cell (same map set everywhere) and so
   carries no spatial information. The sigma panels also correlate ~0.95+ with
   the fitness panels (mean-variance relationship: a genome that eats more has
   more to lose on a bad map), i.e. they are close to a recolouring of them.

3. sigma_evo - sigma_learn is NOT plotted. At R=10 an sd estimate carries ~24%
   relative noise, so their difference is dominated by estimator noise (lag-1
   spatial autocorrelation ~0.15 in baseline = white noise). It is reported as
   a scalar instead, where it is meaningful.

Colour scales are ROBUST (2nd/98th percentile) so a single loud pixel cannot
set the range for all N^2 of them, and the sigma panels use their own range
rather than being anchored at 0.

Three normalisations, all emitted:
  raw               — absolute fitness (m=1 contour lives here, in raw units)
  fitness-normalised— f(alpha,beta)/f(theta0), comparable across environments
  axis-normalised   — alpha,beta divided by ||theta0|| (relative displacement)

Figures
-------
  landscape_raw.png         6-panel surface figure (above)
  landscape_normalised.png  f/f(theta0) pair, shared scale
  landscape_halo.png        the Hinton-Nowlan test: lift vs radius from the
                            anchor, vs radius from the f_evo peak, and vs
                            fitness level, with binned error bars

Scalars -> scalars.json: peak count per panel, mean sigma per pass, EVR (PCA),
f(theta0), the lift decomposition (mean / shape / paired SEM / significance),
sigma diagnostics, and the radial profile. Pass --lambda-json to annotate lambda.

Usage
-----
  python landscape/stage7_plots.py --out-dir <grid dir> [--m1-level 4.375]
        [--lambda-json <stage6/lambda_summary.json>] [--interpolation bilinear]
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


def _robust_limits(grid, lo=2, hi=98, symmetric_about=None):
    """Percentile colour limits, so one outlier pixel can't set the scale for
    the whole panel. symmetric_about=x forces the range to straddle x evenly
    (used for diverging maps, where the neutral colour must mean exactly x)."""
    g = np.asarray(grid)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return None, None
    vmin, vmax = np.percentile(g, [lo, hi])
    if symmetric_about is not None:
        c = float(symmetric_about)
        r = max(abs(vmax - c), abs(c - vmin)) or 1.0
        return c - r, c + r
    if vmin == vmax:
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def _corr(a, b):
    """Pearson r over the cells finite in both."""
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _lag1_autocorr(grid):
    """Nearest-neighbour spatial autocorrelation. ~1 = smooth surface rendered
    at low resolution; ~0 = pixel-for-pixel noise. Distinguishes 'blocky' from
    'noisy', which the eye cannot do on a 25x25 heatmap."""
    g = np.asarray(grid)
    x = np.concatenate([g[:-1, :].ravel(), g[:, :-1].ravel()])
    y = np.concatenate([g[1:, :].ravel(), g[:, 1:].ravel()])
    return _corr(x, y)


def _imshow(ax, grid, extent, cmap, vmin=None, vmax=None, title='',
            interpolation='nearest'):
    # grid is [i=alpha, j=beta]; transpose so alpha=x, beta=y and origin lower.
    im = ax.imshow(grid.T, origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation)
    ax.set_title(title, fontsize=9)
    return im


def _binned_profile(x, y, yerr_cell, nbins=8):
    """Quantile-bin y against x. Returns per-bin (x_mid, mean_y, measurement
    SEM of the bin mean, within-bin spread, n).

    Two error quantities, because they answer different questions:
      - meas_sem: pure measurement error on the bin mean, from the per-cell
        paired SEMs. 'Is this bin's value resolved?' -> yes if the bin-to-bin
        swings exceed it.
      - spread: sd of the lift across cells in the bin, i.e. how much real
        spatial variation the bin averages over (cells are spatially
        correlated, so this is not an error bar on the mean).
    """
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    e = np.asarray(yerr_cell).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y, e = x[m], y[m], e[m]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        sel = idx == k
        n = int(sel.sum())
        if n == 0:
            continue
        ek = e[sel]
        meas = float(np.sqrt(np.nanmean(ek ** 2) / n)) if np.isfinite(ek).any() else float('nan')
        rows.append({'x': float(x[sel].mean()), 'mean': float(y[sel].mean()),
                     'meas_sem': meas, 'spread': float(y[sel].std()), 'n': n})
    return rows


def _halo_sketch(x, kind, amplitude):
    """Schematic Hinton-Nowlan prediction for comparison — SHAPE ONLY, the
    amplitude is arbitrary and scaled to the observed data.

    kind='peak'    : Delta ~ 0 at the summit (a good genome has nothing left to
                     learn), rising to a maximum at intermediate distance, then
                     decaying far out (a hopeless genome cannot learn its way in
                     within one lifetime). H&N's outer edge comes from the trial
                     budget: with k undetermined alleles you need ~2^k guesses,
                     so beyond k ~ log2(budget) learning stops paying.
    kind='fitness' : the same hump expressed against fitness LEVEL — low for
                     hopeless genomes, high in the middle, low at the summit.

    Returned already mean-centred, to overlay on a (Delta - mean Delta) axis.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    t = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    if kind == 'peak':
        tp = 0.35                      # vanishes at t=0, peaks at tp, decays
        h = (t / tp) * np.exp(1.0 - t / tp)
    else:                              # 'fitness' — low at both ends
        h = np.exp(-(((t - 0.5) / 0.22) ** 2))
    h = h - h.mean()
    m = np.max(np.abs(h))
    return h / m * amplitude if m > 0 else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--m1-level', type=float, default=None,
                    help='draw the m=1 population growth/collapse contour at this raw fitness (e.g. 4.375)')
    ap.add_argument('--lambda-json', default=None)
    ap.add_argument('--normalize-axes', action='store_true',
                    help='divide α,β by ||θ0|| for relative-displacement axes')
    ap.add_argument('--interpolation', default='nearest',
                    choices=['nearest', 'bilinear', 'bicubic'],
                    help='applied to the FITNESS panels only (their lag-1 spatial '
                         'autocorrelation is ~0.98, so smoothing reveals resolution '
                         'rather than inventing structure). The lift and sigma panels '
                         'always stay nearest — smoothing those would blend noise into '
                         'shapes that are not in the data.')
    ap.add_argument('--profile-bins', type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    plane = ll_grid.load_plane(os.path.join(out_dir, 'plane.json'))
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    axes = meta['axes']
    N = axes['resolution']
    R = axes['R']

    f_off, s_off = ll_grid.reconstruct_grid(res, axes, rl=False)
    f_on, s_on = ll_grid.reconstruct_grid(res, axes, rl=True)

    # Per-repeat cubes: repeat r is the SAME map index in both passes, so the
    # difference is paired and terrain cancels cell-by-cell rather than only
    # on average.
    cube_off = ll_grid.reconstruct_cube(res, axes, rl=False)
    cube_on = ll_grid.reconstruct_cube(res, axes, rl=True)
    lift_cube = cube_on - cube_off
    lift_sd = np.nanstd(lift_cube, axis=2, ddof=1)      # per-run spread of the paired diff
    lift_sem = lift_sd / np.sqrt(R)                     # error on each cell's lift

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
    mean_lift = float(np.nanmean(diff))
    shape = diff - mean_lift          # the part that is NOT a uniform lift
    sdiff = s_off - s_on

    # plane coordinates of every cell, for the radial profiles
    alphas = np.array(axes['alphas']); betas = np.array(axes['betas'])
    AA, BB = np.meshgrid(alphas, betas, indexing='ij')
    rad_anchor = np.sqrt(AA ** 2 + BB ** 2)
    pi, pj = np.unravel_index(np.nanargmax(f_off), f_off.shape)
    peak_a, peak_b = alphas[pi], betas[pj]
    rad_peak = np.sqrt((AA - peak_a) ** 2 + (BB - peak_b) ** 2)
    # If the best sampled cell is on the grid boundary, the true summit is
    # probably OUTSIDE the window — every "at the summit" statement below is
    # then really "at the highest point we sampled", and the halo test is
    # measuring one flank of a ring rather than a ring.
    peak_on_boundary = bool(pi in (0, N - 1) or pj in (0, N - 1))

    # sigma diagnostics: how much of the R-repeat spread is just map identity?
    dec_off = ll_grid.map_offset_decomposition(cube_off)
    dec_on = ll_grid.map_offset_decomposition(cube_on)

    # how badly the [-1,1] clamp moved grid genomes off their nominal positions
    clipd = ll_grid.clipping_diagnostics(plane, axes)
    cf, cd = clipd['clip_fraction'], clipd['displacement']
    step = float(alphas[1] - alphas[0])
    clip_cells = float(np.mean(cf > 0))

    # ── scalars ───────────────────────────────────────────────────────────────
    prof_anchor = _binned_profile(rad_anchor, diff, lift_sem, args.profile_bins)
    prof_peak = _binned_profile(rad_peak, diff, lift_sem, args.profile_bins)
    prof_fit = _binned_profile(f_off, diff, lift_sem, args.profile_bins)

    scalars = {
        # Kept for continuity, NOT plotted: a "peak" here is only a local max of
        # the 2D SLICE at this resolution — 8 in-plane neighbours out of 3910
        # directions — so it can be a saddle in the full space. A split-half test
        # (5 maps vs 5 maps) reproduces the surface at r=0.99 but only ~13% of
        # peak LOCATIONS, and the count itself swings 15 vs 22 on the same
        # landscape. Use lambda (stage 6) / effective K (stage 8) for ruggedness.
        'peaks_f_evo': ll_grid.count_peaks(f_off),
        'peaks_f_learn': ll_grid.count_peaks(f_on),
        'mean_sigma_evo': float(np.nanmean(s_off)),
        'mean_sigma_learn': float(np.nanmean(s_on)),
        'f_theta0_evo': float(f0_off),
        'f_theta0_learn': float(f0_on),
        'f_evo_range': [float(np.nanmin(f_off)), float(np.nanmax(f_off))],
        'f_learn_range': [float(np.nanmin(f_on)), float(np.nanmax(f_on))],

        # ── the lift, split into "uniform" and "shape" ────────────────────────
        # mean_lift alone is NOT evidence for Hinton-Nowlan; lift_shape_sd vs
        # lift_paired_sem_per_cell is what says whether any shape survives noise.
        'lift_mean': mean_lift,
        'lift_shape_sd': float(np.nanstd(shape)),
        'lift_paired_sem_per_cell': float(np.nanmean(lift_sem)),
        'lift_frac_positive': float(np.nanmean(diff > 0)),
        'lift_frac_cells_2sem': float(np.nanmean(np.abs(diff) > 2 * lift_sem)),
        'lift_range': [float(np.nanmin(diff)), float(np.nanmax(diff))],
        'corr_lift_fitness': _corr(diff, f_off),
        'lift_at_peak_cell': float(diff[pi, pj]),
        'f_evo_peak_cell': {'alpha': float(peak_a), 'beta': float(peak_b),
                            'f_evo': float(f_off[pi, pj]),
                            'on_boundary': peak_on_boundary},
        'lift_radial_profile_anchor': prof_anchor,
        'lift_radial_profile_peak': prof_peak,
        'lift_vs_fitness_profile': prof_fit,

        # ── sigma: what it actually measures ─────────────────────────────────
        'sigma_map_identity_fraction_evo': float(dec_off['explained_fraction']),
        'sigma_map_identity_fraction_learn': float(dec_on['explained_fraction']),
        'sigma_within_map_evo': float(np.nanmean(dec_off['sigma_within'])),
        'sigma_within_map_learn': float(np.nanmean(dec_on['sigma_within'])),
        'map_offsets_evo': [float(x) for x in dec_off['map_offsets']],
        'corr_sigma_fitness_evo': _corr(s_off, f_off),
        'corr_sigma_fitness_learn': _corr(s_on, f_on),
        # reported, not plotted: at R=10 the difference of two sd estimates is
        # mostly estimator noise (see lag1_sigma_diff below).
        'mean_sigma_evo_minus_learn': float(np.nanmean(sdiff)),

        # ── clipping: where the (α,β) coordinate system stops being literal ──
        # The cell is labelled anchor + α·u + β·v but the genome ACTUALLY probed
        # is the clipped one. Displacement is mostly perpendicular to the plane.
        'clip_cells_affected_frac': clip_cells,
        'clip_max_weight_frac': float(cf.max()),
        'clip_displacement_mean': float(cd.mean()),
        'clip_displacement_max': float(cd.max()),
        'clip_displacement_max_grid_steps': float(cd.max() / step) if step else float('nan'),
        'grid_step_l2': step,

        # ── blocky vs noisy ──────────────────────────────────────────────────
        'lag1_f_evo': _lag1_autocorr(f_off),
        'lag1_f_learn': _lag1_autocorr(f_on),
        'lag1_lift': _lag1_autocorr(diff),
        'lag1_sigma_evo': _lag1_autocorr(s_off),
        'lag1_sigma_diff': _lag1_autocorr(sdiff),
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
    pair_min, pair_max = _robust_limits(np.concatenate([f_off.ravel(), f_on.ravel()]))

    pm = meta.get('plane_meta', {})
    evo_traj = np.array(pm.get('evo_traj', [])) if pm else np.array([])
    lrn_traj = np.array(pm.get('lrn_traj', [])) if pm else np.array([])
    tscale = theta_norm if args.normalize_axes else 1.0

    def overlay_traj(ax):
        if evo_traj.size:
            ax.plot(evo_traj[:, 0] / tscale, evo_traj[:, 1] / tscale, '-', color='k', lw=1.3, alpha=0.9)
            ax.plot(evo_traj[0, 0] / tscale, evo_traj[0, 1] / tscale, 'o', color='w', mec='k', ms=5)
            ax.plot(evo_traj[-1, 0] / tscale, evo_traj[-1, 1] / tscale, 's', color='k', ms=5,
                    label='evolution run (training RL off)')
        if lrn_traj.size:
            ax.plot(lrn_traj[:, 0] / tscale, lrn_traj[:, 1] / tscale, '-', color='magenta', lw=1.3, alpha=0.9)
            ax.plot(lrn_traj[0, 0] / tscale, lrn_traj[0, 1] / tscale, 'o', color='w', mec='magenta', ms=5)
            ax.plot(lrn_traj[-1, 0] / tscale, lrn_traj[-1, 1] / tscale, 's', color='magenta', ms=5,
                    label='learning run (training RL on)')

    # ── figure: raw ───────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    e2 = list(extent)
    fint = args.interpolation

    im = _imshow(axs[0, 0], f_off, e2, 'viridis', pair_min, pair_max,
                 f'f(θ) | RL off   f(θ₀)={f0_off:.2f}',
                 interpolation=fint)
    overlay_traj(axs[0, 0])
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046)
    if args.m1_level is not None:
        ca = np.linspace(extent[0], extent[1], N); cb = np.linspace(extent[2], extent[3], N)
        CA, CB = np.meshgrid(ca, cb)
        axs[0, 0].contour(CA, CB, f_off.T, levels=[args.m1_level], colors='red', linewidths=1.5)

    im = _imshow(axs[0, 1], f_on, e2, 'viridis', pair_min, pair_max,
                 f'f(θ) | RL on   f(θ₀)={f0_on:.2f}',
                 interpolation=fint)
    overlay_traj(axs[0, 1]); axs[0, 1].legend(fontsize=7, loc='upper right')
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046)

    # Raw lift: diverging, forced symmetric about 0 so the neutral colour means
    # "learning changes nothing", with robust limits so an outlier can't set the
    # range. This panel answers "does learning help, and where does it hurt" —
    # NOT whether the landscape was reshaped.
    dmin, dmax = _robust_limits(diff, symmetric_about=0.0)
    im = _imshow(axs[0, 2], diff, e2, 'RdBu_r', dmin, dmax,
                 f'Δ = f(θ)|RL on − f(θ)|RL off  (raw lift, same genomes)\nmean={mean_lift:+.3f}  '
                 f'{100*scalars["lift_frac_positive"]:.0f}% cells >0  '
                 f'SEM/cell={scalars["lift_paired_sem_per_cell"]:.3f}')
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046)

    # Shape: the uniform component removed. THIS is the Hinton-Nowlan panel —
    # a halo shows up here as a positive ring with a negative centre.
    smin, smax_ = _robust_limits(shape, symmetric_about=0.0)
    im = _imshow(axs[1, 0], shape, e2, 'RdBu_r', smin, smax_,
                 f'Δ − mean Δ  (Hinton–Nowlan shape)\n'
                 f'sd={scalars["lift_shape_sd"]:.3f} vs SEM/cell={scalars["lift_paired_sem_per_cell"]:.3f}   '
                 f'lag1={scalars["lag1_lift"]:.2f}')
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046)
    axs[1, 0].plot(peak_a / tscale, peak_b / tscale, '*', color='lime', mec='k', ms=13)

    # sigma panels: shared robust range across BOTH passes, NOT anchored at 0
    # (anchoring at 0 pushed all the data into the top sliver of the colourmap).
    sig_lo, sig_hi = _robust_limits(np.concatenate([s_off.ravel(), s_on.ravel()]))
    im = _imshow(axs[1, 1], s_off, e2, 'magma', sig_lo, sig_hi,
                 f'σ | RL off — terrain sensitivity (across {R} maps)\n'
                 f'mean={scalars["mean_sigma_evo"]:.3f}   '
                 f'{100*scalars["sigma_map_identity_fraction_evo"]:.0f}% is map identity   '
                 f'r(σ,f)={scalars["corr_sigma_fitness_evo"]:+.2f}')
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046)
    im = _imshow(axs[1, 2], s_on, e2, 'magma', sig_lo, sig_hi,
                 f'σ | RL on — terrain sensitivity (across {R} maps)\n'
                 f'mean={scalars["mean_sigma_learn"]:.3f}   '
                 f'σ(off)−σ(on)={scalars["mean_sigma_evo_minus_learn"]:+.3f} (scalar only)   '
                 f'r(σ,f)={scalars["corr_sigma_fitness_learn"]:+.2f}')
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)

    for ax in axs.flat:
        ax.set_xlabel(axlabel); ax.set_ylabel(axlabel.replace('α', 'β'))
        # Pin to the grid extent so the trajectory overlay can't expand the axes
        # past the heatmap and leave white margins.
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    kind = pm.get('kind', 'grid'); grid = meta.get('grid', '')
    cfg = meta.get('config_overrides')
    fig.suptitle(f'Fitness landscape — {kind} {grid}{lam_txt}\n'
                 f'{os.path.basename(os.path.normpath(out_dir))}   env_overrides={cfg}\n'
                 f'grid step={step:.2f} L2 (≈{step/1.1:.1f} mutations)   |   '
                 f'clipping: {100*clip_cells:.0f}% of cells have ≥1 weight clamped to [−1,1]; '
                 f'worst cell {100*cf.max():.1f}% of weights, {cd.max():.1f} L2 '
                 f'({cd.max()/step:.1f} steps) off-plane\n'
                 f'Both surface panels probe the SAME {N}×{N} synthetic genome grid on the SAME {R} maps — '
                 f'only the RL flag differs. Overlaid paths are two separate training runs, '
                 f'identical in both panels.',
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    raw_path = os.path.join(out_dir, 'landscape_raw.png')
    fig.savefig(raw_path, dpi=130)
    plt.close(fig)

    # ── figure: fitness-normalised f/f(θ0) for the pair (shared scale) ────────
    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4.6))
    fn_off = f_off / f0_off if f0_off else f_off
    fn_on = f_on / f0_off if f0_off else f_on   # normalise BOTH by the SAME f0 (RL-off anchor)
    nmin, nmax = _robust_limits(np.concatenate([fn_off.ravel(), fn_on.ravel()]))
    im = _imshow(axs2[0], fn_off, e2, 'viridis', nmin, nmax, 'f(θ)|RL off ÷ f(θ₀)', interpolation=fint)
    overlay_traj(axs2[0]); fig2.colorbar(im, ax=axs2[0], fraction=0.046)
    im = _imshow(axs2[1], fn_on, e2, 'viridis', nmin, nmax, 'f(θ)|RL on ÷ f(θ₀)', interpolation=fint)
    overlay_traj(axs2[1]); fig2.colorbar(im, ax=axs2[1], fraction=0.046)
    for ax in axs2:
        ax.set_xlabel(axlabel); ax.set_ylabel(axlabel.replace('α', 'β'))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig2.suptitle(f'Fitness-normalised (θ₀ = RL-off anchor)   {os.path.basename(os.path.normpath(out_dir))}')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    norm_path = os.path.join(out_dir, 'landscape_normalised.png')
    fig2.savefig(norm_path, dpi=130)
    plt.close(fig2)

    # ── figure: the halo test ─────────────────────────────────────────────────
    # The heatmap cannot resolve the halo per-cell (SEM/cell is comparable to
    # the shape amplitude), but binning 100+ cells shrinks the measurement error
    # by ~10x. This is the panel the Hinton-Nowlan claim should be argued from.
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 4.4))
    # y is plotted as Δ − mean Δ, so the "uniform lift" reference sits at ZERO
    # and the visual question becomes "above or below the uniform lift?" — which
    # is the only part that can reshape the landscape. The Δ=0 line (learning
    # does literally nothing) moves down to −mean Δ.
    panels = [
        (prof_anchor, f'distance from anchor θ₀  (αβ radius)\n'
                      '(no halo prediction — the anchor is not the peak)', 'radius from θ₀', None),
        (prof_peak, f'distance from the RL-off peak (α={peak_a:.1f}, β={peak_b:.1f})'
                    + ('\n⚠ peak is ON THE GRID BOUNDARY — true summit may be outside the window'
                       if peak_on_boundary else ''), 'radius from peak', 'peak'),
        (prof_fit, 'f(θ)|RL off level (hopeless → summit)', 'f(θ) | RL off', 'fitness'),
    ]
    for ax, (prof, title, xlab, sketch_kind) in zip(axs3, panels):
        if not prof:
            continue
        x = np.array([p['x'] for p in prof])
        y = np.array([p['mean'] for p in prof]) - mean_lift      # centred on the uniform lift
        ye = np.array([p['meas_sem'] for p in prof])
        sp = np.array([p['spread'] for p in prof])
        ax.fill_between(x, y - sp, y + sp, color='C0', alpha=0.15, lw=0,
                        label='±1 sd across cells in bin')
        ax.errorbar(x, y, yerr=ye, fmt='o-', color='C0', capsize=3, lw=1.6,
                    label='measured, ± SEM on the bin mean')
        ax.axhline(0, color='grey', ls='--', lw=1.2,
                   label=f'uniform lift (Δ̄={mean_lift:+.3f}) — no reshaping')
        ax.axhline(-mean_lift, color='k', lw=0.9, ls=':',
                   label='Δ = 0 (learning does nothing)')
        if sketch_kind:
            amp = float(np.max(np.abs(y))) or 1.0
            ax.plot(x, _halo_sketch(x, sketch_kind, amp), ':', color='crimson', lw=1.8,
                    label='Hinton–Nowlan halo, predicted SHAPE\n(amplitude arbitrary)')
        ax.set_xlabel(xlab); ax.set_ylabel('Δ − mean Δ')
        ax.set_title(title, fontsize=9)
        for xi, yi, p in zip(x, y, prof):
            ax.annotate(str(p['n']), (xi, yi), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=6, color='grey')
    axs3[1].legend(fontsize=6.5, loc='lower left')
    fig3.suptitle('Hinton–Nowlan halo test — the uniform lift is REMOVED (dashed line at 0), so only '
                  'the reshaping part is shown.\nThe prediction is the crimson hump: ~0 at the summit, '
                  'positive at intermediate distance, ~0 far out. A flat line on the dashed level means '
                  'learning\nlifts fitness uniformly and does NOT reshape the landscape.   '
                  f'{os.path.basename(os.path.normpath(out_dir))}', fontsize=9)
    fig3.tight_layout(rect=[0, 0, 1, 0.84])
    halo_path = os.path.join(out_dir, 'landscape_halo.png')
    fig3.savefig(halo_path, dpi=130)
    plt.close(fig3)

    # ── console summary ───────────────────────────────────────────────────────
    print(f'[stage7] {out_dir}')
    print(f'  lift        mean={mean_lift:+.3f}  shape sd={scalars["lift_shape_sd"]:.3f}  '
          f'SEM/cell={scalars["lift_paired_sem_per_cell"]:.3f}  '
          f'{100*scalars["lift_frac_cells_2sem"]:.0f}% of cells >2·SEM')
    print(f'  σ           mean_evo={scalars["mean_sigma_evo"]:.3f} of which '
          f'{100*scalars["sigma_map_identity_fraction_evo"]:.0f}% is map identity '
          f'(within-map σ={scalars["sigma_within_map_evo"]:.3f});  '
          f'r(σ,f)={scalars["corr_sigma_fitness_evo"]:+.3f}')
    print(f'  σ_evo−σ_learn={scalars["mean_sigma_evo_minus_learn"]:+.3f} '
          f'(scalar only; lag1={scalars["lag1_sigma_diff"]:.2f} — '
          f'{"white noise, not mapped" if abs(scalars["lag1_sigma_diff"]) < 0.3 else "some structure"})')
    print(f'  smoothness  lag1 f_evo={scalars["lag1_f_evo"]:.3f} (blocky≠noisy)  '
          f'lift={scalars["lag1_lift"]:.3f}')
    # Radius from the PEAK is the halo-relevant frame; radius from the anchor
    # mixes summit and hinterland in every bin and can manufacture a fake hump
    # whenever the peak sits off-centre.
    print('  lift vs radius FROM PEAK:  ' +
          '  '.join(f'{p["x"]:.0f}:{p["mean"]:+.3f}' for p in prof_peak))
    print('  lift vs f_evo LEVEL:       ' +
          '  '.join(f'{p["x"]:.1f}:{p["mean"]:+.3f}' for p in prof_fit))
    if peak_on_boundary:
        print(f'  ⚠ f_evo peak is on the grid boundary (α={peak_a:.1f}, β={peak_b:.1f}) — '
              'the summit may lie outside the window; treat the halo test as one flank only.')
    print(f'[stage7] -> {raw_path}\n           {norm_path}\n           {halo_path}')


if __name__ == '__main__':
    main()
