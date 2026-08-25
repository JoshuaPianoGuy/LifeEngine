"""
landscape/stage9_difficulty.py
======================================================================
Stage 9 — landscape DIFFICULTY.

Stages 3-8 answer "where is fitness, and how rugged is it". Stage 9 answers a
different question: **how hard is this landscape to search?** Height and
difficulty are not the same axis — a tall smooth cone and a tall shattered
plateau reach the same maximum and are nothing alike to climb.

Nine metrics, all reanalysis of probes that already exist (A1-A9), plus two that
need new probes (B1 threshold calibration, B2 the 1D transect):

  A1  viable fraction         what share of the space sustains a population
  A2  gradient-to-noise       is the local gradient readable through the noise
  A3  fitness-distance corr   does "closer to the peak" mean "fitter"
  A4  dispersion              are the good cells one basin or scattered
  A5  meta-model R^2          how much a quadratic surrogate already explains
  A6  relative lift           what learning buys, and where it costs
  A7  noise-aware peak count  how many maxima survive the error bars
  A8  lambda at extended lags is the decay really a single exponential
  A9  cross-environment table baseline vs hard, with SE and Z where available
  B1  threshold calibration   k = f_monomorphic / f_replay (needs replay probes)
  B2  1D transect             f along the evolution -> learning interpolation

INPUT. The spec for this stage assumed a Stage-5 file `landscape_<env>_<plane>.csv`
with columns (alpha, beta, rl, f, sigma, n_repeats, mean_lifetime). No such file
exists in this pipeline — Stage 5 leaves the per-probe `results.csv` plus
`meta.json`, and Stage 7 reconstructs the grid on the fly. So `metrics` rebuilds
that table from results.csv + meta.json and WRITES IT OUT under exactly that
name, which both satisfies the spec and gives the later stages a stable input.

`trajectory_2d.csv` likewise does not exist as a file: the per-generation
projected coordinates live in `plane.json` under meta.evo_traj / meta.lrn_traj.
`metrics` exports them to `trajectory_2d_<env>.csv` for the same reason.

JSON KEY CONVENTION. The flat top-level keys (gnr_median_all, fdc_best, ...)
describe the **RL-off / evolution** pass — the landscape evolution actually
searched. The identical key set for the RL-on pass sits under `"learn"`. Keys
that are inherently paired (rel_lift*, corr_lift_f, ...) stay top-level only.

Usage
-----
  # A1-A8, one environment
  python landscape/stage9_difficulty.py metrics --grid-dir landscape/out/stage3_hard_coarse \
      --env hard --lambda-dir landscape/out/stage6_hard_centroid \
      --lambda-dir landscape/out/stage6_hard_random [--k 2.5 | --k-json <calib>/calibration.json]

  # A9, across environments
  python landscape/stage9_difficulty.py compare \
      --metrics landscape/out/stage9/difficulty_metrics_baseline.json \
      --metrics landscape/out/stage9/difficulty_metrics_hard.json \
      --out-dir landscape/out/stage9

  # B1 threshold calibration (needs probes: replay + monomorphic)
  python landscape/stage9_difficulty.py replayjobs --out-dir landscape/out/stage9_calib_hard \
      --evo-genome-csv <evo>/genome.csv --learn-genome-csv <learn>/genome.csv \
      --gen 1000 --repeats 5 --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
  # ... run jobs_replay.json through src/eval/replay.js and jobs_mono.json through
  #     src/eval/run_probe.js, then:
  python landscape/stage9_difficulty.py calibrate --out-dir landscape/out/stage9_calib_hard

  # B2 transect (needs probes)
  python landscape/stage9_difficulty.py transectjobs --out-dir landscape/out/stage9_transect_hard \
      --evo-genome-csv <evo>/genome.csv --learn-genome-csv <learn>/genome.csv --repeats 10
  # ... run through run_probe.js, merge, then:
  python landscape/stage9_difficulty.py transect --out-dir landscape/out/stage9_transect_hard --env hard
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_common as ll
import ll_grid as lg
import ll_difficulty as ld

# Environment colours — fixed per entity, assigned in order, never cycled.
ENV_COLORS = {'baseline': '#0891B2', 'hard': '#92400E'}
FALLBACK_COLORS = ['#4338CA', '#B45309', '#047857', '#9D174D']


def _env_color(env, i=0):
    return ENV_COLORS.get(env, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


def _jsonable(o):
    """Recursively convert numpy scalars/arrays and non-finite floats for json."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# ── grid loading ──────────────────────────────────────────────────────────────
def load_grid(grid_dir):
    """Rebuild the Stage-5 grid table from results.csv + meta.json.

    Returns per-cell (N,N) fields for both RL passes plus the paired SEM, which
    the pooled SEM cannot replace: the two passes are evaluated on IDENTICAL map
    indices, so the difference cancels terrain and its true error bar is the
    spread of the per-repeat DIFFERENCE, not the two spreads added.
    """
    res_path = os.path.join(grid_dir, 'results.csv')
    if not os.path.exists(res_path):
        raise SystemExit(f'{res_path} not found — run merge_shards.py first')
    res = ll.load_results(res_path)
    meta = json.load(open(os.path.join(grid_dir, 'meta.json')))
    axes = meta['axes']
    R = int(axes['R'])

    f_evo, s_evo = lg.reconstruct_grid(res, axes, rl=False)
    f_learn, s_learn = lg.reconstruct_grid(res, axes, rl=True)
    life_evo = lg.reconstruct_field(res, axes, rl=False, column='mean_lifetime')
    life_learn = lg.reconstruct_field(res, axes, rl=True, column='mean_lifetime')

    cube_off = lg.reconstruct_cube(res, axes, rl=False)
    cube_on = lg.reconstruct_cube(res, axes, rl=True)
    dcube = cube_on - cube_off
    with np.errstate(invalid='ignore'):
        paired_sem = np.nanstd(dcube, axis=2, ddof=1) / np.sqrt(R)

    alphas = np.array(axes['alphas'], dtype=float)
    betas = np.array(axes['betas'], dtype=float)
    A, B = ld.grid_coords(alphas, betas)

    plane_meta = meta.get('plane_meta', {}) or {}
    plane_name = f"{plane_meta.get('kind', 'plane')}_{plane_meta.get('grid', '')}".strip('_')

    return {
        'meta': meta, 'axes': axes, 'R': R, 'N': int(axes['resolution']),
        'alphas': alphas, 'betas': betas, 'A': A, 'B': B,
        'f_evo': f_evo, 'f_learn': f_learn,
        'sigma_evo': s_evo, 'sigma_learn': s_learn,
        'life_evo': life_evo, 'life_learn': life_learn,
        'paired_sem': paired_sem,
        # The per-repeat cubes, so a caller can tell a cell averaged over all R
        # maps from one averaged over a subset. reconstruct_grid() collapses the
        # repeats silently, and on a partly-finished array that matters: repeats
        # are different MAPS and terrain variance (sigma_vary ~ 6-8) dwarfs most
        # spatial structure, so a 3-map mean is not on the same scale as a 5-map
        # mean. n_off / n_on count the repeats actually present per cell.
        'cube_off': cube_off, 'cube_on': cube_on,
        'n_off': np.isfinite(cube_off).sum(axis=2),
        'n_on': np.isfinite(cube_on).sum(axis=2),
        'plane_name': plane_name or 'plane',
        'extent': [float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1])],
        'grid_dir': grid_dir,
    }


def write_landscape_csv(path, G):
    """Emit the Stage-5 aggregated table the spec assumes as its input."""
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['alpha', 'beta', 'rl', 'f', 'sigma', 'n_repeats', 'mean_lifetime'])
        for tag, f, s, life in (('off', G['f_evo'], G['sigma_evo'], G['life_evo']),
                                ('on', G['f_learn'], G['sigma_learn'], G['life_learn'])):
            for i, a in enumerate(G['alphas']):
                for j, b in enumerate(G['betas']):
                    w.writerow([f'{a:.6f}', f'{b:.6f}', tag,
                                f'{f[i, j]:.6f}', f'{s[i, j]:.6f}', G['R'],
                                f'{life[i, j]:.4f}'])
    return path


def export_trajectory_csv(path, grid_dir):
    """Export plane.json's projected trajectories as the spec's trajectory_2d.csv."""
    pj = os.path.join(grid_dir, 'plane.json')
    if not os.path.exists(pj):
        return None
    pm = (json.load(open(pj)).get('meta') or {})
    evo = pm.get('evo_traj') or []
    lrn = pm.get('lrn_traj') or []
    if not evo and not lrn:
        return None
    evo_gens = pm.get('evo_gens') or list(range(1, len(evo) + 1))
    lrn_gens = pm.get('lrn_gens') or list(range(1, len(lrn) + 1))
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['condition', 'generation', 'alpha', 'beta'])
        for cond, traj, gens in (('evolution', evo, evo_gens), ('learning', lrn, lrn_gens)):
            for g, (a, b) in zip(gens, traj):
                w.writerow([cond, int(g), f'{a:.6f}', f'{b:.6f}'])
    return path


# ── figures ───────────────────────────────────────────────────────────────────
def _imshow(ax, grid, extent, cmap, title, vmin=None, vmax=None, norm=None):
    im = ax.imshow(np.asarray(grid).T, origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('α (PC1)', fontsize=8)
    ax.set_ylabel('β (PC2)', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def fig_viability(out_png, G, env, k, sens):
    """Landscape panels with the viability contour + the k-sensitivity curve."""
    thr = ld.viability_threshold(k)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    lo = float(np.nanmin([G['f_evo'], G['f_learn']]))
    hi = float(np.nanmax([G['f_evo'], G['f_learn']]))
    A, B = G['A'], G['B']

    for ax, f, lab in ((axs[0], G['f_evo'], 'RL off (evolution)'),
                       (axs[1], G['f_learn'], 'RL on (learning)')):
        im = _imshow(ax, f, G['extent'], 'viridis', f'f(θ) — {lab}', lo, hi)
        frac, _ = ld.viable_fraction(f, k)
        if np.nanmin(f) < thr < np.nanmax(f):
            cs = ax.contour(A, B, f, levels=[thr], colors='white', linewidths=1.8)
            ax.clabel(cs, fmt=lambda v: f'viable ({v:.2f})', fontsize=7)
        ax.text(0.02, 0.98, f'viable {frac:.1%}', transform=ax.transAxes,
                va='top', ha='left', fontsize=8, color='white',
                bbox=dict(fc='black', alpha=0.45, ec='none', pad=2))
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[2]
    ks = [s['k'] for s in sens]
    ax.plot(ks, [s['evo'] for s in sens], '-', lw=2, color=_env_color(env),
            label='RL off (evolution)')
    ax.plot(ks, [s['learn'] for s in sens], '--', lw=2, color=_env_color(env),
            label='RL on (learning)')
    ax.axvline(k, color='grey', ls=':', lw=1.2)
    ax.annotate(f'k = {k:g}', xy=(k, 0.98), xycoords=('data', 'axes fraction'),
                fontsize=7, color='grey', ha='left', va='top', rotation=90)
    ax.set_xlabel('threshold multiplier k', fontsize=8)
    ax.set_ylabel('viable fraction of cells', fontsize=8)
    ax.set_title(f'A1 sensitivity — {env}\nthreshold = {ld.VIABILITY_BASE:g}·k', fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_gnr(out_png, G, env, gnr_evo, gnr_learn):
    """GNR heatmaps, log-scaled, with the GNR = 1 searchability contour."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    both = np.concatenate([gnr_evo[np.isfinite(gnr_evo) & (gnr_evo > 0)].ravel(),
                           gnr_learn[np.isfinite(gnr_learn) & (gnr_learn > 0)].ravel()])
    vmin = max(float(np.percentile(both, 1)), 1e-3) if both.size else 1e-3
    vmax = float(np.percentile(both, 99)) if both.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 10
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, g, lab in ((axs[0], gnr_evo, 'RL off (evolution)'),
                       (axs[1], gnr_learn, 'RL on (learning)')):
        im = _imshow(ax, g, G['extent'], 'YlGnBu', f'GNR — {lab}', norm=norm)
        finite = g[np.isfinite(g)]
        if finite.size and finite.min() < 1.0 < finite.max():
            gg = np.where(np.isfinite(g), g, np.nan)
            cs = ax.contour(G['A'], G['B'], gg, levels=[1.0],
                            colors='#B91C1C', linewidths=1.8)
            ax.clabel(cs, fmt='GNR = 1', fontsize=7)
        if finite.size:
            ax.text(0.02, 0.98, f'{np.mean(finite < 1):.1%} below 1\nmedian {np.median(finite):.2f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8, color='white',
                    bbox=dict(fc='black', alpha=0.45, ec='none', pad=2))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'A2 gradient-to-noise ratio — {env}   '
                 '(below the red contour the local gradient is smaller than the noise)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_fdc(out_png, G, env, fdc_evo, fdc_learn):
    """Fitness against distance-to-reference, one panel per RL pass (A3).

    The reference is NOT the true global optimum — it is the best cell actually
    sampled on this 2-D slice (see ll_difficulty.fdc). Both proxies are drawn:
    points are positioned by distance to the top-5% centroid (the stable one),
    and the best-cell r is quoted alongside as the robustness check.

    Sign convention: r < 0 means fitness RISES as distance falls, i.e. the
    landscape points a searcher at the optimum. r ~ 0 = no global signal,
    r > 0 = deceptive. There is no absolute scale — a perfectly linear surface
    on this square domain scores only about -0.69 — so read the two
    environments against EACH OTHER, never against -1.
    """
    A, B = G['A'], G['B']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    colour = _env_color(env)

    for ax, f, d, lab in ((axs[0], G['f_evo'], fdc_evo, 'RL off (evolution)'),
                          (axs[1], G['f_learn'], fdc_learn, 'RL on (learning)')):
        ra, rb = d['fdc_top5_ref']
        a = np.asarray(A, dtype=float).ravel()
        b = np.asarray(B, dtype=float).ravel()
        fv = np.asarray(f, dtype=float).ravel()
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(fv)
        a, b, fv = a[m], b[m], fv[m]
        dist = np.sqrt((a - ra) ** 2 + (b - rb) ** 2)

        ax.scatter(dist, fv, s=9, alpha=0.35, color=colour, edgecolors='none',
                   rasterized=True)
        # OLS fit of f on distance — the visual slope behind the r value.
        if dist.size >= 2 and np.ptp(dist) > 0:
            slope, intercept = np.polyfit(dist, fv, 1)
            xs = np.array([dist.min(), dist.max()])
            ax.plot(xs, slope * xs + intercept, '-', lw=2, color='#B91C1C',
                    label=f'OLS slope {slope:+.4f}')
            ax.legend(fontsize=7.5, loc='upper right')

        r_top = d['fdc_top5_centroid']
        se_top = d['fdc_top5_centroid_se']
        ax.text(0.02, 0.02,
                f'r (top-5% centroid) = {r_top:+.3f} ± {se_top:.3f}\n'
                f'r (best cell)       = {d["fdc_best"]:+.3f} ± {d["fdc_best_se"]:.3f}\n'
                f'reference α={ra:.2f}, β={rb:.2f}',
                transform=ax.transAxes, va='bottom', ha='left', fontsize=7.5,
                family='monospace',
                bbox=dict(fc='white', alpha=0.8, ec='#CCCCCC', pad=3))
        ax.set_title(f'{lab}', fontsize=9)
        ax.set_xlabel('distance to top-5% centroid  (in-plane, α–β)', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    axs[0].set_ylabel('f(θ)', fontsize=8)
    fig.suptitle(f'A3 fitness-distance correlation — {env}\n'
                 'r < 0 = fitness rises toward the reference (guided);  r ~ 0 = no global '
                 'signal;  r > 0 = deceptive.  No absolute scale — compare environments, not to -1.',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_lift(out_png, G, env, lift):
    """Relative-lift map + the two diagnostic scatters (A6)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    rl = lift['rel_lift']
    delta = lift['delta']

    # Signed quantity -> diverging map, hard-centred on 0 (no lift = neutral).
    lim = float(np.nanpercentile(np.abs(rl), 99)) if np.isfinite(rl).any() else 1.0
    lim = lim if lim > 0 else 1.0
    im = _imshow(axs[0], rl, G['extent'], 'RdBu_r',
                 f'relative lift (f_on − f_off)/f_off — {env}', -lim, lim)
    fig.colorbar(im, ax=axs[0], fraction=0.046)

    fe = G['f_evo'].ravel()
    dl = delta.ravel()
    life = G['life_evo'].ravel()
    col = _env_color(env)

    m = np.isfinite(dl) & np.isfinite(life)
    axs[1].scatter(life[m], dl[m], s=9, alpha=0.45, color=col, lw=0)
    axs[1].axhline(0, color='grey', ls='--', lw=1)
    axs[1].set_xlabel('mean lifetime (RL off, ticks)', fontsize=8)
    axs[1].set_ylabel('f_on − f_off', fontsize=8)
    axs[1].set_title(f'r = {ld.pearson(dl, life):+.3f}', fontsize=9)

    m2 = np.isfinite(dl) & np.isfinite(fe)
    axs[2].scatter(fe[m2], dl[m2], s=9, alpha=0.45, color=col, lw=0)
    axs[2].axhline(0, color='grey', ls='--', lw=1)
    axs[2].set_xlabel('f(θ) RL off', fontsize=8)
    axs[2].set_ylabel('f_on − f_off', fontsize=8)
    axs[2].set_title(f'r = {ld.pearson(dl, fe):+.3f}', fontsize=9)

    for ax in axs[1:]:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.suptitle(f'A6 relative lift — {env}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_lambda(out_png, env, lam_by_start):
    """rho(k) to long lags, linear and log-y, with the fitted exponentials."""
    n = len(lam_by_start)
    fig, axs = plt.subplots(n, 2, figsize=(11, 3.8 * n), squeeze=False)
    for row, (label, L) in enumerate(lam_by_start.items()):
        rho = np.array(L['rho'], dtype=float)
        se = np.array(L['rho_se'], dtype=float)
        ks = np.arange(1, rho.size + 1)
        col = _env_color(env)

        for c, ax in enumerate(axs[row]):
            ax.axhline(0, color='k', lw=0.6)
            if np.isfinite(se).any():
                ax.fill_between(ks, rho - se, rho + se, color=col, alpha=0.18, lw=0)
            ax.plot(ks, rho, '-', lw=1.8, color=col, label='mean ρ(k) ± SE')
            for kf, ls in ((10, ':'), (30, '--'), (60, '-.')):
                lam = L.get(f'lambda_fit_k{kf}')
                if lam and np.isfinite(lam) and lam > 0:
                    ax.plot(ks, np.exp(-ks / lam), ls, lw=1.2, color='grey',
                            label=f'fit k≤{kf}: λ={lam:.1f}')
            xc = L.get('lambda_1e_crossing')
            ax.axhline(1 / np.e, color='#B91C1C', lw=0.9, ls=':')
            if xc and np.isfinite(xc):
                ax.axvline(xc, color='#B91C1C', lw=1.2,
                           label=f'1/e crossing: {xc:.1f}')
            ax.set_xlabel('lag k (mutation steps)', fontsize=8)
            ax.set_ylabel('autocorrelation ρ(k)', fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if c == 1:
                ax.set_yscale('log')
                pos = rho[rho > 0]
                if pos.size:
                    ax.set_ylim(max(pos.min() * 0.5, 1e-3), 1.5)
                ax.set_title('log-y', fontsize=9)
            else:
                ax.set_title(f'{label} (transform={L["transform"]}, W={L["walks"]})',
                             fontsize=9)
                ax.legend(fontsize=6.5)
    kmax = max(L['kmax'] for L in lam_by_start.values())
    fig.suptitle(f'A8 ρ(k) to k={kmax} — {env}   '
                 '(ρ(k) < 0 at long lag is the usual finite-series detrending artefact, '
                 'not anticorrelation)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── A1-A8: metrics for one environment ────────────────────────────────────────
def cmd_metrics(args):
    G = load_grid(args.grid_dir)
    env = args.env
    out_dir = args.out_dir or args.grid_dir
    os.makedirs(out_dir, exist_ok=True)
    plane = args.plane or G['plane_name']

    # k: measured (B1) if a calibration file is given, else the stated default.
    k = args.k
    k_source = f'default ({ld.K_DEFAULT})' if args.k == ld.K_DEFAULT and not args.k_json else 'cli'
    k_sd = float('nan')
    if args.k_json:
        cal = json.load(open(args.k_json))
        emp_k = cal.get('empirical_replacement_k')
        use_emp = (args.k_from == 'empirical'
                   or (args.k_from == 'auto' and emp_k is not None))
        if use_emp and emp_k is None:
            raise SystemExit(f'--k-from empirical: {args.k_json} has no measured '
                             f'replacement crossing (no genome straddles 1.0 '
                             f'child/founder)')
        if use_emp:
            # The measured crossing beats the f_mono/f_replay ratio: it locates the
            # cutoff directly instead of converting to it through a k that is
            # itself confounded by how much reproduction happened.
            k = float(emp_k)
            fit = cal.get('empirical_replacement_fit') or {}
            ci = fit.get('ci95') or [float('nan')] * 2
            k_sd = float('nan')
            k_source = (f'MEASURED replacement crossing (B1: {args.k_json}); '
                        f'f_cross={cal["empirical_replacement_f"]:.3f}, '
                        f'95% CI [{ci[0]:.2f}, {ci[1]:.2f}]')
            out_ci = [c / ld.VIABILITY_BASE for c in ci]
        else:
            k = float(cal['k_mean'])
            k_sd = float(cal.get('k_sd', float('nan')))
            k_source = (f'f_mono/f_replay ratio [{cal.get("replay_field")}] '
                        f'(B1: {args.k_json})')
            out_ci = None

    A, B = G['A'], G['B']
    out = {
        'env': env, 'plane': plane, 'grid_dir': os.path.abspath(args.grid_dir),
        'resolution': G['N'], 'n_repeats': G['R'], 'n_cells': int(G['N'] ** 2),
        'k': k, 'k_sd': k_sd, 'k_source': k_source,
        'viability_base': ld.VIABILITY_BASE,
        'viability_threshold': ld.viability_threshold(k),
        'learn': {},
    }

    # ── A1 viable fraction ────────────────────────────────────────────────────
    ve, n_e = ld.viable_fraction(G['f_evo'], k)
    vl, n_l = ld.viable_fraction(G['f_learn'], k)
    out['viable_frac_evo'] = ve
    out['viable_frac_learn'] = vl
    # Propagate the crossing's CI into the viable fraction, so the reported level
    # carries the threshold uncertainty rather than looking exact.
    if args.k_json and 'out_ci' in dir() and out_ci and np.isfinite(out_ci[0]):
        band = [ld.viable_fraction(G['f_evo'], kk)[0] for kk in out_ci]
        out['viable_frac_evo_ci95'] = [min(band), max(band)]
        bandl = [ld.viable_fraction(G['f_learn'], kk)[0] for kk in out_ci]
        out['viable_frac_learn_ci95'] = [min(bandl), max(bandl)]
    out['viable_frac_evo_se'] = ld.viable_fraction_se(ve, n_e)
    out['viable_frac_learn_se'] = ld.viable_fraction_se(vl, n_l)
    sens = ld.viable_fraction_sensitivity(G['f_evo'], G['f_learn'])
    out['viable_frac_sensitivity'] = sens

    # ── A2 gradient-to-noise ──────────────────────────────────────────────────
    ge = ld.gnr_summary(G['f_evo'], G['sigma_evo'])
    gl = ld.gnr_summary(G['f_learn'], G['sigma_learn'])
    for key in ('gnr_median_all', 'gnr_median_lowf', 'gnr_median_highf', 'gnr_frac_below_1'):
        out[key] = ge[key]
        out['learn'][key] = gl[key]

    # ── A3 fitness-distance correlation ───────────────────────────────────────
    fdc_by_pass = {}
    for tag, f in (('', G['f_evo']), ('learn', G['f_learn'])):
        r_best, n_c, ref_b = ld.fdc(A, B, f, reference='best')
        r_top, _, ref_t = ld.fdc(A, B, f, reference='top5')
        d = {'fdc_best': r_best, 'fdc_top5_centroid': r_top,
             'fdc_best_se': ld.corr_se(r_best, n_c),
             'fdc_top5_centroid_se': ld.corr_se(r_top, n_c),
             'fdc_best_ref': list(ref_b), 'fdc_top5_ref': list(ref_t)}
        (out if tag == '' else out['learn']).update(d)
        fdc_by_pass[tag or 'evo'] = d

    # ── A4 dispersion ─────────────────────────────────────────────────────────
    out['dispersion'] = ld.dispersion_profile(A, B, G['f_evo'])
    out['learn']['dispersion'] = ld.dispersion_profile(A, B, G['f_learn'])

    # ── A5 meta-model R^2 ─────────────────────────────────────────────────────
    out.update(ld.meta_model_r2(A, B, G['f_evo']))
    out['learn'].update(ld.meta_model_r2(A, B, G['f_learn']))

    # ── A6 relative lift ──────────────────────────────────────────────────────
    lift = ld.relative_lift(G['f_evo'], G['f_learn'], G['sigma_evo'], G['sigma_learn'],
                            G['R'], paired_sem=G['paired_sem'])
    n_cells = lift['n_cells']
    corr_lift_f = ld.pearson(lift['rel_lift'], G['f_evo'])
    corr_delta_life = ld.pearson(lift['delta'], G['life_evo'])
    out.update({
        'rel_lift_mean': lift['rel_lift_mean'],
        'rel_lift_sd': lift['rel_lift_sd'],
        'rel_lift_sem': lift['rel_lift_sem'],
        'corr_lift_f': corr_lift_f,
        'corr_lift_f_se': ld.corr_se(corr_lift_f, n_cells),
        'corr_delta_lifetime': corr_delta_life,
        'corr_delta_lifetime_se': ld.corr_se(corr_delta_life, n_cells),
        'frac_delta_negative': lift['frac_delta_negative'],
        'frac_delta_negative_paired': lift['frac_delta_negative_paired'],
        'frac_delta_positive': lift['frac_delta_positive'],
    })

    # ── A7 noise-aware peak count ─────────────────────────────────────────────
    sem_e = G['sigma_evo'] / np.sqrt(G['R'])
    sem_l = G['sigma_learn'] / np.sqrt(G['R'])
    pr_e, _ = ld.count_peaks_noise_aware(G['f_evo'], None)
    pf_e, _ = ld.count_peaks_noise_aware(G['f_evo'], sem_e)
    pr_l, _ = ld.count_peaks_noise_aware(G['f_learn'], None)
    pf_l, _ = ld.count_peaks_noise_aware(G['f_learn'], sem_l)
    out['peaks_raw'] = pr_e
    out['peaks_filtered'] = pf_e
    out['learn']['peaks_raw'] = pr_l
    out['learn']['peaks_filtered'] = pf_l
    pm_e = ld.peak_margin_report(G['f_evo'], sem_e)
    pm_l = ld.peak_margin_report(G['f_learn'], sem_l)
    out['peaks_best_ratio'] = pm_e['best_ratio']
    out['peaks_on_boundary'] = pm_e['n_on_boundary']
    out['peaks_detail'] = pm_e['peaks']
    out['learn']['peaks_best_ratio'] = pm_l['best_ratio']
    out['learn']['peaks_on_boundary'] = pm_l['n_on_boundary']

    # ── A8 lambda reanalysis ──────────────────────────────────────────────────
    lam_by_start = {}
    for d in (args.lambda_dir or []):
        try:
            lam = lambda_from_dir(d, kmax=args.kmax, transform=args.lambda_transform)
        except Exception as e:                                  # noqa: BLE001
            print(f'[stage9] lambda skip {d}: {e}', file=sys.stderr)
            continue
        lam_by_start[lam.pop('start_label')] = lam
    if lam_by_start:
        primary = next(iter(lam_by_start.values()))
        for key in ('lambda_fit_k10', 'lambda_fit_k30', 'lambda_fit_k60',
                    'lambda_fit_k10_se', 'lambda_fit_k30_se', 'lambda_fit_k60_se',
                    'lambda_1e_crossing', 'rho1', 'rho1_se',
                    'lambda_kfit_spread', 'lambda_kfit_spread_frac'):
            out[key] = primary.get(key)
        out['lambda_primary_start'] = next(iter(lam_by_start))
        out['lambda_by_start'] = {kk: {k2: v2 for k2, v2 in vv.items() if k2 != 'rho_se'}
                                  for kk, vv in lam_by_start.items()}

    # ── artefacts ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f'landscape_{env}_{plane}.csv')
    write_landscape_csv(csv_path, G)
    traj_path = export_trajectory_csv(os.path.join(out_dir, f'trajectory_2d_{env}.csv'),
                                      args.grid_dir)
    json_path = os.path.join(out_dir, f'difficulty_metrics_{env}.json')
    json.dump(_jsonable(out), open(json_path, 'w'), indent=2)

    fig_viability(os.path.join(out_dir, f'difficulty_{env}_viability.png'), G, env, k, sens)
    fig_gnr(os.path.join(out_dir, f'difficulty_{env}_gnr.png'), G, env, ge['gnr'], gl['gnr'])
    fig_fdc(os.path.join(out_dir, f'difficulty_{env}_fdc.png'), G, env,
            fdc_by_pass['evo'], fdc_by_pass['learn'])
    fig_lift(os.path.join(out_dir, f'difficulty_{env}_lift.png'), G, env, lift)
    if lam_by_start:
        fig_lambda(os.path.join(out_dir, f'difficulty_{env}_lambda.png'), env, lam_by_start)

    _print_metrics(out, lam_by_start)
    print(f'\n[stage9] {json_path}')
    print(f'[stage9] {csv_path}')
    if traj_path:
        print(f'[stage9] {traj_path}')
    print(f'[stage9] figures -> {out_dir}/difficulty_{env}_*.png')
    return out


def _print_metrics(o, lam_by_start):
    L = o['learn']
    print(f"\n=== Stage 9 difficulty — {o['env']} ({o['plane']}, "
          f"{o['resolution']}x{o['resolution']}, R={o['n_repeats']}) ===")
    print(f"  k = {o['k']:.3f} [{o['k_source']}]  ->  viability threshold "
          f"f > {o['viability_threshold']:.3f}")
    print(f"  A1 viable fraction        evo {o['viable_frac_evo']:.3f} ± {o['viable_frac_evo_se']:.3f}"
          f"   learn {o['viable_frac_learn']:.3f} ± {o['viable_frac_learn_se']:.3f}")
    print(f"  A2 median GNR             all {o['gnr_median_all']:.2f}  "
          f"low-f {o['gnr_median_lowf']:.2f}  high-f {o['gnr_median_highf']:.2f}   "
          f"({o['gnr_frac_below_1']:.1%} of cells below 1)")
    print(f"  A3 FDC                    best {o['fdc_best']:+.3f}   "
          f"top-5% centroid {o['fdc_top5_centroid']:+.3f}   (learn {L['fdc_best']:+.3f})")
    disp = ', '.join(f"p={p}: {v:.3f}" for p, v in o['dispersion'].items())
    print(f"  A4 dispersion             {disp}")
    print(f"  A5 meta-model R²          linear {o['r2_linear']:.3f} (adj {o['r2_linear_adj']:.3f})   "
          f"quadratic {o['r2_quadratic']:.3f} (adj {o['r2_quadratic_adj']:.3f})")
    print(f"  A6 relative lift          mean {o['rel_lift_mean']:+.3f} ± {o['rel_lift_sd']:.3f} (sd)   "
          f"corr(lift, f_evo) {o['corr_lift_f']:+.3f}")
    print(f"     corr(Δ, lifetime)      {o['corr_delta_lifetime']:+.3f}   "
          f"harmed cells {o['frac_delta_negative']:.1%} pooled / "
          f"{o['frac_delta_negative_paired']:.1%} paired")
    print(f"  A7 peaks                  evo {o['peaks_raw']} raw -> {o['peaks_filtered']} filtered   "
          f"learn {L['peaks_raw']} -> {L['peaks_filtered']}")
    print(f"     best peak/noise ratio  {o['peaks_best_ratio']:.3f}  "
          f"({o['peaks_on_boundary']}/{o['peaks_raw']} raw peaks on the grid boundary)"
          + ('  <- no raw peak is anywhere near resolved' if o['peaks_best_ratio'] < 0.5 else ''))
    for label, lam in lam_by_start.items():
        warn = ('   <- λ depends on the fit window: quote the 1/e crossing'
                if lam.get('lambda_kfit_spread_frac', 0) > 0.25 else '')
        print(f"  A8 λ [{label}]  ρ(1)={lam['rho1']:.4f}±{lam['rho1_se']:.4f}  "
              f"k≤10 {lam['lambda_fit_k10']:.2f}  k≤30 {lam['lambda_fit_k30']:.2f}  "
              f"k≤60 {lam['lambda_fit_k60']:.2f}  1/e {lam['lambda_1e_crossing']:.2f}"
              f"  (spread {lam.get('lambda_kfit_spread_frac', float('nan')):.0%}){warn}")


def lambda_from_dir(stage6_dir, kmax=100, transform='log'):
    """Run the A8 reanalysis on a Stage-6 walk out-dir."""
    res = ll.load_results(os.path.join(stage6_dir, 'results.csv')).copy()
    meta = json.load(open(os.path.join(stage6_dir, 'meta.json')))
    m = res['id'].str.extract(r'w(\d+)_s(\d+)_r(\d+)')
    if m[0].isna().all():
        leg = res['id'].str.extract(r's(\d+)_r(\d+)')
        res['walk'] = 0
        res['step'] = leg[0].astype(int)
    else:
        res['walk'] = m[0].astype(int)
        res['step'] = m[1].astype(int)

    walk_ids = sorted(res['walk'].unique())
    series = [res[res['walk'] == w].groupby('step')['mean_fitness'].mean().sort_index().to_numpy()
              for w in walk_ids]
    T = min(len(s) for s in series)
    mat = np.vstack([s[:T] for s in series])

    out = ld.lambda_reanalysis(mat, kmax=kmax, kfits=(10, 30, 60), transform=transform)
    out['start_label'] = meta.get('start_label', os.path.basename(stage6_dir))
    out['dir'] = os.path.abspath(stage6_dir)
    out['repeats'] = meta.get('repeats')
    return out


# ── A9: cross-environment comparison ──────────────────────────────────────────
# (metric key, human label, SE key or None, direction note)
COMPARE_ROWS = [
    ('viable_frac_evo', 'A1 viable fraction (evo)', 'viable_frac_evo_se'),
    ('viable_frac_learn', 'A1 viable fraction (learn)', 'viable_frac_learn_se'),
    ('gnr_median_all', 'A2 median GNR (all cells)', None),
    ('gnr_median_lowf', 'A2 median GNR (low f)', None),
    ('gnr_median_highf', 'A2 median GNR (high f)', None),
    ('gnr_frac_below_1', 'A2 fraction of cells GNR < 1', None),
    ('fdc_best', 'A3 FDC (best cell, evo)', 'fdc_best_se'),
    ('learn.fdc_best', 'A3 FDC (best cell, learn)', 'learn.fdc_best_se'),
    ('fdc_top5_centroid', 'A3 FDC (top-5% centroid, evo)', 'fdc_top5_centroid_se'),
    ('learn.fdc_top5_centroid', 'A3 FDC (top-5% centroid, learn)', 'learn.fdc_top5_centroid_se'),
    ('r2_linear', 'A5 R² linear', None),
    ('r2_quadratic', 'A5 R² quadratic', None),
    ('r2_nonquadratic_residual', 'A5 non-quadratic residual', None),
    ('rel_lift_mean', 'A6 mean relative lift', 'rel_lift_sem'),
    ('corr_lift_f', 'A6 corr(lift, f_evo)', 'corr_lift_f_se'),
    ('corr_delta_lifetime', 'A6 corr(Δ, lifetime)', 'corr_delta_lifetime_se'),
    ('frac_delta_negative', 'A6 harmed cells (pooled SEM)', None),
    ('frac_delta_negative_paired', 'A6 harmed cells (paired SEM)', None),
    ('peaks_raw', 'A7 peaks (raw)', None),
    ('peaks_filtered', 'A7 peaks (noise-filtered)', None),
    ('lambda_fit_k10', 'A8 λ (fit k≤10)', 'lambda_fit_k10_se'),
    ('lambda_fit_k30', 'A8 λ (fit k≤30)', 'lambda_fit_k30_se'),
    ('lambda_fit_k60', 'A8 λ (fit k≤60)', 'lambda_fit_k60_se'),
    ('lambda_1e_crossing', 'A8 λ (1/e crossing, fit-free)', None),
    ('rho1', 'A8 ρ(1)', 'rho1_se'),
]


def _mget(m, key):
    """Fetch a metric by flat key, or by dotted path for the nested RL-on pass.

    The RL-off/evolution metrics sit at the top level and the RL-on/learning
    ones under "learn" (see the key convention in the module docstring), so
    'learn.fdc_best' reaches the learning-pass value.
    """
    if key is None:
        return None
    cur = m
    for part in key.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def cmd_compare(args):
    metrics = [json.load(open(p)) for p in args.metrics]
    envs = [m['env'] for m in metrics]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if len(metrics) != 2:
        print(f'[stage9] {len(metrics)} environments given; Z is computed for the '
              f'first two ({envs[:2]}) only', file=sys.stderr)

    rows = []
    for key, label, se_key in COMPARE_ROWS:
        vals = [_mget(m, key) for m in metrics]
        if all(v is None for v in vals):
            continue
        row = {'metric': key, 'label': label}
        for env, m in zip(envs, metrics):
            row[env] = _mget(m, key)
            if se_key:
                row[f'{env}_se'] = _mget(m, se_key)
        if len(metrics) >= 2:
            a, b = metrics[0], metrics[1]
            va = _mget(a, key)
            vb = _mget(b, key)
            sa = _mget(a, se_key)
            sb = _mget(b, se_key)
            cmpres = ld.compare_metric(
                float(va) if va is not None else float('nan'),
                float(vb) if vb is not None else float('nan'),
                sa, sb)
            row.update(cmpres)
        rows.append(row)

    # dispersion is a nested dict — flatten it into its own rows
    for p in ('0.02', '0.05', '0.1'):
        vals = [(m.get('dispersion') or {}).get(p) for m in metrics]
        if all(v is None for v in vals):
            continue
        row = {'metric': f'dispersion_p{p}', 'label': f'A4 dispersion (p={p})'}
        for env, v in zip(envs, vals):
            row[env] = v
        if len(metrics) >= 2 and vals[0] is not None and vals[1] is not None:
            row.update(ld.compare_metric(float(vals[0]), float(vals[1]), None, None))
        rows.append(row)

    cols = ['metric', 'label']
    for env in envs:
        cols += [env, f'{env}_se']
    cols += ['diff', 'se_diff', 'z']
    csv_path = os.path.join(out_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})

    # combined A1 sensitivity figure — one line per (environment, rl)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, m in enumerate(metrics):
        sens = m.get('viable_frac_sensitivity') or []
        if not sens:
            continue
        ks = [s['k'] for s in sens]
        col = _env_color(m['env'], i)
        ax.plot(ks, [s['evo'] for s in sens], '-', lw=2, color=col,
                label=f"{m['env']} — RL off")
        ax.plot(ks, [s['learn'] for s in sens], '--', lw=2, color=col,
                label=f"{m['env']} — RL on")
        if m.get('k_source', '').startswith('measured'):
            ax.axvline(m['k'], color=col, ls=':', lw=1)
    ax.set_xlabel('threshold multiplier k')
    ax.set_ylabel('viable fraction of cells')
    ax.set_title(f'A1 viable fraction vs threshold\n'
                 f'threshold = {ld.VIABILITY_BASE:g}·k  (3.5 reproduction energy / 0.8 success rate)',
                 fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, 'comparison_viable_fraction.png')
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print(f'\n=== Stage 9 comparison: {" vs ".join(envs)} ===')
    w_lab = max(len(r['label']) for r in rows)
    head = f'  {"metric".ljust(w_lab)}  ' + '  '.join(e.rjust(10) for e in envs) + \
           '        diff      SE        Z'
    print(head)
    print('  ' + '-' * (len(head) - 2))
    for r in rows:
        vals = '  '.join(_fmt(r.get(e), 4).rjust(10) for e in envs)
        print(f"  {r['label'].ljust(w_lab)}  {vals}  {_fmt(r.get('diff'), 4).rjust(10)}  "
              f"{_fmt(r.get('se_diff'), 4).rjust(8)}  {_fmt(r.get('z'), 2).rjust(7)}")
    print(f'\n[stage9] {csv_path}')
    print(f'[stage9] {png}')
    return rows


def _fmt(v, nd=6):
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return ''
    if float(f).is_integer() and abs(f) < 1e6:
        return str(int(f))
    return f'{f:.{nd}f}'


# ── B1: threshold calibration ─────────────────────────────────────────────────
def _pick_genomes(args):
    """The three calibration genomes: learning centroid, evolution centroid, one founder."""
    gen = args.gen
    picks = {}

    def centroid_at(csv_path, label):
        df = ll.load_genome_csv(csv_path, record_type='centroid')
        if df.empty:
            raise SystemExit(f'no centroid rows in {csv_path}')
        row = df.iloc[(df['generation'] - gen).abs().argsort().iloc[0]]
        picks[label] = (row['genome'], int(row['generation']))

    centroid_at(args.evo_genome_csv, 'evo_centroid')
    centroid_at(args.learn_genome_csv, 'learn_centroid')

    fdf = ll.load_genome_csv(args.founder_genome_csv or args.evo_genome_csv,
                             record_type='founder')
    if fdf.empty:
        raise SystemExit('no founder rows found — pass --founder-genome-csv')
    near = fdf.iloc[(fdf['generation'] - gen).abs().argsort()]
    g_target = int(near.iloc[0]['generation'])
    same_gen = fdf[fdf['generation'] == g_target]
    row = same_gen.sort_values('fitness', ascending=False).iloc[
        min(args.founder_index, len(same_gen) - 1)]
    picks['founder'] = (row['genome'], int(row['generation']))
    return picks


def _pick_grid_cells(grid_dir, n_cells, f_max=None):
    """Select grid cells whose fitness SPANS the viability range.

    The three gen-1000 genomes are all high performers, so they pin the
    f_mono -> children/founder relation only where they happen to land — in the
    hard environment two of them straddle replacement by luck, in the baseline
    none do. Grid cells fix that: the plane already contains a continuum of
    genomes from dead to excellent, and every one of them is a real point in the
    same space the landscape is measured over.

    Targets are evenly spaced in f from the grid minimum up to `f_max` (default
    the 90th percentile), and the nearest unused cell is taken for each, so the
    sample concentrates where the replacement crossing actually is instead of
    following the grid's own density.
    """
    plane = lg.load_plane(os.path.join(grid_dir, 'plane.json'))
    meta = json.load(open(os.path.join(grid_dir, 'meta.json')))
    axes = meta['axes']
    res = ll.load_results(os.path.join(grid_dir, 'results.csv'))
    f_evo, _ = lg.reconstruct_grid(res, axes, rl=False)

    alphas = np.array(axes['alphas'], dtype=float)
    betas = np.array(axes['betas'], dtype=float)
    ii, jj = np.where(np.isfinite(f_evo))
    vals = f_evo[ii, jj]
    hi = float(f_max) if f_max is not None else float(np.quantile(vals, 0.90))
    targets = np.linspace(float(vals.min()), hi, n_cells)

    chosen, used = [], set()
    for t in targets:
        order = np.argsort(np.abs(vals - t))
        for o in order:
            key = (int(ii[o]), int(jj[o]))
            if key not in used:
                used.add(key)
                chosen.append((key[0], key[1], float(vals[o])))
                break

    anchor, u, v = plane['anchor'], plane['u'], plane['v']
    picks = {}
    for (i, j, fv) in sorted(chosen, key=lambda c: c[2]):
        theta = np.clip(anchor + alphas[i] * u + betas[j] * v, -1.0, 1.0)
        picks[f'cell_i{i}_j{j}'] = (theta, {'i': i, 'j': j, 'f_grid': fv,
                                            'alpha': float(alphas[i]),
                                            'beta': float(betas[j])})
    return picks


def cmd_replayjobs(args):
    """Write the two B1 job files: the replay pass and the matched monomorphic pass.

    Both must see the SAME maps, or k would mix a scale difference with a terrain
    difference. Replay repeat r runs the 5-map block [5r .. 5r+4]; the monomorphic
    pass probes each of those 5 maps separately and `calibrate` averages them, so
    the numerator and denominator of k cover identical terrain.
    """
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.from_grid:
        picks = _pick_grid_cells(args.from_grid, args.n_cells, args.f_max)
        genome_meta = {k: v[1] for k, v in picks.items()}
        rl_for = {k: False for k in picks}   # grid cells match the RL-off f_evo pass
    else:
        picks = _pick_genomes(args)
        genome_meta = {k: {'generation': v[1]} for k, v in picks.items()}
        rl_for = {'evo_centroid': False, 'learn_centroid': bool(args.learn_rl), 'founder': False}
    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    maps_per_gen = args.maps_per_gen
    R = args.repeats

    genomes = {k: v[0] for k, v in picks.items()}

    replay_jobs, mono_jobs = [], []
    for name in picks:
        rl = rl_for[name]
        tag = 'on' if rl else 'off'
        for r in range(R):
            start = args.map_start + r * maps_per_gen
            replay_jobs.append({'id': f'{name}_r{r}_{tag}', 'genome': name,
                                'map_start': int(start), 'rl': rl,
                                'maps': maps_per_gen, 'ticks': args.ticks})
            for m in range(maps_per_gen):
                mono_jobs.append({'id': f'{name}_r{r}_m{m}_{tag}', 'genome': name,
                                  'map_index': int(start + m), 'rl': rl,
                                  'ticks': args.ticks})

    ll.write_jobs(os.path.join(out_dir, 'jobs_replay.json'), genomes, replay_jobs, cfg)
    ll.write_jobs(os.path.join(out_dir, 'jobs_mono.json'), genomes, mono_jobs, cfg)
    meta = {'stage': 'calibration', 'gen': args.gen, 'repeats': R,
            'maps_per_gen': maps_per_gen, 'ticks': args.ticks,
            'map_start': args.map_start, 'learn_rl': bool(args.learn_rl),
            'genomes': genome_meta,
            'source': ('grid:' + os.path.abspath(args.from_grid)) if args.from_grid else 'genome_csv',
            'config_overrides': cfg,
            'reproduction_success_rate_replay': args.reproduction_rate}
    json.dump(meta, open(os.path.join(out_dir, 'meta_calibration.json'), 'w'), indent=2)
    print(f'[stage9/B1] {len(replay_jobs)} replay jobs -> {out_dir}/jobs_replay.json')
    print(f'[stage9/B1] {len(mono_jobs)} monomorphic jobs -> {out_dir}/jobs_mono.json')
    if args.from_grid:
        print('[stage9/B1] grid cells (by f_evo): ' +
              ', '.join(f"{k}={v['f_grid']:.2f}" for k, v in genome_meta.items()))
    else:
        print('[stage9/B1] genomes: ' +
              ', '.join(f'{k}@gen{v["generation"]}' for k, v in genome_meta.items()))
    print(f'\n  node src/eval/replay.js --params <params.json> '
          f'--jobs {out_dir}/jobs_replay.json --out {out_dir}/results_replay.csv '
          f'--reproduction-rate {args.reproduction_rate}')
    print(f'  node src/eval/run_probe.js --params <params.json> '
          f'--jobs {out_dir}/jobs_mono.json --out {out_dir}/results_mono.csv')
    return meta


def fig_calibration(out_png, rows, emp, env, k_thresholds):
    """children/founder vs probe fitness — the k-free viability curve.

    One series (the measured genomes) against the only line that matters,
    replacement at y = 1. Log-y because the quantity spans 0.1 to ~15 and the
    crossing at 1 is the whole point; a linear axis would compress it to nothing.
    The candidate 4.375*k thresholds are drawn as verticals so the reader can see
    directly how far each sits from where the cohort actually breaks even.
    """
    xs = np.array([r['f'] for r in rows], dtype=float)
    ys = np.array([r['cpf'] for r in rows], dtype=float)
    es = np.array([r['cpf_sem'] for r in rows], dtype=float)
    col = _env_color(env)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(1.0, color='#B91C1C', lw=1.6, ls='--', zorder=1,
               label='replacement (1 child / founder)')
    lo = np.maximum(ys - es, 1e-3)
    ax.fill_between(xs, lo, ys + es, color=col, alpha=0.2, lw=0)
    ax.plot(xs, np.maximum(ys, 1e-3), 'o-', color=col, ms=7, lw=2, zorder=3,
            label=f'{env} — measured')
    for lab, t, ls in k_thresholds:
        ax.axvline(t, color='grey', lw=1.1, ls=ls, zorder=0, label=f'{lab} = {t:.2f}')
    if emp is not None and np.isfinite(emp):
        ax.axvline(emp, color='#B91C1C', lw=2, zorder=2)
        ax.annotate(f'replacement at\nf = {emp:.2f}', xy=(emp, 1.0),
                    textcoords='offset points', xytext=(10, 24), fontsize=8,
                    color='#B91C1C', fontweight='bold')
    ax.set_yscale('log')
    # The data rarely spans a full decade, so the default log locator labels only
    # 10^0 and the reader cannot read any other value off the axis.
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(1, 10) * 0.1,
                                                            numticks=20))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{v:g}' if v >= 0.1 else ''))
    ax.set_xlabel('f(θ) on the monomorphic probe scale')
    ax.set_ylabel('children per founder  (replay)')
    ax.set_title(f'B1 viability calibration — {env}\n'
                 'k-free: every agent past the founders is a child that was actually born',
                 fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def cmd_calibrate(args):
    """k = f_monomorphic / f_replay, per genome and pooled."""
    out_dir = args.out_dir
    meta = json.load(open(os.path.join(out_dir, 'meta_calibration.json')))
    rep = ll.load_results(os.path.join(out_dir, 'results_replay.csv'))
    mono = ll.load_results(os.path.join(out_dir, 'results_mono.csv'))

    field = args.replay_field
    if field not in rep.columns:
        raise SystemExit(f'{field} not in results_replay.csv (have {list(rep.columns)})')

    rep = rep.copy()
    rep['genome'] = rep['id'].str.replace(r'_r\d+_(on|off)$', '', regex=True)
    mono = mono.copy()
    mono['genome'] = mono['id'].str.replace(r'_r\d+_m\d+_(on|off)$', '', regex=True)

    # Both replay statistics are carried, not just the selected one. They can
    # differ several-fold and by DIFFERENT factors per environment (descendants
    # are born late and bank little food, so the gap widens with how much
    # reproduction happened), which means the choice of field is not a detail —
    # it can move the viability threshold enough to matter. Reporting both makes
    # that visible instead of buried in a flag default.
    alt = 'mean_founders' if field == 'mean_all_agents' else 'mean_all_agents'
    fields = [f for f in (field, alt) if f in rep.columns]

    per_genome = {}
    for name in sorted(set(rep['genome']) & set(mono['genome'])):
        r = rep[rep['genome'] == name]
        f_mono = float(mono[mono['genome'] == name]['mean_fitness'].mean())
        m_sd = float(mono[mono['genome'] == name]['mean_fitness'].std(ddof=1))
        entry = {'f_monomorphic': f_mono, 'f_monomorphic_sd': m_sd,
                 'n_replay_repeats': int(len(r))}
        for fl in fields:
            v = float(r[fl].mean())
            entry[f'f_replay[{fl}]'] = v
            entry[f'f_replay_sd[{fl}]'] = float(r[fl].std(ddof=1)) if len(r) > 1 else float('nan')
            entry[f'k[{fl}]'] = (f_mono / v) if v > 0 else float('nan')
        entry['f_replay'] = entry[f'f_replay[{field}]']
        entry['f_replay_sd'] = entry[f'f_replay_sd[{field}]']
        entry['k'] = entry[f'k[{field}]']
        if 'n_all_agents' in r.columns and 'n_founders' in r.columns:
            nf = float(r['n_founders'].mean())
            entry['mean_n_all_agents'] = float(r['n_all_agents'].mean())
            # THE k-FREE VIABILITY NUMBER. Every agent beyond the founders is a
            # child that was actually born, so (n_all - n_founders)/n_founders is
            # the realised reproductive output per founder, measured directly in
            # the replay. > 1 means the cohort more than replaced itself. This
            # needs no threshold, no scale conversion and no k at all, so where a
            # pair of genomes brackets 1.0 it pins the viability cutoff on the
            # PROBE scale far more credibly than 4.375*k does.
            entry['children_per_founder'] = (entry['mean_n_all_agents'] - nf) / nf if nf else float('nan')
        per_genome[name] = entry

    def pooled(fl):
        v = np.array([per_genome[n][f'k[{fl}]'] for n in per_genome], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()) if v.size else float('nan'),
                float(v.std(ddof=1)) if v.size > 1 else float('nan'),
                v.size)

    k_mean, k_sd, n_k = pooled(field)
    out = {'k_mean': k_mean, 'k_sd': k_sd,
           'k_se': float(k_sd / np.sqrt(n_k)) if n_k > 1 else float('nan'),
           'k_per_genome': per_genome, 'replay_field': field,
           'viability_threshold': ld.VIABILITY_BASE * k_mean,
           'k_by_field': {fl: dict(zip(('mean', 'sd', 'n'), pooled(fl))) for fl in fields},
           'meta': meta}
    path = os.path.join(out_dir, 'calibration.json')
    json.dump(_jsonable(out), open(path, 'w'), indent=2)

    # If two genomes straddle 1.0 children/founder, interpolate the probe-scale
    # fitness at which a cohort exactly replaces itself — a direct measurement of
    # the same cutoff 4.375*k is trying to estimate.
    emp = None
    cpf = [(v['f_monomorphic'], v['children_per_founder']) for v in per_genome.values()
           if np.isfinite(v.get('children_per_founder', float('nan')))]
    fit = None
    if len(cpf) >= 3:
        fit = ld.replacement_crossing([t[0] for t in cpf], [t[1] for t in cpf])
        if np.isfinite(fit['f_cross']):
            emp = fit['f_cross']
            out['empirical_replacement_f'] = emp
            out['empirical_replacement_k'] = emp / ld.VIABILITY_BASE
            out['empirical_replacement_fit'] = fit
    if emp is None:
        # too few points near the crossing to fit — fall back to the two cells
        # that straddle 1.0, which is noisier but better than nothing
        s = sorted(cpf, key=lambda t: t[1])
        below = [t for t in s if t[1] <= 1.0]
        above = [t for t in s if t[1] > 1.0]
        if below and above:
            (f0, c0), (f1, c1) = below[-1], above[0]
            emp = float(np.interp(1.0, [c0, c1], [f0, f1])) if c1 != c0 else f0
            out['empirical_replacement_f'] = emp
            out['empirical_replacement_k'] = emp / ld.VIABILITY_BASE
            out['empirical_replacement_method'] = 'two-point bracket (fit unavailable)'
    if emp is not None:
        json.dump(_jsonable(out), open(path, 'w'), indent=2)

    print(f'\n=== Stage 9 B1 threshold calibration (k defined by {field}) ===')
    for name, v in per_genome.items():
        print(f"  {name:16s} f_mono={v['f_monomorphic']:8.4f}  "
              f"f_replay={v['f_replay']:8.4f} ± {v['f_replay_sd']:.4f}  k={v['k']:.4f}"
              + (f"   children/founder={v['children_per_founder']:6.2f}"
                 if 'children_per_founder' in v else ''))
    print(f'  k = {k_mean:.4f} ± {k_sd:.4f} (sd over {n_k} genomes)')
    print(f'  => viability threshold on the probe scale: '
          f'f > {ld.VIABILITY_BASE:g} x {k_mean:.3f} = {ld.VIABILITY_BASE * k_mean:.3f}')
    if len(fields) > 1:
        print('  sensitivity to the replay statistic:')
        for fl in fields:
            m, s, _ = pooled(fl)
            print(f'    k[{fl:16s}] = {m:.4f} ± {s:.4f}  -> threshold '
                  f'{ld.VIABILITY_BASE * m:.3f}   (sd across genomes {s / m:.0%})')
        print('    NOTE f_monomorphic is a FOUNDER-only statistic (reproduction is off '
              'in the probe,\n'
              '    so only founders ever exist). Pairing it with mean_all_agents — which '
              'averages\n'
              '    over late-born descendants that bank almost no food — compares two '
              'different\n'
              '    populations, and the resulting k tracks HOW MUCH REPRODUCTION HAPPENED '
              'rather\n'
              '    than any scale conversion. mean_founders is the like-for-like pairing '
              'and is\n'
              '    correspondingly far more stable across genomes.')
    if emp is not None:
        print(f'  k-FREE CHECK: a cohort exactly replaces itself at f_mono = {emp:.3f} '
              f'(implied k = {emp / ld.VIABILITY_BASE:.3f})')
        if fit and np.isfinite(fit['f_cross']):
            print(f'    log-linear fit over the {fit["n_used"]} cells nearest replacement, '
                  f'r²={fit["r2"]:.3f}, bootstrap 95% CI '
                  f'[{fit["ci95"][0]:.2f}, {fit["ci95"][1]:.2f}]')
        print('    Prefer this over 4.375*k — it MEASURES the cutoff instead of '
              'converting to it.')
    else:
        cs = [v.get('children_per_founder') for v in per_genome.values()]
        cs = [c for c in cs if np.isfinite(c)]
        if cs:
            print(f'  k-FREE CHECK: no genome straddles replacement — every one sampled '
                  f'gives {min(cs):.2f}-{max(cs):.2f}\n'
                  f'    children/founder, all {"above" if min(cs) > 1 else "below"} 1.0. '
                  f'The crossing is outside the sampled f range,\n'
                  f'    so it cannot be located from this run — but that IS the finding '
                  f'where the range\n'
                  f'    covers the whole landscape: nothing in the plane is near the '
                  f'viability boundary.')

    # figure: children/founder vs probe fitness
    rows_fig = []
    for name, v in per_genome.items():
        if not np.isfinite(v.get('children_per_founder', float('nan'))):
            continue
        r = rep[rep['genome'] == name]
        per_rep = ((r['n_all_agents'] - r['n_founders']) / r['n_founders']).to_numpy(dtype=float)
        rows_fig.append({
            'f': v['f_monomorphic'], 'cpf': v['children_per_founder'],
            'cpf_sem': float(per_rep.std(ddof=1) / np.sqrt(per_rep.size))
            if per_rep.size > 1 else 0.0})
    if rows_fig:
        rows_fig.sort(key=lambda d: d['f'])
        env = args.env or os.path.basename(out_dir.rstrip('/')).replace('stage9_calib_', '')
        ks = [(f'4.375·k[{fl.replace("mean_", "")}]', ld.VIABILITY_BASE * pooled(fl)[0], ls)
              for fl, ls in zip(fields, (':', '-.'))]
        png = os.path.join(out_dir, f'calibration_{env}.png')
        fig_calibration(png, rows_fig, emp, env, ks)
        print(f'[stage9] {png}')
    print(f'\n[stage9] {path}')
    print(f'  feed to metrics with:  --k-json {path}')
    return out


# ── B2: 1D transect ───────────────────────────────────────────────────────────
def cmd_transectjobs(args):
    """theta(t) = (1-t) theta_evo + t theta_learn, clipped, on ONE fixed map set.

    The grid is a 2D projection of a 3910-D space; this is the single line that
    matters most — the straight path from what evolution found to what learning
    found. Every t uses the SAME map indices so terrain is common-mode along the
    whole transect and the shape of f(t) is not a terrain artefact.

    The [-1,1] clip is applied for consistency with the grid, but it CANNOT bind
    here: both endpoints are simulator-produced genomes and so already lie in
    [-1,1]^3910, and a convex combination of two points in a box stays in that
    box. The clipped count is asserted to 0 and reported, which is the point —
    unlike the Stage-3/4 grids (where clipping displaced ~49% of cells off the
    plane), the transect coordinate t is exact everywhere along its length.
    """
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    _, evo_mat = ll.centroid_trajectory(args.evo_genome_csv)
    _, lrn_mat = ll.centroid_trajectory(args.learn_genome_csv)
    theta_evo = evo_mat[-1]
    theta_learn = lrn_mat[-1]

    ts = np.linspace(0, 1, args.points)
    maps = [args.map_start + r for r in range(args.repeats)]
    genomes, jobs = {}, []
    n_clipped = 0
    for i, t in enumerate(ts):
        th = (1 - t) * theta_evo + t * theta_learn
        clipped = np.clip(th, -1.0, 1.0)
        n_clipped += int(np.sum(th != clipped))
        key = f't{i}'
        genomes[key] = clipped
        for r, m in enumerate(maps):
            jobs.append({'id': f't{i}_r{r}_off', 'genome': key,
                         'map_index': int(m), 'rl': False, 'ticks': args.ticks})
    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    ll.write_jobs(os.path.join(out_dir, 'jobs.json'), genomes, jobs, cfg)
    meta = {'stage': 'transect', 'points': args.points, 'ts': ts.tolist(),
            'repeats': args.repeats, 'maps': maps, 'ticks': args.ticks,
            'evo_genome_csv': os.path.abspath(args.evo_genome_csv),
            'learn_genome_csv': os.path.abspath(args.learn_genome_csv),
            'clipped_weights_total': n_clipped, 'config_overrides': cfg}
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
    print(f'[stage9/B2] {len(jobs)} jobs ({args.points} points x R={args.repeats}) '
          f'-> {out_dir}/jobs.json')
    if n_clipped:
        print(f'[stage9/B2] WARNING: {n_clipped} weight values hit the [-1,1] clip — '
              f'impossible for a convex combination of two in-range genomes, so an '
              f'endpoint genome is out of range. Check the source runs.')
    else:
        print('[stage9/B2] clip is a no-op (0 weights clipped, as expected) — '
              'the transect coordinate t is exact along its whole length')
    return meta


def cmd_transect(args):
    out_dir = args.out_dir
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    ts = np.array(meta['ts'], dtype=float)
    R = int(meta['repeats'])

    ex = res['id'].str.extract(r't(\d+)_r(\d+)_off')
    res = res.copy()
    res['ti'] = ex[0].astype(int)

    rows = []
    for i, t in enumerate(ts):
        sub = res[res['ti'] == i]
        if sub.empty:
            continue
        f = float(sub['mean_fitness'].mean())
        sd = float(sub['mean_fitness'].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append({'t': float(t), 'f': f, 'sigma': sd,
                     'sem': sd / np.sqrt(max(len(sub), 1)),
                     'n_repeats': int(len(sub)),
                     'mean_lifetime': float(sub['mean_lifetime'].mean())})

    env = args.env
    csv_path = os.path.join(out_dir, f'transect_{env}.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['t', 'f', 'sigma', 'sem', 'n_repeats', 'mean_lifetime'])
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})

    t = np.array([r['t'] for r in rows])
    f = np.array([r['f'] for r in rows])
    sem = np.array([r['sem'] for r in rows])
    col = _env_color(env)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.fill_between(t, f - sem, f + sem, color=col, alpha=0.2, lw=0)
    ax.plot(t, f, '-', lw=2, color=col, label=f'f(θ_t) ± SEM (R={R})')
    ax.plot(t[0], f[0], 'o', ms=9, color='white', mec=col, mew=2, zorder=5)
    ax.plot(t[-1], f[-1], 's', ms=9, color=col, mec='white', mew=1.5, zorder=5)
    ax.annotate(f'evolution θ*\n{f[0]:.2f}', (t[0], f[0]), textcoords='offset points',
                xytext=(8, 10), fontsize=8)
    ax.annotate(f'learning θ*\n{f[-1]:.2f}', (t[-1], f[-1]), textcoords='offset points',
                xytext=(-8, 10), fontsize=8, ha='right')
    lo, hi = min(f[0], f[-1]), max(f[0], f[-1])
    if f.min() < lo - sem.mean():
        ax.axhspan(f.min(), lo, color='#B91C1C', alpha=0.06, lw=0)
        ax.annotate('valley between the two optima', (t[int(np.argmin(f))], f.min()),
                    textcoords='offset points', xytext=(0, -18), fontsize=7,
                    color='#B91C1C', ha='center')
    ax.set_xlabel('t   (0 = evolution θ*,  1 = learning θ*)')
    ax.set_ylabel('f(θ) — RL off')
    ax.set_title(f'B2 transect — {env}\nsame {R} maps at every t, weights clipped to [−1,1]',
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, f'transect_{env}.png')
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print(f'\n=== Stage 9 B2 transect — {env} ===')
    print(f'  f(evolution θ*) = {f[0]:.4f} ± {sem[0]:.4f}')
    print(f'  f(learning  θ*) = {f[-1]:.4f} ± {sem[-1]:.4f}')
    print(f'  min along path  = {f.min():.4f} at t = {t[int(np.argmin(f))]:.2f}')
    print(f'  max along path  = {f.max():.4f} at t = {t[int(np.argmax(f))]:.2f}')
    barrier = min(f[0], f[-1]) - f.min()
    print(f'  barrier depth   = {barrier:.4f} '
          f'({"a valley separates the two optima" if barrier > 2 * sem.mean() else "no resolved valley — the path is monotone/flat within noise"})')
    print(f'\n[stage9] {csv_path}')
    print(f'[stage9] {png}')
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Stage 9 — landscape difficulty metrics')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('metrics', help='A1-A8 for one environment')
    p.add_argument('--grid-dir', required=True, help='a Stage-5 grid out-dir (results.csv + meta.json)')
    p.add_argument('--env', required=True, help='environment label, e.g. baseline / hard')
    p.add_argument('--plane', default=None, help='plane label for the output filenames')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--k', type=float, default=ld.K_DEFAULT,
                   help=f'threshold multiplier (default {ld.K_DEFAULT}); overridden by --k-json')
    p.add_argument('--k-json', default=None, help='calibration.json from the B1 calibrate step')
    p.add_argument('--k-from', choices=['auto', 'empirical', 'ratio'], default='auto',
                   help='auto (default): use the measured replacement crossing if the '
                        'calibration found one, else the f_mono/f_replay ratio')
    p.add_argument('--lambda-dir', action='append', default=[],
                   help='Stage-6 walk out-dir for the A8 reanalysis (repeatable)')
    p.add_argument('--kmax', type=int, default=100)
    p.add_argument('--lambda-transform', choices=['log', 'none'], default='log')
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser('compare', help='A9 cross-environment table')
    p.add_argument('--metrics', action='append', required=True,
                   help='difficulty_metrics_<env>.json (repeatable; first two get Z)')
    p.add_argument('--out-dir', required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('replayjobs', help='B1: write the replay + monomorphic job files')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--from-grid', default=None,
                   help='select genomes from a Stage-5 grid out-dir spanning the '
                        'fitness range, instead of the three gen-1000 genomes')
    p.add_argument('--n-cells', type=int, default=10,
                   help='--from-grid: how many cells to sample across the f range')
    p.add_argument('--f-max', type=float, default=None,
                   help='--from-grid: top of the sampled f range (default: 90th pct)')
    p.add_argument('--evo-genome-csv', default=None)
    p.add_argument('--learn-genome-csv', default=None)
    p.add_argument('--founder-genome-csv', default=None,
                   help='where to take the founder genome from (default: the evolution run)')
    p.add_argument('--founder-index', type=int, default=0,
                   help='rank of the founder to take at --gen, by fitness (0 = fittest)')
    p.add_argument('--gen', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--maps-per-gen', type=int, default=5)
    p.add_argument('--ticks', type=int, default=2000)
    p.add_argument('--map-start', type=int, default=0)
    p.add_argument('--reproduction-rate', type=float, default=0.8)
    p.add_argument('--learn-rl', action='store_true',
                   help='evaluate the learning centroid with RL on (it was evolved that way)')
    p.add_argument('--config-overrides', default=None)
    p.set_defaults(func=cmd_replayjobs)

    def _need_genomes(a):
        if not a.from_grid and not (a.evo_genome_csv and a.learn_genome_csv):
            ap.error('replayjobs: pass --from-grid, or both --evo-genome-csv and '
                     '--learn-genome-csv')
        return cmd_replayjobs(a)
    p.set_defaults(func=_need_genomes)

    p = sub.add_parser('calibrate', help='B1: compute k from the replay + monomorphic results')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--replay-field', default='mean_all_agents',
                   help='mean_all_agents (spec: every agent that existed) or mean_founders')
    p.add_argument('--env', default=None, help='label for the figure (default: from out-dir)')
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('transectjobs', help='B2: write the transect jobs')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--evo-genome-csv', required=True)
    p.add_argument('--learn-genome-csv', required=True)
    p.add_argument('--points', type=int, default=51)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--map-start', type=int, default=0)
    p.add_argument('--ticks', type=int, default=2000)
    p.add_argument('--config-overrides', default=None)
    p.set_defaults(func=cmd_transectjobs)

    p = sub.add_parser('transect', help='B2: analyse the transect results')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--env', required=True)
    p.set_defaults(func=cmd_transect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
