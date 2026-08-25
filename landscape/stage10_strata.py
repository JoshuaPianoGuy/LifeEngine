"""
landscape/stage10_strata.py
======================================================================
Stage 10 — stratified (aggregated) landscapes.

Stage 9 reports ONE surface per environment: the mean over the R map repeats.
That mean hides the largest effect in the data. Every grid cell is evaluated on
the SAME R maps (ll_grid.build_grid_jobs pairs the RL passes on identical map
indices), and on the hard grid the per-map mean of f ranges 2.16 .. 9.72 — a
spread of 113% of the grid mean, far larger than anything the genome axes do.
The map a run lands on, not the genome, decides most of the outcome.

So the honest object is not one landscape but a FAMILY of them, indexed by how
favourable the world is:

    LOW    stratum — the maps evolution does worst on ("underperforms")
    HIGH   stratum — the maps it does best on ("performs well")
    ALL              — every map, i.e. exactly the Stage-9 surface

This costs NO new simulation. The cube is already on disk; the strata are three
different reductions of it. Splitting on the per-map grand mean is safe here
because that mean is estimated from all N^2 cells at once: the split-half rank
correlation of the map ordering is rho = 0.96 (hard) and 1.00 (baseline), so the
ranking is essentially noise-free and the usual winner's-curse objection to
selecting on the same data does not bite.

TWO VIABILITY ESTIMATORS, AND THEY DISAGREE.
    'surface' thresholds the map-averaged fitness — the Stage-9 definition.
    'per_map' thresholds each map separately and then averages, which is the
    quantity actually meant by "how much of this space supports a population",
    since a genome is viable in a given world or it is not.
    Thresholding is nonlinear, so averaging first is not free: on the hard grid
    the learn - evo gap is +0.096 by 'surface' but +0.024 by 'per_map', and on
    baseline 'surface' saturates at 1.000 for both passes while 'per_map' gives
    0.903 / 0.914. Both are emitted; 'per_map' is the one to quote.

Usage
-----
  # A. map strata for one environment, from an existing Stage-5 grid dir (FREE)
  python landscape/stage10_strata.py strata \
      --grid-dir landscape/out/stage3_hard_coarse --env hard \
      --k-json landscape/out/stage9_calib_hard_grid/calibration.json \
      --out-dir landscape/out/stage10_hard

  # B. cross-environment / cross-stratum table
  python landscape/stage10_strata.py compare \
      --metrics landscape/out/stage10_hard/strata_hard.json \
      --metrics landscape/out/stage10_baseline/strata_baseline.json \
      --out-dir landscape/out/stage10

MULTI-TRAJECTORY LANDSCAPES (one plane per environment, every run on it)
-----------------------------------------------------------------------
  # C. FREE go/no-go + trajectory preview — ALWAYS do this before spending a grid
  python landscape/stage10_strata.py planecheck \
      --logs-root logs_landscapes --env roam60d5 --fig /tmp/pc_hard.png

  # D. write the grid jobs for the chosen plane (still no simulation)
  python landscape/stage10_strata.py planejobs \
      --logs-root logs_landscapes --env roam60d5 --plane-method within \
      --resolution 21 --repeats 5 --out-dir landscape/out/stage10_hard_traj \
      --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'

  # E. evaluate on the cluster: sbatch run_landscape_stage10_array.slurm
  #    then merge and draw the finished figure
  python landscape/merge_shards.py --out-dir landscape/out/stage10_hard_traj
  python landscape/stage10_strata.py trajplot \
      --grid-dir landscape/out/stage10_hard_traj --env hard \
      --k-json landscape/out/stage9_calib_hard_grid/calibration.json
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import ll_common as ll
import ll_difficulty as ld
import ll_grid as lg


STRATA_ORDER = ['low', 'high', 'all']
STRATUM_LABEL = {'low': 'underperforming maps', 'high': 'favourable maps',
                 'all': 'all maps (Stage-9 mean)'}
STRATUM_COLOR = {'low': '#c2452d', 'high': '#2f7d4f', 'all': '#3a5f9f'}


def _jsonable(o):
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


# ── cube loading and stratification ───────────────────────────────────────────
def load_cube(grid_dir):
    """Per-repeat cubes for both RL passes, shaped (N, N, R), plus the axes."""
    res_path = os.path.join(grid_dir, 'results.csv')
    if not os.path.exists(res_path):
        raise SystemExit(f'{res_path} not found — run merge_shards.py first')
    res = ll.load_results(res_path)
    meta = json.load(open(os.path.join(grid_dir, 'meta.json')))
    axes = meta['axes']
    alphas = np.asarray(axes['alphas'], dtype=float)
    betas = np.asarray(axes['betas'], dtype=float)
    A, B = ld.grid_coords(alphas, betas)
    return {
        'meta': meta, 'axes': axes, 'R': int(axes['R']), 'N': int(axes['resolution']),
        'maps': list(axes.get('maps', range(int(axes['R'])))),
        'cube_evo': lg.reconstruct_cube(res, axes, rl=False),
        'cube_learn': lg.reconstruct_cube(res, axes, rl=True),
        'alphas': alphas, 'betas': betas, 'A': A, 'B': B,
        'extent': [float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1])],
        'grid_dir': grid_dir,
    }


def split_maps(cube_evo, frac=0.5):
    """Rank the R maps by their grand mean f_evo; return the low/high index sets.

    Ranked on the EVOLUTION pass only, so the two passes are never split on
    different criteria — the learning surface must be read on the same maps for
    the learn - evo comparison inside a stratum to stay paired.
    """
    per_map = np.nanmean(cube_evo, axis=(0, 1))
    R = per_map.size
    n = max(1, int(round(frac * R)))
    order = np.argsort(per_map)
    return {'low': order[:n], 'high': order[R - n:], 'all': np.arange(R),
            'per_map_mean': per_map}


def split_half_check(cube_evo):
    """Rank correlation of the map ordering between the two halves of the grid.

    The evidence that selecting strata on the data is safe. Near 1 = the map
    ranking is a property of the maps, not a noise draw off these cells.
    """
    N = cube_evo.shape[0]
    a = np.nanmean(cube_evo[:N // 2], axis=(0, 1))
    b = np.nanmean(cube_evo[N // 2:], axis=(0, 1))
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return ld.pearson(ra, rb), ld.pearson(a, b)


def reduce_stratum(cube, idx):
    """Mean / sd / sem over the selected repeats only."""
    sub = cube[:, :, idx]
    f = np.nanmean(sub, axis=2)
    sd = np.nanstd(sub, axis=2, ddof=1) if len(idx) > 1 else np.zeros_like(f)
    return f, sd, sd / np.sqrt(max(len(idx), 1))


# ── metrics for one stratum ───────────────────────────────────────────────────
def stratum_metrics(C, idx, k, pass_name):
    cube = C['cube_evo'] if pass_name == 'evo' else C['cube_learn']
    f, sd, sem = reduce_stratum(cube, idx)
    A, B = C['A'], C['B']
    thr = ld.viability_threshold(k)

    r_best, n_cells, ref_best = ld.fdc(A, B, f, reference='best')
    r_top5, _, ref_top5 = ld.fdc(A, B, f, reference='top5')
    v_surface, n_v = ld.viable_fraction(f, k)
    sub = cube[:, :, idx]
    v_per_map = float(np.nanmean(sub > thr))
    per_map_v = [float(np.nanmean(sub[:, :, m] > thr)) for m in range(sub.shape[2])]
    gnr = ld.gnr_summary(f, sd)
    n_peaks, _ = ld.count_peaks_noise_aware(f, sem, margin=2.0)
    n_raw, _ = ld.count_peaks_noise_aware(f, None)

    return {
        'pass': pass_name, 'n_maps': int(len(idx)), 'maps': [int(m) for m in idx],
        'mean_f': float(np.nanmean(f)), 'max_f': float(np.nanmax(f)),
        'viability_threshold': thr,
        'viable_frac_surface': v_surface,
        'viable_frac_surface_se': ld.viable_fraction_se(v_surface, n_v),
        'viable_frac_per_map': v_per_map,
        'viable_frac_per_map_spread': per_map_v,
        'fdc_best': r_best, 'fdc_best_se': ld.corr_se(r_best, n_cells),
        'fdc_top5_centroid': r_top5, 'fdc_top5_centroid_se': ld.corr_se(r_top5, n_cells),
        'fdc_ref_best': list(ref_best), 'fdc_ref_top5': list(ref_top5),
        'n_cells': int(n_cells),
        'gnr_median_all': gnr['gnr_median_all'],
        'gnr_frac_below_1': gnr['gnr_frac_below_1'],
        'dispersion': ld.dispersion_profile(A, B, f),
        'neutrality': ld.neutrality_summary(f, sem),
        'peaks_raw': int(n_raw), 'peaks_filtered': int(n_peaks),
        'r2_quadratic': ld.meta_model_r2(A, B, f)['r2_quadratic'],
        '_f': f, '_sd': sd, '_sem': sem,
    }


# ── figures ───────────────────────────────────────────────────────────────────
def _imshow(ax, grid, extent, cmap, title, vmin=None, vmax=None):
    im = ax.imshow(np.asarray(grid).T, origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('α (PC1)', fontsize=8)
    ax.set_ylabel('β (PC2)', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def fig_strata(out_png, C, M, env, k, pass_name):
    """The three surfaces on a SHARED colour scale, each with its viability contour.

    Shared scale is the whole point: on independent scales the three panels look
    alike and the reader concludes the strata differ only in detail, when in fact
    the low stratum sits almost entirely below the replacement threshold.
    """
    thr = ld.viability_threshold(k)
    fs = [M[s][pass_name]['_f'] for s in STRATA_ORDER]
    lo = float(np.nanmin(fs))
    hi = float(np.nanmax(fs))
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, s, f in zip(axs, STRATA_ORDER, fs):
        m = M[s][pass_name]
        im = _imshow(ax, f, C['extent'], 'viridis',
                     f'{s.upper()} — {STRATUM_LABEL[s]}\n'
                     f'{m["n_maps"]} maps, RL {"on" if pass_name == "learn" else "off"}',
                     lo, hi)
        if np.nanmin(f) < thr < np.nanmax(f):
            cs = ax.contour(C['A'], C['B'], f, levels=[thr], colors='white', linewidths=1.8)
            ax.clabel(cs, fmt=lambda v: f'viable ({v:.2f})', fontsize=7)
        ax.text(0.02, 0.98,
                f'viable {m["viable_frac_surface"]:.1%} (surface)\n'
                f'       {m["viable_frac_per_map"]:.1%} (per-map)\n'
                f'FDC {m["fdc_best"]:+.3f}\n'
                f'neutral {m["neutrality"]["neutral_frac"]["0.02"]:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=7, color='white',
                bbox=dict(fc='black', alpha=0.5, ec='none', pad=2))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Stage 10 — stratified landscapes, {env} (shared colour scale, '
                 f'threshold {thr:.2f})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_per_map(out_png, C, sp, env, k):
    """Per-map grand mean and per-map viable fraction — the effect being split on."""
    thr = ld.viability_threshold(k)
    per_map = sp['per_map_mean']
    R = per_map.size
    order = np.argsort(per_map)
    cols = ['#c2452d' if i in set(sp['low'].tolist()) else '#2f7d4f' for i in order]
    vf = [float(np.nanmean(C['cube_evo'][:, :, i] > thr)) for i in order]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].bar(range(R), per_map[order], color=cols)
    axs[0].axhline(thr, color='k', ls=':', lw=1.2)
    axs[0].annotate(f'replacement {thr:.2f}', xy=(0, thr), fontsize=7, va='bottom')
    axs[0].set_xticks(range(R))
    axs[0].set_xticklabels([str(int(C['maps'][i])) for i in order], fontsize=7)
    axs[0].set_xlabel('map index (sorted)', fontsize=8)
    axs[0].set_ylabel('grid-mean f (RL off)', fontsize=8)
    axs[0].set_title(f'per-map mean fitness — {env}\n'
                     f'spread = {np.ptp(per_map):.2f} '
                     f'({100 * np.ptp(per_map) / per_map.mean():.0f}% of the mean)',
                     fontsize=9)

    axs[1].bar(range(R), vf, color=cols)
    axs[1].set_xticks(range(R))
    axs[1].set_xticklabels([str(int(C['maps'][i])) for i in order], fontsize=7)
    axs[1].set_xlabel('map index (sorted)', fontsize=8)
    axs[1].set_ylabel('viable fraction of cells', fontsize=8)
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title(f'per-map viable fraction — {env}', fontsize=9)
    for ax in axs:
        ax.grid(alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_neutrality(out_png, C, M, env, pass_name='evo'):
    """Neutrality curve vs eps, and the plateau map at eps = 2% of the range.

    The curve matters because a single eps is a choice: if the three strata only
    separate at one eps the result is an artefact of that choice, and if they
    stay ordered across the sweep the ordering is real.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))
    eps_grid = np.linspace(0.002, 0.10, 40)
    for s in STRATA_ORDER:
        f = M[s][pass_name]['_f']
        rng = float(np.nanmax(f) - np.nanmin(f))
        ys = [ld.neutral_fraction(f, e * rng)[0] for e in eps_grid]
        axs[0].plot(100 * eps_grid, ys, lw=2, color=STRATUM_COLOR[s],
                    label=f'{s} ({STRATUM_LABEL[s]})')
    axs[0].set_xlabel('ε as % of the stratum fitness range', fontsize=8)
    axs[0].set_ylabel('neutral fraction of adjacent pairs', fontsize=8)
    axs[0].set_title(f'A10 neutrality sweep — {env}', fontsize=9)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=6.5)
    axs[0].tick_params(labelsize=7)

    # Colour each cell by the SIZE of the plateau it belongs to, log-scaled: one
    # component swallows ~77% of the hard grid, so on a linear scale every other
    # plateau collapses to the floor colour and the panel reads as a bare mask.
    for ax, s in zip(axs[1:], STRATA_ORDER):
        f = M[s][pass_name]['_f']
        rng = float(np.nanmax(f) - np.nanmin(f))
        pl = ld.neutral_plateaus(f, 0.02 * rng)
        sizes = np.bincount(pl['labels'].ravel())
        comp = sizes[pl['labels']].astype(float)
        im = ax.imshow(comp.T, origin='lower', extent=C['extent'], cmap='magma',
                       norm=LogNorm(vmin=1, vmax=max(comp.max(), 2)),
                       aspect='auto', interpolation='nearest')
        ax.set_title(f'{s.upper()} plateaus (ε = 2% range)\n'
                     f'{pl["n_plateaus"]} components, largest {pl["largest_frac"]:.0%} of grid',
                     fontsize=9)
        ax.set_xlabel('α (PC1)', fontsize=8)
        ax.set_ylabel('β (PC2)', fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label('cells in this plateau', fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── commands ──────────────────────────────────────────────────────────────────
def _resolve_k(args):
    """Same precedence as Stage 9 --k-from auto: prefer the MEASURED replacement
    crossing, fall back to the f_mono/f_replay ratio, so the two stages threshold
    on identical values and their viable fractions stay comparable."""
    if args.k_json:
        cal = json.load(open(args.k_json))
        for key in ('empirical_replacement_k', 'k_mean'):
            v = cal.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                return float(v), f'{key} from {args.k_json}'
        raise SystemExit(f'no usable k in {args.k_json}')
    return float(args.k), 'command line'


def cmd_strata(args):
    C = load_cube(args.grid_dir)
    k, k_src = _resolve_k(args)
    sp = split_maps(C['cube_evo'], frac=args.frac)
    rho_rank, rho_lin = split_half_check(C['cube_evo'])
    out_dir = args.out_dir or os.path.join('landscape', 'out', f'stage10_{args.env}')
    os.makedirs(out_dir, exist_ok=True)

    M = {}
    for s in STRATA_ORDER:
        M[s] = {p: stratum_metrics(C, sp[s], k, p) for p in ('evo', 'learn')}

    fig_strata(os.path.join(out_dir, f'strata_{args.env}_evo.png'), C, M, args.env, k, 'evo')
    fig_strata(os.path.join(out_dir, f'strata_{args.env}_learn.png'), C, M, args.env, k, 'learn')
    fig_per_map(os.path.join(out_dir, f'per_map_{args.env}.png'), C, sp, args.env, k)
    fig_neutrality(os.path.join(out_dir, f'neutrality_{args.env}.png'), C, M, args.env)

    payload = {
        'env': args.env, 'grid_dir': os.path.abspath(args.grid_dir),
        'resolution': C['N'], 'R': C['R'], 'maps': C['maps'],
        'k': k, 'k_source': k_src, 'viability_threshold': ld.viability_threshold(k),
        'split_frac': args.frac,
        'per_map_mean_f_evo': sp['per_map_mean'],
        'map_rank_split_half_spearman': rho_rank,
        'map_rank_split_half_pearson': rho_lin,
        'strata': {s: {p: {kk: vv for kk, vv in M[s][p].items() if not kk.startswith('_')}
                       for p in ('evo', 'learn')} for s in STRATA_ORDER},
    }
    jpath = os.path.join(out_dir, f'strata_{args.env}.json')
    with open(jpath, 'w') as fh:
        json.dump(_jsonable(payload), fh, indent=2)

    rows = []
    for s in STRATA_ORDER:
        for p in ('evo', 'learn'):
            m = M[s][p]
            rows.append({
                'env': args.env, 'stratum': s, 'pass': p, 'n_maps': m['n_maps'],
                'mean_f': m['mean_f'],
                'viable_surface': m['viable_frac_surface'],
                'viable_per_map': m['viable_frac_per_map'],
                'fdc_best': m['fdc_best'], 'fdc_best_se': m['fdc_best_se'],
                'fdc_top5': m['fdc_top5_centroid'],
                'neutral_1pct': m['neutrality']['neutral_frac']['0.01'],
                'neutral_2pct': m['neutrality']['neutral_frac']['0.02'],
                'neutral_5pct': m['neutrality']['neutral_frac']['0.05'],
                'neutral_noise': m['neutrality'].get('neutral_frac_noise'),
                'plateau_largest_frac': m['neutrality']['plateaus']['largest_frac'],
                'n_plateaus': m['neutrality']['plateaus']['n_plateaus'],
                'gnr_median': m['gnr_median_all'],
                'dispersion_p05': m['dispersion'].get('0.05', m['dispersion'].get(0.05)),
                'peaks_filtered': m['peaks_filtered'],
            })
    cpath = os.path.join(out_dir, f'strata_{args.env}.csv')
    with open(cpath, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _print_strata(payload, rows)
    print(f'\n[stage10] wrote {jpath}\n[stage10] wrote {cpath}\n[stage10] figures in {out_dir}')


def _print_strata(o, rows):
    print(f'\n=== Stage 10 strata — {o["env"]} '
          f'({o["resolution"]}x{o["resolution"]}, R={o["R"]}, k={o["k"]:.4f}, '
          f'threshold={o["viability_threshold"]:.3f}) ===')
    pm = np.asarray(o['per_map_mean_f_evo'], dtype=float)
    print(f'  per-map grand mean f_evo: {np.round(pm, 2).tolist()}')
    print(f'  spread {np.ptp(pm):.2f} = {100 * np.ptp(pm) / pm.mean():.0f}% of the mean '
          f'-> map identity dominates genome position')
    print(f'  map-ranking split-half stability: spearman {o["map_rank_split_half_spearman"]:.3f}, '
          f'pearson {o["map_rank_split_half_pearson"]:.3f} (near 1 = split is not a noise draw)')
    hdr = (f'  {"stratum":<8}{"pass":<7}{"mean_f":>8}{"viab_srf":>9}{"viab_map":>9}'
           f'{"FDC":>8}{"neut2%":>8}{"plateau":>9}{"GNR":>7}{"peaks":>7}')
    print(hdr)
    for r in rows:
        print(f'  {r["stratum"]:<8}{r["pass"]:<7}{r["mean_f"]:8.3f}'
              f'{r["viable_surface"]:9.3f}{r["viable_per_map"]:9.3f}'
              f'{r["fdc_best"]:+8.3f}{r["neutral_2pct"]:8.3f}'
              f'{r["plateau_largest_frac"]:9.3f}{r["gnr_median"]:7.3f}{r["peaks_filtered"]:7d}')
    for s in STRATA_ORDER:
        e = next(r for r in rows if r['stratum'] == s and r['pass'] == 'evo')
        l = next(r for r in rows if r['stratum'] == s and r['pass'] == 'learn')
        print(f'  learning lift in viability, {s:<5}: surface {l["viable_surface"] - e["viable_surface"]:+.3f}'
              f'   per-map {l["viable_per_map"] - e["viable_per_map"]:+.3f}')


def cmd_compare(args):
    """One table across environments AND strata — the Stage-9 A9 view, widened."""
    metrics = [json.load(open(p)) for p in args.metrics]
    os.makedirs(args.out_dir, exist_ok=True)
    fields = ['viable_frac_per_map', 'viable_frac_surface', 'fdc_best',
              'fdc_top5_centroid', 'gnr_median_all', 'peaks_filtered']
    rows = []
    for m in metrics:
        for s in STRATA_ORDER:
            for p in ('evo', 'learn'):
                d = m['strata'][s][p]
                row = {'env': m['env'], 'stratum': s, 'pass': p, 'mean_f': d['mean_f']}
                row.update({f: d.get(f) for f in fields})
                row['neutral_2pct'] = d['neutrality']['neutral_frac']['0.02']
                row['plateau_largest_frac'] = d['neutrality']['plateaus']['largest_frac']
                rows.append(row)
    path = os.path.join(args.out_dir, 'strata_comparison.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[stage10] wrote {path} ({len(rows)} rows)')
    for r in rows:
        print(f'  {r["env"]:<9}{r["stratum"]:<6}{r["pass"]:<7}'
              f'f={r["mean_f"]:6.2f}  viable(per-map)={r["viable_frac_per_map"]:.3f}  '
              f'FDC={r["fdc_best"]:+.3f}  neutral2%={r["neutral_2pct"]:.3f}')


# ── run-level stratification (needs a new grid evaluation) ────────────────────
def run_outcome(run_dir, tail_from=900):
    """Final-phase mean fitness and population of one run, from generations.csv."""
    import pandas as pd
    df = pd.read_csv(os.path.join(run_dir, 'generations.csv'))
    tail = df[df['generation'] > tail_from]
    if len(tail) == 0:
        tail = df.tail(1)
    return {'run': os.path.basename(run_dir.rstrip('/')),
            'dir': os.path.abspath(run_dir),
            'final_fitness': float(tail['avg_fitness'].mean()),
            'final_pop': float(tail['total_agents'].mean())}


# ── plane construction and its go/no-go diagnostic ────────────────────────────
def _dot(a, b):
    """matmul with the macOS/Accelerate spurious FP flags silenced.

    Same quirk ll_common.project_onto_plane documents: a large float64 matmul
    raises divide/overflow/invalid on this platform even though every input is
    finite and every output is correct to ~1e-14.
    """
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return a @ b


def build_plane_basis(stacks, method='joint', groups=None, anchor=None):
    """Return (anchor, U) where U is a (3910, 2) orthonormal basis.

    'joint'      — PCA about one global mean. What Stage 3 uses. Maximises TOTAL
                   variance, which with many runs is spent separating run
                   POSITIONS, leaving almost none for their motion.
    'within'     — PCA on every run's motion about its OWN mean. Maximises
                   visible within-run motion by construction, which is what a
                   trajectory figure is actually for.
    'balanced'   — 'within', but each run is also scaled to unit Frobenius norm
                   before the SVD, so every run contributes equally to the fit.
                   INTENDED for the case where long runs monopolise the plane,
                   but on this data it OVER-CORRECTS and is usually the worst
                   choice: the two collapsed hard runs are SHORT, so normalising
                   amplifies them until they hijack both components and
                   everything else clusters at the origin. Kept because the
                   failure is instructive and the fix is right for data whose
                   run lengths differ without a short-and-degenerate group.

    THERE IS NO UNIVERSAL BEST METHOD — it is per-figure, and the numbers alone
    will mislead you. Measured here: BASELINE (3 evo + 3 learn) is best on
    'joint', which scores a poor 0.058 within-run capture yet draws a clean fan
    of six paths radiating from the shared gen-1 origin with the conditions
    cleanly separated. HARD (13 evo + 3 learn) is best on 'within': its 'joint'
    plane scores 0.010 and spends both components on the 3 long learning runs,
    squashing all 13 evolution paths into a blob. Always render the preview
    (`planecheck --fig`) and look, rather than picking on the score.
    'supervised' — u = mean progress direction (where runs go), v = the outcome
                   contrast (what separates success from collapse),
                   Gram-Schmidt orthogonalised. Not variance-optimal, but both
                   axes have a stated meaning, which is far easier to defend than
                   a principal component that explains 6%.
    """
    X = np.vstack(stacks)
    if anchor is None:
        anchor = X.mean(axis=0)
    if method == 'joint':
        P = ll.joint_pca_plane(stacks, anchor=anchor)
        return anchor, np.column_stack([P['u'], P['v']]), \
            [float(x) for x in P['explained_variance_ratio']]
    if method in ('within', 'balanced'):
        blocks = []
        for C in stacks:
            Cc = C - C.mean(axis=0)
            if method == 'balanced':
                n = np.linalg.norm(Cc)
                if n > 0:
                    Cc = Cc / n
            blocks.append(Cc)
        _, S, Vt = np.linalg.svd(np.vstack(blocks), full_matrices=False)
        evr = (S ** 2 / (S ** 2).sum())[:2]
        return anchor, np.column_stack([Vt[0], Vt[1]]), [float(x) for x in evr]
    if method == 'supervised':
        prog = np.zeros(X.shape[1])
        for C in stacks:
            d = C[-1] - C[0]
            n = np.linalg.norm(d)
            if n > 0:
                prog += d / n
        ends = np.array([C[-1] for C in stacks])
        if groups is not None and len(set(groups)) > 1:
            g = np.asarray(groups)
            second = ends[g == 'succeed'].mean(axis=0) - ends[g != 'succeed'].mean(axis=0)
        else:  # no outcome split: use the dominant endpoint-spread direction
            _, _, Vte = np.linalg.svd(ends - ends.mean(axis=0), full_matrices=False)
            second = Vte[0]
        u = prog / np.linalg.norm(prog)
        v = second - _dot(second, u) * u
        return anchor, np.column_stack([u, v / np.linalg.norm(v)]), [float('nan')] * 2
    raise ValueError(f'unknown plane method {method!r}')


def plane_diagnostics(stacks, U, anchor):
    """Is this plane worth evaluating a grid on? EVR alone cannot say.

    EVR mixes two things that matter differently for a trajectory figure:
    whether runs land in DISTINGUISHABLE places, and whether each run's own
    MOTION is visible. A plane can score a respectable EVR purely by separating
    run positions while rendering every trajectory as a stationary dot — and,
    worse, moving the anchor off-centre INFLATES EVR by counting the constant
    offset as explained variance. So measure the two directly:

      within — median share of a run's own motion (about its own mean) in-plane.
               This is the one that decides whether paths look like journeys.
      disp   — median share of the net gen-1 -> gen-N displacement in-plane.
      sep    — mean pairwise endpoint distance in-plane / in full 3910-D.
      ceiling— median share the BEST possible 2D plane fitted to a single run
               alone could capture. Nothing can beat this, and it is only ~0.66
               here, so every landscape figure in this project loses at least a
               third of the trajectory. Report `within` against this, not against 1.
    """
    within, disp, pp, ff, ceil = [], [], [], [], []
    for C in stacks:
        Cc = C - C.mean(axis=0)
        tot = float((Cc ** 2).sum())
        within.append(float((_dot(Cc, U) ** 2).sum()) / tot if tot > 0 else np.nan)
        d = C[-1] - C[0]
        dn = float(_dot(d, d))
        disp.append(float((_dot(d, U) ** 2).sum()) / dn if dn > 0 else np.nan)
        pp.append(_dot(C[-1] - anchor, U))
        ff.append(C[-1])
        s = np.linalg.svd(Cc, compute_uv=False)
        ceil.append(float((s[:2] ** 2).sum() / (s ** 2).sum()))
    pp, ff = np.array(pp), np.array(ff)
    dp, df = [], []
    for i in range(len(pp)):
        for j in range(i + 1, len(pp)):
            dp.append(np.linalg.norm(pp[i] - pp[j]))
            df.append(np.linalg.norm(ff[i] - ff[j]))
    return {'within_median': float(np.median(within)),
            'within_min': float(np.min(within)), 'within_max': float(np.max(within)),
            'disp_median': float(np.median(disp)),
            'sep_ratio': float(np.mean(dp) / np.mean(df)) if df else float('nan'),
            'ceiling_median': float(np.median(ceil)),
            'within_per_run': [float(x) for x in within]}


CONDITION_COLOR = {'evolution': '#c2452d', 'learning': '#2f7d4f', 'pure_rl': '#7a5bb5'}


def discover_runs(logs_root, env, conditions=('evolution', 'learning'), family='standard'):
    """Find every run dir for one environment across the given conditions.

    Layout, exactly as logs_landscapes already is — NO restructuring needed:

        <logs_root>/<condition>/<family>/auto-run/<env>_*/genome.csv

    Note the two levels are condition x family, not one list of conditions:
        evolution/standard   learning/standard   learning/pure_rl
    so pure_rl is a FAMILY under learning, not a third top-level condition.
    Reach it with --conditions learning --family pure_rl. It will still yield
    nothing usable — pure_rl runs log no genome.csv, so they cannot go on a PCA
    plane at all until that logging is enabled and they are re-run.

    The environment lives in the run-dir PREFIX ('baseline_...', 'roam60d5_...'),
    so one glob per condition finds everything. Returns [(dir, condition)].
    """
    found = []
    for cond in conditions:
        base = os.path.join(logs_root, cond, family)
        if not os.path.isdir(base):
            print(f'  ! no such condition/family: {base} — skipped')
            continue
        hits = sorted(glob.glob(os.path.join(base, 'auto-run', f'{env}_*')))
        if not hits:
            print(f'  ! {base}/auto-run has no runs matching {env}_* — skipped')
        for d in hits:
            if os.path.exists(os.path.join(d, 'genome.csv')):
                found.append((d, cond))
            else:
                print(f'  ! {os.path.basename(d)}: no genome.csv, skipped '
                      f'({cond}/{family} does not log centroid genomes)')
    return found


def fig_trajectories(out_png, stacks, conds, names, fits, methods, groups,
                     anchor0, title):
    """Draw the trajectory overlay for each candidate plane, side by side.

    The whole point of running this BEFORE a grid: the paths need no simulation,
    so the figure that decides the plane choice is free. Every run is drawn
    individually — the only averaging anywhere is the anchor.
    """
    fig, axs = plt.subplots(1, len(methods), figsize=(5.8 * len(methods), 5.2),
                            squeeze=False)
    axs = axs[0]
    for ax, method in zip(axs, methods):
        anchor, U, evr = build_plane_basis(stacks, method=method, groups=groups,
                                           anchor=anchor0)
        d = plane_diagnostics(stacks, U, anchor)
        for C, cond, fit in zip(stacks, conds, fits):
            P = _dot(C - anchor, U)
            collapsed = fit < 5.0
            ax.plot(P[:, 0], P[:, 1], color=CONDITION_COLOR.get(cond, '#555'),
                    lw=2.2 if collapsed else 1.5, ls='--' if collapsed else '-', alpha=0.85)
            ax.plot(P[0, 0], P[0, 1], 'o', ms=4, mfc='white', mew=1.2,
                    color=CONDITION_COLOR.get(cond, '#555'))
            ax.plot(P[-1, 0], P[-1, 1], '*', ms=11 if collapsed else 8,
                    color=CONDITION_COLOR.get(cond, '#555'))
        ax.set_title(f'{method} plane — EVR {evr[0]:.3f}+{evr[1]:.3f}', fontsize=9)
        ax.set_xlabel('α (PC1)', fontsize=8)
        ax.set_ylabel('β (PC2)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.98, f'within {d["within_median"]:.3f} / ceiling '
                            f'{d["ceiling_median"]:.2f}\nsep {d["sep_ratio"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=7,
                bbox=dict(fc='white', alpha=0.8, ec='0.7', pad=2))
    h = [plt.Line2D([], [], color=c, lw=2, label=k)
         for k, c in CONDITION_COLOR.items() if k in set(conds)]
    h += [plt.Line2D([], [], color='0.3', lw=2, ls='--', label='collapsed run'),
          plt.Line2D([], [], color='0.3', marker='o', mfc='white', ls='', label='first gen'),
          plt.Line2D([], [], color='0.3', marker='*', ls='', label='last gen')]
    axs[-1].legend(handles=h, fontsize=6.5, loc='best')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  -> {out_png}')


def _load_run_stacks(run_dirs, stride=5):
    """(runs, stacks, names, groups) with generations subsampled for the SVD."""
    out_runs, stacks, names = [], [], []
    for d in run_dirs:
        gcsv = os.path.join(os.path.abspath(d), 'genome.csv')
        if not os.path.exists(gcsv):
            print(f'  ! skipping {os.path.basename(d.rstrip("/"))}: no genome.csv')
            continue
        r = run_outcome(d)
        gens, cen = ll.centroid_trajectory(gcsv)
        r['gens'] = gens[::stride].tolist()
        out_runs.append(r)
        stacks.append(cen[::stride].astype(np.float64))
        names.append(r['run'])
    return out_runs, stacks, names


def _resolve_run_dirs(args):
    """[(dir, condition)] from either --logs-root/--env discovery or explicit --run."""
    pairs = []
    if getattr(args, 'logs_root', None) and getattr(args, 'env', None):
        pairs = discover_runs(args.logs_root, args.env,
                              conditions=tuple(args.conditions), family=args.family)
        if not pairs:
            raise SystemExit(f'no runs found under {args.logs_root} for env '
                             f'{args.env!r} / conditions {args.conditions}')
    for d in (getattr(args, 'run', None) or []):
        pairs.append((d, _condition_of(d)))
    if not pairs:
        raise SystemExit('pass --logs-root with --env, or one or more --run')
    return pairs


def _condition_of(run_dir):
    """Infer the condition from the run path (…/<condition>/<family>/auto-run/<run>)."""
    parts = os.path.abspath(run_dir).split(os.sep)
    for c in ('evolution', 'learning', 'pure_rl'):
        if c in parts:
            return c
    return 'unknown'


def cmd_planecheck(args):
    """FREE go/no-go: score every plane method on these runs, evaluate nothing.

    Run this before spending a grid. It needs no simulator, only genome.csv.
    """
    pairs = _resolve_run_dirs(args)
    run_dirs = [d for d, _ in pairs]
    conds = [c for _, c in pairs]
    runs, stacks, names = _load_run_stacks(run_dirs, stride=args.stride)
    if not stacks:
        raise SystemExit('no run supplied a genome.csv')
    thr = args.collapse_threshold
    groups = ['underperform' if r['final_fitness'] < thr else 'succeed' for r in runs]

    print(f'\n=== plane check: {len(stacks)} runs (generation stride {args.stride}) ===')
    for r, g, c in zip(runs, groups, conds):
        print(f'  {r["run"][:52]:<52} {c:<10} f={r["final_fitness"]:5.2f} '
              f'pop={r["final_pop"]:7.1f}  {g}')
    traj_meta = [{'run': r['run'], 'group': g} for r, g in zip(runs, groups)]
    anchor0, anchor_desc = _pick_anchor(args, runs, stacks, traj_meta)
    print(f'  anchor: {anchor_desc}')
    print(f'\n  {"method":<12}{"EVR1":>7}{"EVR2":>7}{"within":>8}{"disp":>7}{"sep":>7}   verdict')
    best = None
    for m in ('joint', 'within', 'balanced', 'supervised'):
        anchor, U, evr = build_plane_basis(stacks, method=m, groups=groups, anchor=anchor0)
        d = plane_diagnostics(stacks, U, anchor)
        frac = d['within_median'] / d['ceiling_median'] if d['ceiling_median'] else np.nan
        verdict = ('GOOD' if frac >= 0.40 else 'MARGINAL' if frac >= 0.20 else 'NO-GO')
        print(f'  {m:<12}{evr[0]:7.3f}{evr[1]:7.3f}{d["within_median"]:8.3f}'
              f'{d["disp_median"]:7.3f}{d["sep_ratio"]:7.3f}   {verdict} '
              f'({frac:.0%} of ceiling)')
        if best is None or d['within_median'] > best[1]['within_median']:
            best = (m, d)
    print(f'\n  ceiling (best 2D plane for a single run alone): '
          f'{best[1]["ceiling_median"]:.3f} — nothing can beat this')
    print(f'  best score here: {best[0]!r} at within={best[1]["within_median"]:.3f}')
    if best[1]['within_median'] / best[1]['ceiling_median'] < 0.20:
        print('\n  !! LOW SCORE across every method. That is a reason to LOOK, not\n'
              '     to abandon: a plane can score badly and still draw a clean\n'
              '     figure when what it drops is isotropic drift rather than the\n'
              '     systematic direction. Render --fig and judge on the picture.')
    if args.fig:
        print()
        fig_trajectories(args.fig, stacks, conds, names,
                         [r['final_fitness'] for r in runs],
                         list(args.methods), groups, anchor0,
                         args.title or f'{len(stacks)} runs, one plane per panel '
                                       f'(anchor = mean of all centroids)')


def _pick_anchor(args, runs, stacks, traj_meta):
    """Choose the grid's origin genome. Returns (anchor, description).

    NEVER AVERAGE GENOMES ACROSS INDEPENDENT RUNS. Measured on the 16 hard runs:
    their gen-1000 centroids are mutually ORTHOGONAL (mean pairwise cosine
    +0.002), because hidden unit j in one run has no correspondence to hidden
    unit j in another — the network has permutation symmetry, so independently
    evolved weights are unrelated coordinate-by-coordinate. Averaging N such
    vectors shrinks the norm by ~1/sqrt(N): ||centroid|| falls 35.35 -> 5.44 and
    mean |w| falls 0.493 -> 0.069. That average is a near-dead network. Probed,
    it scores 2.73 (RL off) / 3.44 (RL on) against a 4.798 viability threshold,
    and since the grid is anchor + alpha*u + beta*v, EVERY cell inherits it: the
    pooled-anchor hard grid spanned only f = 1.7 .. 4.9 and read 2/16 real run
    endpoints as viable when all 16 in fact probe at 15.06 mean.

    Gen-1 centroids are no better as a shared origin — they are also mutually
    orthogonal (cosine -0.0004), merely small (||theta|| = 0.836), so runs only
    LOOK like they share a starting point because everything near zero norm
    projects near the origin.

    So the anchor must be ONE REAL GENOME. Averaging WITHIN a single run is fine
    (its own centroids across generations are strongly correlated), which is why
    the trajectory mean of one run is a live network and Stage 3 worked.
    """
    if not args.anchor_run:
        return None, 'pooled mean of all centroids (WARNING: averages across runs)'

    if args.anchor_run == 'auto':
        ok = [(i, r) for i, r in enumerate(runs)
              if r['final_fitness'] >= args.collapse_threshold]
        if not ok:
            raise SystemExit('--anchor-run auto: no run cleared --collapse-threshold')
        fits = np.array([r['final_fitness'] for _, r in ok])
        pick = int(np.argsort(fits)[len(fits) // 2])     # median successful run
        idx, chosen = ok[pick]
        how = f'auto: median successful run by final fitness ({chosen["final_fitness"]:.2f})'
    else:
        want = os.path.basename(os.path.abspath(args.anchor_run).rstrip('/'))
        matches = [i for i, r in enumerate(runs) if r['run'] == want]
        if not matches:
            raise SystemExit(f'--anchor-run {want!r} is not among the discovered runs:\n  '
                             + '\n  '.join(r['run'] for r in runs))
        idx = matches[0]
        chosen = runs[idx]
        how = 'named'

    C = stacks[idx]
    anchor = C.mean(axis=0)   # trajectory mean of ONE run, as stage3_pca.py does
    print(f'[stage10/anchor] {how}: {chosen["run"]}')
    print(f'                 ||anchor|| {np.linalg.norm(anchor):.3f}  '
          f'mean |w| {np.abs(anchor).mean():.4f}   '
          f'(pooled mean would be {np.linalg.norm(np.vstack(stacks).mean(axis=0)):.3f} / '
          f'{np.abs(np.vstack(stacks).mean(axis=0)).mean():.4f})')
    return anchor, f'trajectory mean of {chosen["run"]} ({how})'


def cmd_planejobs(args):
    """Build ONE plane spanning every supplied run, and write its grid jobs.

    The plane is pooled over ALL runs on purpose. Fitting a separate plane per
    outcome group would put the landscapes in different coordinate systems with
    different extents, and FDC / dispersion / neutrality have no absolute scale —
    comparing them across such planes would be meaningless. One shared plane
    costs ONE grid evaluation and keeps every cross-group number on the same
    axes; the outcome split then shows up as the trajectories drawn over that
    common surface, plus the map strata from the `strata` command.

    This matters concretely: out/stage3_hard_coarse was anchored on a single run
    (rerun6) that is one of the two POPULATION COLLAPSES in the hard set — 3.80
    final fitness against ~7.0 for the eleven runs that worked. Every hard number
    in Stage 9 is therefore read off a plane fitted to a failure trajectory.

    ALWAYS RUN `planecheck` FIRST — it is free and it will often say no. Measured
    on the hard evolution set (13 runs), the best of the three plane methods puts
    only 3.9% of each run's own motion in-plane against a 66% ceiling, so the
    trajectories would render as stationary dots. Do NOT make this call on EVR:
    re-anchoring those 13 runs at the old two-run mean RAISES EVR from 0.119 to
    0.291 while LOWERING within-run capture from 0.013 to 0.008, because the
    constant offset between anchor and cloud is counted as explained variance.
    """
    runs = []
    for d, cond in _resolve_run_dirs(args):
        r = run_outcome(d)
        r['condition'] = cond
        runs.append(r)
    thr = args.collapse_threshold
    for r in runs:
        r['group'] = 'underperform' if r['final_fitness'] < thr else 'succeed'

    stacks, traj_meta = [], []
    for r in runs:
        gcsv = os.path.join(r['dir'], 'genome.csv')
        if not os.path.exists(gcsv):
            print(f'  ! skipping {r["run"]}: no genome.csv')
            continue
        gens, cen = ll.centroid_trajectory(gcsv)
        stacks.append(cen)
        traj_meta.append({**r, 'gens': gens.tolist(), '_cen': cen})
    if not stacks:
        raise SystemExit('no run supplied a genome.csv')

    groups = [t['group'] for t in traj_meta]
    anchor0, anchor_desc = _pick_anchor(args, runs, stacks, traj_meta)
    anchor, U, evr = build_plane_basis(stacks, method=args.plane_method, groups=groups,
                                       anchor=anchor0)
    plane = {'anchor': anchor, 'u': U[:, 0], 'v': U[:, 1],
             'explained_variance_ratio': np.asarray(evr)}
    diag = plane_diagnostics(stacks, U, anchor)
    for t in traj_meta:
        t['traj'] = ll.project_onto_plane(t.pop('_cen'), plane).tolist()

    # Extent: the founder cloud of the runs, pooled, so the grid covers the
    # region the populations actually occupied rather than one run's cloud.
    ext = []
    for r in runs:
        gcsv = os.path.join(r['dir'], 'genome.csv')
        if not os.path.exists(gcsv):
            continue
        try:
            fdf = ll.load_genome_csv(gcsv, record_type='founder')
            if len(fdf) == 0:
                continue
            proj = ll.project_onto_plane(np.vstack(fdf['genome'].to_numpy()), plane)
            ext.append(float(np.percentile(np.abs(proj), 98)))
        except Exception as e:  # noqa: BLE001 - a missing founder dump is not fatal
            print(f'  ! founder cloud unavailable for {r["run"]}: {e}')
    founder_ext = float(np.max(ext)) if ext else float('nan')
    # The grid must contain every path it is meant to display. With the anchor on
    # ONE run, the others sit off-centre and a founder-cloud extent can crop them,
    # so take whichever bound is larger and say which one won.
    traj_ext = float(np.max([np.abs(np.asarray(t['traj'], dtype=float)).max()
                             for t in traj_meta])) * args.extent_margin
    if np.isfinite(founder_ext) and founder_ext >= traj_ext:
        extent, ext_how = founder_ext, f'pooled founder cloud p98 over {len(ext)} runs'
    else:
        extent, ext_how = traj_ext, (f'projected trajectory extent x{args.extent_margin:g} '
                                     f'over {len(traj_meta)} runs')
    if not np.isfinite(extent) or extent <= 0:
        extent = float(np.sqrt(ll.GENOME_SIZE * 0.047))
        ext_how = 'L2 fallback'
    print(f'[stage10/extent] {extent:.3f} ({ext_how}); founder p98 {founder_ext:.3f}, '
          f'trajectories {traj_ext:.3f}')
    plane['alpha_extent'] = extent
    plane['beta_extent'] = extent
    plane['meta'] = {
        'kind': f'pooled_{args.plane_method}', 'grid': args.grid, 'stage': 10,
        'extent_method': ext_how,
        'anchor_method': anchor_desc,
        'explained_variance_ratio': evr,
        'plane_method': args.plane_method,
        'diagnostics': _jsonable(diag),
        'runs': [{k: v for k, v in t.items() if k != 'traj'} for t in traj_meta],
        'run_trajectories': [{'run': t['run'], 'group': t['group'],
                              'condition': t.get('condition', 'unknown'),
                              'gens': t['gens'], 'traj': t['traj']} for t in traj_meta],
        'collapse_threshold': thr,
    }

    frac = diag['within_median'] / diag['ceiling_median'] if diag['ceiling_median'] else np.nan
    print(f'[stage10/plane] {len(traj_meta)} runs pooled, method={args.plane_method}, '
          f'extent={extent:.3f}, EVR=[{evr[0]:.3f}, {evr[1]:.3f}]')
    print(f'  within-run motion captured {diag["within_median"]:.3f} '
          f'(ceiling {diag["ceiling_median"]:.3f}, {frac:.0%} of it), '
          f'net displacement {diag["disp_median"]:.3f}, separation {diag["sep_ratio"]:.3f}')
    if frac < 0.20:
        print(f'  !! Only {frac:.0%} of the achievable within-run motion is in this '
              f'plane.\n     Check `planecheck --fig` and confirm the paths still read '
              f'as journeys\n     before spending the grid. A low score is not decisive '
              f'on its own —\n     what it drops may be isotropic drift — but it is '
              f'worth one look.')
    for g in ('underperform', 'succeed'):
        sel = [t for t in traj_meta if t['group'] == g]
        if sel:
            fits = ', '.join('%.2f' % t['final_fitness'] for t in sel)
            print(f'  {g:<13} n={len(sel):2d}  final f = {fits}')

    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    lg.write_grid_jobs(args.out_dir, plane, args.resolution, args.repeats,
                       map_start=args.map_start, rl_passes=(False, True),
                       config_overrides=cfg,
                       extra_meta={'stage': 10, 'grid': args.grid})


def cmd_trajplot(args):
    """The finished figure: the evaluated surface with EVERY run drawn on it.

    Reads the grid the array produced plus the trajectories `planejobs` stored
    in meta.json, so the paths need no recomputation and are guaranteed to be
    the ones the plane was fitted to. Both RL passes are drawn on a shared
    colour scale, with the viability contour, and the runs coloured by condition
    and dashed where the population collapsed.
    """
    G = load_cube(args.grid_dir)
    pm = (G['meta'].get('plane_meta') or {})
    trajs = pm.get('run_trajectories') or []
    if not trajs:
        raise SystemExit(f'{args.grid_dir}/meta.json has no run_trajectories — '
                         f'was it written by stage10_strata.py planejobs?')
    k, _ = _resolve_k(args)
    thr = ld.viability_threshold(k)

    f_evo = np.nanmean(G['cube_evo'], axis=2)
    f_lrn = np.nanmean(G['cube_learn'], axis=2)
    lo = float(np.nanmin([f_evo, f_lrn]))
    hi = float(np.nanmax([f_evo, f_lrn]))

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, f, lab in ((axs[0], f_evo, 'RL off (evolution pass)'),
                       (axs[1], f_lrn, 'RL on (learning pass)')):
        im = _imshow(ax, f, G['extent'], 'viridis', f'f(θ) — {lab}', lo, hi)
        if np.nanmin(f) < thr < np.nanmax(f):
            cs = ax.contour(G['A'], G['B'], f, levels=[thr], colors='white', linewidths=1.6)
            ax.clabel(cs, fmt=lambda v: f'viable ({v:.2f})', fontsize=7)
        for t in trajs:
            P = np.asarray(t['traj'], dtype=float)
            if P.size == 0:
                continue
            collapsed = t.get('group') == 'underperform'
            c = CONDITION_COLOR.get(t.get('condition', 'unknown'), '#dddddd')
            ax.plot(P[:, 0], P[:, 1], color=c, lw=2.0 if collapsed else 1.3,
                    ls='--' if collapsed else '-', alpha=0.9)
            ax.plot(P[0, 0], P[0, 1], 'o', ms=3.5, color=c, mfc='white', mew=1.0)
            ax.plot(P[-1, 0], P[-1, 1], '*', ms=10 if collapsed else 7, color=c)
        ax.set_xlim(G['extent'][0], G['extent'][1])
        ax.set_ylim(G['extent'][2], G['extent'][3])
        fig.colorbar(im, ax=ax, fraction=0.046)

    conds = {t.get('condition', 'unknown') for t in trajs}
    h = [plt.Line2D([], [], color=c, lw=2, label=n)
         for n, c in CONDITION_COLOR.items() if n in conds]
    h += [plt.Line2D([], [], color='0.9', lw=2, ls='--', label='collapsed run'),
          plt.Line2D([], [], color='0.9', marker='o', mfc='white', ls='', label='first gen'),
          plt.Line2D([], [], color='0.9', marker='*', ls='', label='last gen')]
    # Dark legend face: the marker entries are drawn light so they read against
    # the viridis surface, which makes them invisible on the default white box.
    leg = axs[1].legend(handles=h, fontsize=6.5, loc='lower right',
                        facecolor='0.15', edgecolor='0.4', framealpha=0.85)
    for txt in leg.get_texts():
        txt.set_color('white')

    n_by = {c: sum(1 for t in trajs if t.get('condition') == c) for c in conds}
    made = ', '.join(f'{v} {kk}' for kk, v in sorted(n_by.items()))
    # Report the ACTUAL anchor: it is the difference between a live surface and a
    # slice through a dead averaged genome, so it belongs on the figure itself.
    anc = pm.get('anchor_method', 'unrecorded')
    if 'trajectory mean of' in anc:
        anc = 'anchor: trajectory mean of ' + anc.split('trajectory mean of ')[1].split(' (')[0]
    else:
        anc = f'anchor: {anc}'
    fig.suptitle(f'Stage 10 — {args.env}: {made} on one {pm.get("plane_method", "?")} '
                 f'plane\n{anc}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out or os.path.join(args.grid_dir, f'trajectories_{args.env}.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'[stage10/trajplot] {len(trajs)} runs over a {G["N"]}x{G["N"]} grid '
          f'(R={G["R"]}) -> {out}')


def _add_discovery_args(p):
    """Either point at the logs tree and name an environment, or list runs by hand.

    Discovery keeps the cluster invocation short and stable: --logs-root
    logs_landscapes --env roam60d5 picks up every evolution and learning run for
    the hard environment without naming sixteen directories.
    """
    p.add_argument('--logs-root', default=None,
                   help='e.g. logs_landscapes — layout '
                        '<root>/<condition>/<family>/auto-run/<env>_*/genome.csv')
    p.add_argument('--env', default=None,
                   help="run-dir prefix naming the environment: 'baseline' or 'roam60d5'")
    p.add_argument('--conditions', nargs='+', default=['evolution', 'learning'],
                   help='conditions to pool onto the one plane (pure_rl logs no '
                        'genome.csv and will be skipped with a warning)')
    p.add_argument('--family', default='standard',
                   help='the sub-directory under each condition (default: standard)')
    p.add_argument('--run', action='append', default=[],
                   help='an explicit run dir; repeatable, and combines with discovery')


def main():
    ap = argparse.ArgumentParser(description='Stage 10 — stratified (aggregated) landscapes')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('strata', help='low / high / all landscapes for one environment')
    p.add_argument('--grid-dir', required=True, help='a Stage-5 grid out-dir')
    p.add_argument('--env', required=True, help='environment label, e.g. baseline / hard')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--frac', type=float, default=0.5,
                   help='share of maps in each tail stratum (default 0.5 = halves)')
    p.add_argument('--k', type=float, default=ld.K_DEFAULT)
    p.add_argument('--k-json', default=None, help='calibration.json from the Stage-9 B1 step')
    p.set_defaults(func=cmd_strata)

    p = sub.add_parser('planecheck',
                       help='FREE go/no-go: score every plane method + draw the '
                            'trajectory preview, evaluate nothing')
    _add_discovery_args(p)
    p.add_argument('--collapse-threshold', type=float, default=5.0)
    p.add_argument('--stride', type=int, default=5,
                   help='generation subsampling for the SVD (default 5; directions '
                        'are insensitive to it)')
    p.add_argument('--anchor-run', default=None,
                   help="origin genome: a run dir/basename, or 'auto'. Matters here too "
                        "— for --plane-method joint the anchor changes the DIRECTIONS.")
    p.add_argument('--fig', default=None,
                   help='write the trajectory preview PNG here (strongly recommended '
                        '— the scores alone have misled on this data in both directions)')
    p.add_argument('--methods', nargs='+', default=['joint', 'within', 'balanced'],
                   choices=['joint', 'within', 'balanced', 'supervised'],
                   help='which planes to draw in --fig')
    p.add_argument('--title', default=None)
    p.set_defaults(func=cmd_planecheck)

    p = sub.add_parser('planejobs',
                       help='build one pooled plane over many runs + its grid jobs')
    _add_discovery_args(p)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--anchor-run', default=None,
                   help="Origin genome = the trajectory mean of ONE run (a run dir, its "
                        "basename, or 'auto' for the median successful run). STRONGLY "
                        "RECOMMENDED: omitting this averages centroids across runs, "
                        "which are mutually orthogonal, giving a near-dead anchor and a "
                        "grid of dead genomes — see _pick_anchor.__doc__.")
    p.add_argument('--extent-margin', type=float, default=1.05,
                   help='pad the trajectory-derived grid extent by this factor')
    p.add_argument('--plane-method',
                   choices=['joint', 'within', 'balanced', 'supervised'],
                   default='balanced',
                   help='joint = Stage-3 PCA about a global mean; within = PCA on '
                        'each run about its OWN mean (best for trajectory figures); '
                        'supervised = progress direction x outcome contrast')
    p.add_argument('--collapse-threshold', type=float, default=5.0,
                   help='final avg_fitness below this labels a run "underperform" '
                        '(default 5.0: the hard set is bimodal at ~3.8 vs ~7.0)')
    p.add_argument('--grid', choices=['coarse', 'fine'], default='coarse')
    p.add_argument('--resolution', type=int, default=25)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--map-start', type=int, default=0)
    p.add_argument('--config-overrides', default=None)
    p.set_defaults(func=cmd_planejobs)

    p = sub.add_parser('trajplot',
                       help='final figure: evaluated surface + every run drawn on it')
    p.add_argument('--grid-dir', required=True,
                   help='a stage10 planejobs out-dir, after the array + merge_shards')
    p.add_argument('--env', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--k', type=float, default=ld.K_DEFAULT)
    p.add_argument('--k-json', default=None)
    p.set_defaults(func=cmd_trajplot)

    p = sub.add_parser('compare', help='cross-environment / cross-stratum table')
    p.add_argument('--metrics', action='append', required=True,
                   help='strata_<env>.json (repeatable)')
    p.add_argument('--out-dir', required=True)
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
