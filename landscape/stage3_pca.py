"""
landscape/stage3_pca.py
======================================================================
Stage 3 — the PCA plane.

Joint PCA on the evolution + learning centroid trajectories, anchored at the
trajectory MEAN (not gen 0). One plane contains BOTH paths, so "evolution stayed
in the low basin, learning crossed to the peak" becomes a drawing, not an
inference. PCA sees weight vectors only, never fitness — the plane cannot be
accused of being chosen to flatter the result.

Anchor = trajectory mean  -> (0,0) sits mid-journey; the grid reaches gen 0 on
one side and gen 1000 on the other.

Grid extent (two scales, because crossover displaces ~13.6 in L2 while a single
mutation displaces ~1.1 — one grid can't resolve both):
  COARSE: extent = the founder CLOUD radius on the plane (how many basins, is
          there a barrier). Measured from the logged founder-genome dumps.
  FINE:   extent = the trajectory LENGTH on the plane (local texture the coarse
          grid steps over).

Outputs an out-dir with plane.json + jobs.json + meta.json (incl. projected
evo/learn trajectories for overlay). Run with the generic runner, plot with
stage7_plots.py.

Usage
-----
  python landscape/stage3_pca.py --evo-genome-csv <evo/genome.csv> \
      --learn-genome-csv <learn/genome.csv> --out-dir landscape/out/stage3_baseline_coarse \
      --grid coarse --resolution 25 --repeats 8 [--config-overrides '{...}']
"""

import argparse
import json
import os

import numpy as np

import ll_common as ll
import ll_grid


def _project_founder_cloud(genome_csv, plane):
    """Max |alpha|,|beta| of the logged founder genomes projected onto the plane
    (the cloud the population actually occupied). Falls back to the L2 estimate
    from genome variance if no founder dumps exist."""
    try:
        df = ll.load_genome_csv(genome_csv, record_type='founder')
        if len(df) == 0:
            raise ValueError('no founder rows')
        mat = np.vstack(df['genome'].to_numpy())
        proj = ll.project_onto_plane(mat, plane)
        # 98th percentile is robust to a few outliers while still covering the cloud.
        a_ext = float(np.percentile(np.abs(proj[:, 0]), 98))
        b_ext = float(np.percentile(np.abs(proj[:, 1]), 98))
        return a_ext, b_ext, f'founder cloud p98 (n={len(df)})'
    except Exception as e:
        # L2 radius from genome variance ~0.047 -> sqrt(3910*0.047) ~ 13.6
        r = float(np.sqrt(ll.GENOME_SIZE * 0.047))
        return r, r, f'L2 estimate (fallback: {e})'


def build_plane(args):
    evo_gens, evo_cen = ll.centroid_trajectory(args.evo_genome_csv)
    lrn_gens, lrn_cen = ll.centroid_trajectory(args.learn_genome_csv)
    plane = ll.joint_pca_plane([evo_cen, lrn_cen])  # anchor = trajectory mean

    # projected trajectories for overlay
    evo_proj = ll.project_onto_plane(evo_cen, plane)
    lrn_proj = ll.project_onto_plane(lrn_cen, plane)

    if args.grid == 'coarse':
        a_ext, b_ext, how = _project_founder_cloud(args.evo_genome_csv, plane)
    else:  # fine: trajectory length on the plane
        traj = np.vstack([evo_proj, lrn_proj])
        a_ext = float(np.abs(traj[:, 0]).max())
        b_ext = float(np.abs(traj[:, 1]).max())
        how = 'trajectory extent (fine)'
    # square the plotted window to the larger axis so it isn't a sliver
    ext = max(a_ext, b_ext)

    plane['alpha_extent'] = ext
    plane['beta_extent'] = ext
    plane['meta'] = {
        'kind': 'pca', 'grid': args.grid, 'extent_method': how,
        'explained_variance_ratio': [float(x) for x in plane['explained_variance_ratio']],
        'evo_gens': evo_gens.tolist(), 'lrn_gens': lrn_gens.tolist(),
        'evo_traj': evo_proj.tolist(), 'lrn_traj': lrn_proj.tolist(),
        'evo_genome_csv': os.path.abspath(args.evo_genome_csv),
        'learn_genome_csv': os.path.abspath(args.learn_genome_csv),
    }
    return plane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evo-genome-csv', required=True)
    ap.add_argument('--learn-genome-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--grid', choices=['coarse', 'fine'], default='coarse')
    ap.add_argument('--resolution', type=int, default=25)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--map-start', type=int, default=0)
    ap.add_argument('--config-overrides', default=None)
    args = ap.parse_args()

    plane = build_plane(args)
    evr = plane['meta']['explained_variance_ratio']
    print(f'[stage3] PCA plane ({args.grid}): extent={plane["alpha_extent"]:.3f} '
          f'({plane["meta"]["extent_method"]}), EVR=[{evr[0]:.3f}, {evr[1]:.3f}]')

    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    ll_grid.write_grid_jobs(args.out_dir, plane, args.resolution, args.repeats,
                            map_start=args.map_start, rl_passes=(False, True),
                            config_overrides=cfg, extra_meta={'stage': 3, 'grid': args.grid})


if __name__ == '__main__':
    main()
