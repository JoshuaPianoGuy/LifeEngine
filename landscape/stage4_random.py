"""
landscape/stage4_random.py
======================================================================
Stage 4 — random-direction planes, filter-normalised.

PCA directions are derived from the trajectory, so the path lies in the plane by
construction and the terrain along it always looks navigable. To ask "is this
environment rugged?" we need directions chosen WITHOUT reference to where the
population went: random directions.

Filter normalisation (Li et al.): a fixed raw step means different things per
neuron (a small-weight unit is thrown off its operating point; a large-weight
unit barely moves), so a raw random plane looks jagged for weight-scale reasons,
not landscape structure. Each direction is rescaled per-neuron block to that
neuron's weight norm in the anchor (see ll_common.filter_normalize). Blocks:
hidden unit j = 54 W1 + b1[j] (55); output unit k = 64 W2 + b2[k] (65).

Three INDEPENDENT planes: one random plane could be unlucky. If all three show
multiple basins under predators and a single basin in the baseline, direction
choice isn't driving the result.

Anchor = the evolved final centroid (a real genome), so each plane passes
through a point the system actually produced.

Usage
-----
  python landscape/stage4_random.py --genome-csv <run/genome.csv> \
      --out-root landscape/out/stage4_baseline --n-planes 3 --resolution 25 \
      --repeats 8 [--anchor centroid|fittest] [--config-overrides '{...}'] [--seed 0]
Creates <out-root>/plane0, plane1, ... each a grid out-dir.
"""

import argparse
import json
import os

import numpy as np

import ll_common as ll
import ll_grid


def _anchor_genome(genome_csv, which):
    if which == 'fittest':
        df = ll.load_genome_csv(genome_csv, record_type='fittest').sort_values('generation')
        return df.iloc[-1]['genome'].copy()
    gens, mat = ll.centroid_trajectory(genome_csv)
    return mat[-1].copy()


def _random_filter_plane(anchor, rng):
    """Two filter-normalised, orthonormal-ish directions. Filter-normalise each
    random Gaussian direction to the anchor's per-neuron norms, then Gram-Schmidt
    to remove overlap, then re-normalise both to unit length. Extent is set by the
    caller in L2 weight units."""
    d1 = ll.filter_normalize(rng.standard_normal(ll.GENOME_SIZE), anchor)
    d2 = ll.filter_normalize(rng.standard_normal(ll.GENOME_SIZE), anchor)
    u = d1 / np.linalg.norm(d1)
    d2 = d2 - (d2 @ u) * u          # orthogonalise v against u
    v = d2 / np.linalg.norm(d2)
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genome-csv', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--n-planes', type=int, default=3)
    ap.add_argument('--resolution', type=int, default=25)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--anchor', choices=['centroid', 'fittest'], default='centroid')
    ap.add_argument('--extent', type=float, default=None,
                    help='L2 extent of each axis; default = ||theta0|| (relative axis ~[-1,1])')
    ap.add_argument('--map-start', type=int, default=0)
    ap.add_argument('--config-overrides', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    anchor = _anchor_genome(args.genome_csv, args.anchor)
    theta_norm = float(np.linalg.norm(anchor))
    extent = args.extent if args.extent is not None else theta_norm
    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    rng = np.random.default_rng(args.seed)

    print(f'[stage4] anchor={args.anchor} ||theta0||={theta_norm:.3f} extent={extent:.3f} '
          f'({args.n_planes} planes)')
    for p in range(args.n_planes):
        u, v = _random_filter_plane(anchor, rng)
        plane = {'anchor': anchor, 'u': u, 'v': v,
                 'alpha_extent': extent, 'beta_extent': extent,
                 'meta': {'kind': 'random_filter', 'plane_index': p,
                          'theta_norm': theta_norm, 'anchor_kind': args.anchor,
                          'genome_csv': os.path.abspath(args.genome_csv)}}
        out_dir = os.path.join(args.out_root, f'plane{p}')
        ll_grid.write_grid_jobs(out_dir, plane, args.resolution, args.repeats,
                                map_start=args.map_start, rl_passes=(False, True),
                                config_overrides=cfg,
                                extra_meta={'stage': 4, 'plane_index': p})


if __name__ == '__main__':
    main()
