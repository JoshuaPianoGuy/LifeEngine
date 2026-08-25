"""
landscape/make_slice_jobs.py
======================================================================
Write the grid jobs for a SET of landscape slices with a FIXED extent and
resolution, so the slices differ only in WHERE they cut the weight space.

This is the input side of landscape/slice_metrics.py. That script compares
metrics across slices; this one generates the slices to compare. Holding extent
and resolution fixed is the whole point: the first slice_metrics pass mixed slice
choice with grid parameters (25x25 R=10 extent 43.3 vs 21x21 R=5 extent 35.1), so
the spread it measured was not purely slice-to-slice. Everything here is pinned.

ONE TRAJECTORY PER SLICE (the default, and why)
-----------------------------------------------
Each slice is anchored on ONE run's trajectory mean and spanned by that run's own
best 2D plane, so it scores 100% of the achievable within-run ceiling and the run
is drawn faithfully. Putting a SECOND run on the same plane is possible
(--companion) but expensive, and the reason is geometric rather than a tuning
choice:

  * one run's own trajectory spans |alpha|,|beta| <= ~27 (measured over 12
    anchors on the h128 hard runs),
  * but two independent runs' anchors sit ~44 apart in MUTUALLY ORTHOGONAL
    directions, so a plane holding both must span ~41 — and most of that span is
    empty space between two unrelated islands of weight space, not landscape
    anyone's search ever visited.

Holding the sampling density fixed, that wider span needs resolution ~47 instead
of ~31, i.e. 2.3x the probes, to buy mostly dead cells. So: one run per slice by
default, and compare evolution against learning ACROSS slices, which is what
slice_metrics.py is built to do.

CHOOSING THE RESOLUTION — the correlation length sets it
---------------------------------------------------------
The surface must be sampled at least ~2 cells per autocorrelation length or the
grid aliases it and every smoothness metric is meaningless. Measured on the
existing hard slices, lambda ~ 3.7-5.3 weight units (slice_metrics.py reports
lambda_w). With a half-extent of 28 (full width 56):

    resolution 31 -> spacing 1.87 -> lambda/cell ~ 2.0   <- DEFAULT
    resolution 25 -> spacing 2.33 -> lambda/cell ~ 1.6   aliases the hard surface
    resolution 47 -> spacing 1.19 -> lambda/cell ~ 3.1   for the --companion span

R=5 IS ENOUGH — for the metrics this study is for
--------------------------------------------------
R only fights per-cell evaluation noise. For the GLOBAL metrics (FDC, dispersion,
quadratic R2, viable fraction, autocorrelation) the within-slice sampling error is
tiny next to the slice-to-slice spread that is the object of study: FDC SE is
0.005-0.017, while the across-slice spread is 0.13 (hard) to 0.32 (baseline) —
20-65x larger. Given a fixed budget, MORE SLICES AT R=5 beats fewer slices at
R=10.

R=5 is NOT enough for the LOCAL noise-sensitive metrics — at R=5 every cell has
GNR < 1 (vs 92.8% at R=10), so local gradients are unreadable. Nothing is lost by
accepting that here: noise-aware peak counting already returns 0 surviving peaks
at BOTH R=5 and R=10, so those metrics are not informative at either setting.

h128 vs h64 — THE OVERRIDE THAT MUST NOT BE FORGOTTEN
------------------------------------------------------
The existing landscape pipeline was built for hidden_size=64, whose genome is
3910 long (ll_common.GENOME_SIZE is that constant, and asserts on it). The 2026
hard runs under logs/ are hidden_size=128, genome 7814. Grid construction itself
is dimension-agnostic — every cell is anchor + alpha*u + beta*v, so the size comes
from the anchor genome — but the PROBE builds a network from ExperimentParams, so
`hidden_size` MUST be in config_overrides or run_probe.js will build a 64-wide
brain and mis-read a 7814-long genome. This script always emits it.

`map_seed` is emitted per slice too: a genome evolved on seed 2005 must be probed
on seed 2005 terrain, not on whatever the probe defaults to.

Usage
-----
  # 6 slices: the 3 worst and 3 best evolution runs, fixed extent/resolution
  python landscape/make_slice_jobs.py \
      --logs-root logs --condition evolution --n-per-group 3 \
      --out-root landscape/out/slices_h128_hard

  # add the learning arm as its own slices (learning never collapses, so all
  # of its slices are 'succeed')
  python landscape/make_slice_jobs.py \
      --logs-root logs --condition learning --n-per-group 3 \
      --out-root landscape/out/slices_h128_hard

Then evaluate each out-dir on the cluster (see the printed sbatch block), merge
with merge_shards.py, and compare with slice_metrics.py.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import warnings

import numpy as np

# Some macOS BLAS builds raise divide-by-zero/overflow FP warnings from matmul on
# large well-conditioned float64 products. Verified spurious here: the inputs are
# finite (weights live in [-1, 1] after the mutation clamp) and so is the output.
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*matmul.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ll_common as ll
import ll_grid
import stage10_strata as s10


# Measured on the h128 hard runs; see the module docstring.
DEFAULT_EXTENT = 28.0        # half-width, covers a single run's own trajectory
DEFAULT_RESOLUTION = 31      # spacing 1.87 -> ~2 cells per correlation length
DEFAULT_REPEATS = 5
COMPANION_EXTENT = 43.0      # needed if a second run shares the plane
COMPANION_RESOLUTION = 47


def run_final_fitness(run_dir):
    """Mean avg_fitness over the last 50 generations of generations.csv."""
    import csv
    path = os.path.join(run_dir, 'generations.csv')
    vals = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row['avg_fitness']))
            except (KeyError, ValueError):
                pass
    return float(np.mean(vals[-50:])) if vals else float('nan')


def discover(logs_root, condition, family, env_prefix):
    """[(run_dir, seed, final_fitness)] for one condition, newest layout."""
    pat = os.path.join(logs_root, condition, family, 'auto-run', env_prefix + '*')
    out = []
    for d in sorted(glob.glob(pat)):
        pj = os.path.join(d, 'params.json')
        if not (os.path.exists(pj) and os.path.exists(os.path.join(d, 'genome.csv'))):
            continue
        p = json.load(open(pj))
        out.append({'dir': d, 'run': os.path.basename(d),
                    'seed': int(p['map_seed']),
                    'hidden_size': int(p.get('hidden_size', 64)),
                    'roaming': float(p.get('roaming_predator_count', 0)),
                    'drain': float(p.get('predator_drain', 0)),
                    'food_scale': p.get('food_density_scale', 1),
                    'f': run_final_fitness(d)})
    return out


def pick_slices(runs, threshold, n_per_group):
    """The n worst underperformers and the n best successes, spread over seeds.

    Spreading over seeds matters: three replicates of ONE seed share terrain, so
    they would sample three anchors from the same neighbourhood and understate
    how much slices differ."""
    lo = sorted([r for r in runs if r['f'] < threshold], key=lambda r: r['f'])
    hi = sorted([r for r in runs if r['f'] >= threshold], key=lambda r: -r['f'])

    def spread(pool, n):
        chosen, used = [], set()
        for r in pool:                       # first pass: one per seed
            if len(chosen) >= n:
                break
            if r['seed'] not in used:
                chosen.append(r)
                used.add(r['seed'])
        for r in pool:                       # top up if seeds ran out
            if len(chosen) >= n:
                break
            if r not in chosen:
                chosen.append(r)
        return chosen

    return [('underperform', r) for r in spread(lo, n_per_group)] + \
           [('succeed', r) for r in spread(hi, n_per_group)]


def build_slice_plane(run_dir, stride, companion_dir=None):
    """Plane anchored on this run's trajectory mean, spanned by its own best 2D
    directions. With a companion, the plane is fitted to BOTH trajectories."""
    _g, cen = ll.centroid_trajectory(os.path.join(run_dir, 'genome.csv'))
    stacks = [cen[::stride].astype(np.float64)]
    names = [os.path.basename(run_dir)]
    if companion_dir:
        _g2, cen2 = ll.centroid_trajectory(os.path.join(companion_dir, 'genome.csv'))
        stacks.append(cen2[::stride].astype(np.float64))
        names.append(os.path.basename(companion_dir))

    anchor = stacks[0].mean(axis=0)
    method = 'within' if len(stacks) == 1 else 'balanced'
    anchor, U, evr = s10.build_plane_basis(stacks, method=method, anchor=anchor)
    diag = s10.plane_diagnostics(stacks, U, anchor)

    projected = [((C - anchor) @ U).tolist() for C in stacks]
    # anchor/u/v stay ndarrays: ll_grid.build_grid_jobs does
    # `anchor + a*u + b*v` arithmetic on them directly, and lists would
    # concatenate instead of add. ll_common.save_plane serialises them.
    return {
        'anchor': anchor, 'u': U[:, 0], 'v': U[:, 1],
        'explained_variance_ratio': [float(x) for x in evr],
        'meta': {
            'kind': 'slice', 'grid': 'fixed', 'plane_method': method,
            'anchor_method': f'trajectory mean of {names[0]}',
            'runs_on_plane': names,
            'run_trajectories': projected,
            'diagnostics': {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in diag.items()},
            'genome_size': int(anchor.size),
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description='Write fixed-extent, fixed-resolution grid jobs for a set of '
                    'landscape slices, one per anchor run.')
    ap.add_argument('--logs-root', default='logs')
    ap.add_argument('--condition', default='evolution',
                    choices=['evolution', 'learning'])
    ap.add_argument('--family', default='standard')
    ap.add_argument('--env-prefix', default='roam60d5',
                    help='run-dir prefix identifying the environment')
    ap.add_argument('--out-root', required=True,
                    help='parent dir; one sub-dir per slice is created under it')
    ap.add_argument('--n-per-group', type=int, default=3,
                    help='slices per outcome group (underperform / succeed)')
    ap.add_argument('--threshold', type=float, default=4.375,
                    help='viability / collapse threshold on the REPLAY scale '
                         '(default 4.375 = 3.5 food per reproduction attempt / '
                         '0.8 success rate — the projects standing value)')
    ap.add_argument('--extent', type=float, default=None,
                    help=f'half-width, FIXED across slices (default '
                         f'{DEFAULT_EXTENT}, or {COMPANION_EXTENT} with --companion)')
    ap.add_argument('--resolution', type=int, default=None,
                    help=f'cells per axis, FIXED across slices (default '
                         f'{DEFAULT_RESOLUTION}, or {COMPANION_RESOLUTION} with --companion)')
    ap.add_argument('--repeats', type=int, default=DEFAULT_REPEATS)
    ap.add_argument('--stride', type=int, default=5,
                    help='generation subsampling for the plane fit')
    ap.add_argument('--map-start', type=int, default=0)
    ap.add_argument('--learning-rate', type=float, default=0.02,
                    help='RL learning rate pinned into every slice, so the rl=on '
                         'pass means the same thing everywhere (default 0.02, '
                         'the learning arm\'s value).')
    ap.add_argument('--no-pin-rl', action='store_true',
                    help='Do NOT pin epsilon/lr/explore_bonus; let each slice '
                         'inherit them from its anchor run. Makes the rl=on pass '
                         'incomparable across conditions — see the comment in '
                         'main(). Only for reproducing an older run.')
    ap.add_argument('--companion', action='store_true',
                    help='also place the matched run from the OTHER condition on '
                         'each plane. Widens the required extent to ~43 and the '
                         'resolution to ~47 — read the module docstring first.')
    args = ap.parse_args()

    extent = args.extent if args.extent is not None else (
        COMPANION_EXTENT if args.companion else DEFAULT_EXTENT)
    resolution = args.resolution if args.resolution is not None else (
        COMPANION_RESOLUTION if args.companion else DEFAULT_RESOLUTION)

    runs = discover(args.logs_root, args.condition, args.family, args.env_prefix)
    if not runs:
        raise SystemExit(f'no runs with genome.csv under {args.logs_root}/'
                         f'{args.condition}/{args.family}/auto-run/{args.env_prefix}*')
    chosen = pick_slices(runs, args.threshold, args.n_per_group)
    if not chosen:
        raise SystemExit('no runs selected — check --threshold')

    other = None
    if args.companion:
        oc = 'learning' if args.condition == 'evolution' else 'evolution'
        other = discover(args.logs_root, oc, args.family, args.env_prefix)

    spacing = 2.0 * extent / (resolution - 1)
    print(f'FIXED across every slice: extent ±{extent}  resolution {resolution}x{resolution}  '
          f'R={args.repeats}')
    print(f'  cell spacing {spacing:.3f} weight units '
          f'(hard lambda ~3.7-5.3 -> {3.7 / spacing:.1f}-{5.3 / spacing:.1f} cells '
          f'per correlation length; keep this >= 2)')
    print(f'  threshold {args.threshold} on the replay scale\n')

    os.makedirs(args.out_root, exist_ok=True)
    manifest, total = [], 0
    for group, r in chosen:
        companion_dir = None
        if other:
            same_seed = [o for o in other if o['seed'] == r['seed']]
            if same_seed:
                companion_dir = max(same_seed, key=lambda o: o['f'])['dir']

        plane = build_slice_plane(r['dir'], args.stride, companion_dir)
        plane['alpha_extent'] = extent
        plane['beta_extent'] = extent
        plane['meta'].update({'group': group, 'anchor_run': r['run'],
                              'anchor_seed': r['seed'],
                              'anchor_final_fitness': r['f']})

        # The probe builds its network from ExperimentParams, so hidden_size is
        # NOT optional for h128 runs, and map_seed must match the run's terrain.
        cfg = {
            'hidden_size': r['hidden_size'],
            'map_seed': r['seed'],
            'roaming_predator_count': r['roaming'],
            'predator_drain': r['drain'],
        }
        if r['food_scale'] is not None:
            cfg['food_density_scale'] = r['food_scale']

        # ── Pin the RL knobs, or the RL-ON pass is not comparable across slices ──
        # Each slice carries its ANCHOR RUN's params.json, and the two conditions
        # disagree: evolution runs have epsilon_enabled=true (vestigial — they
        # never ran RL at all), learning runs have it false (they ran
        # --no-epsilon). Left alone, the rl=on probe would explore epsilon-greedily
        # on evolution slices and purely on-policy on learning slices, so the A6
        # lift metric (learn - evo) would differ by CONFIG between slices and read
        # as a landscape difference. The old stage10 job script dodged this by
        # feeding a learning run's params.json to every unit; pinning here is the
        # same fix made explicit.
        #
        # The values are the learning arm's real settings — the RL this project
        # actually ran. config.js only honours an explicit `false` for
        # epsilon_enabled, which is what is passed.
        if not args.no_pin_rl:
            cfg['epsilon_enabled'] = False
            cfg['learning_rate'] = args.learning_rate
            cfg['explore_bonus'] = 0

        out_dir = os.path.join(args.out_root, f"{args.condition}_{group}_{r['run']}")
        axes = ll_grid.write_grid_jobs(out_dir, plane, resolution, args.repeats,
                                       map_start=args.map_start,
                                       rl_passes=(False, True),
                                       config_overrides=cfg,
                                       extra_meta={'stage': 'slice', 'group': group})

        # Copy the anchor run's params.json INTO the slice dir. run_probe.js needs
        # a params file for the world settings config_overrides does not carry
        # (grid size, ticks, food); taking it from the anchor run guarantees the
        # probe world matches the world the genome evolved in. Copying rather than
        # referencing makes each slice dir self-contained, so the cluster needs
        # only these dirs — no logs tree at all.
        shutil.copyfile(os.path.join(r['dir'], 'params.json'),
                        os.path.join(out_dir, 'params.json'))
        n_probes = resolution * resolution * args.repeats * 2
        total += n_probes
        d = plane['meta']['diagnostics']
        frac = (d['within_median'] / d['ceiling_median']) if d.get('ceiling_median') else float('nan')
        print(f"  [{group:<12}] f={r['f']:5.2f} seed={r['seed']}  "
              f"plane {frac:.0%} of ceiling  -> {out_dir}")
        manifest.append({'out_dir': out_dir, 'group': group, 'run': r['run'],
                         'seed': r['seed'], 'final_fitness': r['f'],
                         'probes': n_probes, 'config_overrides': cfg,
                         'plane_frac_of_ceiling': frac})

    mpath = os.path.join(args.out_root, f'manifest_{args.condition}.json')
    json.dump({'extent': extent, 'resolution': resolution, 'repeats': args.repeats,
               'threshold': args.threshold, 'slices': manifest},
              open(mpath, 'w'), indent=2)

    print(f'\n  {len(manifest)} slices, {total:,} probes total '
          f'(~{total * 5.5 / 3600:.0f} core-hours at the measured 4-7 s/probe)')
    print(f'  wrote {mpath}')
    print('\nNext — evaluate each out-dir, then:')
    print('  python landscape/merge_shards.py --out-dir <slice out-dir>')
    print('  python landscape/slice_metrics.py \\')
    for m in manifest:
        print(f"      --slice hard:{m['group'][:4]}_{m['seed']}:{m['out_dir']} \\")
    print('      --out-dir landscape/out/slice_metrics_h128')


if __name__ == '__main__':
    main()
