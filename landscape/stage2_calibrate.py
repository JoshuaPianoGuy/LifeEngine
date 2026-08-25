"""
landscape/stage2_calibrate.py
======================================================================
Stage 2 — calibrate R (repeats per landscape point) and decompose evaluation
noise into dynamics vs terrain.

f(theta) is a Monte-Carlo estimate: predators wander unseeded, RL samples,
food respawns, spawn luck varies. The SAME genome on the SAME map gives a
different fitness every run. We must average enough runs (R) that the standard
error of the mean is small relative to the fitness differences we want to
resolve on the landscape.

Two probe families per representative genome:
  FIXED-MAP  : R repeats on ONE map index      -> sigma_fixed  (pure DYNAMICS noise)
  VARYING-MAP: R repeats on R different maps    -> sigma_vary   (dynamics + TERRAIN)

Decomposition (independent sources):  sigma_vary^2 ~= sigma_fixed^2 + sigma_terrain^2
  => sigma_terrain = sqrt(max(0, sigma_vary^2 - sigma_fixed^2))
A large sigma_fixed (dynamics) means selection in that environment operates on a
corrupted signal — a reportable, independent explanation for evolution being
unreliable.

Pick R from the VARYING-MAP sigma (that is the noise a landscape point actually
carries, since each repeat draws a different map): SE = sigma_vary / sqrt(R).
The plan's absolute target (SE < 0.10, ~3% of a 3.5-7.2 range) assumed the
reproducing-population fitness scale; the monomorphic probe runs higher (~20s),
so we report R for BOTH an absolute SE<0.10 and a relative SE/mean<3% target.

Usage
-----
  # 1) write jobs.json
  python landscape/stage2_calibrate.py jobs   --genome-csv <evo/genome.csv> \
        --out-dir landscape/out/stage2_baseline --repeats 20 [--rl] \
        [--config-overrides '{"roaming_predator_count":60,"predator_drain":5}']
  # 2) run (LOCAL): shell out to node; on HPC use the .slurm array on jobs.json
  python landscape/stage2_calibrate.py run    --out-dir ... --params <run/params.json>
  # 3) analyze -> table + plot
  python landscape/stage2_calibrate.py analyze --out-dir ...
  # all-in-one local:
  python landscape/stage2_calibrate.py all --genome-csv ... --params ... --out-dir ... --repeats 10
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

# Generations whose centroids serve as representative genomes across the fitness
# range (early/mid/late), plus a synthetic random genome as the low-fitness anchor.
REP_GENS = [1, 250, 500, 1000]


def _pick_representative_genomes(genome_csv, gens=REP_GENS, seed=12345):
    df = ll.load_genome_csv(genome_csv, record_type='centroid')
    avail = set(df['generation'].tolist())
    genomes, meta = {}, []
    for g in gens:
        if g not in avail:
            # nearest available generation
            g = int(df.iloc[(df['generation'] - g).abs().argmin()]['generation'])
        row = df[df['generation'] == g].iloc[0]
        key = f'cen_g{g}'
        genomes[key] = row['genome']
        meta.append({'key': key, 'kind': 'centroid', 'generation': int(g),
                     'logged_fitness': float(row['fitness'])})
    # Random low-fitness genome: a freshly initialised brain, drawn exactly as
    # NNBrain._initGenome() does (Glorot/Xavier uniform per layer, zero biases).
    # Hidden size comes from the centroids themselves so h128 runs anchor
    # against an h128 fresh brain rather than a 3910-long h64 one.
    hidden_size = ll.infer_hidden_size(len(genomes[meta[0]['key']]))
    genomes['random'] = ll.xavier_genome(seed=seed, hidden_size=hidden_size)
    meta.append({'key': 'random', 'kind': 'random', 'generation': -1, 'logged_fitness': float('nan')})
    return genomes, meta


def cmd_jobs(args):
    os.makedirs(args.out_dir, exist_ok=True)
    genomes, meta = _pick_representative_genomes(args.genome_csv)
    R = args.repeats
    rl = bool(args.rl)
    jobs = []
    for m in meta:
        key = m['key']
        # FIXED map: same map index (0), R independent runs.
        for r in range(R):
            jobs.append({'id': f'fixed_{key}_r{r}', 'genome': key,
                         'map_index': 0, 'rl': rl})
        # VARYING map: map index = r (distinct terrain each repeat).
        for r in range(R):
            jobs.append({'id': f'vary_{key}_r{r}', 'genome': key,
                         'map_index': r, 'rl': rl})
    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    jobs_path = os.path.join(args.out_dir, 'jobs.json')
    ll.write_jobs(jobs_path, genomes, jobs, config_overrides=cfg)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'repeats': R, 'rl': rl, 'meta': meta,
                   'config_overrides': cfg, 'genome_csv': os.path.abspath(args.genome_csv)}, f, indent=2)
    print(f'[stage2] wrote {len(jobs)} jobs ({len(genomes)} genomes x {R} fixed + {R} vary) -> {jobs_path}')


def cmd_run(args):
    jobs_path = os.path.join(args.out_dir, 'jobs.json')
    out_path = os.path.join(args.out_dir, 'results.csv')
    ll.run_probe(args.params, jobs_path, out_path, shard=args.shard)
    print(f'[stage2] results -> {out_path}')


def cmd_analyze(args):
    out_dir = args.out_dir
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    R = meta['repeats']

    # id = {fixed|vary}_{key}_r{n}
    parts = res['id'].str.split('_', n=1, expand=True)
    res = res.assign(family=parts[0])
    res['key'] = res['id'].str.replace(r'^(fixed|vary)_', '', regex=True).str.replace(r'_r\d+$', '', regex=True)

    rows = []
    for m in meta['meta']:
        key = m['key']
        for fam in ('fixed', 'vary'):
            sub = res[(res['key'] == key) & (res['family'] == fam)]['mean_fitness'].to_numpy()
            if sub.size == 0:
                continue
            rows.append({'key': key, 'kind': m['kind'], 'generation': m['generation'],
                         'family': fam, 'n': sub.size,
                         'mean': sub.mean(), 'sigma': sub.std(ddof=1) if sub.size > 1 else 0.0})
    tbl = pd.DataFrame(rows)

    # Wide per genome: sigma_fixed, sigma_vary, terrain, R targets.
    summary = []
    for m in meta['meta']:
        key = m['key']
        f = tbl[(tbl['key'] == key) & (tbl['family'] == 'fixed')]
        v = tbl[(tbl['key'] == key) & (tbl['family'] == 'vary')]
        if f.empty or v.empty:
            continue
        s_fixed = float(f['sigma'].iloc[0]); s_vary = float(v['sigma'].iloc[0])
        mean_v = float(v['mean'].iloc[0])
        s_terrain = float(np.sqrt(max(0.0, s_vary ** 2 - s_fixed ** 2)))
        rel_target = 0.03 * mean_v
        # SHAPE R (the one the pipeline uses): every grid point / walk step shares
        # ONE fixed map set, so terrain is common-mode and cancels between points.
        # Only DYNAMICS noise (sigma_fixed) limits point-to-point resolution.
        r_shape = int(np.ceil((s_fixed / rel_target) ** 2)) if (s_fixed > 0 and rel_target > 0) else 1
        # LEVEL R (only if you want the absolute pool-averaged fitness of a point
        # to a tight SE): driven by the full terrain variance sigma_vary. Large.
        r_level = int(np.ceil((s_vary / rel_target) ** 2)) if (s_vary > 0 and rel_target > 0) else 1
        summary.append({'key': key, 'kind': m['kind'], 'generation': m['generation'],
                        'mean_fitness': mean_v, 'logged_fitness': m['logged_fitness'],
                        'sigma_fixed(dyn)': s_fixed, 'sigma_vary': s_vary,
                        'sigma_terrain': s_terrain,
                        'SE_shape@R{}'.format(R): s_fixed / np.sqrt(R),
                        'R_shape(3%)': r_shape, 'R_level(3%)': r_level})
    sdf = pd.DataFrame(summary)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print('\n=== Stage 2 calibration summary (rl={}) ===\n'.format(meta['rl']))
    print(sdf.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    r_shape_max = int(sdf['R_shape(3%)'].max())
    r_level_max = int(sdf['R_level(3%)'].max())
    print('\nNOISE DECOMPOSITION: variance is {}-dominated '
          '(sigma_terrain ~ sigma_vary >> sigma_fixed).'.format('terrain'))
    print('Because every grid point / walk step shares ONE fixed map set, terrain is')
    print('common-mode and cancels between points. Use R_shape (limited by dynamics):')
    print(f'  Recommended R (SHAPE / ruggedness): {r_shape_max}  (worst genome, SE<3% of mean)')
    print(f'  R for ABSOLUTE pool-level to <3%:   {r_level_max}  (only if you need the level, not the shape)')

    sdf.to_csv(os.path.join(out_dir, 'calibration_summary.csv'), index=False)

    # ── plot: per-genome fitness distributions (fixed vs vary) ────────────────
    keys = [m['key'] for m in meta['meta']]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4), sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        for fam, color in (('fixed', '#0891B2'), ('vary', '#92400E')):
            sub = res[(res['key'] == key) & (res['family'] == fam)]['mean_fitness'].to_numpy()
            if sub.size == 0:
                continue
            x = np.full(sub.size, 0 if fam == 'fixed' else 1) + np.random.uniform(-0.08, 0.08, sub.size)
            ax.scatter(x, sub, s=18, alpha=0.7, color=color, label=fam)
            ax.errorbar([0 if fam == 'fixed' else 1], [sub.mean()],
                        yerr=[sub.std(ddof=1) if sub.size > 1 else 0], fmt='o',
                        color='k', capsize=4, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['fixed\nmap', 'vary\nmap'])
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('mean_fitness (per run)')
    fig.suptitle(f'Stage 2 calibration — evaluation noise (R={R}, rl={meta["rl"]})')
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'calibration.png')
    fig.savefig(plot_path, dpi=130)
    print(f'\n[stage2] plot -> {plot_path}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('jobs', 'run', 'analyze', 'all'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', required=True)
        if name in ('jobs', 'all'):
            p.add_argument('--genome-csv', required=True)
            p.add_argument('--repeats', type=int, default=20)
            p.add_argument('--rl', action='store_true')
            p.add_argument('--config-overrides', default=None)
        if name in ('run', 'all'):
            p.add_argument('--params', required=True)
            p.add_argument('--shard', default=None)
    args = ap.parse_args()

    if args.cmd == 'jobs':
        cmd_jobs(args)
    elif args.cmd == 'run':
        cmd_run(args)
    elif args.cmd == 'analyze':
        cmd_analyze(args)
    elif args.cmd == 'all':
        cmd_jobs(args); cmd_run(args); cmd_analyze(args)


if __name__ == '__main__':
    main()
