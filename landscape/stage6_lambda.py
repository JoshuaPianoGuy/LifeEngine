"""
landscape/stage6_lambda.py
======================================================================
Stage 6 — lambda, the ruggedness scalar (Weinberger 1990).

The 2D landscape pictures are ONE viewing angle through a 3910-D space and
generalise to nothing on their own; slicing also tends to make surfaces look
smoother than they are. lambda walks the FULL space with the ACTUAL mutation
operator the GA applies — no projection, no plane, no choice of angle. It is the
ruggedness CLAIM; the maps are the illustration.

Method (Weinberger's random-walk autocorrelation):
  1. From a start genome, take a random MUTATIONAL walk of ~500 steps using the
     GA's real between-generation operator: each weight, with prob MUT_PROB=0.03,
     gets += N(0, MUT_SIGMA=0.1), then clip to [-1,1]. (Must be the real operator
     — lambda answers "how many mutations until fitness forgets where it started?",
     so it must use the mutations evolution actually experiences.)
  2. Fitness at every step, R-AVERAGED over R different maps. R-averaging inside a
     step is non-negotiable: evaluation noise reduces step-to-step resemblance for
     reasons unrelated to terrain, which would fake a low rho(1) -> fake-low lambda.
  3. rho(1) = lag-1 autocorrelation of the fitness series.
     lambda = -1 / ln(rho(1))   (correlation length, in mutation steps).
     Small lambda  = rugged (fitness decorrelates in few mutations).
     Large lambda  = smooth  (fitness persists over many mutations).

Two start genomes per seed (random + evolved final centroid) probe whether
ruggedness is global or local to one region. Run for BOTH environments and ALL
seeds: consistency across seeds is what proves ruggedness is a property of the
ENVIRONMENT, not a seed accident.

Usage
-----
  python landscape/stage6_lambda.py jobs --genome-csv <run/genome.csv> \
      --out-dir landscape/out/stage6_baseline_random --start random \
      --steps 500 --repeats 8 [--rl] [--config-overrides '{...}'] [--walk-seed 0]
  python landscape/stage6_lambda.py run     --out-dir ... --params <params.json> [--shard i/N]
  python landscape/stage6_lambda.py analyze --out-dir ...
  python landscape/stage6_lambda.py all --genome-csv ... --params ... --out-dir ... \
      --start random --steps 40 --repeats 3         # small LOCAL test
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

MUT_PROB = 0.03    # GAManager MUT_PROB (between-generation)
MUT_SIGMA = 0.1    # GAManager MUT_SIGMA


def mutate(genome, rng):
    """One application of the GA's between-generation mutation operator
    (matches GAManager._mutate: per-weight Bernoulli(MUT_PROB) additive
    N(0,MUT_SIGMA), clip [-1,1]). Returns a NEW array."""
    g = genome.copy()
    mask = rng.random(g.size) < MUT_PROB
    g[mask] += rng.normal(0.0, MUT_SIGMA, size=mask.sum())
    np.clip(g, -1.0, 1.0, out=g)
    return g


def _start_genome(genome_csv, start):
    if start == 'random':
        rng = np.random.default_rng(20260718)
        return np.clip(rng.uniform(-0.1, 0.1, ll.GENOME_SIZE), -1, 1), 'random'
    if start in ('centroid', 'evolved'):
        gens, mat = ll.centroid_trajectory(genome_csv)
        return mat[-1].copy(), f'centroid_g{int(gens[-1])}'
    if start == 'fittest':
        df = ll.load_genome_csv(genome_csv, record_type='fittest').sort_values('generation')
        return df.iloc[-1]['genome'].copy(), f'fittest_g{int(df.iloc[-1]["generation"])}'
    raise ValueError(f'unknown --start {start}')


def cmd_jobs(args):
    os.makedirs(args.out_dir, exist_ok=True)
    theta0, start_label = _start_genome(args.genome_csv, args.start)
    rng = np.random.default_rng(args.walk_seed)

    # Generate the walk: genome at each step (step 0 = start).
    walk = [theta0]
    for _ in range(args.steps):
        walk.append(mutate(walk[-1], rng))

    genomes = {f'w{t}': walk[t] for t in range(len(walk))}
    R = args.repeats
    rl = bool(args.rl)
    jobs = []
    for t in range(len(walk)):
        for r in range(R):
            # Each repeat draws a DIFFERENT map (varying-map averaging) so the
            # step fitness reflects the map pool, not one terrain.
            jobs.append({'id': f's{t}_r{r}', 'genome': f'w{t}', 'map_index': r, 'rl': rl})

    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    ll.write_jobs(os.path.join(args.out_dir, 'jobs.json'), genomes, jobs, config_overrides=cfg)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'start': args.start, 'start_label': start_label, 'steps': args.steps,
                   'repeats': R, 'rl': rl, 'walk_seed': args.walk_seed,
                   'config_overrides': cfg, 'mut_prob': MUT_PROB, 'mut_sigma': MUT_SIGMA}, f, indent=2)
    print(f'[stage6] wrote {len(jobs)} jobs ({len(walk)} steps x {R} repeats) -> {args.out_dir}/jobs.json')


def cmd_run(args):
    ll.run_probe(args.params, os.path.join(args.out_dir, 'jobs.json'),
                 os.path.join(args.out_dir, 'results.csv'), shard=args.shard)


def _autocorr(x, kmax):
    """Weinberger autocorrelation r(k) of series x (population form)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    xm = x.mean()
    d = x - xm
    denom = np.sum(d * d)
    out = []
    for k in range(1, kmax + 1):
        num = np.sum(d[:n - k] * d[k:])
        out.append(num / denom if denom > 0 else np.nan)
    return np.array(out)


def cmd_analyze(args):
    out_dir = args.out_dir
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    # step index from id s{t}_r{r}
    res = res.copy()
    res['step'] = res['id'].str.extract(r's(\d+)_r\d+').astype(int)
    step_mean = res.groupby('step')['mean_fitness'].mean().sort_index()
    steps = step_mean.index.to_numpy()
    fit = step_mean.to_numpy()

    kmax = min(args.kmax, len(fit) - 2) if len(fit) > 2 else 1
    rho = _autocorr(fit, kmax)
    rho1 = rho[0]
    lam = (-1.0 / np.log(rho1)) if (0 < rho1 < 1) else float('inf') if rho1 >= 1 else float('nan')

    print(f'\n=== Stage 6 lambda (start={meta["start_label"]}, rl={meta["rl"]}, '
          f'steps={len(fit)-1}, R={meta["repeats"]}) ===')
    print(f'  fitness range along walk: [{fit.min():.3f}, {fit.max():.3f}]  mean {fit.mean():.3f}')
    print(f'  rho(1) = {rho1:.4f}')
    print(f'  lambda = -1/ln rho(1) = {lam:.3f} mutation steps')
    print('  (small lambda = rugged; large lambda = smooth)')

    summary = {'start_label': meta['start_label'], 'rl': meta['rl'], 'steps': int(len(fit) - 1),
               'repeats': meta['repeats'], 'rho1': float(rho1), 'lambda': float(lam),
               'fit_min': float(fit.min()), 'fit_max': float(fit.max()), 'fit_mean': float(fit.mean()),
               'config_overrides': meta.get('config_overrides')}
    json.dump(summary, open(os.path.join(out_dir, 'lambda_summary.json'), 'w'), indent=2)

    # plots: fitness-along-walk + autocorrelation function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, fit, '-', color='#0891B2', lw=1.2)
    ax1.set_xlabel('mutation step'); ax1.set_ylabel('R-avg fitness')
    ax1.set_title(f'random mutational walk\n{meta["start_label"]} (rl={meta["rl"]})')
    ax1.grid(alpha=0.3)
    ks = np.arange(1, kmax + 1)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.plot(ks, rho, 'o-', color='#92400E', ms=4)
    if np.isfinite(lam) and lam > 0:
        ax2.plot(ks, np.exp(-ks / lam), '--', color='gray', label=f'exp(-k/λ), λ={lam:.1f}')
        ax2.legend()
    ax2.set_xlabel('lag k (mutation steps)'); ax2.set_ylabel('autocorrelation ρ(k)')
    ax2.set_title(f'ρ(1)={rho1:.3f}  →  λ={lam:.2f}')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'lambda.png'), dpi=130)
    print(f'[stage6] plot -> {out_dir}/lambda.png')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('jobs', 'run', 'analyze', 'all'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', required=True)
        if name in ('jobs', 'all'):
            p.add_argument('--genome-csv', required=True)
            p.add_argument('--start', default='random', choices=['random', 'centroid', 'evolved', 'fittest'])
            p.add_argument('--steps', type=int, default=500)
            p.add_argument('--repeats', type=int, default=8)
            p.add_argument('--rl', action='store_true')
            p.add_argument('--walk-seed', type=int, default=0)
            p.add_argument('--config-overrides', default=None)
        if name in ('run', 'all'):
            p.add_argument('--params', required=True)
            p.add_argument('--shard', default=None)
        if name in ('analyze', 'all'):
            p.add_argument('--kmax', type=int, default=30)
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
