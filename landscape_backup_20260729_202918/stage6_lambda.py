"""
landscape/stage6_lambda.py
======================================================================
Stage 6 — lambda, the ruggedness scalar (Weinberger 1990).

The 2D landscape pictures are ONE viewing angle through a 3910-D space and
generalise to nothing on their own; slicing also tends to make surfaces look
smoother than they are. lambda walks the FULL space with the ACTUAL mutation
operator the GA applies — no projection, no plane, no choice of angle. It is the
ruggedness CLAIM; the maps are the illustration.

Method (Weinberger's random-walk autocorrelation), with the statistical
hardening the n=1 / 100-step pilot lacked:
  1. W INDEPENDENT random mutational walks of `steps` (default 500) from a shared
     start genome, each using the GA's real between-generation operator: every
     weight, with prob MUT_PROB=0.03, gets += N(0, MUT_SIGMA=0.1), clip [-1,1].
     (Real operator: lambda answers "how many mutations until fitness forgets
     where it started?", so it must use the mutations evolution experiences.)
       - LONGER walks: with steps=100 and lambda~10 you get ~3 effective
         independent samples, so rho(1) has a huge SE. steps=500 fixes this.
       - MULTIPLE walks: averaging rho(k) across W walks — and the spread ACROSS
         walks — is what turns "7.03 vs 10.70" into a claim with an error bar.
  2. Fitness at every step, R-AVERAGED over R different maps (varying-map noise
     control). R-averaging inside a step is non-negotiable: evaluation noise
     lowers step-to-step resemblance for reasons unrelated to terrain and would
     fake a low rho(1) -> fake-low lambda.
  3. Per walk: optional DETREND (default linear) before autocorrelation. A walk
     from a random start drifts UP (a dead genome can only improve) and one from
     the evolved centroid erodes DOWN; that non-stationary trend inflates rho and
     makes the AR(1) lambda meaningless. Detrend removes it; walks with large
     residual drift are flagged.
  4. rho(k) averaged across walks. lambda from TWO estimators:
       - lambda_rho1 = -1/ln(mean rho(1))           (legacy, rho(1)-only)
       - lambda_fit  = LS fit of ln rho(k) = -k/lambda over k=1..KFIT (default 10)
         — robust, not hostage to the rho(1) noise floor. This is the headline.
     Cross-walk SE on both, so two conditions can be compared honestly.
     Small lambda = rugged; large lambda = smooth.

Two start genomes per seed (random + evolved final centroid) probe whether
ruggedness is global or local to one region. Run for BOTH environments and ALL
seeds: consistency across seeds is what proves ruggedness is a property of the
ENVIRONMENT, not a seed accident.

Job id format: w{walk}_s{step}_r{repeat}.

Usage
-----
  python landscape/stage6_lambda.py jobs --genome-csv <run/genome.csv> \
      --out-dir landscape/out/stage6_baseline_random --start random \
      --walks 5 --steps 500 --repeats 8 [--rl] [--config-overrides '{...}'] [--walk-seed 0]
  python landscape/stage6_lambda.py run     --out-dir ... --params <params.json> [--shard i/N]
  python landscape/stage6_lambda.py analyze --out-dir ... [--kfit 10] [--detrend linear|none]
  python landscape/stage6_lambda.py all --genome-csv ... --params ... --out-dir ... \
      --start random --walks 2 --steps 40 --repeats 2      # small LOCAL test
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ll_common as ll

MUT_PROB = 0.03    # GAManager MUT_PROB (between-generation)
MUT_SIGMA = 0.1    # GAManager MUT_SIGMA

_ID_RE = re.compile(r'w(\d+)_s(\d+)_r(\d+)')


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
    W = args.walks

    # W INDEPENDENT walks from the shared start. SeedSequence.spawn gives
    # statistically independent streams (not merely offset seeds).
    child_seeds = np.random.SeedSequence(args.walk_seed).spawn(W)

    genomes = {}
    for w in range(W):
        rng = np.random.default_rng(child_seeds[w])
        g = theta0
        genomes[f'w{w}_s0'] = g
        for t in range(1, args.steps + 1):
            g = mutate(g, rng)
            genomes[f'w{w}_s{t}'] = g

    R = args.repeats
    rl = bool(args.rl)
    jobs = []
    for w in range(W):
        for t in range(args.steps + 1):
            for r in range(R):
                # Each repeat draws a DIFFERENT map (varying-map averaging), the
                # SAME map set across walks so terrain is common-mode.
                jobs.append({'id': f'w{w}_s{t}_r{r}', 'genome': f'w{w}_s{t}',
                             'map_index': r, 'rl': rl})

    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    ll.write_jobs(os.path.join(args.out_dir, 'jobs.json'), genomes, jobs, config_overrides=cfg)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'start': args.start, 'start_label': start_label,
                   'walks': W, 'steps': args.steps, 'repeats': R, 'rl': rl,
                   'walk_seed': args.walk_seed, 'config_overrides': cfg,
                   'mut_prob': MUT_PROB, 'mut_sigma': MUT_SIGMA}, f, indent=2)
    print(f'[stage6] wrote {len(jobs)} jobs '
          f'({W} walks x {args.steps + 1} steps x {R} repeats) -> {args.out_dir}/jobs.json')


def cmd_run(args):
    ll.run_probe(args.params, os.path.join(args.out_dir, 'jobs.json'),
                 os.path.join(args.out_dir, 'results.csv'), shard=args.shard)


def _autocorr(x, kmax):
    """Weinberger autocorrelation r(k) of series x (population form)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    denom = np.sum(d * d)
    out = []
    for k in range(1, kmax + 1):
        num = np.sum(d[:n - k] * d[k:])
        out.append(num / denom if denom > 0 else np.nan)
    return np.array(out)


def _linear_detrend(y):
    """Subtract a least-squares linear trend; returns residuals + slope."""
    x = np.arange(len(y), dtype=float)
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x), float(b[0])


def _fit_lambda(rho, kfit):
    """LS fit of ln rho(k) = -k/lambda over k=1..kfit where rho(k)>0.
    Returns lambda (correlation length) or nan/inf."""
    ks = np.arange(1, len(rho) + 1)
    m = (ks <= kfit) & (rho > 0) & np.isfinite(rho)
    if m.sum() < 2:
        return float('nan')
    slope = np.polyfit(ks[m], np.log(rho[m]), 1)[0]
    return (-1.0 / slope) if slope < 0 else float('inf')


def _lam_from_rho1(r1):
    return (-1.0 / np.log(r1)) if (0 < r1 < 1) else (float('inf') if r1 >= 1 else float('nan'))


def cmd_analyze(args):
    out_dir = args.out_dir
    res = ll.load_results(os.path.join(out_dir, 'results.csv')).copy()
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    kmax = args.kmax
    kfit = args.kfit

    # parse walk/step from id (tolerate the legacy s{t}_r{r} single-walk format)
    m = res['id'].str.extract(_ID_RE)
    if m[0].isna().all():
        leg = res['id'].str.extract(r's(\d+)_r(\d+)')
        res['walk'] = 0
        res['step'] = leg[0].astype(int)
    else:
        res['walk'] = m[0].astype(int)
        res['step'] = m[1].astype(int)

    # per-walk step-mean fitness series (mean over the R map repeats)
    walk_ids = sorted(res['walk'].unique())
    series = []
    for w in walk_ids:
        s = res[res['walk'] == w].groupby('step')['mean_fitness'].mean().sort_index()
        series.append(s.to_numpy())
    T = min(len(s) for s in series)
    raw = np.vstack([s[:T] for s in series])          # [W, T] pre-detrend
    W = raw.shape[0]
    kmax = min(kmax, T - 2)

    # detrend per walk + record residual drift
    detrend = args.detrend
    slopes = []
    proc = np.empty_like(raw)
    for i in range(W):
        if detrend == 'linear':
            proc[i], sl = _linear_detrend(raw[i])
        else:
            proc[i], sl = raw[i], float(np.polyfit(np.arange(T), raw[i], 1)[0])
        slopes.append(sl)

    # per-walk autocorrelation + lambda, then average across walks
    rho_w = np.vstack([_autocorr(proc[i], kmax) for i in range(W)])   # [W, kmax]
    mean_rho = np.nanmean(rho_w, axis=0)
    se_rho = np.nanstd(rho_w, axis=0, ddof=1) / np.sqrt(W) if W > 1 else np.full(kmax, np.nan)

    lam_fit_w = np.array([_fit_lambda(rho_w[i], kfit) for i in range(W)])
    finite = lam_fit_w[np.isfinite(lam_fit_w)]
    lam_fit_mean = float(np.mean(finite)) if finite.size else float('nan')
    lam_fit_se = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else float('nan')

    rho1_mean = float(mean_rho[0])
    rho1_se = float(se_rho[0]) if np.isfinite(se_rho[0]) else float('nan')
    lam_rho1 = _lam_from_rho1(rho1_mean)              # legacy estimator
    lam_curve = _fit_lambda(mean_rho, kfit)          # fit on the averaged curve

    # drift diagnostic: flag walks whose linear trend spans a big fraction of range
    fit_mean = raw.mean(axis=1)
    spans = np.abs(np.array(slopes)) * T
    rng_span = float(raw.max() - raw.min()) or 1.0
    drifty = [int(walk_ids[i]) for i in range(W) if spans[i] > 0.5 * rng_span]

    print(f'\n=== Stage 6 lambda (start={meta["start_label"]}, rl={meta["rl"]}, '
          f'W={W} walks x {T-1} steps, R={meta["repeats"]}, detrend={detrend}) ===')
    print(f'  fitness (raw, all walks): [{raw.min():.3f}, {raw.max():.3f}]  mean {raw.mean():.3f}')
    print(f'  rho(1) = {rho1_mean:.4f} +/- {rho1_se:.4f}  (mean +/- SE across walks)')
    print(f'  lambda_fit  (LS, k=1..{kfit}) = {lam_fit_mean:.3f} +/- {lam_fit_se:.3f}   <- headline')
    print(f'  lambda_rho1 (legacy)          = {lam_rho1:.3f}')
    print(f'  lambda from averaged curve    = {lam_curve:.3f}')
    if drifty:
        print(f'  WARNING: walks with strong residual drift (>0.5*range): {drifty}')
    print('  (small lambda = rugged; large lambda = smooth)')

    summary = {'start_label': meta['start_label'], 'rl': meta['rl'],
               'walks': int(W), 'steps': int(T - 1), 'repeats': meta['repeats'],
               'detrend': detrend, 'kfit': kfit,
               'rho1': rho1_mean, 'rho1_se': rho1_se,
               'lambda': lam_fit_mean, 'lambda_se': lam_fit_se,      # headline = LS fit
               'lambda_rho1': lam_rho1, 'lambda_curve_fit': lam_curve,
               'fit_min': float(raw.min()), 'fit_max': float(raw.max()),
               'fit_mean': float(raw.mean()), 'drifty_walks': drifty,
               'config_overrides': meta.get('config_overrides')}
    json.dump(summary, open(os.path.join(out_dir, 'lambda_summary.json'), 'w'), indent=2)

    # ── plots ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    steps_x = np.arange(T)
    for i in range(W):
        ax1.plot(steps_x, raw[i], '-', lw=0.8, alpha=0.7)
    ax1.plot(steps_x, raw.mean(axis=0), '-', color='k', lw=1.6, label='walk mean')
    ax1.set_xlabel('mutation step'); ax1.set_ylabel('R-avg fitness')
    ax1.set_title(f'{W} random walks (raw)\n{meta["start_label"]} (rl={meta["rl"]})')
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ks = np.arange(1, kmax + 1)
    ax2.axhline(0, color='k', lw=0.6)
    if W > 1:
        ax2.fill_between(ks, mean_rho - se_rho, mean_rho + se_rho, color='#92400E', alpha=0.2)
    ax2.plot(ks, mean_rho, 'o-', color='#92400E', ms=4, label='mean ρ(k) ± SE')
    if np.isfinite(lam_curve) and lam_curve > 0:
        ax2.plot(ks, np.exp(-ks / lam_curve), '--', color='gray',
                 label=f'LS fit λ={lam_fit_mean:.1f}±{lam_fit_se:.1f}')
    ax2.axvline(kfit, color='#0891B2', lw=0.8, ls=':', label=f'kfit={kfit}')
    ax2.set_xlabel('lag k (mutation steps)'); ax2.set_ylabel('autocorrelation ρ(k)')
    ax2.set_title(f'ρ(1)={rho1_mean:.3f}±{rho1_se:.3f}  →  λ_fit={lam_fit_mean:.2f}  (detrend={detrend})')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
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
            p.add_argument('--walks', type=int, default=5)
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
            p.add_argument('--kfit', type=int, default=10)
            p.add_argument('--detrend', default='linear', choices=['linear', 'none'])
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
