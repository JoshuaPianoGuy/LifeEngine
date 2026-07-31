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


def _walk_step_table(res, value_col):
    """[W, T] matrix of per-(walk, step) means of `value_col`, plus the walk ids."""
    walk_ids = sorted(res['walk'].unique())
    series = [res[res['walk'] == w].groupby('step')[value_col].mean().sort_index().to_numpy()
              for w in walk_ids]
    T = min(len(s) for s in series)
    return np.vstack([s[:T] for s in series]), walk_ids


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

    # per-walk step series. --transform log takes the mean over maps of log f
    # (i.e. the geometric mean), which is the right central tendency when the
    # noise is MULTIPLICATIVE — and Stage 7 measured r(sigma, f) = +0.96..+0.99,
    # so it is. On the log scale an exponential relaxation toward a plateau
    # becomes a straight line, which is exactly what the linear detrender can
    # remove; on the raw scale it leaves curvature that inflates rho and is what
    # the drift warnings were flagging.
    if np.any(res['mean_fitness'] <= 0):
        raise SystemExit('analyze --transform log: non-positive fitness present')
    res['_v'] = np.log(res['mean_fitness']) if args.transform == 'log' else res['mean_fitness']

    raw, walk_ids = _walk_step_table(res, '_v')       # [W, T] pre-detrend
    raw_f, _ = _walk_step_table(res, 'mean_fitness')  # untransformed, for reporting
    W = raw.shape[0]
    T = raw.shape[1]
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
    spans = np.abs(np.array(slopes)) * T
    rng_span = float(raw.max() - raw.min()) or 1.0
    drifty = [int(walk_ids[i]) for i in range(W) if spans[i] > 0.5 * rng_span]
    # residual curvature AFTER detrending: the quantity the log transform is meant
    # to kill. Quadratic coefficient of the detrended residual, scaled by range.
    curv = []
    for i in range(W):
        c2 = np.polyfit(np.arange(T), proc[i], 2)[0]
        curv.append(abs(c2) * T ** 2 / rng_span)
    curv_mean = float(np.mean(curv))

    print(f'\n=== Stage 6 lambda (start={meta["start_label"]}, rl={meta["rl"]}, '
          f'W={W} walks x {T-1} steps, R={meta["repeats"]}, detrend={detrend}, '
          f'transform={args.transform}) ===')
    print(f'  fitness (raw f, all walks): [{raw_f.min():.3f}, {raw_f.max():.3f}]  '
          f'mean {raw_f.mean():.3f}')
    if args.transform == 'log':
        print(f'  analysed on log f: [{raw.min():.3f}, {raw.max():.3f}]  mean {raw.mean():.3f}')
    print(f'  residual curvature after detrend = {curv_mean:.3f} '
          f'(fraction of range; lower = the linear detrend fits better)')
    print(f'  rho(1) = {rho1_mean:.4f} +/- {rho1_se:.4f}  (mean +/- SE across walks)')
    print(f'  lambda_fit  (LS, k=1..{kfit}) = {lam_fit_mean:.3f} +/- {lam_fit_se:.3f}   <- headline')
    print(f'  lambda_rho1 (legacy)          = {lam_rho1:.3f}')
    print(f'  lambda from averaged curve    = {lam_curve:.3f}')
    if drifty:
        print(f'  WARNING: walks with strong residual drift (>0.5*range): {drifty}')
    print('  (small lambda = rugged; large lambda = smooth)')

    summary = {'start_label': meta['start_label'], 'rl': meta['rl'],
               'walks': int(W), 'steps': int(T - 1), 'repeats': meta['repeats'],
               'detrend': detrend, 'transform': args.transform, 'kfit': kfit,
               'rho1': rho1_mean, 'rho1_se': rho1_se,
               'lambda': lam_fit_mean, 'lambda_se': lam_fit_se,      # headline = LS fit
               'lambda_rho1': lam_rho1, 'lambda_curve_fit': lam_curve,
               'fit_min': float(raw_f.min()), 'fit_max': float(raw_f.max()),
               'fit_mean': float(raw_f.mean()), 'drifty_walks': drifty,
               'residual_curvature': curv_mean,
               'config_overrides': meta.get('config_overrides')}
    # keep lambda_summary.json meaning what it has always meant (raw scale) so
    # stage7_plots.py --lambda-json is unaffected; log lands in its own file.
    name = 'lambda_summary.json' if args.transform == 'none' else 'lambda_summary_log.json'
    json.dump(summary, open(os.path.join(out_dir, name), 'w'), indent=2)

    # ── plots ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    steps_x = np.arange(T)
    for i in range(W):
        ax1.plot(steps_x, raw[i], '-', lw=0.8, alpha=0.7)
    ax1.plot(steps_x, raw.mean(axis=0), '-', color='k', lw=1.6, label='walk mean')
    ax1.set_xlabel('mutation step')
    ax1.set_ylabel('mean log f' if args.transform == 'log' else 'R-avg fitness')
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
    pname = 'lambda.png' if args.transform == 'none' else 'lambda_log.png'
    fig.savefig(os.path.join(out_dir, pname), dpi=130)
    plt.close(fig)
    print(f'[stage6] plot -> {out_dir}/{pname}')
    return summary


def cmd_pairjobs(args):
    """Jobs for the STATIONARY-ENSEMBLE pair estimator (needs new probes).

    The walk-derived version in cmd_pairs is a proxy: it reuses consecutive walk
    steps, but a walk that starts at the evolved centroid is relaxing the whole
    way, so no step window of it is stationary (measured drift 0.23-0.61 at every
    window). This draws the ensemble the estimator actually wants:

        parent_i = theta* mutated `burn` times, independently per i
        mutant_i = parent_i mutated ONCE

    Every parent is an independent draw from the same ensemble at a fixed
    mutational distance from theta*, so the ensemble is stationary BY
    CONSTRUCTION — no drift, no detrending, no exponential assumption — and
    rho(1) = corr(f_parent, f_mutant) across pairs is unbiased apart from the
    measurement attenuation, which `pairs` corrects from the R replicates.

    Cost: 2 * n_pairs * R probes (default 300 pairs, R=8 -> 4800, ~2 core-hours).
    """
    os.makedirs(args.out_dir, exist_ok=True)
    theta0, start_label = _start_genome(args.genome_csv, args.start)
    child = np.random.SeedSequence(args.pair_seed).spawn(args.n_pairs)

    genomes, jobs = {}, []
    R, rl = args.repeats, bool(args.rl)
    for i in range(args.n_pairs):
        rng = np.random.default_rng(child[i])
        p = theta0
        for _ in range(args.burn):
            p = mutate(p, rng)
        q = mutate(p, rng)
        genomes[f'p{i}'] = p
        genomes[f'm{i}'] = q
        for r in range(R):
            jobs.append({'id': f'p{i}_r{r}', 'genome': f'p{i}',
                         'map_index': args.map_start + r, 'rl': rl})
            jobs.append({'id': f'm{i}_r{r}', 'genome': f'm{i}',
                         'map_index': args.map_start + r, 'rl': rl})

    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    ll.write_jobs(os.path.join(args.out_dir, 'jobs.json'), genomes, jobs,
                  config_overrides=cfg)
    json.dump({'kind': 'pairs', 'start': args.start, 'start_label': start_label,
               'n_pairs': args.n_pairs, 'burn': args.burn, 'repeats': R, 'rl': rl,
               'map_start': args.map_start, 'pair_seed': args.pair_seed,
               'config_overrides': cfg,
               'mut_prob': MUT_PROB, 'mut_sigma': MUT_SIGMA},
              open(os.path.join(args.out_dir, 'meta.json'), 'w'), indent=2)
    print(f'[stage6] pair ensemble: {args.n_pairs} parents at mutational distance '
          f'burn={args.burn} from {start_label}, each with 1 mutant')
    print(f'[stage6] wrote {len(jobs)} jobs ({args.n_pairs} pairs x 2 x R={R}) '
          f'-> {args.out_dir}/jobs.json')


def _pairs_direct(res, meta, out_dir, use_log, transform_name):
    """rho(1) for a stationary pair ensemble (meta['kind'] == 'pairs')."""
    idx = res['id'].str.extract(r'^([pm])(\d+)_r(\d+)$')
    res = res.assign(side=idx[0], pair=idx[1].astype(float), rep=idx[2].astype(float))
    res = res.dropna(subset=['side', 'pair'])
    res['pair'] = res['pair'].astype(int)

    panel = {}
    for side in ('p', 'm'):
        sub = res[res['side'] == side]
        pv = sub.pivot_table(index='pair', columns='rep', values='_v').sort_index()
        panel[side] = pv
    common = panel['p'].index.intersection(panel['m'].index)
    P = panel['p'].loc[common].to_numpy()
    Q = panel['m'].loc[common].to_numpy()
    p = P.mean(axis=1); q = Q.mean(axis=1)
    n, R = P.shape

    r_obs = float(np.corrcoef(p, q)[0, 1])
    # noise on an R-map mean, from the pair x map interaction (both sides pooled)
    v_n = []
    for A in (P, Q):
        resid = (A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True)
                 + A.mean())
        dof = max((A.shape[0] - 1) * (A.shape[1] - 1), 1)
        v_n.append(float(np.sum(resid ** 2) / dof) / A.shape[1])
    v_n = float(np.mean(v_n))
    v_tot = float((np.var(p, ddof=1) + np.var(q, ddof=1)) / 2.0)
    frac = v_n / v_tot if v_tot > 0 else np.nan
    r_dis = min(r_obs / (1.0 - frac), 0.9999) if np.isfinite(frac) and frac < 0.95 else np.nan
    # bootstrap SE over pairs
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, n, n)
        if np.std(p[k]) > 0 and np.std(q[k]) > 0:
            boot.append(float(np.corrcoef(p[k], q[k])[0, 1]))
    se_obs = float(np.std(boot, ddof=1)) if len(boot) > 2 else float('nan')
    se_dis = se_obs / (1.0 - frac) if np.isfinite(frac) and frac < 0.95 else float('nan')

    print(f'\n=== Stage 6 STATIONARY PAIR rho(1)  ({meta["start_label"]}, '
          f'rl={meta["rl"]}, n={n} pairs, burn={meta.get("burn")}, R={R}, '
          f'{transform_name}) ===')
    print('  Stationary by construction: independent parents from one ensemble,')
    print('  one mutation to each mutant. No drift, no detrending, no exponential fit.')
    print(f'  noise fraction              = {frac:.3f}')
    print(f'  rho(1) observed             = {r_obs:.4f} +/- {se_obs:.4f}  (bootstrap)')
    print(f'  rho(1) noise-corrected      = {r_dis:.4f} +/- {se_dis:.4f}   <- headline')
    print(f'  lambda = -1/ln rho          = {_lam_from_rho1(r_dis):.2f} mutations')
    if np.isfinite(frac) and frac > 0.3:
        print(f'\n  ** NOISE FRACTION {frac:.2f} IS TOO HIGH — DO NOT QUOTE THE CORRECTED '
              f'VALUE. **')
        print(f'     Real fitness differences between parents are smaller than the')
        print(f'     measurement noise, so dividing by (1-{frac:.2f}) amplifies noise, not')
        print(f'     signal. The ensemble radius is too small and/or R too low. Fixes,')
        print(f'     cheapest first:')
        print(f'       - raise --burn (spread the parents further from theta*): noise')
        print(f'         fraction falls as the genuine spread grows. The Stage-6 walk')
        print(f'         windows suggest burn ~30-60 lands near 0.1-0.25 at R=8.')
        print(f'       - raise --repeats (noise on an R-mean scales as 1/R).')
        print(f'     Target < 0.3, ideally < 0.15. Only the OBSERVED rho is safe to')
        print(f'     report until then, and it is a LOWER bound on the true rho(1).')
    summary = {'estimator': 'stationary parent-mutant ensemble',
               'start_label': meta['start_label'], 'rl': meta['rl'],
               'n_pairs': int(n), 'burn': meta.get('burn'), 'repeats': int(R),
               'transform': transform_name, 'noise_frac': frac,
               'rho1_observed': r_obs, 'rho1_observed_se': se_obs,
               'rho1': r_dis, 'rho1_se': se_dis,
               'lambda': _lam_from_rho1(r_dis),
               'config_overrides': meta.get('config_overrides')}
    name = f'pair_rho_summary{"_log" if use_log else ""}.json'
    json.dump(summary, open(os.path.join(out_dir, name), 'w'), indent=2)
    print(f'\n[stage6] pairs -> {out_dir}/{name}')
    return summary


def cmd_pairs(args):
    """rho(1) as a PARENT-MUTANT CORRELATION over an approximately stationary
    ensemble, instead of as the autocorrelation of a drifting walk.

    No new probes are needed: a Stage-6 walk already IS a chain of genomes one
    mutation apart, so every consecutive step pair (g_t, g_{t+1}) is a
    parent-mutant pair and

        rho(1) = corr(f_parent, f_mutant)  across pairs

    needs no detrending, no exponential assumption, and no stationarity of the
    walk as a whole — only of the ENSEMBLE the pairs are drawn from. That is what
    the step window buys: near the start of a centroid walk the ensemble is
    "evolved centroid + a few mutations", i.e. the region of interest.

    TWO THINGS THAT WOULD OTHERWISE FAKE A HIGH rho:
      1. Drift ACROSS the window. If the ensemble is still relaxing, slow
         variation is shared by parent and mutant and appears as correlation.
         Reported as the window's drift, and by sweeping window width so the
         contamination is visible rather than assumed absent.
      2. Level differences ACROSS walks. Five walks that have relaxed to
         different levels contribute between-walk variance to a pooled
         correlation. `centered` removes each walk-window's mean first; the gap
         between pooled and centered is how much of rho was between-walk offset.

    Overlapping pairs share genomes (g_t is the mutant of t-1 and the parent of
    t+1). `--stride 2` makes every genome appear in exactly one pair, at the cost
    of half the sample; the two agree when the estimate is sound.
    """
    out_dir = args.out_dir
    res = ll.load_results(os.path.join(out_dir, 'results.csv')).copy()
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))

    if np.any(res['mean_fitness'] <= 0):
        raise SystemExit('pairs: non-positive fitness present')
    use_log = args.transform == 'log'
    res['_v'] = np.log(res['mean_fitness']) if use_log else res['mean_fitness']

    # a stationary pair ensemble (from `pairjobs`) needs no windowing at all
    if meta.get('kind') == 'pairs':
        return _pairs_direct(res, meta, out_dir, use_log,
                             'log f' if use_log else 'raw f')

    m = res['id'].str.extract(_ID_RE)
    if m[0].isna().all():
        leg = res['id'].str.extract(r's(\d+)_r(\d+)')
        res['walk'] = 0
        res['step'] = leg[0].astype(int)
    else:
        res['walk'] = m[0].astype(int)
        res['step'] = m[1].astype(int)

    V, walk_ids = _walk_step_table(res, '_v')
    Fr, _ = _walk_step_table(res, 'mean_fitness')
    W, T = V.shape

    # per-walk [step, map] panels — needed for the noise-variance estimate below
    panels = []
    for w in walk_ids:
        sub = res[res['walk'] == w]
        pv = sub.pivot_table(index='step', columns='map_index', values='_v')
        panels.append(pv.sort_index().to_numpy()[:T])

    if args.windows:
        windows = []
        for spec in args.windows.split(','):
            a_, b_ = spec.split(':')
            windows.append((int(a_), min(int(b_), T)))
    else:
        windows = [(0, 30), (0, 60), (0, 120), (0, 250), (0, T), (T // 2, T)]
    windows = [(a_, b_) for (a_, b_) in windows if b_ - a_ >= 6]

    rows = []
    for (t0, t1) in windows:
        seg = V[:, t0:t1]
        n_t = seg.shape[1]
        # drift across the window, as a fraction of the window's total spread
        span = float(seg.max() - seg.min()) or 1.0
        drift = float(np.mean([abs(np.polyfit(np.arange(n_t), seg[i], 1)[0]) * n_t
                               for i in range(W)]) / span)

        par_all, mut_all, par_c, mut_c, per_walk = [], [], [], [], []
        corr_walk, noise_frac = [], []
        for i in range(W):
            idx = np.arange(0, n_t - 1, args.stride)
            p = seg[i][idx]
            q = seg[i][idx + 1]
            par_all.append(p); mut_all.append(q)
            par_c.append(p - p.mean()); mut_c.append(q - q.mean())
            if p.size > 2:
                r_obs = float(np.corrcoef(p, q)[0, 1])
                per_walk.append(r_obs)

                # ── DISATTENUATION ──────────────────────────────────────────
                # Parent and mutant are different genomes, so their measurement
                # noise is independent and does NOT enter Cov(p,q) — but it does
                # inflate Var(p) and Var(q), pulling the correlation toward 0:
                #     rho_obs = rho_true * Vs / (Vs + Vn)
                # The shared map set contributes a per-map offset common to every
                # step, which is constant along the walk and so cannot affect a
                # correlation ACROSS steps. What remains is the step x map
                # interaction, estimated as the two-way ANOVA residual of the
                # [step, map] panel; the noise on an R-map mean is that over R.
                pan = panels[i][t0:t1]
                resid = (pan - pan.mean(axis=1, keepdims=True)
                         - pan.mean(axis=0, keepdims=True) + pan.mean())
                dof = max((pan.shape[0] - 1) * (pan.shape[1] - 1), 1)
                v_e = float(np.sum(resid ** 2) / dof)
                v_n = v_e / pan.shape[1]
                v_tot = float((np.var(p, ddof=1) + np.var(q, ddof=1)) / 2.0)
                frac = v_n / v_tot if v_tot > 0 else np.nan
                noise_frac.append(frac)
                if np.isfinite(frac) and frac < 0.95:
                    corr_walk.append(min(r_obs / (1.0 - frac), 0.9999))
                else:
                    corr_walk.append(np.nan)

        p_all = np.concatenate(par_all); q_all = np.concatenate(mut_all)
        p_cen = np.concatenate(par_c); q_cen = np.concatenate(mut_c)
        rho_pool = float(np.corrcoef(p_all, q_all)[0, 1])
        rho_cen = float(np.corrcoef(p_cen, q_cen)[0, 1])
        pw = np.array(per_walk, dtype=float)
        rho_wm = float(np.nanmean(pw)) if pw.size else float('nan')
        rho_se = float(np.nanstd(pw, ddof=1) / np.sqrt(pw.size)) if pw.size > 1 else float('nan')

        cw = np.array(corr_walk, dtype=float)
        rho_dis = float(np.nanmean(cw)) if cw.size else float('nan')
        rho_dis_se = (float(np.nanstd(cw, ddof=1) / np.sqrt(np.isfinite(cw).sum()))
                      if np.isfinite(cw).sum() > 1 else float('nan'))
        nf = float(np.nanmean(noise_frac)) if noise_frac else float('nan')

        rows.append({'t0': t0, 't1': t1, 'n_pairs': int(p_all.size),
                     'drift_frac': drift, 'noise_frac': nf,
                     'rho_pooled': rho_pool, 'rho_centered': rho_cen,
                     'rho_walkmean': rho_wm, 'rho_se': rho_se,
                     'rho_disattenuated': rho_dis, 'rho_disattenuated_se': rho_dis_se,
                     'lambda_centered': _lam_from_rho1(rho_cen),
                     'lambda_walkmean': _lam_from_rho1(rho_wm),
                     'lambda_disattenuated': _lam_from_rho1(rho_dis),
                     'f_mean': float(Fr[:, t0:t1].mean())})

    print(f'\n=== Stage 6 PAIR rho(1)  ({meta["start_label"]}, rl={meta["rl"]}, '
          f'W={W} walks, R={meta["repeats"]}, '
          f'{"log f" if use_log else "raw f"}, stride={args.stride}) ===')
    print('  corr(f_parent, f_mutant) over consecutive walk steps — no detrending')
    hdr = (f'{"window":>10}{"pairs":>7}{"drift":>7}{"noise":>7}{"rho_pool":>10}'
           f'{"rho_walk±SE":>16}{"rho_disatt±SE":>18}{"lam_dis":>9}')
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for r in rows:
        win = f'{r["t0"]}:{r["t1"]}'
        wm = f'{r["rho_walkmean"]:.4f}±{r["rho_se"]:.4f}'
        ds = f'{r["rho_disattenuated"]:.4f}±{r["rho_disattenuated_se"]:.4f}'
        print(f'{win:>10}{r["n_pairs"]:>7d}{r["drift_frac"]:>7.2f}'
              f'{r["noise_frac"]:>7.2f}{r["rho_pooled"]:>10.4f}{wm:>16}{ds:>18}'
              f'{r["lambda_disattenuated"]:>9.2f}')
    print('\n  drift = |linear trend| / window spread. A window at drift ~0 is')
    print('    stationary and needs no correction; large drift INFLATES rho upward.')
    print('  noise = measurement variance / total variance of the pair series.')
    print('    Large noise ATTENUATES rho downward. The two biases run in OPPOSITE')
    print('    directions, so a window is only trustworthy when BOTH are small.')
    print('  rho_walk = per-walk correlation, averaged (removes between-walk offset).')
    print('  rho_disatt = rho_walk / (1 - noise): the noise correction, from the')
    print('    step x map interaction of the R replicates. This is the estimate to use.')

    summary = {'start_label': meta['start_label'], 'rl': meta['rl'],
               'walks': int(W), 'repeats': meta['repeats'],
               'transform': args.transform, 'stride': args.stride,
               'estimator': 'parent-mutant correlation over consecutive walk steps',
               'windows': rows, 'config_overrides': meta.get('config_overrides')}
    name = f'pair_rho_summary{"_log" if use_log else ""}.json'
    json.dump(summary, open(os.path.join(out_dir, name), 'w'), indent=2)
    print(f'\n[stage6] pairs -> {out_dir}/{name}')
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('jobs', 'run', 'analyze', 'all', 'pairs', 'pairjobs'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', required=True)
        if name == 'pairjobs':
            p.add_argument('--genome-csv', required=True)
            p.add_argument('--start', default='centroid',
                           choices=['random', 'centroid', 'evolved', 'fittest'])
            p.add_argument('--n-pairs', type=int, default=300)
            p.add_argument('--burn', type=int, default=5,
                           help='mutations from theta* to each parent (ensemble radius)')
            p.add_argument('--repeats', type=int, default=8)
            p.add_argument('--map-start', type=int, default=0)
            p.add_argument('--rl', action='store_true')
            p.add_argument('--config-overrides', default=None)
            p.add_argument('--pair-seed', type=int, default=6001)
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
        if name in ('analyze', 'all', 'pairs'):
            p.add_argument('--transform', default='none', choices=['none', 'log'],
                           help="'log' analyses mean log f (geometric mean): turns "
                                "exponential relaxation into a line the linear "
                                "detrender can remove, and stabilises multiplicative "
                                "noise")
        if name == 'pairs':
            p.add_argument('--windows', default=None,
                           help='comma list of t0:t1 step windows, e.g. 0:60,0:250')
            p.add_argument('--stride', type=int, default=1,
                           help='2 = non-overlapping pairs (each genome used once)')
    args = ap.parse_args()
    if args.cmd == 'jobs':
        cmd_jobs(args)
    elif args.cmd == 'run':
        cmd_run(args)
    elif args.cmd == 'analyze':
        cmd_analyze(args)
    elif args.cmd == 'pairs':
        cmd_pairs(args)
    elif args.cmd == 'pairjobs':
        cmd_pairjobs(args)
    elif args.cmd == 'all':
        cmd_jobs(args); cmd_run(args); cmd_analyze(args)


if __name__ == '__main__':
    main()
