"""
landscape/stage8_nk.py
======================================================================
Stage 8 — build an NK model FROM the measured landscape, with a tunable K.

Two modes, in increasing strength of claim:

  `quick`   the legacy characterisation: map the Stage-6 random-walk rho(1) to
            Kauffman's K through the isotropic relation rho(d)=(1-(K+1)/N)^d.
            This is a RE-EXPRESSION of lambda, not an independent measurement —
            it carries no information lambda did not already carry, and its
            error bars are correspondingly useless (K_eff = 0.09..0.90 ± 0.3
            across the four seed-999 conditions). Kept for comparison only.

  `jobs` -> `analyze` -> `model` -> `sim`
            the real thing: DISCRETISE the genome into loci, MEASURE the pairwise
            epistasis between every pair of loci with the simulator, and build a
            second-order NK model whose wiring and coefficient magnitudes come
            from that measurement. K is then a property you measured, and a dial
            you can turn.

THE DISCRETISATION (the hard part)
----------------------------------
An NK model needs discrete loci; the genome is 3910 continuous weights. The
resolution is already sitting in ll_common:

  1. LOCI = the 70 filter blocks. `ll.FILTER_BLOCKS` partitions the genome into
     one block per neuron (hidden unit j: its 54 incoming W1 weights + b1[j];
     output unit k: its 64 incoming W2 weights + b2[k]). These are the same units
     Li et al. filter normalisation already treats as atomic in Stages 3/4, and a
     neuron is the natural functional gene of a network. So M = 64 + 6 = 70 loci,
     a size where K in [0, 69] is genuinely tunable.

  2. ALLELES = a fixed filter-normalised nudge, NOT a binning of weight values.
     Binning weights would put the bin edges in the driver's seat and would make
     epsilon incomparable across loci of different weight scale. Instead, for one
     background genome theta0 and one random direction delta, filter-normalised
     to theta0 so that ||delta[block_i]|| = ALPHA * ||theta0[block_i]||:

         allele 0 at locus i  =  theta0[block_i]
         allele 1 at locus i  =  theta0[block_i] + delta[block_i]

     Every locus is displaced by the same RELATIVE amount, so the resulting
     epsilons are on one scale and counting them means something. A genome is
     then a bit string x in {0,1}^70 and f(x) is measurable by the existing probe.

     Note on clipping: theta is clipped to [-1,1] to respect the GA's constraint.
     Because the blocks PARTITION the genome, each coordinate belongs to exactly
     one block, so clip() is separable across loci and introduces NO spurious
     epistasis. That is a property of using the partition; it would not hold for
     overlapping locus definitions.

WHAT IS MEASURED
----------------
With the background, the M singles and all C(M,2) doubles:

    a_i      = f(x_i=1) - f(bg)                                  main effect
    eps_ij   = f(11) - f(10) - f(01) + f(00)                      pairwise epistasis

The corners are SHARED between pairs, so the full 70x70 matrix costs
1 + 70 + 2415 = 2486 genomes, not 4 per pair — 19,888 probes at R=8, the size of
one Stage-6 condition.

Every corner is evaluated on the SAME fixed map set and epsilon is formed
PER MAP before averaging. epsilon is a pure contrast, so terrain is common-mode
and cancels: its noise is governed by Stage 2's sigma_fixed (0.1-1.0), not
sigma_vary (6-8). This pairing is why R=8 is enough for a 4-term contrast.

    K_i = #{ j : |eps_ij| > z * SE(eps_ij) },   K = mean_i K_i

with the detection threshold cross-checked against an EMPIRICAL null: a subset of
pairs re-measured on a disjoint map set, whose replicate differences give SE
directly instead of by propagation.

  ** MEASURED K IS A LOWER BOUND. ** A ground-truth recovery test (a known
  second-order model pushed through this pipeline with sigma_fixed=0.15, R=8,
  terrain sigma=5.6) recovered eps at corr(true,estimated)=0.988 and found 21 of
  28 real edges with ZERO false positives at z=2 — so K came out 3.0 against a
  true 4.0. Edges with |eps| under z*SE are invisible, and they are missing from
  K, never added to it. Quote K as "K >= x" and pair it with the Walsh epistasis
  fraction, which is threshold-free (it uses every eps, debiased) and in the same
  test read 24.6% against a true 28.4%.

Also reported: the Walsh (Fourier) variance decomposition — what fraction of the
fitness variance over the hypercube lives in additive (order-1) vs pairwise
(order-2) terms. In +/-1 coordinates s_i = 2x_i - 1,

    w_i  = a_i/2 + (1/4) sum_{j!=i} eps_ij      w_ij = eps_ij / 4
    V1 = sum w_i^2      V2 = sum_{i<j} w_ij^2   epistasis fraction = V2/(V1+V2)

both debiased for measurement noise (E[sum of squared estimates] overshoots by
the sum of the variances).

Run it with rl=False AND rl=True and you can ask whether learning FLATTENS
epistasis — a claim lambda cannot make.

THE MODEL, AND WHAT IT CAN AND CANNOT SCORE
-------------------------------------------
`model` writes a second-order NK model (an NK landscape with pairwise epistasis;
equivalently an Ising/Hopfield-form landscape):

    f(x) = f0 + sum_i a_i x_i + sum_{(i,j) in E} eps_ij x_i x_j

Order 1 and 2 are exactly what the measurement gives, so this IS the measured
model rather than a guess, extended to all 2^70 strings. K = degree in E.

  * `--wiring measured` keeps the measured edges (locus i's neighbours = its
    significant partners). Real wiring, calibrated magnitudes.
  * `--wiring random --K k` rewires at degree k while resampling coefficients
    from the MEASURED distributions. This is the dial: same ruggedness scale,
    chosen ruggedness amount.

Because the form is multilinear, it also extends to CONTINUOUS x in [0,1]^70 —
the same formula. So the discrete vertices give the NK landscape with tunable K,
while the interior lets a surrogate step be as PARTIAL as a real mutation is.
That matters: a Stage-6 GA step nudges each weight with prob 0.03, so it touches
1-(1-0.03)^|block| of each block — about 82% of all 70 loci per step, but only
PARTIALLY. Treating that as a full allele flip is what makes `quick` a lower
bound on K.

  ** Scoring an arbitrary real genome on this model is only meaningful near the
  hypercube. ** `project` maps a real theta onto x by least squares along each
  locus's allele axis and REPORTS the residual fraction. For an independently
  evolved genome that residual will be large — the 70-D allele subspace is a thin
  slice of 3910-D — and the projected score should not be trusted. The honest use
  of the surrogate is to run adaptive dynamics IN it at several K and compare
  statistics (adaptation rate, local-optima density, autocorrelation) against the
  real system, not to rank real genomes.

Usage
-----
  # 1) jobs (LOCAL) — one background, all pairs, both RL passes
  python landscape/stage8_nk.py jobs \
      --genome-csv <run/genome.csv> --background centroid --alpha 0.25 \
      --repeats 8 --null-pairs 100 \
      --out-dir landscape/out/stage8_hard_centroid \
      --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'

  # 2) probes (CLUSTER) — generic sharded runner, same as every other stage
  OUT_DIR=landscape/out/stage8_hard_centroid PARAMS=<params.json> \
      N_SHARDS=64 sbatch --array=0-63 run_landscape_probe_array.slurm

  # 3) merge + analyse (LOCAL)
  python landscape/merge_shards.py   --out-dir landscape/out/stage8_hard_centroid
  python landscape/stage8_nk.py analyze --out-dir landscape/out/stage8_hard_centroid --z 2

  # 4) build the tunable model, then run dynamics in it
  python landscape/stage8_nk.py model --out-dir landscape/out/stage8_hard_centroid
  python landscape/stage8_nk.py sim   --out-dir landscape/out/stage8_hard_centroid \
      --k-sweep 0,1,2,4,8,16,32
  python landscape/stage8_nk.py project --out-dir landscape/out/stage8_hard_centroid \
      --genome-csv <other run/genome.csv> --record centroid

  # legacy characterisation (unchanged; bare --out-dir still works)
  python landscape/stage8_nk.py quick --out-dir landscape/out/stage6_*
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ll_common as ll

MUT_PROB = 0.03    # GAManager MUT_PROB (between-generation)
MUT_SIGMA = 0.1    # GAManager MUT_SIGMA
M_LOCI = len(ll.FILTER_BLOCKS)          # 70

# series colours (fixed order, never cycled)
C_MEAS = '#3b6fb6'      # measured
C_NULL = '#8a8f98'      # null / noise
C_ALT = '#c2622d'       # second condition (RL on)

_JOB_RE = re.compile(r'^(?P<kind>[a-z]+)(?P<idx>[\d_]*)_r(?P<r>\d+)_(?P<tag>on|off)$')


# ══════════════════════════════════════════════════════════════════════════════
# quick — legacy rho(1) -> K_eff (a re-expression of lambda; kept for comparison)
# ══════════════════════════════════════════════════════════════════════════════
def _k_eff(rho1, N, d_bar):
    """NK effective K from the isotropic relation rho(d_bar)=(1-(K+1)/N)^d_bar."""
    if not (0 < rho1 < 1):
        return float('nan')
    return N * (1.0 - rho1 ** (1.0 / d_bar)) - 1.0


def analyze_dir(out_dir):
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    summ = json.load(open(os.path.join(out_dir, 'lambda_summary.json')))
    N = ll.GENOME_SIZE
    p = float(meta.get('mut_prob', MUT_PROB))
    d_bar = N * p                                   # expected loci moved per step

    rho1 = float(summ['rho1'])
    rho1_se = float(summ.get('rho1_se') or float('nan'))
    lam_step = (-1.0 / np.log(rho1)) if 0 < rho1 < 1 else float('inf')

    k_eff = _k_eff(rho1, N, d_bar)
    k_naive = (N * (1.0 - rho1) - 1.0) if 0 < rho1 < 1 else float('nan')
    lam_locus = d_bar * lam_step if 0 < rho1 < 1 else float('nan')
    # propagate cross-walk SE(rho1) -> SE(K_eff) by +/- one SE on rho1
    if np.isfinite(rho1_se) and rho1_se > 0:
        k_hi = _k_eff(min(rho1 - rho1_se, 0.999999), N, d_bar)   # lower rho -> higher K
        k_lo = _k_eff(min(rho1 + rho1_se, 0.999999), N, d_bar)
        k_se = abs(k_hi - k_lo) / 2.0
    else:
        k_se = float('nan')

    return {
        'out_dir': out_dir,
        'start_label': meta.get('start_label'),
        'rl': meta.get('rl'),
        'N': N, 'mut_prob': p, 'd_bar': d_bar,
        'rho1': rho1, 'rho1_se': rho1_se,
        'lambda_step': lam_step,
        'lambda_locus': lam_locus,
        'K_eff': k_eff, 'K_eff_se': k_se,
        'K_naive_per_step': k_naive,
        'config_overrides': meta.get('config_overrides'),
    }


def cmd_quick(args):
    dirs = []
    for d in args.out_dir:
        dirs.extend(sorted(glob.glob(d)) if any(c in d for c in '*?[') else [d])
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'lambda_summary.json'))]
    if not dirs:
        raise SystemExit('quick: no dirs with a lambda_summary.json')

    rows = []
    for d in dirs:
        r = analyze_dir(d)
        rows.append(r)
        json.dump(r, open(os.path.join(d, 'nk_quick_summary.json'), 'w'), indent=2)

    hdr = f'{"walk":<34}{"rho(1)":>16}{"lam_step":>10}{"K_eff":>16}'
    print(f'\n=== Stage 8 quick - effective NK K  (N={ll.GENOME_SIZE}, d_bar=N*p) ===')
    print(hdr); print('-' * len(hdr))
    for r in rows:
        name = os.path.basename(os.path.normpath(r['out_dir']))
        rho_s = (f'{r["rho1"]:.4f}±{r["rho1_se"]:.4f}' if np.isfinite(r['rho1_se'])
                 else f'{r["rho1"]:.4f}')
        k_s = (f'{r["K_eff"]:.2f}±{r["K_eff_se"]:.2f}' if np.isfinite(r['K_eff_se'])
               else f'{r["K_eff"]:.2f}')
        print(f'{name:<34}{rho_s:>16}{r["lambda_step"]:>10.2f}{k_s:>16}')
    print('\n(K_eff = N*(1-rho1^(1/d_bar)) - 1, multi-locus corrected; '
          'small K = smooth, large K = rugged/epistatic)')
    print('rho(1) is walk-averaged + detrended (Stage 6); ± = cross-walk SE.')
    print('caveat: continuous nudge != full spin flip -> K_eff is a LOWER BOUND, and')
    print('        this number is a re-expression of lambda. Use `jobs`/`analyze` for')
    print('        an independently MEASURED K.')


# ══════════════════════════════════════════════════════════════════════════════
# hypercube construction
# ══════════════════════════════════════════════════════════════════════════════
def block_touch_prob():
    """P(a real GA step perturbs >=1 weight in a block), per block and averaged.

    A Stage-6 step is Bernoulli(MUT_PROB) per weight, so a block of size b is
    touched with prob 1-(1-p)^b. Reported as a diagnostic: it is the reason the
    `quick` estimate is a lower bound (a real step touches most loci, but only
    PARTIALLY, whereas the isotropic relation assumes full re-randomisation).
    """
    sizes = np.array([blk.size for blk in ll.FILTER_BLOCKS], dtype=float)
    per = 1.0 - (1.0 - MUT_PROB) ** sizes
    return per, float(per.mean())


def _background_genome(genome_csv, which):
    """theta0 for the hypercube. 'random' matches Stage 6's random start exactly."""
    if which == 'random':
        # Same fresh-brain draw as Stage 6 (same seed, same helper), so the two
        # stages keep sharing a background genome.
        return (ll.xavier_genome(seed=20260718,
                                 hidden_size=ll.hidden_size_from_genome_csv(genome_csv)),
                'random')
    if which in ('centroid', 'evolved'):
        gens, mat = ll.centroid_trajectory(genome_csv)
        return mat[-1].copy(), f'centroid_g{int(gens[-1])}'
    if which == 'fittest':
        df = ll.load_genome_csv(genome_csv, record_type='fittest').sort_values('generation')
        return df.iloc[-1]['genome'].copy(), f'fittest_g{int(df.iloc[-1]["generation"])}'
    raise ValueError(f'unknown --background {which}')


def build_allele_axis(theta0, alpha, seed):
    """delta such that ||delta[block_i]|| = alpha * ||theta0[block_i]|| for every i.

    ONE direction for the whole experiment, so "allele 1 at locus i" is a fixed,
    well-defined genome and the hypercube is consistent across all pairs.
    """
    rng = np.random.default_rng(seed)
    d = ll.filter_normalize(rng.standard_normal(ll.GENOME_SIZE), theta0)
    return alpha * d


def hypercube_genome(theta0, delta, loci):
    """theta for the bit string with allele 1 at each locus in `loci`.

    Clipping is separable across loci because FILTER_BLOCKS partitions the
    genome, so it cannot manufacture epistasis (see module docstring).
    """
    g = theta0.copy()
    for i in loci:
        blk = ll.FILTER_BLOCKS[i]
        g[blk] = g[blk] + delta[blk]
    return np.clip(g, -1.0, 1.0)


def save_hypercube(path, theta0, delta, meta):
    json.dump({'theta0_b64': ll.encode_b64(theta0),
               'delta_b64': ll.encode_b64(delta), **meta},
              open(path, 'w'))


def load_hypercube(path):
    h = json.load(open(path))
    return ll.decode_b64(h['theta0_b64']), ll.decode_b64(h['delta_b64']), h


# ══════════════════════════════════════════════════════════════════════════════
# jobs
# ══════════════════════════════════════════════════════════════════════════════
def cmd_jobs(args):
    os.makedirs(args.out_dir, exist_ok=True)
    theta0, bg_label = _background_genome(args.genome_csv, args.background)
    delta = build_allele_axis(theta0, args.alpha, args.allele_seed)
    cfg = json.loads(args.config_overrides) if args.config_overrides else None
    rng = np.random.default_rng(args.seed)

    loci = list(range(M_LOCI)) if args.loci is None else \
        sorted({int(t) for t in args.loci.split(',')})
    all_pairs = list(itertools.combinations(loci, 2))
    if args.pairs is not None and args.pairs < len(all_pairs):
        sel = rng.choice(len(all_pairs), size=args.pairs, replace=False)
        pairs = [all_pairs[k] for k in sorted(sel)]
    else:
        pairs = all_pairs

    rl_passes = (False, True) if args.rl_passes == 'both' else \
        ((True,) if args.rl_passes == 'on' else (False,))
    R = args.repeats
    maps = [args.map_start + r for r in range(R)]

    # ── the shared corners: background, singles, doubles ──────────────────────
    genomes = {'bg': hypercube_genome(theta0, delta, [])}
    for i in loci:
        genomes[f's{i}'] = hypercube_genome(theta0, delta, [i])
    for (i, j) in pairs:
        genomes[f'd{i}_{j}'] = hypercube_genome(theta0, delta, [i, j])

    jobs = []
    for r, m in enumerate(maps):
        for rl in rl_passes:
            tag = 'on' if rl else 'off'
            jobs.append({'id': f'bg_r{r}_{tag}', 'genome': 'bg',
                         'map_index': int(m), 'rl': bool(rl)})
            for i in loci:
                jobs.append({'id': f's{i}_r{r}_{tag}', 'genome': f's{i}',
                             'map_index': int(m), 'rl': bool(rl)})
            for (i, j) in pairs:
                jobs.append({'id': f'd{i}_{j}_r{r}_{tag}', 'genome': f'd{i}_{j}',
                             'map_index': int(m), 'rl': bool(rl)})

    # ── empirical null: re-measure a subset on a DISJOINT map set ─────────────
    # Replicate differences give SE(eps) directly, instead of by propagation.
    null_pairs = []
    if args.null_pairs > 0 and pairs:
        n_sel = min(args.null_pairs, len(pairs))
        idx = rng.choice(len(pairs), size=n_sel, replace=False)
        null_pairs = [pairs[k] for k in sorted(idx)]
        null_loci = sorted({i for p in null_pairs for i in p})
        null_maps = [args.map_start + R + r for r in range(R)]
        for r, m in enumerate(null_maps):
            for rl in rl_passes:
                tag = 'on' if rl else 'off'
                jobs.append({'id': f'nbg_r{r}_{tag}', 'genome': 'bg',
                             'map_index': int(m), 'rl': bool(rl)})
                for i in null_loci:
                    jobs.append({'id': f'ns{i}_r{r}_{tag}', 'genome': f's{i}',
                                 'map_index': int(m), 'rl': bool(rl)})
                for (i, j) in null_pairs:
                    jobs.append({'id': f'nd{i}_{j}_r{r}_{tag}', 'genome': f'd{i}_{j}',
                                 'map_index': int(m), 'rl': bool(rl)})

    ll.write_jobs(os.path.join(args.out_dir, 'jobs.json'), genomes, jobs, cfg)

    per_block, mean_touch = block_touch_prob()
    meta = {
        'stage': 8, 'M': M_LOCI, 'loci': loci,
        'n_pairs': len(pairs), 'n_pairs_total': len(all_pairs),
        'pairs_sampled': args.pairs is not None and args.pairs < len(all_pairs),
        'background': args.background, 'background_label': bg_label,
        'alpha': args.alpha, 'allele_seed': args.allele_seed,
        'repeats': R, 'map_start': args.map_start, 'maps': maps,
        'rl_passes': [bool(x) for x in rl_passes],
        'null_pairs': [[int(i), int(j)] for (i, j) in null_pairs],
        'config_overrides': cfg,
        'mut_prob': MUT_PROB, 'mut_sigma': MUT_SIGMA,
        'block_touch_prob_mean': mean_touch,
        'theta0_norm': float(np.linalg.norm(theta0)),
    }
    json.dump(meta, open(os.path.join(args.out_dir, 'meta.json'), 'w'), indent=2)
    save_hypercube(os.path.join(args.out_dir, 'hypercube.json'), theta0, delta,
                   {'M': M_LOCI, 'loci': loci, 'alpha': args.alpha,
                    'allele_seed': args.allele_seed,
                    'background_label': bg_label,
                    'pairs': [[int(i), int(j)] for (i, j) in pairs]})

    print(f'[stage8] background={bg_label} ||theta0||={np.linalg.norm(theta0):.3f} '
          f'alpha={args.alpha}')
    print(f'[stage8] M={len(loci)} loci, {len(pairs)}/{len(all_pairs)} pairs, '
          f'{len(genomes)} distinct genomes')
    print(f'[stage8] {len(jobs)} probes '
          f'(R={R} x {len(rl_passes)} RL pass(es), null pairs={len(null_pairs)}) '
          f'-> {args.out_dir}/jobs.json')
    print(f'[stage8] diagnostic: a real GA step touches {mean_touch*100:.1f}% of the '
          f'70 loci per step (partially) — why `quick` K is a lower bound')


# ══════════════════════════════════════════════════════════════════════════════
# analyze
# ══════════════════════════════════════════════════════════════════════════════
def _parse_results(res, R):
    """results.csv -> {tag: {'bg': (R,), 'sing': {i: (R,)}, 'dbl': {(i,j): (R,)},
                            'nbg':…, 'nsing':…, 'ndbl':…}} with NaN for missing."""
    out = {}
    for row_id, rl, fit in zip(res['id'], res['rl'], res['mean_fitness']):
        m = _JOB_RE.match(str(row_id))
        if not m:
            continue
        kind, idx, r, tag = m.group('kind'), m.group('idx'), int(m.group('r')), m.group('tag')
        if r >= R:
            continue
        d = out.setdefault(tag, {'bg': np.full(R, np.nan), 'nbg': np.full(R, np.nan),
                                 'sing': {}, 'dbl': {}, 'nsing': {}, 'ndbl': {}})
        v = float(fit)
        if kind == 'bg':
            d['bg'][r] = v
        elif kind == 'nbg':
            d['nbg'][r] = v
        elif kind in ('s', 'ns'):
            key = 'sing' if kind == 's' else 'nsing'
            i = int(idx)
            d[key].setdefault(i, np.full(R, np.nan))[r] = v
        elif kind in ('d', 'nd'):
            key = 'dbl' if kind == 'd' else 'ndbl'
            i, j = (int(t) for t in idx.split('_'))
            d[key].setdefault((i, j), np.full(R, np.nan))[r] = v
    return out


def _contrasts(bg, sing, dbl, loci, M):
    """Per-map contrasts -> (a, a_se, eps, eps_se). Pairing is the whole point:
    eps is formed PER MAP, so terrain cancels before averaging."""
    a = np.full(M, np.nan); a_se = np.full(M, np.nan)
    for i, series in sing.items():
        v = series - bg
        v = v[np.isfinite(v)]
        if v.size:
            a[i] = v.mean()
            a_se[i] = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan

    eps = np.full((M, M), np.nan); eps_se = np.full((M, M), np.nan)
    for (i, j), series in dbl.items():
        si, sj = sing.get(i), sing.get(j)
        if si is None or sj is None:
            continue
        v = series - si - sj + bg
        v = v[np.isfinite(v)]
        if v.size:
            eps[i, j] = eps[j, i] = v.mean()
            if v.size > 1:
                eps_se[i, j] = eps_se[j, i] = v.std(ddof=1) / np.sqrt(v.size)
    return a, a_se, eps, eps_se


def _walsh(a, a_se, eps, eps_se):
    """Order-1 / order-2 variance shares in +/-1 coordinates, noise-debiased.

    s_i = 2x_i-1  =>  w_i = a_i/2 + (1/4) sum_{j!=i} eps_ij,  w_ij = eps_ij/4.
    For uniform s, Var(f) = sum w_i^2 + sum_{i<j} w_ij^2. Squared ESTIMATES are
    biased up by their own variance, so subtract it.
    """
    M = a.size
    e0 = np.where(np.isfinite(eps), eps, 0.0)
    np.fill_diagonal(e0, 0.0)
    se0 = np.where(np.isfinite(eps_se), eps_se, 0.0)
    np.fill_diagonal(se0, 0.0)

    w1 = np.where(np.isfinite(a), a, 0.0) / 2.0 + e0.sum(axis=1) / 4.0
    w1_var = (np.where(np.isfinite(a_se), a_se, 0.0) ** 2) / 4.0 + \
             (se0 ** 2).sum(axis=1) / 16.0
    iu = np.triu_indices(M, 1)
    w2 = e0[iu] / 4.0
    w2_var = (se0[iu] ** 2) / 16.0

    V1 = float(max(np.sum(w1 ** 2) - np.sum(w1_var), 0.0))
    V2 = float(max(np.sum(w2 ** 2) - np.sum(w2_var), 0.0))
    tot = V1 + V2
    return {'V1_additive': V1, 'V2_pairwise': V2,
            'epistasis_fraction': (V2 / tot) if tot > 0 else float('nan'),
            'V1_raw': float(np.sum(w1 ** 2)), 'V2_raw': float(np.sum(w2 ** 2)),
            'w1': w1, 'w2_upper': w2}


def _null_se(bg_n, sing_n, dbl_n, eps_main, null_pairs):
    """Empirical SE(eps) from replicate differences on a disjoint map set.

    eps measured twice independently -> Var(diff) = 2 Var(eps), so
    SE = std(diff)/sqrt(2). No propagation assumptions.
    """
    if not null_pairs or not np.isfinite(bg_n).any():
        return float('nan'), np.array([]), np.array([])
    a_n, _, eps_n, _ = _contrasts(bg_n, sing_n, dbl_n,
                                  sorted({i for p in null_pairs for i in p}), eps_main.shape[0])
    diffs, vals = [], []
    for (i, j) in null_pairs:
        if np.isfinite(eps_n[i, j]) and np.isfinite(eps_main[i, j]):
            diffs.append(eps_main[i, j] - eps_n[i, j])
            vals.append(eps_n[i, j])
    if len(diffs) < 2:
        return float('nan'), np.array(vals), np.array(diffs)
    d = np.array(diffs)
    return float(d.std(ddof=1) / np.sqrt(2.0)), np.array(vals), d


def cmd_analyze(args):
    out_dir = args.out_dir
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    res = ll.load_results(os.path.join(out_dir, 'results.csv'))
    M = int(meta['M']); R = int(meta['repeats'])
    loci = meta['loci']
    null_pairs = [tuple(p) for p in meta.get('null_pairs', [])]
    n_pairs = int(meta['n_pairs']); n_pairs_total = int(meta['n_pairs_total'])

    parsed = _parse_results(res, R)
    summary = {'out_dir': out_dir, 'stage': 8, 'M': M, 'repeats': R,
               'background_label': meta.get('background_label'),
               'alpha': meta.get('alpha'), 'z': args.z,
               'n_pairs': n_pairs, 'n_pairs_total': n_pairs_total,
               'config_overrides': meta.get('config_overrides'),
               'block_touch_prob_mean': meta.get('block_touch_prob_mean'),
               'conditions': {}}
    per_tag = {}

    for tag in sorted(parsed):
        d = parsed[tag]
        a, a_se, eps, eps_se = _contrasts(d['bg'], d['sing'], d['dbl'], loci, M)
        se_emp, null_vals, null_diffs = _null_se(d['nbg'], d['nsing'], d['ndbl'],
                                                eps, null_pairs)

        # detection threshold: paired SE per pair, floored by the empirical SE
        se_use = np.where(np.isfinite(eps_se), eps_se, np.nan)
        if np.isfinite(se_emp) and not args.no_null_floor:
            se_use = np.fmax(se_use, se_emp)
        sig = np.isfinite(eps) & np.isfinite(se_use) & (np.abs(eps) > args.z * se_use)
        np.fill_diagonal(sig, False)

        measured = np.isfinite(eps) & ~np.eye(M, dtype=bool)
        n_meas_per_locus = measured.sum(axis=1)
        K_i_raw = sig.sum(axis=1)
        # if pairs were subsampled, scale each locus's count to all M-1 partners
        with np.errstate(divide='ignore', invalid='ignore'):
            K_i = np.where(n_meas_per_locus > 0,
                           K_i_raw * (M - 1) / np.maximum(n_meas_per_locus, 1),
                           np.nan)

        walsh = _walsh(a, a_se, eps, eps_se)
        finite_eps = eps[measured]
        finite_eps = finite_eps[np.isfinite(finite_eps)]

        cond = {
            'rl': tag == 'on',
            'n_eps_measured': int(measured.sum() // 2),
            'K_mean': float(np.nanmean(K_i)), 'K_median': float(np.nanmedian(K_i)),
            'K_sd': float(np.nanstd(K_i, ddof=1)),
            'K_se': float(np.nanstd(K_i, ddof=1) / np.sqrt(np.isfinite(K_i).sum())),
            'K_max': float(np.nanmax(K_i)) if np.isfinite(K_i).any() else float('nan'),
            'frac_pairs_significant': float(sig.sum() / max(measured.sum(), 1)),
            'eps_abs_mean': float(np.mean(np.abs(finite_eps))) if finite_eps.size else float('nan'),
            'eps_abs_p95': float(np.percentile(np.abs(finite_eps), 95)) if finite_eps.size else float('nan'),
            'eps_se_median': float(np.nanmedian(eps_se)),
            'eps_se_empirical': se_emp,
            'a_mean': float(np.nanmean(a)), 'a_abs_mean': float(np.nanmean(np.abs(a))),
            'f_background': float(np.nanmean(d['bg'])),
            'V1_additive': walsh['V1_additive'], 'V2_pairwise': walsh['V2_pairwise'],
            'V1_raw': walsh['V1_raw'], 'V2_raw': walsh['V2_raw'],
            'epistasis_fraction': walsh['epistasis_fraction'],
            'debias_consumed_all': bool(walsh['V1_additive'] + walsh['V2_pairwise'] <= 0
                                        and walsh['V1_raw'] + walsh['V2_raw'] > 0),
            'K_extrapolation_factor': float((M - 1) /
                                            max(np.median(n_meas_per_locus), 1)),
        }
        summary['conditions'][tag] = cond
        per_tag[tag] = {'a': a, 'a_se': a_se, 'eps': eps, 'eps_se': eps_se,
                        'sig': sig, 'K_i': K_i, 'walsh': walsh,
                        'null_vals': null_vals, 'null_diffs': null_diffs,
                        'se_emp': se_emp}
        np.savez_compressed(os.path.join(out_dir, f'epistasis_{tag}.npz'),
                            a=a, a_se=a_se, eps=eps, eps_se=eps_se, K_i=K_i,
                            sig=sig, w1=walsh['w1'])

    # RL-off vs RL-on comparison
    if 'off' in per_tag and 'on' in per_tag:
        eo, en = per_tag['off']['eps'], per_tag['on']['eps']
        both = np.isfinite(eo) & np.isfinite(en) & ~np.eye(M, dtype=bool)
        if both.sum() > 2:
            summary['rl_comparison'] = {
                'K_off': summary['conditions']['off']['K_mean'],
                'K_on': summary['conditions']['on']['K_mean'],
                'dK': summary['conditions']['on']['K_mean'] - summary['conditions']['off']['K_mean'],
                'dK_se': float(np.hypot(summary['conditions']['off']['K_se'],
                                        summary['conditions']['on']['K_se'])),
                'eps_corr': float(np.corrcoef(eo[both], en[both])[0, 1]),
                'eps_abs_ratio_on_over_off': float(np.mean(np.abs(en[both])) /
                                                   max(np.mean(np.abs(eo[both])), 1e-12)),
                'epistasis_fraction_off': summary['conditions']['off']['epistasis_fraction'],
                'epistasis_fraction_on': summary['conditions']['on']['epistasis_fraction'],
            }

    json.dump(summary, open(os.path.join(out_dir, 'nk_summary.json'), 'w'), indent=2)
    _print_analyze(summary, per_tag)
    _plot_analyze(out_dir, summary, per_tag, args.z)
    return summary


def _print_analyze(summary, per_tag):
    print(f"\n=== Stage 8 measured epistasis  ({summary['background_label']}, "
          f"M={summary['M']} loci, alpha={summary['alpha']}, R={summary['repeats']}, "
          f"z={summary['z']}) ===")
    print(f"  pairs measured: {summary['n_pairs']}/{summary['n_pairs_total']}"
          + ('  (SUBSAMPLED — K extrapolated per locus)' if summary['n_pairs'] < summary['n_pairs_total'] else ''))
    for tag, c in summary['conditions'].items():
        se_emp = c['eps_se_empirical']
        print(f"\n  -- RL {tag} --")
        print(f"     f(background)        = {c['f_background']:.4f}")
        print(f"     |a_i| mean           = {c['a_abs_mean']:.4f}   (main effects)")
        print(f"     |eps| mean / p95     = {c['eps_abs_mean']:.4f} / {c['eps_abs_p95']:.4f}")
        print(f"     SE(eps) paired med.  = {c['eps_se_median']:.4f}"
              + (f"   empirical (null) = {se_emp:.4f}" if np.isfinite(se_emp) else "   empirical = n/a"))
        print(f"     significant pairs    = {c['frac_pairs_significant']*100:.1f}%")
        print(f"     K >= {c['K_mean']:.2f} ± {c['K_se']:.2f}  "
              f"(median {c['K_median']:.1f}, max {c['K_max']:.0f}, sd {c['K_sd']:.1f})   "
              f"<- MEASURED (lower bound)")
        if c['K_extrapolation_factor'] > 2.0:
            print(f"       ! each locus was measured against few partners; K is scaled "
                  f"up by x{c['K_extrapolation_factor']:.1f} — coarse")
        print(f"     Walsh variance: additive {c['V1_additive']:.4f} | "
              f"pairwise {c['V2_pairwise']:.4f} -> epistasis fraction "
              f"{c['epistasis_fraction']*100:.1f}%")
        if c['debias_consumed_all']:
            print(f"       ! raw V1={c['V1_raw']:.4f} V2={c['V2_raw']:.4f} but noise "
                  f"debias removed ALL of it: at this R the apparent structure is")
            print(f"         indistinguishable from measurement noise. Raise --repeats.")
    if 'rl_comparison' in summary:
        r = summary['rl_comparison']
        print(f"\n  -- learning vs epistasis --")
        print(f"     K_off={r['K_off']:.2f}  K_on={r['K_on']:.2f}  "
              f"dK={r['dK']:+.2f} ± {r['dK_se']:.2f}")
        print(f"     corr(eps_off, eps_on) = {r['eps_corr']:+.3f}   "
              f"mean|eps| on/off = {r['eps_abs_ratio_on_over_off']:.3f}")
        print(f"     epistasis fraction: off {r['epistasis_fraction_off']*100:.1f}% "
              f"-> on {r['epistasis_fraction_on']*100:.1f}%")
    if not any(np.isfinite(summary['conditions'][t]['eps_se_empirical'])
               for t in summary['conditions']):
        print('\n  NOTE: no empirical null available (--null-pairs 0 at jobs time); the')
        print('        threshold rests on propagated SEs alone.')


def _plot_analyze(out_dir, summary, per_tag, z):
    tags = sorted(per_tag)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    ref = per_tag[tags[0]]
    eps = ref['eps']
    # signed magnitude -> diverging, symmetric, neutral midpoint at 0
    lim = np.nanpercentile(np.abs(eps), 99) if np.isfinite(eps).any() else 1.0
    lim = lim if lim > 0 else 1.0
    im = axs[0, 0].imshow(eps, cmap='RdBu_r', vmin=-lim, vmax=lim,
                          origin='lower', interpolation='nearest')
    axs[0, 0].set_title(f'pairwise epistasis $\\epsilon_{{ij}}$ (RL {tags[0]})', fontsize=10)
    axs[0, 0].set_xlabel('locus j'); axs[0, 0].set_ylabel('locus i')
    axs[0, 0].axhline(63.5, color='k', lw=0.6, ls=':')      # hidden | output split
    axs[0, 0].axvline(63.5, color='k', lw=0.6, ls=':')
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046)

    ax = axs[0, 1]
    meas = eps[np.isfinite(eps) & ~np.eye(eps.shape[0], dtype=bool)]
    bins = np.linspace(-lim * 1.6, lim * 1.6, 45)
    ax.hist(meas, bins=bins, color=C_MEAS, alpha=0.75, label='measured $\\epsilon$')
    if ref['null_diffs'].size > 2:
        ax.hist(ref['null_diffs'] / np.sqrt(2.0), bins=bins, histtype='step',
                lw=1.8, color=C_NULL, label='noise (null replicate)')
    if np.isfinite(ref['se_emp']):
        for s in (-z * ref['se_emp'], z * ref['se_emp']):
            ax.axvline(s, color=C_NULL, lw=1.0, ls='--')
        ax.axvline(np.nan, color=C_NULL, lw=1.0, ls='--', label=f'±{z}·SE threshold')
    ax.set_title('epistasis vs the noise floor', fontsize=10)
    ax.set_xlabel('$\\epsilon_{ij}$'); ax.set_ylabel('pairs')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15)

    ax = axs[1, 0]
    kmax_all = 2.5
    for tag in tags:
        Ki = per_tag[tag]['K_i']
        Ki = Ki[np.isfinite(Ki)]
        if Ki.size:
            kmax_all = max(kmax_all, Ki.max() + 1.5)
    kbins = np.linspace(-0.5, kmax_all, 26)
    # step outlines, not alpha-blended fills: overlapping translucent bars make a
    # third colour that reads as its own series.
    for tag, col, ls in zip(tags, (C_MEAS, C_ALT), ('-', '--')):
        Ki = per_tag[tag]['K_i']
        Ki = Ki[np.isfinite(Ki)]
        if Ki.size:
            ax.hist(Ki, bins=kbins, histtype='step', lw=2.0, color=col, ls=ls,
                    label=f'RL {tag}: K≥{summary["conditions"][tag]["K_mean"]:.2f}')
    ax.set_title('per-locus epistatic degree $K_i$', fontsize=10)
    ax.set_xlabel('$K_i$ (significant partners)'); ax.set_ylabel('loci')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15)

    ax = axs[1, 1]
    labels, v1, v2 = [], [], []
    for tag in tags:
        c = summary['conditions'][tag]
        labels.append(f'RL {tag}'); v1.append(c['V1_additive']); v2.append(c['V2_pairwise'])
    xs = np.arange(len(labels))
    ax.bar(xs, v1, width=0.55, color=C_MEAS, label='additive (order 1)')
    ax.bar(xs, v2, width=0.55, bottom=v1, color=C_ALT, label='pairwise (order 2)')
    for k, tag in enumerate(tags):
        f = summary['conditions'][tag]['epistasis_fraction']
        if np.isfinite(f):
            ax.text(xs[k], (v1[k] + v2[k]) * 1.02, f'{f*100:.1f}% epistatic',
                    ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_title('Walsh variance decomposition (debiased)', fontsize=10)
    ax.set_ylabel('variance over the hypercube')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15, axis='y')

    fig.suptitle(f"Stage 8 — measured NK structure: {summary['background_label']}, "
                 f"M={summary['M']}, alpha={summary['alpha']}, R={summary['repeats']}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'nk_epistasis.png')
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f'\n[stage8] plot -> {path}')

    if 'off' in per_tag and 'on' in per_tag:
        eo, en = per_tag['off']['eps'], per_tag['on']['eps']
        both = np.isfinite(eo) & np.isfinite(en) & ~np.eye(eo.shape[0], dtype=bool)
        if both.sum() > 2:
            fig2, ax2 = plt.subplots(figsize=(5.4, 5.2))
            ax2.axhline(0, color='k', lw=0.6); ax2.axvline(0, color='k', lw=0.6)
            ax2.plot([-lim, lim], [-lim, lim], ls=':', color=C_NULL, lw=1.0,
                     label='no change')
            ax2.scatter(eo[both], en[both], s=7, color=C_MEAS, alpha=0.45,
                        linewidths=0, label='pair')
            r = summary['rl_comparison']
            ax2.set_xlabel('$\\epsilon_{ij}$, RL off'); ax2.set_ylabel('$\\epsilon_{ij}$, RL on')
            ax2.set_title(f"does learning flatten epistasis?\n"
                          f"r={r['eps_corr']:+.3f}, mean|$\\epsilon$| ratio "
                          f"{r['eps_abs_ratio_on_over_off']:.3f}", fontsize=10)
            ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.15)
            fig2.tight_layout()
            p2 = os.path.join(out_dir, 'nk_rl_epistasis.png')
            fig2.savefig(p2, dpi=130); plt.close(fig2)
            print(f'[stage8] plot -> {p2}')


# ══════════════════════════════════════════════════════════════════════════════
# model — the tunable second-order NK landscape
# ══════════════════════════════════════════════════════════════════════════════
class NKModel:
    """f(x) = f0 + sum_i a_i x_i + sum_{(i,j) in E} e_ij x_i x_j

    Defined on {0,1}^M (the NK landscape) and, being multilinear, on [0,1]^M
    (partial allele occupancy — the interior a real, partial mutation lands in).
    """

    def __init__(self, f0, a, edges, e, M=None):
        self.f0 = float(f0)
        self.a = np.asarray(a, dtype=float)
        self.M = int(M if M is not None else self.a.size)
        self.edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        self.e = np.asarray(e, dtype=float)
        self.nbr = [[] for _ in range(self.M)]
        for (i, j), w in zip(self.edges, self.e):
            self.nbr[i].append((j, w))
            self.nbr[j].append((i, w))
        self.degree = np.array([len(n) for n in self.nbr], dtype=float)
        # ACTIVE loci = those the model actually has an effect for. A locus with
        # a_i == 0 and no edges is a flat direction: every point along it has zero
        # gradient, so a hill climber stops wherever it started and each start
        # would be counted as its own "local optimum". Excluding them is what
        # keeps the optima count meaningful.
        self.active = np.flatnonzero((self.a != 0.0) | (self.degree > 0))
        if self.active.size == 0:
            self.active = np.arange(self.M)

    def walsh_variance(self):
        """(V1, V2) — additive and pairwise variance in +/-1 coordinates, i.e. over
        uniformly random vertices of the hypercube."""
        w1 = self.a / 2.0
        for (i, j), w in zip(self.edges, self.e):
            w1[i] += w / 4.0
            w1[j] += w / 4.0
        return float(np.sum(w1 ** 2)), float(np.sum((self.e / 4.0) ** 2))

    def rescale(self, target_variance):
        """Scale a and e by a common factor so V1+V2 == target.

        Common scaling leaves the additive/pairwise SPLIT alone, so this holds the
        landscape's overall fitness spread fixed while K decides how much of that
        spread lives in interactions. Without it, raising K just inflates Var(f)
        and lambda barely moves — K stops being a ruggedness dial.
        """
        V1, V2 = self.walsh_variance()
        tot = V1 + V2
        if tot <= 0 or target_variance <= 0:
            return self
        s = np.sqrt(target_variance / tot)
        self.a = self.a * s
        self.e = self.e * s
        self.nbr = [[] for _ in range(self.M)]
        for (i, j), w in zip(self.edges, self.e):
            self.nbr[i].append((j, w))
            self.nbr[j].append((i, w))
        return self

    # ── fitness ──────────────────────────────────────────────────────────────
    def f(self, x):
        x = np.asarray(x, dtype=float)
        if self.edges.size:
            pair = float(np.sum(self.e * x[self.edges[:, 0]] * x[self.edges[:, 1]]))
        else:
            pair = 0.0
        return self.f0 + float(self.a @ x) + pair

    def gain(self, x, i):
        """f(flip locus i) - f(x). Exact, from the neighbour list."""
        s = self.a[i] + sum(w * x[j] for j, w in self.nbr[i])
        return (1.0 - 2.0 * x[i]) * s

    def all_gains(self, x):
        """Gains for the ACTIVE loci only (flat directions are not moves)."""
        return np.array([self.gain(x, i) for i in self.active])

    # ── dynamics ─────────────────────────────────────────────────────────────
    def hill_climb(self, x, max_steps=100000):
        """Steepest-ascent single-flip climb to a local optimum."""
        x = np.array(x, dtype=float).copy()
        steps = 0
        while steps < max_steps:
            g = self.all_gains(x)
            k = int(np.argmax(g))
            if g[k] <= 1e-12:
                break
            i = int(self.active[k])
            x[i] = 1.0 - x[i]
            steps += 1
        return x, self.f(x), steps

    def optima_density(self, n_samples, rng):
        """P(a uniformly random vertex is a local optimum).

        NOT called by default: at M_active=70 this is ~2^-70 even for a rugged
        landscape, so no feasible sample resolves it (20k samples returned 0.00000
        at every K) while costing n_samples*M gain evaluations. The measures that
        DO discriminate at this M are `distinct_per_start`, the adaptive-walk
        length (`climb_steps_mean`, which shortens as K traps climbers sooner) and
        the fitness actually reached (the complexity catastrophe) — all reported by
        local_optima_stats. Kept for small-M sanity checks.
        """
        n_opt = 0
        for _ in range(n_samples):
            x = np.zeros(self.M)
            x[self.active] = (rng.random(self.active.size) < 0.5).astype(float)
            if np.all(self.all_gains(x) <= 1e-12):
                n_opt += 1
        return n_opt / n_samples

    def local_optima_stats(self, n_starts, rng):
        """Density of local optima and their fitness spread — the NK analogue of
        the Stage-4/7 peak count. Only ACTIVE loci vary, so a purely additive
        model correctly yields exactly one optimum."""
        seen, fits, climbs, already = {}, [], [], 0
        for _ in range(n_starts):
            x0 = np.zeros(self.M)
            x0[self.active] = (rng.random(self.active.size) < 0.5).astype(float)
            if np.all(self.all_gains(x0) <= 1e-12):
                already += 1
            xf, ff, st = self.hill_climb(x0)
            seen[tuple(xf[self.active].astype(int))] = ff
            fits.append(ff); climbs.append(st)
        fits = np.array(fits)
        return {'n_starts': n_starts, 'M_active': int(self.active.size),
                'n_distinct_optima': len(seen),
                'distinct_per_start': len(seen) / n_starts,
                'frac_starts_already_optimal': already / n_starts,
                'optimum_f_mean': float(fits.mean()),
                'optimum_f_sd': float(fits.std(ddof=1)) if len(fits) > 1 else 0.0,
                'optimum_f_max': float(fits.max()),
                'climb_steps_mean': float(np.mean(climbs))}

    def walk_autocorr(self, steps, kmax, rng, operator='single', x_step=None,
                      p_flip=None):
        """Random-walk autocorrelation of the surrogate, Weinberger-style.

        operator='single'  flip exactly one locus per step. lambda is then in
                           single-mutation units — the canonical NK measure, and
                           the one to quote.
        operator='real'    the GA's ACTUAL per-step displacement, in the
                           multilinear interior. A real step adds N(0,MUT_SIGMA)
                           to each weight of a block with prob MUT_PROB, so the
                           component along that block's allele axis has sd
                               MUT_SIGMA*sqrt(MUT_PROB) / ||delta[block]||
                           (the sqrt(block size) from the number of mutated
                           weights cancels the 1/sqrt(block size) from projecting
                           a random direction onto one axis, so it is the same for
                           every block). Pass that per-locus sd as `x_step`;
                           increments reflect off the [0,1] walls. THIS is the
                           operator comparable to Stage 6.
        operator='reach'   flip each locus with p_flip (default: the GA's
                           per-block touch probability, ~0.82). Included only to
                           show why full flips are the wrong model: flipping 82%
                           of loci per step OVERSHOOTS decorrelation and drives
                           rho(1) negative. Not a comparison to Stage 6.
        """
        if p_flip is None:
            p_flip = block_touch_prob()[1]
        x = np.zeros(self.M)
        x[self.active] = (rng.random(self.active.size) < 0.5).astype(float)
        series = np.empty(steps + 1)
        series[0] = self.f(x)
        for t in range(1, steps + 1):
            if operator == 'single':
                i = int(self.active[rng.integers(self.active.size)])
                x[i] = 1.0 - x[i]
            elif operator == 'real':
                if x_step is None:
                    raise ValueError("operator='real' needs x_step (see real_x_step)")
                x = x + rng.normal(0.0, 1.0, self.M) * np.asarray(x_step)
                # reflect into [0,1] so occupancy stays interpretable
                x = np.abs(x)
                x = 1.0 - np.abs(1.0 - x)
                np.clip(x, 0.0, 1.0, out=x)
            else:
                mask = rng.random(self.M) < p_flip
                x[mask] = 1.0 - x[mask]
            series[t] = self.f(x)

        y = series - series.mean()
        # elementwise sum rather than `@`: Apple Accelerate's BLAS raises spurious
        # FP flags on large dot products, which numpy surfaces as bogus
        # divide-by-zero / overflow RuntimeWarnings.
        denom = float(np.sum(y * y))
        rho = np.array([1.0] + [(float(np.sum(y[:-k] * y[k:])) / denom)
                                if denom > 0 else np.nan
                                for k in range(1, kmax + 1)])
        ks = np.arange(1, kmax + 1)
        lam = float('nan')
        if np.isfinite(rho[1]) and rho[1] > 0:
            good = rho[1:] > 0
            if good.sum() >= 2:
                slope = np.polyfit(ks[good], np.log(rho[1:][good]), 1)[0]
                lam = -1.0 / slope if slope < 0 else float('inf')
        return rho, lam

    # ── io ───────────────────────────────────────────────────────────────────
    def to_dict(self, **extra):
        return {'M': self.M, 'f0': self.f0, 'a': self.a.tolist(),
                'edges': self.edges.tolist(), 'e': self.e.tolist(),
                'K_mean': float(self.degree.mean()), **extra}

    @staticmethod
    def from_dict(d):
        return NKModel(d['f0'], d['a'], d['edges'], d['e'], M=d.get('M'))

    @staticmethod
    def from_walsh(c, w1, edges, w2, M=None):
        """Build from +/-1 (Walsh) coefficients, converting exactly to the
        {0,1} form the class stores.

        With s_i = 2x_i - 1:
            sum w_i s_i        = 2 sum w_i x_i - sum w_i
            sum w_ij s_i s_j   = sum w_ij (4 x_i x_j - 2 x_i - 2 x_j + 1)
        hence
            a_i    = 2 w_i - 2 sum_{j in N(i)} w_ij
            eps_ij = 4 w_ij
            f0     = c - sum w_i + sum w_ij

        This constructor is what makes K a real dial: in the {0,1}
        parameterisation an eps_ij x_i x_j term ALSO contributes main effects
        (w_i picks up sum_j eps_ij/4), so adding edges inflates the additive
        variance alongside the pairwise variance and the epistatic SHARE barely
        moves. Choosing w_i and w_ij independently in Walsh space decouples them.
        """
        w1 = np.asarray(w1, dtype=float)
        M = int(M if M is not None else w1.size)
        edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        w2 = np.asarray(w2, dtype=float)
        a = 2.0 * w1.copy()
        for (i, j), w in zip(edges, w2):
            a[i] -= 2.0 * w
            a[j] -= 2.0 * w
        f0 = float(c) - float(w1.sum()) + float(w2.sum())
        return NKModel(f0, a, edges, 4.0 * w2, M=M)


def real_x_step(out_dir):
    """Per-locus sd of a real GA step measured in ALLELE-OCCUPANCY units.

    A step adds N(0,MUT_SIGMA) to each weight of block i with prob MUT_PROB, so
    the block displacement has E||.||^2 = MUT_PROB*b*MUT_SIGMA^2. A random
    direction in a b-dimensional block puts a 1/sqrt(b) fraction of its length
    along the allele axis, so the displacement ALONG that axis has sd
    MUT_SIGMA*sqrt(MUT_PROB) — independent of b. Dividing by the axis length
    ||delta[block_i]|| converts it to x units.

    Returns (x_step per locus, mean) or (None, nan) if hypercube.json is missing.
    """
    p = os.path.join(out_dir, 'hypercube.json')
    if not os.path.exists(p):
        return None, float('nan')
    _, delta, _ = load_hypercube(p)
    axis = np.array([np.linalg.norm(delta[blk]) for blk in ll.FILTER_BLOCKS])
    step_w = MUT_SIGMA * np.sqrt(MUT_PROB)          # weight units, along the axis
    with np.errstate(divide='ignore', invalid='ignore'):
        xs = np.where(axis > 0, step_w / axis, 0.0)
    return xs, float(np.mean(xs[axis > 0])) if np.any(axis > 0) else float('nan')


def _measured_model(out_dir, tag, z, no_null_floor=False):
    """Rebuild (a, eps, sig) for one condition from the saved analyze output."""
    p = os.path.join(out_dir, f'epistasis_{tag}.npz')
    if not os.path.exists(p):
        raise SystemExit(f'model: {p} not found — run `analyze` first')
    d = np.load(p)
    return d['a'], d['eps'], d['eps_se'], d['sig']


def build_model(out_dir, tag='off', wiring='measured', K=None, rng=None,
                keep='significant'):
    """Second-order NK model from the measurement.

    wiring='measured': edges are the significant pairs (or the top-K per locus).
    wiring='random':   a random graph of mean degree K, with coefficients
                       RESAMPLED from the measured distributions — the dial.
    """
    rng = rng or np.random.default_rng(0)
    a, eps, eps_se, sig = _measured_model(out_dir, tag, None)
    summ = json.load(open(os.path.join(out_dir, 'nk_summary.json')))
    f0 = summ['conditions'][tag]['f_background']
    M = int(summ['M'])
    a = np.where(np.isfinite(a), a, 0.0)

    iu = np.triu_indices(M, 1)
    sig_u = sig[iu]
    eps_u = np.where(np.isfinite(eps), eps, 0.0)[iu]

    if wiring == 'measured':
        if keep == 'all':
            mask = np.isfinite(eps[iu])
        else:
            mask = sig_u
        edges = np.column_stack([iu[0][mask], iu[1][mask]])
        e = eps_u[mask]
        if K is not None:
            # thin to mean degree K by keeping the largest |eps| edges
            target = int(round(K * M / 2))
            if 0 <= target < len(e):
                keep_idx = np.argsort(-np.abs(e))[:target]
                edges, e = edges[keep_idx], e[keep_idx]
    elif wiring == 'random':
        # THE DIAL. Built in Walsh space so the epistatic SHARE tracks K instead
        # of saturating (see NKModel.from_walsh).
        if K is None:
            K = float(summ['conditions'][tag]['K_mean'])
        n_e = max(0, min(int(round(K * M / 2)), len(iu[0])))
        pick = rng.choice(len(iu[0]), size=n_e, replace=False)
        edges = np.column_stack([iu[0][pick], iu[1][pick]])

        # resample Walsh coefficients from the MEASURED distributions
        w1_meas = np.load(os.path.join(out_dir, f'epistasis_{tag}.npz'))['w1']
        w1_pool = w1_meas[np.isfinite(w1_meas) & (w1_meas != 0)]
        if w1_pool.size == 0:
            w1_pool = np.array([0.0])
        w2_pool = (eps_u[sig_u] if sig_u.any()
                   else eps_u[np.isfinite(eps_u) & (eps_u != 0)]) / 4.0
        if w2_pool.size == 0:
            w2_pool = np.array([0.0])
        w1 = rng.choice(w1_pool, size=M, replace=True)
        w2 = rng.choice(w2_pool, size=n_e, replace=True)
        return NKModel.from_walsh(f0, w1, edges, w2, M=M)
    else:
        raise ValueError(f'unknown wiring {wiring}')

    return NKModel(f0, a, edges, e, M=M)


def cmd_model(args):
    rng = np.random.default_rng(args.seed)
    tag = args.condition
    m = build_model(args.out_dir, tag=tag, wiring=args.wiring, K=args.K, rng=rng,
                    keep=args.keep)
    summ = json.load(open(os.path.join(args.out_dir, 'nk_summary.json')))

    stats = m.local_optima_stats(args.starts, rng)
    rho_s, lam_s = m.walk_autocorr(args.walk_steps, args.kmax, rng, operator='single')
    xs, xs_mean = real_x_step(args.out_dir)
    if xs is not None:
        rho_r, lam_r = m.walk_autocorr(args.walk_steps, args.kmax, rng,
                                       operator='real', x_step=xs)
    else:
        rho_r, lam_r = np.full(args.kmax + 1, np.nan), float('nan')
    rho_x, lam_x = m.walk_autocorr(args.walk_steps, args.kmax, rng, operator='reach')

    doc = m.to_dict(
        wiring=args.wiring, condition=tag, keep=args.keep,
        source=os.path.abspath(args.out_dir),
        background_label=summ.get('background_label'), alpha=summ.get('alpha'),
        K_requested=args.K,
        n_edges=int(m.edges.shape[0]),
        local_optima=stats,
        x_step_mean=xs_mean,
        rho_single=rho_s.tolist(), lambda_single=lam_s,
        rho_real=rho_r.tolist(), lambda_real=lam_r,
        rho_reach=rho_x.tolist(), lambda_reach=lam_x,
    )
    path = os.path.join(args.out_dir, args.model_out)
    json.dump(doc, open(path, 'w'), indent=2)

    print(f'\n=== Stage 8 NK model  (condition RL {tag}, wiring={args.wiring}) ===')
    print(f'  M={m.M} loci, {m.edges.shape[0]} edges, mean degree K={m.degree.mean():.2f} '
          f'(max {m.degree.max():.0f})')
    print(f'  f0={m.f0:.4f}   |a| mean={np.abs(m.a).mean():.4f}   '
          f'|e| mean={np.abs(m.e).mean() if m.e.size else 0:.4f}')
    print(f'  local optima: {stats["distinct_per_start"]:.3f} distinct per start '
          f'({stats["n_distinct_optima"]}/{stats["n_starts"]}), '
          f'{stats["frac_starts_already_optimal"]*100:.1f}% of random starts are optima')
    print(f'  optimum f: {stats["optimum_f_mean"]:.4f} ± {stats["optimum_f_sd"]:.4f} '
          f'(max {stats["optimum_f_max"]:.4f}), climb {stats["climb_steps_mean"]:.1f} steps')
    print(f'  surrogate lambda / rho(1):')
    print(f'     single-flip  lam={lam_s:8.2f}  rho1={rho_s[1]:+.4f}   '
          f'<- canonical NK measure')
    print(f'     real GA step lam={lam_r:8.2f}  rho1={rho_r[1]:+.4f}   '
          f'<- COMPARE THIS TO STAGE 6 (x_step={xs_mean:.4f} per locus)')
    print(f'     full-flip    lam={lam_x:8.2f}  rho1={rho_x[1]:+.4f}   '
          f'(flips ~82% of loci: overshoots, hence rho1<0)')
    if np.isfinite(xs_mean) and xs_mean > 0:
        print(f'  allele SCALE check: one GA step covers {xs_mean*100:.2f}% of an allele '
              f'axis, so one')
        print(f'     allele change ~ {1/xs_mean:.0f} steps of DIRECTED movement '
              f'(~{1/xs_mean**2:.0f} of undirected drift).')
        print(f'     If that is far from the mutational scale you care about, change '
              f'--alpha at `jobs` time.')
    print(f'\n[stage8] model -> {path}')
    print('  The "real GA step" row is the only one comparable to Stage 6 rho(1): it')
    print('  moves in the multilinear interior by the displacement a GA step actually')
    print('  makes along each allele axis, instead of flipping whole alleles.')
    return doc


def cmd_sim(args):
    """Turn the K dial and report what changes."""
    rng = np.random.default_rng(args.seed)
    ks = [float(t) for t in args.k_sweep.split(',')] if args.k_sweep else [None]
    # Hold total variance fixed across the sweep so K moves the ADDITIVE/EPISTATIC
    # split rather than the overall fitness spread. Target = the measured total.
    summ = json.load(open(os.path.join(args.out_dir, 'nk_summary.json')))
    c = summ['conditions'][args.condition]
    target = (c['V1_additive'] + c['V2_pairwise']) or (c['V1_raw'] + c['V2_raw'])
    if args.no_normalize:
        target = None

    rows = []
    for K in ks:
        m = build_model(args.out_dir, tag=args.condition, wiring='random', K=K,
                        rng=np.random.default_rng(args.seed))
        if target:
            m.rescale(target)
        V1, V2 = m.walsh_variance()
        st = m.local_optima_stats(args.starts, rng)
        rho, lam = m.walk_autocorr(args.walk_steps, args.kmax, rng, operator='single')
        rows.append({'K_requested': K, 'K_actual': float(m.degree.mean()),
                     'n_edges': int(m.edges.shape[0]),
                     'V1': V1, 'V2': V2,
                     'epistasis_fraction': (V2 / (V1 + V2)) if (V1 + V2) > 0 else float('nan'),
                     'rho1': float(rho[1]), 'lambda_single': lam, **st})

    hdr = (f'{"K":>6}{"edges":>8}{"epi%":>7}{"rho(1)":>9}{"lambda":>9}'
           f'{"distinct":>10}{"climb":>8}{"f_opt":>10}')
    print(f'\n=== Stage 8 ruggedness dial  (RL {args.condition}, random wiring, '
          f'coefficients resampled from measurement) ===')
    print(f'    M_active={rows[0]["M_active"]} loci, '
          + (f'total variance held at {target:.4f}' if target
             else 'variance NOT normalised (--no-normalize)'))
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f'{r["K_actual"]:>6.1f}{r["n_edges"]:>8d}'
              f'{r["epistasis_fraction"]*100:>6.1f}%{r["rho1"]:>9.4f}'
              f'{r["lambda_single"]:>9.2f}'
              f'{r["distinct_per_start"]:>10.3f}{r["climb_steps_mean"]:>8.1f}'
              f'{r["optimum_f_mean"]:>10.4f}')
    print('\nK up -> epistatic variance share up, rho(1) down, lambda broadly down,')
    print('more distinct local optima, shorter adaptive walks (climbers trap sooner).')
    print('Total variance is held fixed, so those four track ruggedness and not scale.')
    print('\nf_opt is DESCRIPTIVE ONLY — do not read it as a complexity catastrophe.')
    print('Fixing the variance does not fix the reachable MAXIMUM: adding edges raises')
    print('the best attainable vertex even at constant Var(f), so f_opt drifts UP with K.')
    print('A catastrophe claim needs f_opt relative to the global optimum, which is not')
    print('computable at M=70.')
    print('(local-optima DENSITY is omitted: ~2^-70 at this M, unmeasurable by sampling)')

    path = os.path.join(args.out_dir, 'nk_k_sweep.json')
    json.dump(rows, open(path, 'w'), indent=2)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    Ka = [r['K_actual'] for r in rows]
    axs[0].plot(Ka, [r['lambda_single'] for r in rows], 'o-', color=C_MEAS, lw=2, ms=8)
    axs[0].set_xlabel('K (mean epistatic degree)'); axs[0].set_ylabel('$\\lambda$ (single-flip steps)')
    axs[0].set_title('ruggedness vs K', fontsize=10)
    axs[1].plot(Ka, [r['distinct_per_start'] for r in rows], 'o-', color=C_ALT,
                lw=2, ms=8, label='distinct optima / start')
    axs[1].plot(Ka, [r['epistasis_fraction'] for r in rows], 's--', color=C_MEAS,
                lw=2, ms=7, label='epistatic variance share')
    axs[1].set_xlabel('K'); axs[1].set_ylabel('fraction')
    axs[1].set_ylim(0, 1.05)
    axs[1].legend(fontsize=8, frameon=False)
    axs[1].set_title('multi-peakedness vs K', fontsize=10)
    axs[2].plot(Ka, [r['climb_steps_mean'] for r in rows], 'o-', color=C_MEAS,
                lw=2, ms=8)
    axs[2].set_xlabel('K'); axs[2].set_ylabel('hill-climb steps to a local optimum')
    axs[2].set_title('adaptive walk length vs K', fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.15)
    fig.suptitle(f'Stage 8 — tuning ruggedness in the measured NK model '
                 f'(RL {args.condition})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(args.out_dir, 'nk_k_sweep.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f'[stage8] sweep -> {path}\n[stage8] plot  -> {p}')
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# project — map a real genome onto the hypercube (with an honesty check)
# ══════════════════════════════════════════════════════════════════════════════
def project_genome(theta, theta0, delta):
    """Least-squares allele occupancy per locus, clipped to [0,1].

    x_i = <theta-theta0, delta>_block_i / ||delta_block_i||^2
    Also returns the residual FRACTION: how much of theta's displacement from
    theta0 the 70-D allele subspace fails to capture. Large residual => the
    projected NK score is not about this genome.
    """
    diff = theta - theta0
    x = np.zeros(M_LOCI)
    for i, blk in enumerate(ll.FILTER_BLOCKS):
        dn = float(delta[blk] @ delta[blk])
        x[i] = (float(diff[blk] @ delta[blk]) / dn) if dn > 0 else 0.0
    x_clipped = np.clip(x, 0.0, 1.0)
    recon = theta0.copy()
    for i, blk in enumerate(ll.FILTER_BLOCKS):
        recon[blk] = recon[blk] + x_clipped[i] * delta[blk]
    denom = float(np.linalg.norm(diff))
    resid = float(np.linalg.norm(theta - recon)) / denom if denom > 0 else 0.0
    return x, x_clipped, resid


def cmd_project(args):
    theta0, delta, hyp = load_hypercube(os.path.join(args.out_dir, 'hypercube.json'))
    model = None
    mp = os.path.join(args.out_dir, args.model_out)
    if os.path.exists(mp):
        model = NKModel.from_dict(json.load(open(mp)))

    if args.record == 'fittest':
        df = ll.load_genome_csv(args.genome_csv, record_type='fittest').sort_values('generation')
        theta, label = df.iloc[-1]['genome'].copy(), f'fittest_g{int(df.iloc[-1]["generation"])}'
    else:
        gens, mat = ll.centroid_trajectory(args.genome_csv)
        theta, label = mat[-1].copy(), f'centroid_g{int(gens[-1])}'

    x, xc, resid = project_genome(theta, theta0, delta)
    print(f'\n=== Stage 8 projection: {label} onto the {hyp["background_label"]} '
          f'hypercube (alpha={hyp["alpha"]}) ===')
    print(f'  allele occupancy x: mean {xc.mean():.3f}, '
          f'{int((xc > 0.5).sum())}/{M_LOCI} loci past halfway')
    print(f'  pre-clip range: [{x.min():.2f}, {x.max():.2f}] '
          f'({int(((x < 0) | (x > 1)).sum())} loci outside [0,1])')
    print(f'  RESIDUAL FRACTION = {resid:.3f}  '
          f'({resid*100:.1f}% of this genome\'s displacement from theta0 lies OUTSIDE')
    print(f'                      the 70-D allele subspace)')
    if model is not None:
        print(f'  NK score: vertices f(round(x)) = {model.f(np.round(xc)):.4f}   '
              f'multilinear f(x) = {model.f(xc):.4f}   f(background) = {model.f0:.4f}')
    if resid > args.resid_warn:
        print(f'\n  ** DO NOT QUOTE THAT SCORE AS THIS GENOME\'S FITNESS. ** residual '
              f'{resid:.2f} > {args.resid_warn}:')
        print('     the projection discards most of what makes this genome what it is.')
        print('     Use the surrogate for DYNAMICS at tunable K, not to rank real genomes.')
    out = {'label': label, 'genome_csv': os.path.abspath(args.genome_csv),
           'residual_fraction': resid, 'x': xc.tolist(), 'x_preclip': x.tolist(),
           'f_nk_vertex': (model.f(np.round(xc)) if model else None),
           'f_nk_multilinear': (model.f(xc) if model else None)}
    p = os.path.join(args.out_dir, f'projection_{args.record}.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f'\n[stage8] projection -> {p}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
def main():
    # legacy shim: `stage8_nk.py --out-dir ...` still means `quick`
    argv = sys.argv[1:]
    if argv and argv[0].startswith('-') and argv[0] not in ('-h', '--help'):
        argv = ['quick'] + argv

    ap = argparse.ArgumentParser(description='Stage 8 — NK model from the measured landscape')
    sub = ap.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('quick', help='legacy rho(1)->K_eff from Stage-6 dirs')
    q.add_argument('--out-dir', nargs='+', required=True,
                   help='one or more Stage-6 walk dirs (globs allowed)')

    j = sub.add_parser('jobs', help='write jobs.json for the epistasis measurement')
    j.add_argument('--out-dir', required=True)
    j.add_argument('--genome-csv', required=True)
    j.add_argument('--background', default='centroid',
                   choices=['centroid', 'evolved', 'fittest', 'random'])
    j.add_argument('--alpha', type=float, default=0.25,
                   help='allele displacement as a fraction of each block norm')
    j.add_argument('--repeats', type=int, default=8)
    j.add_argument('--map-start', type=int, default=0)
    j.add_argument('--loci', default=None, help='comma list; default all 70')
    j.add_argument('--pairs', type=int, default=None,
                   help='subsample this many pairs; default all C(M,2)')
    j.add_argument('--null-pairs', type=int, default=100,
                   help='pairs re-measured on a disjoint map set for the empirical SE')
    j.add_argument('--rl-passes', default='both', choices=['both', 'off', 'on'])
    j.add_argument('--config-overrides', default=None)
    j.add_argument('--allele-seed', type=int, default=8001)
    j.add_argument('--seed', type=int, default=0)

    a = sub.add_parser('analyze', help='results.csv -> epsilon, K, Walsh decomposition')
    a.add_argument('--out-dir', required=True)
    a.add_argument('--z', type=float, default=2.0, help='significance threshold in SEs')
    a.add_argument('--no-null-floor', action='store_true',
                   help='use propagated SEs only, ignore the empirical null floor')

    m = sub.add_parser('model', help='build the second-order NK model')
    m.add_argument('--out-dir', required=True)
    m.add_argument('--condition', default='off', choices=['off', 'on'])
    m.add_argument('--wiring', default='measured', choices=['measured', 'random'])
    m.add_argument('--keep', default='significant', choices=['significant', 'all'])
    m.add_argument('--K', type=float, default=None, help='target mean degree')
    m.add_argument('--starts', type=int, default=200)
    m.add_argument('--walk-steps', type=int, default=20000)
    m.add_argument('--kmax', type=int, default=30)
    m.add_argument('--model-out', default='nk_model.json')
    m.add_argument('--seed', type=int, default=0)

    s = sub.add_parser('sim', help='sweep K and report the dynamics that change')
    s.add_argument('--out-dir', required=True)
    s.add_argument('--condition', default='off', choices=['off', 'on'])
    s.add_argument('--k-sweep', default='0,1,2,4,8,16,32')
    s.add_argument('--no-normalize', action='store_true',
                   help='do NOT hold total variance fixed across the K sweep')
    s.add_argument('--starts', type=int, default=200)
    s.add_argument('--walk-steps', type=int, default=20000)
    s.add_argument('--kmax', type=int, default=30)
    s.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('project', help='project a real genome onto the hypercube')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--genome-csv', required=True)
    p.add_argument('--record', default='centroid', choices=['centroid', 'fittest'])
    p.add_argument('--model-out', default='nk_model.json')
    p.add_argument('--resid-warn', type=float, default=0.5)

    args = ap.parse_args(argv)
    {'quick': cmd_quick, 'jobs': cmd_jobs, 'analyze': cmd_analyze,
     'model': cmd_model, 'sim': cmd_sim, 'project': cmd_project}[args.cmd](args)


if __name__ == '__main__':
    main()
