"""
landscape/stage8_nk.py
======================================================================
Stage 8 — characterise the measured landscape in NK terms (effective K).

We do NOT fit a literal NK model — the genome is 3910 CONTINUOUS weights, so
there is no discrete locus table or epistasis wiring to recover from f(theta)
samples. Instead we map the ALREADY-MEASURED random-walk autocorrelation
(Stage 6) to Kauffman's ruggedness parameter K via the standard isotropic-NK
relation (Weinberger 1990; Stadler landscape theory):

    rho(d) = (1 - (K+1)/N)^d          # autocorr after a move of Hamming dist d

A Stage-6 walk STEP is not a single-locus flip: the GA operator perturbs each of
the N weights with prob p = MUT_PROB, so a step moves d_bar = N*p loci at once.
The measured lag-1 autocorrelation is therefore rho(d_bar), and

    K + 1 = N * (1 - rho(1)^(1/d_bar))            # per-locus epistasis
    lambda_locus = d_bar * lambda_step ≈ N/(K+1)  # corr length in single muts

We also print the NAIVE per-step estimate (d_bar treated as 1) so the size of the
multi-locus correction is explicit.

CAVEATS (state these with any number quoted):
  * A spin flip fully re-randomises a locus; the GA's small N(0,MUT_SIGMA) nudge
    only PARTIALLY decorrelates a weight's contribution. Treating each nudged
    weight as one Hamming step therefore UNDER-counts the epistasis needed to
    produce the observed rho -> the true K is likely >= the estimate here.
  * The isotropic rho(d)=(1-(K+1)/N)^d is itself an AR(1) approximation to NK.
  * Hence "effective K" is a ruggedness CHARACTERISATION, not a claim the
    landscape IS an NK landscape.

Consumes the walk-averaged, detrended rho(1) from the Stage-6 lambda_summary.json
(no new sim runs, and the same hardened estimate used for lambda). Propagates the
cross-walk SE on rho(1) into a SE on K_eff so two conditions can be compared.

Usage
-----
  python landscape/stage8_nk.py --out-dir landscape/out/stage6_hard_random
  python landscape/stage8_nk.py --out-dir landscape/out/stage6_*    # many, table
"""

import argparse
import glob
import json
import os

import numpy as np

import ll_common as ll


def _k_eff(rho1, N, d_bar):
    """NK effective K from the isotropic relation rho(d_bar)=(1-(K+1)/N)^d_bar."""
    if not (0 < rho1 < 1):
        return float('nan')
    return N * (1.0 - rho1 ** (1.0 / d_bar)) - 1.0


def analyze_dir(out_dir):
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    summ = json.load(open(os.path.join(out_dir, 'lambda_summary.json')))
    N = ll.GENOME_SIZE
    p = float(meta.get('mut_prob', 0.03))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', nargs='+', required=True,
                    help='one or more Stage-6 walk dirs (globs allowed)')
    args = ap.parse_args()

    dirs = []
    for d in args.out_dir:
        dirs.extend(sorted(glob.glob(d)) if any(c in d for c in '*?[') else [d])
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'results.csv'))]

    rows = []
    for d in dirs:
        r = analyze_dir(d)
        rows.append(r)
        json.dump(r, open(os.path.join(d, 'nk_summary.json'), 'w'), indent=2)

    # table
    hdr = f'{"walk":<34}{"rho(1)":>16}{"lam_step":>10}{"K_eff":>16}'
    print(f'\n=== Stage 8 - effective NK K  (N={ll.GENOME_SIZE}, d_bar=N*p) ===')
    print(hdr); print('-' * len(hdr))
    for r in rows:
        name = os.path.basename(os.path.normpath(r['out_dir']))
        rho_s = f'{r["rho1"]:.4f}±{r["rho1_se"]:.4f}' if np.isfinite(r["rho1_se"]) else f'{r["rho1"]:.4f}'
        k_s = f'{r["K_eff"]:.2f}±{r["K_eff_se"]:.2f}' if np.isfinite(r["K_eff_se"]) else f'{r["K_eff"]:.2f}'
        print(f'{name:<34}{rho_s:>16}{r["lambda_step"]:>10.2f}{k_s:>16}')
    print('\n(K_eff = N*(1-rho1^(1/d_bar)) - 1, multi-locus corrected; '
          'small K = smooth, large K = rugged/epistatic)')
    print('rho(1) is walk-averaged + detrended (Stage 6); ± = cross-walk SE.')
    print('caveat: continuous nudge != full spin flip -> K_eff is a lower bound.')


if __name__ == '__main__':
    main()
