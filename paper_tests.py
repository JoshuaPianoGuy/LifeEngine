"""
paper_tests.py
======================================================================
The reduced set of tests the PAPER reports — one test per claim, chosen so that
each is defensible on its own and needs no further machinery to explain.

    python paper_tests.py

Reads only durable per-run outputs that the big analyses already wrote, so this
never re-parses a log directory:

    output/conditions_by_environment/time_to_threshold.csv   one row per run
    output/conditions_by_environment/runs.csv                one row per run
    output/conditions_by_environment/variance_components.csv ICC / design effect
    output/genome_evaluations/assimilation_arm_by_generation.csv

Writes output/stats/paper_tests.csv (one row per test) and prints the same.

THE FOUR FAMILIES, AND WHY EACH IS THE RIGHT ONE
------------------------------------------------
1. LOG-RANK on time to sustained viability, per environment, THREE conditions at
   once, STRATIFIED BY MAP SEED. The headline result is a difference in *when*
   viability is reached (median generation 51 vs 405 in the predator
   environment), and the log-rank is the test of exactly that, with censoring at
   generation 1000 handled and no proportional-hazards assumption to defend.
   Stratifying compares runs only against other runs on the same terrain, which
   is what the ICCs say is necessary. The unstratified statistic is computed too
   and reported beside it, so the effect of the clustering is visible rather than
   asserted.

2. FISHER'S EXACT on below-threshold counts, predator environment. 10/100 versus
   0/100 is the cleanest binary outcome in the study, and at those counts an
   exact test is the honest one: a chi-square approximation is not trustworthy
   with a zero cell. Pairwise A-vs-B and A-vs-C, Holm-adjusted over the two.
   Fisher treats the 100 runs as independent, which they are not; the design
   effect in family 4 is what says how far that overstates the evidence, and the
   seed-stratified CMH in output/stats/viability_tests.csv is the clustered
   version for anyone who wants it.

3. BROWN-FORSYTHE on converged fitness, per environment, on the 100 RUN-LEVEL
   values per condition. The claim that the learning condition is the *consistent*
   one is a claim about spread, and no test of means can carry it. Brown-Forsythe
   is Levene's test centred on the MEDIAN, which is what makes it robust to the
   non-normal, bimodal distributions the predator environment produces — the
   mean-centred version would be reading the outliers it is supposed to survive.
   Run-level values are correct here: aggregating to seed means would destroy the
   within-seed variance that is the object of the claim. The omnibus is reported
   with all three pairwise contrasts, Holm-adjusted over the three, because the
   omnibus alone cannot say WHICH spreads differ or in which direction.

4. WILCOXON SIGNED-RANK on the inherited-genome curves, paired across the 10 map
   seeds at each founder-dump checkpoint. Read from the assimilation analysis
   rather than recomputed. Rank-based because n = 10 is small enough that
   normality is an assumption worth not making.

Deliberately NOT here: a Cox model (the log-rank already answers the question and
a hazard ratio would need its own paragraph to explain, and its PH assumption is
violated anyway), generation-by-generation tests of the fitness curves, and the
landscape metrics (n = 3 slices per cell; the descriptive contrast is the honest
form of that claim).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

VIABILITY = 4.375          # ExperimentParams: 3.5 food per child / 0.8 success
CONDITIONS = ['evolution', 'learning', 'pure_rl']
SHORT = {'evolution': 'A', 'learning': 'B', 'pure_rl': 'C'}


# ── 1. log-rank ───────────────────────────────────────────────────────────────
def _logrank_terms(time, event, group, groups):
    """One stratum's O-E vector and covariance matrix (Mantel-Haenszel)."""
    k = len(groups)
    U = np.zeros(k)
    V = np.zeros((k, k))
    idx = {g: i for i, g in enumerate(groups)}
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n_t = at_risk.sum()
        d_t = ((time == t) & (event == 1)).sum()
        if n_t <= 1 or d_t == 0:
            continue
        n_j = np.array([((group == g) & at_risk).sum() for g in groups], dtype=float)
        d_j = np.array([((group == g) & (time == t) & (event == 1)).sum()
                        for g in groups], dtype=float)
        p = n_j / n_t
        U += d_j - d_t * p
        c = d_t * (n_t - d_t) / (n_t - 1)
        V += c * (np.diag(p) - np.outer(p, p))
    return U, V


def logrank(df, groups, strata=None):
    """Multivariate log-rank, optionally stratified. Returns (chi2, df, p)."""
    k = len(groups)
    U = np.zeros(k)
    V = np.zeros((k, k))
    blocks = [df] if strata is None else [g for _s, g in df.groupby(strata)]
    for b in blocks:
        u, v = _logrank_terms(b['time'].to_numpy(float),
                              b['event'].to_numpy(int),
                              b['condition'].to_numpy(), groups)
        U += u
        V += v
    # Drop one group: the terms sum to zero, so V is singular at full rank.
    u, v = U[:-1], V[:-1, :-1]
    chi2 = float(u @ np.linalg.pinv(v) @ u)
    dof = k - 1
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


def check_against_lifelines(df, groups):
    """The unstratified statistic must reproduce lifelines exactly."""
    try:
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        return None
    r = multivariate_logrank_test(df['time'], df['condition'], df['event'])
    mine = logrank(df, groups)[0]
    return mine, float(r.test_statistic), abs(mine - float(r.test_statistic))


# ── 2-4 ───────────────────────────────────────────────────────────────────────
def fisher_below(runs, env):
    """Pairwise Fisher's exact on below-viability counts, Holm over the pair."""
    sub = runs[runs.environment == env]
    below = {c: int((sub[sub.condition == c].avg_fitness < VIABILITY).sum())
             for c in CONDITIONS}
    n = {c: int((sub.condition == c).sum()) for c in CONDITIONS}
    rows, ps = [], []
    for other in ('learning', 'pure_rl'):
        table = [[below['evolution'], n['evolution'] - below['evolution']],
                 [below[other], n[other] - below[other]]]
        odds, p = stats.fisher_exact(table, alternative='two-sided')
        rows.append({'family': "Fisher's exact, runs below viability",
                     'environment': env,
                     'contrast': f'{SHORT["evolution"]} vs {SHORT[other]}',
                     'detail': f'{below["evolution"]}/{n["evolution"]} vs '
                               f'{below[other]}/{n[other]}',
                     'statistic': odds, 'statistic_name': 'odds ratio',
                     'n': n['evolution'] + n[other], 'p': p})
        ps.append(p)
    for r, a in zip(rows, _holm(ps)):
        r['p_holm'] = a
    return rows


def brown_forsythe(runs, env):
    """Levene centred on the MEDIAN = Brown-Forsythe. Omnibus then pairwise."""
    sub = runs[runs.environment == env]
    vals = {c: sub[sub.condition == c].avg_fitness.to_numpy(float)
            for c in CONDITIONS}
    rows = []
    W, p = stats.levene(*[vals[c] for c in CONDITIONS], center='median')
    rows.append({'family': 'Brown-Forsythe, equality of variance',
                 'environment': env, 'contrast': 'A vs B vs C (omnibus)',
                 'detail': '  '.join(f'sd({SHORT[c]})={vals[c].std(ddof=1):.4f}'
                                     for c in CONDITIONS),
                 'statistic': float(W), 'statistic_name': 'W',
                 'n': sum(len(v) for v in vals.values()), 'p': float(p)})
    # All THREE pairs, not just the two involving B: the omnibus W says only
    # that the spreads are not all equal, and a reader who wants "is A wider
    # than C" must not have to infer it from two tests that do not mention C
    # together. Holm over the three.
    ps, pair_rows = [], []
    for a, b in (('learning', 'evolution'), ('learning', 'pure_rl'),
                 ('evolution', 'pure_rl')):
        W, p = stats.levene(vals[a], vals[b], center='median')
        pair_rows.append({'family': 'Brown-Forsythe, equality of variance',
                          'environment': env,
                          'contrast': f'{SHORT[a]} vs {SHORT[b]}',
                          'detail': f'sd({SHORT[a]})={vals[a].std(ddof=1):.4f} vs '
                                    f'sd({SHORT[b]})={vals[b].std(ddof=1):.4f}',
                          'statistic': float(W), 'statistic_name': 'W',
                          'n': len(vals[a]) + len(vals[b]),
                          'p': float(p)})
        ps.append(p)
    for r, a in zip(pair_rows, _holm(ps)):
        r['p_holm'] = a
    return rows + pair_rows


def _holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj = np.empty(ps.size)
    run = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, (ps.size - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def checkpoint_wilcoxon(path):
    """Read the per-checkpoint paired test the assimilation analysis computed."""
    if not os.path.exists(path):
        return []
    d = pd.read_csv(path)
    rows = []
    for env, g in d.groupby('env'):
        adj = _holm(g['p_wilcoxon'].to_numpy(float))
        for (_i, r), a in zip(g.iterrows(), adj):
            rows.append({'family': 'Wilcoxon signed-rank, inherited genomes',
                         'environment': env,
                         'contrast': f'B vs A at generation {int(r.generation)}',
                         'detail': f'mean difference {r["mean"]:+.2f} raw '
                                   f'f_off, {int(r.n_positive)}/{int(r.n_seeds)} '
                                   f'seeds favour B',
                         'statistic': float(r.w_stat),
                         'statistic_name': 'W',
                         'n': int(r.n_seeds), 'p': float(r.p_wilcoxon),
                         'p_holm': float(a),
                         'n_favouring_b': int(r.n_positive),
                         'mean_difference': float(r['mean'])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conditions-root', default='output/conditions_by_environment')
    ap.add_argument('--assimilation-root', default='output/genome_evaluations')
    ap.add_argument('--out', default='output/stats')
    args = ap.parse_args()

    tt = pd.read_csv(os.path.join(args.conditions_root, 'time_to_threshold.csv'))
    runs = pd.read_csv(os.path.join(args.conditions_root, 'runs.csv'))
    vc = pd.read_csv(os.path.join(args.conditions_root, 'variance_components.csv'))

    rows = []

    # 1. log-rank ------------------------------------------------------------
    chk = check_against_lifelines(tt[tt.environment == 'baseline'], CONDITIONS)
    if chk:
        print(f'[check] unstratified log-rank vs lifelines: '
              f'{chk[0]:.10f} vs {chk[1]:.10f}  (diff {chk[2]:.2e})')
    for env in ('baseline', 'hard'):
        sub = tt[tt.environment == env]
        for name, strata in (('stratified by map seed', 'seed'),
                             ('unstratified', None)):
            chi2, dof, p = logrank(sub, CONDITIONS, strata=strata)
            rows.append({'family': 'Log-rank, time to sustained viability',
                         'environment': env,
                         'contrast': f'A vs B vs C, {name}',
                         'detail': f'{dof} df, censored at generation 1000',
                         'statistic': chi2, 'statistic_name': 'chi-square',
                         'n': len(sub), 'p': p, 'p_holm': np.nan})

    # 2. Fisher ---------------------------------------------------------------
    rows += fisher_below(runs, 'hard')

    # 3. Brown-Forsythe -------------------------------------------------------
    for env in ('baseline', 'hard'):
        rows += brown_forsythe(runs, env)

    # 4. checkpoint Wilcoxon --------------------------------------------------
    rows += checkpoint_wilcoxon(os.path.join(args.assimilation_root,
                                             'assimilation_arm_by_generation.csv'))

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'paper_tests.csv')
    df.to_csv(path, index=False)

    # the design-effect table comes ready-made from the variance components
    eff = vc[['environment', 'condition', 'icc', 'design_effect', 'n_effective']]
    eff.to_csv(os.path.join(args.out, 'effective_n.csv'), index=False)

    pd.set_option('display.width', 200)
    for fam, g in df.groupby('family', sort=False):
        print(f'\n=== {fam} ===')
        for _i, r in g.iterrows():
            ph = ('' if not np.isfinite(r.get('p_holm', np.nan))
                  else f'  p(Holm)={r["p_holm"]:.4g}')
            print(f'  {r.environment:<9}{r.contrast:<34}'
                  f'{r.statistic_name} = {r.statistic:>9.4g}   '
                  f'n={r.n:<5} p={r.p:.4g}{ph}')
            print(f'  {"":<9}{r.detail}')

    print('\n=== effective sample size (design effect = 1 + (m-1) x ICC, m = 10) ===')
    for _i, r in eff.iterrows():
        print(f'  {r.environment:<9}{r.condition:<11}ICC={r.icc:.3f}   '
              f'deff={r.design_effect:.2f}   n_eff={r.n_effective:.0f} of 100')
    print(f'\nwrote {path}\nwrote {os.path.join(args.out, "effective_n.csv")}')


if __name__ == '__main__':
    main()
