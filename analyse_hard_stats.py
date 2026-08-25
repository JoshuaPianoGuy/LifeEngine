"""
analyse_hard_stats.py
======================================================================
Clustered inference for the three-condition hard-environment experiment.

WHY THIS SCRIPT EXISTS
----------------------
The design is 10 map seeds x 10 replicate runs per condition. That is **not**
n = 100 independent runs. Every run sharing a seed also shares its terrain --
the same 50-map pool, the same food layout, the same cave placement -- so the
outcomes within a seed are positively correlated. Treating them as independent
inflates the effective sample size by roughly the number of replicates and
produces p-values that no reviewer should accept.

The fix is to make the seed an explicit stratum / cluster in every test:

  P(viable)          Cochran-Mantel-Haenszel, stratified by seed
                     + a binomial GLMM with a random intercept for seed,
                       which additionally reports how much of the outcome
                       variance the terrain accounts for
  time-to-threshold  Cox proportional hazards with cluster-robust (Lin-Wei
                     sandwich) standard errors clustered on seed
                     + the same model with a gamma frailty on seed

Each of those is reported next to the NAIVE version that ignores clustering, so
the size of the pseudo-replication problem is visible rather than asserted --
the naive/clustered ratio of standard errors is the design effect, and it is
printed for every comparison.

OUTCOMES
--------
viable            A run is viable if its converged average fitness (mean of the
                  last --final-window generations) is at or above the
                  replacement threshold 3.5 / 0.8 = 4.375. This is exactly the
                  threshold analyse_learning_hard_runs.py draws, imported from
                  there rather than restated, so the figures and the tests can
                  never drift apart.

time_to_threshold The first generation whose avg_fitness reaches the threshold.
                  A run that never reaches it is RIGHT-CENSORED at its last
                  generation -- it is not a missing value and it is not an
                  infinite time, and dropping those runs (or coding them as the
                  maximum) is what a survival model exists to avoid.

USAGE
-----
  python analyse_hard_stats.py
  python analyse_hard_stats.py --env hard --reference evolution
  python analyse_hard_stats.py --threshold 5.39      # Stage-9 sensitivity
  python analyse_hard_stats.py --logs-root logs_hard --out output/stats

OUTPUT
------
  output/stats/run_outcomes.csv     one row per run: seed, rep, condition,
                                    converged fitness, viable, time-to-threshold,
                                    censoring indicator
  output/stats/viability_tests.csv  every P(viable) test, naive and clustered
  output/stats/survival_tests.csv   every time-to-threshold model
  output/stats/stats_report.txt     the printed report, saved verbatim
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Discovery, loading and the viability threshold all come from the figure script
# so the tests are defined on exactly the numbers the figures draw.
import analyse_learning_hard_runs as la

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
    HAVE_SM = True
except ImportError:                                          # pragma: no cover
    HAVE_SM = False

try:
    from lifelines import CoxPHFitter
    HAVE_LIFELINES = True
except ImportError:                                          # pragma: no cover
    HAVE_LIFELINES = False

from scipy import stats as sps
from scipy import optimize as spo


# ── The three arms ────────────────────────────────────────────────────────────
# Same definition as analyse_hard_conditions_combined.ARMS: the arm is identified
# by (condition, mode) out of params.json, never by folder name, because pure RL
# is logged with condition == "learning".
ARMS = [
    {'key': 'evolution', 'label': 'Evolution (GA only)',
     'logs_root': 'evolution', 'condition': 'evolution', 'mode': 'standard'},
    {'key': 'learning', 'label': 'Learning (GA + in-life RL)',
     'logs_root': 'learning', 'condition': 'learning', 'mode': 'standard'},
    {'key': 'pure_rl', 'label': 'Pure RL (no GA)',
     'logs_root': 'learning', 'condition': 'learning', 'mode': 'pure_rl'},
    # The chance anchor. Included so every fitness number in the report has a
    # floor to be read against: without it "converged fitness 6.65" is a number
    # with no scale, and a run at 3.5 cannot be called bad rather than merely
    # unlucky. Runs come from --random-floor and are selected by that flag, not
    # by their condition string, which reads "evolution" (see la.discover).
    {'key': 'random_floor', 'label': 'Random policy (chance floor)',
     'logs_root': 'evolution', 'condition': 'evolution', 'mode': 'standard',
     'random_floor': True},
]


# ─────────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_outcomes(logs_root, env_key, seeds, reps, max_gen, final_window, threshold):
    """One row per run: the two outcomes plus the clustering variable.

    converged_fitness  mean avg_fitness over the last `final_window` generations
    viable             converged_fitness >= threshold
    time               first generation whose avg_fitness reaches threshold, or
                       the run's last generation if it never does
    event              1 if the threshold was reached, 0 if right-censored
    """
    env = la.ENVIRONMENTS[env_key]
    rows = []
    for arm in ARMS:
        want = {'condition': arm['condition'], 'mode': arm['mode'],
                'roaming': env['roaming'], 'drain': env['drain'],
                'random_floor': arm.get('random_floor', False)}
        root = os.path.join(logs_root, arm['logs_root'])
        runs = la.discover(root, want, seeds, reps)
        for r in runs:
            df = pd.read_csv(r['gen_csv'])
            df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
            df['avg_fitness'] = pd.to_numeric(df['avg_fitness'], errors='coerce')
            df = df.dropna(subset=['generation', 'avg_fitness'])
            if max_gen:
                df = df[df['generation'] <= max_gen]
            if df.empty:
                continue
            df = df.sort_values('generation')

            converged = float(df['avg_fitness'].tail(final_window).mean())
            reached = df[df['avg_fitness'] >= threshold]
            if len(reached):
                time, event = float(reached['generation'].iloc[0]), 1
            else:
                time, event = float(df['generation'].iloc[-1]), 0

            rows.append({
                'condition': arm['key'],
                'label': arm['label'],
                'seed': int(r['seed']),
                'rep': int(r['rep']),
                'run': r['name'],
                'n_gens': int(df['generation'].iloc[-1]),
                'converged_fitness': converged,
                'viable': int(converged >= threshold),
                'time': time,
                'event': event,
                'ever_reached': int(event),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# P(viable): naive, CMH, GLMM
# ─────────────────────────────────────────────────────────────────────────────

def naive_2x2(sub, a_key, b_key):
    """Fisher's exact and a chi-square on the POOLED table -- the test that
    treats 10 seeds x 10 reps as 100 independent runs. Reported only so the
    clustered results have something to be compared against."""
    a = sub[sub.condition == a_key]
    b = sub[sub.condition == b_key]
    tab = np.array([[int(a.viable.sum()), int((1 - a.viable).sum())],
                    [int(b.viable.sum()), int((1 - b.viable).sum())]])
    odds, p_fisher = sps.fisher_exact(tab)
    # Haldane-Anscombe continuity correction, so a zero cell still yields a
    # finite standard error to compare the clustered one against.
    t = tab + 0.5
    log_or = np.log((t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0]))
    se = np.sqrt((1 / t).sum())
    return {'table': tab, 'odds_ratio': odds, 'p_fisher': p_fisher,
            'log_or': log_or, 'se_log_or': se}


def _nchg_pmf(m1, m2, n1, psi):
    """Fisher noncentral hypergeometric pmf over the [0,0] cell of one 2x2 table
    with row totals (m1, m2) and first-column total n1, at odds ratio psi.

    Written out rather than pulled from a library because the exact CMH below
    needs it evaluated at arbitrary psi during a root search, and because the
    log-space form keeps the binomials from overflowing on 10x10 tables.
    """
    lo = max(0, n1 - m2)
    hi = min(n1, m1)
    k = np.arange(lo, hi + 1)
    from scipy.special import gammaln
    logw = (gammaln(m1 + 1) - gammaln(k + 1) - gammaln(m1 - k + 1) +
            gammaln(m2 + 1) - gammaln(n1 - k + 1) - gammaln(m2 - n1 + k + 1) +
            k * np.log(psi if psi > 0 else 1e-300))
    logw -= logw.max()
    w = np.exp(logw)
    return k, w / w.sum()


def exact_cmh(tables, alpha=0.05):
    """Exact conditional Cochran-Mantel-Haenszel test and confidence interval.

    Conditioning on both margins of every stratum, the sum S of the [0,0] cells
    is distributed as the convolution of the strata's noncentral hypergeometrics.
    That gives an exact p-value and, by inverting the test on psi, an exact CI --
    which is what makes this usable when a stratum is completely separated and
    the Mantel-Haenszel odds ratio is infinite. The asymptotic RBG interval
    returns nan there; this one returns a finite lower bound and an infinite
    upper bound, which is the honest statement.

    `tables` is a list of 2x2 arrays laid out [[a, b], [c, d]] with the exposure
    of interest in row 0 and the outcome of interest in column 0.
    """
    supports = []
    for t in tables:
        m1 = int(t[0].sum())          # row-0 total
        m2 = int(t[1].sum())          # row-1 total
        n1 = int(t[:, 0].sum())       # column-0 total
        supports.append((m1, m2, n1))
    s_obs = int(sum(int(t[0, 0]) for t in tables))

    def dist(psi):
        """Distribution of S at odds ratio psi, as (offset, pmf array)."""
        off, pmf = 0, np.array([1.0])
        for (m1, m2, n1) in supports:
            k, w = _nchg_pmf(m1, m2, n1, psi)
            pmf = np.convolve(pmf, w)
            off += int(k[0])
        return off, pmf

    def p_ge(psi):                     # P(S >= s_obs)
        off, pmf = dist(psi)
        i = s_obs - off
        if i <= 0:
            return 1.0
        if i >= len(pmf):
            return 0.0
        return float(pmf[i:].sum())

    def p_le(psi):                     # P(S <= s_obs)
        off, pmf = dist(psi)
        i = s_obs - off
        if i < 0:
            return 0.0
        if i >= len(pmf) - 1:
            return 1.0
        return float(pmf[:i + 1].sum())

    # Two-sided exact p at the null psi = 1, by doubling the smaller tail
    # (Cox's convention -- monotone in psi, so the CI it inverts to is coherent).
    p_two = min(1.0, 2 * min(p_ge(1.0), p_le(1.0)))

    # Conditional MLE of the common odds ratio: the psi at which E[S] = s_obs.
    def mean_at(psi):
        off, pmf = dist(psi)
        return float((np.arange(len(pmf)) + off) @ pmf)

    lo_b, hi_b = 1e-8, 1e8
    if mean_at(hi_b) < s_obs:
        cmle = np.inf
    elif mean_at(lo_b) > s_obs:
        cmle = 0.0
    else:
        cmle = spo.brentq(lambda lp: mean_at(np.exp(lp)) - s_obs,
                          np.log(lo_b), np.log(hi_b), xtol=1e-6)
        cmle = float(np.exp(cmle))

    # CI by inverting the exact test: the lower bound is the psi at which
    # P(S >= s_obs) = alpha/2, the upper the psi at which P(S <= s_obs) = alpha/2.
    def invert(fn, target, lo, hi):
        f_lo, f_hi = fn(lo) - target, fn(hi) - target
        if f_lo * f_hi > 0:
            return None
        return float(np.exp(spo.brentq(lambda lp: fn(np.exp(lp)) - target,
                                       np.log(lo), np.log(hi), xtol=1e-6)))

    ci_lo = invert(p_ge, alpha / 2, 1e-8, 1e8)
    ci_hi = invert(p_le, alpha / 2, 1e-8, 1e8)
    return {'p_exact': p_two, 'or_cmle': cmle,
            'ci_lo': 0.0 if ci_lo is None else ci_lo,
            'ci_hi': np.inf if ci_hi is None else ci_hi,
            's_obs': s_obs, 'k': len(tables)}


def cmh_test(sub, a_key, b_key):
    """Cochran-Mantel-Haenszel: the 2x2 table is built SEPARATELY WITHIN EACH
    SEED and the strata are combined. Terrain is conditioned out rather than
    averaged over, so a seed where both arms do well contributes no spurious
    association. Exact, distribution-free, and the easiest of the two options to
    defend in a write-up.

    Strata with no variation in the outcome (all viable, or none) carry no
    information about the association and are dropped by the statistic itself;
    they are counted and reported so the drop is visible.
    """
    tables, used, dropped = [], [], []
    for seed, g in sub.groupby('seed'):
        a = g[g.condition == a_key]
        b = g[g.condition == b_key]
        if a.empty or b.empty:
            dropped.append((seed, 'arm missing'))
            continue
        tab = np.array([[int(a.viable.sum()), int((1 - a.viable).sum())],
                        [int(b.viable.sum()), int((1 - b.viable).sum())]])
        if tab.sum(axis=0).min() == 0 or tab.sum(axis=1).min() == 0:
            dropped.append((seed, 'no outcome variation'))
            continue
        tables.append(tab)
        used.append(seed)

    if not tables:
        return None

    # StratifiedTable wants (2, 2, K).
    arr = np.dstack(tables)
    st = StratifiedTable(arr)
    res = st.test_null_odds(correction=True)
    or_mh = st.oddsratio_pooled
    ci = st.oddsratio_pooled_confint()
    # Robins-Breslow-Greenland SE of log(OR_MH) -- what statsmodels' CI is built
    # from; recovered here so the design effect can be computed against the naive SE.
    lo, hi = st.logodds_pooled_confint()
    se = (hi - lo) / (2 * 1.959963984540054)
    homog = st.test_equal_odds(adjust=True)      # Breslow-Day (Tarone-adjusted)
    # Complete separation: one arm has no failures anywhere, so the MH odds ratio
    # is infinite and the RBG interval is undefined. The CMH test statistic and
    # its p-value are still valid -- it is the interval that has no upper end --
    # so the exact conditional version is run alongside to supply a finite lower
    # bound instead of reporting nan.
    separated = (not np.isfinite(or_mh)) or or_mh == 0 or (not np.isfinite(se))
    exact = exact_cmh(tables) if separated else None

    return {'k_strata': len(tables), 'seeds_used': used, 'seeds_dropped': dropped,
            'or_mh': or_mh, 'ci': ci, 'log_or': st.logodds_pooled, 'se_log_or': se,
            'stat': res.statistic, 'p': res.pvalue,
            'bd_stat': homog.statistic, 'bd_p': homog.pvalue,
            'separated': separated, 'exact': exact, 'tables': tables}


def glmm_viability(sub, reference):
    """Binomial GLMM: viable ~ condition + (1 | seed).

    Fitted by Laplace approximation via statsmodels' BinomialBayesMixedGLM is not
    what is wanted here (it is variational/Bayesian); instead the model is fitted
    as a GEE with an exchangeable working correlation clustered on seed, which
    gives population-averaged log-odds with cluster-robust standard errors, AND
    separately the variance component is estimated by fitting the same random
    intercept with BinomialBayesMixedGLM so the terrain variance can be reported.

    Two different quantities, deliberately labelled as such:
      GEE           population-averaged effect, robust to the within-seed
                    correlation structure -- the inferential result
      variance      sigma^2_seed on the logit scale, and the latent-scale ICC
                    sigma^2 / (sigma^2 + pi^2/3) -- the descriptive result,
                    "terrain accounts for X% of outcome variance"
    """
    d = sub.copy()
    d['condition'] = pd.Categorical(d['condition'],
                                    categories=[reference] + [c for c in d.condition.unique()
                                                              if c != reference])
    out = {}

    # ── population-averaged, cluster-robust ──────────────────────────────────
    try:
        gee = smf.gee('viable ~ C(condition)', groups='seed', data=d,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        out['gee'] = {'params': gee.params, 'bse': gee.bse, 'pvalues': gee.pvalues,
                      'alpha': float(getattr(gee.cov_struct, 'dep_params', np.nan)),
                      'summary': str(gee.summary())}
    except Exception as e:
        out['gee_error'] = str(e)

    # ── the same fit ignoring clustering, for the design effect ──────────────
    try:
        glm = smf.glm('viable ~ C(condition)', data=d,
                      family=sm.families.Binomial()).fit()
        out['glm'] = {'params': glm.params, 'bse': glm.bse, 'pvalues': glm.pvalues}
    except Exception as e:
        out['glm_error'] = str(e)

    # ── the variance component ───────────────────────────────────────────────
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        vcf = {'seed': '0 + C(seed)'}
        mdl = BinomialBayesMixedGLM.from_formula('viable ~ C(condition)', vcf, d)
        fit = mdl.fit_vb(verbose=False)
        # vcp_mean is log(sd) of the variance component under this parameterisation.
        log_sd = float(fit.vcp_mean[0])
        sd = float(np.exp(log_sd))
        var = sd ** 2
        icc = var / (var + np.pi ** 2 / 3)
        out['vc'] = {'sd_seed': sd, 'var_seed': var, 'icc_latent': icc,
                     'log_sd': log_sd, 'log_sd_sd': float(fit.vcp_sd[0])}
    except Exception as e:
        out['vc_error'] = str(e)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Time-to-threshold: Cox with clustered SEs
# ─────────────────────────────────────────────────────────────────────────────

def cox_models(sub, reference):
    """Cox proportional hazards on time-to-threshold, three ways.

      naive     no clustering -- the standard errors a run-as-independent
                analysis would report
      robust    Lin-Wei cluster-robust sandwich standard errors clustered on
                seed; the point estimates are unchanged, the standard errors are
                corrected for the within-seed correlation. This is the direct
                analogue of the CMH for a time-to-event outcome.
      frailty   a stratified fit by seed, which conditions the baseline hazard on
                terrain entirely rather than modelling it as a random effect --
                a stronger, assumption-free alternative when the number of seeds
                is small (lifelines has no gamma-frailty Cox, and 10 clusters is
                thin for estimating one anyway).

    Censoring: runs that never reach the threshold enter with event = 0 at their
    last generation. Efron's method handles the heavy ties that a
    generation-resolution time axis produces.
    """
    d = sub.copy()
    d['condition'] = pd.Categorical(d['condition'],
                                    categories=[reference] + [c for c in ARMS_ORDER
                                                              if c != reference and
                                                              c in set(d.condition)])
    X = pd.get_dummies(d[['condition']], drop_first=True, dtype=float)
    X['time'] = d['time'].values
    X['event'] = d['event'].values
    X['seed'] = d['seed'].values

    out = {}
    covariates = [c for c in X.columns if c.startswith('condition_')]
    if not covariates:
        return out

    try:
        naive = CoxPHFitter()
        naive.fit(X.drop(columns=['seed']), 'time', 'event')
        out['naive'] = naive.summary
    except Exception as e:
        out['naive_error'] = str(e)

    try:
        robust = CoxPHFitter()
        robust.fit(X, 'time', 'event', cluster_col='seed', robust=True)
        out['robust'] = robust.summary
        out['robust_fit'] = robust
    except Exception as e:
        out['robust_error'] = str(e)

    try:
        strat = CoxPHFitter()
        strat.fit(X, 'time', 'event', strata=['seed'])
        out['stratified'] = strat.summary
    except Exception as e:
        out['stratified_error'] = str(e)

    return out


ARMS_ORDER = [a['key'] for a in ARMS]


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    """Print to stdout and collect for the saved report."""
    def __init__(self):
        self.lines = []

    def __call__(self, s=''):
        print(s)
        self.lines.append(str(s))

    def text(self):
        return '\n'.join(self.lines) + '\n'


def describe_design(out, df, say):
    say('=' * 78)
    say('DESIGN')
    say('=' * 78)
    counts = df.groupby(['condition', 'seed']).size().unstack(fill_value=0)
    say(f'  runs discovered      : {len(df)}')
    say(f'  conditions           : {", ".join(sorted(df.condition.unique()))}')
    say(f'  map seeds (clusters) : {df.seed.nunique()}  -> '
        f'{[int(x) for x in sorted(df.seed.unique())]}')
    say(f'  replicates per cell  : min {counts.values.min()}, max {counts.values.max()}')
    say('')
    say('  Runs per condition x seed:')
    say('  ' + counts.to_string().replace('\n', '\n  '))
    say('')
    say('  The unit of INDEPENDENT replication is the seed, not the run. Every')
    say('  test below that ignores that is labelled NAIVE and is reported only')
    say('  as the comparison point.')
    say('')


def describe_outcomes(df, threshold, final_window, say):
    say('=' * 78)
    say('OUTCOMES')
    say('=' * 78)
    say(f'  viability threshold  : {threshold:.4f}  (replacement fitness 3.5 / 0.8)')
    say(f'  converged fitness    : mean of the last {final_window} generations')
    say('  time-to-threshold    : first generation with avg_fitness >= threshold;')
    say('                         runs that never reach it are RIGHT-CENSORED at')
    say('                         their last generation, not dropped.')
    say('')
    for key in ARMS_ORDER:
        s = df[df.condition == key]
        if s.empty:
            continue
        per_seed = s.groupby('seed').viable.mean()
        say(f'  {key:<10}  n={len(s):>3}  viable {int(s.viable.sum()):>3}/{len(s):<3} '
            f'({s.viable.mean():.3f})   reached threshold {int(s.event.sum())}/{len(s)}')
        say(f'              per-seed P(viable): '
            f'{", ".join(f"{k}:{v:.1f}" for k, v in per_seed.items())}')
        say(f'              between-seed sd of P(viable) = {per_seed.std(ddof=1):.4f}')
    say('')


def degenerate_arms(df, col, say=None):
    """Arms whose outcome has no variation at all (all 0 or all 1).

    A logistic model cannot put a finite coefficient on such an arm -- the
    likelihood is still increasing at the edge of the parameter space -- so any
    beta, SE or p it prints for that arm is an artefact of where the optimiser
    stopped, not an estimate. They are named up front so nothing downstream is
    read as a number.
    """
    out = {}
    for key, g in df.groupby('condition', observed=True):
        v = g[col]
        if v.nunique() == 1:
            out[str(key)] = f'{"all" if v.iloc[0] else "no"} runs ({len(g)}) — {col}={int(v.iloc[0])}'
    return out


def report_viability(df, reference, say, rows_out):
    say('=' * 78)
    say('P(VIABLE) — CLUSTERED INFERENCE')
    say('=' * 78)

    deg = degenerate_arms(df, 'viable')
    if deg:
        say('  DEGENERATE ARMS (no variation in the outcome):')
        for k, why in deg.items():
            say(f'    {k:<14} {why}')
        say('  A logistic model has no finite MLE for these, so their coefficients,')
        say('  standard errors and p-values below are optimiser artefacts, not')
        say('  estimates. The CMH TEST statistic is still valid; read the effect size')
        say('  off the exact conditional interval, or off the linear mixed model on')
        say('  converged fitness, which does not degenerate.')
        say('')

    others = [k for k in ARMS_ORDER if k != reference and k in set(df.condition)]
    for other in others:
        sub = df[df.condition.isin([reference, other])]
        say(f'\n--- {other} vs {reference} (reference) ---')

        nv = naive_2x2(sub, other, reference)
        say(f'  NAIVE (runs treated as independent, n={len(sub)})')
        say(f'    2x2 pooled table [[viable, not], ...] = {nv["table"].tolist()}')
        say(f'    Fisher exact  OR = {nv["odds_ratio"]:.4g}   p = {nv["p_fisher"]:.5g}')
        say(f'    log-OR = {nv["log_or"]:+.4f}   SE = {nv["se_log_or"]:.4f}')

        cm = cmh_test(sub, other, reference)
        if cm is None:
            say('  CMH: no informative strata (every seed has a constant outcome).')
        else:
            say(f'  COCHRAN-MANTEL-HAENSZEL (stratified by seed, k={cm["k_strata"]})')
            if cm['separated']:
                ex = cm['exact']
                # Which way it separated matters for how the interval reads: an
                # arm with no failures pushes the odds ratio to +inf (no upper
                # bound), an arm with no successes pushes it to 0 (no lower one).
                if cm['or_mh'] == 0:
                    which = (f'    OR_MH = 0  — COMPLETE SEPARATION: {other} has no VIABLE runs\n'
                             '      in any stratum, so the odds ratio has no finite lower bound\n'
                             '      and the Robins-Breslow-Greenland interval is undefined.')
                else:
                    which = (f'    OR_MH = inf  — COMPLETE SEPARATION: {other} has no FAILED runs\n'
                             '      in any stratum, so the odds ratio has no finite upper bound\n'
                             '      and the Robins-Breslow-Greenland interval is undefined.')
                for line in which.split('\n'):
                    say(line)
                say('      The CMH TEST is unaffected; the interval is replaced by the')
                say('      exact conditional one below.')
                fmt = lambda v: 'inf' if not np.isfinite(v) else format(v, '.4g')
                say(f'    exact conditional OR = {fmt(ex["or_cmle"])}'
                    f'   95% CI [{fmt(ex["ci_lo"])}, {fmt(ex["ci_hi"])}]')
                say(f'    exact conditional p = {ex["p_exact"]:.5g}   '
                    f'(k={ex["k"]} informative strata)')
                # statsmodels underflows a very small p to exactly 0; recover it
                # from the statistic so the report never prints "p = 0", which is
                # not a p-value anyone should quote.
                p_cmh = cm['p'] if cm['p'] > 0 else sps.chi2.sf(cm['stat'], 1)
                p_txt = f'{p_cmh:.5g}' if p_cmh > 0 else '< 1e-308'
                say(f'    CMH chi2 = {cm["stat"]:.4f}   p = {p_txt}   <- QUOTE THIS')
            else:
                say(f'    OR_MH = {cm["or_mh"]:.4g}   95% CI '
                    f'[{cm["ci"][0]:.4g}, {cm["ci"][1]:.4g}]')
                say(f'    log-OR = {cm["log_or"]:+.4f}   SE(RBG) = {cm["se_log_or"]:.4f}')
                p_cmh = cm['p'] if cm['p'] > 0 else sps.chi2.sf(cm['stat'], 1)
                say(f'    CMH chi2 = {cm["stat"]:.4f}   p = {p_cmh:.5g}')
            if np.isfinite(cm['bd_p']):
                say(f'    Breslow-Day homogeneity of OR across seeds: '
                    f'chi2 = {cm["bd_stat"]:.4f}, p = {cm["bd_p"]:.5g}'
                    + ('   (heterogeneous — a single pooled OR is a summary, not a '
                       'description)' if cm['bd_p'] < 0.05 else ''))
            else:
                say('    Breslow-Day homogeneity: undefined under separation.')
            if cm['seeds_dropped']:
                say('    strata carrying no information (dropped by the statistic): '
                    + ', '.join(f'{s}({why})' for s, why in cm['seeds_dropped']))
            if np.isfinite(cm['se_log_or']) and np.isfinite(nv['se_log_or']):
                de = (cm['se_log_or'] / nv['se_log_or']) ** 2
                say(f'    DESIGN EFFECT (var ratio clustered/naive) = {de:.2f}'
                    f'  -> effective n ~ {len(sub) / de:.0f} of {len(sub)} runs')
            else:
                de = np.nan
                say('    DESIGN EFFECT: not computable under separation (both standard '
                    'errors are undefined). See the time-to-threshold models, where the '
                    'same clustering is quantified on an outcome that does not separate.')
            rows_out.append({
                'comparison': f'{other}_vs_{reference}', 'test': 'CMH',
                'estimate_or': cm['or_mh'], 'ci_lo': cm['ci'][0], 'ci_hi': cm['ci'][1],
                'se_log_or': cm['se_log_or'],
                'p': cm['p'] if cm['p'] > 0 else sps.chi2.sf(cm['stat'], 1),
                'k_strata': cm['k_strata'],
                'design_effect': de,
            })
        rows_out.append({
            'comparison': f'{other}_vs_{reference}', 'test': 'Fisher (NAIVE)',
            'estimate_or': nv['odds_ratio'], 'ci_lo': np.nan, 'ci_hi': np.nan,
            'se_log_or': nv['se_log_or'], 'p': nv['p_fisher'], 'k_strata': np.nan,
            'design_effect': np.nan,
        })

    # One GLMM over all three arms at once rather than pairwise, so the variance
    # component is estimated from the whole design.
    say('\n--- Binomial mixed model over all arms: viable ~ condition + (1 | seed) ---')
    g = glmm_viability(df, reference)
    if 'gee' in g:
        say('  GEE, exchangeable working correlation, clustered on seed')
        say('  (population-averaged log-odds; SEs are cluster-robust)')
        p = g['gee']['params']
        for name in p.index:
            # A coefficient this large is the signature of complete separation,
            # not an effect size: the likelihood is still increasing at the edge
            # of the parameter space. Say so rather than printing it straight.
            flag = '   <- SEPARATED: no finite MLE, read this as "always viable", ' \
                   'not as an odds ratio of e^beta' if abs(p[name]) > 10 else ''
            say(f'    {name:<28} beta = {p[name]:+.4f}  SE = {g["gee"]["bse"][name]:.4f}  '
                f'p = {g["gee"]["pvalues"][name]:.5g}{flag}')
        say(f'    within-seed correlation (working alpha) = {g["gee"]["alpha"]:.4f}')
        if 'glm' in g:
            say('  Same model WITHOUT clustering (NAIVE), for the design effect:')
            for name in g['glm']['params'].index:
                naive_se = g['glm']['bse'][name]
                rob_se = g['gee']['bse'].get(name, np.nan)
                de = (rob_se / naive_se) ** 2 if naive_se else np.nan
                say(f'    {name:<28} SE naive = {naive_se:.4f}  SE clustered = {rob_se:.4f}'
                    f'   design effect = {de:.2f}')
    else:
        say(f'  GEE failed: {g.get("gee_error")}')

    if 'vc' in g:
        v = g['vc']
        say('  Variance component for seed (terrain), logit scale:')
        say(f'    sigma_seed = {v["sd_seed"]:.4f}   sigma^2_seed = {v["var_seed"]:.4f}')
        say(f'    latent-scale ICC = sigma^2 / (sigma^2 + pi^2/3) = {v["icc_latent"]:.4f}')
        say(f'    -> terrain accounts for {100 * v["icc_latent"]:.1f}% of the variance in')
        say('       the latent viability propensity. That is a result in its own right:')
        say('       it says how much of whether a run survives is decided by the map it')
        say('       was dealt rather than by the adaptation mechanism.')
    else:
        say(f'  variance component unavailable: {g.get("vc_error")}')
    say('')
    return g


def report_continuous(df, reference, say, rows_out):
    """Converged fitness as a CONTINUOUS outcome, with a random intercept for seed.

    P(viable) throws away almost everything the runs measured -- it reduces a
    fitness trajectory to one bit, and at ~90-100% viability that bit is nearly
    constant, which is what produced the complete separation above. The same
    clustering question asked of the underlying fitness has no separation
    problem, far more power, and gives the terrain variance component directly
    as a proportion of variance rather than through a latent-scale
    approximation.

    Reported as:
      LMM      converged_fitness ~ condition + (1 | seed), REML
      ICC      sigma^2_seed / (sigma^2_seed + sigma^2_residual) -- the fraction
               of variance in converged fitness attributable to terrain
      OLS      the same fixed effects with no random effect, for the design effect
    """
    say('=' * 78)
    say('CONVERGED FITNESS — LINEAR MIXED MODEL WITH A RANDOM INTERCEPT FOR SEED')
    say('=' * 78)
    say('  Same clustering question on the continuous outcome the viability bit was')
    say('  thresholded from. No separation, and the variance component is a plain')
    say('  proportion of variance rather than a latent-scale one.')
    say('')
    d = df.copy()
    cats = [reference] + [c for c in ARMS_ORDER if c != reference and c in set(d.condition)]
    d['condition'] = pd.Categorical(d['condition'], categories=cats)

    for key in cats:
        s_ = d[d.condition == key]
        say(f'  {key:<10} converged fitness: mean {s_.converged_fitness.mean():.4f}  '
            f'sd {s_.converged_fitness.std(ddof=1):.4f}  '
            f'[{s_.converged_fitness.min():.3f}, {s_.converged_fitness.max():.3f}]')
    say('')

    try:
        md = smf.mixedlm('converged_fitness ~ C(condition)', d, groups=d['seed'])
        fit = md.fit(reml=True)
        var_seed = float(fit.cov_re.iloc[0, 0])
        var_res = float(fit.scale)
        icc = var_seed / (var_seed + var_res)
        say('  LMM  converged_fitness ~ condition + (1 | seed)   [REML]')
        for name in fit.params.index:
            if name.startswith('Group') or name == 'seed Var':
                continue
            say(f'    {name:<28} beta = {fit.params[name]:+.4f}  SE = {fit.bse[name]:.4f}  '
                f'p = {fit.pvalues[name]:.5g}')
            rows_out.append({'model': 'LMM', 'term': name, 'estimate': fit.params[name],
                             'se': fit.bse[name], 'p': fit.pvalues[name]})
        say(f'    sigma^2_seed = {var_seed:.5f}   sigma^2_residual = {var_res:.5f}')
        say(f'    ICC = sigma^2_seed / (sigma^2_seed + sigma^2_resid) = {icc:.4f}')
        say(f'    -> TERRAIN ACCOUNTS FOR {100 * icc:.1f}% OF THE VARIANCE IN CONVERGED')
        say('       FITNESS. This is the number to report as the seed variance component:')
        say('       it is what justifies clustering, and it is a finding in itself.')
        rows_out.append({'model': 'LMM', 'term': 'ICC_seed', 'estimate': icc,
                         'se': np.nan, 'p': np.nan})
    except Exception as e:
        say(f'  LMM failed: {e}')
        return

    try:
        ols = smf.ols('converged_fitness ~ C(condition)', data=d).fit()
        say('')
        say('  Same fixed effects with NO random effect (NAIVE), for the design effect:')
        for name in ols.params.index:
            if name not in fit.bse.index:
                continue
            de = (fit.bse[name] / ols.bse[name]) ** 2
            say(f'    {name:<28} SE naive = {ols.bse[name]:.4f}  SE mixed = '
                f'{fit.bse[name]:.4f}   design effect = {de:.2f}')
        say('')
        say('  A design effect near 1 here does NOT mean the design is free of')
        say('  pseudo-replication -- it means that for THIS outcome the between-seed')
        say('  variance is small, so clustering costs almost nothing. The correction')
        say('  still has to be applied and reported; what varies is how much it moves')
        say('  the answer. Compare with the time-to-threshold models below, where the')
        say('  same 10 seeds produce design effects well above 1.')
    except Exception as e:
        say(f'  OLS comparison failed: {e}')
    say('')


def report_survival(df, reference, say, rows_out):
    say('=' * 78)
    say('TIME TO THRESHOLD — COX WITH CLUSTER-ROBUST STANDARD ERRORS')
    say('=' * 78)
    n_cens = int((1 - df.event).sum())
    say(f'  {len(df)} runs, {int(df.event.sum())} reached the threshold, '
        f'{n_cens} right-censored ({n_cens / len(df):.1%})')
    say(f'  reference level: {reference}')
    per_arm = df.groupby('condition', observed=True).agg(
        n=('event', 'size'), events=('event', 'sum'))
    for key, row in per_arm.iterrows():
        say(f'    {str(key):<14} {int(row.events):>3}/{int(row.n):<3} reached '
            f'({1 - row.events / row.n:.0%} censored)')
    no_event = [str(k) for k, r in per_arm.iterrows() if r.events == 0]
    if no_event:
        say('')
        say(f'  MONOTONE LIKELIHOOD: {", ".join(no_event)} had NO events at all.')
        say('  Cox cannot estimate a finite coefficient for an arm that never')
        say('  experiences the event — the partial likelihood is maximised by driving')
        say('  the hazard ratio to 0 — so its HR (near zero, with a huge standard')
        say('  error) and its p-value are numerical artefacts and are suppressed below.')
        say('  The correct statement is the descriptive one: 0 of N runs ever reached')
        say('  the threshold. Report that, not a hazard ratio.')
    say('')

    if not HAVE_LIFELINES:
        say('  lifelines is not installed — skipping. pip install lifelines')
        return

    m = cox_models(df, reference)
    for name, title in [('naive', 'NAIVE (runs independent)'),
                        ('robust', 'CLUSTER-ROBUST on seed (Lin-Wei sandwich)'),
                        ('stratified', 'STRATIFIED by seed (baseline hazard per terrain)')]:
        if name not in m:
            say(f'  {title}: failed — {m.get(name + "_error")}')
            continue
        s = m[name]
        say(f'  {title}')
        for cov in s.index:
            hr = s.loc[cov, 'exp(coef)']
            se = s.loc[cov, 'se(coef)']
            p = s.loc[cov, 'p']
            lo = s.loc[cov, 'exp(coef) lower 95%']
            hi = s.loc[cov, 'exp(coef) upper 95%']
            arm = cov.replace('condition_', '')
            if arm in no_event:
                n = int(per_arm.loc[arm, 'n'])
                say(f'    {cov:<30} SUPPRESSED — 0/{n} runs reached the threshold; '
                    'no finite HR exists')
                rows_out.append({'model': name, 'covariate': cov, 'HR': np.nan,
                                 'ci_lo': np.nan, 'ci_hi': np.nan, 'se_coef': np.nan,
                                 'p': np.nan, 'note': f'monotone likelihood: 0/{n} events'})
                continue
            say(f'    {cov:<30} HR = {hr:.4g} [{lo:.4g}, {hi:.4g}]  '
                f'SE(coef) = {se:.4f}  p = {p:.5g}')
            rows_out.append({'model': name, 'covariate': cov, 'HR': hr,
                             'ci_lo': lo, 'ci_hi': hi, 'se_coef': se, 'p': p})
        say('')

    if 'naive' in m and 'robust' in m:
        say('  DESIGN EFFECT per covariate (variance ratio robust/naive):')
        for cov in m['robust'].index:
            if cov not in m['naive'].index:
                continue
            if cov.replace('condition_', '') in no_event:
                say(f'    {cov:<30} n/a (no events — both standard errors are artefacts)')
                continue
            se_r = m['robust'].loc[cov, 'se(coef)']
            se_n = m['naive'].loc[cov, 'se(coef)']
            de = (se_r / se_n) ** 2
            # A design effect below 1 means the robust sandwich came out SMALLER
            # than the model-based variance. That is possible, but at these
            # cluster counts it is more often a sign that 10 clusters is too few
            # for the sandwich to be stable than a real efficiency gain -- so it
            # is reported as such rather than as "effective n > n", which is not
            # a thing.
            if de < 1:
                say(f'    {cov:<30} {de:.2f}   <- BELOW 1: the sandwich estimate came '
                    'out smaller than the')
                say(f'    {"":<30}          model-based one. With only '
                    f'{df.seed.nunique()} clusters the sandwich is')
                say(f'    {"":<30}          noisy; treat this as "clustering costs '
                    'nothing here",')
                say(f'    {"":<30}          not as a gain in effective sample size.')
            else:
                say(f'    {cov:<30} {de:.2f}   -> effective n ~ {len(df) / de:.0f} '
                    f'of {len(df)} runs')
        say('')
        say('  A design effect above 1 is the pseudo-replication, quantified: it is the')
        say('  factor by which treating runs as independent understates the variance.')
    say('')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Seed-clustered inference for the hard-environment trio.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--logs-root', default='logs_hard',
                    help='directory holding evolution/ and learning/')
    ap.add_argument('--env', default='hard', choices=sorted(la.ENVIRONMENTS),
                    help='environment preset to filter runs by')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='restrict to these map seeds')
    ap.add_argument('--reps', type=int, default=None,
                    help='keep only the first N replicates of each seed')
    ap.add_argument('--max-gen', type=int, default=None,
                    help='truncate every run at this generation')
    ap.add_argument('--final-window', type=int, default=50,
                    help='generations averaged for the converged fitness')
    ap.add_argument('--threshold', type=float, default=la.VIABILITY_THRESHOLD,
                    help='viability / time-to-threshold fitness level')
    ap.add_argument('--reference', default='evolution', choices=ARMS_ORDER,
                    help='condition used as the reference level')
    ap.add_argument('--out', default=os.path.join('output', 'stats'),
                    help='output directory')
    args = ap.parse_args()

    if not HAVE_SM:
        print('ERROR: statsmodels is required.  pip install statsmodels', file=sys.stderr)
        return 2

    df = build_outcomes(args.logs_root, args.env, args.seeds, args.reps,
                        args.max_gen, args.final_window, args.threshold)
    if df.empty:
        print(f'No runs found under {args.logs_root} for env={args.env}.', file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    say = Tee()

    say(f'analyse_hard_stats.py — {la.ENVIRONMENTS[args.env]["desc"]}')
    say(f'logs root: {args.logs_root}')
    say('')
    describe_design(args.out, df, say)
    describe_outcomes(df, args.threshold, args.final_window, say)

    viab_rows, surv_rows, cont_rows = [], [], []
    report_viability(df, args.reference, say, viab_rows)
    report_continuous(df, args.reference, say, cont_rows)
    report_survival(df, args.reference, say, surv_rows)

    say('=' * 78)
    say('WHAT TO REPORT')
    say('=' * 78)
    say('  For P(viable): quote the CMH odds ratio, its 95% CI and p, and state the')
    say('  stratification ("stratified by map seed, k strata"). Quote the ICC from the')
    say('  mixed model beside it as the terrain variance component. Do NOT quote the')
    say('  pooled Fisher result except as the thing being corrected.')
    say('')
    say('  For time-to-threshold: quote the Cox hazard ratio with the CLUSTER-ROBUST')
    say('  CI and p, and say the standard errors are clustered on map seed. Report the')
    say('  censoring fraction — runs that never reach the threshold are informative and')
    say('  the reader needs to know how many there were.')
    say('')
    say('  For converged fitness: quote the mixed-model fixed effects and the ICC. The')
    say('  continuous outcome is the one worth leading with — thresholding a fitness')
    say('  trajectory into one viable/not bit discards most of what the run measured,')
    say('  and at these viability rates it is what forces the separation seen above.')
    say('')
    say('  In every case, state the clustering in the Methods sentence, not just in a')
    say('  table footnote: "n = 100 runs per condition, 10 replicates on each of 10 map')
    say('  seeds; all tests treat the seed as the unit of independent replication."')
    say('')

    df.to_csv(os.path.join(args.out, 'run_outcomes.csv'), index=False)
    if viab_rows:
        pd.DataFrame(viab_rows).to_csv(os.path.join(args.out, 'viability_tests.csv'),
                                       index=False)
    if surv_rows:
        pd.DataFrame(surv_rows).to_csv(os.path.join(args.out, 'survival_tests.csv'),
                                       index=False)
    if cont_rows:
        pd.DataFrame(cont_rows).to_csv(os.path.join(args.out, 'fitness_mixed_model.csv'),
                                       index=False)
    with open(os.path.join(args.out, 'stats_report.txt'), 'w') as f:
        f.write(say.text())
    print(f'wrote {args.out}/run_outcomes.csv, viability_tests.csv, '
          f'survival_tests.csv, stats_report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
