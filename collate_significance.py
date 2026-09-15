"""
collate_significance.py
======================================================================
Every significance test in the project, in one table, keyed to the figure it
supports.

The tests themselves live with the analyses that produce them — condition and
environment contrasts in analyse_conditions_by_environment.py, the assimilation
signature in assimilation/analyse_assimilation.py, the landscape tests in
landscape/{plot_slices,slice_metrics}.py. That is the right place for them: each
sits beside the data it was computed from and is re-derived whenever that
analysis re-runs. What was missing is a single sheet answering "is this figure's
difference significant, and by what test", without opening six CSVs.

This script only READS those outputs and reshapes them. It computes no statistic
of its own, so it can never disagree with the analysis it summarises — if a
number here looks wrong, the fix belongs upstream and this file will follow.

    python collate_significance.py

Writes <out>/significance_summary.csv, one row per test:

    figure          the figure or claim the test backs
    comparison      what was compared
    unit            the unit of replication (always the seed, or the slice)
    n               how many of that unit
    estimate        the effect, in the units of `metric`
    ci_lo / ci_hi   95% interval where the source provides one
    effect_size     Cohen's dz / d, or a rank statistic where that is what fits
    p / p_adjusted  raw and multiplicity-adjusted where the source adjusts
    *_norm          estimate and CI restated on the figures' [0, 1] axis, for the
                    rows carried in raw fitness units (see add_normalised)
    p_floor         the SMALLEST p that design could return (see below)
    verdict         significant / not significant / untestable at this n
    source          the file the row was read from

WHY p_floor IS A COLUMN
-----------------------
Several comparisons here are rank tests on tiny samples, where the p-value is
bounded below by the number of possible orderings. A 3-vs-3 Mann-Whitney cannot
return anything under 0.100 no matter how cleanly the groups separate, and a
sign test on 6 slices bottoms out at 0.031. A reader who does not know that will
read p = 0.100 as "no effect" when it is in fact "this design cannot answer the
question". Every row therefore carries the floor, and `verdict` says
"untestable at this n" rather than "not significant" when the two coincide.

VERDICTS ARE AT alpha = 0.05 ON THE ADJUSTED p WHERE ONE EXISTS. No row is
adjusted across analyses: the families are the six condition contrasts, the
three environment contrasts, and so on, each corrected within itself. Pooling
every test in the project into one correction would penalise each figure for
questions the others asked.
"""

import argparse
import os

import numpy as np
import pandas as pd

SRC = {
    'conditions': 'output/conditions_by_environment',
    'assimilation': 'output/genome_evaluations',
    'landscape': 'output/fitness_landscapes',
}


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _verdict(p, floor=np.nan, alpha=0.05):
    if p is None or not np.isfinite(p):
        return 'no p-value'
    if np.isfinite(floor) and abs(p - floor) < 1e-9 and p >= alpha:
        return 'untestable at this n'
    return 'significant' if p < alpha else 'not significant'


def row(**kw):
    base = {'figure': '', 'comparison': '', 'metric': '', 'unit': '', 'n': np.nan,
            'estimate': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
            'effect_size': np.nan, 'effect_type': '', 'p': np.nan,
            'p_adjusted': np.nan, 'p_floor': np.nan, 'test': '',
            'verdict': '', 'source': '', 'note': ''}
    base.update(kw)
    return base


# Metrics carried in RAW fitness units, and which normalisation.csv rescales them.
# Everything absent from this map is already unitless (a hazard ratio, an odds
# ratio, a correlation, a proportion) or already normalised at source (the
# landscape lift), and gets no normalised twin.
RAW_FITNESS_METRICS = {
    'converged avg_fitness': ('conditions', 'avg_fitness'),
    'd_f_off': ('assimilation', 'f_off'),
    'd_f_on': ('assimilation', 'f_on'),
    'd_lift': ('assimilation', 'lift'),
    'f_off': ('assimilation', 'f_off'),
}


def add_normalised(df, roots):
    """Restate raw-fitness estimates and CIs on the figures' [0, 1] axis.

    The figures are drawn on the min-max normalised scale, so a CI quoted in raw
    food-score units cannot be read off them. This adds the normalised twin of
    every raw-fitness estimate, dividing by the span from the same
    normalisation.csv the figures were drawn with.

    WHY ONLY THE ESTIMATE AND THE INTERVAL. Normalisation here is affine with
    CONSTANTS FIXED ACROSS EVERY CELL, z = (x - lo)/span. A paired t statistic is
    scale-invariant, Cohen's dz is a ratio of two quantities that scale together,
    and the rank tests see only the ordering — so p, p_adjusted and effect_size
    are numerically identical on either scale (verified to machine precision).
    Only the estimate, its interval and the standard deviation carry units, and
    only those are converted. Reporting a normalised Delta beside a p computed on
    raw values is therefore not a mismatch: it is the same test, restated.
    """
    spans = {}
    for key, root in roots.items():
        n = _read(os.path.join(root, 'normalisation.csv'))
        if n is None:
            continue
        for _i, r in n.iterrows():
            spans[(key, r.metric)] = float(r.span)

    def span_of(metric):
        k = RAW_FITNESS_METRICS.get(metric)
        return spans.get(k, np.nan) if k else np.nan

    sp = df['metric'].map(span_of)
    df['norm_span'] = sp
    for col in ('estimate', 'ci_lo', 'ci_hi'):
        df[col + '_norm'] = df[col] / sp
    return df


def from_conditions(root):
    out = []

    d = _read(os.path.join(root, 'paired_by_seed.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='fitness_avg_normalised / compare_fitness_avg_normalised',
                comparison=f'{r.environment}: {r.condition_a} vs {r.condition_b}',
                metric='converged avg_fitness', unit='map seed', n=r.n_seeds,
                estimate=r.mean_diff, ci_lo=r.ci95_lo, ci_hi=r.ci95_hi,
                effect_size=r.cohens_dz, effect_type="Cohen's dz",
                p=r.p_paired_t, p_adjusted=r.p_holm,
                test='paired t (Holm over 6); Wilcoxon p=%.4g' % r.p_wilcoxon,
                verdict=_verdict(r.p_holm), source='paired_by_seed.csv',
                note='terrain differenced out by pairing on seed'))

    d = _read(os.path.join(root, 'paired_by_environment.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='strip_final_fitness / compare_strip_final_fitness',
                comparison=f'{r.condition}: {r.env_a} vs {r.env_b}',
                metric='converged avg_fitness', unit='map seed', n=r.n_seeds,
                estimate=r.mean_diff, ci_lo=r.ci95_lo, ci_hi=r.ci95_hi,
                effect_size=r.cohens_dz, effect_type="Cohen's dz",
                p=r.p_paired_t, p_adjusted=r.p_holm,
                test='paired t (Holm over 3); Wilcoxon p=%.4g' % r.p_wilcoxon,
                verdict=_verdict(r.p_holm), source='paired_by_environment.csv',
                note=f'{int(r.seeds_favouring_a)}/{int(r.n_seeds)} seeds favour '
                     f'{r.env_a}; same terrain both sides'))

    d = _read(os.path.join(root, 'collapse_rate.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='strip_final_fitness (collapsed runs)',
                comparison=f'{r.environment}/{r.condition}: share below viability',
                metric='P(converged < 4.375)', unit='map seed (clustered)',
                n=r.n_seeds, estimate=r.rate, ci_lo=r.boot_lo, ci_hi=r.boot_hi,
                test='seed-clustered bootstrap interval (10k)',
                verdict='interval excludes 0' if r.boot_lo > 0 else 'interval includes 0',
                source='collapse_rate.csv',
                note=f'{int(r.n_collapsed)}/{int(r.n_runs)} runs, '
                     f'{int(r.seeds_with_any_collapse)}/{int(r.n_seeds)} seeds affected'))

    d = _read(os.path.join(root, 'time_to_threshold_logrank.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='time_to_viability_km',
                comparison=f'{r.environment}: all three conditions',
                metric='time to sustained viability', unit='run (log-rank)',
                n=r.n_runs, estimate=r.test_statistic, p=r.p,
                effect_type='chi-square', effect_size=r.test_statistic,
                test='log-rank (omnibus, no PH assumption)',
                verdict=_verdict(r.p), source='time_to_threshold_logrank.csv',
                note='right-censored at last generation if never reached'))

    d = _read(os.path.join(root, 'time_to_threshold_cox.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            ph = r.get('ph_assumption_min_p', np.nan)
            out.append(row(
                figure='time_to_viability_km',
                comparison=f'{r.environment}: {r.term} vs {r.reference}',
                metric='hazard of reaching viability', unit='map seed (clustered SE)',
                n=r.n_seeds, estimate=r.hazard_ratio,
                ci_lo=r.hr_ci95_lo, ci_hi=r.hr_ci95_hi,
                effect_size=r.hazard_ratio, effect_type='hazard ratio',
                p=r.p, test='Cox PH, Lin-Wei cluster-robust SE on seed',
                verdict=_verdict(r.p), source='time_to_threshold_cox.csv',
                note=('PH VIOLATED (p=%.1g) — HR is a time-average; quote RMST'
                      % ph) if np.isfinite(ph) and ph < 0.05 else ''))

    d = _read(os.path.join(root, 'time_to_threshold_km.csv'))
    if d is not None and 'rmst_gen' in d.columns:
        col = [c for c in d.columns if c.startswith('rmst_vs_')]
        ref = col[0].replace('rmst_vs_', '') if col else None
        for _i, r in d.iterrows():
            if ref is None or r.condition == ref:
                continue
            out.append(row(
                figure='time_to_viability_km',
                comparison=f'{r.environment}: {r.condition} vs {ref}',
                metric='RMST (generations not yet viable)', unit='run',
                n=r.n_runs, estimate=r[col[0]],
                test='restricted mean survival time (no PH assumption)',
                verdict='effect size, no p', source='time_to_threshold_km.csv',
                note=f'tau={r.rmst_tau:.0f}; negative = reaches viability sooner'))
    return out


def from_assimilation(root):
    out = []
    d = _read(os.path.join(root, 'assimilation_tests.csv'))
    if d is not None:
        claim = {'d_f_off': 'innate genome improves on its own (>0)',
                 'd_lift': 'learning advantage shrinks (<0)',
                 'd_f_on': 'learned phenotype improves (>0)',
                 'ratio_of_means': 'innate gaining on learned (>1)'}
        for _i, r in d.iterrows():
            if r.quantity == 'ratio_of_means':
                v = ('interval excludes 1'
                     if (r.ci95_lo > 1 or r.ci95_hi < 1) else 'interval includes 1')
                test = 'seed-clustered bootstrap of the ratio of means (10k)'
                p = np.nan
            else:
                v = _verdict(r.p_t)
                test = 'one-sample t; Wilcoxon p=%.4g' % r.p_wilcoxon
                p = r.p_t
            out.append(row(
                figure='assimilation_gap / innate_fitness / lift',
                comparison=f'{r.env}/{r.condition}: {claim.get(r.quantity, r.quantity)}',
                metric=r.quantity, unit='map seed', n=r.n_seeds,
                estimate=r['mean'], ci_lo=r.ci95_lo, ci_hi=r.ci95_hi,
                effect_size=r.cohens_d, effect_type="Cohen's d",
                p=p, test=test, verdict=v, source='assimilation_tests.csv',
                note='CONFOUNDED: RL-on pass not matched to the learning arm'
                     if r.confounded else 'gen 250 -> 1000'))

    d = _read(os.path.join(root, 'assimilation_arm_by_generation.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='innate_fitness',
                comparison=(f'{r.env}: {r.arm_a} vs {r.arm_b} on {r.quantity} '
                            f'at generation {int(r.generation)}'),
                metric=r.quantity, unit='map seed', n=r.n_seeds,
                estimate=r['mean'], ci_lo=r.ci95_lo, ci_hi=r.ci95_hi,
                effect_size=r.cohens_d, effect_type="Cohen's dz",
                p=r.p_t, p_adjusted=r.p_holm,
                test='paired t (Holm over %d generations); Wilcoxon p=%.4g'
                     % (r.n_generations_in_family, r.p_wilcoxon),
                verdict=_verdict(r.p_holm),
                source='assimilation_arm_by_generation.csv',
                note='level at one generation, not the 250->1000 change'))

    d = _read(os.path.join(root, 'assimilation_arm_contrast.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='innate_fitness',
                comparison=f'{r.env}: {r.arm_a} vs {r.arm_b} on {r.quantity}',
                metric=r.quantity, unit='map seed', n=r.n_seeds,
                estimate=r['mean'], ci_lo=r.ci95_lo, ci_hi=r.ci95_hi,
                effect_size=r.cohens_d, effect_type="Cohen's dz",
                p=r.p_t, test='paired t; Wilcoxon p=%.4g' % r.p_wilcoxon,
                verdict=_verdict(r.p_t), source='assimilation_arm_contrast.csv',
                note='the clean cross-arm contrast — RL-off pass runs no RL'))
    return out


def from_landscape(root):
    out = []
    for env in ('hard', 'baseline'):
        d = _read(os.path.join(root, env, 'lift_significance.csv'))
        if d is None:
            continue
        for _i, r in d.iterrows():
            out.append(row(
                figure='slices_lift / slices_off_vs_on / rl_effect',
                comparison=f'{env}: {r.test}', metric='median lift per slice',
                unit='slice', n=r.n_slices, estimate=r.median_of_medians,
                p=r.p_wilcoxon if np.isfinite(r.p_wilcoxon) else r.p_sign,
                p_floor=r.p_floor_at_this_n,
                test='sign / Wilcoxon' if r.n_positive >= 0 else 'Mann-Whitney U',
                verdict=_verdict(
                    r.p_wilcoxon if np.isfinite(r.p_wilcoxon) else r.p_sign,
                    r.p_floor_at_this_n),
                source=f'{env}/lift_significance.csv',
                note=(f'{int(r.n_positive)}/{int(r.n_slices)} slices positive'
                      if r.n_positive >= 0 else 'deliberately chosen anchors, not a sample')))

    d = _read(os.path.join(root, 'metrics', 'slice_metric_tests.csv'))
    if d is not None:
        for _i, r in d.iterrows():
            out.append(row(
                figure='slice_metrics',
                comparison=f'{r.metric}: {r.comparison}'
                           + (f' ({r.group_a} vs {r.group_b})'
                              if isinstance(r.group_b, str) and r.group_b else ''),
                metric=r.metric, unit='slice',
                n=int(r.n_a + (r.n_b or 0)), estimate=r['diff'],
                p=r.p, p_floor=r.p_floor, test=r.test,
                verdict=_verdict(r.p, r.p_floor),
                source='metrics/slice_metric_tests.csv', note=r.label))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join('output', 'stats'))
    for k, v in SRC.items():
        ap.add_argument(f'--{k}-root', default=v)
    args = ap.parse_args()

    rows = (from_conditions(args.conditions_root)
            + from_assimilation(args.assimilation_root)
            + from_landscape(args.landscape_root))
    if not rows:
        raise SystemExit('nothing found — run the analyses first')

    df = pd.DataFrame(rows)
    df = add_normalised(df, {'conditions': args.conditions_root,
                             'assimilation': args.assimilation_root,
                             'landscape': args.landscape_root})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'significance_summary.csv')
    df.to_csv(path, index=False)
    print(f'wrote {path}   ({len(df)} tests)')

    counts = df['verdict'].value_counts()
    print('\nverdicts:')
    for k, v in counts.items():
        print(f'  {k:<26} {v}')

    print('\n--- significant (alpha = 0.05, on the adjusted p where one exists) ---')
    sig = df[df.verdict.isin(['significant', 'interval excludes 0',
                              'interval excludes 1'])]
    for _i, r in sig.iterrows():
        p = r.p_adjusted if np.isfinite(r.p_adjusted) else r.p
        ptxt = 'CI' if not np.isfinite(p) else f'p={p:.3g}'
        nz = ('' if not np.isfinite(r.estimate_norm)
              else f'  ({r.estimate_norm:+.4f} normalised)')
        print(f'  {r.comparison:<62} {r.estimate:>+9.3f}  {ptxt}{nz}')

    print('\n--- NOT significant, and worth stating as such ---')
    for _i, r in df[df.verdict == 'not significant'].iterrows():
        p = r.p_adjusted if np.isfinite(r.p_adjusted) else r.p
        print(f'  {r.comparison:<62} {r.estimate:>+9.3f}  p={p:.3g}')

    unt = df[df.verdict == 'untestable at this n']
    if len(unt):
        print('\n--- UNTESTABLE at this n (p sits on its own floor) ---')
        for _i, r in unt.iterrows():
            print(f'  {r.comparison:<62} {r.estimate:>+9.3f}  '
                  f'p={r.p:.3g} = floor {r.p_floor:.3g}')


if __name__ == '__main__':
    main()
