"""
analyse_takeoff_hazard.py
======================================================================
When does a trapped evolution run ESCAPE the low basin, and is the waiting time
memoryless?

The hard environment (60 roaming predators, drain 5, seed 999) was re-run 10x
with identical terrain. Only the MAP is seeded — founder weights come from
`xavierRandom` on an unseeded `Math.random()` (NNBrain.js:168,232), and GA
mutation is unseeded too — so the 10 replicates differ purely by algorithmic
chance. Eight of them eventually climb out of the ~3.7 low basin to ~7.0; two
(rerun1, rerun6) never do within 1000 generations.

That gives a survival-analysis question with a mechanistic payoff:

    If escape is an UNDIRECTED event — the population diffuses through neutral
    genotype space until it happens on the crossing — then the escape hazard is
    constant in time and the waiting time is EXPONENTIAL (memoryless). If instead
    the population is being pushed toward the crossing by selection, the hazard
    RISES with time (Weibull shape k > 1): the longer it has been going, the
    readier it is, and escapes cluster at a characteristic generation.

So: constant hazard is the signature of escape-by-diffusion; increasing hazard is
the signature of escape-by-directed-progress. This script measures which.

Two runs never escaped. They are NOT dropped — they are right-CENSORED
observations (escape time > 1000), and dropping them would bias the rate upward.
Every estimator here handles censoring: the exponential MLE uses total-time-on-
test over all 10 runs, the survival curve is Kaplan-Meier, and the Weibull shape
is fitted by censored likelihood.

Defining "takeoff"
------------------
The low basin and the escaped level are both flat and well separated (~3.7 vs
~7.0 in population-mean fitness), so takeoff is a threshold crossing:

    threshold = basin + frac * (escaped - basin)

with `basin` the median smoothed fitness over the early plateau (--basin-window)
across all runs, and `escaped` the median final level of the runs that got out.
A crossing only counts if fitness STAYS above the threshold for --hold
generations, so a one-generation spike is not a takeoff. Because the answer must
not hinge on that choice, `--frac-sweep` re-derives every takeoff time over a
range of fracs and reports how much the fitted rate moves.

Outputs (into --out / output/takeoff_hazard):
    takeoff_trajectories.png -- all 10 fitness curves, takeoff marked, censored
                                runs flagged. What the event actually looks like.
    takeoff_hazard.png       -- Kaplan-Meier survival of "still trapped" with the
                                fitted exponential, the cumulative hazard (straight
                                line <=> constant hazard), a Q-Q plot of the
                                escape times against the fitted (truncated)
                                exponential, and the frac sensitivity.
    takeoff_times.csv        -- per run: takeoff generation, censored flag, levels.
    takeoff_stats.json       -- rate, mean waiting time, CIs, Weibull shape, tests.

Usage
-----
    python analyse_takeoff_hazard.py                        # hard reruns, frac 0.5
    python analyse_takeoff_hazard.py --frac 0.4 --hold 40
    python analyse_takeoff_hazard.py --metric top20percent_fitness
"""

import argparse
import glob
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

TRAPPED_C = '#B91C1C'    # runs that never escaped
ESCAPED_C = '#2563EB'
FIT_C = '#059669'

_RUN_RE = re.compile(r'rerun(\d+)$')


def discover(root, pattern):
    """Find the replicate run dirs (generations.csv present), sorted by rerun
    number so the legend reads rerun1..rerun10 rather than lexicographically."""
    runs = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        gen_csv = os.path.join(d, 'generations.csv')
        if not os.path.exists(gen_csv):
            continue
        m = _RUN_RE.search(os.path.basename(d))
        runs.append({'name': os.path.basename(d), 'dir': d, 'gen_csv': gen_csv,
                     'idx': int(m.group(1)) if m else 0})
    return sorted(runs, key=lambda r: r['idx'])


def load_curve(gen_csv, metric, smooth):
    """One run's smoothed fitness curve, indexed by generation."""
    df = pd.read_csv(gen_csv, usecols=['generation', metric])
    df = df.drop_duplicates('generation', keep='last').sort_values('generation')
    s = pd.to_numeric(df.set_index('generation')[metric], errors='coerce').dropna()
    return s.rolling(smooth, min_periods=1, center=True).mean()


def takeoff_generation(curve, threshold, hold):
    """First generation whose smoothed fitness is >= threshold AND stays there for
    `hold` generations. None if the run never sustains the crossing — i.e. the
    observation is right-censored at the run's last generation.

    The hold requirement is what separates a takeoff from a lucky generation: in
    the low basin the population mean rattles by ~0.3 between generations, so a
    bare first-crossing rule would fire on noise near the threshold."""
    above = (curve >= threshold).to_numpy()
    gens = curve.index.to_numpy()
    n = len(gens)
    for i in range(n):
        if not above[i]:
            continue
        # window = the next `hold` generations that exist in this run
        j = np.searchsorted(gens, gens[i] + hold, side='right')
        if j >= n and gens[-1] - gens[i] < hold:
            return None          # not enough run left to confirm it held
        if above[i:j].all():
            return float(gens[i])
    return None


def exponential_mle(times, censored):
    """Rate of a constant-hazard (exponential) escape process under Type-I
    censoring, with an exact chi-square CI.

    lambda_hat = d / TTT, where d is the number of observed escapes and TTT is
    total time on test = sum over ALL runs of (escape time, or censoring time for
    the two that never escaped). Dropping the censored runs would divide by 4190
    generations of exposure instead of 6190 and inflate the rate by ~50%.
    2*d*lambda_hat/lambda ~ chi2_2d gives the CI without any bootstrap."""
    d = int(np.sum(~censored))
    ttt = float(np.sum(times))
    if d == 0 or ttt <= 0:
        return dict(rate=np.nan, mean=np.nan, ci=(np.nan, np.nan), d=d, ttt=ttt)
    rate = d / ttt
    lo = stats.chi2.ppf(0.025, 2 * d) / (2 * ttt)
    hi = stats.chi2.ppf(0.975, 2 * d) / (2 * ttt)
    return dict(rate=rate, mean=1.0 / rate, ci=(lo, hi), d=d, ttt=ttt,
                mean_ci=(1.0 / hi, 1.0 / lo))


def weibull_mle(times, censored):
    """Weibull (shape k, scale s) by censored maximum likelihood.

    k is the whole point: k = 1 IS the exponential (constant hazard, memoryless);
    k > 1 means the hazard rises with time (runs 'get ready' to escape — directed
    progress); k < 1 means it falls. A CI on k that contains 1 is the closest this
    sample size can come to supporting the diffusion picture."""
    t = np.asarray(times, float)
    obs = ~np.asarray(censored, bool)

    def nll(p):
        lk, ls = p
        k, s = np.exp(lk), np.exp(ls)
        z = (t / s) ** k
        # log h(t) for the observed events + log S(t) for everyone
        return -(np.sum(np.log(k / s) + (k - 1) * np.log(t[obs] / s)) - np.sum(z))

    # Multiple restarts: Nelder-Mead silently stalls on this likelihood from a
    # bad start (it did, at one threshold, returning k=1.03 at a visibly worse
    # NLL than the k=1.9 optimum), which would read as "constant hazard" when it
    # is really a failed fit. Keep the best converged run.
    best, best_res = None, None
    for k0 in (0.5, 1.0, 2.0, 4.0):
        for s0 in (np.mean(t), np.max(t)):
            res = optimize.minimize(nll, x0=[np.log(k0), np.log(s0)],
                                    method='Nelder-Mead',
                                    options={'xatol': 1e-8, 'fatol': 1e-10,
                                             'maxiter': 4000})
            if np.isfinite(res.fun) and (best is None or res.fun < best):
                best, best_res = res.fun, res
    k, s = float(np.exp(best_res.x[0])), float(np.exp(best_res.x[1]))
    return k, s, best_res


def weibull_shape_ci(times, censored, n_boot, rng):
    """Bootstrap CI for the Weibull shape by resampling RUNS (the independent
    unit), keeping each run's censoring flag with it."""
    t = np.asarray(times, float)
    c = np.asarray(censored, bool)
    n = len(t)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (~c[idx]).sum() < 2:      # need >=2 events to fit a shape
            continue
        try:
            k, _s, res = weibull_mle(t[idx], c[idx])
            if res.success and 0.05 < k < 20:
                ks.append(k)
        except (ValueError, FloatingPointError):
            continue
    if len(ks) < 20:
        return (np.nan, np.nan), np.array(ks)
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))), np.array(ks)


def null_calibration(rate, censor_t, n_runs, n_sim, rng):
    """What does the Weibull shape estimate look like when the process REALLY IS
    memoryless? Simulate n_sim datasets from an exponential at the fitted rate,
    censored at the same time with the same number of runs, and refit k on each.

    This is the test that matters here. The MLE of a Weibull shape is badly biased
    upward at n = 8 events, so a raw k-hat of 1.8 does NOT by itself mean the
    hazard rises — the null distribution of k-hat is centred well above 1. The
    fraction of simulated memoryless datasets that produce a k-hat at least as
    large as the observed one is an exact, calibrated p-value for
    'is this more clustered than pure diffusion would give?'"""
    ks = []
    for _ in range(n_sim):
        t = rng.exponential(1.0 / rate, n_runs)
        c = t > censor_t
        t = np.minimum(t, censor_t)
        if (~c).sum() < 2:
            continue
        try:
            k, _s, res = weibull_mle(t, c)
            if np.isfinite(k) and 0.05 < k < 50:
                ks.append(k)
        except (ValueError, FloatingPointError):
            continue
    return np.array(ks)


def kaplan_meier(times, censored):
    """KM estimate of S(t) = P(still trapped at t). Returns step (t, S) arrays."""
    order = np.argsort(times)
    t = np.asarray(times, float)[order]
    c = np.asarray(censored, bool)[order]
    n_at_risk = len(t)
    ts, ss = [0.0], [1.0]
    s = 1.0
    for i in range(len(t)):
        if not c[i]:
            s *= (1 - 1.0 / n_at_risk)
            ts.append(t[i]); ss.append(s)
        n_at_risk -= 1
    return np.array(ts), np.array(ss)


def ks_truncated_exponential(escape_times, rate, censor_t):
    """KS test of the OBSERVED escape times against the fitted exponential
    CONDITIONED on escaping before the censoring time.

    Testing them against a plain exponential would be wrong: these 8 times are
    exactly the ones that landed inside [0, C], so their reference distribution is
    the exponential truncated to that window, F(t)/F(C)."""
    t = np.asarray(escape_times, float)
    denom = 1 - np.exp(-rate * censor_t)
    cdf = lambda x: (1 - np.exp(-rate * np.asarray(x))) / denom
    return stats.kstest(t, cdf)


def lag_check(escape_times, rate, censor_t, n_runs):
    """Is there a refractory period? Under a constant hazard from generation 0,
    the expected number of the n_runs that escape before the FIRST observed escape
    is n_runs*(1-exp(-rate*t_first)); seeing none is evidence against pure
    memorylessness from t=0. Reported as an exact binomial tail — it is the one
    place this data can speak to the early-time behaviour."""
    t_first = float(np.min(escape_times))
    p_before = 1 - np.exp(-rate * t_first)
    return dict(first_escape=t_first, p_escape_before_first=p_before,
                expected_before=n_runs * p_before,
                p_value=float((1 - p_before) ** n_runs))


def plot_trajectories(curves, takeoffs, threshold, basin, escaped, metric, out_dir):
    """All replicates on one axis with their takeoff marked — the plot that
    justifies the threshold rule before any statistics are quoted."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, curve in curves.items():
        t = takeoffs[name]
        col = TRAPPED_C if t is None else ESCAPED_C
        ax.plot(curve.index, curve.values, color=col, lw=1.4,
                alpha=0.85 if t is not None else 1.0)
        if t is not None:
            ax.plot([t], [curve.reindex([t]).iloc[0]], 'o', ms=7, color=col,
                    mec='white', mew=1.2, zorder=5)
            ax.annotate(name.replace('roam60d5_g1k_evolution_lscape_seed999_', ''),
                        (t, curve.reindex([t]).iloc[0]), textcoords='offset points',
                        xytext=(0, 9), fontsize=7, ha='center', color=col)
        else:
            ax.annotate(name.replace('roam60d5_g1k_evolution_lscape_seed999_', '')
                        + ' (never)', (curve.index[-1], curve.values[-1]),
                        textcoords='offset points', xytext=(-4, 4), fontsize=7,
                        ha='right', color=TRAPPED_C)
    ax.axhline(threshold, color='#6B7280', ls='--', lw=1)
    ax.text(0.005, threshold, ' takeoff threshold', color='#6B7280', fontsize=8,
            va='bottom', transform=ax.get_yaxis_transform())
    ax.axhline(basin, color='#9CA3AF', ls=':', lw=1)
    ax.axhline(escaped, color='#9CA3AF', ls=':', lw=1)
    ax.set_xlabel('Generation')
    ax.set_ylabel(metric)
    ax.set_title('Escape from the low basin — 10 evolution replicates, hard environment\n'
                 'identical terrain (seed 999); founder weights and GA mutation unseeded',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    fig.tight_layout()
    path = os.path.join(out_dir, 'takeoff_trajectories.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_hazard(times, censored, esc, fit, wb_k, wb_ci, censor_t, sweep,
                null_ks, null_p, ks_res, lag, out_dir):
    """The views that decide constant-vs-rising hazard."""
    fig, axs = plt.subplots(2, 3, figsize=(19, 9.5))

    # (1) Kaplan-Meier survival + fitted exponential.
    ax = axs[0, 0]
    kt, ks_ = kaplan_meier(times, censored)
    ax.step(kt, ks_, where='post', color=ESCAPED_C, lw=2,
            label='Kaplan–Meier (still trapped)')
    grid = np.linspace(0, censor_t, 300)
    ax.plot(grid, np.exp(-fit['rate'] * grid), color=FIT_C, lw=2, ls='--',
            label=f"exponential MLE (rate {fit['rate']*1000:.2f}/1000 gen)")
    ct = np.asarray(times)[np.asarray(censored)]
    if len(ct):
        ax.plot(ct, np.interp(ct, kt, ks_), '|', ms=14, mew=2, color=TRAPPED_C,
                label=f'censored ({len(ct)} never escaped)')
    ax.set_xlabel('Generation'); ax.set_ylabel('P(still in the low basin)')
    ax.set_title('Survival of the trapped state', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (2) Cumulative hazard: a STRAIGHT LINE THROUGH THE ORIGIN is what constant
    # hazard looks like. Curvature up = the hazard rises with time.
    ax = axs[0, 1]
    with np.errstate(divide='ignore'):
        ch = -np.log(np.clip(ks_, 1e-12, None))
    ax.step(kt, ch, where='post', color=ESCAPED_C, lw=2, label='−ln S(t) (Nelson–Aalen-ish)')
    ax.plot(grid, fit['rate'] * grid, color=FIT_C, lw=2, ls='--',
            label='constant hazard (fitted)')
    ax.set_xlabel('Generation'); ax.set_ylabel('cumulative hazard')
    ax.set_title('Cumulative hazard — straight ⇔ memoryless', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (3) Q-Q of the observed escapes against the fitted TRUNCATED exponential.
    ax = axs[1, 0]
    n = len(esc)
    p = (np.arange(1, n + 1) - 0.5) / n
    denom = 1 - np.exp(-fit['rate'] * censor_t)
    q_theory = -np.log(1 - p * denom) / fit['rate']
    ax.plot(q_theory, np.sort(esc), 'o', color=ESCAPED_C, ms=7)
    lim = [0, max(q_theory.max(), np.max(esc)) * 1.08]
    ax.plot(lim, lim, color=FIT_C, lw=1.5, ls='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('theoretical quantile (truncated exponential)')
    ax.set_ylabel('observed takeoff generation')
    ax.set_title(f'Q–Q vs fitted exponential (n={n} escapes)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (4) The calibrated test: k-hat's null distribution when the process really
    # is memoryless. If the observed k sits inside this cloud, a rising hazard is
    # NOT established — the estimator alone produces k > 1 at this sample size.
    ax = axs[1, 1]
    if len(null_ks):
        ax.hist(null_ks, bins=40, color='#9CA3AF', alpha=0.75,
                label=f'k̂ under a memoryless process\n(median {np.median(null_ks):.2f}, n={len(null_ks)} sims)')
        ax.axvline(1.0, color='#111827', lw=1.5, ls=':', label='true k = 1')
        ax.axvline(wb_k, color=TRAPPED_C, lw=2.5,
                   label=f'observed k̂ = {wb_k:.2f}   (p = {null_p:.3f})')
    ax.set_xlabel('Weibull shape k̂'); ax.set_ylabel('simulated datasets')
    ax.set_title('Is k̂ > 1 just small-sample bias?', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    # (5) Does the conclusion depend on where the threshold sits?
    ax = axs[1, 2]
    fr = [s['frac'] for s in sweep]
    ax.errorbar(fr, [s['rate'] * 1000 for s in sweep],
                yerr=[[(s['rate'] - s['ci'][0]) * 1000 for s in sweep],
                      [(s['ci'][1] - s['rate']) * 1000 for s in sweep]],
                marker='o', lw=2, capsize=4, color=ESCAPED_C, ecolor='#6B7280')
    ax2 = ax.twinx()
    ax2.plot(fr, [s['k'] for s in sweep], marker='s', lw=1.5, ls=':', color='#D97706')
    ax2.axhline(1.0, color='#D97706', lw=1, alpha=0.4)
    ax2.set_ylabel('Weibull shape k  (1 = constant hazard)', color='#D97706')
    ax2.tick_params(axis='y', colors='#D97706')
    ax.set_xlabel('threshold position between basin and escaped level')
    ax.set_ylabel('escape rate per 1000 generations', color=ESCAPED_C)
    ax.set_title('Sensitivity to the takeoff threshold', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (6) The numbers, so the figure stands alone.
    ax = axs[0, 2]
    ax.axis('off')
    txt = (
        f"escapes                  {fit['d']} of {len(times)}\n"
        f"time on test             {fit['ttt']:.0f} generations\n"
        f"\n"
        f"escape rate              {fit['rate']*1000:.2f} per 1000 gen\n"
        f"   95% CI                {fit['ci'][0]*1000:.2f} – {fit['ci'][1]*1000:.2f}\n"
        f"mean waiting time        {fit['mean']:.0f} generations\n"
        f"   95% CI                {fit['mean_ci'][0]:.0f} – {fit['mean_ci'][1]:.0f}\n"
        f"\n"
        f"KS vs exponential        D = {ks_res.statistic:.3f},  p = {ks_res.pvalue:.3f}\n"
        f"Weibull shape k̂          {wb_k:.2f}   (boot CI {wb_ci[0]:.2f}–{wb_ci[1]:.2f})\n"
        f"   calibrated p(k̂≥obs)   {null_p:.3f}\n"
        f"\n"
        f"first escape             generation {lag['first_escape']:.0f}\n"
        f"   expected by then      {lag['expected_before']:.1f} of {len(times)} runs\n"
        f"   p(none that early)    {lag['p_value']:.3f}"
    )
    ax.text(0.0, 0.98, txt, transform=ax.transAxes, fontsize=9.5,
            family='monospace', va='top')
    ax.set_title('Summary', fontsize=11, fontweight='bold', loc='left')

    verdict = ('consistent with constant hazard' if null_p > 0.05
               else 'hazard rises with time')
    plt.suptitle('Is escape from the low basin memoryless? — hard environment, 10 replicates\n'
                 f'k = 1 ⇔ constant hazard ⇔ undirected (diffusive) escape.   '
                 f'Observed k̂ = {wb_k:.2f}, calibrated p = {null_p:.3f} → {verdict}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'takeoff_hazard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def derive_takeoffs(curves, frac, hold, basin_window):
    """Threshold + per-run takeoff generation for one choice of `frac`."""
    basin = float(np.median([c[(c.index >= basin_window[0]) &
                               (c.index <= basin_window[1])].median()
                             for c in curves.values()]))
    finals = np.array([c.tail(50).mean() for c in curves.values()])
    # The escaped level = the upper cluster's median. Split at the midpoint of the
    # observed range, which is unambiguous here (3.7 vs 7.0, nothing between).
    hi = finals[finals > (finals.min() + finals.max()) / 2]
    escaped = float(np.median(hi)) if len(hi) else float(finals.max())
    threshold = basin + frac * (escaped - basin)
    takeoffs = {name: takeoff_generation(c, threshold, hold)
                for name, c in curves.items()}
    return basin, escaped, threshold, takeoffs


def main():
    ap = argparse.ArgumentParser(
        description='Takeoff (low-basin escape) times across evolution replicates, '
                    'and whether the waiting time is memoryless.')
    ap.add_argument('--root', default='logs_landscapes/evolution/standard/auto-run',
                    help='Folder holding the replicate run dirs.')
    ap.add_argument('--pattern', default='roam60d5_g1k_evolution_lscape_seed999_rerun*',
                    help='Glob for the replicate dirs (default: the 10 hard reruns).')
    ap.add_argument('--metric', default='avg_fitness',
                    help='generations.csv column defining the level (default avg_fitness).')
    ap.add_argument('--smooth', type=int, default=11,
                    help='Centred rolling-mean window on the fitness curve.')
    ap.add_argument('--frac', type=float, default=0.5,
                    help='Takeoff threshold as a fraction of the basin->escaped gap.')
    ap.add_argument('--hold', type=int, default=25,
                    help='Generations the curve must STAY above the threshold.')
    ap.add_argument('--basin-window', type=int, nargs=2, default=[50, 150],
                    help='Generation range defining the low-basin level.')
    ap.add_argument('--frac-sweep', type=float, nargs='+',
                    default=[0.3, 0.4, 0.5, 0.6, 0.7],
                    help='Thresholds for the sensitivity panel.')
    ap.add_argument('--n-boot', type=int, default=2000,
                    help='Bootstrap resamples for the Weibull shape CI.')
    ap.add_argument('--n-sim', type=int, default=2000,
                    help='Simulated memoryless datasets calibrating the shape test.')
    ap.add_argument('--seed', type=int, default=0, help='Bootstrap RNG seed.')
    ap.add_argument('-o', '--out', default='output/takeoff_hazard',
                    help='Output directory.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = discover(args.root, args.pattern)
    if not runs:
        print(f"No runs matched {os.path.join(args.root, args.pattern)}")
        return
    print(f"Found {len(runs)} replicates under {args.root}")

    curves = {r['name']: load_curve(r['gen_csv'], args.metric, args.smooth)
              for r in runs}
    censor_t = float(min(c.index.max() for c in curves.values()))

    basin, escaped, threshold, takeoffs = derive_takeoffs(
        curves, args.frac, args.hold, args.basin_window)
    print(f"  basin level {basin:.3f}  escaped level {escaped:.3f}  "
          f"-> threshold {threshold:.3f} (frac {args.frac})")
    print(f"  censoring time {censor_t:.0f} generations\n")

    rows = []
    for r in runs:
        name = r['name']
        t = takeoffs[name]
        rows.append({'run': name,
                     'takeoff_generation': t if t is not None else '',
                     'censored': int(t is None),
                     'time_on_test': t if t is not None else censor_t,
                     'final_level': float(curves[name].tail(50).mean())})
        short = name.replace('roam60d5_g1k_evolution_lscape_seed999_', '')
        print(f"    {short:<10} " + (f"takeoff at generation {t:.0f}" if t is not None
                                     else f"NEVER escaped (censored at {censor_t:.0f})")
              + f"   final {rows[-1]['final_level']:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'takeoff_times.csv'), index=False)

    times = np.array([r['time_on_test'] for r in rows], float)
    censored = np.array([bool(r['censored']) for r in rows])
    esc = times[~censored]
    if len(esc) < 3:
        print("\nToo few escapes to fit a waiting-time distribution.")
        return

    fit = exponential_mle(times, censored)
    wb_k, wb_s, _res = weibull_mle(times, censored)
    rng = np.random.default_rng(args.seed)
    wb_ci, _ks = weibull_shape_ci(times, censored, args.n_boot, rng)
    ks_res = ks_truncated_exponential(esc, fit['rate'], censor_t)
    lag = lag_check(esc, fit['rate'], censor_t, len(times))
    null_ks = null_calibration(fit['rate'], censor_t, len(times), args.n_sim, rng)
    null_p = float(np.mean(null_ks >= wb_k)) if len(null_ks) else np.nan

    print(f"\n  Escapes {fit['d']}/{len(times)}   total time on test {fit['ttt']:.0f} generations")
    print(f"  Exponential MLE: rate {fit['rate']*1000:.2f} per 1000 gen "
          f"(95% CI {fit['ci'][0]*1000:.2f}–{fit['ci'][1]*1000:.2f})")
    print(f"  Mean waiting time {fit['mean']:.0f} generations "
          f"(95% CI {fit['mean_ci'][0]:.0f}–{fit['mean_ci'][1]:.0f})")
    print(f"  Weibull shape k = {wb_k:.2f} (bootstrap CI {wb_ci[0]:.2f}–{wb_ci[1]:.2f})"
          f"   [k=1 ⇔ constant hazard]")
    if len(null_ks):
        print(f"  Calibrated against a TRULY memoryless process: k̂ median "
              f"{np.median(null_ks):.2f}, 95th pct {np.percentile(null_ks, 95):.2f} "
              f"-> p(k̂ ≥ {wb_k:.2f}) = {null_p:.3f}")
    print(f"  KS vs truncated exponential: D = {ks_res.statistic:.3f}, "
          f"p = {ks_res.pvalue:.3f}")
    print(f"  First escape at generation {lag['first_escape']:.0f}; under the fitted "
          f"rate {lag['expected_before']:.1f} of {len(times)} runs should already have "
          f"escaped by then (p = {lag['p_value']:.3f} for seeing none)")

    sweep = []
    for f in args.frac_sweep:
        b, e, th, tk = derive_takeoffs(curves, f, args.hold, args.basin_window)
        tt = np.array([tk[r['name']] if tk[r['name']] is not None else censor_t
                       for r in runs], float)
        cc = np.array([tk[r['name']] is None for r in runs])
        if (~cc).sum() < 3:
            continue
        ff = exponential_mle(tt, cc)
        kk, _s, _r = weibull_mle(tt, cc)
        sweep.append({'frac': f, 'threshold': th, 'n_escaped': int((~cc).sum()),
                      'rate': ff['rate'], 'ci': ff['ci'], 'k': kk})
    print("\n  Threshold sensitivity:")
    for s in sweep:
        print(f"    frac {s['frac']:.1f} (f={s['threshold']:.2f}): {s['n_escaped']} escapes, "
              f"rate {s['rate']*1000:.2f}/1000 gen, k = {s['k']:.2f}")

    stats_out = {
        'metric': args.metric, 'frac': args.frac, 'hold': args.hold,
        'basin_level': basin, 'escaped_level': escaped, 'threshold': threshold,
        'censor_time': censor_t, 'n_runs': len(times), 'n_escaped': fit['d'],
        'total_time_on_test': fit['ttt'],
        'exponential_rate_per_gen': fit['rate'], 'exponential_rate_ci': list(fit['ci']),
        'mean_waiting_generations': fit['mean'], 'mean_waiting_ci': list(fit['mean_ci']),
        'weibull_shape': wb_k, 'weibull_shape_ci': list(wb_ci), 'weibull_scale': wb_s,
        'weibull_shape_null_median': float(np.median(null_ks)) if len(null_ks) else None,
        'weibull_shape_null_p95': float(np.percentile(null_ks, 95)) if len(null_ks) else None,
        'weibull_shape_calibrated_p': null_p,
        'ks_statistic': float(ks_res.statistic), 'ks_pvalue': float(ks_res.pvalue),
        'lag_check': lag,
        'threshold_sweep': [{k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in s.items()} for s in sweep],
        'takeoff_times': {r['run']: r['takeoff_generation'] for r in rows},
    }
    with open(os.path.join(args.out, 'takeoff_stats.json'), 'w') as fh:
        json.dump(stats_out, fh, indent=2, default=float)

    print("\nPlotting ...")
    plot_trajectories(curves, takeoffs, threshold, basin, escaped, args.metric, args.out)
    plot_hazard(times, censored, esc, fit, wb_k, wb_ci, censor_t, sweep,
                null_ks, null_p, ks_res, lag, args.out)
    print(f"  Saved: {os.path.join(args.out, 'takeoff_times.csv')}")
    print(f"  Saved: {os.path.join(args.out, 'takeoff_stats.json')}")
    print(f"Done. Outputs in {os.path.abspath(args.out)}\n")


if __name__ == '__main__':
    main()
