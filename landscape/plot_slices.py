"""
landscape/plot_slices.py
======================================================================
Draw the fitness-landscape SLICES themselves, grouped by run outcome.

slice_metrics.py reduces each slice to scalars (FDC, dispersion, lambda_w, ...)
and plots the distribution of those scalars. This script plots the surfaces the
scalars were computed from, so that "the underperforming runs sit on a worse
piece of landscape" can be looked at rather than inferred from a number.

WHAT A PANEL IS
---------------
One slice = one 31x31 grid of f(theta) over the plane anchored at ONE run's
trajectory mean and spanned by that run's own best 2D directions (see
make_slice_jobs.py). Every cell is a real probe: 100 clones of theta, GA off,
averaged over R=5 fixed maps. The run's own trajectory is overlaid in the same
(alpha, beta) coordinates — it is the ONLY overlay on a panel, so nothing on the
surface can be mistaken for it. Viability is reported as a percentage in the
panel title instead of as a contour.

Slices are NOT a common coordinate system. Each run has its own anchor and its
own u/v, so (alpha, beta) means something different in every panel and the
panels cannot be subtracted from one another. What IS comparable across panels
is the fitness axis — every probe is the same measurement on the same scale —
which is why every panel shares one colour scale and one viability threshold.

VIABILITY = 4.375, THE SAME NUMBER EVERYWHERE
----------------------------------------------
A lineage holds its numbers when it banks 3.5 food per reproduction attempt /
0.8 success rate = 4.375 food per child. Both constants are ExperimentParams
values, so the threshold is theoretical, is identical for every run, condition
and environment, and needs no calibration job to exist.

It is a LOWER bound on the true cost: AdvancedOrganism.reproduce() also requires
isClear() and canAddOrganism(), and the counter resets whether or not placement
succeeded, so some banked food buys no child. The viable percentages here are
therefore UPPER bounds. That is the one caveat to state, and it is one-sided in a
convenient direction — inflating every slice equally cannot manufacture a
difference between arms, only understate one.

Reproduction costs NO energy in this simulator (verified: reproduce() deducts
nothing), so the probe running with reproduction off does NOT put f(theta) on a
different scale. The threshold applies to it directly.

--calib-json opts into Stage-9 B1's measured crossing (5.39 for hard) as a
sensitivity check. resolve_replacement() has the full argument for why that is
the sensitivity check and not the default.

RL OFF vs RL ON
---------------
Every cell is probed TWICE: once with in-lifetime learning disabled (f_evo, the
'RL off' pass) and once with it enabled (f_learn, 'RL on'), at the SAME theta on
the SAME 5 maps. The two passes are paired by map index, so

    lift = f_learn - f_evo

is what in-lifetime learning adds at that genome, with terrain cancelled. That
subtraction is the only honest way to see it: terrain variance (sigma_vary ~ 6-8)
dwarfs the lift (~0-3), which is exactly why the effect is invisible when the two
surfaces are eyeballed side by side.

NORMALISATION — ONE MIN-MAX RANGE, SHARED BY EVERY SLICE AND BOTH ENVIRONMENTS
-------------------------------------------------------------------------------
Fitness is rescaled by the same affine map the condition comparison uses
(analyse_conditions_by_environment.py, NORMALISATION):

    z = (f - lo) / (hi - lo)

lo/hi are the min and max over EVERY probed cell of EVERY slice in ALL
environments passed on the command line, RL-off and RL-on pooled
(--norm-scope global, the default). The constants are written to
normalisation.csv and stamped on every figure subtitle, because a normalised
axis with an unstated range is not interpretable.

Three deliberate choices, each different from the condition comparison's:

1. f_evo and f_learn SHARE one range. There the per-metric ranges exist because
   best_fitness and avg_fitness live on different scales; here the two passes are
   the SAME measurement at the SAME genome on the SAME maps, and the whole point
   is that one can be subtracted from the other. Separate ranges would destroy
   that.
2. The lift is a DIFFERENCE, so it carries no offset and the map reduces to a
   division by the span — `lo` cancels. A normalised lift reads as "this
   fraction of the full observed fitness range", exactly as the normalised sigmas
   do in the variance table.
3. Scope is global across environments for the same reason it is there: a
   per-environment range would map hard and baseline onto the same endpoints and
   delete the vertical gap between them, which is the comparison the figures
   exist to make. The cost is that neither environment spans the whole of [0, 1],
   and that is the honest picture rather than a defect.

FDC, the autocorrelation length and the neutral fraction are unaffected by this
in the sense that matters: FDC is a correlation and dispersion is a ratio, so
both are invariant under an affine rescaling. The neutral fraction is NOT — see
slice_metrics.py --norm-range, which is the reason that flag exists.

--raw turns normalisation off and plots f(theta) in its own food-score units.

INCOMPLETE RL-ON COVERAGE
-------------------------
Every cell is probed R=5 times, once per fixed map, in each pass. If the SLURM
array did not finish, cells are missing repeats — and because a shard is a
contiguous range of job ids, the gaps are contiguous BANDS of alpha rows, not
scattered noise.

reconstruct_grid() averages whatever repeats it finds, silently. That is fine for
a full cell and wrong for a partial one: the repeats are different MAPS, and
terrain variance (sigma_vary ~ 6-8) is far larger than most of the spatial
structure on a slice, so a 3-map mean and a 5-map mean are not the same
estimator. This script therefore MASKS any cell without all R RL-on repeats out
of f_learn, and prints the coverage per slice.

The LIFT is treated differently, because it does not have that problem: it is a
per-repeat paired difference on identical terrain, so terrain cancels inside each
repeat and a mean over 3 shared repeats is unbiased for the same quantity as a
mean over 5 — only noisier. It is computed over whatever repeats are paired, and
masked only where none are.

f_evo is unaffected: the RL-off pass is the cheap one and finished everywhere.

FIGURES
-------
  slices_f_evo.png    f(theta), RL off — the surface the weights alone reach
  slices_f_learn.png  f(theta), RL on  — same cells, same maps, learning enabled
  slices_lift.png     f_learn - f_evo, the paired difference, per cell
  rl_effect.png       the lift as numbers: viable % off vs on, mean f off -> on,
                      and lift against the underlying RL-off fitness
  outcome_endpoints.png  where the successful runs end up on the landscape and
                      where the failed ones do — the fitness along each projected
                      path, the headroom left on its own plane, and the endpoint's
                      rank within its own slice

Usage
-----
  python landscape/plot_slices.py \
      --slices-root hard:landscape/out/slices_h128_companion \
      --slices-root baseline:landscape/out/slices_h128_baseline_companion \
      --out-dir landscape/out/slice_landscapes

--slices-root is repeatable and takes ENV:PATH. Every root given shares ONE
normalisation range; figures are written per environment into <out-dir>/<env>/.
A bare PATH still works and is treated as a single unnamed environment.
"""

import argparse
import glob
import json
import os
import sys
import textwrap
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

import ll_difficulty as ld
from stage9_difficulty import load_grid


INK = '#374151'
GRID_C = '#D1D5DB'

# Row order and colour of the group label.
#
# THE SPLIT IS NOT THE SAME QUANTITY IN BOTH ENVIRONMENTS, so the label cannot be
# hard-coded. make_slice_jobs.py --threshold decides which runs are
# 'underperform', and the two slice sets were built with different values:
#
#   hard      --threshold 4.375  = replacement. Its underperformers finish at
#                                  f = 3.60-3.68, genuinely below replacement.
#                                  Those runs FAILED.
#   baseline  --threshold 7.478  = a median split. Its underperformers finish at
#                                  f = 7.31-7.37 — the bottom of the distribution
#                                  but still 1.7x replacement. Those runs did NOT
#                                  fail, and calling them failures would invent a
#                                  baseline collapse that never happened.
#
# group_rows() reads the threshold out of the slice set's own manifest and says
# which of the two a given row is.
GROUP_ORDER = [
    ('evolution', 'underperform', '#DC2626'),
    ('evolution', 'succeed',      '#059669'),
    ('learning',  'succeed',      '#2563EB'),
]


def group_label(condition, group, threshold, replacement, norm=None):
    """The row label, given the split the slice set was actually built with.

    `norm` maps the split threshold onto the figure's own axis, so the label
    quotes the same units as every other number on the page. Left raw only when
    the figure itself is raw.
    """
    cond = condition.capitalize()
    if group != 'underperform':
        return f'{cond} — succeeded'
    if threshold is None or not np.isfinite(threshold):
        return f'{cond} — underperformed'
    t = norm(threshold) if norm else threshold
    fmt = '{:.3f}'.format(t) if norm else '{:g}'.format(t)
    if abs(threshold - replacement) < 1e-6:
        return f'{cond} — failed (f < {fmt}, below viability)'
    return (f'{cond} — weakest runs (f < {fmt}, a median split — '
            f'still above viability)')

# How each condition's TRAJECTORY is drawn. These are not the row-label colours
# above: a row label says which group of runs the panel is about, while these say
# which mechanism drew a given path — and with --companion a single panel carries
# one of each. Near-black and magenta both survive a white halo on viridis and on
# RdBu_r; the dash pattern is the redundant channel.
TRAJ_STYLE = {
    'evolution': {'colour': '#111827', 'ls': '-',  'label': 'evolution run'},
    'learning':  {'colour': '#E11D8F', 'ls': (0, (5, 2)), 'label': 'learning run'},
}


def manifest_threshold(slices_root):
    """The --threshold make_slice_jobs.py used to split succeed / underperform.

    It is written into every manifest_<condition>.json in the slice root. Reading
    it back is what keeps the row labels honest — see GROUP_ORDER.
    """
    for mf in sorted(glob.glob(os.path.join(slices_root, 'manifest_*.json'))):
        try:
            t = json.load(open(mf)).get('threshold')
        except (ValueError, OSError):
            continue
        if t is not None:
            return float(t)
    return None


def discover(slices_root, env=''):
    """Load every slice under slices_root, keyed by (condition, group)."""
    slices = []
    threshold = manifest_threshold(slices_root)
    for d in sorted(glob.glob(os.path.join(slices_root, '*/'))):
        d = d.rstrip('/')
        if not os.path.exists(os.path.join(d, 'results.csv')):
            print(f'  [skip] {os.path.basename(d)} — no results.csv '
                  f'(run merge_shards.py first)')
            continue
        G = load_grid(d)
        pm = json.load(open(os.path.join(d, 'plane.json')))['meta']
        name = os.path.basename(d)
        G['plane_meta'] = pm
        G['condition'] = name.split('_')[0]
        G['group'] = pm.get('group', name.split('_')[1])
        G['run'] = pm.get('anchor_run', name)
        G['seed'] = pm.get('anchor_seed')
        G['final_fitness'] = pm.get('anchor_final_fitness', float('nan'))
        # Pair each projected trajectory with the run that produced it, in the
        # order build_slice_plane wrote them: runs_on_plane[i] is the run behind
        # run_trajectories[i], and the anchor run is always index 0. With
        # --companion the second entry is the matched run of the OTHER condition,
        # so the two paths must be told apart by condition, not by position.
        names = pm.get('runs_on_plane') or []
        G['traj'] = []
        for i, t in enumerate(pm.get('run_trajectories') or []):
            nm = names[i] if i < len(names) else ''
            cond = ('learning' if '_learning_' in nm
                    else 'evolution' if '_evolution_' in nm
                    else G['condition'])
            G['traj'].append({'xy': np.asarray(t, dtype=float), 'run': nm,
                              'condition': cond, 'is_anchor': i == 0})
        diag = pm.get('diagnostics') or {}
        w, c = diag.get('within_median'), diag.get('ceiling_median')
        G['plane_frac'] = (float(w) / float(c)) if (w and c) else float('nan')
        G['env'] = env
        G['split_threshold'] = threshold

        # Coverage. See INCOMPLETE RL-ON COVERAGE in the module docstring: a
        # partial cell is a mean over a SUBSET of the fixed map set, which is a
        # different estimator from a mean over all of it, so f_learn is masked
        # wherever the RL-on pass is short.
        R = G['R']
        full_on = G['n_on'] >= R
        G['cov_on'] = float(np.mean(full_on))
        G['cov_off'] = float(np.mean(G['n_off'] >= R))
        G['f_learn'] = np.where(full_on, G['f_learn'], np.nan)

        # The lift keeps every PAIRED repeat, because a paired difference on
        # identical terrain is unbiased at any R — cube_on - cube_off is NaN
        # exactly where a repeat is missing on either side, so nanmean over
        # axis 2 averages the repeats that survived on both.
        dcube = G['cube_on'] - G['cube_off']
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            # A cell with no paired repeat at all is expected on a partly
            # finished array; it is masked on the next line, not silently used.
            warnings.simplefilter('ignore', RuntimeWarning)
            G['lift'] = np.where(np.isfinite(dcube).any(axis=2),
                                 np.nanmean(dcube, axis=2), np.nan)
        G['n_paired'] = np.isfinite(dcube).sum(axis=2)
        slices.append(G)
    return slices


def ordered(slices, replacement, norm=None):
    """Rows of (label, colour, [slice, ...]) in GROUP_ORDER, best-first inside."""
    rows = []
    for cond, group, colour in GROUP_ORDER:
        sel = [s for s in slices if s['condition'] == cond and s['group'] == group]
        sel.sort(key=lambda s: -s['final_fitness'])
        if sel:
            label = group_label(cond, group, sel[0].get('split_threshold'),
                                replacement, norm)
            rows.append((label, colour, sel))
    return rows


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalisation_range(slices, fields=('f_evo', 'f_learn')):
    """(lo, hi) over every probed cell of every slice given, both RL passes.

    One range for both fields on purpose — they are the same measurement at the
    same genome, and the lift is their difference. See NORMALISATION.
    """
    vals = []
    for G in slices:
        for f in fields:
            v = np.asarray(G[f], dtype=float).ravel()
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
    if not vals:
        return None
    allv = np.concatenate(vals)
    return float(allv.min()), float(allv.max())


def rescale(values, rng):
    """z = (f - lo) / span, the same affine map the condition comparison uses."""
    if rng is None:
        return values
    lo, hi = rng
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return values
    return (np.asarray(values, dtype=float) - lo) / span


def rescale_difference(values, rng):
    """A SPREAD carries no offset, so the map reduces to a division by the span.

    Identical to the normalised sigmas in the condition comparison's variance
    table: `lo` cancels in any difference, and the result reads as a fraction of
    the full observed fitness range.
    """
    if rng is None:
        return values
    span = rng[1] - rng[0]
    if not np.isfinite(span) or span <= 0:
        return values
    return np.asarray(values, dtype=float) / span


def apply_normalisation(slices, rng):
    """Rescale every slice in place, keeping the raw fields under a _raw name."""
    for G in slices:
        for f in ('f_evo', 'f_learn'):
            G[f + '_raw'] = G[f]
            G[f] = rescale(G[f], rng)
        G['lift_raw'] = G['lift']
        G['lift'] = rescale_difference(G['lift'], rng)


def write_normalisation_csv(path, rng, replacement, scope, n_slices, envs):
    """The rescaling constants, in the same shape the comparison folder uses."""
    lo, hi = rng
    span = hi - lo
    with open(path, 'w') as fh:
        fh.write('scope,group,metric,lo,hi,span,'
                 f'viability_{replacement:g}_normalised,n_slices,environments\n')
        for metric in ('f_evo', 'f_learn'):
            fh.write(f'{scope},*,{metric},{lo!r},{hi!r},{span!r},'
                     f'{(replacement - lo) / span!r},{n_slices},'
                     f'"{"|".join(envs)}"\n')
        # The lift is scaled by the span alone, so its lo/hi are not the same
        # constants and are recorded as such rather than silently reused.
        fh.write(f'{scope},*,lift,0.0,{span!r},{span!r},,{n_slices},'
                 f'"{"|".join(envs)}"\n')
    print(f'  wrote {path}')


def viable_pct(grid, threshold):
    f = grid[np.isfinite(grid)]
    return 100.0 * float(np.mean(f > threshold)) if f.size else float('nan')


def resolve_replacement(explicit, calib_json):
    """The self-replacement level ON THE PROBE AXIS, and a one-line provenance.

    ONE criterion throughout: a lineage holds its numbers when it banks enough
    food per child actually LANDED. 4.375 and 5.39 are not two criteria, and the
    gap between them is NOT an energy cost of reproducing.

    REPRODUCTION IS ENERGY-NEUTRAL — verified, not assumed.
    AdvancedOrganism.reproduce() deducts nothing: it builds the child, tries to
    place it, and resets energyGainedSinceReproduction whether or not placement
    succeeded. 3.5 is a counter threshold, not an energy debit. So "the probe has
    reproduction off, therefore its fitness is on another scale" is NOT the
    reason the two numbers differ.

    WHY THEY DIFFER: 4.375 COUNTS ONE GATE OF THREE.
    reproduce() lands a child only when

        child.isClear(new_c, new_r, ...) && env.canAddOrganism() && rand() < 0.8

    and the counter resets on ALL paths. 4.375 = 3.5 / 0.8 prices only the third
    gate — it is the food per child that would be needed if every attempt passing
    the roll also found space. The first two gates (room at the spawn point, the
    global population cap) burn attempts too, so the real food per child landed
    is >= 4.375 BY CONSTRUCTION. ll_difficulty's docstring says exactly this and
    calls the resulting viable fraction "correspondingly OPTIMISTIC".

    Stage-9 B1 measures the effective value instead of bounding it: replay the
    probed genomes with reproduction ON and interpolate the f(θ) at which
    children per founder = 1. Hard environment: 5.387, bracketed by two probed
    genomes at f = 5.365 (0.976 children/founder) and f = 5.533 (1.162).
    4.375 / 5.387 = 0.81 — i.e. the two placement gates jointly pass ~81% of the
    time, which is the whole of the discrepancy.

    So: 4.375 is the theoretical LOWER BOUND, 5.39 the MEASURED value of the same
    quantity. Using 4.375 on f(θ) is not wrong, it is optimistic, and that is what
    the collapsed run scoring 82% "viable" was showing.

    THIS SCRIPT DEFAULTS TO 4.375, and the reason is defensibility, not accuracy.
    4.375 falls out of two ExperimentParams constants and holds identically for
    every run, condition and environment — nothing about it is fitted, so it needs
    no per-environment calibration job and cannot drift between arms. The 5.39
    crossing is better centred but far weaker as evidence: three probed genomes,
    only two of them straddling replacement, no confidence interval, hard
    environment only. There is no baseline equivalent and getting one means
    re-running B1 with genomes that bracket 1.0 — so adopting it would make the
    hard and baseline landscapes incomparable until that job exists.

    What this costs, and the sentence that pays for it: the viable fraction is an
    UPPER BOUND. Report it as such and the collapsed runs' high viable % is a
    stated property of the estimator rather than an anomaly. Note the bound is
    one-sided in a convenient direction — it cannot manufacture a difference
    between arms, it can only understate one, since it inflates every slice.
    --calib-json landscape/out/stage9_calib_hard/calibration.json switches to the
    measured crossing for a sensitivity check.

    DO NOT USE k_mean (1.80 -> 7.89). It is a composition artefact, not a
    conversion. calibration.json's k_mean divides f_monomorphic by the replay's
    mean over ALL agents, and that average is dragged down by children born
    mid-generation who had less time to eat. The same genome scores 3.64 as
    all-agents but 6.44 over founders alone. Against founders — the like-for-like
    comparison for a fixed full-lifetime cohort — k is 0.86, i.e. BELOW 1, which
    would push the threshold to ~3.76 rather than up to 7.89. The crossing avoids
    the choice entirely: it maps f_monomorphic straight to children per founder
    and never averages a mixed-age population. Stage 9 prefers it for that reason
    (stage9_difficulty.py:458).

    CAVEAT for the write-up: the crossing rests on the two genomes of three that
    straddle replacement, so the bracket is narrow (5.37-5.53) but thin.

    Baseline is NOT covered: stage9_calib_baseline/calibration.json has no
    crossing (no probed genome straddles replacement there), only k_mean = 2.16,
    which per the above should not be used as a conversion. A baseline slice set
    needs B1 re-run with genomes that bracket 1.0.
    """
    if explicit is not None:
        return float(explicit), 'set explicitly via --replacement'
    if calib_json and os.path.exists(calib_json):
        cal = json.load(open(calib_json))
        f_star = cal.get('empirical_replacement_f')
        if f_star:
            return (float(f_star),
                    f'MEASURED self-replacement crossing, {os.path.basename(calib_json)}'
                    f' (= {ld.VIABILITY_BASE:g} x {float(f_star) / ld.VIABILITY_BASE:.2f})')
    if calib_json:
        print(f'  [warn] no measured crossing in {calib_json!r} — using the '
              f'theoretical {ld.VIABILITY_BASE:g} instead.')
    return (float(ld.VIABILITY_BASE),
            f'theoretical replacement 3.5/0.8 = {ld.VIABILITY_BASE:g}; '
            f'lower bound, so viable % is an UPPER bound')


def _panel(ax, grid, G, cmap, norm, threshold=None):
    """One slice as a heatmap with the anchor run's trajectory — and nothing else.

    grid is indexed [i=alpha, j=beta]; imshow wants [row=y=beta], hence the .T.

    THE VIABILITY CONTOUR IS OFF BY DEFAULT (--viability-contour turns it on).
    A bare threshold line does not say WHICH SIDE is viable, and on a marginal
    slice that is actively misleading: hard seed 2004 ends at f = 0.150 against a
    0.136 threshold, so its endpoint reads as sitting on the boundary when the
    run it came from finished at 3.60, below replacement. The contour is not
    wrong there — probe viability is an UPPER bound, because 4.375 prices one of
    the three gates in reproduce() — but a reader cannot see that from a dashed
    line, and will read "inside the line" as "this run was fine".

    The viable PERCENTAGE stays in every panel title, which carries the same
    information without inviting a per-pixel reading of a bound that is only
    one-sided.
    """
    im = ax.imshow(grid.T, origin='lower', extent=G['extent'], cmap=cmap,
                   norm=norm, interpolation='nearest', aspect='auto')
    if threshold is not None:
        g = np.asarray(grid, dtype=float)
        if np.nanmin(g) < threshold < np.nanmax(g):
            ax.contour(G['alphas'], G['betas'], g.T, levels=[threshold],
                       colors='white', linestyles='--', linewidths=1.3,
                       zorder=3.5)
    for t in G['traj']:
        xy = t['xy']
        if xy.size == 0:
            continue
        st = TRAJ_STYLE.get(t['condition'], TRAJ_STYLE['evolution'])
        # A white halo under every path, so both stay legible over the dark and
        # the bright end of viridis and over both lobes of RdBu_r. The condition
        # is carried by the line colour AND its dash pattern, because the two
        # paths cross and a reader who cannot separate the hues still can.
        ax.plot(xy[:, 0], xy[:, 1], color='#FFFFFF', lw=2.6, alpha=.9, zorder=4,
                solid_capstyle='round')
        ax.plot(xy[:, 0], xy[:, 1], color=st['colour'], lw=1.3, ls=st['ls'],
                alpha=.95, zorder=5, solid_capstyle='round')
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=30, marker='o', c=st['colour'],
                   edgecolors='white', linewidths=1.0, zorder=6)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=62, marker='*', c=st['colour'],
                   edgecolors='white', linewidths=.8, zorder=6)
    ax.tick_params(colors=INK, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID_C)
    return im


def _surface_figure(rows, field, cmap, title, sub, out_png, annotate,
                    norm_kind='linear', viab=None, run_f=None):
    """One row per outcome group, one column per slice in that group.

    annotate(G) -> the short string appended to each panel title (the viable
    percentage on the fitness figures, the median lift on the difference one).
    """
    vals = np.concatenate([s[field][np.isfinite(s[field])].ravel()
                           for _, _, sel in rows for s in sel])
    if norm_kind == 'diverging':
        # Symmetric about 0 so +0.3 and -0.3 read equally strongly — an
        # asymmetric norm makes small negatives look like large ones. Capped at
        # the 95th percentile of |lift| rather than the max: the learning row
        # runs to +4.7, and scaling to that washes the other six panels white.
        # The tails clip, hence extend='both' on the colourbar.
        cap = float(np.nanpercentile(np.abs(vals), 95))
        norm = plt.Normalize(-cap, cap)
    else:
        norm = plt.Normalize(float(np.nanpercentile(vals, 1)),
                             float(np.nanpercentile(vals, 99)))

    ncol = max(len(sel) for _, _, sel in rows)
    nrow = len(rows)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol + 1.4, 3.35 * nrow),
                            squeeze=False)

    im = None
    for r, (label, colour, sel) in enumerate(rows):
        for c in range(ncol):
            ax = axs[r][c]
            if c >= len(sel):
                ax.set_visible(False)
                continue
            G = sel[c]
            im = _panel(ax, G[field], G, cmap, norm, threshold=viab)
            # The plane score belongs on the panel: it is how much of the runs'
            # motion this 2D cut actually shows, and it is NOT equal across
            # panels — failed runs share a plane better than successful ones, so
            # comparing "how much path is visible" between rows is invalid
            # unless the reader can see both numbers.
            plane = (f"  ·  plane {G['plane_frac']:.0%}"
                     if np.isfinite(G.get('plane_frac', np.nan)) else '')
            # run f in the SAME units as everything else on the figure. It is
            # the run's own logged avg_fitness, a population measurement rather
            # than a probe, so the shared min-max map is a convenience for
            # comparability and not a claim that the two are the same estimator.
            # The raw value stays in slice_summary.csv.
            rf = run_f(G) if run_f else G['final_fitness']
            ax.set_title(f"seed {G['seed']}  ·  run f = {rf:.3f}"
                         f'{plane}\n{annotate(G)}',
                         fontsize=9, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel('β', fontsize=9, color=INK)
                # Wrapped: these labels carry the split threshold, and a single
                # rotated line long enough to hold it runs into the suptitle.
                ax.text(-0.30, 0.5, textwrap.fill(label, 26),
                        transform=ax.transAxes, rotation=90, va='center',
                        ha='center', fontsize=9.5, color=colour,
                        fontweight='bold', linespacing=1.35)
            if r == nrow - 1:
                ax.set_xlabel('α', fontsize=9, color=INK)

    # The subtitle carries the viability definition AND the normalisation range,
    # which is far too long for one line — wrap it and give the axes back only
    # what is left, or it lands on top of the first row's panel titles.
    title = '\n'.join(l for part in title.split('\n')
                      for l in (textwrap.wrap(part, 98) or ['']))
    n_title = title.count('\n') + 1
    # hspace leaves room for the two-line panel titles; left for the wrapped
    # row labels; top for however many lines the suptitle turned into.
    fig.subplots_adjust(left=.085, right=.9, top=.90 - .035 * n_title,
                        hspace=.45)
    cax = fig.add_axes([.915, .12, .014, .74])
    cb = fig.colorbar(im, cax=cax,
                      extend='both' if norm_kind == 'diverging' else 'neither')
    cb.set_label(sub, fontsize=9, color=INK)
    cb.ax.tick_params(colors=INK, labelsize=8)
    cb.outline.set_edgecolor(GRID_C)

    # Only advertise the conditions actually drawn: on a single-run slice set a
    # "learning run" key would promise a path that is not in any panel.
    drawn = {t['condition'] for _l, _c, sel in rows for G in sel for t in G['traj']}
    handles = [Line2D([], [], color=TRAJ_STYLE[c]['colour'], lw=1.8,
                      ls=TRAJ_STYLE[c]['ls'], label=TRAJ_STYLE[c]['label'])
               for c in ('evolution', 'learning') if c in drawn]
    handles += [Line2D([], [], color='none', marker='o', mfc=INK,
                       mec='white', ms=6, label='gen 0'),
                Line2D([], [], color='none', marker='*', mfc=INK,
                       mec='white', ms=11, label='final generation')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(.45, -.005))

    fig.suptitle(title, fontsize=12.5, color=INK, y=.985)
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def fig_paired(rows, out_path, threshold, unit, title, sub, viab, run_f):
    """Each slice twice, RL OFF beside RL ON, on one shared colour scale.

    The separate f_evo / f_learn figures answer "what does this surface look
    like" but make the comparison that matters an act of memory: the reader has
    to hold one page in mind while looking at the other. Here the two passes for
    a given slice are ADJACENT, so what learning adds at a genome is a saccade
    rather than a page turn.

    ONE NORM ACROSS BOTH PASSES, and it has to be: the pair is only readable as
    a comparison if a colour means the same fitness on the left as on the right.
    Normalising each pass separately would make the RL-on panel look better (or
    worse) purely by rescaling.

    Columns are ordered off, on, off, on, ... rather than all-off then all-on,
    so the pairing is positional and needs no legend to decode.
    """
    fields = ('f_evo', 'f_learn')
    vals = np.concatenate([s[f][np.isfinite(s[f])].ravel()
                           for _l, _c, sel in rows for s in sel for f in fields])
    norm = plt.Normalize(float(np.nanpercentile(vals, 1)),
                         float(np.nanpercentile(vals, 99)))

    n_slices = max(len(sel) for _l, _c, sel in rows)
    ncol, nrow = n_slices * 2, len(rows)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.9 * ncol + 1.6, 3.5 * nrow),
                            squeeze=False)

    im = None
    for r, (label, colour, sel) in enumerate(rows):
        for c in range(n_slices):
            for k, field in enumerate(fields):
                ax = axs[r][2 * c + k]
                if c >= len(sel):
                    ax.set_visible(False)
                    continue
                G = sel[c]
                im = _panel(ax, G[field], G, 'viridis', norm, threshold=viab)
                pct = viable_pct(G[field], threshold)
                ax.set_title(('RL OFF' if k == 0 else 'RL ON')
                             + f'\n{pct:.1f}% viable', fontsize=8.5,
                             color=INK, pad=3)
                if k == 0:
                    # The slice's identity sits over the PAIR, not over one half.
                    ax.text(1.03, 1.20, f"seed {G['seed']}  ·  run f = "
                            f"{run_f(G):.3f}  ·  plane "
                            f"{G['plane_frac']:.0%}",
                            transform=ax.transAxes, ha='center', va='bottom',
                            fontsize=9.5, color=INK)
                if k == 0 and c == 0:
                    ax.set_ylabel('β', fontsize=9, color=INK)
                    ax.text(-0.34, 0.5, textwrap.fill(label, 26),
                            transform=ax.transAxes, rotation=90, va='center',
                            ha='center', fontsize=9.5, color=colour,
                            fontweight='bold', linespacing=1.35)
                else:
                    ax.set_yticklabels([])
                if r == nrow - 1:
                    ax.set_xlabel('α', fontsize=9, color=INK)

    # Each PAIR carries a header line above its two panel titles, so this needs
    # more headroom than the single-surface figures: wrap the suptitle first,
    # then give the axes only what is left under it.
    title = '\n'.join(l for part in title.split('\n')
                      for l in (textwrap.wrap(part, 104) or ['']))
    n_title = title.count('\n') + 1
    fig.subplots_adjust(left=.075, right=.9, top=.86 - .045 * n_title,
                        hspace=.46, wspace=.12)
    cax = fig.add_axes([.915, .12, .012, .62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(sub, fontsize=9, color=INK)
    cb.ax.tick_params(colors=INK, labelsize=8)
    cb.outline.set_edgecolor(GRID_C)

    handles = [Line2D([], [], color=TRAJ_STYLE[c]['colour'], lw=1.8,
                      ls=TRAJ_STYLE[c]['ls'], label=TRAJ_STYLE[c]['label'])
               for c in ('evolution', 'learning')
               if c in {t['condition'] for _l, _c, sel in rows
                        for G in sel for t in G['traj']}]
    if viab is not None:
        handles += [Line2D([], [], color='#9CA3AF', lw=1.4, ls='--',
                           label=f'viability boundary ({viab:.3f})')]
    handles += [Line2D([], [], color='none', marker='o', mfc=INK, mec='white',
                       ms=6, label='gen 0'),
                Line2D([], [], color='none', marker='*', mfc=INK, mec='white',
                       ms=11, label='final generation')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(.49, -.02))

    fig.suptitle(title, fontsize=12.5, color=INK, y=.985)
    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_path}')


def fig_rl_effect(rows, out_png, threshold, unit=''):
    """The RL-on minus RL-off difference as numbers rather than colour.

    The heatmaps show the lift's spatial shape but not its size: on a colour
    scale that has to hold the learning slices, the failed slices' lift is a
    barely-tinted white. These three panels are the same quantity read off
    directly, so 'what learning adds' is comparable across slices.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5))
    flat = [(lab, col, G) for lab, col, sel in rows for G in sel]
    xs = []
    x = 0
    for lab, col, sel in rows:
        for _ in sel:
            xs.append(x); x += 1
        x += .7

    # 1 — viable % of the slice, RL off vs RL on, at the one threshold.
    ax = axs[0]
    w = .38
    for xi, (lab, col, G) in zip(xs, flat):
        ax.bar(xi - w / 2, viable_pct(G['f_evo'], threshold), width=w,
               color=col, alpha=.45)
        ax.bar(xi + w / 2, viable_pct(G['f_learn'], threshold), width=w,
               color=col)
    ax.set_ylabel(f'% of slice cells with f > {threshold:.3f}', fontsize=9, color=INK)
    ax.set_title(f'Viable percentage of each slice (f > {threshold:.3f})\n'
                 'pale = RL off, solid = RL on   ·   RL-on bar uses only the '
                 'fully-probed cells', fontsize=10.5, color=INK)

    # 2 — mean f, RL off -> RL on, as a paired dumbbell.
    ax = axs[1]
    for xi, (lab, col, G) in zip(xs, flat):
        a = float(np.nanmean(G['f_evo'])); b = float(np.nanmean(G['f_learn']))
        ax.plot([xi, xi], [a, b], color=col, lw=2.4, zorder=2)
        ax.scatter([xi], [a], s=52, facecolor='white', edgecolor=col,
                   linewidths=2, zorder=3)
        ax.scatter([xi], [b], s=52, color=col, zorder=3)
        ax.annotate(f'{b - a:+.3f}', xy=(xi, max(a, b)), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    color=col)
    ax.axhline(threshold, color=INK, lw=1, ls=':')
    ax.set_ylabel(f'mean f(θ) over the slice{unit}', fontsize=9, color=INK)
    ax.set_title('What learning adds, per slice\n'
                 'hollow = RL off, filled = RL on', fontsize=10.5, color=INK)

    # 3 — lift against the RL-off fitness underneath it. Answers WHERE learning
    # helps: on cells the weights alone already handle, or on the poor ones.
    ax = axs[2]
    edges = np.linspace(0, 100, 11)
    for lab, col, sel in rows:
        for k, G in enumerate(sel):
            fo = G['f_evo'].ravel(); li = G['lift'].ravel()
            m = np.isfinite(fo) & np.isfinite(li)
            fo, li = fo[m], li[m]
            q = np.percentile(fo, edges)
            cx, cy = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                sel_c = (fo >= lo) & (fo <= hi)
                if sel_c.sum() >= 5:
                    cx.append(float(np.mean(fo[sel_c])))
                    cy.append(float(np.mean(li[sel_c])))
            ax.plot(cx, cy, color=col, lw=1.8, alpha=.55 + .15 * k,
                    marker='o', ms=3, label=lab if k == 0 else None)
    ax.axhline(0, color=GRID_C, lw=1)
    ax.axvline(threshold, color=INK, lw=1, ls=':')
    ax.annotate(f'f = {threshold:.3f}', xy=(threshold, ax.get_ylim()[1]),
                xytext=(3, -10), textcoords='offset points', fontsize=7.5, color=INK)
    ax.set_xlabel(f'f(θ) with RL off, binned by decile{unit}', fontsize=9,
                  color=INK)
    ax.set_ylabel(f'mean lift  f_on − f_off{unit}', fontsize=9, color=INK)
    ax.set_title('Where learning helps\nlift vs the fitness underneath it',
                 fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper right')

    for ax in axs[:2]:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(G['seed']) for _l, _c, G in flat], fontsize=8,
                           color=INK)
        ax.set_xlabel('anchor run seed', fontsize=9, color=INK)
    for ax in axs:
        ax.grid(True, axis='y', alpha=.25, lw=.7)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID_C)
        ax.tick_params(colors=INK, labelsize=8)

    fig.suptitle('RL on vs RL off — the paired difference at identical θ on '
                 'identical maps', fontsize=13, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


# make_slice_jobs.py --stride default: the projected trajectories carry every
# 5th generation of the run, so trajectory index i is generation 5*i.
TRAJ_STRIDE = 5


def sample_surface(G, xy, field='f_evo'):
    """f(theta) at arbitrary (alpha, beta) points, by bilinear interpolation.

    Points outside the probed rectangle come back NaN rather than clamped: a
    trajectory that leaves the grid has left the measured surface, and pinning it
    to the boundary would invent a fitness for it.
    """
    a, b = G['alphas'], G['betas']
    f = np.asarray(G[field], dtype=float)
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    out = np.full(x.shape, np.nan)
    inside = (x >= a[0]) & (x <= a[-1]) & (y >= b[0]) & (y <= b[-1])
    if not inside.any():
        return out
    xi = np.clip(np.searchsorted(a, x[inside]) - 1, 0, len(a) - 2)
    yi = np.clip(np.searchsorted(b, y[inside]) - 1, 0, len(b) - 2)
    tx = (x[inside] - a[xi]) / (a[xi + 1] - a[xi])
    ty = (y[inside] - b[yi]) / (b[yi + 1] - b[yi])
    out[inside] = ((1 - tx) * (1 - ty) * f[xi, yi]
                   + tx * (1 - ty) * f[xi + 1, yi]
                   + (1 - tx) * ty * f[xi, yi + 1]
                   + tx * ty * f[xi + 1, yi + 1])
    return out


# Which probe pass a trajectory must be read on. An evolution run's genome is
# the whole of what it produced, so f_evo — the RL-off pass — is the right
# surface for it. A LEARNING run's genome is not: that condition's agents adapt
# in their own lifetime, and scoring their weights with learning disabled marks
# them on a task they never faced. Reading both paths off f_evo would understate
# the learning run by construction, so each is read on the pass matching the
# condition it actually ran under.
CONDITION_FIELD = {'evolution': 'f_evo', 'learning': 'f_learn'}


def find_run_dir(run_name, roots=('logs', 'logs_hard')):
    """The run's own log directory, if it is on this machine."""
    for root in roots:
        for d in glob.glob(os.path.join(root, '*', '*', 'auto-run', run_name)):
            if os.path.exists(os.path.join(d, 'generations.csv')):
                return d
    return None


_FINAL_F_CACHE = {}


def own_final_fitness(run_name):
    """Mean avg_fitness over the run's last 50 generations, or NaN if absent.

    The anchor's own value is in plane.json, but the COMPANION's is not — and
    stamping the anchor's fitness on the companion's row would report one run's
    outcome under another run's name.
    """
    if run_name not in _FINAL_F_CACHE:
        d = find_run_dir(run_name)
        v = float('nan')
        if d:
            try:
                import csv as _csv
                vals = []
                with open(os.path.join(d, 'generations.csv')) as fh:
                    for row in _csv.DictReader(fh):
                        try:
                            vals.append(float(row['avg_fitness']))
                        except (KeyError, ValueError):
                            pass
                v = float(np.mean(vals[-50:])) if vals else float('nan')
            except OSError:
                pass
        _FINAL_F_CACHE[run_name] = v
    return _FINAL_F_CACHE[run_name]


def endpoint_stats(rows):
    """Per trajectory: fitness along the projected path and where it stopped."""
    recs = []
    for label, colour, sel in rows:
        for G in sel:
            for t in G['traj']:
                xy = t['xy']
                if xy.size == 0:
                    continue
                field = CONDITION_FIELD.get(t['condition'], 'f_evo')
                f = np.asarray(G[field], dtype=float)
                cells = f[np.isfinite(f)]
                path = sample_surface(G, xy, field)
                good = path[np.isfinite(path)]
                end = float(path[-1]) if np.isfinite(path[-1]) else (
                    float(good[-1]) if good.size else float('nan'))
                start = float(path[0]) if np.isfinite(path[0]) else (
                    float(good[0]) if good.size else float('nan'))
                recs.append({
                    'row': label, 'colour': colour, 'group': G['group'],
                    'condition': t['condition'], 'is_anchor': t['is_anchor'],
                    'seed': G['seed'], 'run': t['run'], 'env': G.get('env', ''),
                    'read_on': field,
                    # The run's OWN final fitness, never the anchor's.
                    'run_final_fitness': (G['final_fitness'] if t['is_anchor']
                                          else own_final_fitness(t['run'])),
                    'plane_frac': G['plane_frac'],
                    'gens': np.arange(len(path)) * TRAJ_STRIDE,
                    'path': path,
                    'start_f': start, 'end_f': end,
                    'slice_best': float(cells.max()) if cells.size else np.nan,
                    'slice_median': float(np.median(cells)) if cells.size else np.nan,
                    # How much of its OWN plane the run ended above. A run that
                    # stops at the 30th percentile of a surface it was free to
                    # move over stopped somewhere the surface itself did not
                    # force it to.
                    'end_pct': (100.0 * float(np.mean(cells < end))
                                if cells.size and np.isfinite(end) else np.nan),
                    # Points with no fitness behind them: off the probed
                    # rectangle, or over a cell masked for short RL-on coverage.
                    'n_unsampled': int(np.sum(~np.isfinite(path))),
                })
    return recs


def fig_outcome_endpoints(rows, out_png, threshold, unit, env_note):
    """Where the successful runs end up on the landscape, and where the failed
    ones do.

    The surface figures already SHOW the endpoint — it is the star on each panel
    — but they cannot be compared cell for cell, because every slice has its own
    anchor and its own u/v and (alpha, beta) means something different in each.
    What IS common to all of them is the fitness axis: every cell is the same
    measurement on the same normalised scale. These three panels use only that.

    THE ONE CAVEAT, and it is not small: the curve is f(theta) READ OFF THE
    PLANE at the projected centroid, not the run's own recorded fitness. The
    plane holds 62-69% of each run's motion (the per-panel plane score), so the
    remaining third of the journey is perpendicular to the page and invisible
    here. Read the curves as "the landscape under the part of the path this cut
    can see", never as a replacement for the run's fitness curve.
    """
    recs = endpoint_stats(rows)
    if not recs:
        return None
    anchors = [r for r in recs if r['is_anchor']]

    fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.8))

    # 1 — fitness along the projected path, gen 0 to the last generation.
    ax = axs[0]
    for r in recs:
        st = TRAJ_STYLE.get(r['condition'], TRAJ_STYLE['evolution'])
        ax.plot(r['gens'], r['path'], color=r['colour'], lw=1.5, ls=st['ls'],
                alpha=.9 if r['is_anchor'] else .55, zorder=3)
        if np.isfinite(r['end_f']):
            ax.scatter([r['gens'][-1]], [r['end_f']], s=58, marker='*',
                       color=r['colour'], edgecolors='white', linewidths=.7,
                       zorder=4)
    ax.axhline(threshold, color=INK, lw=1.1, ls=':', zorder=2)
    ax.annotate(f'replacement  {threshold:.3f}', xy=(0, threshold),
                xytext=(4, 4), textcoords='offset points', fontsize=7.5, color=INK)
    ax.set_xlabel('generation', fontsize=9, color=INK)
    ax.set_ylabel(f'f(θ) at the run’s centroid, on its own pass{unit}',
                  fontsize=9, color=INK)
    ax.set_title('The landscape along each run’s own path\n'
                 'solid = evolution (RL off), dashed = its matched learning run '
                 '(RL on)', fontsize=10.5, color=INK)

    # 2 — where it stopped vs the best cell on the plane it stopped on.
    ax = axs[1]
    xs = np.arange(len(anchors))
    order = sorted(range(len(anchors)), key=lambda i: -anchors[i]['end_f']
                   if np.isfinite(anchors[i]['end_f']) else 0)
    for x, i in zip(xs, order):
        r = anchors[i]
        ax.plot([x, x], [r['end_f'], r['slice_best']], color=r['colour'],
                lw=2.4, zorder=2)
        ax.scatter([x], [r['slice_best']], s=46, facecolor='white',
                   edgecolor=r['colour'], linewidths=2, zorder=3)
        ax.scatter([x], [r['end_f']], s=58, marker='*', color=r['colour'],
                   edgecolors='white', linewidths=.7, zorder=4)
        ax.annotate(f"{r['slice_best'] - r['end_f']:+.3f}",
                    xy=(x, r['slice_best']), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    color=r['colour'])
    ax.axhline(threshold, color=INK, lw=1.1, ls=':')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(anchors[i]['seed']) for i in order], fontsize=8,
                       color=INK)
    ax.set_xlabel('anchor run seed', fontsize=9, color=INK)
    ax.set_ylabel(f'f(θ){unit}', fontsize=9, color=INK)
    ax.set_title('Headroom left on its own plane\n'
                 '★ = where the evolution run ended, ○ = best cell probed there',
                 fontsize=10.5, color=INK)

    # 3 — the endpoint's rank inside its own slice.
    ax = axs[2]
    groups, seen = [], set()
    for r in anchors:
        if r['row'] not in seen:
            seen.add(r['row'])
            groups.append((r['row'], r['colour']))
    for gi, (row_label, colour) in enumerate(groups):
        vals = [r['end_pct'] for r in anchors if r['row'] == row_label]
        labs = [r['seed'] for r in anchors if r['row'] == row_label]
        jit = np.linspace(-.13, .13, len(vals))
        ax.scatter(gi + jit, vals, s=70, color=colour, zorder=3,
                   edgecolors='white', linewidths=.9)
        if len(vals) > 1:
            ax.plot([gi - .3, gi + .3], [np.mean(vals)] * 2, color=colour, lw=2.2)
        for x, v, sd in zip(jit, vals, labs):
            ax.annotate(str(sd), xy=(gi + x, v), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=7,
                        color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0].split('—')[-1].strip().split('(')[0].strip()
                        for g in groups], fontsize=8.5, color=INK)
    ax.set_xlim(-.6, len(groups) - .4)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of the slice’s cells scoring BELOW the endpoint',
                  fontsize=9, color=INK)
    ax.set_title('Did the run stop at a high point of its own plane?\n'
                 '100 = ended on the best cell probed, 50 = ended at the median',
                 fontsize=10.5, color=INK)

    for ax in axs:
        ax.grid(True, axis='y', alpha=.25, lw=.7)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID_C)
        ax.tick_params(colors=INK, labelsize=8)

    fig.suptitle('Where the runs end up on the landscape' + env_note,
                 fontsize=13, color=INK, y=1.02)
    # Wrapped by hand: bbox_inches='tight' grows the canvas to fit whatever this
    # text needs, so one long line silently stretches the whole figure.
    fig.text(.5, -.10, textwrap.fill(
        'f(θ) is read off the plane at the projected centroid, not the run’s own '
        'recorded fitness: the plane holds 62–69% of each run’s motion, so about a '
        'third of the journey is perpendicular to the page. Each path is read on '
        'the pass it ran under — evolution on RL off, learning on RL on — so the '
        'dashed curves break wherever a cell was masked for short RL-on coverage. '
        'The plane is anchored on the EVOLUTION run, so the learning path is a '
        'guest on it and its curve understates that run: read the learning runs’ '
        'own final fitness from endpoints.csv, not off this axis.',
        104), ha='center', va='top', fontsize=8.5, color=INK, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')
    return recs


def write_endpoint_csv(recs, path):
    cols = ['env', 'row', 'group', 'condition', 'is_anchor', 'seed', 'run',
            'run_final_fitness', 'read_on', 'plane_frac', 'start_f', 'end_f',
            'slice_median', 'slice_best', 'headroom', 'end_pct', 'n_unsampled']
    with open(path, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in recs:
            r = dict(r, headroom=r['slice_best'] - r['end_f'])
            fh.write(','.join(
                f'"{r[c]}"' if c in ('run', 'row') else
                (f'{r[c]:.5f}' if isinstance(r[c], float) else str(r[c]))
                for c in cols) + '\n')
    print(f'  wrote {path}')


def write_summary(rows, path, threshold, normalised):
    """Per-slice numbers, in normalised units where the figures are, plus the
    raw food-score columns beside them and the RL-on coverage the f_learn
    columns were computed on."""
    cols = ['env', 'condition', 'group', 'seed', 'run', 'run_final_fitness',
            'resolution', 'repeats', 'extent', 'plane_frac',
            'coverage_rl_off', 'coverage_rl_on',
            'f_evo_min', 'f_evo_median', 'f_evo_max', 'f_evo_at_anchor',
            'f_evo_range', 'viable_pct_rl_off', 'viable_pct_rl_on',
            'mean_f_rl_off', 'mean_f_rl_on', 'median_lift', 'mean_lift',
            'f_evo_median_raw', 'f_evo_max_raw', 'mean_f_rl_off_raw',
            'median_lift_raw', 'fdc_top5', 'grid_dir']
    with open(path, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for _label, _c, sel in rows:
            for G in sel:
                f = G['f_evo']; fin = f[np.isfinite(f)]
                raw = G.get('f_evo_raw', f); rawfin = raw[np.isfinite(raw)]
                lift_raw = G.get('lift_raw', G['lift'])
                mid = G['N'] // 2
                row = [G.get('env', ''), G['condition'], G['group'], G['seed'],
                       G['run'], f"{G['final_fitness']:.4f}", G['N'], G['R'],
                       f"{G['alphas'][-1] - G['alphas'][0]:.2f}",
                       f"{G['plane_frac']:.4f}",
                       f"{G['cov_off']:.4f}", f"{G['cov_on']:.4f}",
                       f'{fin.min():.4f}', f'{np.median(fin):.4f}',
                       f'{fin.max():.4f}', f'{f[mid, mid]:.4f}',
                       f'{fin.max() - fin.min():.4f}',
                       f"{viable_pct(f, threshold):.2f}",
                       f"{viable_pct(G['f_learn'], threshold):.2f}",
                       f'{np.nanmean(f):.4f}', f"{np.nanmean(G['f_learn']):.4f}",
                       f"{np.nanmedian(G['lift']):.4f}",
                       f"{np.nanmean(G['lift']):.4f}",
                       f'{np.median(rawfin):.4f}', f'{rawfin.max():.4f}',
                       f'{np.nanmean(raw):.4f}', f'{np.nanmedian(lift_raw):.4f}',
                       f"{ld.fdc(G['A'], G['B'], f, reference='top5', top_frac=.05)[0]:+.4f}",
                       G['grid_dir']]
                fh.write(','.join(str(v) for v in row) + '\n')
    print(f'  wrote {path}'
          + ('   [fitness columns are normalised unless suffixed _raw]'
             if normalised else '   [raw food-score units]'))


# How an environment key is NAMED on a figure. The key stays as it is — it is
# the directory name, the CSV column and the --slices-root argument, and every
# path already written depends on it — while the label is free to be the words a
# reader should see. 'hard' is the internal shorthand for the roaming-predator
# environment; 'predator' is what it actually is.
ENV_TITLE = {'hard': 'predator', 'baseline': 'baseline'}


def env_label_for(env, overrides):
    return overrides.get(env, ENV_TITLE.get(env, env))


def parse_root(spec):
    """ENV:PATH, or a bare PATH for a single unnamed environment."""
    if ':' in spec and not os.path.exists(spec):
        env, path = spec.split(':', 1)
        return env, path
    return '', spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slices-root', action='append', metavar='ENV:PATH',
                    help='Repeatable. Every root given shares ONE normalisation '
                         'range and gets its own <out-dir>/<env>/ of figures.')
    ap.add_argument('--out-dir', default='landscape/out/slice_plots')
    ap.add_argument('--replacement', type=float, default=None,
                    help='Replacement threshold on f(θ). Default: the theoretical '
                         '3.5/0.8 = 4.375, which needs no calibration and is '
                         'identical for every run, condition and environment.')
    ap.add_argument('--calib-json', default='',
                    help='OPT-IN sensitivity check: a Stage-9 B1 calibration.json '
                         'whose measured crossing (empirical_replacement_f, 5.39 '
                         'for the hard environment) replaces the theoretical '
                         'threshold. Off by default — see resolve_replacement().')
    ap.add_argument('--format', default='png', choices=('png', 'pdf', 'both'),
                    help='Figure format. pdf keeps the panels as vectors, which '
                         'is what a thesis wants — the heatmap itself is raster '
                         'either way, but the trajectories, contours and every '
                         'label stay sharp at any zoom.')
    ap.add_argument('--suffix-env', action='store_true',
                    help='Name every figure <stem>_<env>.<ext> using the DISPLAY '
                         'name, e.g. slices_off_vs_on_predator.pdf. Useful when '
                         'figures from both environments end up in one folder — '
                         'a thesis figures/ directory — where bare stems collide.')
    ap.add_argument('--env-title', action='append', default=[], metavar='KEY=NAME',
                    help='Repeatable. Rename an environment ON THE FIGURES only '
                         '— the directory, the CSV column and --slices-root keep '
                         "the key. Defaults: hard=predator. e.g. --env-title "
                         'hard=roaming-predator')
    ap.add_argument('--viability-contour', action='store_true',
                    help='Draw a white dashed line at the viability threshold on '
                         'each surface. OFF by default — see _panel() for why a '
                         'bare threshold line misleads on marginal slices.')
    ap.add_argument('--raw', action='store_true',
                    help='Plot f(θ) in raw food-score units instead of min-max '
                         'normalising it to [0, 1]. See NORMALISATION.')
    ap.add_argument('--norm-scope', choices=('global', 'env'), default='global',
                    help="'global' (default) puts every environment on ONE ruler, "
                         "so the gap between them is real. 'env' rescales each "
                         'environment against its own range and deletes that gap.')
    args = ap.parse_args()
    roots = [parse_root(r) for r in (args.slices_root
                                     or ['landscape/out/slices_h128_hard'])]
    threshold_raw, thr_note = resolve_replacement(args.replacement, args.calib_json)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load EVERY root first: the normalisation range is global by default, so it
    # cannot be known until all of them are in memory.
    loaded = []
    for env, root in roots:
        print(f'loading slices from {root}'
              + (f'   [{env}]' if env else '') + ' ...')
        sl = discover(root, env)
        if not sl:
            print(f'  [skip] no slices with a results.csv under {root}')
            continue
        for G in sl:
            if G['cov_on'] < 1.0:
                print(f"  [partial] {os.path.basename(G['grid_dir'])[:56]:56s} "
                      f"RL-on complete in {G['cov_on']:.1%} of cells "
                      f'— f_learn masked elsewhere')
        loaded.append((env, root, sl))
    if not loaded:
        raise SystemExit('no slices with a results.csv found')

    every = [G for _e, _r, sl in loaded for G in sl]
    rng = None
    if not args.raw:
        if args.norm_scope == 'global':
            rng = normalisation_range(every)
            apply_normalisation(every, rng)
            write_normalisation_csv(
                os.path.join(args.out_dir, 'normalisation.csv'), rng,
                threshold_raw, 'global', len(every),
                [e or '(unnamed)' for e, _r, _s in loaded])
        else:
            for env, _root, sl in loaded:
                r = normalisation_range(sl)
                apply_normalisation(sl, r)
                write_normalisation_csv(
                    os.path.join(args.out_dir, f'normalisation_{env or "all"}.csv'),
                    r, threshold_raw, 'env', len(sl), [env or '(unnamed)'])
            rng = normalisation_range(every)   # for the axis label only

    threshold = (threshold_raw if rng is None
                 else float(rescale(np.array([threshold_raw]), rng)[0]))
    unit = '' if rng is None else '  — normalised'
    print(f'viability threshold = {threshold:.4f} normalised '
          f'(raw f(θ) > {threshold_raw:.3f})   [{thr_note}]')
    if rng is not None:
        print(f'normalised range [{rng[0]:.4f}, {rng[1]:.4f}] '
              f'({args.norm_scope} scope) -> threshold {threshold:.4f}')

    # ONE SHORT LINE, in normalised units. The provenance of the threshold and
    # the normalisation constants belong in the prose and in normalisation.csv,
    # not baked into an image where they cannot be edited — the earlier
    # three-line subtitle crowded the panels and had to be re-wrapped for every
    # figure size. What a reader needs on the page is the number the viable
    # percentages were computed against.
    viab_note = (f'viability threshold {threshold:.3f}'
                 if rng is not None else
                 f'viability threshold f(θ) > {threshold_raw:g}')

    exts = ('png', 'pdf') if args.format == 'both' else (args.format,)
    contour = threshold if args.viability_contour else None
    titles = {}
    for spec in args.env_title:
        if '=' not in spec:
            raise SystemExit(f'--env-title wants KEY=NAME, got {spec!r}')
        k, v = spec.split('=', 1)
        titles[k] = v

    def run_f(G):
        """The run's logged final fitness, on the figure's own scale."""
        v = G['final_fitness']
        return v if rng is None else (v - rng[0]) / (rng[1] - rng[0])

    def paths(out_dir, stem, env_name=''):
        tail = f'_{env_name}' if (args.suffix_env and env_name) else ''
        return [os.path.join(out_dir, f'{stem}{tail}.{e}') for e in exts]

    for env, root, slices in loaded:
        out_dir = os.path.join(args.out_dir, env) if env else args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        shown = env_label_for(env, titles)
        env_note = f'  —  {shown} environment' if env else ''
        print(f'\n{shown or "slices"}: {len(slices)} slice(s) -> {out_dir}')
        to_norm = (None if rng is None
                   else (lambda v: (v - rng[0]) / (rng[1] - rng[0])))
        rows = ordered(slices, threshold_raw, to_norm)
        for label, _c, sel in rows:
            print(f'  {label:<62} {len(sel)} slice(s): '
                  + ', '.join('seed %s (f=%.3f)' % (s['seed'], run_f(s))
                              for s in sel))

        def pct_off(G):
            return f"{viable_pct(G['f_evo'], threshold):.1f}% viable"

        def pct_on(G):
            cov = ('' if G['cov_on'] >= 1.0
                   else f"  ·  {G['cov_on']:.0%} of cells complete")
            return f"{viable_pct(G['f_learn'], threshold):.1f}% viable{cov}"

        def med_lift(G):
            return f"median lift {np.nanmedian(G['lift']):+.3f}"

        for out in paths(out_dir, 'slices_off_vs_on', shown):
            fig_paired(
                rows, out, threshold, unit,
                'Fitness landscape slices, RL OFF beside RL ON — identical '
                f'cells, identical maps{env_note}\n{viab_note}',
                f'f(θ){unit}  (mean cumulative food score, 100 clones, GA off)',
                viab=contour, run_f=run_f)

        for out in paths(out_dir, 'slices_f_evo', shown):
            _surface_figure(
                rows, 'f_evo', 'viridis',
                'Fitness landscape slices, RL OFF — one plane per run, anchored '
                f'on that run’s own trajectory{env_note}\n{viab_note}',
                f'f(θ){unit}  (mean cumulative food score, 100 clones, GA off)',
                out, annotate=pct_off, viab=contour, run_f=run_f)

        for out in paths(out_dir, 'slices_f_learn', shown):
            _surface_figure(
                rows, 'f_learn', 'viridis',
                'Fitness landscape slices, RL ON — identical cells and identical '
                f'maps, in-lifetime learning enabled{env_note}\n{viab_note}',
                f'f(θ) with RL on{unit}',
                out, annotate=pct_on, viab=contour, run_f=run_f)

        for out in paths(out_dir, 'slices_lift', shown):
            # No viability contour here: the threshold is a level on f(θ), and
            # this panel shows a DIFFERENCE of two f(θ) values. A contour at
            # 0.136 on a lift surface would mean nothing.
            _surface_figure(
                rows, 'lift', 'RdBu_r',
                'What learning adds:  lift = f(θ)|RL on − f(θ)|RL off, same θ, '
                f'same 5 maps{env_note}',
                f'lift  (food score added by in-lifetime learning){unit}',
                out, annotate=med_lift, norm_kind='diverging', run_f=run_f)

        for out in paths(out_dir, 'rl_effect', shown):
            fig_rl_effect(rows, out, threshold, unit)
        recs = None
        for out in paths(out_dir, 'outcome_endpoints', shown):
            recs = fig_outcome_endpoints(rows, out, threshold, unit, env_note)
        if recs:
            write_endpoint_csv(recs, os.path.join(out_dir, 'endpoints.csv'))
        write_summary(rows, os.path.join(out_dir, 'slice_summary.csv'),
                      threshold, rng is not None)


if __name__ == '__main__':
    main()
