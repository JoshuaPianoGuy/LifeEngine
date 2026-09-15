"""
assimilation/analyse_assimilation.py
======================================================================
Genetic assimilation: does an inherited genome get better on its own?

Reads the founder-probe results built by make_founder_jobs.py and run through
slurm/run_founder_probe_array.slurm — every logged founder genome re-evaluated as a
monomorphic 100-clone cohort with the EA off, once with in-lifetime learning
disabled and once with it enabled, on the same maps.

    f_off  what the INHERITED weights do alone      <- the primary measurement
    f_on   what they do with learning switched on
    lift   f_on - f_off, paired per map

WHAT IS CLEAN HERE AND WHAT IS NOT — READ THIS BEFORE QUOTING A NUMBER
-----------------------------------------------------------------------
The probe takes its RL hyperparameters from each run's own params.json. For the
LEARNING runs that is exactly right: the RL-on pass reproduces the learning those
lineages actually experienced. For the EVOLUTION runs it is not, because those
runs never ran RL at all and their params.json carries whatever the RL fields
were left at:

    evolution   lr=0.02   epsilon_enabled=TRUE    (both environments)
    learning    lr=0.02   epsilon_enabled=false   (hard)
    learning    lr=0.01   epsilon_enabled=false   (baseline)

epsilon_enabled=true is not a formality. NNBrain pins epsilon to 0 only when the
flag is false; otherwise it decays epsilon_start -> epsilon_end (0.3 -> 0.05)
across each organism's lifetime, injecting that fraction of RANDOM ACTIONS. So
the evolution arm's RL-on pass is "learning PLUS 30%-decaying-to-5% random
actions", and in the baseline it also runs at double the learning rate. That is a
different intervention from the one the learning arm received.

Consequences, and they are not symmetric:

  f_off        CLEAN for both arms. No RL runs at all, so learning_rate and
               epsilon are inert. Every f_off number and every behaviour
               measured on the RL-off pass is valid as it stands.
  f_on, lift   CLEAN for the LEARNING arm — its settings are the ones its
               lineages evolved under.
               CONFOUNDED for the EVOLUTION arm. Its negative lift is not
               evidence that learning damages an unadapted genome; random-action
               injection alone would produce it.

So the within-learning-arm assimilation signature is fully measurable now, and
the cross-condition lift contrast needs the evolution arm's RL-on pass re-run
with matched hyperparameters. Figures built on the confounded quantity are
written with CONFOUNDED in the title rather than left out, because the shape is
still informative once the reader knows what produced it.

THE SIGNATURE, AND WHY THE RATIO IS THE STATISTIC
--------------------------------------------------
Assimilation is the innate phenotype catching up to the learned one. Within the
learning arm both curves rise, so neither alone says anything; what matters is
which rises FASTER:

    delta_f_off / delta_f_on  > 1   innate gaining on learned  -> assimilation
                              ~ 1   both improving together    -> no assimilation
                              < 1   learned pulling away

Equivalently, the lift shrinks. That ratio is computed per SEED and the spread
across seeds is the error bar, because the seed is the unit of replication —
100 founders inside one run share a lineage and are not independent.

Usage
-----
  python assimilation/analyse_assimilation.py \\
      --units hard:assimilation/out/founders_hard \\
      --units baseline:assimilation/out/founders_baseline \\
      --out-dir assimilation/out/analysis
"""

import argparse
import glob
import json
import os
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#374151'
GRID_C = '#D1D5DB'
COND_COLOUR = {'evolution': '#D98A2B', 'learning': '#152A47'}  # grey 154 / 46
COND_STYLE  = {'evolution': ('--', 'o'), 'learning': ('-', 's')}
ENV_ORDER = ['baseline', 'hard']

# Display names. The keys stay as they are — they are the --units labels and a
# column in every CSV — while the figures say what the environment actually is.
# Matches landscape/plot_slices.py ENV_TITLE.
ENV_TITLE = {'hard': 'predator', 'baseline': 'baseline'}


def env_title(env):
    return ENV_TITLE.get(env, env)

# Behaviours the probe records, all of them read off the RL-OFF pass where they
# are clean for both arms.
BEHAVIOURS = [
    ('mean_lifetime',          'Lifetime (ticks)'),
    ('mean_cave_entries',      'Cave entries per organism'),
    ('mean_predator_touches',  'Predator contacts per organism'),
    ('mean_drained_ticks',     'Ticks drained by predators'),
]

VIABILITY = 4.375


def load(units):
    """Every probe row, joined to the generation/founder it came from."""
    frames = []
    for env, root in units:
        for d in sorted(glob.glob(os.path.join(root, '*/'))):
            rp = os.path.join(d, 'results.csv')
            mp = os.path.join(d, 'meta.json')
            if not (os.path.exists(rp) and os.path.exists(mp)):
                continue
            m = json.load(open(mp))
            df = pd.read_csv(rp)
            idx = {g['genome']: g for g in m['genome_index']}
            df['genome'] = df['id'].str.replace(r'_r\d+_(on|off)$', '', regex=True)
            df['generation'] = df['genome'].map(lambda g: idx[g]['generation'])
            df['founder_index'] = df['genome'].map(lambda g: idx[g]['founder_index'])
            df['env'] = env
            df['condition'] = m['condition']
            df['seed'] = m['seed']
            df['run'] = m['run']
            df['run_final_fitness'] = m['run_final_fitness']
            df['rl_matched'] = bool(m.get('rl_matched', False))
            # Present only in results produced by a run_probe.js that supports
            # "metrics": "founder". The first cluster run predated it, so this
            # column is missing from those units and has to be tolerated rather
            # than required — see the stale-binary note in the slurm header.
            if 'mean_weight_drift' not in df.columns:
                df['mean_weight_drift'] = np.nan
            frames.append(df)
    if not frames:
        raise SystemExit('no results.csv found — run merge_shards.py first')
    out = pd.concat(frames, ignore_index=True)

    # LATER --units ROOTS OVERRIDE EARLIER ONES for the same probe. That is how a
    # correction is layered on: an --rl-match re-run emits only the RL-on pass
    # for the control, into its own root, and listing it after the original makes
    # those rows replace the confounded ones while every RL-off row and the whole
    # learning arm stay exactly as they were. Order of --units is therefore
    # significant, and the count of replaced rows is printed so a correction that
    # silently matched nothing is visible.
    key = ['env', 'condition', 'seed', 'generation', 'founder_index',
           'map_index', 'rl']
    before = len(out)
    out = out.drop_duplicates(subset=key, keep='last').reset_index(drop=True)
    if before != len(out):
        print(f'  {before - len(out):,} probe rows overridden by a later --units root')
    return out


def pair_passes(raw):
    """One row per (genome, map): the two passes side by side.

    Pairing on map_index is the whole point — the two passes ran on IDENTICAL
    terrain, so the difference cancels it. Terrain variance dwarfs the lift, and
    an unpaired difference of means would be mostly noise.
    """
    keys = ['env', 'condition', 'seed', 'run', 'run_final_fitness',
            'generation', 'founder_index', 'genome', 'map_index']
    # rl_matched belongs to the RL-ON pass, so it is carried on that side only.
    raw = raw.copy()
    cols = ['mean_fitness'] + [c for c, _ in BEHAVIOURS] + ['mean_weight_drift']
    off = raw[raw.rl == 0].set_index(keys)[cols].add_suffix('_off')
    on = raw[raw.rl == 1].set_index(keys)[cols + ['rl_matched']].add_suffix('_on')
    p = off.join(on, how='inner').reset_index()
    p['rl_matched'] = p['rl_matched_on']
    p['f_off'] = p['mean_fitness_off']
    p['f_on'] = p['mean_fitness_on']
    p['lift'] = p['f_on'] - p['f_off']
    return p


def normalisation_range(p):
    """One min-max range over every probed value, both passes, both environments.

    Same affine map and the same GLOBAL scope as
    analyse_conditions_by_environment.py: rescaling each environment against its
    own range would delete the gap between them, which is the comparison the
    side-by-side panels exist to make. f_off and f_on share the range because
    they are the same measurement and their difference has to stay meaningful.
    """
    v = np.concatenate([p['f_off'].to_numpy(dtype=float),
                        p['f_on'].to_numpy(dtype=float)])
    v = v[np.isfinite(v)]
    return float(v.min()), float(v.max())


def per_seed(p):
    """Seed-level means. The seed is the unit of replication, not the founder.

    The 100 founders inside one dump share a lineage and a map pool, so they are
    replicates of the same run, not independent samples. Collapsing to the seed
    first is what makes the across-seed spread an honest error bar.
    """
    g = (p.groupby(['env', 'condition', 'seed', 'generation'])
           .agg(f_off=('f_off', 'mean'), f_on=('f_on', 'mean'),
                lift=('lift', 'mean'),
                weight_drift=('mean_weight_drift_on', 'mean'),
                **{f'{c}_off': (f'{c}_off', 'mean') for c, _ in BEHAVIOURS},
                **{f'{c}_on': (f'{c}_on', 'mean') for c, _ in BEHAVIOURS},
                n_genomes=('genome', 'nunique'))
           .reset_index())
    return g


def attach_gen0(seed_tbl):
    """Fold the generation-0 anchor into BOTH arms' curves.

    The gen-0 units are built one per SEED, not one per condition, because at
    generation 0 there is no condition: both arms' founders are drawn from the
    same Xavier distribution on the same terrain, and GAManager spawns exactly
    that when gene_pool is null. Plotting two separate gen-0 points would draw a
    difference that cannot exist.

    So the single row is copied into both conditions to give each curve its
    origin, and flagged is_anchor so the figures can mark it as one shared
    measurement rather than two. If both arms were probed at gen 0 anyway (they
    are not, by design), their f_off agreeing would be the validity check.
    """
    if 'gen0' not in set(seed_tbl.condition):
        return seed_tbl, False
    anchor = seed_tbl[seed_tbl.condition == 'gen0']
    rest = seed_tbl[seed_tbl.condition != 'gen0'].copy()
    rest['is_anchor'] = False
    copies = []
    for cond in sorted(set(rest.condition)):
        c = anchor.copy()
        c['condition'] = cond
        c['is_anchor'] = True
        copies.append(c)
    out = pd.concat([rest] + copies, ignore_index=True)
    return out.sort_values(['env', 'condition', 'seed', 'generation']), True


def assimilation_ratio(seed_tbl):
    """delta_f_off / delta_f_on per seed, first dump to last.

    > 1 means the innate phenotype gained more than the learned one over the
    same span, which is what assimilation predicts. Reported per condition with
    the across-seed spread; only the LEARNING arm's value is interpretable,
    since the evolution arm's f_on is the confounded pass.
    """
    rows = []
    # Dumps only. Spanning from generation 0 would measure the climb out of a
    # random brain, where learning's contribution is enormous and the ratio says
    # nothing about assimilation between established genomes.
    seed_tbl = seed_tbl[seed_tbl.generation > 0]
    for (env, cond, seed), sub in seed_tbl.groupby(['env', 'condition', 'seed']):
        sub = sub.sort_values('generation')
        if len(sub) < 2:
            continue
        a, b = sub.iloc[0], sub.iloc[-1]
        d_off, d_on = b['f_off'] - a['f_off'], b['f_on'] - a['f_on']
        rows.append({'env': env, 'condition': cond, 'seed': seed,
                     'gen_from': int(a['generation']), 'gen_to': int(b['generation']),
                     'd_f_off': d_off, 'd_f_on': d_on,
                     'ratio': (d_off / d_on) if d_on != 0 else np.nan,
                     'lift_from': a['lift'], 'lift_to': b['lift'],
                     'd_lift': b['lift'] - a['lift']})
    return pd.DataFrame(rows)


def _one_sample(v, conf=0.95):
    """Mean, CI, Cohen's d, paired-t and Wilcoxon for one column of seed values.

    The Wilcoxon test STATISTIC is kept, not just its p-value: a table that
    reports a test is expected to show what was computed, and the signed-rank
    statistic is the only thing a reader can check the p against by hand at
    n = 10.
    """
    from scipy import stats
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    k = v.size
    if k < 3:
        return None
    sd = v.std(ddof=1)
    sem = sd / np.sqrt(k)
    tcrit = float(stats.t.ppf(0.5 + conf / 2, k - 1))
    tstat, tp = stats.ttest_1samp(v, 0.0)
    try:
        w, wp = stats.wilcoxon(v)
    except ValueError:
        w, wp = np.nan, np.nan
    return {'n_seeds': k, 'mean': v.mean(), 'sd': sd, 'sem': sem,
            'w_stat': float(w),
            'ci95_lo': v.mean() - tcrit * sem, 'ci95_hi': v.mean() + tcrit * sem,
            'cohens_d': v.mean() / sd if sd > 0 else np.nan,
            't_stat': float(tstat), 'p_t': float(tp), 'p_wilcoxon': float(wp),
            'n_positive': int((v > 0).sum()), 'n_negative': int((v < 0).sum())}


def assimilation_tests(ratios, matched=None, n_boot=10000, seed=20260831):
    """Significance tests for the assimilation signature. The seed is the unit.

    WHY THE RATIO IS NOT THE TEST STATISTIC, THOUGH IT IS THE HEADLINE NUMBER
    -------------------------------------------------------------------------
    `ratio = delta_f_off / delta_f_on` is the right thing to *quote* and the wrong
    thing to *test*. Its denominator is a difference that is not bounded away from
    zero: on this data two of ten seeds in baseline/evolution and two of ten in
    hard/learning have a non-positive numerator or denominator, which flips the
    ratio negative (-0.054, -0.049) for reasons that have nothing to do with
    assimilation. A mean and a t-test over per-seed ratios inherits that, so the
    ratio is bootstrapped as a RATIO OF MEANS — mean(d_f_off) / mean(d_f_on) over
    resampled whole seeds — which has no such singularity, and the hypothesis
    "ratio > 1" is read off whether the interval clears 1.

    THE TESTABLE FORM OF THE SAME CLAIM
    ------------------------------------
    Assimilation is the innate phenotype catching up to the learned one, so the
    learning advantage shrinks:

        d_lift = lift(gen_to) - lift(gen_from) < 0

    That is a plain difference of two paired measurements, well-behaved for every
    seed, and it is the primary test here. `d_f_off > 0` is reported beside it
    because the two together separate the two ways d_lift can go negative:
    innate rising to meet learned (assimilation) versus learned falling.

    WHAT IS CLEAN. `d_f_off` is ALWAYS clean — the RL-off pass runs no RL at all,
    so a mismatched epsilon or learning rate is inert there. Whether `d_lift`,
    `d_f_on` and the ratio are clean depends on the provenance of that cell's
    RL-ON pass, passed in as `matched` (the per-cell mean of the `rl_matched`
    flag). That is a property of WHICH PROBE produced the rows, not of the arm's
    name: an --rl-match re-run layered over the original makes the evolution
    arm's RL-on pass comparable, and these tests have to notice that rather than
    assume the control is confounded forever. With no `matched` given, only the
    learning arm counts as clean — the state before such a re-run exists.
    """
    rng = np.random.default_rng(seed)
    matched = {} if matched is None else dict(matched)
    rows = []
    for (env, cond), g in ratios.groupby(['env', 'condition']):
        # The LEARNING arm must never be rl-matched — its own params.json holds
        # the settings its lineages actually evolved under, so overriding them
        # would be the error, not the fix. It is clean exactly when it was left
        # alone. Every other arm is the reverse: clean only once an --rl-match
        # re-run has replaced its RL-on pass. (Same rule as the provenance block
        # in main(); rl_matched means "came from a re-run", not "is correct".)
        # < 0.001 / < 0.999 rather than == 0 / == 1 so a cell whose re-run
        # covered only some genomes is flagged, not rounded clean.
        frac = matched.get((env, cond), 0.0)
        conf = (frac > 0.001) if cond == 'learning' else (frac < 0.999)
        for col, claim, confounded in (
                ('d_f_off', 'inherited genome improves on its own', False),
                ('d_lift', 'learning advantage shrinks', conf),
                ('d_f_on', 'learned phenotype improves', conf)):
            st = _one_sample(g[col])
            if st is None:
                continue
            rows.append({'env': env, 'condition': cond, 'quantity': col,
                         'claim': claim, 'confounded': confounded, **st})

        # Ratio of means, seed-clustered bootstrap.
        off = g['d_f_off'].to_numpy(dtype=float)
        on = g['d_f_on'].to_numpy(dtype=float)
        ok = np.isfinite(off) & np.isfinite(on)
        off, on = off[ok], on[ok]
        k = off.size
        if k >= 3:
            boot = np.empty(n_boot)
            for b in range(n_boot):
                pick = rng.integers(0, k, k)
                den = on[pick].mean()
                boot[b] = off[pick].mean() / den if den != 0 else np.nan
            boot = boot[np.isfinite(boot)]
            den = on.mean()
            rows.append({
                'env': env, 'condition': cond, 'quantity': 'ratio_of_means',
                'claim': 'innate gaining on learned (>1)',
                'confounded': conf,
                'n_seeds': k, 'mean': (off.mean() / den) if den != 0 else np.nan,
                'sd': np.nan, 'sem': np.nan,
                'ci95_lo': float(np.percentile(boot, 2.5)) if boot.size else np.nan,
                'ci95_hi': float(np.percentile(boot, 97.5)) if boot.size else np.nan,
                'cohens_d': np.nan, 't_stat': np.nan,
                'p_t': np.nan, 'p_wilcoxon': np.nan,
                'n_positive': int((g['ratio'] > 1).sum()),
                'n_negative': int((g['ratio'] <= 1).sum())})
    return pd.DataFrame(rows)


def contrast_arms(ratios, col='d_f_off'):
    """Learning vs evolution on the same seeds, paired.

    Only `d_f_off` is a fair contrast: it is the one quantity measured
    identically in both arms (no RL runs on that pass, so the evolution arm's
    epsilon/learning-rate mismatch cannot touch it).
    """
    from scipy import stats
    rows = []
    for env, g in ratios.groupby('env'):
        a = g[g.condition == 'learning'].set_index('seed')[col]
        b = g[g.condition == 'evolution'].set_index('seed')[col]
        shared = sorted(set(a.index) & set(b.index))
        if len(shared) < 3:
            continue
        da, db = a.loc[shared].to_numpy(float), b.loc[shared].to_numpy(float)
        st = _one_sample(da - db)
        rows.append({'env': env, 'quantity': col, 'arm_a': 'learning',
                     'arm_b': 'evolution', 'mean_a': da.mean(),
                     'mean_b': db.mean(), **st})
    return pd.DataFrame(rows)


def _holm(ps):
    """Holm step-down adjustment over one family. Returns adjusted p in input order."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty(ps.size)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def contrast_arms_by_generation(seed_tbl, col='f_off'):
    """Learning vs evolution at EACH probe generation, paired by seed.

    contrast_arms() tests d_f_off — the whole generation-250-to-1000 CHANGE — and
    that is the right statistic for "did the inherited genome improve more in one
    arm". It is the wrong one for the claim the innate_fitness figure actually
    invites, which is about the LEVEL of the two curves at a point: "the arms
    separate early and have converged by generation 1000". A difference in levels
    that opens and then closes leaves d_f_off near zero, so the change-based test
    reports nothing where the figure plainly shows something.

    So this is the per-generation companion: at each dump generation, the same
    seeds, paired, learning minus evolution on f_off. Positive means the LEARNING
    arm's inherited genomes score higher with RL off.

    f_off is the only quantity worth contrasting across arms — it is measured with
    no RL running at all, so the evolution arm's epsilon/learning-rate mismatch
    (see the module docstring) cannot reach it. f_on and lift are not comparable
    between arms and are deliberately not offered here.

    MULTIPLICITY. The four generations are one family per environment, so p is
    Holm-adjusted within each environment and both the raw and adjusted values are
    kept. Reading the four raw p-values as four independent findings is exactly the
    error the adjustment exists to stop: on this data the baseline generation-250
    gap is p = 0.049 raw and p = 0.12 adjusted, which is the difference between
    "evolution is ahead early" and "we cannot say that".
    """
    rows = []
    for env, g in seed_tbl[seed_tbl.generation > 0].groupby('env'):
        env_rows = []
        for gen, sub in g.groupby('generation'):
            a = sub[sub.condition == 'learning'].set_index('seed')[col]
            b = sub[sub.condition == 'evolution'].set_index('seed')[col]
            shared = sorted(set(a.index) & set(b.index))
            if len(shared) < 3:
                continue
            da = a.loc[shared].to_numpy(dtype=float)
            db = b.loc[shared].to_numpy(dtype=float)
            st = _one_sample(da - db)
            if st is None:
                continue
            env_rows.append({'env': env, 'generation': int(gen), 'quantity': col,
                             'arm_a': 'learning', 'arm_b': 'evolution',
                             'mean_a': da.mean(), 'mean_b': db.mean(), **st})
        if not env_rows:
            continue
        for r, adj in zip(env_rows, _holm([r['p_t'] for r in env_rows])):
            r['p_holm'] = float(adj)
            r['n_generations_in_family'] = len(env_rows)
        rows.extend(env_rows)
    return pd.DataFrame(rows)


# Figure footnotes are OFF by default. Every caveat they carried is stated in
# this module's docstring and belongs in the surrounding prose of a write-up,
# where it can be edited; baked into the image it cannot be, and it forces the
# figure into a letterbox because bbox_inches='tight' grows the canvas to fit it.
# --captions puts them back for a standalone figure that has to travel alone.
CAPTIONS = False


def _caption(fig, text, width=104, y=-.06):
    """Footnote, hard-wrapped — only when --captions is on."""
    if not CAPTIONS:
        return
    fig.text(.5, y, textwrap.fill(text, width), ha='center', va='top',
             fontsize=8.5, color=INK, linespacing=1.5)


def _gen_axis(ax, gens):
    """Generation axis anchored at 0.

    The founder dumps start at 250 (Logger.js FULL_GENOME_DUMP_EVERY_N_GENS), so
    without this the leftmost tick is 250 and the curves appear to begin from a
    already-competent state — the rise from an unevolved brain is off the page.
    Starting the axis at 0 keeps the missing stretch visible as a gap rather than
    hiding it, and it is where the --gen0 Xavier anchor lands when it is probed.
    """
    ax.set_xlim(0, max(gens) * 1.04)
    ax.set_xticks([0] + sorted(g for g in gens if g > 0))


def _style(ax):
    ax.grid(True, axis='y', alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK, labelsize=8)


def _band(ax, sub, col, colour, label, ls=None, marker=None):
    """Seed mean with a +/-1 SD band across seeds.

    Line style and marker shape are redundant with colour on purpose: the two
    arms have to stay separable in a black-and-white printout, where hue is
    gone and only luminance and texture survive.
    """
    dls, dmk = COND_STYLE.get(label, ('-', 'o'))
    g = sub.groupby('generation')[col]
    m, sd, gens = g.mean(), g.std(ddof=1), sorted(sub.generation.unique())
    ax.plot(gens, m.loc[gens], color=colour, lw=2, ls=(ls or dls),
            marker=(marker or dmk), ms=4, label=label, zorder=3)
    ax.fill_between(gens, (m - sd).loc[gens], (m + sd).loc[gens],
                    color=colour, alpha=.15, lw=0, zorder=2)


def fig_innate(seed_tbl, rng, out_png):
    """The primary figure: what the INHERITED genome does on its own."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.1), sharey=True)
    for ax, env in zip(axs, ENV_ORDER):
        sub = seed_tbl[seed_tbl.env == env]
        for cond in ('evolution', 'learning'):
            s = sub[sub.condition == cond]
            if len(s):
                _band(ax, s, 'f_off_norm', COND_COLOUR[cond], cond)
        # One black marker at the origin: the arms share it, so two coloured
        # points would imply a gen-0 difference that cannot exist.
        a = sub[sub.get('is_anchor', False) == True] if 'is_anchor' in sub else sub.iloc[:0]
        if len(a):
            # The marker stays — the two arms genuinely share this point and the
            # diamond says so by being one shape rather than two. The label that
            # explained it is gone: what generation 0 is belongs in the caption,
            # where it can be written once instead of twice on the page.
            m = a.groupby('generation')['f_off_norm'].mean()
            ax.scatter(m.index, m.values, s=70, marker='D', color=INK,
                       edgecolors='white', linewidths=1.1, zorder=6)
        thr = (VIABILITY - rng[0]) / (rng[1] - rng[0])
        ax.axhline(thr, color=INK, lw=1, ls=':')
        ax.annotate(f'viability threshold {thr:.3f}', xy=(0, thr), xytext=(2, 4),
                    textcoords='offset points', fontsize=7.5, color=INK)
        ax.set_title(f'{env_title(env)} environment', fontsize=11, color=INK)
        ax.set_xlabel('generation of the founder dump', fontsize=9, color=INK)
        _gen_axis(ax, sorted(seed_tbl.generation.unique()))
        _style(ax)
    axs[0].set_ylabel('f_off — inherited genome, learning OFF\n(normalised)',
                      fontsize=9, color=INK)
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=INK,
                  loc='lower right', handlelength=1.8, handletextpad=0.6,
                  labelspacing=0.35, borderaxespad=0.6)
    _caption(fig,
             'Seed means with ±1 SD across the 10 seeds. This pass runs no RL at '
             'all, so it is unaffected by the hyperparameter mismatch that '
             'confounds the evolution arm’s RL-on pass. The generation-0 point is a '
             'RESAMPLE of the Xavier distribution the real founders were drawn '
             'from, not a replay of them — its error bar carries terrain and '
             'Monte-Carlo noise only, since no lineage has diverged yet.')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', pad_inches=0.01,
                facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def fig_gap(seed_tbl, ratios, out_png):
    """The assimilation signature, inside the learning arm where it is clean."""
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.6))

    for ax, env in zip(axs[:2], ENV_ORDER):
        s = seed_tbl[(seed_tbl.env == env) & (seed_tbl.condition == 'learning')]
        if not len(s):
            continue
        _band(ax, s, 'f_on_norm', '#7C3AED', 'f_on — with learning', ls='--')
        _band(ax, s, 'f_off_norm', '#2563EB', 'f_off — inherited alone')
        gens = sorted(s.generation.unique())
        mo = s.groupby('generation')['f_off_norm'].mean().loc[gens]
        mn = s.groupby('generation')['f_on_norm'].mean().loc[gens]
        ax.fill_between(gens, mo, mn, color='#F59E0B', alpha=.30, lw=0, zorder=1)
        for g, a, b in zip(gens, mo, mn):
            ax.annotate(f'{b - a:.3f}', xy=(g, (a + b) / 2), fontsize=7.5,
                        color='#B45309', ha='center', va='center')
        ax.set_title(f'{env_title(env)} — learning condition', fontsize=11,
                     color=INK)
        ax.set_xlabel('generation of the founder dump', fontsize=9, color=INK)
        _gen_axis(ax, gens)
        ax.set_ylabel('normalised f(θ)', fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='lower right')
        _style(ax)

    ax = axs[2]
    r = ratios[ratios.condition == 'learning']
    for i, env in enumerate(ENV_ORDER):
        v = r[r.env == env]['ratio'].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        ax.scatter(np.full(v.size, i) + np.linspace(-.14, .14, v.size), v, s=60,
                   color='#2563EB', edgecolors='white', linewidths=.8, zorder=3)
        ax.plot([i - .28, i + .28], [v.mean()] * 2, color='#2563EB', lw=2.4, zorder=4)
        ax.annotate(f'{v.mean():.2f}', xy=(i + .32, v.mean()), fontsize=9,
                    color='#2563EB', va='center')
    ax.axhline(1.0, color=INK, lw=1.2, ls=':')
    ax.annotate('1.0 — innate and learned improving equally', xy=(-.45, 1.0),
                xytext=(0, 5), textcoords='offset points', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(ENV_ORDER)))
    ax.set_xticklabels([env_title(e) for e in ENV_ORDER], fontsize=9,
                       color=INK)
    ax.set_xlim(-.5, len(ENV_ORDER) - .1)
    ax.set_ylabel('Δf_off / Δf_on  (gen 250 → 1000)', fontsize=9, color=INK)
    ax.set_title('Which rose faster, innate or learned?\n> 1 = the innate '
                 'phenotype gained more', fontsize=10.5, color=INK)
    _style(ax)

    fig.suptitle('The assimilation signature, inside the learning condition',
                 fontsize=13, color=INK, y=1.01)
    _caption(fig,
             'Both passes here use the learning runs’ own RL settings, so this '
             'contrast is clean. The shaded gap is the lift — what learning still '
             'adds once the genome has done its part.')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def fig_behaviours(seed_tbl, out_png):
    """Behaviours of the inherited genome — RL-off pass, clean for both arms."""
    fig, axs = plt.subplots(len(BEHAVIOURS), 2, figsize=(11.2, 3.0 * len(BEHAVIOURS)),
                            squeeze=False)
    for r, (col, label) in enumerate(BEHAVIOURS):
        for c, env in enumerate(ENV_ORDER):
            ax = axs[r][c]
            sub = seed_tbl[seed_tbl.env == env]
            for cond in ('evolution', 'learning'):
                s = sub[sub.condition == cond]
                if len(s):
                    _band(ax, s, f'{col}_off', COND_COLOUR[cond], cond)
            _gen_axis(ax, sorted(seed_tbl.generation.unique()))
            _style(ax)
            if r == 0:
                ax.set_title(f'{env_title(env)} environment', fontsize=11, color=INK)
            if r == len(BEHAVIOURS) - 1:
                ax.set_xlabel('generation of the founder dump', fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
    axs[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Behaviour of the inherited genome, learning OFF', fontsize=13,
                 color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def fig_lift(seed_tbl, out_png):
    """Both arms' lift — with the evolution arm marked as what it is."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, env in zip(axs, ENV_ORDER):
        sub = seed_tbl[seed_tbl.env == env]
        for cond in ('evolution', 'learning'):
            s = sub[sub.condition == cond]
            if len(s):
                _band(ax, s, 'lift_norm', COND_COLOUR[cond], cond)
        ax.axhline(0, color=INK, lw=1.1, ls=':')
        ax.set_title(f'{env_title(env)} environment', fontsize=11, color=INK)
        ax.set_xlabel('generation of the founder dump', fontsize=9, color=INK)
        _gen_axis(ax, sorted(seed_tbl.generation.unique()))
        _style(ax)
    axs[0].set_ylabel('lift = f_on − f_off  (normalised)', fontsize=9, color=INK)
    axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('What learning still adds, by arm', fontsize=13, color=INK, y=1.0)
    _caption(fig,
             'The evolution runs never ran RL, so their params.json left '
             'epsilon_enabled=TRUE: their RL-on pass injects 30%→5% random actions '
             '(and in the baseline runs at double the learning rate). Their '
             'negative lift is that, not a property of unadapted genomes. Only the '
             'learning curve is quotable.')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--units', action='append', required=True, metavar='ENV:PATH',
                    help='Repeatable. ORDER MATTERS: a later root overrides an '
                         'earlier one for the same probe, which is how an '
                         '--rl-match correction is layered over the original run.')
    ap.add_argument('--out-dir', default='assimilation/out/analysis')
    ap.add_argument('--format', default='png', choices=('png', 'pdf', 'both'),
                    help='Figure format. pdf keeps every line and label as a '
                         'vector, which is what a thesis wants.')
    ap.add_argument('--captions', action='store_true',
                    help='Print the explanatory footnote under each figure. Off '
                         'by default — see CAPTIONS.')
    args = ap.parse_args()

    global CAPTIONS
    CAPTIONS = args.captions
    exts = ('png', 'pdf') if args.format == 'both' else (args.format,)

    units = []
    for spec in args.units:
        env, path = spec.split(':', 1)
        units.append((env, path))
    os.makedirs(args.out_dir, exist_ok=True)

    raw = load(units)
    print(f'loaded {len(raw):,} probe rows')
    p = pair_passes(raw)
    print(f'paired into {len(p):,} (genome, map) rows')

    # Say plainly whether the control's RL-on pass is the corrected one, since
    # every lift number downstream is only comparable across arms if it is.
    print('\n  RL-on pass provenance (rl_matched = built with the learning '
          'arm\'s RL settings):')
    # Per (env, condition, generation), not just per arm: a correction truncated
    # on walltime loses a CONTIGUOUS block of the job list, and because on-only
    # units are emitted in founder order that block is the LATEST generations —
    # exactly the endpoint every trend rests on. An arm-level mean would show
    # "75% matched" and hide that the last dump is barely covered at all.
    bygen = p.groupby(['env', 'condition', 'generation'])['rl_matched'].mean()
    partial = bygen[(bygen > 0) & (bygen < 1)]
    if len(partial):
        print('    partial rl-match coverage by generation '
              '(mixed provenance — not averaged):')
        for (env, cond, gen), frac in partial.items():
            print(f'      {env:<9} {cond:<10} gen {gen:>4}  {frac:.0%} matched')

    prov = p.groupby(['env', 'condition'])['rl_matched'].mean()
    ok = True
    for (env, cond), frac in prov.items():
        if cond == 'learning':
            # The learning arm must NOT be matched: its own params are the
            # settings its lineages actually evolved under, and overriding them
            # would be the error, not the fix.
            tag = ('its own settings — correct' if frac == 0 else
                   f'OVERRIDDEN ({frac:.0%}) — this arm should NOT be rl-matched')
            ok = ok and frac == 0
        else:
            tag = ('rl-matched to the learning arm — lift IS comparable'
                   if frac == 1 else
                   'NOT rl-matched — its RL-on pass injects epsilon-greedy '
                   'noise, so lift is NOT comparable across arms' if frac == 0
                   else f'PARTIAL ({frac:.0%}) — mixed provenance, do not quote lift')
            ok = ok and frac == 1
        print(f'    {env:<9} {cond:<10} {tag}')
    print('    => cross-arm lift contrast is '
          + ('USABLE' if ok else 'NOT usable; only the within-learning-arm '
                                 'signature and every f_off number are'))

    rng = normalisation_range(p)
    span = rng[1] - rng[0]
    for c in ('f_off', 'f_on'):
        p[c + '_norm'] = (p[c] - rng[0]) / span
    p['lift_norm'] = p['lift'] / span          # a difference carries no offset
    with open(os.path.join(args.out_dir, 'normalisation.csv'), 'w') as fh:
        fh.write('scope,metric,lo,hi,span,viability_4.375_normalised\n')
        for m in ('f_off', 'f_on'):
            fh.write(f'global,{m},{rng[0]!r},{rng[1]!r},{span!r},'
                     f'{(VIABILITY - rng[0]) / span!r}\n')
        fh.write(f'global,lift,0.0,{span!r},{span!r},\n')
    print(f'  normalised on [{rng[0]:.4f}, {rng[1]:.4f}] '
          f'-> viability threshold at {(VIABILITY - rng[0]) / span:.4f}')

    # An arm whose RL-on pass is only PARTLY corrected cannot have a lift: the
    # matched and confounded rows are different measurements, and averaging them
    # produces a number that is neither. Blank the lift for those arms and keep
    # everything else, which costs nothing — f_off is untouched by any of this.
    mixed = {(e, c) for (e, c), f in
             p.groupby(['env', 'condition'])['rl_matched'].mean().items()
             if c != 'learning' and 0 < f < 1}
    if mixed:
        key = list(zip(p['env'], p['condition']))
        drop = np.array([k in mixed for k in key])
        p.loc[drop, ['f_on', 'lift']] = np.nan
        for e, c in sorted(mixed):
            print(f'    [held back] {e}/{c}: lift suppressed until the rl-match '
                  f're-run completes ({int(drop.sum()):,} rows)')

    seed_tbl = per_seed(p)
    for c in ('f_off', 'f_on'):
        seed_tbl[c + '_norm'] = (seed_tbl[c] - rng[0]) / span
    seed_tbl['lift_norm'] = seed_tbl['lift'] / span
    seed_tbl, has_anchor = attach_gen0(seed_tbl)
    # How much learning had to move the weights — the third assimilation
    # signature, and the one the stale binary cost us on the first run.
    drift = p.groupby(['env', 'condition'])['mean_weight_drift_on'].apply(
        lambda v: float(np.isfinite(v).mean()))
    print('\n  mean_weight_drift coverage (RL-on pass):')
    for (env, cond), frac in drift.items():
        print(f'    {env:<9} {cond:<10} {frac:.0%}'
              + ('' if frac == 1 else
                 '  — from a run_probe.js predating "metrics": "founder"'))

    print('  generation-0 Xavier anchor: '
          + ('present, shared by both arms' if has_anchor else
             'ABSENT — curves start at the first dump (250). Build it with '
             'make_founder_jobs.py --gen0 100 --gen0-only --rl-match'))

    ratios = assimilation_ratio(seed_tbl)

    p.to_csv(os.path.join(args.out_dir, 'per_genome.csv.gz'), index=False,
             compression='gzip')
    seed_tbl.to_csv(os.path.join(args.out_dir, 'seed_summary.csv'), index=False)
    ratios.to_csv(os.path.join(args.out_dir, 'assimilation_ratio.csv'), index=False)

    matched = p.groupby(['env', 'condition'])['rl_matched'].mean().to_dict()
    tests = assimilation_tests(ratios, matched=matched)
    arms = contrast_arms(ratios)
    by_gen = contrast_arms_by_generation(seed_tbl)
    tests.to_csv(os.path.join(args.out_dir, 'assimilation_tests.csv'), index=False)
    arms.to_csv(os.path.join(args.out_dir, 'assimilation_arm_contrast.csv'), index=False)
    by_gen.to_csv(os.path.join(args.out_dir, 'assimilation_arm_by_generation.csv'),
                  index=False)

    print('\n=== significance, seed as the unit (k = 10 per cell) ===')
    print(f'  {"env":<9}{"cond":<11}{"quantity":<16}{"mean":>9}{"95% CI":>20}'
          f'{"d":>7}{"p(t)":>10}{"p(Wilc)":>10}   sign')
    for _i, r in tests.iterrows():
        ci = f'[{r["ci95_lo"]:+.3f}, {r["ci95_hi"]:+.3f}]'
        d = '     —' if not np.isfinite(r['cohens_d']) else f'{r["cohens_d"]:>6.2f}'
        pt = '        —' if not np.isfinite(r['p_t']) else f'{r["p_t"]:>9.2g}'
        pw = '        —' if not np.isfinite(r['p_wilcoxon']) else f'{r["p_wilcoxon"]:>9.2g}'
        flag = ' CONFOUNDED' if r['confounded'] else ''
        print(f'  {r["env"]:<9}{r["condition"]:<11}{r["quantity"]:<16}'
              f'{r["mean"]:>9.3f}{ci:>20}{d} {pt} {pw}   '
              f'{int(r["n_positive"])}+/{int(r["n_negative"])}-{flag}')
    print('  (d_lift < 0 is the assimilation direction; ratio_of_means > 1 likewise.')
    print('   CONFOUNDED marks a cell whose RL-ON pass was NOT built with the')
    print('   learning arm\'s RL settings; d_f_off never depends on that and is')
    print('   clean everywhere. See the provenance block above.)')

    if not arms.empty:
        print('\n=== learning vs evolution on d_f_off, paired by seed (the clean contrast) ===')
        for _i, r in arms.iterrows():
            ci = f'[{r["ci95_lo"]:+.3f}, {r["ci95_hi"]:+.3f}]'
            print(f'  {r["env"]:<9} learning {r["mean_a"]:>7.3f}  vs  evolution '
                  f'{r["mean_b"]:>7.3f}   Δ {r["mean"]:>+7.3f}  {ci}  '
                  f'dz {r["cohens_d"]:>5.2f}  p(t) {r["p_t"]:.2g}  '
                  f'p(Wilc) {r["p_wilcoxon"]:.2g}')

    if not by_gen.empty:
        print('\n=== learning vs evolution on f_off AT EACH GENERATION, paired by seed ===')
        print(f'  {"env":<9}{"gen":>5}{"learning":>10}{"evolution":>11}{"Δ":>9}'
              f'{"95% CI":>20}{"dz":>7}{"p(t)":>9}{"p(Holm)":>9}   sign')
        for _i, r in by_gen.iterrows():
            ci = f'[{r["ci95_lo"]:+.3f}, {r["ci95_hi"]:+.3f}]'
            print(f'  {r["env"]:<9}{int(r["generation"]):>5}{r["mean_a"]:>10.3f}'
                  f'{r["mean_b"]:>11.3f}{r["mean"]:>+9.3f}{ci:>20}'
                  f'{r["cohens_d"]:>7.2f}{r["p_t"]:>9.2g}{r["p_holm"]:>9.2g}   '
                  f'{int(r["n_positive"])}+/{int(r["n_negative"])}-')
        print('  (Δ > 0 = the LEARNING arm\'s inherited genomes score higher with RL off.')
        print('   p is Holm-adjusted over the generations WITHIN each environment.)')

    print('\n=== f_off (inherited genome alone), seed mean ± SD, RAW units ===')
    t = (seed_tbl.groupby(['env', 'condition', 'generation'])['f_off']
         .agg(['mean', 'std']).round(3))
    print(t.to_string())

    print('\n=== assimilation ratio, learning arm (gen 250 → 1000) ===')
    for env in ENV_ORDER:
        r = ratios[(ratios.env == env) & (ratios.condition == 'learning')]
        v = r['ratio'].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        dl = r['d_lift'].to_numpy(dtype=float)
        if v.size:
            n_gt1 = int((v > 1).sum())
            print(f'  {env:<9} Δf_off/Δf_on = {v.mean():.3f} ± {v.std(ddof=1):.3f}'
                  f'   ({n_gt1}/{v.size} seeds > 1)   Δlift = '
                  f'{dl.mean():+.3f} ± {dl.std(ddof=1):.3f}')

    for e in exts:
        fig_innate(seed_tbl, rng, os.path.join(args.out_dir, f'innate_fitness.{e}'))
        fig_gap(seed_tbl, ratios,
                os.path.join(args.out_dir, f'assimilation_gap.{e}'))
        fig_behaviours(seed_tbl,
                       os.path.join(args.out_dir, f'behaviours_rl_off.{e}'))
        fig_lift(seed_tbl, os.path.join(args.out_dir, f'lift.{e}'))
    print(f'\n  wrote {args.out_dir}/')


if __name__ == '__main__':
    main()
