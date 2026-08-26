"""
assimilation/analyse_assimilation.py
======================================================================
Genetic assimilation: does an inherited genome get better on its own?

Reads the founder-probe results built by make_founder_jobs.py and run through
run_founder_probe_array.slurm — every logged founder genome re-evaluated as a
monomorphic 100-clone cohort with the GA off, once with in-lifetime learning
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
COND_COLOUR = {'evolution': '#059669', 'learning': '#2563EB'}
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


def _band(ax, sub, col, colour, label, ls='-'):
    """Seed mean with a +/-1 SD band across seeds."""
    g = sub.groupby('generation')[col]
    m, sd, gens = g.mean(), g.std(ddof=1), sorted(sub.generation.unique())
    ax.plot(gens, m.loc[gens], color=colour, lw=2, ls=ls, marker='o', ms=4,
            label=label, zorder=3)
    ax.fill_between(gens, (m - sd).loc[gens], (m + sd).loc[gens],
                    color=colour, alpha=.15, lw=0, zorder=2)


def fig_innate(seed_tbl, rng, out_png):
    """The primary figure: what the INHERITED genome does on its own."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
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
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    fig.suptitle('Inherited genomes evaluated in the predator and baseline '
                 'environments', fontsize=13, color=INK, y=1.0)
    _caption(fig,
             'Seed means with ±1 SD across the 10 seeds. This pass runs no RL at '
             'all, so it is unaffected by the hyperparameter mismatch that '
             'confounds the evolution arm’s RL-on pass. The generation-0 point is a '
             'RESAMPLE of the Xavier distribution the real founders were drawn '
             'from, not a replay of them — its error bar carries terrain and '
             'Monte-Carlo noise only, since no lineage has diverged yet.')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
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
                _band(ax, s, 'lift_norm', COND_COLOUR[cond], cond,
                      ls='--' if cond == 'evolution' else '-')
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
