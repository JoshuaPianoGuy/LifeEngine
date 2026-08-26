"""
assimilation/make_founder_jobs.py
======================================================================
Build probe jobs for EVERY logged founder genome, to test genetic assimilation.

Logger.js dumps the full genome of all 100 founders every
FULL_GENOME_DUMP_EVERY_N_GENS = 250 generations, so a 1000-generation run carries
400 founder genomes (100 each at generations 250, 500, 750 and 1000) in its
genome.csv under record_type='founder'. This script turns those into a jobs.json
per run for src/eval/run_probe.js, which re-evaluates each genome as a
monomorphic 100-clone cohort with the GA off — ONCE WITH LEARNING DISABLED and
ONCE WITH IT ENABLED, on the same maps.

WHAT THE TEST ACTUALLY IS
-------------------------
Genetic assimilation means a behaviour that at first REQUIRED in-lifetime
learning ends up encoded in the inherited genome. The probe gives the two halves
of that directly, at the same theta on the same terrain:

    f_off(theta)  what the inherited weights do on their own   = INNATE
    f_on(theta)   what they do with learning switched on       = EXPRESSED
    lift = f_on - f_off                                        = what learning still adds

Assimilation predicts three things about the LEARNING condition as the
generation of the dump increases:

    1. f_off rises            — the innate behaviour gets better
    2. lift shrinks           — learning has less left to contribute
    3. weight drift shrinks   — and it has to move the weights less to do it
                                (mean_weight_drift, the "metrics": "founder"
                                column; see probe.js _collect)

THE CONTROL IS THE POINT, NOT AN EXTRA
---------------------------------------
None of those three is evidence on its own. Fitness rises over generations in
every condition, and a fitter genome has less headroom, so "lift shrinks" is
exactly what a plain ceiling effect also predicts. That is why the EVOLUTION
condition is probed too: its lineages never learned during evolution, so
whatever their lift does over generations is the ceiling effect with the
assimilation removed.

The claim is therefore a DIFFERENCE OF TRAJECTORIES, not a trajectory:

    assimilation  <=>  lift falls FASTER in learning than in evolution,
                       and lower at MATCHED innate fitness f_off.

The matched-f_off form is the stronger one, because it does not depend on the
two conditions reaching the same fitness at the same generation. Both are
available from these results, and neither needs another simulation.

Pure RL is NOT probed and cannot be: those runs set log_genomes=false, so no
pure_rl run on disk has a genome.csv at all.

PAIRING — WHY SEEDS ARE MATCHED ACROSS CONDITIONS
--------------------------------------------------
map_seed fixes the terrain pool, so map_index 0 of the evolution run on seed 2001
is the SAME map as map_index 0 of the learning run on seed 2001. Selecting one
run per (condition, seed) therefore makes the condition contrast paired on
terrain, which matters here for the same reason it matters everywhere else in
this project: terrain variance (sigma_vary ~ 6-8) is far larger than the effect
being measured.

Within a seed there are up to 10 replicate runs. Taking the best or the worst
would bias the comparison, so the default picks the MEDIAN-fitness replicate
(--rep-pick median). The seed remains the unit of replication.

CONFIG THE PROBE CANNOT INFER — both are mandatory here
--------------------------------------------------------
  hidden_size  these are h128 runs (genome 7814). run_probe.js builds its
               network from ExperimentParams, so without the override it builds
               a 64-wide brain and mis-reads every genome QUIETLY.
  map_seed     a genome evolved on seed 2005 has to be probed on seed 2005
               terrain.
Both are written into each unit's config_overrides, and params.json is copied in
beside jobs.json so the unit dir is self-contained on the cluster.

JOB ORDER: ALL RL-OFF FIRST, THEN ALL RL-ON
--------------------------------------------
run_probe.js shards with `idx % N`, so the order the jobs are written in decides
both how evenly the array is loaded and what survives a task that runs out of
walltime. Two things are wanted at once, and this ordering gets both.

BALANCE. If the jobs ALTERNATE off/on — the natural way to emit them, and what
the landscape jobs files do — then an EVEN N preserves parity: every even shard
draws only RL-off jobs and every odd shard only RL-on. The RL-on pass is ~2.4x
slower, so half the array hits the walltime and truncates while the other half
finishes early. That is exactly how the h128 companion slice run lost 8,468
probes. Writing the off jobs as one contiguous block and the on jobs as another
removes the parity coupling for ANY N: shard i takes every Nth index, and since
the off jobs occupy the first half of the list it receives about half its work
from each block whatever N is.

PRIORITY, which is the reason this beats a shuffle. f_off is the primary
measurement — it is what the INHERITED genome does on its own, and the whole
experiment is about how that changes over evolutionary time. Because the off
jobs come first globally, they also come first WITHIN every shard, so a task
killed on walltime loses RL-on probes and never RL-off ones. The primary result
completes by construction, and only the secondary half degrades.

A shuffle would balance the load equally well but spread the damage of a
truncation across both passes, leaving the primary measurement with holes in it.
That is the trade this ordering exists to avoid.

MATCHING THE RL SETTINGS OF THE CONTROL — --rl-match
-----------------------------------------------------
The probe reads its RL hyperparameters from each run's own params.json. For a
LEARNING run that is right: the RL-on pass reproduces the learning that lineage
actually experienced. For an EVOLUTION run it is NOT, because those runs never
ran RL and their params.json carries whatever the RL fields were left at:

    evolution   lr=0.02   epsilon_enabled=TRUE
    learning    lr=0.02   epsilon_enabled=false   (hard)
    learning    lr=0.01   epsilon_enabled=false   (baseline)

epsilon_enabled=true is not cosmetic. NNBrain pins epsilon to 0 only when the
flag is FALSE; otherwise it decays epsilon_start -> epsilon_end (0.3 -> 0.05)
over each organism's lifetime, injecting that share of RANDOM ACTIONS. So the
control's RL-on pass is "learning plus 30%-decaying-to-5% random actions", at
double the learning rate in the baseline. Its lift is then not comparable with
the learning arm's, and its negative sign is largely that injection.

--rl-match copies the RL hyperparameters of the SAME-SEED learning run into each
evolution unit's config_overrides, so both arms' RL-on pass is the identical
intervention and the lift contrast means what it claims to. It changes nothing
about the learning units, and nothing about either arm's RL-OFF pass — that pass
runs no RL, so these settings are inert in it.

Only the keys src/eval/config.js actually applies are emitted: learning_rate,
epsilon_start, epsilon_end, epsilon_decay_shape, epsilon_enabled, explore_bonus.
trace_decay is NOT among them, so a difference there cannot be corrected this
way — the builder checks and refuses rather than emitting an override that would
be silently dropped. (All runs on disk are trace_decay=0.9, so it does not bite.)

REBUILDING WITHOUT THE LOGS TREE — --from-units
------------------------------------------------
Every unit already carries its genomes as base64 in jobs.json plus the
generation/founder index in meta.json, so a corrected unit can be rebuilt from
an existing one without touching genome.csv at all. That is what --from-units
does, and it is the only option once the run logs have been cleaned up. It is
also much faster — no 100 MB CSV parse per run.

--rl-passes then keeps the re-run small: the RL-OFF results are unaffected by any
of this, so a correction only needs the RL-on half.

Usage
-----
  python assimilation/make_founder_jobs.py \\
      --logs-root logs_hard --env-label hard \\
      --out-root assimilation/out/founders_hard

  # correct the control's RL-on pass, reusing the genomes already built:
  python assimilation/make_founder_jobs.py \\
      --from-units assimilation/out/founders_hard --env-label hard \\
      --conditions evolution --rl-match --rl-passes on \\
      --out-root assimilation/out/founders_hard_evomatch

  python assimilation/make_founder_jobs.py \\
      --logs-root logs --env-label baseline \\
      --out-root assimilation/out/founders_baseline

Then transfer the out-root and submit run_founder_probe_array.slurm; the
preflight there prints the --array line to use.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'landscape'))
import ll_common as ll


# Logger.js FULL_GENOME_DUMP_EVERY_N_GENS = 250 over 1000 generations.
DEFAULT_GENERATIONS = (250, 500, 750, 1000)

# The condition families that actually log genomes. pure_rl sets
# log_genomes=false, so it has no genome.csv and cannot be included.
CONDITIONS = {
    'evolution': ('evolution', 'standard'),
    'learning':  ('learning',  'standard'),
}

# The RL keys src/eval/config.js actually applies. Anything outside this list
# would be accepted into config_overrides and then silently ignored, which is
# worse than refusing — see check_unmatchable().
RL_OVERRIDE_KEYS = ('learning_rate', 'epsilon_start', 'epsilon_end',
                    'epsilon_decay_shape', 'epsilon_enabled', 'explore_bonus')

# RL settings that differ between runs but CANNOT be overridden through
# config.js. If two runs disagree on one of these, --rl-match cannot make their
# RL-on passes equivalent and says so instead of pretending.
RL_UNMATCHABLE_KEYS = ('trace_decay',)


def rl_settings(params):
    return {k: params[k] for k in RL_OVERRIDE_KEYS if k in params}


def check_unmatchable(evo_params, learn_params, label):
    bad = [k for k in RL_UNMATCHABLE_KEYS
           if evo_params.get(k) != learn_params.get(k)]
    if bad:
        raise SystemExit(
            f'{label}: {", ".join(bad)} differs between the arms '
            f'({[evo_params.get(k) for k in bad]} vs {[learn_params.get(k) for k in bad]}) '
            f'and src/eval/config.js does not apply it, so --rl-match cannot '
            f'equalise the two RL-on passes. Add the key to config.js first.')


def run_final_fitness(run_dir):
    """Mean avg_fitness over the last 50 generations of generations.csv."""
    path = os.path.join(run_dir, 'generations.csv')
    try:
        v = pd.read_csv(path, usecols=['avg_fitness'])['avg_fitness']
    except (OSError, ValueError, KeyError):
        return float('nan')
    v = v.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v[-50:].mean()) if v.size else float('nan')


def discover(logs_root, condition):
    """Every run of one condition that has BOTH a genome.csv and params.json."""
    cond, family = CONDITIONS[condition]
    out = []
    pat = os.path.join(logs_root, cond, family, 'auto-run', '*')
    for d in sorted(glob.glob(pat)):
        gp, pp = os.path.join(d, 'genome.csv'), os.path.join(d, 'params.json')
        if not (os.path.exists(gp) and os.path.exists(pp)):
            continue
        p = json.load(open(pp))
        out.append({
            'dir': d, 'name': os.path.basename(d),
            'seed': int(p['map_seed']),
            'hidden_size': int(p.get('hidden_size', 64)),
            'ticks': int(p.get('ticks_per_map', 2000)),
            'maps_per_gen': int(p.get('maps_per_gen', 5)),
            'params': p,
            'f': run_final_fitness(d),
        })
    return out


def gen0_founders(n, hidden_size, seed):
    """n freshly Xavier-initialised genomes — a RESAMPLE of true generation 0.

    Not a proxy. GAManager.spawnGeneration() spawns the founding cohort
    "Xavier-random if gene_pool is null, i.e. generation 0", and
    ll_common.xavier_genome() reproduces NNBrain._initGenome() exactly: Glorot
    UNIFORM per layer (limits sqrt(6/(fan_in+fan_out)), different for W1 and W2)
    with both bias blocks zero. So these are drawn from precisely the
    distribution the real generation 0 came from.

    WHAT THIS IS NOT: a replay. xavierRandom() uses unseeded Math.random(), so
    the original gen-0 genomes are gone for good. Averaging 100 i.i.d. draws
    gives an unbiased estimate of the same population quantity, but it is an
    estimate, and its across-seed spread is terrain plus Monte-Carlo only —
    there is no lineage divergence at generation 0, so that error bar is
    narrower than the ones at 250+ for an uninteresting reason.

    WHY THE 'DIFFERENT WEIGHT SPACES' WORRY DOES NOT APPLY: independent runs
    really are near-orthogonal in weight space (permutation symmetry), which is
    why a shared PCA plane collapses past ~4 trajectories and why averaging
    genomes ACROSS runs yields a near-dead network. But f(theta) here is a
    SCALAR measured by running the simulator, and each genome is probed on its
    own — the 100 fitnesses are averaged, never the 100 genomes. No geometry is
    involved, so where a genome sits relative to any evolved basin is irrelevant
    to what it scores.

    The RNG is seeded from the map seed so the draw is reproducible.
    """
    rng = np.random.default_rng(20260826 + int(seed))
    out = []
    for i in range(n):
        g = ll.xavier_genome(hidden_size=hidden_size, rng=rng)
        out.append({'generation': 0, 'founder_index': i,
                    'logged_fitness': float('nan'),
                    'b64': ll.encode_b64(g)})
    return out


def discover_units(units_root):
    """Runs sourced from an ALREADY-BUILT units root instead of the logs tree.

    jobs.json carries every genome as base64 and meta.json the generation /
    founder index behind each, so a unit is a complete substitute for the
    genome.csv it came from — and the only source left once the logs are gone.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(units_root, '*/'))):
        d = d.rstrip('/')
        mp, jp, pp = (os.path.join(d, 'meta.json'), os.path.join(d, 'jobs.json'),
                      os.path.join(d, 'params.json'))
        if not all(os.path.exists(x) for x in (mp, jp, pp)):
            continue
        m = json.load(open(mp))
        out.append({'dir': d, 'name': m['run'], 'seed': int(m['seed']),
                    'hidden_size': int(m['hidden_size']),
                    'ticks': int(m['ticks']),
                    'condition': m['condition'],
                    'params': json.load(open(pp)),
                    'f': m.get('run_final_fitness', float('nan')),
                    'unit_meta': m, 'unit_jobs': jp})
    return out


def founders_from_unit(run):
    """(meta, b64) pairs recovered from a built unit — no genome.csv needed."""
    genomes = json.load(open(run['unit_jobs']))['genomes']
    out = []
    for g in run['unit_meta']['genome_index']:
        b64 = genomes.get(g['genome'])
        if b64 is None:
            continue
        out.append({'generation': int(g['generation']),
                    'founder_index': int(g['founder_index']),
                    'logged_fitness': float(g['logged_fitness']),
                    'b64': b64})
    return out


def pick_replicate(runs, how):
    """One run per seed, so the seed stays the unit of replication.

    'median' is the default because best/worst would bias the very comparison
    the design exists to make — an assimilation signal read off the best
    learning replicate against the worst evolution one would be an artefact of
    the picking, not of the lineage.
    """
    by_seed = {}
    for r in runs:
        by_seed.setdefault(r['seed'], []).append(r)
    chosen = []
    for seed in sorted(by_seed):
        reps = [r for r in by_seed[seed] if np.isfinite(r['f'])]
        if not reps:
            continue
        reps.sort(key=lambda r: r['f'])
        if how == 'median':
            chosen.append(reps[len(reps) // 2])
        elif how == 'best':
            chosen.append(reps[-1])
        elif how == 'worst':
            chosen.append(reps[0])
        else:                                    # 'first' — by run name
            chosen.append(sorted(by_seed[seed], key=lambda r: r['name'])[0])
    return chosen


def load_founders(run_dir, generations, n_founders):
    """The founder rows for the requested generations, as (meta, b64) pairs.

    genome.csv runs to ~100 MB per run and is mostly the per-generation centroid
    and fittest rows, so it is read with usecols and filtered before anything is
    decoded — the genomes stay as base64 strings the whole way through, since
    that is exactly the form jobs.json wants. Nothing here needs the float array.
    """
    df = pd.read_csv(os.path.join(run_dir, 'genome.csv'),
                     usecols=['generation', 'record_type', 'founder_index',
                              'fitness', 'genome_b64'],
                     dtype={'genome_b64': str})
    df = df[(df['record_type'] == 'founder')
            & (df['generation'].isin(generations))]
    out = []
    for gen in sorted(generations):
        sub = df[df['generation'] == gen].sort_values('founder_index')
        if sub.empty:
            continue
        if n_founders and len(sub) > n_founders:
            # Evenly spaced by founder_index rather than the first k: the
            # founders are ordered by the GA's selection, so a prefix is the
            # fitter end of the cohort, not a sample of it.
            idx = np.linspace(0, len(sub) - 1, n_founders).round().astype(int)
            sub = sub.iloc[np.unique(idx)]
        for _, row in sub.iterrows():
            out.append({'generation': int(gen),
                        'founder_index': int(row['founder_index']),
                        'logged_fitness': float(row['fitness']),
                        'b64': row['genome_b64']})
    return out


def build_unit(run, condition, env_label, generations, n_founders, repeats,
               out_root, founders=None, rl_overrides=None, rl_passes='both'):
    """One run -> one self-contained probe unit directory."""
    if founders is None:
        founders = load_founders(run['dir'], generations, n_founders)
    if not founders:
        return None

    genome_size = len(ll.decode_b64(founders[0]['b64']))
    expected = 61 * run['hidden_size'] + 6
    if genome_size != expected:
        raise SystemExit(
            f"{run['name']}: genome length {genome_size} does not match "
            f"hidden_size={run['hidden_size']} (expects {expected}). Refusing to "
            f"build jobs that run_probe.js would silently mis-read.")

    genomes, index = {}, []
    off_jobs, on_jobs = [], []
    for f in founders:
        key = f"g{f['generation']}_{f['founder_index']}"
        genomes[key] = f['b64']
        index.append({'genome': key, 'generation': f['generation'],
                      'founder_index': f['founder_index'],
                      'logged_fitness': f['logged_fitness']})
        for r in range(repeats):
            for rl, bucket in ((False, off_jobs), (True, on_jobs)):
                bucket.append({'id': f"{key}_r{r}_{'on' if rl else 'off'}",
                               'genome': key, 'map_index': r, 'rl': rl,
                               'ticks': run['ticks']})

    # See JOB ORDER in the module docstring. Two blocks, RL-off first: balances
    # every shard for any N, AND makes a walltime kill cost RL-on probes rather
    # than the primary RL-off measurement.
    if rl_passes == 'off':
        jobs = off_jobs
    elif rl_passes == 'on':
        jobs = on_jobs
    else:
        jobs = off_jobs + on_jobs
    if not jobs:
        return None

    cfg = {'hidden_size': run['hidden_size'], 'map_seed': run['seed']}
    # --rl-match: the paired learning run's RL settings, so both arms' RL-on
    # pass is the same intervention. Inert in the RL-off pass by construction.
    if rl_overrides:
        cfg.update(rl_overrides)
    unit = f"{condition}_{env_label}_seed{run['seed']}_{run['name']}"
    out_dir = os.path.join(out_root, unit)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'jobs.json'), 'w') as fh:
        json.dump({'metrics': 'founder', 'config_overrides': cfg,
                   'genomes': genomes, 'jobs': jobs}, fh)
    # params.json for the world settings config_overrides does not carry (grid
    # size, food density, predator counts) — the unit dir must stand alone on
    # the cluster, which runs Node only.
    with open(os.path.join(out_dir, 'params.json'), 'w') as fh:
        json.dump(run['params'], fh, indent=2)
    meta = {'kind': 'founder_assimilation', 'condition': condition,
            'environment': env_label, 'run': run['name'], 'seed': run['seed'],
            'run_final_fitness': run['f'], 'hidden_size': run['hidden_size'],
            'ticks': run['ticks'], 'repeats': repeats,
            'rl_passes': rl_passes,
            'rl_matched': bool(rl_overrides),
            'rl_overrides': rl_overrides or {},
            'generations': sorted(set(g['generation'] for g in index)),
            'n_genomes': len(genomes), 'n_jobs': len(jobs),
            'config_overrides': cfg, 'genome_index': index}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-root', default=None,
                    help='logs (baseline) or logs_hard (predator environment). '
                         'Not needed when --from-units is given.')
    ap.add_argument('--from-units', default=None, metavar='PATH',
                    help='Source the genomes from an ALREADY-BUILT units root '
                         'instead of the logs tree. The only option once the run '
                         'logs are gone, and far faster either way.')
    ap.add_argument('--rl-match', action='store_true',
                    help="Copy the SAME-SEED learning run's RL hyperparameters "
                         'into each evolution unit, so both arms\' RL-on pass is '
                         'the identical intervention. See --rl-match in the '
                         'module docstring for why the control needs it.')
    ap.add_argument('--gen0', type=int, default=0, metavar='N',
                    help='Emit an extra generation-0 unit per seed holding N '
                         'freshly Xavier-initialised genomes — the anchor the '
                         'founder dumps lack, because Logger.js only dumps from '
                         'generation 250. Use the SAME N as the dumps (100). '
                         'See gen0_founders() for why this is a resample of true '
                         'generation 0 rather than a proxy.')
    ap.add_argument('--gen0-only', action='store_true',
                    help='Build ONLY the generation-0 units, skipping the founder '
                         'dumps — for adding the anchor to results you already have.')
    ap.add_argument('--rl-passes', choices=('both', 'off', 'on'), default='both',
                    help="Which passes to emit. 'on' is what a --rl-match "
                         'correction needs: the RL-off results are unaffected by '
                         'RL settings, so re-running them would be waste.')
    ap.add_argument('--env-label', required=True, help='baseline | hard')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--conditions', default='evolution,learning',
                    help='Comma-separated. evolution is the CONTROL and dropping '
                         'it leaves the result uninterpretable — see the module '
                         'docstring.')
    ap.add_argument('--generations', default=','.join(str(g) for g in DEFAULT_GENERATIONS),
                    help='Founder dump generations to probe.')
    ap.add_argument('--n-founders', type=int, default=0,
                    help='Founders per dump, 0 = all 100 (the default).')
    ap.add_argument('--repeats', type=int, default=3,
                    help='Maps per genome per RL pass. Averages terrain; map r '
                         'is the same terrain in every condition on the same seed.')
    ap.add_argument('--seeds', default='',
                    help='Comma-separated map seeds. Default: all seeds found.')
    ap.add_argument('--rep-pick', choices=('median', 'best', 'worst', 'first'),
                    default='median',
                    help='Which replicate to take per seed. Default median — see '
                         'pick_replicate().')
    ap.add_argument('--shards-per-unit', type=int, default=4,
                    help='Only used for the cost estimate and the --array line.')
    args = ap.parse_args()

    generations = [int(g) for g in args.generations.split(',') if g.strip()]
    want_seeds = {int(s) for s in args.seeds.split(',') if s.strip()}
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f'unknown condition {c!r}; known: {list(CONDITIONS)}')
    if 'evolution' not in conditions:
        print('  [warn] no evolution arm: without the control, a shrinking lift '
              'cannot be told from a ceiling effect.')

    if not args.logs_root and not args.from_units:
        raise SystemExit('give either --logs-root or --from-units')

    # Source the runs. From an existing units root they arrive already grouped
    # by condition and already one-per-seed, so no replicate picking is needed —
    # doing it again could silently choose a DIFFERENT run than the results
    # being corrected came from.
    from_units = bool(args.from_units)
    if from_units:
        all_units = discover_units(args.from_units)
        by_cond = {}
        for u in all_units:
            by_cond.setdefault(u['condition'], []).append(u)
        print(f'sourcing genomes from {args.from_units} '
              f'({len(all_units)} units: ' +
              ', '.join(f'{c} x{len(v)}' for c, v in sorted(by_cond.items())) + ')')
    else:
        by_cond = None

    # --rl-match needs the learning arm's settings, keyed by seed. Read them from
    # whichever source is in play; both carry the run's params.json verbatim.
    learn_params = {}
    if args.rl_match:
        if from_units:
            for u in by_cond.get('learning', []):
                learn_params[u['seed']] = u['params']
        else:
            for r in discover(args.logs_root, 'learning'):
                learn_params.setdefault(r['seed'], r['params'])
        if not learn_params:
            raise SystemExit('--rl-match: no learning runs found to match against')
        seen = sorted({tuple(sorted(rl_settings(p).items()))
                       for p in learn_params.values()})
        print(f'--rl-match: learning-arm RL settings from {len(learn_params)} seeds')
        for combo in seen:
            print('    ' + '  '.join(f'{k}={v}' for k, v in combo))

    os.makedirs(args.out_root, exist_ok=True)
    manifest, total_jobs = [], 0

    # ── generation-0 anchor ──────────────────────────────────────────────────
    # ONE unit per seed, not one per condition. At generation 0 the two arms are
    # the same population: the genome distribution is Xavier for both, and the
    # terrain is the seed's. Emitting a separate gen-0 point per arm would draw a
    # difference that cannot exist, and would cost twice as much to say it.
    #
    # Its RL settings are the LEARNING arm's, which is what both arms use once
    # --rl-match is applied — so the anchor is shared by both f_off AND f_on
    # curves. Without --rl-match the arms' RL-on settings differ and this unit
    # is only a valid anchor for f_off; the builder says so.
    if args.gen0:
        if not args.rl_match:
            print('  [warn] --gen0 without --rl-match: the arms use different RL '
                  'settings, so this anchor is valid for f_off only, not f_on.')
        anchor_src = {}
        source_runs = (by_cond.get('learning', []) if from_units
                       else discover(args.logs_root, 'learning'))
        for r in source_runs:
            anchor_src.setdefault(r['seed'], r)
        if want_seeds:
            anchor_src = {k: v for k, v in anchor_src.items() if k in want_seeds}
        print(f'\ngen0: {len(anchor_src)} units ({args.gen0} Xavier genomes each, '
              f'one per seed, shared by both arms)')
        for seed in sorted(anchor_src):
            r = dict(anchor_src[seed])
            r['name'] = f'gen0_xavier_seed{seed}'
            r['f'] = float('nan')
            founders = gen0_founders(args.gen0, r['hidden_size'], seed)
            rl_over = rl_settings(r['params']) if args.rl_match else None
            meta = build_unit(r, 'gen0', args.env_label, [0], 0, args.repeats,
                              args.out_root, founders=founders,
                              rl_overrides=rl_over, rl_passes=args.rl_passes)
            if meta is None:
                continue
            manifest.append(meta)
            total_jobs += meta['n_jobs']
            print(f'  seed {seed:<5} {meta["n_genomes"]:>4} genomes  '
                  f'{meta["n_jobs"]:>6} jobs  {r["name"]}')
        if args.gen0_only:
            conditions = []

    for cond in conditions:
        if from_units:
            chosen = by_cond.get(cond, [])
            print(f'\n{cond}: {len(chosen)} units from {args.from_units}')
        else:
            runs = discover(args.logs_root, cond)
            if want_seeds:
                runs = [r for r in runs if r['seed'] in want_seeds]
            chosen = pick_replicate(runs, args.rep_pick)
            print(f'\n{cond}: {len(runs)} runs with genomes -> {len(chosen)} units '
                  f'({args.rep_pick} replicate per seed)')
        if want_seeds:
            chosen = [r for r in chosen if r['seed'] in want_seeds]

        for r in chosen:
            # Only the CONTROL is corrected: a learning run's own params are
            # already the settings its lineage evolved under.
            rl_over = None
            if args.rl_match and cond != 'learning':
                lp = learn_params.get(r['seed'])
                if lp is None:
                    print(f'  [skip] {r["name"]}: no learning run on seed '
                          f'{r["seed"]} to match against')
                    continue
                check_unmatchable(r['params'], lp, r['name'])
                rl_over = rl_settings(lp)

            founders = founders_from_unit(r) if from_units else None
            meta = build_unit(r, cond, args.env_label, generations,
                              args.n_founders, args.repeats, args.out_root,
                              founders=founders, rl_overrides=rl_over,
                              rl_passes=args.rl_passes)
            if meta is None:
                print(f'  [skip] {r["name"]}: no jobs for {generations} '
                      f'/ --rl-passes {args.rl_passes}')
                continue
            manifest.append(meta)
            total_jobs += meta['n_jobs']
            note = ''
            if rl_over:
                note = ('  [rl-matched: ' +
                        ' '.join(f'{k}={v}' for k, v in sorted(rl_over.items())) + ']')
            print(f'  seed {r["seed"]:<5} f={r["f"]:.2f}  {meta["n_genomes"]:>4} '
                  f'genomes  {meta["n_jobs"]:>6} jobs  {r["name"]}{note}')

    if not manifest:
        raise SystemExit('no units built — check --logs-root and --generations')

    n_units = len(manifest)
    tasks = n_units * args.shards_per_unit
    # Measured on the h128 companion slices: ~7 s per RL-off probe and ~17 s per
    # RL-on one at w500/2000 ticks. An --rl-passes on re-run is ALL slow probes,
    # so the two-pass average would understate it by more than half.
    sec = {'both': 12.0, 'off': 7.0, 'on': 17.0}[args.rl_passes]
    core_h = total_jobs * sec / 3600.0
    per_task_h = core_h / tasks if tasks else float('nan')

    with open(os.path.join(args.out_root, 'manifest.json'), 'w') as fh:
        json.dump({'environment': args.env_label, 'logs_root': args.logs_root,
                   'conditions': conditions, 'generations': generations,
                   'repeats': args.repeats, 'rep_pick': args.rep_pick,
                   'rl_passes': args.rl_passes, 'rl_match': args.rl_match,
                   'from_units': args.from_units,
                   'n_units': n_units, 'n_jobs': total_jobs,
                   'units': manifest}, fh, indent=2)

    print(f'\n  {n_units} units, {total_jobs:,} probes total')
    print(f'  ~{core_h:.0f} core-hours at ~{sec:.0f} s/probe '
          f'(--rl-passes {args.rl_passes})')
    print(f'  SHARDS_PER_UNIT={args.shards_per_unit} -> {tasks} tasks, '
          f'~{per_task_h:.1f} h each')
    print(f'\n  UNITS_ROOT={args.out_root} SHARDS_PER_UNIT={args.shards_per_unit} \\')
    print(f'      sbatch --array=0-{tasks - 1} run_founder_probe_array.slurm')
    print(f'\n  wrote {os.path.join(args.out_root, "manifest.json")}')


if __name__ == '__main__':
    main()
