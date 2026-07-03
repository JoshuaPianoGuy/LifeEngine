# Headless runner — learning-vs-evolution experiment

`src/headless.js` runs the **learning-vs-evolution experiment** (GA / RL / frozen-policy
conditions, seeded map pool, predators, structured CSV logging) as a pure Node.js
process with no browser, canvas, or display. It is built for the CHPC (PBS) cluster.

> **Two different runners exist.** The root `headless.js` is the *base* Life Engine
> sim (single origin organism, `fossil_record` JSON output). **This experiment uses
> `src/headless.js`** — different flags and output. Everything below is about `src/headless.js`.

---

## Relationship to "CHPC Guide.pdf"

The guide's **environment & workflow** steps apply unchanged:

- Install NVM, install **Node 16** (`nvm install 16`), `nvm alias default 16`.
- Copy/clone the project, run `npm install`.
- **Do not** `npm run build` — the headless runner uses the raw `src/` files.
- Submit with `qsub`, monitor with `qstat -u <user>` / `tail -f logs/*.out`, cancel with `qdel`.
- Walltime, `mem`, `ppn=1`, queue `serial`, project `CSCI1142`, `mkdir -p logs` — all as in the guide.

What **differs** from the guide (because it documents the base runner):

| | Guide / base runner | This experiment runner |
|---|---|---|
| Stop flag | `--max-ticks` | `--generations` (or `--max-ticks`) |
| Key flags | `--config`, `--load`, `--cell-size` | `--map-seed`, `--condition`, `--mode`, tunables |
| Output | one `results_*.json` (`fossil_record`) | `generations.csv` / `organisms.csv` / `events.csv` + `params.txt` per run folder |

The code is kept **Node 16 compatible** (no syntax newer than ES2018 in the hot path).

---

## Quick start

```bash
# one learning run on world seed 42, 50 generations
node src/headless.js --map-seed 42 --condition learning --generations 50

# same world, evolution condition, with a few tuned hyperparameters
node src/headless.js --map-seed 42 --condition evolution --generations 50 \
    --learning-rate 0.01 --hidden-size 64 --predator-drain 2.0
```

`node src/headless.js` with no args prints the full flag reference.

---

## Flags

**Run control**

| Flag | Default | Meaning |
|---|---|---|
| `--map-seed <N>` | *required* | Terrain seed (deterministic world). |
| `--generations <N>` | — | Run this many generations (1 gen = `ticks-per-map × maps-per-gen` ticks). |
| `--max-ticks <N>` | — | Alternative stop in raw ticks. Provide one of generations/max-ticks. |
| `--condition <s>` | `learning` | `learning` \| `evolution`. |
| `--mode <s>` | `standard` | `standard` \| `frozen_pg` \| `pure_rl`. |
| `--width <N>` | `400` | Grid columns. |
| `--height <N>` | `300` | Grid rows. |
| `--cell-size <N>` | `2` | Pixels per cell. |
| `--ticks-per-map <N>` | `2000` | Generation map length. |
| `--maps-per-gen <N>` | `5` | Maps per generation. |
| `--log-every <N>` | `10000` | Progress line every N ticks (0 = silent). |
| `--run-name <s>` | auto | Output folder name (see below). |

**Tunable hyperparameters** (default = current in-code value; see `src/ExperimentParams.js`)

| Flag | Default | Target |
|---|---|---|
| `--learning-rate <f>` | `0.02` | RL learning rate (NNBrain) |
| `--epsilon-start <f>` | `0.5` | exploration ε at birth |
| `--epsilon-end <f>` | `0.05` | exploration ε at death |
| `--hidden-size <N>` | `64` | NN hidden width (resizes the genome) |
| `--population-size <N>` | `100` | founders per generation |
| `--mut-prob <f>` | `0.03` | between-generation mutation rate |
| `--mut-sigma <f>` | `0.1` | between-generation mutation std-dev |

**Predators** (define the *world type* — see the presets below)

| Flag | Default | Target |
|---|---|---|
| `--roaming-predators <N>` | `0` | roaming predator count (chase prey anywhere) |
| `--predators-per-patch <N>` | `2` | patrol predators per prestige patch (guard food) |
| `--predator-drain <f>` | `1.0` | energy drained per predator contact |

> Patrol pool size = `predators-per-patch × patches on the current map`, so the
> count scales with each map. Setting either flag to `0` removes that predator
> type entirely. `--predator-drain` only matters when predators are present.
>
> Predator *spatial* radii (detection/leash) are intentionally not exposed —
> `PredatorHyperparameters.resolveForGrid` resets them from a load-time snapshot,
> so a flag override wouldn't stick.

**Natural disaster** (optional mass-mortality cull; off unless `--disaster` is passed)

| Flag | Default | Target |
|---|---|---|
| `--disaster` | off | enable natural-disaster culls (master toggle) |
| `--disaster-prob <f>` | `0.1` | per-generation probability a disaster strikes |
| `--disaster-fraction <f>` | `0.2` | **fixed** fraction of the population culled per strike |
| `--disaster-cooldown <N>` | `0` | min generations between strikes (safety window; 0 = none) |
| `--disaster-seed <N>` | `0` | PRNG seed for reproducible timing/victims (0 = unseeded) |

> A disaster removes a **fixed** `--disaster-fraction` of the generation's agents
> (chosen at random, fitness-blind) from the selection pool *before* tournament
> selection, wiping those genes from the next gene pool. **When** it strikes is
> random (per-generation Bernoulli at `--disaster-prob`), so strikes land on
> irregular generations. After a strike at gen *G*, `--disaster-cooldown` blocks
> any further strike until gen *G + cooldown* — a safety window that lets the
> population recover. Set `--disaster-seed` to make the whole pattern reproducible
> across runs (independent of `--map-seed`).

---

## World types (predator presets)

The two predator flags select which of the four world types a run uses. Everything
else (RL tuning, generations, world size/seed, disaster) is orthogonal and can be
combined with any of them.

| World type | `--roaming-predators` | `--predators-per-patch` |
|---|---|---|
| **Baseline** — no predators | `0` | `0` |
| **Roaming only** | `N` (e.g. `8`) | `0` |
| **Patrol only** | `0` | `M` (e.g. `2`) |
| **Roaming + patrol** | `N` | `M` |

> The default is *patrol only* (`predators-per-patch=2`, `roaming-predators=0`), so
> a **baseline** run must zero **both** flags explicitly.

Direct `node` invocations (same world seed, so identical terrain across types):

```bash
# 1. Baseline — no predators
node src/headless.js --map-seed 42 --condition learning --generations 200 \
    --roaming-predators 0 --predators-per-patch 0

# 2. Roaming predators only
node src/headless.js --map-seed 42 --condition learning --generations 200 \
    --roaming-predators 8 --predators-per-patch 0 --predator-drain 1.0

# 3. Patrol predators only
node src/headless.js --map-seed 42 --condition learning --generations 200 \
    --roaming-predators 0 --predators-per-patch 2 --predator-drain 1.0

# 4. Roaming + patrol
node src/headless.js --map-seed 42 --condition learning --generations 200 \
    --roaming-predators 8 --predators-per-patch 2 --predator-drain 1.0
```

Add a natural disaster to any of the above by appending, e.g.:

```bash
    --disaster --disaster-prob 0.1 --disaster-fraction 0.2 \
    --disaster-cooldown 50 --disaster-seed 7
```

The same presets via the job-script environment variables (for `qsub`/`sbatch`) are
in the cluster sections below.

---

## Output

Each run writes to:

```
logs/<condition>/<mode>/auto-run/<run-folder>/
    generations.csv    one row per generation (incl. map_seed)
    organisms.csv      one row per organism (incl. predator_touches, drained_ticks,
                       cave_entries[_day|_night], death_cause, map_seed, ...)
    events.csv         generation start / evolve events
    params.txt         human-readable resolved config for this run
    params.json        same, machine-readable
```

`<run-folder>` is, in priority order:
1. `--run-name <s>` if given;
2. `seed<MAP_SEED>_job<PBS_JOBID>` when `$PBS_JOBID` is set (i.e. on the cluster) — **unique per job**;
3. otherwise the auto-incremented `run_N` (fine for serial/local runs).

---

## Map seed: multiple runs & conditions on the same seed

The map seed identifies a **world**. Anything sharing a `--map-seed` plays on
**identical terrain** — that is the intended way to compare across conditions.

- **The pool file is shared and safe.** A seed maps to `src/maps/map_pool_seed<N>.json`,
  generated on first use and reused thereafter. Generation is deterministic, and the
  file is written atomically (temp + rename), so parallel jobs that first-touch the same
  seed cannot read a half-written file or corrupt each other.
- **Conditions don't collide.** Different `--condition`/`--mode` write under different
  `logs/<condition>/<mode>/...` folders, so learning/evolution/pure_rl on the same seed
  keep separate outputs.
- **Replicates don't collide** — *as long as the run folder is unique.* On the cluster the
  `seed<N>_job<JOBID>` naming guarantees this even when many jobs of the **same** condition
  and seed run at once. Locally, serial runs get `run_N`; if you launch parallel local runs
  of the same condition, pass distinct `--run-name` values.

**Why replicates matter here:** only the *terrain* is seeded. GA mutation, RL exploration,
and predator wandering use unseeded `Math.random()` (left to chance, by design — see
`MATHEMATICAL_REFERENCE.md` / the experiment notes). So two runs with identical
`--map-seed` and `--condition` will diverge — run several per (world × condition) to
sample the emergent distribution.

---

## Testing locally before the cluster

`#PBS` directives are bash comments, so the job script body runs anywhere:

```bash
./run_local.sh                              # quick 2-generation smoke test
GENERATIONS=10 MODE=pure_rl ./run_local.sh
MAP_SEED=7 CONDITION=evolution HIDDEN_SIZE=64 ./run_local.sh
```

`run_local.sh` execs the body of `run_simulation.pbs` with `node` from your PATH and the
current directory — if it works locally, the PBS job will too.

---

## Submitting on the CHPC

Edit `NODE_BIN` and `PROJECT_DIR` (and the `#PBS -M` email) at the top of
`run_simulation.pbs`, then:

```bash
mkdir -p logs
qsub run_simulation.pbs
```

**Per-job parameters without editing the file.** Every parameter uses `${VAR:-default}`,
so override at submit time — ideal for sweeps and job arrays. The variable name is the
upper-snake-case of the CLI flag. Full list:

| Variable | Flag | Default |
|---|---|---|
| `MAP_SEED` | `--map-seed` | *required* |
| `CONDITION` | `--condition` | `learning` |
| `MODE` | `--mode` | `standard` |
| `GENERATIONS` | `--generations` | `50` |
| `WIDTH` / `HEIGHT` | `--width` / `--height` | `400` / `300` |
| `LOG_EVERY` | `--log-every` | `10000` |
| `LEARNING_RATE` | `--learning-rate` | `0.02` |
| `EPSILON_START` / `EPSILON_END` | `--epsilon-start` / `--epsilon-end` | `0.5` / `0.05` |
| `HIDDEN_SIZE` | `--hidden-size` | `64` |
| `POPULATION_SIZE` | `--population-size` | `100` |
| `MUT_PROB` / `MUT_SIGMA` | `--mut-prob` / `--mut-sigma` | `0.03` / `0.1` |
| `ROAMING_PREDATORS` | `--roaming-predators` | `0` |
| `PREDATORS_PER_PATCH` | `--predators-per-patch` | `2` |
| `PREDATOR_DRAIN` | `--predator-drain` | `1.0` |
| `DISASTER` | `--disaster` (`1` = on) | `0` |
| `DISASTER_PROB` | `--disaster-prob` | `0.1` |
| `DISASTER_FRACTION` | `--disaster-fraction` | `0.2` |
| `DISASTER_COOLDOWN` | `--disaster-cooldown` | `0` |
| `DISASTER_SEED` | `--disaster-seed` | `0` |

```bash
# one world, all three conditions
qsub -v MAP_SEED=42,CONDITION=learning  run_simulation.pbs
qsub -v MAP_SEED=42,CONDITION=evolution run_simulation.pbs
qsub -v MAP_SEED=42,MODE=pure_rl        run_simulation.pbs

# a hyperparameter point
qsub -v MAP_SEED=7,GENERATIONS=100,LEARNING_RATE=0.01,HIDDEN_SIZE=64,PREDATOR_DRAIN=2.0 run_simulation.pbs
```

**The four world types** (same seed → identical terrain, just different predators):

```bash
# 1. Baseline — no predators
qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=0,PREDATORS_PER_PATCH=0 run_simulation.pbs
# 2. Roaming only
qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=8,PREDATORS_PER_PATCH=0,PREDATOR_DRAIN=1.0 run_simulation.pbs
# 3. Patrol only
qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=0,PREDATORS_PER_PATCH=2,PREDATOR_DRAIN=1.0 run_simulation.pbs
# 4. Roaming + patrol
qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=8,PREDATORS_PER_PATCH=2,PREDATOR_DRAIN=1.0 run_simulation.pbs
```

**With a natural disaster** (combine with any world type):

```bash
qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=600,PREDATORS_PER_PATCH=2,\
DISASTER=1,DISASTER_PROB=0.1,DISASTER_FRACTION=0.2,DISASTER_COOLDOWN=50,DISASTER_SEED=7 run_simulation.pbs
```

Sweep wrappers — the world type is just another loop dimension:

```bash
# conditions × seeds
for seed in 42 43 44; do
  for cond in learning evolution; do
    qsub -v MAP_SEED=$seed,CONDITION=$cond,GENERATIONS=100 run_simulation.pbs
  done
done

# all four world types on one seed (roaming N, patrol M)
for world in "0 0" "8 0" "0 2" "8 2"; do
  set -- $world
  qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=$1,PREDATORS_PER_PATCH=$2 run_simulation.pbs
done

# disaster-cooldown recovery sweep (fixed seed → comparable strike pattern)
for cd in 25 50 100; do
  qsub -v MAP_SEED=42,CONDITION=learning,GENERATIONS=600,DISASTER=1,DISASTER_COOLDOWN=$cd,DISASTER_SEED=7 run_simulation.pbs
done
```

Each job's `params.txt` records exactly what it ran with.
