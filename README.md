# LifeEngine — learning vs. evolution in a predator-laden foraging world

A research build of [The Life Engine](https://thelifeengine.net/). The original is a
cellular-automaton evolution sandbox; this fork keeps its world and rebuilds the
organism on top of it as a **neural-network agent**, so that three ways of acquiring
adaptive behaviour can be run head-to-head under identical physics.

The upstream project's own README — the cell types, anatomies and brains of the
sandbox — is preserved at [`README_upstream.md`](README_upstream.md). Note that the
*experimental* organism here does **not** use the mutating anatomies described there:
its body is fixed and only its network weights change. That is deliberate — it keeps
the network's input and output dimensions constant so organisms are comparable.

---

## 1. What the experiment asks

Three mechanisms, one world, one yardstick:

| Condition | Mechanism | What is inherited | Lamarckian? | Code |
|---|---|---|---|---|
| **Evolution** | Genetic algorithm only | `genome_weights` | n/a — nothing is learned | `GAManager` |
| **Learning** | GA **plus** in-lifetime REINFORCE | `genome_weights` only | **No** — RL moves `active_weights`, the genome stays frozen and is all the GA sees. A Baldwin-effect setup. | `GAManager` with `rl_enabled` |
| **Pure RL** | In-lifetime REINFORCE only, no GA | `active_weights`, synced into the genome at reproduction | **Yes** | `PureRLManager` |

All three share the same grid, terrain, food, energy rules, day/night cycle, caves,
predators, network architecture and generation timing. They differ **only** in the
adaptation mechanism, so a difference in outcome is attributable to the mechanism.

Each is run in two environments:

| Environment | Roaming predators | Drain | Learning rate |
|---|---|---|---|
| **baseline** (predator-free) | 0 | — | 0.01 |
| **predator** (`hard` internally) | 60 | 5 | 0.02 |

## 2. What it measures

**Fitness is `cumulative_food_score`** — the total food value an organism ate over its
life. It is the GA's selection signal and is *independent of the RL reward*, so the
learning and evolution arms are scored on the same yardstick rather than each on its
own objective.

Four things anchor that number, and they matter more than the number itself:

- **Viability threshold = 4.375.** A lineage holds its size when it banks
  `3.5 food per reproduction attempt ÷ 0.8 success rate` = 4.375 food per child. Both
  constants are `ExperimentParams` values, so the threshold is **derived, not fitted** —
  identical for every run, condition and environment, and needing no calibration.
- **The chance floor = 1.537 ± 0.139.** `--random-floor` runs the evolution condition
  *minus between-generation selection*: the full GA loop, then the gene pool is thrown
  away at the boundary so every generation starts from fresh Xavier initialisation.
  Without it a converged fitness of 3.6 is a number with no scale; with it, that run is
  2.3× chance and ~16 sd above it — genuine adaptation that stalls below replacement,
  which is a different claim from "it learned nothing".
- **Time to viability**, not just endpoint fitness. The Baldwin effect is a claim about
  *acceleration*, so the quantity it predicts is a time. The event is the first
  generation starting 10 consecutive generations at or above threshold, right-censored
  if a run never manages it.
- **The seed is the unit of replication.** 10 seeds × 10 replicates is **not** n = 100.
  Runs sharing a seed share their terrain. Measured design effects on this data run up
  to 3.6, so every test clusters or stratifies on seed and reports the naive version
  beside it.

Secondary measures: population size, lifetime, death-cause composition, genome
variance (GA diversity), mean absolute weight difference `mean|active − genome|` (the
learning signal, identically zero in the evolution arm), and per-organism behaviour
from `organisms.csv` — cells explored, cave entries by day/night, food eaten by tier,
predator encounters.

## 3. Repository map

```
src/                        the simulator (JS, Node 16 compatible)
  headless.js                 the experiment runner — this is the one the cluster uses
  BrowserPreset.js            makes a browser run match a cluster run
  ExperimentParams.js         every tunable, overridden at module load
  Organism/                   GAManager, PureRLManager, NNBrain, predators, perception
  eval/                       the probe harness: run one fixed genome, GA off
    validate.js               the verification suite
headless.js                 the ORIGINAL Life Engine runner — not used by the experiments
analyse_*.py                one script per experiment family; all discover runs via params.json
landscape/                  fitness-landscape and ruggedness pipeline
assimilation/               the founder-genome assimilation probe
run_*.slurm                 cluster job arrays, one per experiment
output/                     figures and tables (gitignored)
logs/  logs_hard/           run output, baseline and predator (gitignored)
```

### Documentation map

| Question | Read |
|---|---|
| What are the experiments, exactly how do they run, what do they write | [`EXPERIMENTS.md`](EXPERIMENTS.md) — the authoritative reference |
| The RL / GA / energy equations | [`MATHEMATICAL_REFERENCE.md`](MATHEMATICAL_REFERENCE.md) |
| Every headless flag | [`HEADLESS.md`](HEADLESS.md) |
| The landscape pipeline stage by stage | [`landscape/README.md`](landscape/README.md) |
| Cluster environment setup | `CHPC Guide.pdf` |

---

## 4. Running it

### 4.1 Install

```bash
nvm install 16 && nvm use 16      # Node 16; the hot path stays ES2018-compatible
npm install
python3 -m venv .venv && ./.venv/bin/pip install -r landscape/requirements.txt
```

The Python venv is only needed for analysis. It additionally wants `statsmodels` and
`lifelines` for the statistics and survival scripts.

**Do not `npm run build` before a headless run** — the runner uses the raw `src/` files.
The build is only for the browser.

### 4.2 In the browser

The browser build runs the *same* experiment code as the cluster — same managers, same
map pool, same predators, same logger. It has no CLI, so the run is configured by
editing the constants at the top of `src/BrowserPreset.js`:

```js
const CONDITION   = 'learning';   // 'learning' | 'evolution' | 'pure_rl'
const ENVIRONMENT = 'hard';       // 'baseline' | 'hard'
const MAP_SEED    = 1;
const GRID_COLS   = 500;          // 500x500 matches every cluster run
const GRID_ROWS   = 500;
```

```bash
node generate_map_pool.js --seed <MAP_SEED>   # only when MAP_SEED changes
npm run build                                 # or: npm run build-watch
npm run serve                                 # http://localhost:3000
```

`BrowserPreset` is imported on the first line of `src/index.js`, above `Engine`, because
`NNBrain`, `AdvancedOrganism` and `GAManager` bake their constants in at module load.
Consequently **editing `WorldConfig.js` / `ExperimentParams.js` / `PredatorHyperparameters.js`
directly no longer changes a browser run** for the preset-owned fields — change the
preset. Confirm what is actually running from the About tab or the
`[BrowserPreset] …` banner in the devtools console.

Use `npm run serve`, not `dist/index.html` opened as a file: CSV logging works by POSTing
to `server.js`, which writes the same directory layout as headless.

Two limits worth knowing: there is **no generation cap** in the browser (it runs until
you stop it), and a 500×500 world in a browser is far slower than a headless task. The
browser build is for *watching behaviour*, not for producing comparison data.

### 4.3 Headless, one run

```bash
# smallest useful thing
node src/headless.js --map-seed 42 --condition learning --generations 50

# a production predator-environment learning run
node src/headless.js --map-seed 2001 --condition learning --mode standard \
    --generations 1000 --width 500 --height 500 --hidden-size 128 \
    --learning-rate 0.02 --no-epsilon --explore-bonus 0 \
    --roaming-predators 60 --predator-drain 5 --predators-per-patch 0
```

`node src/headless.js` with no arguments prints the full flag reference. Overrides are
applied *before* any simulation module loads, so a run is fully described by the
`params.json` it writes.

The three conditions map to flags as `--condition evolution`, `--condition learning`,
and `--condition learning --mode pure_rl`.

### 4.4 On the cluster

Each experiment is a SLURM job array, one array task per point in the grid:

```bash
sbatch run_learning_condition_hard_w500_h128_array.slurm      # 100 runs
GENERATIONS=200 sbatch run_evolution_condition_hard_w500_h128_array.slurm
SEEDS="2001 2002 2003" sbatch --array=0-29 run_pure_rl_condition_hard_w500_h128_array.slurm
```

The current comparison set is six of these — `{evolution, learning, pure_rl} ×
{baseline, hard}` — plus `run_random_floor_condition_hard_w500_h128_array.slurm` for the
chance anchor.

### 4.5 Verifying the build

The invariants the design rests on are *checked*, not argued for in prose:

```bash
node src/eval/validate.js
node src/eval/validate.js --width 500 --height 500 --hidden-size 128 --seed 2001
node src/eval/validate.js --only grad,mut
```

Eight checks, each printing the measured quantity beside its expected value, with the
whole run written to `output/validation/verification_report.json`. Non-zero exit if
anything fails. They cover: weight difference identically zero in the evolution arm;
all three conditions simulating a byte-identical runtime grid; `reinforce()` matching a
finite-difference ∇log π(a); the mutation kernels being bit-identical across conditions;
caves costing exactly zero energy at night; predation energy balancing exactly against
contacts; whole-organism energy conservation; and the random floor carrying nothing
across a generation boundary.

---

## 5. What a run writes

`logs<_env>/<condition>/<mode>/auto-run/<run-name>/`:

| File | Contents |
|---|---|
| `params.json` / `params.txt` | the authoritative resolved parameters. **Analysis scripts key off this, never off folder names.** |
| `generations.csv` | one row per generation: `avg_fitness`, `top20percent_fitness`, `best_fitness`, `total_agents`, `peak_population`, `avg_lifetime`, `genome_variance`, `avg_learned_weight_diff`, `avg_network_weight_mag`, … |
| `organisms.csv` | one row per organism per generation: fitness, lifetime, `death_cause`, food by tier, `cells_visited`, `predator_touches`, `drained_ticks`, `cave_entries{,_day,_night}`, … |
| `events.csv` | timestamped lifecycle events (generation close, disasters, pure-RL collapse/top-up) |
| `genome.csv` | with `--log-genomes`: the population centroid and the fittest genome every generation, plus **every** founder at generations 250/500/750/1000 (2400 rows over a 1000-generation run). Base64-encoded weights; the input to the landscape and assimilation pipelines. |

One trap worth carrying: **`drained_ticks` de-duplicates** several predators draining on
one global tick, so `drained_ticks × drain` is a *lower bound* on energy lost, not an
equality. The exact figure is contacts × drain.

---

## 6. Generating the figures

### 6.1 The three-condition comparison

```bash
./.venv/bin/python analyse_conditions_by_environment.py
```

Discovers all six cells, min-max normalises fitness on **one range shared by both
environments**, and writes to `output/conditions_by_environment/`: one PNG+PDF per
metric per environment, a `compare/` set with the two environments side by side on a
shared y-axis, Kaplan–Meier time-to-viability curves, and the tables everything is
quoted from — `fitness_summary.csv`, `runs.csv` (one row per run, so any aggregate can
be recomputed and any outlier traced to its seed), `paired_by_seed.csv`,
`bootstrap_ci.csv`, `collapse_rate.csv`, `variance_components.csv`,
`time_to_threshold_{km,cox,logrank}.csv`.

**The value cache.** The two log roots are tens of GB each and do not fit on one disk
together, so the script stores everything the figures and tests need under
`output/conditions_by_environment/.cache_values/` (~13 MB per environment) and reads a
cell back from there when its logs are gone:

```bash
python analyse_conditions_by_environment.py    # environment A on disk -> cached
# swap the logs
python analyse_conditions_by_environment.py    # A from cache, B from logs
```

Verified both ways: every figure pixel-identical, every table agreeing to float
round-trip. The cache stores per-generation mean/std/median/quartiles/n (so `--band`
and `--aggregate` still work and `--smooth` is still applied at draw time), the per-run
converged table every significance test consumes, the min/max extent that keeps both
environments on one normalised axis, and the survival event table. What it cannot do is
change `--final-window`, `--max-gen`, `--reps` or `--seeds` after the fact — those chose
what went into the stored numbers, so `meta.json` stamps them and a mismatched cache is
refused out loud.

### 6.2 The seed-clustered statistics

```bash
./.venv/bin/python analyse_hard_stats.py
./.venv/bin/python analyse_hard_stats.py --reference random_floor
```

Cochran–Mantel–Haenszel stratified by seed for viability, a linear mixed model with a
random intercept for seed for converged fitness, and Cox PH with cluster-robust
(Lin–Wei) SEs for time-to-threshold. The design effect and the effective *n* are printed
for every comparison. Writes `output/stats/`.

### 6.3 Fitness-landscape slices

The landscape work has two layers: the **JS probe** (`src/eval/`) is the only code that
runs the world, measuring `f(θ)` for one fixed genome as a monomorphic population of 100
clones with the GA disabled; the **Python side** (`landscape/`) chooses the θ points,
calls the probe, and draws the results.

A *slice* is a 31×31 (or 47×47) grid of `f(θ)` over a 2D plane cut through the 7814-dimensional
weight space, anchored on one run's trajectory mean and spanned by that run's own best
two directions. Every cell is probed **twice** at the same θ on the same 5 maps — once
with in-lifetime learning off (`f_evo`) and once with it on (`f_learn`) — so
`lift = f_learn − f_evo` is what learning adds at that genome with terrain cancelled.
That subtraction is the only honest way to see it: terrain variance (σ ≈ 6–8) dwarfs the
lift (≈ 0–3).

High-level steps:

```bash
# 1. LOCAL — pick the anchor runs and write the probe jobs (base64 genomes, ~52 MB/slice)
python landscape/make_slice_jobs.py --out-root landscape/out/slices_h128_companion ...

# 2. CLUSTER — run the probes as a sharded array
sbatch run_landscape_slices_h128_array.slurm

# 3. LOCAL — merge the shards, draw the landscapes, compute the metrics
bash landscape/run_slice_landscapes.sh
```

Step 3 is idempotent and safe to re-run over a partly finished array, which is the point
— it merges (dedup by job id, last wins), then calls `plot_slices.py` with **both**
environments on one shared normalisation range, then `slice_metrics.py` for FDC,
autocorrelation length and neutrality.

To redraw the figures alone without re-merging:

```bash
./.venv/bin/python landscape/plot_slices.py \
    --slices-root "hard:landscape/out/slices_h128_companion" \
    --slices-root "baseline:landscape/out/slices_h128_baseline_companion" \
    --out-dir output/fitness_landscapes --format pdf --suffix-env
```

Producing `slices_f_evo`, `slices_f_learn`, `slices_lift`, `slices_off_vs_on` (the two
passes adjacent, so the comparison is a saccade rather than a page turn), `rl_effect`
and `outcome_endpoints`, plus `slice_summary.csv`.

Three things to know before reading a slice panel:

- **Panels are not a common coordinate system.** Each run has its own anchor and its own
  u/v, so (α, β) means something different in every panel and panels cannot be
  subtracted. What *is* comparable is the fitness axis — every probe is the same
  measurement on the same scale, which is why all panels share one colour scale.
- **`plane N%`** in each title is `within_median / ceiling_median`: how much of that
  run's own motion the chosen plane captures, as a fraction of what the *best possible*
  2D plane for that run could capture. It is not equal across panels (69% for the failed
  runs here, 62–64% for the successful ones), so comparing apparent path shape between
  rows without it is invalid.
- **Viable percentages are upper bounds.** The 4.375 threshold prices only one of the
  three gates in `reproduce()`. The bound is one-sided and identical for every panel, so
  it can understate a difference between arms but cannot manufacture one.

### 6.4 Genetic assimilation

```bash
python assimilation/make_founder_jobs.py ...     # LOCAL
sbatch run_founder_probe_array.slurm             # CLUSTER
./.venv/bin/python assimilation/analyse_assimilation.py
```

Re-probes the 400 logged founder genomes per run — 100 founders at each of generations
250, 500, 750 and 1000 — as monomorphic cohorts with the GA off,
once with RL off (`f_off` — what the *inherited* weights do alone) and once with RL on
(`f_on`). Assimilation is the innate phenotype catching up to the learned one, so the
statistic is the ratio `Δf_off / Δf_on`: above 1 means innate is gaining on learned.

**Read the caveat in the script header before quoting a lift number.** `f_off` is clean
for both arms. `f_on` and `lift` are clean for the *learning* arm but **confounded for
the evolution arm**, whose `params.json` carries `epsilon_enabled: true` — those runs
never ran RL, so the field was never set — meaning its RL-on pass is "learning plus
30%-decaying-to-5% random actions". Figures built on the confounded quantity are labelled
CONFOUNDED in the title rather than omitted.

### 6.5 The sweeps

One script per family, all discovering runs via `params.json` and aggregating across
seeds: `analyse_lr_sweep_noeps.py`, `analyse_hidden_size_sweep.py` (widths 32/64/128
across all three conditions), `analyse_epsilon_sweep.py`, `analyse_trace_decay_sweep.py`,
`analyse_buffer_sweep.py`, `analyse_predator_sweep.py`, `analyse_disaster_replicates.py`,
`analyse_food_shuffle.py`.

---

## 7. How the project got here

Roughly the order the work was done, and why each step exists:

1. **Rebuild the organism.** Fixed anatomy, 54-input → ReLU hidden → 6-action softmax
   policy, two weight vectors per organism (`genome_weights`, `active_weights`). Fitness
   defined as cumulative food score, deliberately decoupled from the RL reward.
2. **Tune each mechanism separately** — learning rate, epsilon schedule, trace decay,
   pure-RL buffer size, network width — so the head-to-head is between three *tuned*
   mechanisms rather than three arbitrary ones. This is where LR 0.01 (baseline) and
   0.02 (predator) come from.
3. **Fix the operating point and run the comparison.** 6 cells × 100 runs = 600 runs at
   1000 generations, 500×500, hidden 128, epsilon off, on 10 fresh seeds (2001–2010) ×
   10 replicates.
4. **Anchor the scale.** Derive the 4.375 viability threshold from the reproduction
   constants; measure the random-policy floor so "below replacement" and "no better than
   chance" become distinguishable claims.
5. **Verify rather than assert.** `src/eval/validate.js` turns the design's load-bearing
   invariants into a runnable suite.
6. **Do the statistics with the seed as the unit.** Paired-by-seed tests, seed-clustered
   bootstrap, mixed models, cluster-robust survival — and report the design effect every
   time.
7. **Ask *why* the arms differ, not just *whether*.** The landscape slices probe the
   surface the runs were actually searching, and the paired RL-off / RL-on passes
   isolate what learning contributes at a fixed genome.
8. **Test for assimilation.** Re-probe logged founders with learning off to see whether
   the inherited genome improves on its own — the Baldwin-effect prediction stated as a
   measurement.

---

## 8. Key constants as actually run

The production comparison runs, from their own `params.json` (300 runs per environment):

| | |
|---|---|
| grid | 500 × 500 |
| generations | 1000 (1 gen = 5 maps × 2000 ticks = 10 000 ticks) |
| population | 100 founders per generation |
| network | 54 → 128 → 6, genome = 61·H + 6 = **7814** weights |
| learning rate | 0.01 baseline / 0.02 predator |
| epsilon | **off** — pure on-policy REINFORCE with baseline and traces (γ = 0.90) |
| explore bonus | 0 |
| mutation | 0.03 / σ 0.1 between generations; 0.05 / σ 0.1 at asexual reproduction |
| selection | tournament k = 2 over the whole generation, uniform crossover, no elitism |
| predators | 60 roaming @ drain 5 (predator env), 0 patrol |
| pure-RL buffer | 100 |
| seeds × replicates | 2001–2010 × 10 |

`SELECTION_PERCENT = 0.2` is **not** a selection cutoff — it only defines the top-20%
cohort used for reporting `top20percent_fitness`. Parent selection always uses the full
population. The full table, including the values the defaults sit at, is
[`EXPERIMENTS.md`](EXPERIMENTS.md) §10.

## 9. Reproducibility

- **Seeded:** map terrain (`--map-seed`) and, optionally, the disaster PRNG. A seed fixes
  the same 50-map pool for every condition, which is what makes the comparison controlled.
- **Unseeded, deliberately:** GA mutation, RL exploration, reproduction chance, predator
  wandering. Each run is an independent sample of stochastic dynamics on a fixed world —
  which is exactly why the seed, not the run, is the unit of independent replication.
- **Fully recorded:** overrides bake in at module load, so `params.json` describes a run
  completely, up to those unseeded draws.

## 10. Licence and credit

Simulator and world derived from **The Life Engine** by Max Robinson
([original repo](https://github.com/MaxRobinsonTheGreat/EvolutionSimulatorV2)) — see
[`LICENSE`](LICENSE) and [`README_upstream.md`](README_upstream.md). The neural policy,
RL and GA managers, predators, headless experiment runner, evaluation harness and the
whole analysis pipeline are additions made for this project.
