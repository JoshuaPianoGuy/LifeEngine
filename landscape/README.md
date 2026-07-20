# Fitness-landscape & ruggedness pipeline

Maps the fitness landscape of the evolved / learned neural policies and computes
the **ruggedness scalar λ** (Weinberger 1990 random-walk autocorrelation), using
the loss-landscape visualisation method of **Li et al. 2018** (filter-normalised
directions). Two layers:

- **Simulator (JS, `src/eval/`)** — the only code that runs the world. It
  measures `f(θ)`: the fitness of one fixed genome as a monomorphic population of
  100 clones, GA disabled.
- **Analysis (Python, `landscape/`)** — generates the θ points for each stage
  (PCA plane, random directions, mutational walk), calls the JS probe, and makes
  the plots + scalars.

## What f(θ) is (and the constraints that make it honest)

`f(θ)` = mean `cumulative_food_score` over 100 clones of θ, GA fully off,
`reproduction_success_rate = 0` (fixed cohort), on one seeded map. The unit of
replication is the **run**, not the clone — the 100 clones share a world and are
correlated, so the error bar comes from spread across runs, never
`std_clones/√100`.

Two findings from building this that **correct the original plan's premises**:

1. **Reproduction is energy-neutral.** The plan assumed 3.5 energy is deducted on
   reproduction; it is not (the threshold is a separate counter, not an energy
   debit — verified in `AdvancedOrganism.reproduce()` and by `selftest.js`
   DIAG B). `reproduction_success_rate = 0` is therefore justified by **fixed
   cohort / density matching**, not energy calibration. Its conclusion stands.

2. **Evaluation variance is terrain-dominated, so use a common map set.**
   Stage-2 calibration: σ_fixed (dynamics, same map) ≈ 0.1–1.0 but σ_vary
   (different maps) ≈ 6–8 — terrain is ~98% of the variance. The pipeline
   evaluates **every grid point / walk step on the SAME fixed set of R maps**, so
   terrain is common-mode and cancels between points (the plan's RL-off/RL-on
   pairing, extended across the whole grid). Only σ_fixed limits the **shape** and
   **ruggedness**, so **R ≈ 8–12 suffices**. The huge σ_vary only affects a
   surface's absolute *level* — which is "fitness on this map set"; maps 0..R-1
   are a profile-balanced sample of the pool, not one terrain.

## `src/eval/` (JS)

| file | role |
|---|---|
| `probe.js` | `Probe(rlEnabled).evaluate(θ,{mapIndex,ticks})` → f(θ) as 100 clones, GA off |
| `config.js` | `applyConfig({paramsPath,overrides})` (must run before requiring the sim); genome base64 codec |
| `run_probe.js` | batch driver: jobs.json → results CSV, `--shard i/N`, resumable |
| `anchor.js` | Stage-2 anchor: monomorphic centroid, **reproduction ON**, full 5-map generation |
| `selftest.js` | `node src/eval/selftest.js` — RL-off MAD assert + reproduction-energy diagnostic + smoke |

## `landscape/` (Python)

| file | stage | role |
|---|---|---|
| `ll_common.py` | — | genome I/O, network layout + **filter normalisation**, joint PCA (numpy SVD), jobs/results plumbing |
| `ll_grid.py` | 3/4/5/7 | plane save/load, **paired-RL grid jobs**, reconstruction, peak counting |
| `stage2_calibrate.py` | 2 | R calibration: σ_fixed vs σ_vary decomposition, recommended R |
| `stage3_pca.py` | 3 | joint-PCA plane (coarse=founder cloud, fine=trajectory), grid jobs |
| `stage4_random.py` | 4 | 3 random **filter-normalised** planes, grid jobs |
| `run_grid.py` | 5 | run a grid out-dir's jobs locally |
| `stage6_lambda.py` | 6 | mutational random walk → ρ(1) → **λ = −1/ln ρ(1)** |
| `stage7_plots.py` | 7 | raw + normalised panels, halo/difference, σ, peak counts, trajectory overlay |
| `merge_shards.py` | — | concat `results_shard_*.csv` → `results.csv` after a SLURM array |

`run_landscape_probe_array.slurm` (repo root) is the generic sharded HPC runner —
point it at any out-dir containing a `jobs.json`.

## Where each part runs (NO venv needed on the cluster)

Three phases; only phase 2 is heavy / cluster-bound:

| phase | what | where | needs |
|---|---|---|---|
| 1. generate jobs | `stageN_*.py … jobs` writes `jobs.json` | **local** (or login node) | Python + numpy/pandas |
| 2. run probes | `run_landscape_probe_array.slurm` (sharded) | **cluster** | **Node only** |
| 3. merge + plot | `merge_shards.py` then `stageN analyze` / `stage7_plots.py` | **local** | Python + matplotlib |

The Python scripts do **jobs generation and plotting only** — both lightweight
(seconds) and run **locally**, where the env already works. The compute nodes run
**only Node** (`src/eval/run_probe.js`), so no Python virtualenv is required on the
cluster. Recommended flow: generate `jobs.json` locally → `scp` the out-dir to
scratch → `sbatch` the array → `scp` the `results_shard_*.csv` back → merge + plot
locally.

If you'd rather run the Python on an HPC login node instead, make a venv there:
`python -m venv .venv && . .venv/bin/activate && pip install -r landscape/requirements.txt`.
The `run_landscape_probe_array.slurm` job itself never touches Python.

**The SLURM script does NOT do everything** — by design it runs only phase 2 (the
`f(θ)` probes). Phases 1 and 3 are the quick local Python steps above. Merge can't
be folded into the array because it must wait for all shards; run it after.

## Environments + which run to use (seed 999)

Only seed 999 was run. Per-seed stochasticity means evolution converges to
different levels across replicates; use the **lowest-fitness evolution replicate**
(the run where evolution "struggled") so the landscape shows the trapped/low-basin
case. Ranked by mean centroid fitness over the last 50 gens:

- **evolution roam60d5** (hard): the hard environment was later re-run **10×**
  (`..._rerun1..10`). Two replicates never escaped the low basin while the other
  eight converged to ~13–16: rerun6=**6.77** < rerun1=6.97 < rerun10=12.78 < … <
  rerun8=16.01  → **use rerun6** (lowest centroid fitness; rerun1 is a near-tie
  backup). The original 3-replicate ranking (r2=15.72 < r3=15.85 < r1=16.12) is
  superseded — all three of those sit up in the converged band.
- **evolution baseline**: r2=**18.89** < r1=18.97 < r3=19.10  → **use r2** (barely
  varies — no predators).

So the canonical runs are:
- **baseline** — no predators. `logs/{evolution,learning}/standard/auto-run/baseline_g1k_*_seed999_r2/`
- **hard (roam60d5)** — 60 roaming predators, drain 5. Evolution θ\*:
  `.../roam60d5_g1k_evolution_lscape_seed999_rerun6/`; learning θ\* + params:
  `.../roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/`.

Re-derive the ranking any time with:
`awk -F',' '$4=="centroid" && $1>=951{s+=$6;n++} END{print s/n}' <run>/genome.csv`

Use the **learning r2** run's `params.json` as `--params` for grids so the RL-on
pass matches the learning condition; retarget predators per plane with
`--config-overrides '{"roaming_predator_count":60,"predator_drain":5}'`.

## Quick local smoke (each stage ships a small config)

```bash
# Stage 2 — calibrate R (5 genomes x 6+6 runs)
python landscape/stage2_calibrate.py all \
  --genome-csv logs/evolution/standard/auto-run/baseline_g1k_evolution_lscape_seed999_r1/genome.csv \
  --params     logs/evolution/standard/auto-run/baseline_g1k_evolution_lscape_seed999_r1/params.json \
  --out-dir    landscape/out/stage2_baseline_local --repeats 6

# Stage 6 — lambda (25-step walk, R=2)
python landscape/stage6_lambda.py all --start random --steps 25 --repeats 2 \
  --genome-csv <evo genome.csv> --params <evo params.json> \
  --out-dir landscape/out/stage6_baseline_random_local

# Stage 3 + 7 — PCA landscape (5x5, R=2)
python landscape/stage3_pca.py --grid coarse --resolution 5 --repeats 2 \
  --evo-genome-csv <evo genome.csv> --learn-genome-csv <learn genome.csv> \
  --out-dir landscape/out/stage3_baseline_coarse_local
python landscape/run_grid.py  --out-dir landscape/out/stage3_baseline_coarse_local --params <learn params.json>
python landscape/stage7_plots.py --out-dir landscape/out/stage3_baseline_coarse_local
```

## HPC run (real scale)

```bash
# 1) generate jobs LOCALLY (or login node), e.g. hard 25x25 coarse PCA plane, R=10.
#    Evolution θ* = rerun6 (lowest-fitness); learning θ* = r2 (see run-selection above).
python landscape/stage3_pca.py --grid coarse --resolution 25 --repeats 10 \
  --evo-genome-csv   logs/evolution/standard/auto-run/roam60d5_g1k_evolution_lscape_seed999_rerun6/genome.csv \
  --learn-genome-csv logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/genome.csv \
  --out-dir landscape/out/stage3_hard_coarse \
  --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
#    (then scp landscape/out/stage3_hard_coarse to your scratch clone)

# 2) submit sharded array on the cluster (N_SHARDS must equal the array size)
OUT_DIR=landscape/out/stage3_hard_coarse \
PARAMS=logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json \
N_SHARDS=64 sbatch --array=0-63 run_landscape_probe_array.slurm
#    (then scp landscape/out/stage3_hard_coarse/results_shard_*.csv back)

# 3) merge + plot
python landscape/merge_shards.py --out-dir landscape/out/stage3_hard_coarse
python landscape/stage7_plots.py --out-dir landscape/out/stage3_hard_coarse \
  --lambda-json landscape/out/stage6_hard_random/lambda_summary.json
```

For **λ across all seeds / both environments / two start genomes** (the claim that
ruggedness is a property of the *environment*, not a seed accident), loop
`stage6_lambda.py jobs` over seeds {1,42,999} × {baseline,hard} × start
{random,centroid}, submit each out-dir as its own array, merge, and
`stage6_lambda.py analyze` each.
