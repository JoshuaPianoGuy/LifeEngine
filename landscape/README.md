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
| `replay.js` | Stage-9 B1: same 100 clones but **reproduction ON** over a full generation → the scale the 4.375 viability threshold is defined on |
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
| `stage8_nk.py` | 8 | discretise → **measure pairwise epistasis** → NK model with **tunable K** |
| `ll_difficulty.py` | 9 | pure metric kernel: viability, GNR, FDC, dispersion, meta-model R², lift, peaks, λ |
| `stage9_difficulty.py` | 9 | **how HARD is the landscape** — A1-A9 reanalysis + B1 calibration + B2 transect |
| `test_ll_difficulty.py` | 9 | ground-truth tests (cone → FDC = −1, AR(1) → λ recovered, …) |
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

## Stage 6 — two better ρ(1) estimators

**`analyze --transform log`** analyses mean `log f` (the geometric mean) instead of
mean `f`. Justified by Stage 7's `r(σ, f) = +0.96..+0.99`: the noise is
**multiplicative**, so log stabilises the variance, and an exponential relaxation
toward a plateau becomes a straight line that the linear detrender can actually
remove. Pure reanalysis — no new probes. It writes `lambda_summary_log.json` and
`lambda_log.png`, leaving `lambda_summary.json` on the raw scale so
`stage7_plots.py --lambda-json` is unaffected. `analyze` also now reports
**residual curvature after detrend** (the quantity log is meant to reduce).

Seed-999 result: curvature falls exactly where the theory says it should — the
centroid walks, which relax exponentially down from an evolved peak
(hard 0.98→0.68, baseline 0.67→0.55) — and barely moves for the random starts
(0.69→0.65, 0.67→0.68). λ shifts modestly, and the apparent environment effect at
the centroid **halves**: baseline−hard goes from +18.0±13.2 (t=1.4) raw to
+9.4±13.5 (t=0.7) on log. Part of that gap was a curvature artefact.

**`pairs`** estimates ρ(1) as `corr(f_parent, f_mutant)` across pairs rather than
as the autocorrelation of a drifting series — no detrending, no exponential
assumption. A walk is already a chain of one-mutation-apart genomes, so `pairs`
reuses the existing walk data over a step window. ⚠️ **On the seed-999 walks this
is undetermined**: every window shows drift 0.23–0.61 (never stationary), and the
two biases run in opposite directions — wide windows are drift-**inflated**, narrow
windows are noise-**attenuated** — so λ ranges from 5 to >10⁴ purely with the
window. The table prints both diagnostics so a window is only believed when both
are small. It also disattenuates using the step×map interaction of the R
replicates (independent measurement noise inflates `Var(p)`/`Var(q)` but not
`Cov(p,q)`, so `ρ_true = ρ_obs/(1−noise)`).

**`pairjobs`** builds the ensemble the estimator actually wants, which does need
new probes: `n_pairs` independent parents each `burn` mutations from θ*, one
mutation to each mutant. Stationary **by construction** — one ensemble, no drift,
independent pairs. `pairs` then detects the ensemble and reports a single
bootstrap-SE ρ(1) with no windowing. Cost is `2·n_pairs·R` probes (300 pairs at
R=8 → 4800, ~2 core-hours), and it is directly comparable between environments.

⚠️ **Pick `--burn` by its noise fraction, not by intuition.** A 25-pair smoke test
at `burn=5, R=3` gave noise fraction **0.68** — that close to θ\* the genuine
fitness spread between parents is smaller than the measurement noise, so
disattenuating divides by 0.32 and amplifies noise into a meaningless ρ=0.9999.
`pairs` prints the fraction and refuses to let the corrected value stand above
0.3. Raise `--burn` first (cheaper than R): the Stage-6 walk windows put
burn≈30–60 near 0.1–0.25 at R=8. Below that threshold only the *observed* ρ is
reportable, and it is a lower bound on the true ρ(1).

```bash
python landscape/stage6_lambda.py pairjobs --start centroid --n-pairs 300 --burn 5 \
  --genome-csv <evo genome.csv> --out-dir landscape/out/stage6_pairs_hard \
  --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
# ...run through run_landscape_probe_array.slurm, merge, then:
python landscape/stage6_lambda.py pairs --out-dir landscape/out/stage6_pairs_hard \
  --transform log
```

## Stage 8 — the NK model (discretisation, measured K, tunable K)

λ says how fast fitness decorrelates; it cannot say what fraction of the
landscape's structure is *interaction*. That needs discrete loci, so Stage 8
discretises — and the discretisation is not arbitrary:

- **Loci = the 70 `FILTER_BLOCKS`** (one per neuron: hidden j = its 54 W1 weights
  + b1[j]; output k = its 64 W2 weights + b2[k]). Same units filter normalisation
  already treats as atomic in Stages 3/4, so Stage 8 is methodologically
  consistent with them. Because the blocks *partition* the genome, the `[-1,1]`
  clip is separable across loci and **cannot manufacture epistasis**.
- **Alleles = one filter-normalised nudge**, not a binning of weight values:
  allele 1 at locus *i* is `theta0[block_i] + delta[block_i]` with
  `||delta[block_i]|| = ALPHA * ||theta0[block_i]||`. Equal *relative* displacement
  at every locus, so the ε's are on one scale and counting them means something.

Then ε_ij = f(11) − f(10) − f(01) + f(00) over all C(70,2)=2415 pairs. Corners are
shared, so it costs 1 + 70 + 2415 = **2486 genomes** (19,888 probes at R=8 — one
Stage-6 condition), not 4 per pair. ε is formed **per map** before averaging, so
terrain is common-mode and its noise is governed by σ_fixed, not σ_vary.

Outputs: `K_i` (significant partners per locus), `K`, the threshold-free **Walsh
epistasis fraction** (order-2 share of variance, debiased), and — from the paired
RL passes — whether **learning flattens epistasis**, which λ cannot answer.

**K is a lower bound.** A ground-truth recovery test (known model, σ_fixed=0.15,
R=8, terrain σ=5.6) got corr(true, estimated) ε = 0.988 and found 21/28 real edges
with zero false positives at z=2 → K=3.0 against a true 4.0. Sub-threshold edges
are missing from K, never added. Quote `K ≥ x` alongside the Walsh fraction
(24.6% measured vs 28.4% true in the same test).

The model is second-order, `f(x) = f0 + Σ a_i x_i + Σ_E eps_ij x_i x_j` — exactly
the orders measured, extended to all 2^70 strings, with K = degree in E.
`--wiring measured` keeps the measured edges; `--wiring random --K k` is the dial
(rewires at degree k, coefficients resampled from the measured distributions).
Being multilinear it also extends to `[0,1]^70`, so a surrogate step can be as
**partial** as a real mutation: `sim`/`model` report ρ(1) under a *real GA step*
(per-locus sd `MUT_SIGMA*sqrt(MUT_PROB)/||delta[block]||`), which is the only row
comparable to Stage 6. Full allele flips reach ~82% of loci per step and overshoot
into ρ(1) < 0 — that mismatch is exactly why `quick` K is a lower bound.

⚠️ **`project` scores real genomes only near the hypercube.** It reports a residual
fraction; for an independently evolved genome most of the displacement lies outside
the 70-D allele subspace and the score is meaningless. Use the surrogate for
*dynamics* at tunable K, not to rank real genomes.

```bash
# 1) jobs (LOCAL)
python landscape/stage8_nk.py jobs --background centroid --alpha 0.25 --repeats 8 \
  --genome-csv logs/evolution/standard/auto-run/roam60d5_g1k_evolution_lscape_seed999_rerun6/genome.csv \
  --out-dir landscape/out/stage8_hard_centroid \
  --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
# 2) probes (CLUSTER) — the generic runner, nothing stage-specific
OUT_DIR=landscape/out/stage8_hard_centroid \
PARAMS=logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json \
N_SHARDS=64 sbatch --array=0-63 run_landscape_probe_array.slurm
# 3) merge + analyse + build + tune (LOCAL)
python landscape/merge_shards.py     --out-dir landscape/out/stage8_hard_centroid
python landscape/stage8_nk.py analyze --out-dir landscape/out/stage8_hard_centroid
python landscape/stage8_nk.py model   --out-dir landscape/out/stage8_hard_centroid
python landscape/stage8_nk.py sim     --out-dir landscape/out/stage8_hard_centroid \
                                      --k-sweep 0,1,2,4,8,16,32
```

`stage8_nk.py quick --out-dir landscape/out/stage6_*` keeps the old ρ(1)→K_eff
characterisation. Treat it as a re-expression of λ, not an independent
measurement — for seed 999 it gives K_eff = 0.09–0.90 ± ~0.3, i.e. no information
λ did not already carry. (A bare `--out-dir` still routes to `quick`.)

For **λ across all seeds / both environments / two start genomes** (the claim that
ruggedness is a property of the *environment*, not a seed accident), loop
`stage6_lambda.py jobs` over seeds {1,42,999} × {baseline,hard} × start
{random,centroid}, submit each out-dir as its own array, merge, and
`stage6_lambda.py analyze` each.

## Stage 9 — difficulty, not height

Stages 3-8 say where fitness **is** (level, shape, ruggedness). Stage 9 asks what
a searcher would **experience** climbing it. A tall smooth cone and a tall
shattered plateau reach the same maximum and are nothing alike to search.

Nine metrics are pure reanalysis of probes that already exist — **no new compute**:

| | metric | what it answers |
|---|---|---|
| A1 | viable fraction | what share of the space sustains a population at all |
| A2 | gradient-to-noise (GNR) | is the local gradient readable through the evaluation noise |
| A3 | fitness-distance corr (FDC) | does "closer to the optimum" mean "fitter" |
| A4 | dispersion | are the good cells one basin or scattered |
| A5 | meta-model R² | how much a linear / quadratic surrogate already explains |
| A6 | relative lift | what learning buys per cell, and where it costs |
| A7 | noise-aware peak count | how many local maxima survive the error bars |
| A8 | λ at extended lags | is the autocorrelation decay really a single exponential |
| A9 | comparison table | baseline vs hard, with SE and Z where defensible |

```bash
python landscape/stage9_difficulty.py metrics --grid-dir landscape/out/stage3_hard_coarse \
  --env hard --lambda-dir landscape/out/stage6_hard_centroid \
  --lambda-dir landscape/out/stage6_hard_random --out-dir landscape/out/stage9
# ... same for baseline, then
python landscape/stage9_difficulty.py compare \
  --metrics landscape/out/stage9/difficulty_metrics_baseline.json \
  --metrics landscape/out/stage9/difficulty_metrics_hard.json --out-dir landscape/out/stage9
python landscape/test_ll_difficulty.py     # ground-truth checks, ~2s
```

`metrics` also **writes the two files the Stage-9 spec assumed already existed**:
`landscape_<env>_<plane>.csv` (the aggregated grid table — alpha, beta, rl, f,
sigma, n_repeats, mean_lifetime) and `trajectory_2d_<env>.csv` (from
`plane.json`'s `meta.evo_traj` / `meta.lrn_traj`). Neither was ever a real file
in this pipeline; Stage 7 reconstructed the grid on the fly from `results.csv`.

### The viability threshold is a derived constant, not a guess

`AdvancedOrganism.js:291` fires a reproduction attempt at
`energyGainedSinceReproduction >= 3.5`, and the counter resets *"whether or not
placement succeeded"* (line 477) — so 3.5 food buys one **attempt**, not one
child. Attempts land with probability `REPRODUCTION_SUCCESS_RATE = 0.8`, so the
food needed to actually **replace itself** is 3.5 / 0.8 = **4.375**. That is the
replacement point: below it a lineage shrinks, above it grows.

It is a *lower* bound — the gate is `isClear() && canAddOrganism() &&
rand() < 0.8`, and the first two also fail in a crowded world — so the reported
viable fractions are **optimistic**. That shifts both environments the same way
and so does not affect the baseline-vs-hard comparison, which is the point of the
metric. (Note this is a threshold on a *counter*, not an energy debit:
reproduction costs no energy here, as the correction at the top of this file
records.)

The catch is units. 4.375 lives on the **replay** scale (a real reproducing
generation over 5 maps); f(θ) lives on the **monomorphic probe** scale (fixed
cohort, 1 map, reproduction off). So the cutoff on the probe scale is 4.375·k
with `k = f_monomorphic / f_replay`, which **B1 measures**. Until B1 is run,
k = 2.5 is a stated assumption — which is exactly why A1 also emits the full
sweep over k ∈ [1.5, 3.5] and the figure plots it.

### Seed-999 results (25×25 PCA coarse planes, R=10, k measured by B1)

| metric | baseline | hard | Z |
|---|---|---|---|
| A1 viable fraction (evo) — measured crossing | **1.000** | **0.733** | 15.1 |
| A2 median GNR (all cells) | 0.28 | 0.21 | — |
| A2 cells with GNR < 1 | 99.4% | 92.8% | — |
| A3 FDC (best) | **−0.031** | **−0.762** | 16.8 |
| A4 dispersion (p=0.05) | **0.884** | **0.257** | — |
| A5 R² quadratic | 0.713 | 0.885 | — |
| A6 corr(lift, f_evo) | **−0.729** | **+0.191** | −21.4 |
| A7 peaks raw → filtered | 13 → **0** | 11 → **0** | — |
| A8 λ (1/e crossing, centroid) | 36.1 | 22.5 | — |

Four things fall out of this, and three of them cut against the intuition that
"hard = harder to search":

1. **The hard environment is the one with global structure.** FDC −0.76 vs −0.03
   and dispersion 0.26 vs 0.88: under predation the surface is a single coherent
   basin whose gradient points at the optimum, while the baseline's good cells are
   scattered as widely as random points (dispersion ≈ 1 at p=0.10) and distance to
   the optimum predicts nothing. The baseline is a broad viable plateau with **no
   directional information**; the hard environment is a narrow ridge with a strong
   one. Hard to *survive* on and easy to *navigate* are different axes.
2. **Not one peak in either plane survives the noise.** All 24 raw local maxima
   beat their best neighbour by 0.003-0.089 while 2·SEM is 1.2-4.6 — margins
   20-500× too small, best ratio 0.052 (hard) / 0.073 (baseline), and 7/11 and
   5/13 of them sit on the grid boundary. The `peaks_f_evo: 11` in Stage 7's
   `scalars.json` is **entirely an artefact of finite R**; `peaks_detail` in the
   Stage-9 JSON lists the margin of every one. Multimodality claims from these
   planes are not supported at R=10.
3. **Learning does opposite things in the two environments.** corr(lift, f_evo)
   is −0.73 in the baseline (RL lifts the *weak* regions — it flattens the
   landscape) but +0.19 in the hard one (RL sharpens what is already good). And
   corr(Δ, lifetime) = +0.84 under predation vs −0.12 in the baseline: when
   predators are the binding constraint, everything learning buys comes through
   *staying alive longer*. Learning measurably harms almost nothing anywhere
   (0.3% of hard cells, 0% baseline).
4. **λ depends on where you stop fitting, so quote the 1/e crossing.** The random
   starts have a fit-window spread of 46-50% (λ ranges 27→16 as kfit goes 10→60),
   i.e. ρ(k) is **not** a single exponential and no one λ summarises it. The
   fit-free 1/e crossing is reported alongside every fit, and the environment gap
   it gives (36.1 vs 22.5) is the number to use. Note also that λ's baseline−hard
   difference is only Z ≈ 1.1-2.1 — much weaker evidence than A1/A3/A6.

⚠️ **A2 is the caveat on all of it.** 93-99% of cells have GNR < 1: at R=10 the
local gradient is smaller than the noise almost everywhere, so these surfaces
describe *shape at grid scale*, not anything a hill-climber could follow cell to
cell. Raising R is the only fix, and GNR scales as √R — going from 10 to 40
repeats buys a factor of 2.

⚠️ **The SEs are optimistic.** Grid cells are spatially autocorrelated, so the
binomial and Fisher-z SEs in `comparison.csv` treat 625 correlated cells as 625
independent ones. They are a floor. The λ rows (cross-walk SE over 5 independent
walks) are the only honest ones, and they are the rows that fail to reach
significance — which is itself informative.

### B1 — threshold calibration (needs new probes)

`replay.js` runs the same 100 monomorphic clones but with **reproduction ON**
across a full generation (5 maps × 2000 ticks, energy carryover, `NEXT_MAP`
transitions firing), stopping one tick short of the generation boundary so
`evolve()` never runs — the GA is still fully disabled. It reports mean
`cumulative_food_score` over `ga.all_agents` (**every agent that existed**,
founders + descendants) and, alongside, over founders only.

Those two differ a lot — in a 2-map smoke the hard evolution centroid gave
all-agents 3.62 vs founders 6.85, because descendants are born late and
accumulate less food. `--replay-field` picks which defines k; the spec's
statistic (`mean_all_agents`) is the default.

```bash
# PREFERRED: sample cells spanning the plane's fitness range, so replacement is
# bracketed by construction. ~10 min locally for both environments.
python landscape/stage9_difficulty.py replayjobs --out-dir landscape/out/stage9_calib_hard_grid \
  --from-grid landscape/out/stage3_hard_coarse --n-cells 10 --f-max 9.0 --repeats 4 \
  --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'

# or the three gen-1000 genomes (what the original spec asked for; see the caveat below)
python landscape/stage9_difficulty.py replayjobs --out-dir landscape/out/stage9_calib_hard \
  --evo-genome-csv   logs/evolution/standard/auto-run/roam60d5_g1k_evolution_lscape_seed999_rerun6/genome.csv \
  --learn-genome-csv logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/genome.csv \
  --gen 1000 --repeats 5 \
  --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
# CLUSTER (or local — it is only 15 + 75 probes):
node src/eval/replay.js    --params <learn params.json> --jobs <dir>/jobs_replay.json --out <dir>/results_replay.csv
node src/eval/run_probe.js --params <learn params.json> --jobs <dir>/jobs_mono.json   --out <dir>/results_mono.csv
python landscape/stage9_difficulty.py calibrate --out-dir landscape/out/stage9_calib_hard
# then re-run A1 with the measured k:
python landscape/stage9_difficulty.py metrics ... --k-json landscape/out/stage9_calib_hard/calibration.json
```

Both job files probe the **same map blocks** (replay repeat *r* runs maps
[5r..5r+4]; the monomorphic pass probes each of those 5 separately and
`calibrate` averages them), so k is a pure scale ratio and not a terrain
difference in disguise.

#### B1 results — and why the three gen-1000 genomes were not enough

**Run 1 (3 gen-1000 genomes, R=5)** exposed that k is confounded.
`f_monomorphic` is a **founder-only** number — reproduction is off in the probe,
so only the 100 founders ever exist. Pairing it with `mean_all_agents`, which
averages over ~1400 late-born descendants that banked almost no food, compares
two different populations, so the resulting k tracks *how much reproduction
happened* rather than any scale conversion. Watch it do exactly that — within the
hard environment k rose 1.52 → 1.59 → 2.30 in lockstep with children/founder
1.16 → 0.98 → 14.56. The like-for-like pairing (`--replay-field mean_founders`)
gave 3% spread instead of 24%, but a threshold that low made 99.5% of the hard
plane "viable". The two choices bracketed the hard viable fraction from 0.18 to
0.99 — i.e. A1 was not identified at all.

**Run 2 (`--from-grid`, 10 cells spanning the f range, R=4)** fixes it. The plane
already contains a continuum of genomes from dead to excellent, so sampling cells
across it brackets replacement by construction instead of by luck:

| env | cells sampled (f_mono) | children/founder | replacement crossing |
|---|---|---|---|
| hard | 3.28 → 8.62 | 0.67 → 4.91 | **f = 4.80**, 95% CI [4.34, 5.61] |
| baseline | 7.98 → 13.57 | 4.54 → 9.88 | **not reached — nothing in the plane is near it** |

**Use `children/founder`, not k.** Every agent past the founders is a child that
was actually born, so `(n_all − n_founders)/n_founders` measures realised
reproductive output directly — no threshold, no scale conversion, no k.
`calibrate` fits `ln(cpf) = a + b·f` over the cells nearest replacement (a local
fit: far above 1.0 the curve steepens and would drag the line) and solves for
cpf = 1, with a bootstrap CI. `metrics --k-json` then prefers that crossing over
the ratio automatically (`--k-from` to override).

The three thresholds for hard, and what each implies:

| threshold source | hard threshold | hard viable frac |
|---|---|---|
| `mean_all_agents` k = 1.38 | 6.03 | 0.35 |
| **measured crossing** | **4.80** | **0.733** (CI 0.43-0.92) |
| `mean_founders` k = 0.81 | 3.56 | 0.99 |

**Baseline's result is the stronger one for being negative.** The lowest cell in
the *entire* baseline plane (f = 7.98) still produces 4.54 children per founder —
4.5× replacement. So the baseline viable fraction is **1.000**, and that holds
under any threshold below 7.98 rather than depending on a calibration at all.
`calibrate` detects this case and says so instead of extrapolating a crossing it
cannot see.

That the hard *evolution* centroid and founder sat at 0.98-1.16 children per
founder in run 1 — dead on replacement — independently confirms the run
selection: rerun6 was picked as the replicate that "never escaped the low basin",
and it is measurably at the edge of viability. The hard *learning* centroid, at
14.56, is indistinguishable from the baseline genomes.

⚠️ **A1's headline is now baseline 1.000 vs hard 0.733** (evo pass; 1.000 vs
0.829 with RL on). The contrast is real and large but much smaller than the
k = 2.5 guess suggested (0.94 vs 0.12) — most of that original gap was the
threshold being far too strict for the hard environment, not a property of the
landscapes. The hard level still carries a wide CI (0.43-0.92) because the
crossing is fitted from 6 noisy cells; tightening it means more cells or more
repeats near f ≈ 5, not a better estimator.

### B2 — the 1D transect (needs new probes)

θ(t) = (1−t)·θ_evo + t·θ_learn over 51 points, **the same fixed map set at every
t**, RL off. The 2D planes are one viewing angle; this is the single line that
matters most — the straight path from what evolution found to what learning
found, and whether a valley separates them.

The `[-1,1]` clip is applied for consistency but **provably cannot bind**: both
endpoints are simulator-produced (hence already in range) and a convex
combination of two points in a box stays in the box. The job generator asserts
the clipped count is 0 and says so. Unlike the Stage-3/4 grids — where clipping
displaced ~49% of cells off the plane — the transect coordinate t is exact along
its whole length.

```bash
python landscape/stage9_difficulty.py transectjobs --out-dir landscape/out/stage9_transect_hard \
  --evo-genome-csv <evo>/genome.csv --learn-genome-csv <learn>/genome.csv \
  --points 51 --repeats 10 --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
# 510 probes — one small array, or ~15 min locally
OUT_DIR=landscape/out/stage9_transect_hard PARAMS=<learn params.json> \
  N_SHARDS=8 sbatch --array=0-7 run_landscape_probe_array.slurm
python landscape/merge_shards.py --out-dir landscape/out/stage9_transect_hard
python landscape/stage9_difficulty.py transect --out-dir landscape/out/stage9_transect_hard --env hard
```

`transect` reports the barrier depth against the mean SEM and refuses to call a
valley resolved when it is not.
