# UCT HPC ("hex") Guide — Running the Life Engine

This guide covers setting up Node.js and running the learning-vs-evolution headless
experiment (`src/headless.js`) on **UCT's own HPC cluster** ("hex" / `hpc.uct.ac.za`),
as documented at
[ucthpc.uct.ac.za/index.php/hpc-cluster](https://ucthpc.uct.ac.za/index.php/hpc-cluster/#softwareselection).

> **This is a different cluster from "CHPC Guide.pdf".** That PDF documents the
> national **CHPC** (PBS scheduler, `qsub`/`qstat`). This guide documents **UCT's
> in-house cluster** (SLURM scheduler, `sbatch`/`squeue`, environment `module`
> system). The two are not interchangeable — check which one your account/allocation
> is actually on before following either guide. Everything *project-specific*
> (flags, output layout, map seeds) is identical; only the cluster plumbing differs.

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Cluster basics: SLURM + modules](#2-cluster-basics-slurm--modules)
3. [Installing NVM and Node 16](#3-installing-nvm-and-node-16)
4. [Cloning / copying the project](#4-cloning--copying-the-project)
5. [Installing dependencies](#5-installing-dependencies)
6. [Smoke-testing on the login node](#6-smoke-testing-on-the-login-node)
7. [Submitting a job with SLURM](#7-submitting-a-job-with-slurm)
8. [Monitoring, cancelling, sweeps](#8-monitoring-cancelling-sweeps)
9. [Output](#9-output)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

- An active UCT HPC account. Accounts are **department / research-group based**
  (apply via the ICTS HPC team if you don't have one). You will be told which
  **allocation/account name** to use — you must pass it as `--account=<name>` on
  every job (e.g. `maths`, `science`, `csci`). Jobs without a valid `--account`
  are rejected.
- An SSH client. Log in to the head node:

  ```bash
  ssh <username>@hpc.uct.ac.za      # also reachable as hex.uct.ac.za (same cluster)
  ```

- Basic familiarity with bash.

Your `/home` directory and the larger `/scratch` filesystem are both mounted on
**every** node, so files you create on the login node are visible to your jobs.

The machine you land on after `ssh` is the **login / head node**. It is shared by
everyone — never run the actual simulation there beyond a 1–2 generation smoke
test. Real jobs go through `sbatch`; for anything interactive, grab a compute node
with `sintx` (see §2 and §6).

---

## 2. Cluster basics: SLURM + modules

Two things differ from a PBS-style cluster:

**Scheduler is SLURM, not PBS.**

| PBS | SLURM | Purpose |
|---|---|---|
| `qsub job.pbs` | `sbatch job.slurm` | submit a job |
| `qstat -u $USER` | `squeue -u $USER` (or just `qstat`*) | check queued/running jobs |
| `qdel <id>` | `scancel <id>` | cancel a job |
| `#PBS -l walltime=24:00:00` | `#SBATCH --time=24:00:00` | walltime |
| `#PBS -l nodes=1:ppn=1` | `#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1` | resources |
| `#PBS -P CSCI1142` | `#SBATCH --account=<name>` | charge to an allocation |
| `$PBS_JOBID` | `$SLURM_JOB_ID` | unique job id |

> *On this cluster `qstat` is a UCT-provided wrapper that shows **your own** jobs
> (handy if PBS muscle-memory kicks in); `squeue` shows the whole queue.

**Every job must declare an account and a partition.** Unlike the CHPC, where the
project code (`-P`) and a generic `serial` queue suffice, the UCT cluster requires:

- `#SBATCH --account=<name>` — your department / research-group allocation.
- `#SBATCH --partition=<name>` — which set of nodes to run on. The standard
  CPU partition is **`ada`**; the table below is the current layout:

| Partition | Use | Cores/node | Max cores/user | Walltime cap |
|---|---|---|---|---|
| `ada` (100-series) | general CPU — faster cores, less RAM | 48 | 200 | 250 h |putty
| `ada` (200-series) | general CPU — slower cores, more RAM (~9 GB/core) | 40 | 200 | 250 h |
| `l40s` | GPU (4 GPUs/node) | 48 | 96 | 48 h |
| `a100` | GPU (4 GPUs/node) | 56 | — | — |
| `gpumk` / `sadacc` | private (CS GPU / SADaCC) | varies | varies | varies |

This experiment is a **single-core CPU job**, so `--partition=ada` with
`--nodes=1 --ntasks=1` is all you need; the 250 h cap is far more than a run
requires.

**Interactive work uses a compute node, not the login node.** To get a shell on a
real node for smoke tests / debugging:

```bash
sintx --account=<name> --partition=ada --ntasks=1   # interactive session (UCT wrapper)
# or, the raw SLURM equivalent:
salloc --account=<name> --partition=ada --time=01:00:00 --nodes=1 --ntasks=1
```

**Software is provisioned via environment modules, not pre-installed.** The cluster
keeps packages under `/opt/exp_soft` and you "activate" what you need per session
or per job script:

```bash
module avail            # list everything installed
module load <name>      # activate a package (e.g. module load python/miniconda3-py3.12)
module list              # show what's currently active
module unload <name>
module purge             # clear everything
```

**Node.js is not provided as a module** on this cluster (only Python/R/MATLAB are
listed on the [supported software page](https://ucthpc.uct.ac.za/index.php/supported-software/)).
So, exactly as on the CHPC, you self-install Node into your home directory with
**NVM** — no module load needed for it, no sudo required.

---

## 3. Installing NVM and Node 16

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

If the installer didn't already append these to `~/.bashrc`, add them:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
```

Reload and verify:

```bash
source ~/.bashrc
nvm --version        # e.g. 0.39.7
```

Install and pin Node 16 (the codebase is kept Node 16-compatible):

```bash
nvm install 16
nvm use 16
nvm alias default 16   # make 16 the default in every new shell
node --version          # v16.20.2
which node               # note this path — you'll need it for the job script
```

---

## 4. Cloning / copying the project

```bash
git clone <repository-url> ~/LifeEngine
cd ~/LifeEngine
```

Or, from your local machine:

```bash
scp -r /path/to/LifeEngine <username>@hex.uct.ac.za:~/LifeEngine
```

---

## 5. Installing dependencies

```bash
cd ~/LifeEngine
npm install
```

**Do not run `npm run build`.** `src/headless.js` runs the raw source directly; the
webpack bundle is only needed for the browser UI.

---

## 6. Smoke-testing on a compute node

`run_local.sh` execs the same command body as the SLURM job, using `node` from
your PATH — a quick sanity check before queueing a real job. Run it on an
**interactive compute node**, not the login node:

```bash
sintx --account=<name> --partition=ada --ntasks=1   # grab a node first

./run_local.sh                                   # 2-generation smoke test
GENERATIONS=10 MODE=pure_rl ./run_local.sh
MAP_SEED=7 CONDITION=evolution HIDDEN_SIZE=64 ./run_local.sh
```

A 1–2 generation test is light enough to tolerate on the login node in a pinch,
but anything longer belongs on a compute node (via `sintx`) or in a batch job.

If this runs cleanly and writes a CSV folder under `logs/`, the SLURM job will too.

---

## 7. Submitting a job with SLURM

`run_simulation.slurm` is the SLURM counterpart of `run_simulation.pbs`
(same flags, same output layout — only the scheduler directives differ).

**Step 1 — edit the script.** Open `run_simulation.slurm` and set:

```bash
NODE_BIN="$HOME/.nvm/versions/node/v16.20.2/bin/node"   # from `which node` above
PROJECT_DIR="$HOME/LifeEngine"
```

Then fix the SLURM directives at the top of the file:

- `#SBATCH --account=YOUR_ACCOUNT` — **required**; set it to your allocation name
  (see §1). A job with a missing/invalid account is rejected at submit time.
- `#SBATCH --mail-user=` — your address for begin/end/fail mail.
- `#SBATCH --partition=ada` — already correct for this CPU job; `ada` is the real
  general-purpose partition. Run `sinfo` to confirm it's available to you.

**Step 2 — create the logs directory and submit:**

```bash
mkdir -p logs
sbatch run_simulation.slurm
```

You'll get back a job id, e.g. `Submitted batch job 123456`.

**Per-job parameters without editing the file** — every variable in the script
uses `${VAR:-default}`, so override at submit time with `--export`:

```bash
# one world, all three conditions
sbatch --export=MAP_SEED=42,CONDITION=learning  run_simulation.slurm
sbatch --export=MAP_SEED=42,CONDITION=evolution run_simulation.slurm
sbatch --export=MAP_SEED=42,MODE=pure_rl        run_simulation.slurm

# a hyperparameter point
sbatch --export=MAP_SEED=7,GENERATIONS=100,LEARNING_RATE=0.01,HIDDEN_SIZE=64,PREDATOR_DRAIN=2.0 run_simulation.slurm
```

> The full list of override variables (RL tuning, world size/seed, generations,
> predators, disaster) is in **[HEADLESS.md](HEADLESS.md)** — the variable names are
> identical on both clusters; only `qsub -v` vs `sbatch --export=` differs. Remember
> the `--export` replace-vs-`ALL` caveat in the note above.

**The four world types** — select via the two predator variables; same seed gives
identical terrain so only the predators differ:

```bash
# 1. Baseline — no predators
sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=0,PREDATORS_PER_PATCH=0 run_simulation.slurm
# 2. Roaming only
sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=8,PREDATORS_PER_PATCH=0,PREDATOR_DRAIN=1.0 run_simulation.slurm
# 3. Patrol only
sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=0,PREDATORS_PER_PATCH=2,PREDATOR_DRAIN=1.0 run_simulation.slurm
# 4. Roaming + patrol
sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=8,PREDATORS_PER_PATCH=2,PREDATOR_DRAIN=1.0 run_simulation.slurm
```

> The default is *patrol only* (`PREDATORS_PER_PATCH=2`, `ROAMING_PREDATORS=0`), so a
> **baseline** run must zero **both** predator variables explicitly.

**With a natural disaster** (combine with any world type):

```bash
sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=600,PREDATORS_PER_PATCH=2,DISASTER=1,DISASTER_PROB=0.1,DISASTER_FRACTION=0.2,DISASTER_COOLDOWN=50,DISASTER_SEED=7 run_simulation.slurm
```

> `--export` by default *replaces* the inherited environment unless you also pass
> `ALL`. Use `--export=ALL,MAP_SEED=42,...` if your script relies on anything else
> from your login environment (NVM setup lives in `~/.bashrc`, which the job
> doesn't source anyway — that's why `NODE_BIN` is an absolute path, not a bare
> `node`).

Sweep wrappers, identical in spirit to the CHPC ones:

```bash
# conditions × seeds
for seed in 42 43 44; do
  for cond in learning evolution; do
    sbatch --export=MAP_SEED=$seed,CONDITION=$cond,GENERATIONS=100 run_simulation.slurm
  done
done

# all four world types on one seed (roaming N, patrol M)
for world in "0 0" "8 0" "0 2" "8 2"; do
  set -- $world
  sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=200,ROAMING_PREDATORS=$1,PREDATORS_PER_PATCH=$2 run_simulation.slurm
done

# disaster-cooldown recovery sweep (fixed seed → comparable strike pattern)
for cd in 25 50 100; do
  sbatch --export=MAP_SEED=42,CONDITION=learning,GENERATIONS=600,DISASTER=1,DISASTER_COOLDOWN=$cd,DISASTER_SEED=7 run_simulation.slurm
done
```

---

## 8. Monitoring, cancelling, sweeps

```bash
squeue -u $USER                              # queued / running jobs (whole-queue tool)
qstat                                         # UCT wrapper: just your jobs
tail -f logs/life_engine_<jobid>.out         # live progress
cat logs/life_engine_<jobid>.err              # errors
scancel <jobid>                               # cancel (scancel id1 id2 id3 for several)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # post-mortem resource usage
```

---

## 9. Output

Identical to the CHPC runner — see [HEADLESS.md](HEADLESS.md) for the full
reference. Each run writes to:

```
logs/<condition>/<mode>/auto-run/<run-folder>/
    generations.csv
    organisms.csv
    events.csv
    params.txt
    params.json
```

`run_simulation.slurm` exports `PBS_JOBID=$SLURM_JOB_ID` before invoking
`src/headless.js`, because the run-folder naming logic only recognises
`PBS_JOBID` — this keeps `seed<N>_job<ID>` folders unique per SLURM job too,
exactly as they are on the CHPC.

---

## 10. Troubleshooting

**`nvm: command not found`** — run `source ~/.bashrc`, or log out/in again.

**`ERROR: Node binary not found at ...`** — your `NODE_BIN` path in
`run_simulation.slurm` doesn't match where NVM installed Node. Re-run
`nvm use 16 && which node` and update the script.

**`sbatch: error: Invalid account` / `Invalid qos`** — you left
`--account=YOUR_ACCOUNT` unset or used a name you're not a member of. Set it to
your real allocation (see §1); ask ICTS HPC if you don't know it.

**`sbatch: error: Invalid partition`** — `ada` isn't available to your account.
It's the standard CPU partition, but run `sinfo` to list the partitions you can
actually use and update the `#SBATCH --partition=` line.

**Job exits immediately with no output** — check the `.err` file; common causes
are `PROJECT_DIR` pointing at the wrong path, or the `logs/` directory not
existing yet (`mkdir -p logs` before submitting).

**Job killed before finishing** — increase `#SBATCH --time=` past your estimate,
or reduce `--generations`.

**Checking your account/allocation** — ask the UCT ICTS HPC team, or check the
[cluster docs](https://ucthpc.uct.ac.za/index.php/hpc-cluster/) for current
status pages; there's no `accounts`-style CLI equivalent documented for this
cluster the way there is on the CHPC.
