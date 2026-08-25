"""
landscape/ll_common.py
======================================================================
Shared utilities for the fitness-landscape / ruggedness pipeline (Stages 2-7).

The heavy numeric work (PCA planes, filter-normalised random directions,
mutational walks) lives in Python; the actual f(theta) evaluation lives in the
JS simulator and is reached via src/eval/run_probe.js. This module is the glue:

  - genome I/O:      load genome.csv, decode/encode little-endian float32 base64
  - network layout:  the 54->64->6 weight blocks + Li et al. filter normalisation
  - PCA:             joint trajectory PCA via numpy SVD (no sklearn)
  - job plumbing:    write a jobs.json for run_probe.js, read its results.csv,
                     and (for local tests) shell out to node to run it

Network (matches src/Organism/Perception/NNBrain.js at hidden_size=64):
    input  54 -> hidden 64 (ReLU) -> output 6 (softmax)
    genome = W1(54x64) + b1(64) + W2(64x6) + b2(6) = 3456+64+384+6 = 3910
    Flat layout (row-major per the JS forward pass):
        W1[j*54 + i]         hidden unit j, input i     idx    0 .. 3455
        b1[j]                hidden unit j              idx 3456 .. 3519
        W2[3520 + k*64 + j]  output unit k, hidden j    idx 3520 .. 3903
        b2[k]                output unit k              idx 3904 .. 3909
"""

import base64
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# ── Network layout constants ──────────────────────────────────────────────────
STATE_SIZE  = 54
HIDDEN_SIZE = 64
OUTPUT_SIZE = 6
W1_SIZE = STATE_SIZE * HIDDEN_SIZE   # 3456
B1_SIZE = HIDDEN_SIZE                # 64
W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE  # 384
B2_SIZE = OUTPUT_SIZE                # 6
GENOME_SIZE = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE  # 3910

W1_OFF = 0
B1_OFF = W1_SIZE                 # 3456
W2_OFF = W1_SIZE + B1_SIZE       # 3520
B2_OFF = W2_OFF + W2_SIZE        # 3904

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_filter_blocks():
    """Per-neuron index blocks for Li et al. filter normalisation.

    A 'filter' here = one neuron's incoming weight row + its bias:
      - hidden unit j: 54 incoming W1 weights (idx j*54 .. j*54+53) + b1[j] -> 55
      - output unit k: 64 incoming W2 weights (idx 3520+k*64 .. +63) + b2[k] -> 65
    Returns a list of int arrays (index sets), 64 + 6 = 70 blocks covering all
    3910 weights exactly once.
    """
    blocks = []
    for j in range(HIDDEN_SIZE):
        idx = list(range(W1_OFF + j * STATE_SIZE, W1_OFF + j * STATE_SIZE + STATE_SIZE))
        idx.append(B1_OFF + j)
        blocks.append(np.array(idx, dtype=np.int64))
    for k in range(OUTPUT_SIZE):
        idx = list(range(W2_OFF + k * HIDDEN_SIZE, W2_OFF + k * HIDDEN_SIZE + HIDDEN_SIZE))
        idx.append(B2_OFF + k)
        blocks.append(np.array(idx, dtype=np.int64))
    return blocks


FILTER_BLOCKS = _build_filter_blocks()

# Sanity: blocks partition all weights exactly once.
_all = np.concatenate(FILTER_BLOCKS)
assert _all.size == GENOME_SIZE and np.array_equal(np.sort(_all), np.arange(GENOME_SIZE)), \
    "filter blocks do not partition the genome"


def filter_normalize(direction, theta, eps=1e-12):
    """Rescale a random direction per-neuron to that neuron's weight norm in theta
    (Li et al. 2018). For each block: d_block <- (d_block/||d_block||)*||theta_block||.

    A fixed raw step means different things per neuron (a small-weight unit gets
    thrown far off its operating point, a large-weight unit barely moves); this
    makes the step relative to each neuron's own scale, so surface texture
    reflects landscape structure, not weight-scale artefacts.
    """
    d = np.array(direction, dtype=np.float64, copy=True)
    for blk in FILTER_BLOCKS:
        dn = np.linalg.norm(d[blk])
        tn = np.linalg.norm(theta[blk])
        if dn > eps:
            d[blk] *= (tn / dn)
        else:
            d[blk] = 0.0
    return d


# ── genome base64 codec (little-endian float32) ───────────────────────────────
def decode_b64(s):
    return np.frombuffer(base64.b64decode(s), dtype='<f4').astype(np.float64)


def encode_b64(arr):
    return base64.b64encode(np.asarray(arr, dtype='<f4').tobytes()).decode('ascii')


# ── Fresh-brain (Xavier) genomes ──────────────────────────────────────────────
def infer_hidden_size(genome_size):
    """Recover H from a flat genome length: 54H + H + 6H + 6 = 61H + 6.

    Lets the probes work with h128 runs without hard-coding a second set of
    constants: the centroid genomes carry their own width.
    """
    per_unit = STATE_SIZE + 1 + OUTPUT_SIZE          # W1 row + b1 + one W2 column
    h, rem = divmod(int(genome_size) - B2_SIZE, per_unit)
    if rem != 0 or h <= 0:
        raise ValueError(f'genome length {genome_size} is not 61*H + 6 for any positive H')
    return h


def hidden_size_from_genome_csv(path):
    """Hidden width of the network a genome.csv was logged from (first row only).

    Cheap enough to call on the 'random' branches, which otherwise never touch
    the CSV: pandas reads a single row, and the base64 blob's byte length gives
    the genome length without decoding the whole file.
    """
    row = pd.read_csv(path, dtype={'genome_b64': str}, nrows=1)
    return infer_hidden_size(len(decode_b64(row['genome_b64'].iloc[0])))


def xavier_genome(seed=None, hidden_size=HIDDEN_SIZE, rng=None):
    """A freshly initialised brain, matching NNBrain._initGenome() exactly.

    The sim uses Glorot/Xavier *uniform*: each weight is drawn i.i.d. from
    U(-L, +L) with L = sqrt(6 / (fan_in + fan_out)) — so the limit differs per
    layer (W1: fan 54->H, W2: fan H->6) — and both bias blocks start at zero.
    An earlier version of this helper used a flat U(-0.1, 0.1) for the whole
    genome, which is ~2-3x too narrow and gives biases the sim never has.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    h = int(hidden_size)
    w1_lim = np.sqrt(6.0 / (STATE_SIZE + h))
    w2_lim = np.sqrt(6.0 / (h + OUTPUT_SIZE))
    return np.concatenate([
        rng.uniform(-w1_lim, w1_lim, STATE_SIZE * h),   # W1
        np.zeros(h),                                    # b1
        rng.uniform(-w2_lim, w2_lim, h * OUTPUT_SIZE),  # W2
        np.zeros(OUTPUT_SIZE),                          # b2
    ])


# ── genome.csv loading ────────────────────────────────────────────────────────
def load_genome_csv(path, record_type=None, decode=True):
    """Load a run's genome.csv. Columns:
        generation, map_seed, condition, record_type, founder_index,
        fitness, num_founders, genome_b64
    Returns a DataFrame; if decode=True adds a 'genome' column of float64 arrays.
    Filter by record_type ('centroid' | 'fittest' | 'founder') if given.
    """
    df = pd.read_csv(path, dtype={'genome_b64': str})
    if record_type is not None:
        df = df[df['record_type'] == record_type].reset_index(drop=True)
    if decode:
        df = df.copy()
        df['genome'] = df['genome_b64'].map(decode_b64)
    return df


def centroid_trajectory(path):
    """Return (generations, matrix) of the per-generation centroid genomes,
    sorted by generation. matrix shape (n_gens, 3910)."""
    df = load_genome_csv(path, record_type='centroid')
    df = df.sort_values('generation')
    gens = df['generation'].to_numpy()
    mat = np.vstack(df['genome'].to_numpy())
    return gens, mat


# ── joint trajectory PCA (numpy SVD) ──────────────────────────────────────────
def joint_pca_plane(centroid_stacks, anchor=None):
    """Fit a 2D plane by PCA on stacked centroid trajectories.

    centroid_stacks: list of (n_i, 3910) arrays (e.g. [evo_centroids, learn_centroids]).
    anchor: origin of the coordinate system (default = mean of all stacked rows,
            i.e. the trajectory mean — puts (0,0) mid-journey, not at gen 0).

    Returns dict with:
        anchor (3910,), u (3910,) v (3910,) unit basis vectors (top-2 PCs),
        explained_variance_ratio (2,), singular_values.
    PCA never sees fitness — it operates on weight vectors only, so the plane
    cannot be accused of being chosen to flatter the result.
    """
    X = np.vstack(centroid_stacks).astype(np.float64)
    if anchor is None:
        anchor = X.mean(axis=0)
    Xc = X - anchor
    # SVD: rows of Vt are principal directions.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = np.sum(S ** 2)
    evr = (S ** 2 / total_var) if total_var > 0 else np.zeros_like(S)
    u = Vt[0]
    v = Vt[1]
    return {
        'anchor': anchor,
        'u': u / np.linalg.norm(u),
        'v': v / np.linalg.norm(v),
        'explained_variance_ratio': evr[:2],
        'singular_values': S[:2],
    }


def project_onto_plane(points, plane):
    """Project genome vectors onto (u,v) plane coords relative to the anchor.
    Returns (n,2) array of (alpha, beta).

    errstate: on macOS/Accelerate (numpy 2.0) a large float64 matmul raises
    spurious divide/overflow/invalid flags even when every input is finite and
    every output is correct — verified against einsum and an explicit dot to
    3e-14 on a (400, 3910) projection. Silencing them here keeps the real
    warnings visible instead of drowning them in per-run noise.
    """
    P = np.atleast_2d(points).astype(np.float64) - plane['anchor']
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        a = P @ plane['u']
        b = P @ plane['v']
    return np.column_stack([a, b])


# ── jobs.json plumbing for src/eval/run_probe.js ──────────────────────────────
def write_jobs(path, genomes, jobs, config_overrides=None):
    """Write a jobs file for run_probe.js.
      genomes: dict {key: float64 array or b64 str}
      jobs:    list of {id, genome(key), map_index, rl(bool), ticks?}
    """
    genc = {}
    for k, v in genomes.items():
        genc[k] = v if isinstance(v, str) else encode_b64(v)
    spec = {'genomes': genc, 'jobs': jobs}
    if config_overrides:
        spec['config_overrides'] = config_overrides
    with open(path, 'w') as f:
        json.dump(spec, f)
    return path


def load_results(path):
    return pd.read_csv(path)


def run_probe(params_path, jobs_path, out_path, shard=None, node='node', check=True):
    """Shell out to node src/eval/run_probe.js (for LOCAL tests; on HPC the
    .slurm script calls it directly with --shard). Returns the results DataFrame.
    """
    cmd = [node, os.path.join(REPO_ROOT, 'src', 'eval', 'run_probe.js'),
           '--params', params_path, '--jobs', jobs_path, '--out', out_path]
    if shard is not None:
        cmd += ['--shard', shard]
    print('[ll_common] $', ' '.join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=check, cwd=REPO_ROOT)
    return load_results(out_path)
