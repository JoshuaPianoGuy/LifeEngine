"""
landscape/run_grid.py — run one grid out-dir's jobs.json through the JS probe.

  python landscape/run_grid.py --out-dir <grid dir> --params <run/params.json> [--shard i/N]

On HPC, call src/eval/run_probe.js directly from the .slurm array with --shard;
this wrapper is for local runs. Results stream to <out-dir>/results.csv (resumable).
"""
import argparse
import os
import ll_common as ll

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', required=True)
ap.add_argument('--params', required=True)
ap.add_argument('--shard', default=None)
a = ap.parse_args()
ll.run_probe(a.params, os.path.join(a.out_dir, 'jobs.json'),
             os.path.join(a.out_dir, 'results.csv'), shard=a.shard)
print(f'[run_grid] -> {os.path.join(a.out_dir, "results.csv")}')
