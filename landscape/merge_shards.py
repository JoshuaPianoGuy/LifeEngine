"""
landscape/merge_shards.py — concatenate results_shard_*.csv into results.csv.

After the SLURM array (run_landscape_probe_array.slurm) finishes, each task has
written results_shard_<i>.csv. This merges them (dedup by job id, last wins) into
the single results.csv the stageN analyze/plot scripts expect.

  python landscape/merge_shards.py --out-dir landscape/out/<grid dir>
"""
import argparse
import glob
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', required=True)
ap.add_argument('--pattern', default='results_shard_*.csv')
a = ap.parse_args()

shards = sorted(glob.glob(os.path.join(a.out_dir, a.pattern)))
if not shards:
    raise SystemExit(f'no shard files matching {a.pattern} in {a.out_dir}')
frames = [pd.read_csv(s) for s in shards]
df = pd.concat(frames, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
out = os.path.join(a.out_dir, 'results.csv')
df.to_csv(out, index=False)
print(f'[merge] {len(shards)} shards, {before} rows -> {len(df)} unique -> {out}')
