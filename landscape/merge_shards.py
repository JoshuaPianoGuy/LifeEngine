"""
landscape/merge_shards.py — concatenate results_shard_*.csv into results.csv.

After the SLURM array finishes, each task has written results_shard_<i>.csv.
This merges them (dedup by job id, last wins) into the single results.csv the
stageN analyze/plot scripts expect.

  python landscape/merge_shards.py --out-dir landscape/out/<grid dir>

NUL-BYTE HOLES FROM A WALLTIME KILL
-----------------------------------
A task SIGKILLed on walltime can leave the shard CSV with a run of NUL bytes:
the filesystem recorded the extended file length but never flushed the data
blocks, so the gap reads back as \\x00. Whatever rows lived in that gap are
gone, and the row immediately after it survives with the NULs glued to its
front — so its id parses as empty and pandas reads it as NaN.

That is the dangerous shape, because nothing downstream errors. The row is
merged with a missing id, a completeness check counts its probe as absent, and
any groupby that keys on the id silently drops or mis-buckets it.

An interrupted write also shows up as CHARACTER-LEVEL garbling with no NUL
anywhere near it — observed once as `g1000_92_2_r1_on`, an id with a stray `_2`
spliced into it. That one is nastier: the row looks structurally fine, so a
shard containing it is not obviously damaged, and only the id is wrong.

So the merge validates EVERY shard, not just the ones holding NULs:
  * leading NULs are stripped, which recovers the intact row behind a hole;
  * every id is then checked against the probe-id grammar, and any row failing
    it is DROPPED rather than merged;
  * the counts are reported, because a shard needing repair is also a shard that
    lost rows into the hole — the ones the NULs replaced. Re-run the unit to
    fill those; resume skips everything already present.

Dropping is safe precisely because the id is what identifies a probe: a row
whose id cannot be trusted cannot be attributed to a genome, so it has no use
downstream, and the completeness check will report its probe as absent and send
it back through the queue.

NULs are stripped from the FRONT only (lstrip, not replace). Removing NULs from
the middle of a line would splice two fragments of different rows into one
syntactically valid record — a silent fabrication, and far worse than a dropped
row.

Observed 2026-08-26 on founders_baseline_evomatch: 36 of 50 shard files holed,
189 rows recovered from leading NULs, 1 row dropped for a garbled id.
"""
import argparse
import glob
import io
import os
import re

import pandas as pd

# Probe ids are the grid form p{i}_{j}_r{r}_{on|off} or the founder form
# g{gen}_{idx}_r{r}_{on|off}. Anything else is not a row this merge should keep.
ID_RE = re.compile(rb'^[pg]\d+_\d+_r\d+_(on|off),')

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', required=True)
ap.add_argument('--pattern', default='results_shard_*.csv')
ap.add_argument('--quiet', action='store_true')
a = ap.parse_args()

shards = sorted(glob.glob(os.path.join(a.out_dir, a.pattern)))
if not shards:
    raise SystemExit(f'no shard files matching {a.pattern} in {a.out_dir}')

frames, repaired, dropped, holed = [], 0, 0, 0
for s in shards:
    raw = open(s, 'rb').read()
    if b'\x00' in raw:
        holed += 1
    header, keep = None, []
    for line in raw.split(b'\n'):
        if not line:
            continue
        if line.startswith(b'id,'):
            header = line
            continue
        if ID_RE.match(line):
            keep.append(line)
            continue
        # Leading NULs only — see the module docstring on why not replace().
        stripped = line.lstrip(b'\x00')
        if stripped is not line and ID_RE.match(stripped):
            keep.append(stripped)
            repaired += 1
        else:
            dropped += 1
    if header is None:
        raise SystemExit(f'{s}: no header row survived — delete it and re-run '
                         f'that shard.')
    frames.append(pd.read_csv(io.BytesIO(b'\n'.join([header] + keep) + b'\n')))

df = pd.concat(frames, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
out = os.path.join(a.out_dir, 'results.csv')
df.to_csv(out, index=False)
if not a.quiet:
    note = ''
    if holed or repaired or dropped:
        note = (f'  [{holed} shard(s) with NUL holes; {repaired} rows repaired, '
                f'{dropped} dropped for an unusable id]')
    print(f'[merge] {len(shards)} shards, {before} rows -> {len(df)} unique '
          f'-> {out}{note}')
