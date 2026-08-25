'use strict';

/**
 * src/eval/run_probe.js — batch driver for the Stage-1 probe harness.
 *
 * Reads a JOBS file (JSON) describing many f(θ) evaluations, runs them, and
 * streams one CSV row per job to --out. Every later stage (PCA plane, random
 * directions, λ walk) generates its θ points in Python, writes a jobs file, and
 * calls this. This is the ONLY process that touches the JS simulator.
 *
 *   node src/eval/run_probe.js --params <run/params.json> --jobs jobs.json --out results.csv
 *   node src/eval/run_probe.js --params ... --jobs jobs.json --out r.csv --shard 3/16
 *
 * Jobs file schema:
 * {
 *   "config_overrides": { "roaming_predator_count": 60, "predator_drain": 5, ... },
 *      // optional; overrides params.json so one θ set can be probed in a
 *      // DIFFERENT environment than the run it came from (baseline vs hard).
 *   "genomes": { "g0": "<b64>", "g1": "<b64>", ... },   // deduped genome pool
 *   "jobs": [
 *     { "id": "coarse_i0_j0_r0", "genome": "g0", "map_index": 4, "rl": false, "ticks": 2000 },
 *     ...
 *   ]
 * }
 *
 * --shard i/N runs only jobs whose position ≡ i (mod N) — split a sweep across
 * N array-job tasks. Each shard writes its own --out; concatenate afterwards.
 * Re-running is safe: if --out already has a row for a job id, it is skipped
 * (resume after a crash / preemption).
 */

const fs   = require('fs');
const path = require('path');

// ── args ──────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
    const o = {};
    for (let i = 0; i < argv.length; i++) {
        if (!argv[i].startsWith('--')) continue;
        const k = argv[i].slice(2);
        const nx = argv[i + 1];
        if (nx !== undefined && !nx.startsWith('--')) { o[k] = nx; i++; } else o[k] = true;
    }
    return o;
}
const opts = parseArgs(process.argv.slice(2));
if (!opts.params || !opts.jobs || !opts.out) {
    console.error('Usage: node src/eval/run_probe.js --params <params.json> --jobs <jobs.json> ' +
        '--out <results.csv> [--shard i/N]');
    process.exit(1);
}

const spec = JSON.parse(fs.readFileSync(opts.jobs, 'utf8'));
const genomes = spec.genomes || {};
let jobs = spec.jobs || [];

// Shard filter
let shardI = 0, shardN = 1;
if (typeof opts.shard === 'string' && opts.shard.includes('/')) {
    const [a, b] = opts.shard.split('/').map(x => parseInt(x, 10));
    shardI = a; shardN = b;
    if (!(shardN > 0 && shardI >= 0 && shardI < shardN)) {
        console.error(`Bad --shard ${opts.shard}: need i/N with 0<=i<N.`); process.exit(1);
    }
    jobs = jobs.filter((_, idx) => (idx % shardN) === shardI);
}

// ── apply config BEFORE requiring the probe ─────────────────────────────────────
const { applyConfig, decodeGenomeB64 } = require('./config');
const cfg = applyConfig({ paramsPath: opts.params, overrides: spec.config_overrides || {} });
console.error(`[run_probe] config: ${JSON.stringify(cfg)}`);
console.error(`[run_probe] ${jobs.length} jobs${shardN > 1 ? ` (shard ${shardI}/${shardN})` : ''}`);

const { Probe } = require('./probe');

// ── output schema ─────────────────────────────────────────────────────────────
// BASE is what every landscape stage has always written; do not reorder or
// insert into it, several readers index these by name and the merged
// results.csv files on disk carry exactly this header.
const BASE_COLUMNS = ['id', 'rl', 'map_index', 'ticks', 'mean_fitness', 'std_clones', 'n_clones',
    'mean_lifetime', 'mean_cave_entries', 'mean_drained_ticks', 'mean_predator_touches',
    'deaths_survived', 'deaths_starved', 'deaths_drained', 'deaths_lifespan'];

// Opt-in extras, requested by the jobs file as "metrics": "founder". APPENDED,
// never interleaved, so a BASE reader that selects by name is unaffected.
// Off by default precisely so an existing landscape shard CSV can still be
// resumed by a newer build of this script without the row width changing
// underneath it.
const EXTRA_COLUMNS = { founder: ['mean_weight_drift'] };

const metricsMode = typeof spec.metrics === 'string' ? spec.metrics : 'base';
if (metricsMode !== 'base' && !EXTRA_COLUMNS[metricsMode]) {
    console.error(`Unknown "metrics": ${JSON.stringify(metricsMode)} in the jobs file. ` +
        `Known: base, ${Object.keys(EXTRA_COLUMNS).join(', ')}.`);
    process.exit(1);
}
const COLUMNS = BASE_COLUMNS.concat(EXTRA_COLUMNS[metricsMode] || []);

// ── resume support: read already-done ids from --out ───────────────────────────
const done = new Set();
let needHeader = true;
if (fs.existsSync(opts.out)) {
    const prev = fs.readFileSync(opts.out, 'utf8').split('\n');
    if (prev.length && prev[0].startsWith('id,')) {
        // Schema guard: appending rows of a different width to an existing file
        // produces a ragged CSV that pandas rejects only later, after the
        // walltime has been spent. Refuse now and say what to do.
        const prevHeader = prev[0].trim();
        if (prevHeader !== COLUMNS.join(',')) {
            console.error(`Schema mismatch: ${opts.out} has header\n  ${prevHeader}\n` +
                `but this run would write\n  ${COLUMNS.join(',')}\n` +
                `(jobs file "metrics": ${JSON.stringify(metricsMode)}). Resuming would ` +
                `append rows of a different width. Either match the jobs file to the ` +
                `existing output, or delete it and start that shard clean.`);
            process.exit(1);
        }
        needHeader = false;
        for (let i = 1; i < prev.length; i++) {
            const c = prev[i].split(',');
            if (c[0]) done.add(c[0]);
        }
    }
}
const outFd = fs.openSync(opts.out, 'a');
if (needHeader) fs.writeSync(outFd, COLUMNS.join(',') + '\n');

// ── lazily-built probes, one per RL flag, reused across jobs ────────────────────
let probeOff = null, probeOn = null;
function probeFor(rl) {
    if (rl) return (probeOn  || (probeOn  = new Probe(true)));
    return       (probeOff || (probeOff = new Probe(false)));
}

// ── run ─────────────────────────────────────────────────────────────────────────
const t_start = Date.now();
let ran = 0, skipped = 0;
for (let j = 0; j < jobs.length; j++) {
    const job = jobs[j];
    if (done.has(job.id)) { skipped++; continue; }

    const b64 = genomes[job.genome];
    if (!b64) { console.error(`[run_probe] job ${job.id}: unknown genome key '${job.genome}' — skipping`); continue; }
    const theta = decodeGenomeB64(b64);
    const rl = !!job.rl;

    const res = probeFor(rl).evaluate(theta, { mapIndex: job.map_index | 0, ticks: job.ticks });

    const cells = [
        job.id, rl ? 1 : 0, res.map_index, res.ticks,
        res.mean_fitness.toFixed(6), res.std_clones.toFixed(6), res.n_clones,
        res.mean_lifetime.toFixed(2), res.mean_cave_entries.toFixed(4),
        res.mean_drained_ticks.toFixed(4), res.mean_predator_touches.toFixed(4),
        res.deaths.survived, res.deaths.starved, res.deaths.drained, res.deaths.lifespan,
    ];
    if (metricsMode === 'founder') cells.push(res.mean_weight_drift.toFixed(8));
    const row = cells.join(',');
    fs.writeSync(outFd, row + '\n');
    ran++;

    if (ran % 10 === 0 || j === jobs.length - 1) {
        const el = (Date.now() - t_start) / 1000;
        const rate = ran / Math.max(el, 1e-9);
        const remain = (jobs.length - skipped - ran) / Math.max(rate, 1e-9);
        console.error(`[run_probe] ${ran} ran, ${skipped} skipped / ${jobs.length}  ` +
            `${rate.toFixed(2)} job/s  eta ${(remain / 60).toFixed(1)}min`);
    }
}
fs.closeSync(outFd);
console.error(`[run_probe] done: ${ran} ran, ${skipped} skipped, ${((Date.now() - t_start) / 1000).toFixed(1)}s -> ${opts.out}`);
