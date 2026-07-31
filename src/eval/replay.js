'use strict';

/**
 * src/eval/replay.js — Stage 9 / B1: the REPLAY probe.
 *
 * The landscape's f(θ) is a MONOMORPHIC probe: 100 clones, reproduction off, one
 * 2000-tick map. That is the right construct for a landscape (fixed cohort, so
 * every grid cell is measured at the same density), but its units are not the
 * units a viability threshold is defined in. Reproduction fires at
 * `energyGainedSinceReproduction >= 3.5` (AdvancedOrganism.js) and succeeds with
 * probability 0.8, so the food an agent must bank to actually LAND one child is
 * 3.5 / 0.8 = 4.375 — and that number lives on the scale of a REAL generation: a
 * reproducing population over MAPS_PER_GEN maps with energy carryover.
 *
 * This driver measures that scale. Same monomorphic start (100 clones of θ, GA
 * never evolves), but reproduction ON, run across a full generation:
 *
 *     k = f_monomorphic(θ) / f_replay(θ)
 *
 * and the viability threshold on the probe scale is 4.375 * k. Stage 9's
 * `calibrate` computes k from these results; `metrics --k-json` consumes it.
 *
 * WHAT IS AVERAGED. `mean_all_agents` is the mean cumulative_food_score over
 * ga.all_agents — every organism that existed this generation, founders AND the
 * descendants reproduction created. That is the spec's statistic and the default
 * for k. `mean_founders` (the same statistic over the 100 founders only) is
 * emitted alongside because it is what src/eval/anchor.js reports and what
 * Logger stores on a centroid row — the two differ whenever reproduction is
 * doing much, and which one you want depends on whether "fitness" means the
 * lineage's or the individual's. Both are in the CSV; `--replay-field` picks.
 *
 * GA STILL DISABLED. Like anchor.js, the run stops ONE TICK SHORT of the
 * generation boundary, so env.update()'s NEXT_MAP transitions (load the next
 * map, carry survivors + their energy) all fire while the terminal
 * NEXT_GENERATION — which would call evolve() and respawn — never does. No
 * selection, no crossover, no between-generation mutation. The only genetic
 * event is the asexual within-generation mutation reproduce() applies to each
 * child, which is part of what a real generation IS and must not be removed.
 *
 * Jobs file schema (same shape as run_probe.js, different per-job fields):
 * {
 *   "config_overrides": { ... },
 *   "genomes": { "evo_centroid": "<b64>", ... },
 *   "jobs": [ { "id": "evo_centroid_r0_off", "genome": "evo_centroid",
 *               "map_start": 0, "maps": 5, "ticks": 2000, "rl": false } ]
 * }
 *
 *   node src/eval/replay.js --params <run/params.json> --jobs jobs_replay.json \
 *       --out results_replay.csv [--reproduction-rate 0.8] [--shard i/N]
 *
 * --shard / resume behave exactly as in run_probe.js.
 */

const fs = require('fs');

function parseArgs(argv) {
    const o = {};
    for (let i = 0; i < argv.length; i++) {
        if (!argv[i].startsWith('--')) continue;
        const k = argv[i].slice(2), nx = argv[i + 1];
        if (nx !== undefined && !nx.startsWith('--')) { o[k] = nx; i++; } else o[k] = true;
    }
    return o;
}
const opts = parseArgs(process.argv.slice(2));
if (!opts.params || !opts.jobs || !opts.out) {
    console.error('Usage: node src/eval/replay.js --params <params.json> --jobs <jobs.json> ' +
        '--out <results.csv> [--reproduction-rate 0.8] [--shard i/N]');
    process.exit(1);
}
const REPRO_RATE = opts['reproduction-rate'] != null
    ? parseFloat(opts['reproduction-rate']) : 0.8;

const spec = JSON.parse(fs.readFileSync(opts.jobs, 'utf8'));
const genomes = spec.genomes || {};
let jobs = spec.jobs || [];

let shardI = 0, shardN = 1;
if (typeof opts.shard === 'string' && opts.shard.includes('/')) {
    const [a, b] = opts.shard.split('/').map(x => parseInt(x, 10));
    shardI = a; shardN = b;
    if (!(shardN > 0 && shardI >= 0 && shardI < shardN)) {
        console.error(`Bad --shard ${opts.shard}: need i/N with 0<=i<N.`); process.exit(1);
    }
    jobs = jobs.filter((_, idx) => (idx % shardN) === shardI);
}

// ── config BEFORE the sim modules load ────────────────────────────────────────
// keepReproductionRate:true is the whole point of this driver — config.js
// otherwise forces reproduction_success_rate = 0 for the landscape probe.
// AdvancedOrganism.js reads the rate into a module-level const at load, so it
// MUST be set here, before any sim module is required.
const { applyConfig, decodeGenomeB64 } = require('./config');
const cfg = applyConfig({
    paramsPath: opts.params,
    overrides: Object.assign({}, spec.config_overrides || {},
        { reproduction_success_rate: REPRO_RATE }),
    keepReproductionRate: true,
});
const ExperimentParams = require('../ExperimentParams');
if (ExperimentParams.reproduction_success_rate !== REPRO_RATE) {
    console.error(`[replay] FATAL: reproduction_success_rate is ` +
        `${ExperimentParams.reproduction_success_rate}, expected ${REPRO_RATE}. ` +
        `The replay scale would be meaningless — aborting.`);
    process.exit(1);
}
console.error(`[replay] config: ${JSON.stringify(cfg)}`);
console.error(`[replay] ${jobs.length} jobs${shardN > 1 ? ` (shard ${shardI}/${shardN})` : ''}`);

const WorldConfig = require('../WorldConfig');
const NNBrain = require('../Organism/Perception/NNBrain');
const GenerationConstants = require('../Organism/GenerationConstants');
const TICKS_PER_MAP = GenerationConstants.TICKS_PER_MAP;
const MAPS_PER_GEN = GenerationConstants.MAPS_PER_GEN;
const POP = ExperimentParams.population_size;

// ── one env per RL flag (learning_enabled is read at env construction) ─────────
const envs = {};
function envFor(rl) {
    const key = rl ? 'on' : 'off';
    if (envs[key]) return envs[key];
    WorldConfig.headless = true;
    WorldConfig.auto_pause = false;
    WorldConfig.auto_reset = false;
    WorldConfig.learning_enabled = !!rl;
    WorldConfig.experiment_mode = 'standard';
    const WorldEnvironment = require('../Environments/WorldEnvironment');
    const engine = {
        fps: Infinity, last_fps: Infinity, running: true,
        stop() { this.running = false; }, start() {}, restart() {},
    };
    const _log = console.log; console.log = () => {};
    let env;
    try { env = new WorldEnvironment(engine, cfg.cell_size || 2); }
    finally { console.log = _log; }
    envs[key] = { env, ga: env.ga_manager };
    return envs[key];
}

function runReplay(theta, { mapStart, maps, ticks, rl }) {
    const { env, ga } = envFor(rl);
    const _log = console.log; console.log = () => {};
    try {
        const poolLen = (env._map_pool && env._map_pool.length) ? env._map_pool.length : 50;
        const seq = [];
        for (let m = 0; m < maps; m++) seq.push((mapStart + m) % poolLen);

        env.total_ticks = 0;
        env._map_sequence = seq;
        env._sequence_index = 0;
        env.generateWorld();

        const pool = new Array(POP);
        for (let i = 0; i < POP; i++) pool[i] = new Float32Array(theta);
        ga.gene_pool = pool;
        ga.spawnGeneration();
        env.predator_manager.spawnAll();
        const founders = ga.founders.slice();

        // One tick short of the generation boundary: every NEXT_MAP fires, the
        // terminal NEXT_GENERATION (evolve + respawn) does not.
        const total = maps * ticks - 1;
        for (let t = 0; t < total; t++) env.update();

        const all = ga.all_agents;
        let sumAll = 0, lifeAll = 0;
        for (const a of all) { sumAll += a.getFitness(); lifeAll += a.lifetime || 0; }
        let sumF = 0, alive = 0;
        for (const a of founders) { sumF += a.getFitness(); if (a.living) alive++; }

        return {
            mean_all_agents: all.length ? sumAll / all.length : 0,
            mean_founders: founders.length ? sumF / founders.length : 0,
            n_all_agents: all.length,
            n_founders: founders.length,
            alive_founders: alive,
            final_pop: env.organisms.length,
            mean_lifetime: all.length ? lifeAll / all.length : 0,
        };
    } finally { console.log = _log; }
}

// ── resume support ────────────────────────────────────────────────────────────
const COLUMNS = ['id', 'rl', 'map_start', 'maps', 'ticks', 'mean_all_agents', 'mean_founders',
    'n_all_agents', 'n_founders', 'alive_founders', 'final_pop', 'mean_lifetime'];

const done = new Set();
let needHeader = true;
if (fs.existsSync(opts.out)) {
    const prev = fs.readFileSync(opts.out, 'utf8').split('\n');
    if (prev.length && prev[0].startsWith('id,')) {
        needHeader = false;
        for (let i = 1; i < prev.length; i++) {
            const c = prev[i].split(',');
            if (c[0]) done.add(c[0]);
        }
    }
}
const outFd = fs.openSync(opts.out, 'a');
if (needHeader) fs.writeSync(outFd, COLUMNS.join(',') + '\n');

// ── run ───────────────────────────────────────────────────────────────────────
const t_start = Date.now();
let ran = 0, skipped = 0;
for (let j = 0; j < jobs.length; j++) {
    const job = jobs[j];
    if (done.has(job.id)) { skipped++; continue; }

    const b64 = genomes[job.genome];
    if (!b64) { console.error(`[replay] job ${job.id}: unknown genome '${job.genome}' — skipping`); continue; }
    const theta = decodeGenomeB64(b64);
    if (theta.length !== NNBrain.GENOME_SIZE) {
        console.error(`[replay] job ${job.id}: genome length ${theta.length} != ${NNBrain.GENOME_SIZE}`);
        process.exit(1);
    }
    const maps = job.maps != null ? (job.maps | 0) : MAPS_PER_GEN;
    const ticks = job.ticks != null ? (job.ticks | 0) : TICKS_PER_MAP;
    const rl = !!job.rl;

    const r = runReplay(theta, { mapStart: job.map_start | 0, maps, ticks, rl });
    const row = [
        job.id, rl ? 1 : 0, job.map_start | 0, maps, ticks,
        r.mean_all_agents.toFixed(6), r.mean_founders.toFixed(6),
        r.n_all_agents, r.n_founders, r.alive_founders, r.final_pop,
        r.mean_lifetime.toFixed(2),
    ].join(',');
    fs.writeSync(outFd, row + '\n');
    ran++;

    const el = (Date.now() - t_start) / 1000;
    const rate = ran / Math.max(el, 1e-9);
    console.error(`[replay] ${job.id}: all=${r.mean_all_agents.toFixed(4)} ` +
        `(n=${r.n_all_agents}) founders=${r.mean_founders.toFixed(4)} ` +
        `final_pop=${r.final_pop} | ${ran}/${jobs.length - skipped} ` +
        `eta ${(((jobs.length - skipped - ran) / Math.max(rate, 1e-9)) / 60).toFixed(1)}min`);
}
fs.closeSync(outFd);
console.error(`[replay] done: ${ran} ran, ${skipped} skipped, ` +
    `${((Date.now() - t_start) / 1000).toFixed(1)}s -> ${opts.out}`);
