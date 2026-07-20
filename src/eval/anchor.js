'use strict';

/**
 * src/eval/anchor.js — Stage 2 anchor check (the ONE reproduction-on figure).
 *
 * The landscape everywhere uses the fixed-100 monomorphic probe. That probe's
 * fitness is not directly comparable to the logged generation fitness, because a
 * real generation reproduces (rate 0.8) into a large, food-competing population
 * over 5 maps. This script closes that gap for a SINGLE validation number:
 *
 *   Take the gen-G centroid genome, seed 100 monomorphic founders, run a FULL
 *   real generation (MAPS_PER_GEN maps, energy carryover, reproduction ON),
 *   stop BEFORE evolve(), and report the mean founder cumulative_food_score —
 *   the exact statistic Logger stores on the centroid row. Average over R
 *   repeats (each a different 5-map block) and compare to the logged fitness
 *   (that generation, and a ±window average to beat terrain noise).
 *
 * If anchor ~= logged  -> the centroid is a valid unimodal representative and is
 *                         a sound landscape origin.
 * If anchor << logged  -> the population was multimodal (clades); the centroid
 *                         sits in a valley — switch the anchor to the logged
 *                         fittest founder everywhere.
 *
 *   node src/eval/anchor.js --params <run/params.json> --genome-csv <run/genome.csv> \
 *       --gen 500 --repeats 5 [--rl] [--window 10] [--config-overrides '{...}']
 *
 * --rl selects the RL flag the centroid was evolved under (learning centroid =>
 * --rl; evolution centroid => omit). --config-overrides can retarget the
 * environment (e.g. probe a baseline genome under the hard predator config).
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
if (!opts.params || !opts['genome-csv'] || opts.gen == null) {
    console.error('Usage: node src/eval/anchor.js --params <params.json> --genome-csv <genome.csv> ' +
        '--gen <G> [--repeats 5] [--rl] [--window 10] [--config-overrides <json>]');
    process.exit(1);
}
const TARGET_GEN = parseInt(opts.gen, 10);
const REPEATS    = opts.repeats != null ? parseInt(opts.repeats, 10) : 5;
const RL         = !!opts.rl;
const WINDOW     = opts.window != null ? parseInt(opts.window, 10) : 10;

// ── Read the centroid genome at TARGET_GEN + logged fitness (and window avg) ───
function readCentroids(csvPath) {
    const text = fs.readFileSync(csvPath, 'utf8').split('\n');
    const h = text[0].split(',');
    const iGen = h.indexOf('generation'), iRt = h.indexOf('record_type'),
          iFit = h.indexOf('fitness'), iB64 = h.indexOf('genome_b64');
    const rows = [];
    for (let i = 1; i < text.length; i++) {
        if (!text[i]) continue;
        const c = text[i].split(',');
        if (c[iRt] !== 'centroid') continue;
        rows.push({ gen: parseInt(c[iGen], 10), fitness: parseFloat(c[iFit]), b64: c[iB64] });
    }
    return rows;
}

const centroids = readCentroids(opts['genome-csv']);
const target = centroids.find(r => r.gen === TARGET_GEN) ||
    centroids.reduce((best, r) => Math.abs(r.gen - TARGET_GEN) < Math.abs(best.gen - TARGET_GEN) ? r : best);
const windowRows = centroids.filter(r => Math.abs(r.gen - target.gen) <= WINDOW);
const loggedWindowMean = windowRows.reduce((s, r) => s + r.fitness, 0) / windowRows.length;

// ── Apply config (KEEP reproduction rate — the anchor's whole point) ──────────
const { applyConfig, decodeGenomeB64 } = require('./config');
const cfg = applyConfig({
    paramsPath: opts.params,
    overrides: opts['config-overrides'] ? JSON.parse(opts['config-overrides']) : {},
    keepReproductionRate: true,
});
const ExperimentParams = require('../ExperimentParams');
console.error(`[anchor] gen=${target.gen} rl=${RL} repeats=${REPEATS} ` +
    `reproduction_rate=${ExperimentParams.reproduction_success_rate} ` +
    `logged_fit(gen)=${target.fitness.toFixed(4)} logged_fit(±${WINDOW}win)=${loggedWindowMean.toFixed(4)}`);

const theta = decodeGenomeB64(target.b64);
const NNBrain = require('../Organism/Perception/NNBrain');
if (theta.length !== NNBrain.GENOME_SIZE) { console.error('[anchor] genome length mismatch'); process.exit(1); }

// ── Boot env (mirrors headless.js, minus logging) ─────────────────────────────
const WorldConfig = require('../WorldConfig');
WorldConfig.headless = true; WorldConfig.auto_pause = false; WorldConfig.auto_reset = false;
WorldConfig.learning_enabled = RL; WorldConfig.experiment_mode = 'standard';
const WorldEnvironment = require('../Environments/WorldEnvironment');
const GenerationConstants = require('../Organism/GenerationConstants');
const TICKS_PER_MAP = GenerationConstants.TICKS_PER_MAP;
const MAPS_PER_GEN  = GenerationConstants.MAPS_PER_GEN;
const POP = ExperimentParams.population_size;

const engine = { fps: Infinity, last_fps: Infinity, running: true,
    stop() { this.running = false; }, start() {}, restart() {} };

// One env, reused across repeats. Silence the sim's per-map logging.
const _log = console.log; console.log = () => {};
const env = new WorldEnvironment(engine, cfg.cell_size || 2);
const ga = env.ga_manager;
const poolLen = (env._map_pool && env._map_pool.length) ? env._map_pool.length : 50;

function runOneGeneration(mapBlockStart) {
    // 5-map block for this repeat (different terrain per repeat, wrapping the pool).
    const seq = [];
    for (let m = 0; m < MAPS_PER_GEN; m++) seq.push((mapBlockStart + m) % poolLen);
    env.total_ticks = 0;
    env._map_sequence = seq;
    env._sequence_index = 0;
    env.generateWorld();                 // loads seq[0]
    const pool = new Array(POP);
    for (let i = 0; i < POP; i++) pool[i] = new Float32Array(theta);
    ga.gene_pool = pool;
    ga.spawnGeneration();                // 100 monomorphic founders
    env.predator_manager.spawnAll();
    const founders = ga.founders.slice();

    // Run the full generation EXCEPT its final tick, so env.update()'s NEXT_MAP
    // transitions (reload next map, carry survivors) all fire but the terminal
    // NEXT_GENERATION (which would call evolve + respawn) never does.
    const total = MAPS_PER_GEN * TICKS_PER_MAP - 1;
    for (let t = 0; t < total; t++) env.update();

    let sum = 0, alive = 0;
    for (const a of founders) { sum += a.getFitness(); if (a.living) alive++; }
    const pop = env.organisms.length;
    return { founder_mean: sum / founders.length, alive, final_pop: pop,
             n_all: ga.all_agents.length };
}

const results = [];
for (let r = 0; r < REPEATS; r++) {
    const res = runOneGeneration(r * MAPS_PER_GEN);
    results.push(res);
    console.error(`[anchor] repeat ${r}: founder_mean=${res.founder_mean.toFixed(4)} ` +
        `alive=${res.alive}/${POP} final_pop=${res.final_pop} total_agents=${res.n_all}`);
}
console.log = _log;

const means = results.map(r => r.founder_mean);
const mean = means.reduce((s, x) => s + x, 0) / means.length;
const sd = means.length > 1
    ? Math.sqrt(means.reduce((s, x) => s + (x - mean) ** 2, 0) / (means.length - 1)) : 0;
const se = sd / Math.sqrt(means.length);

console.log('\n=== ANCHOR RESULT ===');
console.log(`  centroid gen:            ${target.gen}  (rl=${RL})`);
console.log(`  anchor founder-mean:     ${mean.toFixed(4)}  ± ${sd.toFixed(4)} (sd, R=${REPEATS}), SE=${se.toFixed(4)}`);
console.log(`  logged fitness (gen):    ${target.fitness.toFixed(4)}`);
console.log(`  logged fitness (±${WINDOW}win): ${loggedWindowMean.toFixed(4)}`);
const ratio = loggedWindowMean !== 0 ? mean / loggedWindowMean : NaN;
console.log(`  anchor / logged(window): ${ratio.toFixed(3)}`);
console.log(ratio >= 0.75 && ratio <= 1.5
    ? '  ⇒ Anchor consistent with logged fitness — centroid is a valid unimodal anchor. ✅'
    : '  ⇒ Anchor DIVERGES from logged fitness — consider the fittest-founder anchor instead. ⚠');
