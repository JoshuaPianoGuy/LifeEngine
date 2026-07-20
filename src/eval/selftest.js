'use strict';

/**
 * src/eval/selftest.js — validate the Stage-1 probe harness.
 *
 *   node src/eval/selftest.js
 *
 * Runs the two load-bearing asserts from the plan plus a smoke probe:
 *   ASSERT A — RL-off ⇒ MAD (|active − genome|) is exactly 0 for every founder.
 *   DIAG   B — measure the actual energy cost of reproduction (the plan claims
 *              3.5 is deducted even when no child spawns; this reports the truth).
 *   SMOKE    — evaluate the real seed-999 baseline evolution final centroid,
 *              RL-off and RL-on, on one map; print mean fitness + diagnostics.
 *
 * Uses the baseline (no-predator) run so the smoke test is fast and
 * deterministic-ish (only food dynamics vary).
 */

const path = require('path');
const fs   = require('fs');

const RUN_DIR = path.join(__dirname, '..', '..', 'logs', 'evolution', 'standard',
    'auto-run', 'baseline_g1k_evolution_lscape_seed999_r1');
const PARAMS  = path.join(RUN_DIR, 'params.json');
const GENOME  = path.join(RUN_DIR, 'genome.csv');

// ── Apply the run's config BEFORE requiring the probe / sim modules ───────────
const { applyConfig, decodeGenomeB64 } = require('./config');
const cfg = applyConfig({ paramsPath: PARAMS });
console.log('[selftest] resolved config:', JSON.stringify(cfg, null, 2));

const NNBrain = require('../Organism/Perception/NNBrain');

// ── Read the last centroid genome from genome.csv ─────────────────────────────
function lastRecordGenome(csvPath, recordType) {
    const text = fs.readFileSync(csvPath, 'utf8');
    const lines = text.split('\n');
    const header = lines[0].split(',');
    const rtIdx = header.indexOf('record_type');
    const gbIdx = header.indexOf('genome_b64');
    const genIdx = header.indexOf('generation');
    let last = null;
    for (let i = 1; i < lines.length; i++) {
        const ln = lines[i];
        if (!ln) continue;
        // genome_b64 is the last column and has no commas, so a plain split is safe.
        const cols = ln.split(',');
        if (cols[rtIdx] === recordType) last = { gen: cols[genIdx], b64: cols[gbIdx] };
    }
    return last;
}

const centroidRow = lastRecordGenome(GENOME, 'centroid');
const theta = decodeGenomeB64(centroidRow.b64);
console.log(`[selftest] loaded final centroid (gen ${centroidRow.gen}), length=${theta.length} ` +
    `(GENOME_SIZE=${NNBrain.GENOME_SIZE})`);
if (theta.length !== NNBrain.GENOME_SIZE) {
    console.error(`[selftest] FATAL: genome length mismatch.`);
    process.exit(1);
}

const { Probe } = require('./probe');

// ── DIAG B: does reproduction deduct energy? ──────────────────────────────────
function reproductionEnergyCost() {
    const AdvancedOrganism = require('../Organism/AdvancedOrganism');
    const probe = new Probe(false);          // borrow its booted env
    const env = probe.env, ga = probe.ga;
    // Need a valid map so isClear/spawn checks run against real cells.
    env.total_ticks = 0;
    env._map_sequence = [0]; env._sequence_index = 0;
    env.generateWorld();
    const org = new AdvancedOrganism(0, 0, env, null, false, ga);
    // Give it the standard experiment anatomy so it's a well-formed organism.
    const CellStates = require('../Organism/Cell/CellStates');
    org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);
    org.anatomy.addDefaultCell(CellStates.mover, 0, 1);
    org.anatomy.checkTypeChange();
    org.c = ga.spawn_col; org.r = ga.spawn_row;
    const before = org.energy;
    org.energyGainedSinceReproduction = 3.5;   // at threshold → reproduce() fires
    org.reproduce();
    const after = org.energy;
    return { before, after, delta: after - before,
             counter_after: org.energyGainedSinceReproduction };
}

// ── Run everything ────────────────────────────────────────────────────────────
console.log('\n=== DIAG B: reproduction energy cost ===');
const diagB = reproductionEnergyCost();
console.log(`  parent energy before reproduce(): ${diagB.before}`);
console.log(`  parent energy after  reproduce(): ${diagB.after}`);
console.log(`  ΔEenergy on reproduction:         ${diagB.delta}`);
console.log(`  reproduction counter after:       ${diagB.counter_after} (reset to 0)`);
console.log(diagB.delta === 0
    ? '  ⇒ Reproduction is ENERGY-NEUTRAL (no 3.5 deduction). The plan\'s assert B\n' +
      '    premise does not hold; reproduction_success_rate=0 is used for the\n' +
      '    fixed-cohort/density reason, which is unaffected.'
    : `  ⇒ Reproduction deducts ${(-diagB.delta).toFixed(3)} energy.`);

console.log('\n=== ASSERT A + SMOKE (RL-off) ===');
const probeOff = new Probe(false);
const t0 = Date.now();
const rOff = probeOff.evaluate(theta, { mapIndex: 0 });
const invariant = probeOff.assertRlOffInvariant();
console.log(`  RL-off invariant: max MAD=${invariant.mad}, aliased_ok=${invariant.aliased_ok}  ✅`);
console.log(`  mean_fitness=${rOff.mean_fitness.toFixed(4)}  std_clones=${rOff.std_clones.toFixed(4)} ` +
    `n=${rOff.n_clones}  mean_lifetime=${rOff.mean_lifetime.toFixed(0)}  ` +
    `deaths=${JSON.stringify(rOff.deaths)}  (${((Date.now()-t0)/1000).toFixed(1)}s)`);

console.log('\n=== SMOKE (RL-on) ===');
const probeOn = new Probe(true);
const t1 = Date.now();
const rOn = probeOn.evaluate(theta, { mapIndex: 0 });
console.log(`  mean_fitness=${rOn.mean_fitness.toFixed(4)}  std_clones=${rOn.std_clones.toFixed(4)} ` +
    `n=${rOn.n_clones}  mean_lifetime=${rOn.mean_lifetime.toFixed(0)}  ` +
    `deaths=${JSON.stringify(rOn.deaths)}  (${((Date.now()-t1)/1000).toFixed(1)}s)`);

console.log('\n=== SMOKE (random genome, RL-off, should be ~low fitness) ===');
const rnd = new Float32Array(NNBrain.GENOME_SIZE);
for (let i = 0; i < rnd.length; i++) rnd[i] = (Math.random() * 2 - 1) * 0.1;
const rRnd = probeOff.evaluate(rnd, { mapIndex: 0 });
console.log(`  mean_fitness=${rRnd.mean_fitness.toFixed(4)}  mean_lifetime=${rRnd.mean_lifetime.toFixed(0)}`);

console.log('\n[selftest] done.');
