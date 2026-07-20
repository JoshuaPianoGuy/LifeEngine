'use strict';

/**
 * src/eval/config.js — apply run configuration BEFORE the sim modules load.
 *
 * The fitness-landscape probe must evaluate θ in the SAME environment that
 * produced it (grid size, network width, predator config, RL hyperparameters).
 * The easiest, least-error-prone way is to point at the run's own params.json;
 * every field there mirrors ExperimentParams / WorldConfig / the predator knobs.
 *
 * Call applyConfig(...) ONCE, before requiring probe.js (or any sim module).
 * It mirrors the override sequence in src/headless.js, then forces
 * reproduction_success_rate = 0 (the probe's fixed-cohort requirement).
 *
 * Usage:
 *   const { applyConfig } = require('./config');
 *   const cfg = applyConfig({ paramsPath: '.../params.json', overrides: {...} });
 *   const { Probe } = require('./probe');   // sized/configured per cfg
 */

const fs = require('fs');

// Mock browser globals so canvas modules load headless (mirrors headless.js).
function installBrowserStubs() {
    if (global.document) return;
    const mockCtx = new Proxy({}, {
        get: (_, prop) => typeof prop === 'string' ? () => {} : undefined,
        set: () => true,
    });
    const mockEl = {
        getContext: () => mockCtx,
        addEventListener: () => {}, removeEventListener: () => {},
        onwheel: null, width: 800, height: 600,
    };
    global.document = {
        getElementById: () => mockEl, querySelector: () => mockEl, createElement: () => mockEl,
    };
    global.window = global;
    try { global.navigator = { userAgent: '' }; } catch (_) {}
    global.confirm = () => false;
    const jqChain = new Proxy({}, {
        get(_, prop) {
            const primitives = { height: 600, width: 800, length: 1 };
            if (prop in primitives) return () => primitives[prop];
            if (prop === 'is') return () => false;
            if (prop === '0') return { click: () => {} };
            return function () { return jqChain; };
        }
    });
    global.$ = new Proxy(function () { return jqChain; }, {
        get(_, prop) {
            if (prop === 'fn') return {};
            return jqChain[prop] || function () { return jqChain; };
        }
    });
}

/**
 * @param {object}  args
 * @param {string}  [args.paramsPath]  path to a run's params.json to inherit from
 * @param {object}  [args.overrides]   explicit field overrides (win over params.json)
 * @param {boolean} [args.keepReproductionRate=false]  by default the probe forces
 *                  reproduction_success_rate=0 (fixed monomorphic cohort). Set true
 *                  for the Stage-2 ANCHOR run only, which needs the real rate (0.8)
 *                  so a reproducing population matches the logged generation fitness.
 * @returns {object} the resolved config actually applied (for logging / manifests)
 */
function applyConfig(args = {}) {
    installBrowserStubs();

    let base = {};
    if (args.paramsPath) {
        base = JSON.parse(fs.readFileSync(args.paramsPath, 'utf8'));
    }
    const o = Object.assign({}, base, args.overrides || {});

    // Fields must be numbers where numeric — params.json already stores them so,
    // but CLI overrides may arrive as strings; coerce defensively.
    const num = (v, d) => (v == null ? d : (typeof v === 'string' ? parseFloat(v) : v));
    const int = (v, d) => (v == null ? d : (typeof v === 'string' ? parseInt(v, 10) : v));

    // ── ExperimentParams (read at sim-module load) ────────────────────────────
    const ExperimentParams = require('../ExperimentParams');
    ExperimentParams.applyOverrides({
        learning_rate:          num(o.learning_rate),
        epsilon_start:          num(o.epsilon_start),
        epsilon_end:            num(o.epsilon_end),
        epsilon_decay_shape:    o.epsilon_decay_shape,
        epsilon_enabled:        (o.epsilon_enabled === false) ? false : null,
        explore_bonus:          num(o.explore_bonus),
        hidden_size:            int(o.hidden_size),
        population_size:        int(o.population_size),
        mut_prob:               num(o.mut_prob),
        mut_sigma:              num(o.mut_sigma),
        // Predators mirrored here for the run manifest (real knobs set below).
        predator_drain:         num(o.predator_drain),
        predators_per_patch:    int(o.predators_per_patch),
        roaming_predator_count: int(o.roaming_predator_count),
    });
    // The probe's non-negotiable: fixed monomorphic cohort — no descendants.
    // The anchor run is the sole exception (keepReproductionRate), where a
    // reproducing population must match the logged generation fitness.
    if (!args.keepReproductionRate) {
        ExperimentParams.reproduction_success_rate = 0;
    } else if (o.reproduction_success_rate != null) {
        ExperimentParams.reproduction_success_rate = num(o.reproduction_success_rate);
    }

    // ── WorldConfig ───────────────────────────────────────────────────────────
    const WorldConfig = require('../WorldConfig');
    WorldConfig.headless   = true;
    WorldConfig.auto_pause = false;
    WorldConfig.auto_reset = false;
    WorldConfig.MAP_SEED   = int(o.map_seed);
    WorldConfig.MAP_COLS   = int(o.grid_cols, 400);
    WorldConfig.MAP_ROWS   = int(o.grid_rows, 300);
    if (o.ticks_per_map != null) WorldConfig.TICKS_PER_MAP = int(o.ticks_per_map);
    if (o.maps_per_gen  != null) WorldConfig.MAPS_PER_GEN  = int(o.maps_per_gen);

    // ── Predator knobs (mutable object read at runtime) ───────────────────────
    const PredatorHyperparameters = require('../Organism/PredatorHyperparameters');
    if (o.predator_drain != null)         PredatorHyperparameters.drainAmount = num(o.predator_drain);
    if (o.roaming_predator_count != null) PredatorHyperparameters.count = int(o.roaming_predator_count);
    if (o.predators_per_patch != null)    PredatorHyperparameters.patrol.predatorsPerPatch = int(o.predators_per_patch);

    // Founder-genome logging must stay OFF — the probe writes nothing to logs.
    const logger = require('../Logger');
    if (typeof logger.setGenomeLogging === 'function') logger.setGenomeLogging(false);

    return {
        map_seed:   WorldConfig.MAP_SEED,
        grid_cols:  WorldConfig.MAP_COLS,
        grid_rows:  WorldConfig.MAP_ROWS,
        cell_size:  int(o.cell_size, 2),
        hidden_size: ExperimentParams.hidden_size,
        population_size: ExperimentParams.population_size,
        ticks_per_map: require('../Organism/GenerationConstants').TICKS_PER_MAP,
        maps_per_gen:  require('../Organism/GenerationConstants').MAPS_PER_GEN,
        predator_drain: PredatorHyperparameters.drainAmount,
        roaming_predator_count: PredatorHyperparameters.count,
        predators_per_patch: PredatorHyperparameters.patrol.predatorsPerPatch,
        learning_rate: ExperimentParams.learning_rate,
        epsilon_enabled: ExperimentParams.epsilon_enabled,
        explore_bonus: ExperimentParams.explore_bonus,
        reproduction_success_rate: ExperimentParams.reproduction_success_rate,
    };
}

/** Decode a genome_b64 field (little-endian float32) to a Float32Array. */
function decodeGenomeB64(b64) {
    const buf = Buffer.from(b64, 'base64');
    // Ensure the byte offset/length line up with Float32Array (4-byte) view.
    return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
}

/** Encode a Float32Array to a genome_b64 field (little-endian float32). */
function encodeGenomeB64(f32) {
    const arr = (f32 instanceof Float32Array) ? f32 : Float32Array.from(f32);
    return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
}

module.exports = { applyConfig, decodeGenomeB64, encodeGenomeB64, installBrowserStubs };
