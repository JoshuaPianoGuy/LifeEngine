'use strict';

/**
 * BrowserPreset.js — one-line experiment configuration for the BROWSER build.
 *
 * The headless runner (src/headless.js) configures a run from CLI flags before
 * any simulation module loads. The browser has no CLI, so historically you had
 * to hand-edit three separate files to reproduce one of the SLURM runs:
 *   src/WorldConfig.js               (condition + mode + map)
 *   src/ExperimentParams.js          (learning rate, epsilon)
 *   src/Organism/PredatorHyperparameters.js  (roaming count, drain, patrol)
 * — and forgetting any one of them (usually patrol.predatorsPerPatch, which
 * every production run disables) silently produced a run that did NOT match the
 * cluster results it was being compared against.
 *
 * This module replaces that with the two constants below. It mirrors exactly
 * what the run_{learning,evolution,pure_rl}_condition_{baseline,roaming}_array
 * SLURM scripts pass to src/headless.js — see EXPERIMENTS.md.
 *
 *   ── EDIT THESE TWO, REBUILD, RUN ──
 *       CONDITION   = 'learning' | 'evolution' | 'pure_rl'
 *       ENVIRONMENT = 'baseline' | 'hard'
 *
 * Then:
 *       node generate_map_pool.js --seed <MAP_SEED>   # only when MAP_SEED changes
 *       npm run build && npm run serve                # http://localhost:3000
 *
 * ── LOAD ORDER (important) ────────────────────────────────────────────────────
 * NNBrain, AdvancedOrganism and GAManager read their constants from
 * ExperimentParams AT MODULE LOAD, so this file must be evaluated BEFORE they
 * are. That is why src/index.js imports it on the line above `import Engine`:
 * ES import declarations are hoisted but evaluate in source order, so this
 * module (which requires only the three leaf config modules — no sim code) runs
 * first and the values are baked in correctly.
 *
 * This file is imported ONLY by src/index.js (the webpack entry point), so the
 * headless runner, the eval/probe harness and the landscape tooling are
 * completely unaffected — they keep configuring themselves from their own flags.
 */

const WorldConfig             = require('./WorldConfig');
const ExperimentParams        = require('./ExperimentParams');
const PredatorHyperparameters = require('./Organism/PredatorHyperparameters');

// ─────────────────────────────────────────────────────────────────────────────
// EDIT HERE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Which of the three comparison conditions to run.
 *   'learning'  — GA + within-life REINFORCE   (isolates nothing on its own)
 *   'evolution' — GA only, no within-life learning
 *   'pure_rl'   — within-life REINFORCE only, no GA (PureRLManager)
 * learning vs pure_rl isolates the GA's contribution; learning vs evolution
 * isolates within-life learning.
 */
const CONDITION = 'learning';

/**
 * Which environment to run the condition in.
 *   'baseline' — predator-free
 *   'hard'     — 60 roaming predators, drain 5 (patrol off in both)
 * The learning rate is selected per environment from that environment's own LR
 * sweep, so changing this also changes the LR (see ENVIRONMENTS below).
 */
const ENVIRONMENT = 'hard';

/**
 * Terrain seed. One of the five comparison seeds: 1, 42, 999, 123, 456.
 * The ACTIVE bundled pool (src/maps/map_pool.json) must be generated for this
 * seed or WorldEnvironment logs a mismatch warning at startup:
 *     node generate_map_pool.js --seed <MAP_SEED>
 * Seeds 1 / 42 / 999 are already cached in src/maps/ and regenerate instantly.
 */
const MAP_SEED = 999;

/**
 * Grid dimensions, in cells. 500x500 matches every comparison / sweep run on
 * the cluster. At cell_size 2 that is a 1000x1000 px canvas — heavy but usable.
 * Set both to null to fall back to the canvas-derived (browser window) size,
 * which is fine for eyeballing behaviour but NOT comparable to cluster results.
 */
const GRID_COLS = 500;
const GRID_ROWS = 500;

/**
 * RESIZING THE WORLD — read before changing GRID_COLS / GRID_ROWS.
 *
 * Nothing breaks structurally at any size (tested 40x40 to 1000x1000: no
 * errors), but a resized run is NOT comparable to the reported results, because
 * two things change at once and only one of them is automatic:
 *
 *   SCALES automatically — predator detection / give-up / patrol radii, via
 *     PredatorHyperparameters.resolveForGrid(cols), which rescales from the
 *     400-column reference. detectionRadius is 31 at 500 wide, 63 at 1000.
 *
 *   DOES NOT scale — food. A map pool places a FIXED number of food cells
 *     (~2,533 for seed 42, identical at every width), so coverage falls as the
 *     world grows: 17.6% at 120x120, 1.0% at 500x500, 0.25% at 1000x1000.
 *     Pass the AREA RATIO to ExperimentParams.food_density_scale to hold
 *     density constant (500->1000 is 4), exactly as --food-density-scale does
 *     headless.
 *
 *   DOES NOT scale — predator COUNT. It is an absolute number, so predator
 *     density moves inversely with area: 60 predators is ~240 per million
 *     cells at 500x500 but ~60 at 1000x1000.
 *
 *   DOES NOT scale — GAManager.SPAWN_RADIUS (a hardcoded 30 cells), so the
 *     founding cohort occupies a quarter of a 120-wide world but 3% of a
 *     1000-wide one.
 *
 * Prestige-patch COUNT also shifts with size for the same map (5 patches at
 * 500x500 vs 11 at 1000x1000 on seed 42, map 1). Harmless while patrol
 * predators are off, but it would multiply the patrol pool if they were on.
 * See EXPERIMENTS.md 5.2b.
 */

/**
 * Network width. The reported runs all pass --hidden-size 128; the
 * ExperimentParams default is 64, which is a DIFFERENT network (genome 3910
 * weights instead of 7814) and not comparable to the published results. Set
 * here so the browser matches the experiments rather than the file default.
 * Lower it to 64 if you want a faster browser run and do not need comparability.
 */
const HIDDEN_SIZE = 128;

/**
 * Pure-RL respawn buffer. Only used by --mode pure_rl, where it is the sole
 * mechanism carrying strategies across a generation boundary. The reported runs
 * use 100 (chosen so the founder count matches the 100 founders of the other
 * two conditions); the ExperimentParams default is 25.
 */
const COLLAPSE_BUFFER_SIZE = 100;

/**
 * Per-tick reward for stepping on an unvisited cell. The explore-bonus sweep
 * found it did not help, so every reported run passes --explore-bonus 0. This
 * already matches the ExperimentParams default; set explicitly so the preset
 * stays correct if that default ever moves.
 */
const EXPLORE_BONUS = 0;

/**
 * Runtime food cells placed per reference food cell. 1 = the historical
 * behaviour and what every 500x500 run used. Raise it to the area ratio if you
 * enlarge the world — see the resizing note above.
 */
const FOOD_DENSITY_SCALE = 1;

// ─────────────────────────────────────────────────────────────────────────────
// PRESET TABLES — these encode the SLURM scripts; you should not need to edit
// them unless the production run configuration itself changes.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CONDITION -> the WorldConfig switches WorldEnvironment reads to pick a manager.
 * `label` / `detail` are display-only, for the About tab's run-config readout.
 */
const CONDITIONS = {
    // --condition learning --mode standard
    learning:  {
        learning_enabled: true,  experiment_mode: 'standard',
        label: 'Learning', detail: 'EA + within-life REINFORCE',
    },
    // --condition evolution --mode standard  (no RL knobs passed at all)
    evolution: {
        learning_enabled: false, experiment_mode: 'standard',
        label: 'Evolution', detail: 'EA only, no within-life learning',
    },
    // --condition learning --mode pure_rl
    pure_rl:   {
        learning_enabled: true,  experiment_mode: 'pure_rl',
        label: 'Pure RL', detail: 'within-life REINFORCE only, no EA',
    },
};

/** ENVIRONMENT -> learning rate + the predator settings that define it. */
const ENVIRONMENTS = {
    // Predator-free. --predators-per-patch 0 --roaming-predators 0, no
    // --predator-drain flag (so drain keeps the headless default of 1.0; it is
    // inert here anyway with zero predators).
    baseline: {
        learning_rate:          0.01,   // baseline-environment LR-sweep optimum
        roaming_predator_count: 0,
        predator_drain:         1.0,
        predators_per_patch:    0,
        label: 'Baseline', detail: 'predator-free',
    },
    // Roaming-predator environment: --roaming-predators 60 --predator-drain 5
    // --predators-per-patch 0.
    hard: {
        learning_rate:          0.02,   // roaming-environment LR-sweep optimum
        roaming_predator_count: 60,
        predator_drain:         5.0,
        predators_per_patch:    0,
        label: 'Hard', detail: '60 roaming predators, drain 5, patrol off',
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// APPLY
// ─────────────────────────────────────────────────────────────────────────────

const cond = CONDITIONS[CONDITION];
if (!cond) {
    throw new Error(
        `BrowserPreset: unknown CONDITION '${CONDITION}'. ` +
        `Expected one of: ${Object.keys(CONDITIONS).join(', ')}.`
    );
}
const env = ENVIRONMENTS[ENVIRONMENT];
if (!env) {
    throw new Error(
        `BrowserPreset: unknown ENVIRONMENT '${ENVIRONMENT}'. ` +
        `Expected one of: ${Object.keys(ENVIRONMENTS).join(', ')}.`
    );
}

// ── World / condition ────────────────────────────────────────────────────────
WorldConfig.learning_enabled = cond.learning_enabled;
WorldConfig.experiment_mode  = cond.experiment_mode;
WorldConfig.MAP_SEED         = MAP_SEED;
WorldConfig.MAP_COLS         = GRID_COLS;
WorldConfig.MAP_ROWS         = GRID_ROWS;

// ── RL hyperparameters (read at NNBrain module load — hence the load order) ──
// The learning rate is inert in the 'evolution' condition (no RL runs at all),
// but is still set so a params snapshot reports the environment's chosen value.
ExperimentParams.learning_rate = env.learning_rate;
// All production runs are --no-epsilon: pure on-policy REINFORCE, with the
// importance ratio identically 1. epsilon_start/end/decay_shape are ignored
// entirely when this is false, so they are deliberately left alone.
ExperimentParams.epsilon_enabled = false;

// Network / buffer / reward knobs the SLURM scripts pass explicitly. Without
// these four assignments the browser silently ran on the ExperimentParams
// defaults (hidden_size 64, collapse_buffer_size 25), which do NOT match any
// reported run — see the constant definitions above.
ExperimentParams.hidden_size          = HIDDEN_SIZE;
ExperimentParams.collapse_buffer_size = COLLAPSE_BUFFER_SIZE;
ExperimentParams.explore_bonus        = EXPLORE_BONUS;
ExperimentParams.food_density_scale   = FOOD_DENSITY_SCALE;

// ── Predators ────────────────────────────────────────────────────────────────
// PredatorHyperparameters is the ONLY source of truth the browser reads at
// runtime (PredatorManager.spawnAll). The matching ExperimentParams fields are
// mirrors kept in sync purely so a params snapshot records the real values —
// src/headless.js copies them across itself, but nothing does so in the browser.
PredatorHyperparameters.count                    = env.roaming_predator_count;
PredatorHyperparameters.drainAmount              = env.predator_drain;
PredatorHyperparameters.patrol.predatorsPerPatch = env.predators_per_patch;

ExperimentParams.roaming_predator_count = env.roaming_predator_count;
ExperimentParams.predator_drain         = env.predator_drain;
ExperimentParams.predators_per_patch    = env.predators_per_patch;

// ── Publish a read-only descriptor for the UI ────────────────────────────────
// ControlPanel renders this in the About tab so the running build's full
// configuration is visible without opening devtools. Published onto WorldConfig
// (rather than having ControlPanel require this module) because requiring
// BrowserPreset re-applies nothing but IS a side-effecting import — parking the
// data on a config object everyone already reads keeps the dependency one-way.
// Absent (undefined) under headless, where the panel never renders anyway.
WorldConfig.browser_preset = {
    condition:        CONDITION,
    condition_label:  cond.label,
    condition_detail: cond.detail,
    environment:        ENVIRONMENT,
    environment_label:  env.label,
    environment_detail: env.detail,
    map_seed:      MAP_SEED,
    grid_cols:     GRID_COLS,
    grid_rows:     GRID_ROWS,
    learning_rate: env.learning_rate,
    epsilon_enabled: false,

};

// ── Startup banner ───────────────────────────────────────────────────────────
// The browser writes no params.txt (unlike headless), so this console line is
// the only record of what the running build is actually configured as. Check it
// in devtools before trusting a browser run against cluster results.
console.log(
    `[BrowserPreset] condition=${CONDITION} (learning_enabled=${cond.learning_enabled}, ` +
    `mode=${cond.experiment_mode})  environment=${ENVIRONMENT}  ` +
    `lr=${ExperimentParams.learning_rate}  epsilon=off  ` +
    `predators: roaming=${PredatorHyperparameters.count} ` +
    `drain=${PredatorHyperparameters.drainAmount} ` +
    `patrol=${PredatorHyperparameters.patrol.predatorsPerPatch}  ` +
    `map_seed=${MAP_SEED}  grid=${GRID_COLS}x${GRID_ROWS}`
);

module.exports = { CONDITION, ENVIRONMENT, MAP_SEED, GRID_COLS, GRID_ROWS };
