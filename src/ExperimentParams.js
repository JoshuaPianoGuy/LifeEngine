/**
 * ExperimentParams.js
 *
 * Single mutable holder for the hyperparameters that a headless/HPC job is
 * allowed to vary per run (via CLI flags / the PBS script). Modules that own
 * these knobs (NNBrain, GAManager) read their defaults from here AT MODULE
 * LOAD, so the headless runner can override them BEFORE any simulation module
 * is required and the new values bake in correctly (including hidden_size,
 * which resizes the network genome).
 *
 * Defaults below are identical to the original in-code constants, so the
 * browser build — which never calls applyOverrides — behaves exactly as before.
 *
 * Predator knobs (drain / patrol count / roaming count) live on the already
 * mutable PredatorHyperparameters object; the headless runner copies the values
 * here onto it at runtime. They are mirrored here only so a run's full
 * parameter set is recorded in one place (see snapshot() / params.txt).
 *
 * Note: Math.random itself is NOT seeded — GA mutation, RL exploration and
 * predator wandering are deliberately left to chance (only the map terrain is
 * seeded, via WorldConfig.MAP_SEED). See [[map-seed-workflow]].
 */

'use strict';

const ExperimentParams = {
    // ── RL (NNBrain) ──────────────────────────────────────────────────────
    learning_rate: 0.02,   // RL_LR
    epsilon_start: 0.3,    // EPSILON_START
    epsilon_end:   0.05,   // EPSILON_END

    // Per-tick REINFORCE reward added the first time an organism steps onto a
    // grid cell it has not visited this lifetime (AdvancedOrganism.EXPLORE_BONUS).
    // Read at AdvancedOrganism module load, so set it before the sim modules
    // are required. Larger values push organisms to roam/explore more; 0
    // disables the exploration bonus entirely.
    // Default 0 (bonus OFF): the explore-bonus sweep found it did not help
    // fitness or meaningfully shift cave usage, so RL conditions now run without
    // it. The sweep scripts still pass explicit --explore-bonus values to probe
    // the effect; every other run inherits this 0 default.
    explore_bonus: 0,   // EXPLORE_BONUS

    // Master switch for epsilon-greedy exploration. When false, epsilon is
    // forced to 0 for the whole lifetime regardless of start/end/shape: the
    // organism acts purely on-policy (samples straight from the softmax) and
    // REINFORCE still runs with an importance ratio of exactly 1. Use this to
    // isolate whether epsilon exploration adds anything on top of REINFORCE.
    epsilon_enabled: true, // EPSILON_ENABLED

    // Shape of the epsilon decay over lifetime_frac (0 at birth -> 1 at max
    // lifespan). Applied as lifetime_frac ** exponent in NNBrain.decide():
    //   'sublinear' (√, exponent 0.5): epsilon drops fast early then flattens
    //                                  near EPSILON_END (concave, explore-early).
    //   'linear'    (exponent 1):      constant-rate decay.
    //   'quadratic' (², exponent 2):   epsilon holds near EPSILON_START then
    //                                  drops sharply late (convex, explore-late).
    //                                  This is the original in-code behaviour.
    // Unknown values fall back to 'quadratic'.
    epsilon_decay_shape: 'quadratic', // EPSILON_DECAY_SHAPE

    // ── Network (NNBrain) ─────────────────────────────────────────────────
    // Hidden-layer width. Changing this resizes the genome (W1/W2), so it is
    // read once at NNBrain load — set it before the sim modules are required.
    hidden_size:  64,     // HIDDEN_SIZE

    // ── GA (GAManager) ────────────────────────────────────────────────────
    population_size: 100,  // POPULATION_SIZE (founders per generation)
    mut_prob:        0.03, // MUT_PROB  (between-generation per-weight mutation rate)
    mut_sigma:       0.1,  // MUT_SIGMA (between-generation Gaussian std-dev)

    // ── Natural disaster (GAManager) ──────────────────────────────────────
    // Optional mass-mortality event applied BEFORE tournament selection: a
    // FIXED fraction of the generation's agents are culled from the SELECTION
    // pool, wiping their genes from the gene pool that seeds the next gen.
    //
    // Two independent random mechanisms:
    //   - WHEN it strikes: a per-generation Bernoulli draw (disaster_prob). The
    //     gaps between strikes are irregular, so disasters land on random
    //     generations (e.g. 56, 82, 250, ...) without a fixed cadence.
    //   - WHO dies: the population is shuffled and exactly disaster_fraction of
    //     it is removed (fitness-blind). The fraction itself is NOT random.
    //
    // Driven by a dedicated PRNG so both are reproducible across runs,
    // independently of the map seed and the unseeded GA/RL chance.
    disaster_enabled:   false, // master toggle (off = browser/default behaviour)
    disaster_prob:      0.1,   // per-generation probability a disaster strikes
    disaster_fraction:  0.2,   // FIXED fraction of agents removed when it strikes
    disaster_cooldown:  0,     // min generations between disasters (0 = no limit).
                               // After a strike at gen G, the next can only
                               // occur at gen >= G + cooldown — a safety window
                               // that lets population/fitness recover.
    disaster_seed:      0,     // PRNG seed (0 = unseeded, use Math.random)
    disaster_recovery_rate: 0, // GRADUAL-RECOVERY variant. 0 = classic once-off
                               // cull (default). If > 0, a strike does NOT cull a
                               // fixed slice once; instead it tapers over
                               // successive generations, shrinking the culled
                               // fraction by this many percentage points each
                               // generation until it reaches 0 and the full
                               // population is used for selection again. E.g.
                               // disaster_fraction=0.30, recovery_rate=0.05 →
                               // culls 0.30, 0.25, 0.20, ... 0.05, 0 over 6 gens.
                               // No new strike can fire while a taper is in
                               // progress; the cooldown counts from its last gen.

    // ── Predators (mirrors PredatorHyperparameters) ───────────────────────
    predator_drain:         1.0, // PredatorHyperparameters.drainAmount
    predators_per_patch:    2,   // PredatorHyperparameters.patrol.predatorsPerPatch
    roaming_predator_count: 60,   // PredatorHyperparameters.count
};

// Keys that may be overridden (everything above; methods are excluded).
const TUNABLE_KEYS = Object.keys(ExperimentParams);

/**
 * Apply a flat overrides object (e.g. parsed CLI flags). Unknown keys are
 * ignored; null/undefined values are skipped so partial overrides are fine.
 * Returns the list of keys actually changed (for logging).
 */
ExperimentParams.applyOverrides = function (obj) {
    if (!obj) return [];
    const changed = [];
    for (const key of TUNABLE_KEYS) {
        if (obj[key] != null && obj[key] !== this[key]) {
            this[key] = obj[key];
            changed.push(key);
        }
    }
    return changed;
};

/** Plain object of the current values (no methods) — for params.txt / logging. */
ExperimentParams.snapshot = function () {
    const out = {};
    for (const key of TUNABLE_KEYS) out[key] = this[key];
    return out;
};

ExperimentParams.TUNABLE_KEYS = TUNABLE_KEYS;

module.exports = ExperimentParams;
