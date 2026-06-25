/**
 * PredatorHyperparameters.js
 *
 * Tunable parameters for the predation experiment. Kept in its own module
 * (rather than folded into Hyperparameters.js) so predator-specific knobs
 * are easy to find and so this file can be the single place referenced
 * when sweeping values during hyperparameter tuning on a small map.
 *
 * Spatial parameters (count, detectionRadius) are written as comments
 * indicating how they should scale with map size — see notes inline.
 */

'use strict';

const PredatorHyperparameters = {
    // ── Population ─────────────────────────────────────────────────────────
    // Fixed for the entire run. Predators do not reproduce and the population
    // never grows or shrinks below this number — PredatorManager respawns any
    // predator that somehow dies (e.g. future extension) or is removed, so
    // count(tick=1) === count(tick=10,000,000).
    count: 15,

    // ── Energy drain ───────────────────────────────────────────────────────
    // Energy subtracted from a prey organism per drain-cell contact per tick.
    // This is the main lever for how punishing predation is. Exposed as a
    // plain mutable field (not a constant) so it can be swept programmatically
    // during hyperparameter tuning, e.g.:
    //   for (const d of [0.5, 1, 2, 4]) { PredatorHyperparameters.drainAmount = d; ... }
    drainAmount: 5.0,

    // ── Movement ───────────────────────────────────────────────────────────
    // Ticks-per-move ratio relative to prey. Prey attempt a move every tick
    // (subject to NN decision); predator_move_interval = 1 means the same
    // cadence. Use 2 to make the predator move every other tick (i.e. slower
    // than prey), values < 1 are not supported (use 1 and treat prey as the
    // slower party by other means). This ratio is the one parameter that
    // should NOT be rescaled between a small tuning map and the final map —
    // it's a speed relationship, not a spatial one.
    moveInterval: 1,

    // ── Detection ──────────────────────────────────────────────────────────
    // Omnidirectional radius (in grid cells) within which a predator detects
    // prey, scanned directly via grid_map rather than EyeCell raycasts (see
    // PredatorBrain). Prey have no sensory access to predators at all in this
    // design — detection is one-directional (predator -> prey only).
    //
    // SCALING NOTE: this should scale proportionally with map size. If tuning
    // on a map that is e.g. 1/4 the width/height of the final map, use
    // detectionRadius * (small_map_width / final_map_width) on the small map,
    // or equivalently store this as a fraction of map width and derive the
    // absolute radius at world-generation time. Left as an absolute value
    // here for simplicity; PredatorManager.spawnAll() reads it directly.
    detectionRadius: 25,

    // Radius (in grid cells) within which the predator "loses" a target it
    // can no longer detect and falls back to wandering. Kept separate from
    // detectionRadius (with hysteresis, give_up > detection) so predators
    // don't flicker between chase/wander when prey sit right at the boundary.
    giveUpRadius: 35,

    // ── Wander behaviour (no target in range) ─────────────────────────────
    // Number of ticks to commit to a wander direction before re-rolling,
    // mirroring the prey's own move_range/neutral-decision wander cadence.
    wanderRange: 6,

    // ── Cave exclusion ─────────────────────────────────────────────────────
    // Predators cannot enter cave cells (movement-level) and will not target
    // prey currently standing in a cave (targeting-level). Both are hardcoded
    // behaviour in PredatorOrganism/PredatorBrain rather than toggles here,
    // since "caves are safe from predators" is a fixed design decision, not
    // a hyperparameter to sweep.

    // ── Respawn ────────────────────────────────────────────────────────────
    // If a predator is ever removed (not currently triggered by anything in
    // this implementation, but kept for robustness / future extension, e.g.
    // if armor or future prey counter-adaptations are added), ticks to wait
    // before respawning it at a random non-cave location, to maintain the
    // exact fixed count above at all times.
    respawnDelay: 50,
};

module.exports = PredatorHyperparameters;
