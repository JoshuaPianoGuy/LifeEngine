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

// NOTE: `count`, `drainAmount` and `patrol.predatorsPerPatch` define which
// ENVIRONMENT (baseline vs hard) a run happens in, and are set per run — by
// src/BrowserPreset.js in the browser, by src/headless.js from its CLI flags on
// the cluster. Editing their literals below has no effect on either path; change
// BrowserPreset.js's ENVIRONMENT constant instead. Everything else here (radii,
// leash, move intervals, patch detection) is fixed design, not swept per run.
const PredatorHyperparameters = {
    // ── Population ─────────────────────────────────────────────────────────
    // Fixed for the entire run. Predators do not reproduce and the population
    // never grows or shrinks below this number — PredatorManager respawns any
    // predator that somehow dies (e.g. future extension) or is removed, so
    // count(tick=1) === count(tick=10,000,000).
    count: 60,

    // ── Spatial scaling ────────────────────────────────────────────────────
    // Every distance below (detection/leash/link radii) is an ABSOLUTE number
    // of grid cells calibrated for a reference grid this many columns wide.
    // resolveForGrid(cols) rescales them all proportionally when the actual
    // grid differs, so the predators behave identically whether the map is
    // 400 cells wide (reference) or screen-sized. Call it once after the grid
    // dimensions are known (PredatorManager.spawnAll does this).
    referenceCols: 400,

    // ── Energy drain ───────────────────────────────────────────────────────
    // Energy subtracted from a prey organism per drain-cell contact per tick.
    // This is the main lever for how punishing predation is. Exposed as a
    // plain mutable field (not a constant) so it can be swept programmatically
    // during hyperparameter tuning, e.g.:
    //   for (const d of [0.5, 1, 2, 4]) { PredatorHyperparameters.drainAmount = d; ... }
    drainAmount: 1.0,

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

    // ── Patrol predators ───────────────────────────────────────────────────
    // Patrol predators guard prestige food patches — one per patch, spawned
    // at the patch centre and leashed to it. They are defenders, not
    // aggressors: they chase prey only while within patrolRadius of home,
    // then return. Prey can escape by pulling them away from the patch.
    //
    // Kept as a sub-object so patrol settings are clearly distinct from
    // roaming predator settings and can be tuned independently.
    patrol: {
        // Number of patrol predators per prestige food patch.
        // Total patrol count = predatorsPerPatch × number of patches on the
        // current map (6–8 per map), and is recalculated on each map swap so
        // density stays constant regardless of how many patches the new map has.
        predatorsPerPatch: 2,

        // Leash radius (Manhattan distance) from home. The patrol predator
        // will not pursue prey beyond this distance from its patch centre.
        // Should be large enough to cover the patch blob (~12 cell radius)
        // plus a buffer for meaningful deterrence, but small enough that
        // prey can escape by moving away from the patch.
        patrolRadius: 20,

        // Detection radius for prey scan — independent of roaming detectionRadius.
        // Smaller than roaming predators: patrol predators only react to prey
        // that have actually entered the patch area.
        patrolDetectionRadius: 15,

        // Give-up radius: patrol drops a target it can no longer detect.
        // Kept slightly larger than detection to avoid flicker at boundary.
        patrolGiveUpRadius: 20,

        // Move interval relative to prey (same semantics as top-level moveInterval).
        // Patrol predators can be slightly slower than roaming ones to give
        // skilled/learned organisms a chance to grab food and retreat.
        moveInterval: 2,

        // Link distance (grid cells) for connected-components patch detection
        // in WorldEnvironment._extractPrestigeCentres. Two prestige cells are
        // joined into the same patch if within this distance of EACH OTHER
        // (not of a fixed seed — that older approach fragmented any patch wider
        // than its radius). Must exceed the largest internal gap within one
        // patch blob but stay well below the spacing between distinct patches.
        // The safe window is wide (~5..80 on a 400-wide grid), so this rarely
        // needs tuning; raise slightly if a single sparse patch splits into
        // multiple homes, lower if two nearby patches merge into one.
        patchLinkRadius: 12,

        // Minimum prestige-cell count for a connected component to count as a
        // real patch. Stray Gaussian-tail cells that detach from a blob would
        // otherwise become singleton "patches", each spawning its own patrol
        // home. Components smaller than this are ignored.
        minPatchCells: 8,
    },
};

// ── Grid-relative rescaling ────────────────────────────────────────────────
// Spatial radii above are calibrated for `referenceCols`. Rescale them all in
// place for the actual grid width so behaviour is resolution-independent.
// Idempotent: rescales from an immutable snapshot of the reference values, so
// calling it repeatedly (or with different widths) always yields the same
// result for a given `cols`. predatorsPerPatch / moveInterval / wanderRange /
// counts are NOT spatial and are deliberately left untouched.
const _SPATIAL_BASE = {
    detectionRadius:       PredatorHyperparameters.detectionRadius,
    giveUpRadius:          PredatorHyperparameters.giveUpRadius,
    wanderRange:           PredatorHyperparameters.wanderRange,
    patrol: {
        patrolRadius:          PredatorHyperparameters.patrol.patrolRadius,
        patrolDetectionRadius: PredatorHyperparameters.patrol.patrolDetectionRadius,
        patrolGiveUpRadius:    PredatorHyperparameters.patrol.patrolGiveUpRadius,
        patchLinkRadius:       PredatorHyperparameters.patrol.patchLinkRadius,
    },
};

PredatorHyperparameters.resolveForGrid = function (cols) {
    if (!cols || cols <= 0) return;
    const s = cols / this.referenceCols;
    const scale = (base, floor) => Math.max(floor, Math.round(base * s));
    this.detectionRadius = scale(_SPATIAL_BASE.detectionRadius, 3);
    this.giveUpRadius    = scale(_SPATIAL_BASE.giveUpRadius, 4);
    this.wanderRange     = _SPATIAL_BASE.wanderRange; // a duration, not a distance
    this.patrol.patrolRadius          = scale(_SPATIAL_BASE.patrol.patrolRadius, 4);
    this.patrol.patrolDetectionRadius = scale(_SPATIAL_BASE.patrol.patrolDetectionRadius, 3);
    this.patrol.patrolGiveUpRadius    = scale(_SPATIAL_BASE.patrol.patrolGiveUpRadius, 4);
    this.patrol.patchLinkRadius       = scale(_SPATIAL_BASE.patrol.patchLinkRadius, 4);
};

module.exports = PredatorHyperparameters;