/**
 * PredatorBrain.js
 *
 * Scripted policy for predators. Deliberately NOT a neural network and NOT
 * evolved/mutated — predators are a fixed environmental hazard, not part of
 * the experimental population under selection. This sidesteps co-evolution
 * entirely: the predator's behaviour on tick 1 is identical to its behaviour
 * on tick 10,000,000.
 *
 * Detection model: omnidirectional radius scan directly over grid_map,
 * representing something closer to "smell/hearing" than "sight." This is
 * deliberately decoupled from the prey's 4-direction raycast EyeCells — the
 * predator's own targeting doesn't depend on facing the right way. This is
 * independent of how PREY perceive predators: prey now have a dedicated
 * percept for "predator visible via eye raycast" (NNBrain.PERCEPT_INDEX,
 * index 11 — see CellStates.js / EyeCell.js), so avoidance can be learned
 * or evolved either from direct sighting or from the energy-loss
 * consequence of contact (PredatorDrainCell), or both.
 *
 * Policy:
 *   1. Scan a radius (detectionRadius) around the predator for the nearest
 *      living, non-caved prey organism.
 *   2. If one is found (and still within giveUpRadius once acquired), move
 *      directly toward it each tick — simple greedy pursuit, no pathing
 *      around obstacles beyond what attemptMove()/isClear() already do.
 *   3. If no prey is in range, wander: pick a random direction and commit to
 *      it for `wanderRange` ticks, same cadence pattern as the base
 *      Organism's neutral-decision wander behaviour.
 *
 * Caves are off-limits as both a movement constraint (PredatorOrganism
 * overrides isPassableCell to exclude caves) and a targeting constraint
 * (prey standing in a cave are skipped here, so they are not just
 * "blocked-but-still-chased" — they are genuinely ignored).
 */

'use strict';

const CellStates = require('../Cell/CellStates');
const Directions  = require('../Directions');
const PredatorHyperparameters = require('../PredatorHyperparameters');

class PredatorBrain {
    constructor(owner) {
        this.owner = owner;
        this.is_predator_brain = true;

        this._current_target  = null;
        this._wander_dir       = Directions.getRandomDirection();
        this._wander_ticks_left = PredatorHyperparameters.wanderRange;
    }

    /**
     * Called once per predator tick. Returns a movement direction
     * (Directions.up/right/down/left) for the predator to attempt.
     */
    decide() {
        const target = this._acquireTarget();
        if (target) {
            return this._directionToward(target);
        }
        return this._wanderDirection();
    }

    // ── Targeting ─────────────────────────────────────────────────────────

    _acquireTarget() {
        // Re-validate the currently held target first (cheaper than a full
        // rescan, and avoids target-switching jitter when multiple prey are
        // in range at similar distances).
        if (this._current_target && this._isValidTarget(this._current_target)) {
            const dist = this._distanceTo(this._current_target);
            if (dist <= PredatorHyperparameters.giveUpRadius) {
                return this._current_target;
            }
        }

        const found = this._scanForNearestPrey();
        this._current_target = found;
        return found;
    }

    _isValidTarget(org) {
        if (!org || !org.living) return false;
        if (org.is_predator) return false;
        if (this._isInCave(org)) return false;
        return true;
    }

    _isInCave(org) {
        const cell = this.owner.env.grid_map.cellAt(org.c, org.r);
        return cell != null && cell.state === CellStates.cave;
    }

    _distanceTo(org) {
        return Math.abs(this.owner.c - org.c) + Math.abs(this.owner.r - org.r);
    }

    /**
     * Omnidirectional scan within detectionRadius for the nearest valid prey.
     * Implemented as a direct grid scan rather than per-organism iteration so
     * cost scales with detectionRadius (a tunable, bounded area) rather than
     * with total population size.
     */
    _scanForNearestPrey() {
        const grid = this.owner.env.grid_map;
        const radius = PredatorHyperparameters.detectionRadius;
        const oc = this.owner.c;
        const or_ = this.owner.r;

        let best = null;
        let best_dist = Infinity;

        for (let dc = -radius; dc <= radius; dc++) {
            const remaining = radius - Math.abs(dc);
            for (let dr = -remaining; dr <= remaining; dr++) {
                if (dc === 0 && dr === 0) continue;
                const cell = grid.cellAt(oc + dc, or_ + dr);
                if (cell == null || cell.owner == null) continue;

                const candidate = cell.owner;
                if (!this._isValidTarget(candidate)) continue;

                const dist = Math.abs(dc) + Math.abs(dr);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = candidate;
                }
            }
        }
        return best;
    }

    // ── Movement direction selection ─────────────────────────────────────

    _directionToward(target) {
        const dc = target.c - this.owner.c;
        const dr = target.r - this.owner.r;
        // Greedy: close the larger axis gap first. Ties broken toward
        // horizontal movement; this is an arbitrary but deterministic choice
        // (no randomness here keeps chase behaviour fully reproducible).
        if (Math.abs(dc) >= Math.abs(dr)) {
            return dc >= 0 ? Directions.right : Directions.left;
        }
        return dr >= 0 ? Directions.down : Directions.up;
    }

    _wanderDirection() {
        if (this._wander_ticks_left <= 0) {
            this._wander_dir = Directions.getRandomDirection();
            this._wander_ticks_left = PredatorHyperparameters.wanderRange;
        }
        this._wander_ticks_left--;
        return this._wander_dir;
    }

    // ── Anatomy-tracking stubs ───────────────────────────────────────────
    // Anatomy.addDefaultCell()/removeCell() call these on whatever brain the
    // owner organism has, regardless of brain type (mirrors the pattern
    // NNBrain already uses for the same reason — see NNBrain.checkAddedCell).
    // PredatorBrain has fixed targeting logic independent of cell count, so
    // these are no-ops.
    checkAddedCell(cell) {}
    checkRemovedCell(cell) {}
    countCells() {}
}

module.exports = PredatorBrain;
