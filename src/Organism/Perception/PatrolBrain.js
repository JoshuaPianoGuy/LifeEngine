/**
 * PatrolBrain.js
 *
 * Scripted patrol policy for prestige-food guardian predators.
 * Subclasses PredatorBrain, overriding only decide() to add a home-leash:
 *
 *   - Within patrolRadius of home: behaves exactly like PredatorBrain —
 *     scans for prey (using patrolDetectionRadius), chases greedily,
 *     wanders when no prey is in range.
 *   - Outside patrolRadius of home: drops any current target and moves
 *     directly back toward home, regardless of whether prey are visible.
 *     This is the "defender not aggressor" guarantee — prey can escape by
 *     simply leading the patrol predator away from the patch.
 *
 * Home position is set by PredatorManager when the predator is spawned or
 * relocated (on map swap). It is stored as {c, r} on the organism itself
 * (org.patrol_home) rather than in the brain, so PredatorManager can update
 * it without touching the brain directly.
 *
 * Detection and give-up radii are read from PredatorHyperparameters under
 * the 'patrol' sub-object, kept separate from roaming predator settings so
 * both can be tuned independently.
 */

'use strict';

const PredatorBrain        = require('./PredatorBrain');
const Directions           = require('../Directions');
const PredatorHyperparams  = require('../PredatorHyperparameters');

class PatrolBrain extends PredatorBrain {
    constructor(owner) {
        super(owner);
        this.is_patrol_brain = true;
    }

    /**
     * Override decide():
     *   1. If outside patrol leash → return home.
     *   2. Otherwise → standard PredatorBrain chase/wander logic,
     *      but using patrol-specific detection radius.
     */
    decide() {
        const home = this.owner.patrol_home;
        if (!home) {
            // No home set yet — fall back to wander until PredatorManager
            // assigns one (happens immediately after spawn, so this is
            // only reachable on the very first tick).
            return this._wanderDirection();
        }

        const dist_from_home = Math.abs(this.owner.c - home.c) +
                               Math.abs(this.owner.r - home.r);
        const leash = PredatorHyperparams.patrol.patrolRadius;

        if (dist_from_home > leash) {
            // Outside leash — drop target, return home.
            this._current_target = null;
            return this._directionToward(home);
        }

        // Inside leash — use standard acquire/chase/wander, but with
        // patrol-specific detection and give-up radii.
        const target = this._acquirePatrolTarget();
        if (target) {
            return this._directionToward(target);
        }
        return this._wanderDirection();
    }

    // ── Patrol-radius-aware target acquisition ────────────────────────────

    _acquirePatrolTarget() {
        // Re-validate held target using patrol give-up radius.
        if (this._current_target && this._isValidTarget(this._current_target)) {
            const dist = this._distanceTo(this._current_target);
            if (dist <= PredatorHyperparams.patrol.patrolGiveUpRadius) {
                return this._current_target;
            }
        }
        const found = this._scanPatrolArea();
        this._current_target = found;
        return found;
    }

    /**
     * Same diamond scan as PredatorBrain._scanForNearestPrey() but uses
     * patrol.patrolDetectionRadius instead of the roaming detectionRadius.
     */
    _scanPatrolArea() {
        const grid   = this.owner.env.grid_map;
        const radius = PredatorHyperparams.patrol.patrolDetectionRadius;
        const oc     = this.owner.c;
        const or_    = this.owner.r;

        let best      = null;
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
                    best      = candidate;
                }
            }
        }
        return best;
    }

    // _directionToward, _wanderDirection, _isValidTarget, _isInCave,
    // _distanceTo are all inherited unchanged from PredatorBrain.
}

module.exports = PatrolBrain;