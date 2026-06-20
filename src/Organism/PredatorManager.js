/**
 * PredatorManager.js
 *
 * Maintains a FIXED number of predators for the lifetime of the simulation.
 * Deliberately separate from GAManager: predators are not part of the
 * experimental population, have no generations, no fitness, and persist
 * across prey generation boundaries (map resets, evolve() calls, etc).
 *
 * "Fixed count" guarantee: PredatorHyperparameters.count is read once at
 * spawnAll() time and enforced forever after. If a predator is ever removed
 * (nothing in the current design does this, but the respawn path exists for
 * robustness — see PredatorOrganism.die() docs), it respawns after
 * `respawnDelay` ticks at a random non-cave location, so the live count at
 * tick 1 equals the live count at tick 10,000,000.
 */

'use strict';

const PredatorOrganism = require('./PredatorOrganism');
const CellStates = require('./Cell/CellStates');
const PredatorHyperparameters = require('./PredatorHyperparameters');

class PredatorManager {
    constructor(env) {
        this.env = env;
        this.predators = [];
        // Tracks ticks-until-respawn for each predator slot that's currently
        // dead. Indexed the same way as `predators` (parallel array) so a
        // respawned predator can be slotted back into the same index.
        this._respawn_timers = [];
        this._tick_counter = 0;
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    spawnAll() {
        this.predators = [];
        this._respawn_timers = [];
        const target_count = PredatorHyperparameters.count;
        for (let i = 0; i < target_count; i++) {
            const org = this._spawnOne();
            this.predators.push(org);
            this._respawn_timers.push(0);
        }
    }

    /**
     * Call once per world tick, after prey have updated. Advances predator
     * movement (subject to moveInterval) and handles respawning.
     */
    tick() {
        this._tick_counter++;
        const should_move = (this._tick_counter % PredatorHyperparameters.moveInterval) === 0;

        for (let i = 0; i < this.predators.length; i++) {
            const org = this.predators[i];

            if (!org || !org.living) {
                // Respawn countdown for this slot.
                if (this._respawn_timers[i] > 0) {
                    this._respawn_timers[i]--;
                } else {
                    const replacement = this._spawnOne();
                    this.predators[i] = replacement;
                }
                continue;
            }

            if (should_move) {
                org.update();
            }
        }
    }

    /**
     * Called by WorldEnvironment when the world layout changes (new map /
     * new generation) so predators are repositioned onto valid, non-cave
     * terrain on the new layout while preserving the fixed count. Predators
     * are NOT reset to a "fresh" state otherwise — they have no internal
     * state to reset (no energy, no learning, no traces).
     */
    relocateAll() {
        for (let i = 0; i < this.predators.length; i++) {
            const org = this.predators[i];
            if (!org || !org.living) continue;
            const pos = this._findValidPosition(org);
            if (pos) {
                org.c = pos[0];
                org.r = pos[1];
                org.updateGrid();
            }
        }
    }

    // ── Spawning helpers ───────────────────────────────────────────────

    _spawnOne() {
        const org = PredatorOrganism.create(this.env);
        const pos = this._findValidPosition(org);
        if (pos) {
            org.c = pos[0];
            org.r = pos[1];
            org.updateGrid();
        } else {
            // Extremely unlikely on any reasonably-sized map, but avoid
            // leaving the predator un-placed — park it at grid centre with
            // grid update skipped; it will path away on its own next tick.
            const center = this.env.grid_map.getCenter();
            org.c = center[0];
            org.r = center[1];
        }
        return org;
    }

    /**
     * Find a random position that is clear for the predator AND not a cave
     * cell (predators may never spawn or be relocated into a cave).
     */
    _findValidPosition(org, attempts = 200) {
        const grid = this.env.grid_map;
        for (let i = 0; i < attempts; i++) {
            const col = Math.floor(Math.random() * grid.cols);
            const row = Math.floor(Math.random() * grid.rows);
            const cell = grid.cellAt(col, row);
            if (cell == null) continue;
            if (cell.state === CellStates.cave) continue;
            if (org.isClear(col, row, org.rotation)) {
                return [col, row];
            }
        }
        return null;
    }

    // ── Introspection ──────────────────────────────────────────────────

    livingCount() {
        let n = 0;
        for (const org of this.predators) {
            if (org && org.living) n++;
        }
        return n;
    }
}

module.exports = PredatorManager;
