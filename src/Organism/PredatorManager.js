/**
 * PredatorManager.js
 *
 * Maintains a FIXED roaming predator pool AND a variable-size patrol pool
 * whose size scales with the current map's prestige patch count.
 *
 * Roaming predators: chase prey anywhere on the map (existing behaviour).
 *   Fixed count = PredatorHyperparams.count, constant for the whole run.
 *
 * Patrol predators: guard prestige food patches.
 *   Count = predatorsPerPatch × patches_on_current_map.
 *   Recalculated on every map swap so density stays constant regardless
 *   of how many patches the new map has. The entire patrol pool is
 *   despawned and rebuilt on each map change rather than resized in-place —
 *   simpler and avoids stale home assignments from the previous map.
 */

'use strict';

const PredatorOrganism    = require('./PredatorOrganism');
const PatrolBrain         = require('./Perception/PatrolBrain');
const CellStates          = require('./Cell/CellStates');
const PredatorHyperparams = require('./PredatorHyperparameters');

class PredatorManager {
    constructor(env) {
        this.env = env;
        this.predators       = [];   // roaming pool
        this._respawn_timers = [];

        this.patrol_predators       = [];  // patrol pool — variable size
        this._patrol_home_slots     = [];  // parallel: home {c,r} for each slot

        this._tick_counter = 0;
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    spawnAll() {
        this.predators       = [];
        this._respawn_timers = [];

        // Rescale all predator distances (detection/leash/patch-link radii) to
        // the actual grid width so behaviour is resolution-independent. Done
        // here because spawnAll runs once per reset, after the grid exists.
        if (typeof PredatorHyperparams.resolveForGrid === 'function') {
            PredatorHyperparams.resolveForGrid(this.env.grid_map.cols);
        }

        const roaming_count = PredatorHyperparams.count;
        for (let i = 0; i < roaming_count; i++) {
            this.predators.push(this._spawnOne());
            this._respawn_timers.push(0);
        }

        this.patrol_predators   = [];
        this._patrol_home_slots = [];
        this._rebuildPatrolPool();
    }

    tick() {
        this._tick_counter++;

        // Roaming
        const roaming_move = (this._tick_counter % PredatorHyperparams.moveInterval) === 0;
        for (let i = 0; i < this.predators.length; i++) {
            const org = this.predators[i];
            if (!org || !org.living) {
                if (this._respawn_timers[i] > 0) {
                    this._respawn_timers[i]--;
                } else {
                    this.predators[i] = this._spawnOne();
                }
                continue;
            }
            if (roaming_move) org.update();
        }

        // Patrol — pool may vary in size between maps; respawn uses stored home
        const patrol_move = (this._tick_counter % PredatorHyperparams.patrol.moveInterval) === 0;
        for (let i = 0; i < this.patrol_predators.length; i++) {
            const org = this.patrol_predators[i];
            if (!org || !org.living) {
                // Respawn at the same home slot
                this.patrol_predators[i] = this._spawnPatrolOne(this._patrol_home_slots[i]);
                continue;
            }
            if (patrol_move) org.update();
        }
    }

    /**
     * Called by WorldEnvironment on every map swap. Roaming predators are
     * repositioned randomly. The patrol pool is fully rebuilt for the new
     * map so each patch gets exactly predatorsPerPatch guards.
     */
    relocateAll() {
        // Roaming — reposition randomly on new map
        for (let i = 0; i < this.predators.length; i++) {
            const org = this.predators[i];
            if (!org || !org.living) continue;
            const pos = this._findValidPosition(org);
            if (pos) { org.c = pos[0]; org.r = pos[1]; org.updateGrid(); }
        }

        // Patrol — kill old pool, rebuild fresh from new map's prestige centres.
        // Killing first prevents ghost organisms that still update but have
        // stale home positions from the previous map.
        for (const org of this.patrol_predators) {
            if (org && org.living) org.die();
        }
        this.patrol_predators   = [];
        this._patrol_home_slots = [];
        this._rebuildPatrolPool();
    }

    // ── Patrol pool management ─────────────────────────────────────────

    /**
     * Build the patrol pool from scratch using the current map's prestige
     * patch centres. Spawns predatorsPerPatch organisms per centre.
     * Called at spawnAll() and after every map swap via relocateAll().
     */
    _rebuildPatrolPool() {
        if (!this.env.prestige_patch_centres || this.env.prestige_patch_centres.length === 0) {
            this.env.prestige_patch_centres = this.env._extractPrestigeCentres
                ? this.env._extractPrestigeCentres()
                : [];
        }
        const centres      = this.env.prestige_patch_centres;
        const perPatch     = PredatorHyperparams.patrol.predatorsPerPatch;

        for (const centre of centres) {
            for (let j = 0; j < perPatch; j++) {
                this._patrol_home_slots.push(centre);
                this.patrol_predators.push(this._spawnPatrolOne(centre));
            }
        }

        // Diagnostic: confirms the pool matches expected density. If
        // total != patches * perPatch, the centre extraction is over- or
        // under-counting patches (check MERGE_RADIUS in _extractPrestigeCentres).
        console.log(`[PredatorManager] patrol pool rebuilt: ` +
            `${centres.length} patches x ${perPatch} = ${this.patrol_predators.length} predators`);
    }

    _spawnPatrolOne(home) {
        const org      = PredatorOrganism.create(this.env);
        org.brain      = new PatrolBrain(org);
        org.patrol_home = home || null;

        const pos = home
            ? (this._findValidPositionNear(org, home.c, home.r, 10) || this._findValidPosition(org))
            : this._findValidPosition(org);

        if (pos) {
            org.c = pos[0];
            org.r = pos[1];
            org.updateGrid();
        } else {
            const center = this.env.grid_map.getCenter();
            org.c = center[0];
            org.r = center[1];
        }
        return org;
    }

    // ── Roaming spawn helpers ──────────────────────────────────────────

    _spawnOne() {
        const org = PredatorOrganism.create(this.env);
        const pos = this._findValidPosition(org);
        if (pos) {
            org.c = pos[0];
            org.r = pos[1];
            org.updateGrid();
        } else {
            const center = this.env.grid_map.getCenter();
            org.c = center[0];
            org.r = center[1];
        }
        return org;
    }

    _findValidPosition(org, attempts = 200) {
        const grid = this.env.grid_map;
        for (let i = 0; i < attempts; i++) {
            const col  = Math.floor(Math.random() * grid.cols);
            const row  = Math.floor(Math.random() * grid.rows);
            const cell = grid.cellAt(col, row);
            if (cell == null) continue;
            if (cell.state === CellStates.cave) continue;
            if (org.isClear(col, row, org.rotation)) return [col, row];
        }
        return null;
    }

    _findValidPositionNear(org, home_c, home_r, radius, attempts = 100) {
        const grid = this.env.grid_map;
        for (let i = 0; i < attempts; i++) {
            const dc  = Math.floor((Math.random() * 2 - 1) * radius);
            const dr  = Math.floor((Math.random() * 2 - 1) * radius);
            const col = home_c + dc;
            const row = home_r + dr;
            if (col < 0 || row < 0 || col >= grid.cols || row >= grid.rows) continue;
            const cell = grid.cellAt(col, row);
            if (cell == null) continue;
            if (cell.state === CellStates.cave) continue;
            if (org.isClear(col, row, org.rotation)) return [col, row];
        }
        return null;
    }

    // ── Introspection ──────────────────────────────────────────────────

    livingCount() {
        let n = 0;
        for (const org of this.predators)        { if (org && org.living) n++; }
        for (const org of this.patrol_predators) { if (org && org.living) n++; }
        return n;
    }
}

module.exports = PredatorManager;