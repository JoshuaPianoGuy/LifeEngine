/**
 * PredatorOrganism.js
 *
 * A fixed, non-evolving, non-learning hazard organism. Built as an Organism
 * subclass (rather than a bespoke single-cell hack) so it reuses the
 * existing cell-based movement/collision/rendering pipeline
 * (attemptMove/isClear/updateGrid) instead of reimplementing it outside the
 * established architecture.
 *
 * Key differences from AdvancedOrganism (the prey class):
 *   - Brain is PredatorBrain (scripted chase/wander), not NNBrain. Never
 *     mutated, never subject to GA selection/crossover.
 *   - No GA registration, no species/FossilRecord tracking, no fitness.
 *   - No food-seeking, no reproduction, no energy of its own to manage.
 *   - Cannot enter or path through cave cells (isPassableCell/isClear
 *     overrides) and will not target prey standing in a cave (enforced in
 *     PredatorBrain, the second half of the cave-exclusion guarantee).
 *   - Population count is held fixed externally by PredatorManager, not by
 *     anything in this class — PredatorOrganism never reproduces or grows
 *     the population on its own.
 *
 * Anatomy (local coordinates):
 *        predator-mover (0,-1)
 *  predator-eye(-1,0)  predator-body(0,0)  predator-eye(1,0)
 *        predator-drain (0,1)
 *
 * Four cells: one mover (marks is_mover so the base Organism.update() loop
 * drives movement), two eyes (cosmetic/structural — actual detection is a
 * radius scan in PredatorBrain, not a raycast from these cells), one drain
 * cell (the only cell with a behavioural performFunction — see
 * PredatorDrainCell).
 */

'use strict';

const Organism    = require('./Organism');
const CellStates   = require('./Cell/CellStates');
const Directions   = require('./Directions');
const PredatorBrain = require('./Perception/PredatorBrain');

class PredatorOrganism extends Organism {
    constructor(col, row, env) {
        // parent=null always — predators are never spawned via inheritance.
        super(col, row, env, null);

        this.is_predator = true;
        // Roaming vs patrol distinction, exposed so prey perception can tell the
        // two hazard types apart (NNBrain promotes a patrol predator to its own
        // percept index via this flag). Defaults to roaming; PredatorManager
        // flips it to true when spawning a predator into the patrol pool.
        this.is_patrol = false;

        // Replace the base Brain with the scripted PredatorBrain. super()
        // already constructed a (decision-table) Brain via the base
        // Organism constructor and there is no parent to copy from, so it's
        // safe to just overwrite it here.
        this.brain = new PredatorBrain(this);

        // Predators have no energy/food/fitness concept of their own. These
        // fields exist only so generic code paths that might inspect
        // organism.living / organism.anatomy continue to work; nothing in
        // this class reads org.energy for the predator itself.
        this.food_collected = 0;

        this.move_count = 0;
        this.move_range = 1; // unused directly — PredatorBrain drives direction every tick

        // Per-anatomy-cell memory of the food state currently underneath each
        // predator cell (null = nothing to restore). Lets the predator draw
        // itself over food — staying visible to prey eye raycasts and blocking
        // other organisms — while restoring the food intact once it moves off.
        this._covered = [];
    }

    // ── Lifespan ──────────────────────────────────────────────────────────
    // Predators do not age out or starve. lifespan()/foodNeeded() from the
    // base Organism are overridden to neutralise the natural-death checks in
    // the inherited update() — but update() itself is fully overridden below
    // anyway, so these exist mainly for safety if any external code calls
    // them directly.

    lifespan() {
        return Infinity;
    }

    foodNeeded() {
        return Infinity; // never triggers reproduction
    }

    // ── Disabled mechanics ───────────────────────────────────────────────
    // Predators never reproduce, mutate, or take anatomy/brain damage through
    // the legacy harm() pathway. These are no-ops rather than removed
    // entirely, in case any shared code (e.g. future tooling) calls them
    // generically across all organisms in env.organisms.

    reproduce() {
        // no-op: population size is fixed and managed by PredatorManager
    }

    mutate() {
        return false; // no-op: anatomy and brain are both fixed for life
    }

    harm() {
        // no-op: predators are not killable through the legacy
        // damage/KillerCell pathway. If a "prey fights back" mechanic is
        // ever added, implement it as an explicit, separate method rather
        // than wiring it through harm()/maxHealth(), which assume the
        // anatomy-cell-count health model the predator doesn't use.
    }

    // ── Passable cell override ──────────────────────────────────────────
    // Predators traverse empty space, food, and food-region landmarks. They
    // may NOT enter caves, walls, or any cell occupied by another organism
    // (prey or predator). Food and landmark cells are passable but never
    // altered — updateGrid draws the predator over them and restores them on
    // departure (see _covered). Landmarks are deliberately passable so
    // predators can see and walk through them (they mark food regions rather
    // than acting as terrain barriers).

    isPassableCell(cell, parent) {
        if (cell == null) return false;
        if (cell.owner === this || cell.owner === parent) return true;
        return this._isTraversableTerrain(cell.state);
    }

    isClear(col, row, rotation = this.rotation) {
        for (const loccell of this.anatomy.cells) {
            const cell = this.getRealCell(loccell, col, row, rotation);
            if (cell == null) return false;
            if (cell.owner === this) continue;
            // Only empty space, food, and landmarks are traversable. Walls and
            // caves still block the predator. Food/landmark traversal is
            // required so patrol predators aren't trapped inside the prestige
            // patch they guard (every neighbour there is food or landmark).
            if (!this._isTraversableTerrain(cell.state)) return false;
            // Even on traversable terrain, the cell must not be occupied by
            // another organism — this is what stops predators overlapping each
            // other or sharing a cell with prey.
            if (cell.owner != null && cell.owner !== this) return false;
        }
        return true;
    }

    // ── Movement / rotation overrides ───────────────────────────────────
    // Draw-remember-restore model: the predator always draws its cells onto
    // the grid (so it stays visible to prey eye raycasts and occupies — and
    // therefore blocks — its cells), recording any food underneath. When it
    // moves off, the remembered food is restored, so the predator passes over
    // food without deleting it. Food is never erased by the predator; only
    // actual eating (which predators don't do) removes it.

    attemptMove() {
        const direction = Directions.scalars[this.direction];
        const new_c = this.c + direction[0];
        const new_r = this.r + direction[1];

        if (this.isClear(new_c, new_r)) {
            this._clearOwnedCells();   // restores any covered food at the old position
            this.c = new_c;
            this.r = new_r;
            this.updateGrid();         // draws at new position, records covered food
            return true;
        }
        return false;
    }

    // Terrain the predator may move onto: empty, food (all tiers), and
    // food-region landmarks (all tiers). Walls and caves are excluded.
    _isTraversableTerrain(state) {
        const name = state.name;
        return name === CellStates.empty.name
            || name === CellStates.food.name
            || name === CellStates.lowFood.name
            || name === CellStates.mediumFood.name
            || name === CellStates.prestigeFood.name
            || name === CellStates.lowFoodLandmark.name
            || name === CellStates.mediumFoodLandmark.name
            || name === CellStates.prestigeFoodLandmark.name;
    }

    // Non-empty terrain the predator must remember and restore when it draws
    // itself over a square (so passing over it doesn't erase it): food and
    // landmark cells. Empty squares need no restoration.
    _isRestorableUnder(state) {
        return state === CellStates.food
            || state === CellStates.lowFood
            || state === CellStates.mediumFood
            || state === CellStates.prestigeFood
            || state === CellStates.lowFoodLandmark
            || state === CellStates.mediumFoodLandmark
            || state === CellStates.prestigeFoodLandmark;
    }

    // Restore the predator's currently-occupied cells: put back the food that
    // was underneath (if any), otherwise clear to empty. Only touches cells the
    // predator still owns, so it never disturbs prey/landmarks/caves.
    _clearOwnedCells() {
        const cells = this.anatomy.cells;
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing = this.env.grid_map.cellAt(real_c, real_r);
            if (existing && existing.owner === this) {
                const restore = this._covered[i];
                this.env.changeCell(real_c, real_r, restore != null ? restore : CellStates.empty, null);
            }
            this._covered[i] = null;
        }
    }

    updateGrid() {
        const cells = this.anatomy.cells;
        if (this._covered.length !== cells.length) {
            this._covered = new Array(cells.length).fill(null);
        }
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing = this.env.grid_map.cellAt(real_c, real_r);
            if (existing == null) continue;
            // Record the food underneath the first time we occupy this square
            // (before we've drawn ourselves here). If we already own it, keep
            // the value recorded earlier — re-reading would see our own cell.
            if (existing.owner !== this) {
                this._covered[i] = this._isRestorableUnder(existing.state) ? existing.state : null;
            }
            // Always draw the predator — visible to the NN, blocks others.
            this.env.changeCell(real_c, real_r, cell.state, cell);
        }
    }

    // ── Death ────────────────────────────────────────────────────────────
    // Nothing in the current design kills a predator, but this is
    // implemented (rather than left as the base Organism.die(), which
    // touches this.species — predators have no species/FossilRecord entry)
    // so PredatorManager can safely respawn a predator if anything ever does
    // remove one. Restores any food the predator was covering.

    die() {
        this._clearOwnedCells();
        this.living = false;
    }

    // ── Core update loop ────────────────────────────────────────────────
    // Deliberately does not call the base Organism.update() (which contains
    // food/lifespan/reproduction/Brain.Decision logic that doesn't apply
    // here). PredatorManager calls this directly once per predator per tick
    // (subject to moveInterval, handled by the manager).

    update() {
        this.lifetime++;

        for (const cell of this.anatomy.cells) {
            cell.performFunction();
        }

        const direction = this.brain.decide();
        this.direction = direction;
        let moved = this.attemptMove();
        if (!moved) {
            // Blocked (wall, cave boundary, another organism) — try a
            // different direction this tick rather than idling, so
            // predators don't get stuck pacing against an obstacle forever.
            this.direction = Directions.getRandomDirection();
            this.attemptMove();
        }

        return this.living;
    }
}

module.exports = PredatorOrganism;

/**
 * Factory: builds a PredatorOrganism with its fixed anatomy attached.
 * Kept as a static method (mirroring the inline anatomy-construction
 * pattern in GAManager.spawnGeneration()) rather than building anatomy in
 * the constructor, so PredatorManager can construct-then-place without
 * anatomy cells briefly existing at an invalid (0,0) grid position.
 *
 * Anatomy (local coords):
 *        predator-mover (0,-1)
 *  predator-eye(-1,0)  predator-body(0,0)  predator-eye(1,0)
 *        predator-drain (0,1)
 */
PredatorOrganism.create = function(env) {
    const org = new PredatorOrganism(0, 0, env);

    org.anatomy.addDefaultCell(CellStates.predatorMover, 0, -1);
    org.anatomy.addDefaultCell(CellStates.predatorEye, -1, 0);
    org.anatomy.addDefaultCell(CellStates.predatorBody, 0, 0);
    org.anatomy.addDefaultCell(CellStates.predatorEye, 1, 0);
    org.anatomy.addDefaultCell(CellStates.predatorDrain, 0, 1);

    org.anatomy.checkTypeChange();
    return org;
};