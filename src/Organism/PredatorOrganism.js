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
    // Predators may walk through empty space and through prey-related
    // landscape (food/landmarks) exactly like prey can, but — unlike prey —
    // may NOT enter caves. This is half of the cave-exclusion guarantee;
    // the other half is PredatorBrain skipping caved prey as targets.

    isPassableCell(cell, parent) {
        if (cell == null) return false;
        if (cell.owner === this || cell.owner === parent) return true;
        const name = cell.state.name;
        // Predators cannot enter caves, food tiles, or landmark tiles.
        // Food/landmark exclusion prevents the predator's attemptMove() from
        // calling changeCell(..., empty) on a food tile it was standing on,
        // which would permanently erase that food from the environment.
        // Predators only need to traverse empty cells to chase prey — they
        // gain nothing from walking over food, and excluding it here means
        // their movement is naturally channelled through open corridors,
        // which is also more realistic behaviour.
        return name === CellStates.empty.name;
    }

    isClear(col, row, rotation = this.rotation) {
        for (const loccell of this.anatomy.cells) {
            const cell = this.getRealCell(loccell, col, row, rotation);
            if (cell == null) return false;
            if (cell.owner === this) continue;
            // Only empty cells are traversable — everything else (walls, caves,
            // food, landmarks, other organisms) blocks movement.
            if (cell.state.name !== CellStates.empty.name) return false;
        }
        return true;
    }

    // ── Movement / rotation overrides ───────────────────────────────────
    // Mirrors AdvancedOrganism's pattern of preserving landmarks/caves when
    // clearing old cell positions, so predators don't erase landscape
    // features as they move across them.

    attemptMove() {
        const direction = Directions.scalars[this.direction];
        const new_c = this.c + direction[0];
        const new_r = this.r + direction[1];

        if (this.isClear(new_c, new_r)) {
            for (const cell of this.anatomy.cells) {
                const real_c = this.c + cell.rotatedCol(this.rotation);
                const real_r = this.r + cell.rotatedRow(this.rotation);
                const existing = this.env.grid_map.cellAt(real_c, real_r);
                if (existing && this._isPreservedLandscape(existing.state)) continue;
                this.env.changeCell(real_c, real_r, CellStates.empty, null);
            }
            this.c = new_c;
            this.r = new_r;
            this.updateGrid();
            return true;
        }
        return false;
    }

    _isPreservedLandscape(state) {
        // Every environmental cell that must survive a predator moving over it.
        // Food cells are included here because food can respawn beneath a
        // stationary predator between ticks, and the clearance sweep in
        // attemptMove() fires on the *old* position regardless of what's there
        // now — so without food in this list, any food that appeared under the
        // predator since its last move would be silently erased.
        // Landmarks must also be preserved for the same reason (they're fixed
        // and never re-placed, so any erasure would be permanent).
        return state === CellStates.food
            || state === CellStates.lowFood
            || state === CellStates.mediumFood
            || state === CellStates.prestigeFood
            || state === CellStates.lowFoodLandmark
            || state === CellStates.mediumFoodLandmark
            || state === CellStates.prestigeFoodLandmark
            || state === CellStates.cave;
    }

    updateGrid() {
        for (const cell of this.anatomy.cells) {
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing = this.env.grid_map.cellAt(real_c, real_r);
            if (existing && this._isPreservedLandscape(existing.state)) continue;
            this.env.changeCell(real_c, real_r, cell.state, cell);
        }
    }

    // ── Death ────────────────────────────────────────────────────────────
    // Nothing in the current design kills a predator, but this is
    // implemented (rather than left as the base Organism.die(), which
    // touches this.species — predators have no species/FossilRecord entry)
    // so PredatorManager can safely respawn a predator if anything ever does
    // remove one.

    die() {
        for (const cell of this.anatomy.cells) {
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing = this.env.grid_map.cellAt(real_c, real_r);
            if (existing && this._isPreservedLandscape(existing.state)) continue;
            this.env.changeCell(real_c, real_r, CellStates.empty, null);
        }
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
