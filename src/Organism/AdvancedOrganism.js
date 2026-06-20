/**
 * AdvancedOrganism.js
 *
 * Subclass of Organism for the learning-vs-evolution experiment.
 *
 * Key differences from base Organism:
 *
 * 1. Brain is NNBrain (neural network policy) instead of Brain (lookup table).
 *
 * 2. Energy system replaces the food-for-reproduction mechanic as the survival
 *    signal. Energy decays over time; eating food restores it. Reaching 0
 *    energy kills the agent. This drives the reward function for RL.
 *
 * 3. Lifespan is capped at MAX_LIFETIME (1500 ticks). An agent dies at
 *    whichever comes first: energy hits 0, or 1500 ticks elapse.
 *
 * 4. Reproduction is asexual genome inheritance (Model C):
 *    - Triggered by the standard LifeEngine mechanic: eat food equal to body
 *      size (food_collected >= anatomy.cells.length).
 *    - Child genome = parent genome_weights + Gaussian mutation.
 *    - anatomy mutation (add/change/remove cell) is DISABLED — anatomy is
 *      fixed so the NN input/output dimensions never change.
 *    - Child is registered with GAManager so the generation tracker knows
 *      about it.
 *    - This is non-Lamarckian: the child inherits genome_weights only, not
 *      active_weights (RL drift is discarded).
 *
 * 5. Between-generation crossover is handled by GAManager.evolve() and is
 *    completely separate from this per-organism reproduction.
 *
 * 6. Fitness = cumulative food score across the lifetime (sum of energy values
 *    of all food eaten). Used by GAManager for top-5 selection.
 *
 * 7. Die() does NOT drop food (per experiment spec). Cells become empty.
 *
 * Reward function (drives RL in Condition A):
 *   +food energy value    when food is eaten
 *   -DECAY_PENALTY        each time energy ticks down
 *   +EXPLORE_BONUS        for each newly visited grid cell
 */

'use strict';

const Organism   = require('./Organism');
const NNBrain    = require('./Perception/NNBrain');
const Directions = require('./Directions');
const CellStates = require('./Cell/CellStates');
const Hyperparams = require('../Hyperparameters');
const FossilRecord = require('../Stats/FossilRecord');

// ── Energy constants ──────────────────────────────────────────────────────────

const MAX_LIFETIME      = 1000000;  // increased from 1500 to allow more time for learning

// Generation length in ticks: TICKS_PER_MAP * MAPS_PER_GEN from GAManager.js.
// No organism can outlive a generation, so this is the true ceiling for
// within-lifetime epsilon decay. Must be kept in sync with GAManager's
// TICKS_PER_MAP and MAPS_PER_GEN constants — if those change, update this.
const TICKS_PER_GEN = 10000; // 2000 ticks/map * 5 maps/gen
const ENERGY_CAPACITY   = 500;   // maximum energy set to 500
const START_ENERGY      = 300;   // increased from 100 to give organisms buffer for reproduction
const ENERGY_DECAY_RATE = 1;     // energy lost per decay event
const ENERGY_DECAY_INTERVAL = 10; // ticks between decay events (per spec: lose energy every 10 ticks)

// ── Day/night energy mechanics ─────────────────────────────────────────────────
//
// Day (300 ticks):
//   - Outside cave: normal decay (1 per 10 ticks) — organisms search for food
//   - Inside cave: accelerated decay (2-3x rate) — not a refuge during day
//
// Night (300 ticks):
//   - Inside cave: energy frozen — safe sleeping spot
//   - Outside cave: fast energy loss — incentive to find cave before night
//
// These multipliers are applied to the base ENERGY_DECAY_RATE each decay event.

const DAY_CAVE_DECAY_MULTIPLIER     = 2;   // 2x faster loss in caves during day
const NIGHT_OUTSIDE_DECAY_MULTIPLIER = 2;  // 2x faster loss outside caves at night

// Food energy values: single source of truth is MouthCell.FOOD_ENERGY_VALUES.
// Importing from there avoids duplicating the table.
// MouthCell sets org.last_eaten_state before each food_collected increment
// so _foodValue() knows which food type was just eaten.
const MouthCell = require('./Cell/BodyCells/MouthCell');
const FOOD_ENERGY = MouthCell.FOOD_ENERGY_VALUES;
const DEFAULT_FOOD_ENERGY = FOOD_ENERGY.food || 0.01;

// ── Reward constants ──────────────────────────────────────────────────────────

const DECAY_PENALTY  = 0.05;   // negative reward each time energy decays
//try 0.01--0.1?
const EXPLORE_BONUS  = 0.15;  // positive reward for visiting a new cell
//try 0.1--0.3? initially 0.02, but organisms had less incentive to explore after eating high tier food

// ── Debugging ───────────────────────────────────────────────────────────────
const DEBUG_ACTIONS = false;  // Set to true to log NN actions and movement outcomes

// ── GA mutation constants (within-generation asexual reproduction) ────────────
// Applied when an agent reproduces mid-generation.
// Enabled: Gaussian mutation allows natural exploration and prevents local maxima

const ASEXUAL_MUT_PROB  = 0.05;  // 5% per-weight mutation probability
const ASEXUAL_MUT_SIGMA = 0.1;   // Gaussian noise std-dev

// ── Reproduction control ──────────────────────────────────────────────────────
const REPRODUCTION_SUCCESS_RATE = 0.8;  // 80% chance a child successfully spawns

// ─────────────────────────────────────────────────────────────────────────────

class AdvancedOrganism extends Organism {
    /**
     * @param {number}   col
     * @param {number}   row
     * @param {object}   env
     * @param {AdvancedOrganism|null} parent   null for founding generation agents
     * @param {boolean}  rl_enabled            Condition A = true, B = false
     * @param {object|null} ga_manager         GAManager reference for child registration
     */
    constructor(col, row, env, parent = null, rl_enabled = true, ga_manager = null) {
        super(col, row, env, parent);

        this.rl_enabled = rl_enabled;
        this.ga_manager = ga_manager;

        // Replace Brain with NNBrain.
        // super() already called inherit() which called brain.copy(parent.brain)
        // using the old Brain — we replace it now and re-copy if parent is NNBrain.
        this.brain = new NNBrain(this, rl_enabled);
        if (parent !== null && parent.brain instanceof NNBrain) {
            this.brain.copy(parent.brain);
        }

        // Energy
        this.energy     = START_ENERGY;
        this.max_energy = ENERGY_CAPACITY;

        // Reward accumulator — cleared each tick after RL update
        this.pending_reward = 0;

        // Fitness — cumulative food score (used by GAManager for selection)
        this.cumulative_food_score = 0;

        // Exploration — visited cells for exploration bonus
        this.visited_cells = new Set();

        // Metrics: energy snapshots at 20% of MAX_LIFETIME (per spec)
        this.energy_at_early_sample = null;
        this._early_sample_tick     = Math.floor(MAX_LIFETIME * 0.20);  // tick 300

        // Per-food-type counts for Logger detail rows
        this.food_by_type = {};

        // Energy-based reproduction: child spawns when parent gains 10 energy
        this.energyGainedSinceReproduction = 0;
    }

    // ── Lifespan ──────────────────────────────────────────────────────────────

    lifespan() {
        return MAX_LIFETIME;
    }

    // ── Reproduction Threshold ─────────────────────────────────────────────────
    // [LEGACY] Left for base Organism compatibility, but not used functionally.
    // Reproduction is now triggered by energy gain (10 points), not food count.
    foodNeeded() {
        return 4;  // unused; reproduction based on energy gain instead
    }

    // ── Core update loop ──────────────────────────────────────────────────────

    update() {
        this.lifetime++;

        // Hard lifespan cap
        if (this.lifetime > MAX_LIFETIME) {
            this.die();
            return false;
        }

        // Early-lifetime energy snapshot (20% mark, used as metric)
        if (this.lifetime === this._early_sample_tick) {
            this.energy_at_early_sample = this.energy;
        }

        // ── Energy decay with day/night and cave mechanics ────────────────────
        // Day: normal outside, 2x faster in caves
        // Night: frozen in caves, 2x faster outside
        if (this.lifetime % ENERGY_DECAY_INTERVAL === 0) {
            const is_night = this.env.isNight();
            const in_cave = this._isInCave();

            let decay = ENERGY_DECAY_RATE;

            if (is_night) {
                // Night time: only lose energy if outside
                if (!in_cave) {
                    decay *= NIGHT_OUTSIDE_DECAY_MULTIPLIER;
                } else {
                    decay = 0;  // Energy frozen in caves at night
                }
            } else {
                // Day time: lose energy faster in caves
                if (in_cave) {
                    decay *= DAY_CAVE_DECAY_MULTIPLIER;
                }
                // else: normal decay outside
            }

            this.energy -= decay;
            this.pending_reward -= DECAY_PENALTY;
        }

        if (this.energy <= 0) {
            this.die();
            return false;
        }

        // Check if enough energy gained to reproduce (energy-based trigger)
        // This happens before cell functions so the parent isn't mid-move when it
        // spawns a child. Decouples reproduction from specific food items.
        if (this.energyGainedSinceReproduction >= 3.5) {
            this.reproduce();
        }

        // Run cell functions: mouth eats food, eye cells observe
        const food_before = this.food_collected;
        for (const cell of this.anatomy.cells) {
            cell.performFunction();
            if (!this.living) return false;
        }

        // Convert any newly eaten food to energy + reward
        if (this.food_collected > food_before) {
            this._processFoodEaten(this.food_collected - food_before);
        }

        // NNBrain movement decision + RL update
        if (this.anatomy.is_mover) {
            this._nnMove();
        }

        return this.living;
    }

    // ── Food → energy conversion ──────────────────────────────────────────────

    _processFoodEaten(units) {
        for (let i = 0; i < units; i++) {
            const value  = this._foodValue();
            const gained = Math.min(value, this.max_energy - this.energy);
            this.energy                     += gained;
            this.cumulative_food_score      += value;
            this.pending_reward             += value;
            this.energyGainedSinceReproduction += gained;  // track for reproduction trigger

            // Track per-food-type counts for Logger
            const type = this.last_eaten_state || 'food';
            this.food_by_type[type] = (this.food_by_type[type] || 0) + 1;
        }
    }

    _foodValue() {
        // Returns the energy value of the last eaten food cell.
        // Once you track which cell state was eaten (e.g. via last_eaten_state
        // set by MouthCell before incrementing food_collected), look it up here.
        if (this.last_eaten_state && FOOD_ENERGY[this.last_eaten_state]) {
            return FOOD_ENERGY[this.last_eaten_state];
        }
        return DEFAULT_FOOD_ENERGY;
    }

    // ── NNBrain movement ──────────────────────────────────────────────────────

    _nnMove() {
        // lifetime_frac runs 0 → 1 over the generation's tick length, not over
        // MAX_LIFETIME (1,000,000) which no organism can ever reach — the
        // generation ends at TICKS_PER_GEN (10,000) and all organisms are
        // replaced. Using MAX_LIFETIME as the denominator meant lifetime_frac
        // never exceeded 0.01, so epsilon barely moved from EPSILON_START and
        // the quadratic decay was effectively disabled. Using TICKS_PER_GEN
        // means epsilon correctly decays from EPSILON_START to EPSILON_END
        // over each organism's actual lifespan within the generation.
        const lifetime_frac = Math.min(1, this.lifetime / TICKS_PER_GEN);
        const reward        = this.pending_reward;
        this.pending_reward = 0;

        // Exploration bonus for visiting new cells
        const cell_key = `${this.c},${this.r}`;
        if (!this.visited_cells.has(cell_key)) {
            this.visited_cells.add(cell_key);
            // Add to next tick's reward (after the RL update this tick)
            this.pending_reward += EXPLORE_BONUS;
        }

        const action = this.brain.decide(reward, this.max_energy, lifetime_frac);
        let moved = false;
        let rotated = false;
        
        // Actions 0-3: movement (up, right, down, left)
        // Actions 4-5: rotation (rotate-left, rotate-right)
        if (action === 4) {
            // Rotate left
            rotated = this.attemptRotate(Directions.getLeftDirection(this.rotation));
        } else if (action === 5) {
            // Rotate right
            rotated = this.attemptRotate(Directions.getRightDirection(this.rotation));
        } else {
            // Movement action
            this.direction = action;
            moved = this.attemptMove();
            if (!moved) {
                this.direction = Directions.getRandomDirection();
                moved = this.attemptMove();
            }
        }

        if (DEBUG_ACTIONS) {
            const actionName = action <= 3 ? ['UP', 'RIGHT', 'DOWN', 'LEFT'][action] : (action === 4 ? 'ROTATE_LEFT' : 'ROTATE_RIGHT');
            console.log(`[Org #${this.id} tick ${this.lifetime}] Action:${actionName} moved:${moved} rotated:${rotated} pos:(${this.c},${this.r}) energy:${(this.energy || 0).toFixed(2)}`);
            if (action <= 3 && !moved) {
                const dir = Directions.scalars[this.direction];
                const block_c = this.c + dir[0];
                const block_r = this.r + dir[1];
                const block_cell = this.env.grid_map.cellAt(block_c, block_r);
                const block_state = block_cell && block_cell.state ? block_cell.state.name : 'out-of-bounds';
                console.log(`[Org #${this.id} tick ${this.lifetime}] Move blocked by ${block_state} at (${block_c},${block_r})`);
            }
            if (action >= 4 && !rotated) {
                console.log(`[Org #${this.id} tick ${this.lifetime}] Rotation blocked at pos:(${this.c},${this.r})`);
            }
        }
    }

    // ── Asexual reproduction (within-generation) ──────────────────────────────
    //
    // Overrides base Organism.reproduce().
    // Child genome = parent genome_weights + Gaussian mutation.
    // Anatomy mutation is disabled — anatomy is fixed so NN dimensions are stable.
    // Child is registered with GAManager so the generation tracker sees it.
    // Non-Lamarckian: genome_weights only, never active_weights.

    reproduce() {
        // Place the child using the same placement logic as base Organism
        const child = new AdvancedOrganism(
            0, 0,           // position set below after placement check
            this.env,
            this,           // parent — NNBrain.copy() called in constructor
            this.rl_enabled,
            this.ga_manager
        );

        if (Hyperparams.rotationEnabled) {
            child.rotation = Directions.getRandomDirection();
        }

        // Apply genome mutation to child (asexual: parent genome + noise)
        // This is the only mutation that happens within a generation.
        const child_genome = new Float32Array(this.brain.getGenome());
        this._mutateGenome(child_genome);
        child.brain.setGenome(child_genome);

        // Spawn at randomized location within spawn radius
        let new_c, new_r;
        if (this.ga_manager) {
            [new_c, new_r] = this.ga_manager._getRandomSpawnPosition();
        } else {
            new_c = 0;
            new_r = 0;
        }

        if (
            child.isClear(new_c, new_r, child.rotation, true) &&
            this.env.canAddOrganism() &&
            Math.random() < REPRODUCTION_SUCCESS_RATE
        ) {
            child.c = new_c;
            child.r = new_r;
            this.env.addOrganism(child);

            // Register with GAManager so it's tracked for fitness/generation end
            if (this.ga_manager) {
                this.ga_manager.registerAgent(child);
            }

            // FossilRecord: treat each child as same species (no anatomy mutation)
            if (this.species) {
                child.species = this.species;
                this.species.addPop();
            }
        }

        // Reset energy gain counter whether or not placement succeeded
        this.energyGainedSinceReproduction = 0;
    }

    // Gaussian additive mutation on a genome array, in-place.
    // Same scheme as GAManager._mutate() but using asexual-specific rate constants.
    // Weights are clipped to [-1, 1] after mutation.
    _mutateGenome(genome) {
        for (let i = 0; i < genome.length; i++) {
            if (Math.random() < ASEXUAL_MUT_PROB) {
                const u1 = 1 - Math.random();
                const u2 = 1 - Math.random();
                const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                genome[i] += ASEXUAL_MUT_SIGMA * z;
                // Clip to [-1, 1]
                genome[i] = Math.max(-1, Math.min(1, genome[i]));
            }
        }
    }

    // ── Day/night and cave helpers ────────────────────────────────────────────
    // Check if organism is currently in a cave cell

    _isInCave() {
        const current_cell = this.env.grid_map.cellAt(this.c, this.r);
        return current_cell && current_cell.state === CellStates.cave;
    }

    // ── Death override ────────────────────────────────────────────────────────
    // Per spec: dead organisms do NOT drop food. Cells become empty.

    die() {
        for (const cell of this.anatomy.cells) {
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing_cell = this.env.grid_map.cellAt(real_c, real_r);
            
            // Don't clear landmarks or caves when organism dies — keep the landscape intact
            if (existing_cell) {
                const name = existing_cell.state.name;
                if (name === 'low food landmark'
                    || name === 'medium food landmark'
                    || name === 'prestige food landmark'
                    || name === 'cave') {
                    continue; // Skip clearing this cell
                }
            }
            
            this.env.changeCell(real_c, real_r, CellStates.empty, null);
        }
        if (this.species) this.species.decreasePop();
        this.living = false;
    }

    // ── Passable cell override ───────────────────────────────────────────────
    // Landmarks and caves are walkable — agents move through them freely.
    // Without this override, movers treat them as walls and get stuck.

    isPassableCell(cell, parent) {
        if (cell == null) return false;
        if (cell.owner === this || cell.owner === parent) return true;
        const name = cell.state.name;
        return name === 'empty'
            || name === 'food'
            || name === 'low food'
            || name === 'medium food'
            || name === 'prestige food'
            || name === 'low food landmark'
            || name === 'medium food landmark'
            || name === 'prestige food landmark'
            || name === 'cave';
    }

    // ── Spawn position override ───────────────────────────────────────────────
    // Override isClear() to allow spawning on landmarks and caves, since they
    // are passable for movement. This prevents reproduction failures when
    // spawn areas are densely populated with landmarks/caves.

    isClear(col, row, rotation = this.rotation) {
        for (const loccell of this.anatomy.cells) {
            const cell = this.getRealCell(loccell, col, row, rotation);
            if (cell == null) {
                return false;
            }
            // Allow empty, owned, food, landmarks, and caves
            const name = cell.state.name;
            if (cell.owner === this || name === 'empty'
                || (!Hyperparams.foodBlocksReproduction && (name === 'food' || name === 'low food' || name === 'medium food' || name === 'prestige food'))
                || name === 'low food landmark'
                || name === 'medium food landmark'
                || name === 'prestige food landmark'
                || name === 'cave') {
                continue;
            }
            return false;
        }
        return true;
    }

    // ── Movement override ────────────────────────────────────────────────────
    // Override attemptMove() to preserve landmarks and caves when clearing old
    // cell positions. Base Organism clears everything without checking, which
    // overwrites landscape features.

    attemptMove() {
        const Directions = require('./Directions');
        const direction = Directions.scalars[this.direction];
        const direction_c = direction[0];
        const direction_r = direction[1];
        const new_c = this.c + direction_c;
        const new_r = this.r + direction_r;
        
        if (this.isClear(new_c, new_r)) {
            // Clear old cell positions, but preserve landmarks and caves
            for (const cell of this.anatomy.cells) {
                const real_c = this.c + cell.rotatedCol(this.rotation);
                const real_r = this.r + cell.rotatedRow(this.rotation);
                const existing_cell = this.env.grid_map.cellAt(real_c, real_r);
                
                // Only clear if NOT a landmark or cave
                if (existing_cell) {
                    const name = existing_cell.state.name;
                    if (name === 'low food landmark'
                        || name === 'medium food landmark'
                        || name === 'prestige food landmark'
                        || name === 'cave') {
                        continue; // Keep landscape intact
                    }
                }
                
                this.env.changeCell(real_c, real_r, CellStates.empty, null);
            }
            
            this.c = new_c;
            this.r = new_r;
            this.updateGrid();
            return true;
        }
        return false;
    }

    // ── Rotation override ────────────────────────────────────────────────────
    // Same as attemptMove — preserve landmarks when rotating.

    attemptRotate(rotation = null) {
        const Directions = require('./Directions');
        if (!Hyperparams.rotationEnabled) {
            this.direction = Directions.getRandomDirection();
            this.move_count = 0;
            return true;
        }
        if (rotation == null) {
            rotation = Directions.getRandomDirection();
        }
        if (this.isClear(this.c, this.r, rotation)) {
            // Clear old cell positions, but preserve landmarks and caves
            for (const cell of this.anatomy.cells) {
                const real_c = this.c + cell.rotatedCol(this.rotation);
                const real_r = this.r + cell.rotatedRow(this.rotation);
                const existing_cell = this.env.grid_map.cellAt(real_c, real_r);
                
                // Only clear if NOT a landmark or cave
                if (existing_cell) {
                    const name = existing_cell.state.name;
                    if (name === 'low food landmark'
                        || name === 'medium food landmark'
                        || name === 'prestige food landmark'
                        || name === 'cave') {
                        continue; // Keep landscape intact
                    }
                }
                
                this.env.changeCell(real_c, real_r, CellStates.empty, null);
            }
            
            this.rotation = rotation;
            this.direction = Directions.getRandomDirection();
            this.updateGrid();
            this.move_count = 0;
            return true;
        }
        return false;
    }

    // ── Fitness accessor ──────────────────────────────────────────────────────

    getFitness() {
        return this.cumulative_food_score;
    }

    // ── Grid update override ──────────────────────────────────────────────────
    // Override updateGrid() to preserve landmarks and caves when organisms move
    // over them. This prevents landscape cells from being overwritten by organism
    // body cells, allowing organisms to walk through while keeping the terrain.

    updateGrid() {
        for (const cell of this.anatomy.cells) {
            const real_c = this.c + cell.rotatedCol(this.rotation);
            const real_r = this.r + cell.rotatedRow(this.rotation);
            const existing_cell = this.env.grid_map.cellAt(real_c, real_r);
            
            // If the space has a landmark or cave, preserve it - don't overwrite
            if (existing_cell) {
                const name = existing_cell.state.name;
                if (name === 'low food landmark'
                    || name === 'medium food landmark'
                    || name === 'prestige food landmark'
                    || name === 'cave') {
                    // Skip updating this cell to preserve the landscape
                    continue;
                }
            }
            
            // Otherwise, place the organism cell as normal
            this.env.changeCell(real_c, real_r, cell.state, cell);
        }
    }

    // ── GA genome accessors ───────────────────────────────────────────────────

    getGenome() {
        return this.brain.getGenome();   // genome_weights only, never active_weights
    }

    setGenome(weights) {
        this.brain.setGenome(weights);
    }
}

AdvancedOrganism.MAX_LIFETIME    = MAX_LIFETIME;
AdvancedOrganism.ENERGY_CAPACITY = ENERGY_CAPACITY;
AdvancedOrganism.START_ENERGY    = START_ENERGY;
AdvancedOrganism.FOOD_ENERGY     = FOOD_ENERGY;
AdvancedOrganism.ASEXUAL_MUT_PROB  = ASEXUAL_MUT_PROB;
AdvancedOrganism.ASEXUAL_MUT_SIGMA = ASEXUAL_MUT_SIGMA;

module.exports = AdvancedOrganism;