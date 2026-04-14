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

const MAX_LIFETIME      = 5000;  // increased from 1500 to allow more time for learning
const ENERGY_CAPACITY   = 200;   // maximum energy
const START_ENERGY      = 150;   // increased from 100 to give organisms buffer for reproduction
const ENERGY_DECAY_RATE = 1;     // energy lost per decay event
const ENERGY_DECAY_INTERVAL = 10; // ticks between decay events (per spec: lose energy every 10 ticks)

// ── Day/night energy mechanics ─────────────────────────────────────────────────
//
// Day (150 ticks):
//   - Outside cave: normal decay (1 per 10 ticks) — organisms search for food
//   - Inside cave: accelerated decay (2-3x rate) — not a refuge during day
//
// Night (75 ticks):
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

const DECAY_PENALTY  = 0.1;   // negative reward each time energy decays
const EXPLORE_BONUS  = 0.02;  // positive reward for visiting a new cell

// ── GA mutation constants (within-generation asexual reproduction) ────────────
// Applied when an agent reproduces mid-generation.
// DISABLED: only testing within-lifetime learning, not morphological changes

const ASEXUAL_MUT_PROB  = 0.0;   // DISABLED: per-weight mutation probability
const ASEXUAL_MUT_SIGMA = 0.1;   // Gaussian noise std-dev (unused when ASEXUAL_MUT_PROB = 0)

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
    }

    // ── Lifespan ──────────────────────────────────────────────────────────────

    lifespan() {
        return MAX_LIFETIME;
    }

    // ── Reproduction Threshold ─────────────────────────────────────────────────
    // Override to reduce reproduction cost for faster population growth.
    // Base: anatomy.cells.length = 10 food needed.
    // Adjusted: 4 food needed for a 10-cell organism.
    // This allows children to achieve reproduction more easily and build stable pops.
    foodNeeded() {
        return 4;  // reduced from this.anatomy.cells.length (10)
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
        // Day: normal outside, 3x faster in caves
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

        // Check if enough food collected to reproduce (standard LifeEngine trigger)
        // This happens before cell functions so the parent isn't mid-move when it
        // spawns a child.
        if (this.food_collected >= this.foodNeeded()) {
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
            this.energy                += gained;
            this.cumulative_food_score += value;
            this.pending_reward        += value;

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
        const lifetime_frac = Math.min(1, this.lifetime / MAX_LIFETIME);
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
        this.direction = action;

        const moved = this.attemptMove();
        if (!moved) {
            this.direction = Directions.getRandomDirection();
            this.attemptMove();
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

        // Spawn at fixed location (same as founders) so all organisms start from
        // the same place regardless of parent location
        const new_c = this.ga_manager ? this.ga_manager.spawn_col : 0;
        const new_r = this.ga_manager ? this.ga_manager.spawn_row : 0;

        if (
            child.isClear(new_c, new_r, child.rotation, true) &&
            this.env.canAddOrganism()
        ) {
            child.c = new_c;
            child.r = new_r;
            this.env.addOrganism(child);
            child.updateGrid();

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

        // Deduct food cost whether or not placement succeeded (base LifeEngine behaviour)
        this.food_collected = Math.max(0, this.food_collected - this.foodNeeded());
    }

    // Gaussian additive mutation on a genome array, in-place.
    // Same scheme as GAManager._mutate() but using asexual-specific rate constants.
    _mutateGenome(genome) {
        for (let i = 0; i < genome.length; i++) {
            if (Math.random() < ASEXUAL_MUT_PROB) {
                const u1 = 1 - Math.random();
                const u2 = 1 - Math.random();
                const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                genome[i] += ASEXUAL_MUT_SIGMA * z;
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

    // ── Fitness accessor ──────────────────────────────────────────────────────

    getFitness() {
        return this.cumulative_food_score;
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