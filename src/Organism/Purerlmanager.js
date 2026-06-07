/**
 * PureRLManager.js
 *
 * Condition D: Pure Reinforcement Learning
 *
 * What this condition is:
 *   Organisms learn entirely within their own lifetimes via REINFORCE (same as
 *   Condition A). There is NO crossover, NO between-generation gradient update,
 *   and NO generational reset. The only cross-generation improvement mechanism
 *   is differential reproduction — fitter organisms reproduce more and
 *   contribute their starting genome (not their RL-drifted weights) to a
 *   larger share of the next population. Small Gaussian mutation on reproduction
 *   prevents diversity collapse.
 *
 * Key differences from other conditions:
 *   vs Condition A (RL + GA): no crossover, no between-generation genetic ops.
 *   vs Condition B (GA only): RL is active within each lifetime.
 *   vs Condition C (FrozenPG): RL runs during lifetimes, not as a between-gen
 *                               gradient update on a frozen policy.
 *
 * Generation / episode boundary:
 *   There is no hard generational reset. The simulation runs continuously.
 *   "Generations" are purely measurement windows of TICKS_PER_GEN ticks —
 *   the same 10 000-tick boundary used by other conditions (5 maps × 2 000
 *   ticks). At the end of each window, metrics are snapshotted and logged
 *   using the same schema as GAManager, so all three conditions can be
 *   compared on a shared tick axis.
 *
 * Inheritance:
 *   Offspring inherit the parent's STARTING genome (genome_weights at birth),
 *   not the RL-drifted active_weights. This is non-Lamarckian, matching
 *   Condition A. The 3% asexual mutation rate used in AdvancedOrganism
 *   .reproduce() applies as normal; no extra between-generation mutation step
 *   is needed because reproduction is continuous.
 *
 * Population management:
 *   No founding population reset. The sim starts with POPULATION_SIZE
 *   organisms seeded from random Xavier weights and runs until stopped.
 *   If the population collapses to zero this is treated as a legitimate
 *   result (within-lifetime RL alone could not sustain the population).
 */

'use strict';

const NNBrain          = require('./Perception/NNBrain');
const AdvancedOrganism = require('./AdvancedOrganism');
const CellStates       = require('./Cell/CellStates');
const logger           = require('../Logger');

// ── Hyper-parameters ──────────────────────────────────────────────────────────

const POPULATION_SIZE = 100;   // founding population only; grows/shrinks naturally after
const SPAWN_RADIUS    = 30;

// Measurement window — matches other conditions exactly (5 maps × 2 000 ticks)
const TICKS_PER_MAP   = 2000;
const MAPS_PER_GEN    = 5;
const TICKS_PER_GEN   = TICKS_PER_MAP * MAPS_PER_GEN; // 10 000

// ─────────────────────────────────────────────────────────────────────────────

class PureRLManager {
    /**
     * @param {WorldEnvironment} env
     * @param {number} spawn_col  fixed spawn column for founding agents
     * @param {number} spawn_row  fixed spawn row
     */
    constructor(env, spawn_col, spawn_row) {
        this.env        = env;
        this.spawn_col  = spawn_col;
        this.spawn_row  = spawn_row;

        // RL is always active in this condition
        this.rl_enabled      = true;
        this.condition_label = 'pure_rl';

        // Measurement window counters — reset each window, never trigger a
        // population reset.
        this.generation          = 0;   // measurement window index (logged as 'generation')
        this.current_map_index   = 0;
        this.map_tick_count      = 0;
        this.tick_count          = 0;   // ticks elapsed in the current measurement window
        this.total_ticks         = 0;   // total ticks ever (matches env.total_ticks semantics)

        // Agent tracking — agents are added on birth and retained until logged.
        // Unlike GAManager, living_agents is the primary population; there is no
        // clean founding/descendant boundary.
        this.all_agents_this_window = [];  // every agent alive or born this window
        this.living_agents          = new Set();
        this.peak_population        = 0;

        // No gene_pool — population is self-sustaining via reproduction
    }

    // ── Initial population spawn ──────────────────────────────────────────────

    /**
     * Spawn the founding population once at the start of the experiment.
     * Called once from WorldEnvironment (same call site as GAManager.spawnGeneration).
     * After this, organisms reproduce and die naturally — no further resets.
     */
    spawnGeneration(options = {}) {
        const spawn_col = options.spawn_col !== undefined ? options.spawn_col : this.spawn_col;
        const spawn_row = options.spawn_row !== undefined ? options.spawn_row : this.spawn_row;
        const seed_weights = options.seed_weights || null;
        const preserve_window_state = Boolean(options.preserve_window_state);

        if (!preserve_window_state) {
            this.all_agents_this_window = [];
            this.tick_count             = 0;
            this.map_tick_count         = 0;
            this.current_map_index      = 0;
            this.peak_population        = 0;
            this.generation             = 1;
        }

        this.living_agents = new Set();
        this.env.organisms = [];

        for (let i = 0; i < POPULATION_SIZE; i++) {
            const org = new AdvancedOrganism(
                0,
                0,
                this.env,
                null,       // no parent — random Xavier init
                true,       // rl_enabled = true
                this        // manager reference so reproduce() calls registerAgent()
            );

            // Anatomy matches other conditions exactly
            org.anatomy.addDefaultCell(CellStates.mover, 0, -2);

            const eye_ul = org.anatomy.addDefaultCell(CellStates.eye, -1, -1);
            if (eye_ul) eye_ul.direction = 0;
            const eye_ur = org.anatomy.addDefaultCell(CellStates.eye,  1, -1);
            if (eye_ur) eye_ur.direction = 1;
            const eye_dl = org.anatomy.addDefaultCell(CellStates.eye, -1,  1);
            if (eye_dl) eye_dl.direction = 3;
            const eye_dr = org.anatomy.addDefaultCell(CellStates.eye,  1,  1);
            if (eye_dr) eye_dr.direction = 2;

            org.anatomy.addDefaultCell(CellStates.mouth,  0, -1);
            org.anatomy.addDefaultCell(CellStates.mouth, -1,  0);
            org.anatomy.addDefaultCell(CellStates.mouth,  0,  0);
            org.anatomy.addDefaultCell(CellStates.mouth,  1,  0);
            org.anatomy.addDefaultCell(CellStates.mouth,  0,  1);

            org.anatomy.checkTypeChange();

            if (seed_weights) {
                org.brain.setGenome(seed_weights);
            }

            // Genome left as random Xavier — no gene_pool for generation 0

            const spawn = this._findSpawnPosition(org, spawn_col, spawn_row);
            if (!spawn) continue;
            org.c = spawn[0];
            org.r = spawn[1];

            this.env.addOrganism(org);
            this.registerAgent(org);
        }

        if (!preserve_window_state) {
            logger.logEvent('PureRL', `Founding population | ${POPULATION_SIZE} organisms | RL=true | no generational resets`);
        }
    }

    respawnPopulationAtCenter() {
        const template_agent = this._getBestWindowAgent();
        if (!template_agent || !template_agent.brain) {
            return false;
        }

        const center = this.env.grid_map ? this.env.grid_map.getCenter() : [this.spawn_col, this.spawn_row];
        const seed_weights = new Float32Array(template_agent.brain.active_weights || template_agent.brain.genome_weights);

        this.spawnGeneration({
            preserve_window_state: true,
            spawn_col: center[0],
            spawn_row: center[1],
            seed_weights,
        });

        logger.logEvent('PureRL',
            `Population extinct | respawned at center | best_fit=${template_agent.getFitness().toFixed(2)}`
        );
        return true;
    }

    // ── Agent registration ────────────────────────────────────────────────────

    /**
     * Register a newly born agent (founding or asexual descendant).
     * Called by AdvancedOrganism.reproduce() for every child and by
     * spawnGeneration() for founders.
     */
    registerAgent(agent) {
        this.all_agents_this_window.push(agent);
        this.living_agents.add(agent);
        if (this.living_agents.size > this.peak_population) {
            this.peak_population = this.living_agents.size;
        }
    }

    // ── Per-tick update ───────────────────────────────────────────────────────

    /**
     * Call once per world tick (after WorldEnvironment.update()).
     * Returns 'NEXT_MAP' at each map boundary, 'NEXT_WINDOW' at each
     * measurement window boundary (every TICKS_PER_GEN ticks).
     * Returns null otherwise.
     *
     * Unlike GAManager, 'NEXT_WINDOW' does NOT trigger a population reset —
     * it only triggers metric logging and counter resets.
     */
    tick() {
        this.tick_count++;
        this.map_tick_count++;
        this.total_ticks++;

        // Remove newly dead agents from the living set.
        // Collect first, then delete — modifying a Set while iterating it is
        // undefined behaviour in older runtimes and can skip entries in V8.
        const dead_agents = [];
        for (const agent of this.living_agents) {
            if (!agent.living) dead_agents.push(agent);
        }
        for (const agent of dead_agents) {
            this.living_agents.delete(agent);
        }

        if (this.map_tick_count >= TICKS_PER_MAP) {
            this.current_map_index++;
            if (this.current_map_index >= MAPS_PER_GEN) {
                return 'NEXT_WINDOW';
            }
            return 'NEXT_MAP';
        }

        return null;
    }

    /**
     * Transition to the next map within a measurement window.
     * Organisms keep their positions and energy — same behaviour as GAManager.
     */
    startNextMap() {
        this.map_tick_count = 0;
        logger.logEvent('PureRL', `Window ${this.generation} -> Map ${this.current_map_index + 1}/${MAPS_PER_GEN}`);

        const survivors = Array.from(this.living_agents);
        this.env.organisms = [];

        for (const org of survivors) {
            if (org.brain && org.brain.resetTraces) {
                org.brain.resetTraces();
            }
            this.env.addOrganism(org);
        }
    }

    /**
     * Close a measurement window: log metrics, reset window counters.
     * Does NOT reset the population — organisms continue living.
     * Call when tick() returns 'NEXT_WINDOW'.
     */
    closeWindow() {
        for (const agent of this.living_agents) {
            if (agent.brain && typeof agent.brain.syncGenomeFromActive === 'function') {
                agent.brain.syncGenomeFromActive();
            }
        }

        // Snapshot all agents active this window (dead + alive)
        const sorted = [...this.all_agents_this_window]
            .sort((a, b) => b.getFitness() - a.getFitness());

        // Log using the same Logger.logGeneration path as GAManager so CSV
        // schemas are identical across all conditions
        logger.logGeneration(this, sorted);

        logger.logEvent('PureRL',
            `Window ${this.generation} closed | ` +
            `living=${this.living_agents.size} | ` +
            `window_agents=${sorted.length} | ` +
            `best_fit=${sorted.length > 0 ? sorted[0].getFitness().toFixed(2) : 'n/a'}`
        );

        // Reset window counters — population continues uninterrupted
        this.generation++;
        this.tick_count          = 0;
        this.map_tick_count      = 0;
        this.current_map_index   = 0;
        this.peak_population     = this.living_agents.size;

        // New window tracks agents alive at the start of the window plus any
        // born during it — seed with currently living agents
        this.all_agents_this_window = Array.from(this.living_agents);
    }

    _getBestWindowAgent() {
        if (this.all_agents_this_window.length === 0) {
            return null;
        }

        let best_agent = this.all_agents_this_window[0];
        let best_fitness = best_agent.getFitness();
        for (let i = 1; i < this.all_agents_this_window.length; i++) {
            const agent = this.all_agents_this_window[i];
            const fitness = agent.getFitness();
            if (fitness > best_fitness) {
                best_agent = agent;
                best_fitness = fitness;
            }
        }

        return best_agent;
    }

    // ── Spawn helpers ─────────────────────────────────────────────────────────

    _getRandomSpawnPosition(spawn_col = this.spawn_col, spawn_row = this.spawn_row) {
        const grid = this.env.grid_map;
        if (!grid) return [spawn_col, spawn_row];

        for (let i = 0; i < 50; i++) {
            const rx = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            const ry = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            if (rx * rx + ry * ry <= SPAWN_RADIUS * SPAWN_RADIUS) {
                const col = Math.floor(spawn_col + rx);
                const row = Math.floor(spawn_row + ry);
                if (col >= 0 && col < grid.cols && row >= 0 && row < grid.rows) {
                    return [col, row];
                }
            }
        }
        return [spawn_col, spawn_row];
    }

    _findSpawnPosition(org, spawn_col = this.spawn_col, spawn_row = this.spawn_row, attempts = 200) {
        for (let i = 0; i < attempts; i++) {
            const [col, row] = this._getRandomSpawnPosition(spawn_col, spawn_row);
            if (org.isClear(col, row, org.rotation)) {
                return [col, row];
            }
        }
        return null;
    }
}

PureRLManager.POPULATION_SIZE = POPULATION_SIZE;
PureRLManager.TICKS_PER_GEN   = TICKS_PER_GEN;

module.exports = PureRLManager;