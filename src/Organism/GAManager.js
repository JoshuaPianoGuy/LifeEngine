/**
 * GAManager.js
 *
 * Controls the between-generation genetic algorithm for the
 * learning-vs-evolution experiment (Model C).
 *
 * What a generation is:
 *   A generation starts when the founding population is spawned and ends when
 *   every organism in the lineage is dead — founders AND all their asexual
 *   descendants. There is no fixed tick count. A successful generation with
 *   good food-seeking behaviour will last longer and produce more descendants.
 *   A failing generation dies out quickly.
 *
 * Within-generation reproduction (handled by AdvancedOrganism.reproduce()):
 *   Asexual. Child genome = parent genome_weights + Gaussian mutation.
 *   Triggered by the standard LifeEngine mechanic: food_collected >= body size.
 *   Every new child calls GAManager.registerAgent() so it is tracked.
 *
 * Between-generation evolution (handled here):
 *   When the lineage dies out entirely, GAManager.evolve() runs:
 *     1. Collect all agents that lived this generation (founders + descendants)
 *     2. Sort by fitness (cumulative food score)
 *     3. Take top N_PARENTS (5) as parents
 *     4. Uniform crossover between parents → POPULATION_SIZE offspring genomes
 *     5. Gaussian mutation on each offspring genome
 *     6. No elitism — every genome in the next generation comes from crossover
 *   This seeds the founding population of generation G+1.
 *
 * Metrics recorded per generation (per spec):
 *   avg_energy_early  — average energy of all agents at their 20% lifetime mark
 *   avg_energy_end    — average energy of all agents at death / end of lifetime
 *   top20percent_fitness — average cumulative food score of the top 20% agents
 *   generation_ticks  — how many world ticks the generation lasted
 *   population_peak   — largest simultaneous population during the generation
 */

'use strict';

const NNBrain         = require('./Perception/NNBrain');
const AdvancedOrganism = require('./AdvancedOrganism');
const CellStates      = require('./Cell/CellStates');
const logger          = require('../Logger');

// ── GA hyper-parameters ───────────────────────────────────────────────────────

const POPULATION_SIZE = 100;   // founding population per generation; raised for larger experiments
const SELECTION_PERCENT = 0.2; // top 20% GA selection
const MUT_PROB        = 0.03;  // 3% per-weight mutation probability between generations
const TOURNAMENT_K    = 2;     // tournament size for parent selection
const MUT_SIGMA       = 0.1;   // Gaussian noise std-dev
const SPAWN_RADIUS    = 30;    // spawn organisms within a 30-cell radius to fit 100 organisms

const TICKS_PER_MAP   = 2000;
const MAPS_PER_GEN    = 5;

// ─────────────────────────────────────────────────────────────────────────────

class GAManager {
    /**
     * @param {WorldEnvironment} env
     * @param {boolean} rl_enabled   Condition A = true, Condition B = false
     * @param {number}  spawn_col    fixed spawn column for founding agents
     * @param {number}  spawn_row    fixed spawn row
     */
    constructor(env, rl_enabled, spawn_col, spawn_row) {
        this.env        = env;
        this.rl_enabled = rl_enabled;
        this.condition_label = rl_enabled ? 'learning' : 'natural_selection';
        this.spawn_col  = spawn_col;
        this.spawn_row  = spawn_row;

        this.generation    = 0;
        this.current_map_index = 0;
        this.map_tick_count    = 0;   // world ticks elapsed in the current map
        this.tick_count        = 0;   // total world ticks this generation

        // All agents ever registered this generation (founders + descendants).
        // Agents are never removed from this list — dead agents stay here so
        // their fitness scores are available for evolve().
        this.all_agents    = [];

        // Living agents this generation — updated by tick() and registerAgent().
        this.living_agents = new Set();

        // Peak population seen this generation
        this.peak_population = 0;

        // Genomes for the next founding generation (null = random init for gen 0)
        this.gene_pool = null;

        this.metrics = [];

        // Expose hyper-parameters so Logger and other consumers can read them
        // without importing the module-level constants directly.
        this.selection_percent = SELECTION_PERCENT;
    }

    // ── Generation lifecycle ──────────────────────────────────────────────────

    /**
     * Spawn POPULATION_SIZE founding agents from gene_pool (or Xavier-random
     * if gene_pool is null, i.e. generation 0).
     * Founders spawn within SPAWN_RADIUS of the base spawn location.
     */
    spawnGeneration() {
        this.all_agents      = [];
        this.living_agents   = new Set();
        this.tick_count      = 0;
        this.map_tick_count  = 0;
        this.current_map_index = 0;
        this.peak_population = 0;

        // Clear any leftover organisms in env to be safe
        this.env.organisms = [];

        for (let i = 0; i < POPULATION_SIZE; i++) {
            const org = new AdvancedOrganism(
                0,
                0,
                this.env,
                null,             // no parent — genome set below
                this.rl_enabled,
                this             // pass GA reference so reproduce() can call registerAgent()
            );

            // Build anatomy: multiple mouths for redundant food access from all directions
            // Eyes positioned at diagonals so they don't block eating.
            //
            // Grid positions (local coords):
            //   (-1,-2)  (0,-2)  (1,-2)
            //   (-1,-1)  (0,-1)  (1,-1)
            //   (-1, 0)  (0, 0)  (1, 0)
            //   (-1, 1)  (0, 1)  (1, 1)
            //   (-1, 2)  (0, 2)  (1, 2)
            //
            // Our layout:
            //           mover
            // eye      mouth      eye
            // mouth    mouth     mouth  
            // eye      mouth      eye
            //
            // 5 mouths + 1 mover + 4 eyes = 10 cells for max food eating capability.

            // Mover at (0, -2) above all - can move up to find food
            org.anatomy.addDefaultCell(CellStates.mover, 0, -2);

            // Diagonal eye positions (don't block eating)
            const eye_up_left = org.anatomy.addDefaultCell(CellStates.eye, -1, -1);
            if (eye_up_left) eye_up_left.direction = 0;  // Directions.up

            const eye_up_right = org.anatomy.addDefaultCell(CellStates.eye, 1, -1);
            if (eye_up_right) eye_up_right.direction = 1;  // Directions.right

            const eye_down_left = org.anatomy.addDefaultCell(CellStates.eye, -1, 1);
            if (eye_down_left) eye_down_left.direction = 3;  // Directions.left

            const eye_down_right = org.anatomy.addDefaultCell(CellStates.eye, 1, 1);
            if (eye_down_right) eye_down_right.direction = 2;  // Directions.down

            // Cardinal mouths for redundant directional coverage
            org.anatomy.addDefaultCell(CellStates.mouth, 0, -1);  // top mouth
            org.anatomy.addDefaultCell(CellStates.mouth, -1, 0);  // left mouth
            org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);   // center mouth
            org.anatomy.addDefaultCell(CellStates.mouth, 1, 0);   // right mouth
            org.anatomy.addDefaultCell(CellStates.mouth, 0, 1);   // bottom mouth

            // Update anatomy type flags (is_mover, has_eyes, is_producer)
            org.anatomy.checkTypeChange();

            if (this.gene_pool !== null) {
                org.setGenome(this.gene_pool[i % this.gene_pool.length]);
            }

            const spawn = this._findSpawnPosition(org);
            if (!spawn) {
                continue;
            }
            org.c = spawn[0];
            org.r = spawn[1];

            this.env.addOrganism(org);
            this.registerAgent(org);
        }

        this.generation++;
        logger.logEvent('GA', `Gen ${this.generation} started | ${POPULATION_SIZE} founders | RL=${this.rl_enabled}`);
    }

    /**
     * Register a newly created agent (founder or asexual descendant).
     * Called by AdvancedOrganism.reproduce() for every child born mid-generation,
     * and by spawnGeneration() for founding agents.
     *
     * @param {AdvancedOrganism} agent
     */
    registerAgent(agent) {
        this.all_agents.push(agent);
        this.living_agents.add(agent);
        if (this.living_agents.size > this.peak_population) {
            this.peak_population = this.living_agents.size;
        }
    }

    /**
     * Call once per world tick (after WorldEnvironment.update()).
     * Removes newly-dead agents from living_agents.
     * Returns 'NEXT_MAP' when map epoch ends, 'NEXT_GENERATION' when max maps reached or all dead.
     * Returns null otherwise.
     */
    tick() {
        this.tick_count++;
        this.map_tick_count++;

        // Remove agents that died this tick
        const dead_agents = [];
        for (const agent of this.living_agents) {
            if (!agent.living) {
                dead_agents.push(agent);
            }
        }
        for (const agent of dead_agents) {
            this.living_agents.delete(agent);
        }

        // Map ends exactly at TICKS_PER_MAP, generating evolving epoch when maps exhausted
        if (this.map_tick_count >= TICKS_PER_MAP) {
            this.current_map_index++;
            if (this.current_map_index >= MAPS_PER_GEN) {
                return 'NEXT_GENERATION'; // Reached map limit -> Evolve
            } else {
                return 'NEXT_MAP'; // Move to next map snippet
            }
        }
        return null;
    }

    startNextMap() {
        this.map_tick_count = 0;
        logger.logEvent('GA', `Gen ${this.generation} -> starting Map ${this.current_map_index + 1}/${MAPS_PER_GEN}`);

        // Keep living agents in the environment array but reposition them and reset traces
        const survivors = Array.from(this.living_agents);
        this.env.organisms = []; // Temporarily clear list to let addOrganism work cleanly without duplicates
        
        for (const org of survivors) {
            // Do not reset energy or position between maps as requested
            // Keep the exact same state, just register them back with the new map

            if (org.brain && org.brain.resetTraces) {
                org.brain.resetTraces();
            }

            // Keep existing position
            // Ensure they are registered in the new environment's grid
            this.env.addOrganism(org);
        }
    }

    /**
     * Run selection → crossover → mutation → record metrics.
     * Call when tick() returns true.
     * After this, gene_pool is ready for the next spawnGeneration().
     */
    evolve() {
        // Sort all agents by fitness descending
        const sorted = [...this.all_agents].sort((a, b) => b.getFitness() - a.getFitness());

        this._recordMetrics(sorted);

        // Tournament selection (k=TOURNAMENT_K) replaces truncation selection.
        // Each parent slot runs a mini-tournament: draw k agents at random from
        // the full sorted population, the fitter one wins. This gives every agent
        // a non-zero selection chance while still favouring higher fitness —
        // selection pressure is controlled by k without a hard truncation cutoff.
        // The between-generation mutation step (_mutate) is unchanged.
        const next_pool = [];

        logger.logEvent('GA',
            `Gen ${this.generation - 1} | ${sorted.length} agents | ` +
            `tournament k=${TOURNAMENT_K} → ${POPULATION_SIZE} offspring`);

        while (next_pool.length < POPULATION_SIZE) {
            const pa = this._tournamentSelect(sorted);
            let pb;
            // Ensure two distinct parents when pool is large enough
            do {
                pb = this._tournamentSelect(sorted);
            } while (pb === pa && sorted.length > 1);

            const child = this._uniformCrossover(pa.getGenome(), pb.getGenome());
            this._mutate(child);
            next_pool.push(child);
        }

        this.gene_pool = next_pool;
    }

    // ── Genetic operators ─────────────────────────────────────────────────────

    /**
     * Tournament selection: draw TOURNAMENT_K agents at random from the pool
     * and return the one with the highest fitness. With k=2 this gives a
     * selection probability of p(rank i) = (n-i)/(n*(n-1)/2) approximately,
     * preserving diversity while still favouring fit agents.
     */
    _tournamentSelect(pool) {
        let best = null;
        for (let i = 0; i < TOURNAMENT_K; i++) {
            const candidate = pool[Math.floor(Math.random() * pool.length)];
            if (best === null || candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        return best;
    }

    /**
     * Uniform crossover: each weight independently from parent A or B at 50/50.
     * Preferred over single-point for neural weight arrays — avoids the
     * permutation problem where neuron j in A may not match neuron j in B.
     * Weights are clipped to [-1, 1] after crossover.
     */
    _uniformCrossover(genome_a, genome_b) {
        const child = new Float32Array(NNBrain.GENOME_SIZE);
        for (let i = 0; i < NNBrain.GENOME_SIZE; i++) {
            const weight = Math.random() < 0.5 ? genome_a[i] : genome_b[i];
            // Clip to [-1, 1]
            child[i] = Math.max(-1, Math.min(1, weight));
        }
        return child;
    }

    /**
     * Gaussian additive mutation, in-place.
     * Each weight has MUT_PROB chance of being perturbed by N(0, MUT_SIGMA).
     * Applied to every genome entering the next generation's founding pool.
     * Weights are clipped to [-1, 1] after mutation.
     *
     * Note: AdvancedOrganism.reproduce() applies a SEPARATE mutation step
     * (ASEXUAL_MUT_PROB / ASEXUAL_MUT_SIGMA) when spawning children within
     * a generation. These two mutation rates can be tuned independently:
     *   - ASEXUAL_MUT: controls within-generation genome drift
     *   - MUT_PROB/SIGMA here: controls between-generation disruption
     */
    _mutate(genome) {
        for (let i = 0; i < genome.length; i++) {
            if (Math.random() < MUT_PROB) {
                genome[i] += this._gaussianSample(0, MUT_SIGMA);
                // Clip to [-1, 1]
                genome[i] = Math.max(-1, Math.min(1, genome[i]));
            }
        }
    }

    _gaussianSample(mean, sigma) {
        const u1 = 1 - Math.random();
        const u2 = 1 - Math.random();
        return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Generate a random spawn position within SPAWN_RADIUS of the centre.
     *
     * @returns {[number, number]} [col, row] spawn position
     */
    _getRandomSpawnPosition() {
        const grid = this.env.grid_map;
        if (!grid) {
            return [this.spawn_col, this.spawn_row];
        }

        // Try a few times to get a point strictly inside the circular radius
        for (let i = 0; i < 50; i++) {
            const rx = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            const ry = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            
            if (rx * rx + ry * ry <= SPAWN_RADIUS * SPAWN_RADIUS) {
                const col = Math.floor(this.spawn_col + rx);
                const row = Math.floor(this.spawn_row + ry);
                
                if (col >= 0 && col < grid.cols && row >= 0 && row < grid.rows) {
                    return [col, row];
                }
            }
        }

        return [this.spawn_col, this.spawn_row];
    }

    _findSpawnPosition(org, attempts = 200) {
        for (let i = 0; i < attempts; i++) {
            const [col, row] = this._getRandomSpawnPosition();
            if (org.isClear(col, row, org.rotation)) {
                return [col, row];
            }
        }
        return null;
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    _recordMetrics(sorted_agents) {
        if (sorted_agents.length === 0) return;

        // Delegate all logging and analysis to Logger — it computes drift,
        // genome variance, per-food counts, and writes both summary and
        // per-organism rows in one call.
        logger.logGeneration(this, sorted_agents);

        // Keep a lightweight local copy for exportMetricsCSV() backward compat
        const n = sorted_agents.length;
        let num_top = Math.floor(n * SELECTION_PERCENT);
        num_top = Math.max(1, Math.min(num_top, n));
        const top_parents = sorted_agents.slice(0, num_top);
        const top_parents_fitness = top_parents.reduce((s, a) => s + a.getFitness(), 0) / top_parents.length;
        const avg_lifetime = sorted_agents.reduce((s, a) => s + (a.lifetime || 0), 0) / n;
        const early_samples = sorted_agents.map(a => a.energy_at_early_sample).filter(e => e != null);
        const avg_energy_early = early_samples.length > 0
            ? early_samples.reduce((s, e) => s + e, 0) / early_samples.length : 0;

        this.metrics.push({
            generation:       this.generation,
            condition:        this.rl_enabled ? 'learning' : 'natural_selection',
            generation_ticks: this.tick_count,
            total_agents:     n,
            peak_population:  this.peak_population,
            avg_energy_early: avg_energy_early.toFixed(3),
            avg_energy_end:   (sorted_agents.reduce((s, a) => s + (a.energy || 0), 0) / n).toFixed(3),
            top20percent_fitness: top_parents_fitness.toFixed(4),
            best_fitness:     sorted_agents[0].getFitness().toFixed(4),
            avg_lifetime:     avg_lifetime.toFixed(1),
        });
    }

    // ── Export ────────────────────────────────────────────────────────────────

    exportMetricsCSV() {
        if (this.metrics.length === 0) return '';
        const headers = Object.keys(this.metrics[0]).join(',');
        const rows    = this.metrics.map(m => Object.values(m).join(',')).join('\n');
        return headers + '\n' + rows;
    }

    downloadMetrics(filename = 'experiment_metrics.csv') {
        const csv  = this.exportMetricsCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

GAManager.POPULATION_SIZE = POPULATION_SIZE;
GAManager.SELECTION_PERCENT = SELECTION_PERCENT;
GAManager.MUT_PROB        = MUT_PROB;
GAManager.MUT_SIGMA       = MUT_SIGMA;
GAManager.TOURNAMENT_K    = TOURNAMENT_K;

module.exports = GAManager;