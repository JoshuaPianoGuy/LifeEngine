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
 *   top5_fitness      — average cumulative food score of the top 5 agents
 *   generation_ticks  — how many world ticks the generation lasted
 *   population_peak   — largest simultaneous population during the generation
 */

'use strict';

const NNBrain         = require('./Perception/NNBrain');
const AdvancedOrganism = require('./AdvancedOrganism');
const CellStates      = require('./Cell/CellStates');
const logger          = require('../Logger');

// ── GA hyper-parameters ───────────────────────────────────────────────────────

const POPULATION_SIZE = 1;   // founding population per generation; raised for larger experiments
const N_PARENTS       = 5;     // top-5 per spec
const MUT_PROB        = 0.0;   // DISABLED: only testing within-lifetime learning, not morphological changes
const MUT_SIGMA       = 0.1;   // Gaussian noise std-dev (unused when MUT_PROB = 0)

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
        this.spawn_col  = spawn_col;
        this.spawn_row  = spawn_row;

        this.generation    = 0;
        this.tick_count    = 0;   // world ticks elapsed in the current generation

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
    }

    // ── Generation lifecycle ──────────────────────────────────────────────────

    /**
     * Spawn POPULATION_SIZE founding agents from gene_pool (or Xavier-random
     * if gene_pool is null, i.e. generation 0).
     * All founders start at (spawn_col, spawn_row).
     */
    spawnGeneration() {
        this.all_agents      = [];
        this.living_agents   = new Set();
        this.tick_count      = 0;
        this.peak_population = 0;

        for (let i = 0; i < POPULATION_SIZE; i++) {
            const org = new AdvancedOrganism(
                this.spawn_col,
                this.spawn_row,
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
     * Returns true when the entire lineage has died out and evolve() should run.
     *
     * @returns {boolean}
     */
    tick() {
        this.tick_count++;

        // Remove agents that died this tick
        for (const agent of this.living_agents) {
            if (!agent.living) {
                this.living_agents.delete(agent);
            }
        }

        // Generation ends when no living agents remain
        return this.living_agents.size === 0 && this.all_agents.length > 0;
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

        // Use min(N_PARENTS, population_size) to handle small populations
        const num_parents = Math.min(N_PARENTS, sorted.length);
        const parents = sorted.slice(0, num_parents);
        const next_pool = [];
        
        logger.logEvent('GA', `Gen ${this.generation - 1} size: ${sorted.length} agents, selecting top ${num_parents} parents`);

        // Fill entire next generation with crossover offspring — no elitism.
        // Every genome comes from uniform crossover between two parents drawn
        // randomly from the top-5, then mutated.
        while (next_pool.length < POPULATION_SIZE) {
            const pa    = parents[Math.floor(Math.random() * parents.length)];
            const pb    = parents[Math.floor(Math.random() * parents.length)];
            const child = this._uniformCrossover(pa.getGenome(), pb.getGenome());
            this._mutate(child);
            next_pool.push(child);
        }

        this.gene_pool = next_pool;
    }

    // ── Genetic operators ─────────────────────────────────────────────────────

    /**
     * Uniform crossover: each weight independently from parent A or B at 50/50.
     * Preferred over single-point for neural weight arrays — avoids the
     * permutation problem where neuron j in A may not match neuron j in B.
     */
    _uniformCrossover(genome_a, genome_b) {
        const child = new Float32Array(NNBrain.GENOME_SIZE);
        for (let i = 0; i < NNBrain.GENOME_SIZE; i++) {
            child[i] = Math.random() < 0.5 ? genome_a[i] : genome_b[i];
        }
        return child;
    }

    /**
     * Gaussian additive mutation, in-place.
     * Each weight has MUT_PROB chance of being perturbed by N(0, MUT_SIGMA).
     * Applied to every genome entering the next generation's founding pool.
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
            }
        }
    }

    _gaussianSample(mean, sigma) {
        const u1 = 1 - Math.random();
        const u2 = 1 - Math.random();
        return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
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
        const top5 = sorted_agents.slice(0, Math.min(N_PARENTS, n));
        const top5_fitness = top5.reduce((s, a) => s + a.getFitness(), 0) / top5.length;
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
            top5_fitness:     top5_fitness.toFixed(4),
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
GAManager.N_PARENTS       = N_PARENTS;
GAManager.MUT_PROB        = MUT_PROB;
GAManager.MUT_SIGMA       = MUT_SIGMA;

module.exports = GAManager;