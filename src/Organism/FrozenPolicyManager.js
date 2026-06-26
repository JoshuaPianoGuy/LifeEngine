'use strict';

const NNBrain = require('./Perception/NNBrain');
const FrozenPolicyOrganism = require('./FrozenPolicyOrganism');
const CellStates = require('./Cell/CellStates');
const logger = require('../Logger');

const POPULATION_SIZE = 100;
const MUT_SIGMA = 0.1;
const SPAWN_RADIUS = 30;

// Generation length — single source of truth (honors WorldConfig overrides).
const { TICKS_PER_MAP, MAPS_PER_GEN } = require('./GenerationConstants');

class FrozenPolicyManager {
    constructor(env, spawn_col, spawn_row) {
        this.env = env;
        this.rl_enabled = true;
        this.condition_label = 'frozen_pg';
        this.spawn_col = spawn_col;
        this.spawn_row = spawn_row;

        this.generation = 0;
        this.current_map_index = 0;
        this.map_tick_count = 0;
        this.tick_count = 0;

        this.all_agents = [];
        this.living_agents = new Set();
        this.peak_population = 0;
        this.gene_pool = null;
    }

    spawnGeneration() {
        this.all_agents = [];
        this.living_agents = new Set();
        this.tick_count = 0;
        this.map_tick_count = 0;
        this.current_map_index = 0;
        this.peak_population = 0;

        this.env.organisms = [];

        for (let i = 0; i < POPULATION_SIZE; i++) {
            const org = new FrozenPolicyOrganism(
                0,
                0,
                this.env,
                null,
                true,
                this
            );

            // Mover at (0, -2) above all - can move up to find food
            org.anatomy.addDefaultCell(CellStates.mover, 0, -2);

            // Diagonal eye positions (don't block eating)
            const eye_up_left = org.anatomy.addDefaultCell(CellStates.eye, -1, -1);
            if (eye_up_left) eye_up_left.direction = 0;

            const eye_up_right = org.anatomy.addDefaultCell(CellStates.eye, 1, -1);
            if (eye_up_right) eye_up_right.direction = 1;

            const eye_down_left = org.anatomy.addDefaultCell(CellStates.eye, -1, 1);
            if (eye_down_left) eye_down_left.direction = 3;

            const eye_down_right = org.anatomy.addDefaultCell(CellStates.eye, 1, 1);
            if (eye_down_right) eye_down_right.direction = 2;

            // Cardinal mouths for redundant directional coverage
            org.anatomy.addDefaultCell(CellStates.mouth, 0, -1);
            org.anatomy.addDefaultCell(CellStates.mouth, -1, 0);
            org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);
            org.anatomy.addDefaultCell(CellStates.mouth, 1, 0);
            org.anatomy.addDefaultCell(CellStates.mouth, 0, 1);

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
        logger.logEvent('GA', `Gen ${this.generation} started | ${POPULATION_SIZE} founders | mode=frozen_pg`);
    }

    registerAgent(agent) {
        this.all_agents.push(agent);
        this.living_agents.add(agent);
        if (this.living_agents.size > this.peak_population) {
            this.peak_population = this.living_agents.size;
        }
    }

    tick() {
        this.tick_count++;
        this.map_tick_count++;

        const dead_agents = [];
        for (const agent of this.living_agents) {
            if (!agent.living) {
                dead_agents.push(agent);
            }
        }
        for (const agent of dead_agents) {
            this.living_agents.delete(agent);
        }

        if (this.map_tick_count >= TICKS_PER_MAP) {
            this.current_map_index++;
            if (this.current_map_index >= MAPS_PER_GEN) {
                return 'NEXT_GENERATION';
            }
            return 'NEXT_MAP';
        }
        return null;
    }

    startNextMap() {
        this.map_tick_count = 0;
        logger.logEvent('GA', `Gen ${this.generation} -> starting Map ${this.current_map_index + 1}/${MAPS_PER_GEN}`);

        const survivors = Array.from(this.living_agents);
        this.env.organisms = [];

        for (const org of survivors) {
            this.env.addOrganism(org);
        }
    }

    evolve() {
        if (this.all_agents.length === 0) {
            this.gene_pool = null;
            return;
        }

        const sorted = [...this.all_agents].sort((a, b) => b.getFitness() - a.getFitness());
        logger.logGeneration(this, sorted);

        const best = sorted[0];
        const best_reward = best.getFitness();

        if (best.brain && typeof best.brain.applyFrozenUpdate === 'function') {
            best.brain.applyFrozenUpdate(best_reward);
        }

        const base_genome = best.getGenome();
        const next_pool = [];

        while (next_pool.length < POPULATION_SIZE) {
            const child = new Float32Array(base_genome);
            this._mutateAll(child);
            next_pool.push(child);
        }

        this.gene_pool = next_pool;
    }

    _mutateAll(genome) {
        for (let i = 0; i < genome.length; i++) {
            genome[i] += this._gaussianSample(0, MUT_SIGMA);
            genome[i] = Math.max(-1, Math.min(1, genome[i]));
        }
    }

    _gaussianSample(mean, sigma) {
        const u1 = 1 - Math.random();
        const u2 = 1 - Math.random();
        return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    _getRandomSpawnPosition() {
        const grid = this.env.grid_map;
        if (!grid) {
            return [this.spawn_col, this.spawn_row];
        }

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
}

module.exports = FrozenPolicyManager;
