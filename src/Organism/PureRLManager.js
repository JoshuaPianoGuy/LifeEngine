/**
 * PureRLManager.js — Condition D: Pure Reinforcement Learning
 *
 * Design
 * ------
 * A population of organisms learn entirely through within-lifetime
 * REINFORCE. There is no GA crossover and no fitness-based selection.
 * The only mechanisms driving improvement are:
 *
 *   1. Within-lifetime RL   — active_weights updated each tick via REINFORCE
 *   2. Natural reproduction  — energy-triggered, child inherits parent's
 *                              current active_weights (Lamarckian continuity)
 *                              plus Gaussian mutation via _mutateGenome()
 *   3. Between-episode mutation (optional, see BETWEEN_EPISODE_MUTATION) —
 *                              undirected Gaussian noise on all survivors'
 *                              active_weights at each episode boundary
 *
 * Why active_weights as the heritable unit
 * -----------------------------------------
 * In Condition A (RL+GA), genome_weights are the heritable unit — they
 * are frozen during a lifetime and only moved by the GA between generations.
 * Here there is no GA step, so using genome_weights as the heritable unit
 * would discard all within-lifetime learning at every reproduction event.
 * Instead, active_weights (the RL-learned state) are synced to genome_weights
 * immediately before each child is created, so the child inherits what the
 * parent actually learned rather than its starting point. This maintains
 * continuity of learning across the population over time.
 *
 * This is intentionally Lamarckian and is the defining characteristic of
 * this condition: learned weights persist and propagate through reproduction,
 * contrasting with the non-Lamarckian RL+GA condition where learned weights
 * die with the organism.
 *
 * Population management
 * ---------------------
 * Within an episode the population grows freely via natural reproduction and
 * shrinks as organisms die. At each episode boundary it is forced back to
 * exactly POPULATION_SIZE so every window STARTS full — matching the
 * learning/evolution conditions, which found each generation with
 * POPULATION_SIZE organisms (without this, a window that started with 3
 * survivors would not be comparable to their 100):
 *
 *   - above POPULATION_SIZE : the excess is removed RANDOMLY (not by fitness)
 *                             — random removal has no fitness signal and is
 *                             not selection pressure.
 *   - below POPULATION_SIZE : the deficit is topped up with organisms whose
 *                             weights are sampled randomly from a rolling
 *                             buffer of the last COLLAPSE_BUFFER_SIZE dead
 *                             organisms' active_weights, each mutated around
 *                             (same Gaussian kernel as between-episode
 *                             mutation). A full collapse to zero is just the
 *                             deficit == POPULATION_SIZE case.
 *
 * Buffering learned weights (rather than always spawning fresh Xavier) means a
 * shortfall is refilled with real, already-learned strategies — a shrink is
 * often environmental (harsh map, food scarcity, insufficient learning time)
 * rather than a sign the weights are bad, so discarding them would throw away
 * useful learned structure. Random sampling preserves real individual
 * strategies rather than averaging (which can produce weight vectors no
 * organism ever actually had); the per-seed mutation makes the refill a diverse
 * cloud around those strategies rather than exact clones. The top-up happens
 * AFTER the window is logged, so the fresh spawns never pollute that window's
 * metrics.
 *
 * Between-episode mutation
 * ------------------------
 * Controlled by BETWEEN_EPISODE_MUTATION toggle.
 *
 *   false — within-episode RL and reproduction only. No weight perturbation
 *           at episode boundaries. The only noise source is _mutateGenome()
 *           at reproduction.
 *
 *   true  — at each episode boundary, all surviving organisms receive
 *           undirected Gaussian noise on their active_weights (same kernel as
 *           the GA between-generation mutation in GAManager — ExperimentParams
 *           mut_prob / mut_sigma, default 3% per weight, σ=0.1). The mutated
 *           state is synced to genome_weights so children inherit it.
 *           Mutation without selection is not evolutionary search — it is
 *           an undirected random walk that adds diversity without gradient.
 *
 * Comparison axis
 * ---------------
 * Episode boundaries are 10,000-tick windows (5 maps × 2,000 ticks),
 * matching GAManager exactly. Logger.logGeneration is called at each
 * boundary with the same schema so all conditions are directly comparable.
 */

'use strict';

const AdvancedOrganism = require('./AdvancedOrganism');
const CellStates       = require('./Cell/CellStates');
const logger           = require('../Logger');
const ExperimentParams = require('../ExperimentParams');  // tunable: mutation prob/sigma

// ── Hyper-parameters ──────────────────────────────────────────────────────────

const POPULATION_SIZE = 100;
const SPAWN_RADIUS    = 30;

// Generation length — single source of truth (honors WorldConfig overrides).
const { TICKS_PER_MAP, MAPS_PER_GEN, TICKS_PER_GEN } = require('./GenerationConstants');

// Rolling buffer size for collapse respawn — recent enough to reflect
// current learning, large enough to have meaningful diversity.
const COLLAPSE_BUFFER_SIZE = 25;

// Toggle between-episode mutation. Does not affect within-episode
// reproduction mutation, which always runs via _mutateGenome() (0.05 / 0.1).
const BETWEEN_EPISODE_MUTATION = true;
// Inter-generation (episode-boundary + collapse top-up) mutation kernel, sourced
// from ExperimentParams so it is IDENTICAL to the GA between-generation mutation
// in GAManager (default 0.03 / 0.1) and honours --mut-prob / --mut-sigma. This
// keeps inter-generation mutation consistent across the learning, evolution and
// pure-RL conditions. Read at module load, after headless.js applies overrides
// (same pattern as GAManager).
const EPISODE_MUT_PROB         = ExperimentParams.mut_prob;   // tunable (default 0.03)
const EPISODE_MUT_SIGMA        = ExperimentParams.mut_sigma;  // tunable (default 0.1)

// ─────────────────────────────────────────────────────────────────────────────

class PureRLManager {
    constructor(env, spawn_col, spawn_row) {
        this.env       = env;
        this.spawn_col = spawn_col;
        this.spawn_row = spawn_row;

        this.rl_enabled        = true;
        this.condition_label   = 'pure_rl';
        this.selection_percent = 0.2;  // used by Logger for top-N slice

        this.generation        = 0;
        this.current_map_index = 0;
        this.map_tick_count    = 0;
        this.tick_count        = 0;
        this.total_ticks       = 0;

        this.all_agents_this_window = [];
        this.living_agents          = new Set();
        this.peak_population        = 0;

        // Rolling buffer of active_weight snapshots from recently dead
        // organisms. Used for collapse respawn. Oldest entry dropped when
        // full. Stored as Float32Array copies (organism may be GC'd).
        this._dead_weight_buffer = [];
    }

    // ── Initial population spawn ──────────────────────────────────────────────

    spawnGeneration() {
        this.all_agents_this_window = [];
        this.living_agents          = new Set();
        this.tick_count             = 0;
        this.map_tick_count         = 0;
        this.current_map_index      = 0;
        this.peak_population        = 0;
        this.generation             = 1;
        this.env.organisms          = [];

        for (let i = 0; i < POPULATION_SIZE; i++) {
            // Founding population: random Xavier weights (buffer is empty)
            const org = this._createOrganism(null);
            if (!org) continue;
            this.env.addOrganism(org);
            this._trackAgent(org);
        }

        logger.logEvent('PureRL',
            `Founding population | ${POPULATION_SIZE} organisms | RL=true | ` +
            `between_episode_mutation=${BETWEEN_EPISODE_MUTATION}`);
    }

    // ── Agent registration ────────────────────────────────────────────────────

    /**
     * Called by AdvancedOrganism.reproduce() for naturally born children.
     *
     * Before registering the child, we sync the parent's active_weights →
     * genome_weights so the child inherits the learned state rather than
     * the original starting weights. This uses the existing inheritance
     * path in AdvancedOrganism without modifying it.
     *
     * The child then receives Gaussian mutation via _mutateGenome() as
     * normal — this is the within-episode mutation source.
     */
    registerAgent(agent) {
        // agent.parent is set by AdvancedOrganism.reproduce() before calling
        // registerAgent. Sync parent's active → genome so this child
        // inherits the learned state.
        if (agent.parent && agent.parent.brain) {
            agent.parent.brain.syncGenomeFromActive();
        }
        this._trackAgent(agent);
    }

    _trackAgent(agent) {
        this.all_agents_this_window.push(agent);
        this.living_agents.add(agent);
        if (this.living_agents.size > this.peak_population) {
            this.peak_population = this.living_agents.size;
        }
    }

    // ── Per-tick update ───────────────────────────────────────────────────────

    tick() {
        this.tick_count++;
        this.map_tick_count++;
        this.total_ticks++;

        // Remove dead agents, snapshot their active_weights into buffer
        const dead = [];
        for (const agent of this.living_agents) {
            if (!agent.living) dead.push(agent);
        }
        for (const agent of dead) {
            this.living_agents.delete(agent);
            this._bufferDeadWeights(agent);
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

    startNextMap() {
        this.map_tick_count = 0;
        logger.logEvent('PureRL',
            `Window ${this.generation} -> Map ` +
            `${this.current_map_index + 1}/${MAPS_PER_GEN}`);

        // Reset RL eligibility traces — each map is a fresh episode
        for (const org of this.living_agents) {
            if (org.brain && org.brain.resetTraces) {
                org.brain.resetTraces();
            }
        }

        this.env.organisms = [];
        for (const org of this.living_agents) {
            this.env.addOrganism(org);
        }
    }

    closeWindow() {
        const living_arr = Array.from(this.living_agents);

        // ── Random cap DOWN to POPULATION_SIZE (NOT fitness-based) ────────────
        // If reproduction pushed the population above the cap, remove the excess
        // randomly — random removal carries no fitness signal, so it is not
        // selection pressure.
        if (living_arr.length > POPULATION_SIZE) {
            // Fisher-Yates shuffle, remove the tail
            for (let i = living_arr.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [living_arr[i], living_arr[j]] = [living_arr[j], living_arr[i]];
            }
            const excess = living_arr.slice(POPULATION_SIZE);
            for (const org of excess) {
                // Buffer their weights before removing, so they can feed a future top-up.
                this._bufferDeadWeights(org);
                org.living = false;
                this.living_agents.delete(org);
            }
            logger.logEvent('PureRL',
                `Window ${this.generation} | random cap: removed ` +
                `${excess.length} organisms randomly ` +
                `(${living_arr.length} → ${this.living_agents.size})`);
        }

        // ── Between-episode mutation of survivors (optional) ──────────────────
        // Undirected Gaussian noise on active_weights of all survivors.
        // Synced to genome_weights so children next episode inherit the mutated
        // learned state. No fitness signal — not selection.
        if (BETWEEN_EPISODE_MUTATION) {
            for (const org of this.living_agents) {
                if (!org.brain) continue;
                this._mutateWeights(org.brain.active_weights);
                org.brain.genome_weights =
                    new Float32Array(org.brain.active_weights);
            }
        }

        // ── Log THIS window BEFORE topping up ─────────────────────────────────
        // logGeneration averages over all_agents_this_window (everyone who
        // actually lived this window). Top-up organisms are fresh, ~0-fitness
        // spawns for the NEXT window, so we must log first — otherwise they
        // would drag down this window's fitness and inflate its agent count.
        const all_sorted = [...this.all_agents_this_window]
            .sort((a, b) => b.getFitness() - a.getFitness());

        logger.logGeneration(this, all_sorted);

        // ── Top UP to POPULATION_SIZE from the buffer ─────────────────────────
        // Every window must START with a full POPULATION_SIZE so pure-RL is
        // comparable to the learning/evolution conditions, which found each
        // generation with exactly POPULATION_SIZE organisms. Survivors are kept;
        // the deficit is filled with organisms sampled RANDOMLY from the
        // dead-weight buffer and mutated around (see _respawnFromBuffer). A full
        // collapse (0 survivors) is just the deficit == POPULATION_SIZE case, so
        // it needs no special branch.
        const deficit = POPULATION_SIZE - this.living_agents.size;
        if (deficit > 0) {
            logger.logEvent('PureRL',
                `Window ${this.generation} | topping up ${deficit} from buffer ` +
                `(survivors=${this.living_agents.size}, ` +
                `buffer_size=${this._dead_weight_buffer.length})`);
            this._respawnFromBuffer(deficit);
        }

        logger.logEvent('PureRL',
            `Window ${this.generation} closed | ` +
            `living=${this.living_agents.size} | ` +
            `window_agents=${all_sorted.length} | ` +
            `best_fit=${all_sorted.length > 0
                ? all_sorted[0].getFitness().toFixed(2) : 'n/a'} | ` +
            `between_mutation=${BETWEEN_EPISODE_MUTATION}`);

        // ── Reset window counters — next window starts with the full roster ───
        // living_agents is now POPULATION_SIZE (survivors + top-ups), so
        // all_agents_this_window and peak_population reset to that full roster.
        this.generation++;
        this.tick_count          = 0;
        this.map_tick_count      = 0;
        this.current_map_index   = 0;
        this.peak_population     = this.living_agents.size;
        this.all_agents_this_window = Array.from(this.living_agents);
    }

    // ── Collapse respawn ──────────────────────────────────────────────────────

    /**
     * Respawn N organisms by randomly sampling from the dead weight buffer.
     * Each organism gets one randomly chosen buffer entry — preserving real
     * individual strategies rather than averaging (which can produce weight
     * vectors no organism ever actually had) — then MUTATED around with
     * undirected Gaussian noise (same kernel as the survivor between-episode
     * mutation: 5% per weight, σ=0.1). Mutating the seeds means the reseeded
     * episode is a diverse cloud AROUND the buffered strategies rather than
     * 100 exact clones drawn from ~COLLAPSE_BUFFER_SIZE vectors. No fitness
     * signal — this adds diversity, not selection.
     *
     * If the buffer is empty (very first episode collapsed before anyone
     * died, which is extremely unlikely), falls back to random Xavier (no
     * seed to mutate around).
     */
    _respawnFromBuffer(n) {
        for (let i = 0; i < n; i++) {
            let seed = null;
            if (this._dead_weight_buffer.length > 0) {
                const idx = Math.floor(
                    Math.random() * this._dead_weight_buffer.length);
                seed = new Float32Array(this._dead_weight_buffer[idx]);
                this._mutateWeights(seed);  // mutate around the learned weights
            }
            const org = this._createOrganism(seed);
            if (!org) continue;
            this.env.addOrganism(org);
            this._trackAgent(org);
        }
    }

    /**
     * Snapshot an organism's active_weights into the rolling buffer.
     * Oldest entry is dropped when the buffer exceeds COLLAPSE_BUFFER_SIZE.
     */
    _bufferDeadWeights(agent) {
        if (!agent.brain || !agent.brain.active_weights) return;
        this._dead_weight_buffer.push(
            new Float32Array(agent.brain.active_weights));
        if (this._dead_weight_buffer.length > COLLAPSE_BUFFER_SIZE) {
            this._dead_weight_buffer.shift();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Gaussian mutation on a weight array in-place — the inter-generation
     * kernel. Uses ExperimentParams mut_prob / mut_sigma (default 3% per-weight
     * probability, σ=0.1, clipped to [-1, 1]), identical to GAManager's
     * between-generation mutation so all conditions match.
     */
    _mutateWeights(weights) {
        if (!weights) return;
        for (let i = 0; i < weights.length; i++) {
            if (Math.random() < EPISODE_MUT_PROB) {
                const u1 = 1 - Math.random();
                const u2 = 1 - Math.random();
                const z  = Math.sqrt(-2 * Math.log(u1)) *
                           Math.cos(2 * Math.PI * u2);
                weights[i] += EPISODE_MUT_SIGMA * z;
                weights[i]  = Math.max(-1, Math.min(1, weights[i]));
            }
        }
    }

    _createOrganism(seed_weights) {
        const org = new AdvancedOrganism(
            0, 0, this.env,
            null,   // no parent — genome set below
            true,   // rl_enabled
            this    // manager ref so reproduce() calls registerAgent()
        );

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

        const spawn = this._findSpawnPosition(
            org, this.spawn_col, this.spawn_row);
        if (!spawn) return null;
        org.c = spawn[0];
        org.r = spawn[1];
        return org;
    }

    _getRandomSpawnPosition(
        spawn_col = this.spawn_col,
        spawn_row = this.spawn_row
    ) {
        const grid = this.env.grid_map;
        if (!grid) return [spawn_col, spawn_row];
        for (let i = 0; i < 50; i++) {
            const rx = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            const ry = (Math.random() * 2 - 1) * SPAWN_RADIUS;
            if (rx * rx + ry * ry <= SPAWN_RADIUS * SPAWN_RADIUS) {
                const col = Math.floor(spawn_col + rx);
                const row = Math.floor(spawn_row + ry);
                if (col >= 0 && col < grid.cols &&
                    row >= 0 && row < grid.rows) {
                    return [col, row];
                }
            }
        }
        return [spawn_col, spawn_row];
    }

    _findSpawnPosition(org, spawn_col, spawn_row, attempts = 200) {
        for (let i = 0; i < attempts; i++) {
            const [col, row] = this._getRandomSpawnPosition(
                spawn_col, spawn_row);
            if (org.isClear(col, row, org.rotation)) return [col, row];
        }
        return null;
    }
}

PureRLManager.POPULATION_SIZE = POPULATION_SIZE;
PureRLManager.TICKS_PER_GEN   = TICKS_PER_GEN;

module.exports = PureRLManager;