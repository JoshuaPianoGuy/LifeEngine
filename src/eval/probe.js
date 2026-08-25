'use strict';

/**
 * src/eval/probe.js — Stage 1 of the fitness-landscape pipeline.
 *
 * The probe harness. It measures f(θ): the fitness of ONE fixed genome θ,
 * evaluated as a monomorphic founding population of `population_size` clones,
 * with the genetic algorithm fully disabled, on a single seeded map.
 *
 *   probe.evaluate({ genome, mapIndex, ticks }) -> { mean_fitness, ... }
 *
 * WHY the constraints (see the plan for the full rationale):
 *   - GA fully disabled. We never call GAManager.evolve() or a second
 *     spawnGeneration(), so selection / crossover / between-generation mutation
 *     never run. Each probe measures the coordinate θ itself, not its
 *     descendants a few hundred ticks downstream.
 *   - reproduction_success_rate = 0 (set by the caller via ExperimentParams).
 *     Within-generation asexual children never spawn, so the cohort stays a
 *     fixed `population_size` clones of θ. This is density-matching: fitness is
 *     density-dependent (clones compete for the same food + predators), so a
 *     100-clone cohort measures "fitness of a monomorphic population of θ" —
 *     an adaptive-dynamics construct — at the same founder density a real
 *     generation used. (Reproduction costs no energy in this sim, so 0 vs 0.8
 *     is energy-neutral; it only controls whether the population grows.)
 *   - RL flag. rl_enabled is fixed at env construction (GAManager reads it).
 *     Build one Probe per RL flag; reuse it across genomes and map indices.
 *   - One estimate per run. The 100 clones share a world, so their fitnesses
 *     are correlated — NOT independent samples. The unit of replication is the
 *     RUN (one call to evaluate). mean_fitness is ONE Monte-Carlo estimate;
 *     average many runs (different map indices) for a landscape value, and take
 *     the spread ACROSS RUNS as the error bar — never std-across-clones/√N.
 *
 * This module requires the simulation modules at load, so the caller MUST apply
 * all ExperimentParams / WorldConfig / PredatorHyperparameters overrides BEFORE
 * requiring this file (exactly as src/headless.js does), or the network size,
 * predator counts, reproduction rate, etc. will bake in at their defaults.
 */

const WorldConfig            = require('../WorldConfig');
const NNBrain                = require('../Organism/Perception/NNBrain');
const GenerationConstants    = require('../Organism/GenerationConstants');

// ── quiet-console helper ──────────────────────────────────────────────────────
// The sim modules log per-map / per-predator-pool lines. A landscape sweep runs
// thousands of probes; silence stdout during the hot loop, restore it after.
let _saved_log = null;
function _muteConsole() {
    if (_saved_log) return;
    _saved_log = console.log;
    console.log = () => {};
}
function _unmuteConsole() {
    if (!_saved_log) return;
    console.log = _saved_log;
    _saved_log = null;
}

class Probe {
    /**
     * @param {boolean} rlEnabled  true = RL-on pass (active_weights drift via
     *                             REINFORCE); false = RL-off pass
     *                             (active_weights === genome_weights, no drift).
     * @param {object}  [opts]
     * @param {number}  [opts.cellSize=2]
     */
    constructor(rlEnabled, opts = {}) {
        this.rlEnabled = !!rlEnabled;
        this.cellSize  = opts.cellSize != null ? opts.cellSize : 2;

        // GAManager reads rl_enabled from WorldConfig.learning_enabled at env
        // construction. Set it before building the env.
        WorldConfig.headless         = true;
        WorldConfig.auto_pause       = false;
        WorldConfig.auto_reset       = false;
        WorldConfig.learning_enabled = this.rlEnabled;
        WorldConfig.experiment_mode  = 'standard';   // GAManager, not frozen/pure_rl

        // Require WorldEnvironment lazily so the caller's overrides (applied
        // before this file was required) are already in effect module-wide.
        const WorldEnvironment = require('../Environments/WorldEnvironment');
        const engine = { fps: Infinity, last_fps: Infinity, running: true,
            stop() { this.running = false; }, start() {}, restart() {} };

        _muteConsole();
        try {
            this.env = new WorldEnvironment(engine, this.cellSize);
        } finally {
            _unmuteConsole();
        }
        this.ga = this.env.ga_manager;

        // Disable the generation/map lifecycle. With tick() pinned to null,
        // env.update() runs the per-tick simulation (organism updates, predator
        // movement, periodic food respawn) but NEVER fires NEXT_MAP /
        // NEXT_GENERATION — so generateWorld/startNextMap/evolve/spawnGeneration
        // are never triggered mid-run. The probe drives exactly one map of a
        // fixed length itself. (living_agents bookkeeping is unused by the probe.)
        this.ga.tick = () => null;

        this.ticksPerMap = GenerationConstants.TICKS_PER_MAP;
        this.popSize     = this.ga.founders ? 0 : 0; // set per-run from founders
    }

    /**
     * Evaluate one genome on one map for one run.
     *
     * @param {Float32Array|number[]} genome  θ, length NNBrain.GENOME_SIZE (3910)
     * @param {object} opts
     * @param {number} opts.mapIndex  which map in the seed's pool to load
     * @param {number} [opts.ticks]   ticks to run (default TICKS_PER_MAP=2000)
     * @returns {object} run result (see below)
     */
    evaluate(genome, opts = {}) {
        if (genome.length !== NNBrain.GENOME_SIZE) {
            throw new Error(`probe.evaluate: genome length ${genome.length} != GENOME_SIZE ${NNBrain.GENOME_SIZE}`);
        }
        const mapIndex = opts.mapIndex | 0;
        const ticks    = opts.ticks != null ? (opts.ticks | 0) : this.ticksPerMap;

        _muteConsole();
        try {
            const env = this.env;
            const ga  = this.ga;

            // Independent run: reset the world clock so day/night phase and the
            // 1200-tick food-respawn cadence start identically every probe
            // (terrain + unseeded dynamics are the only things allowed to vary).
            env.total_ticks = 0;

            // Load exactly the requested map from the seed's pool.
            env._map_sequence  = [mapIndex];
            env._sequence_index = 0;
            env.generateWorld();

            // Seed the founding cohort with population_size clones of θ. gene_pool
            // is what spawnGeneration reads per founder; every slot is a fresh
            // copy so RL drift in one clone can't alias into another.
            const N = require('../ExperimentParams').population_size;
            const pool = new Array(N);
            for (let i = 0; i < N; i++) pool[i] = new Float32Array(genome);
            ga.gene_pool = pool;
            ga.spawnGeneration();

            // Fresh predators for this run (spawnAll clears the old pool). Their
            // wandering is unseeded — the Monte-Carlo noise R repeats average out.
            env.predator_manager.spawnAll();

            // Capture the founders now: they persist in ga.founders even after
            // death (dead agents are never removed from that list), carrying
            // their final cumulative_food_score. reproduction_success_rate=0
            // means no descendants join, so these N objects are the whole cohort.
            const founders = ga.founders.slice();

            // Run one map. env.update() ticks organisms + predators + food.
            for (let t = 0; t < ticks; t++) env.update();

            return this._collect(founders, mapIndex, ticks);
        } finally {
            _unmuteConsole();
        }
    }

    _collect(founders, mapIndex, ticks) {
        const n = founders.length;
        const fitness = new Array(n);
        let sum = 0, sum2 = 0;
        let lifeSum = 0, caveSum = 0, drainSum = 0, touchSum = 0;
        let driftSum = 0, driftN = 0;
        const deaths = { drained: 0, starved: 0, lifespan: 0, survived: 0 };
        for (let i = 0; i < n; i++) {
            const a = founders[i];
            // How far in-lifetime learning moved this organism's weights away
            // from the genome it inherited: mean |active - genome| over the
            // whole weight vector, at end of life.
            //
            // This is the same quantity assertRlOffInvariant() checks, promoted
            // from an assertion to a measurement. With RL OFF it is exactly 0
            // (active_weights IS genome_weights, same array reference), so the
            // column doubles as a per-row proof that the RL-off pass really had
            // learning disabled. With RL ON it is the size of the lifetime
            // adjustment — which is what makes it worth collecting for
            // assimilation: a genome that already encodes the behaviour needs
            // LESS adjustment, so this should fall over evolutionary time even
            // as fitness rises.
            const br = a.brain;
            if (br && br.genome_weights && br.active_weights) {
                const g = br.genome_weights, w = br.active_weights;
                let d = 0;
                for (let k = 0; k < g.length; k++) d += Math.abs(w[k] - g[k]);
                driftSum += d / g.length;
                driftN++;
            }
            const f = a.getFitness();
            fitness[i] = f;
            sum += f; sum2 += f * f;
            lifeSum  += a.lifetime || 0;
            caveSum  += a.cave_entry_count || 0;
            drainSum += a.drained_ticks || 0;
            touchSum += a.predator_touch_count || 0;
            const cause = a.living ? 'survived' : (a.death_cause || 'starved');
            if (deaths[cause] != null) deaths[cause]++; else deaths.survived++;
        }
        const mean = n > 0 ? sum / n : 0;
        // Std ACROSS CLONES within this one run — reported as a within-run
        // dispersion diagnostic ONLY (σ_evo/σ_learn in the plots). It is NOT a
        // standard error for mean_fitness: the clones are correlated, so the
        // real error bar comes from the spread across independent runs.
        const varClones = n > 1 ? Math.max(0, sum2 / n - mean * mean) : 0;
        return {
            mean_fitness:  mean,
            std_clones:    Math.sqrt(varClones),
            n_clones:      n,
            map_index:     mapIndex,
            ticks,
            rl_enabled:    this.rlEnabled,
            mean_lifetime: n > 0 ? lifeSum / n : 0,
            mean_cave_entries: n > 0 ? caveSum / n : 0,
            mean_drained_ticks: n > 0 ? drainSum / n : 0,
            mean_predator_touches: n > 0 ? touchSum / n : 0,
            // Exactly 0 on an RL-off probe, by the aliasing invariant above.
            mean_weight_drift: driftN > 0 ? driftSum / driftN : 0,
            deaths,
            fitness,       // per-clone, for callers that want the full vector
        };
    }

    /**
     * ASSERT A (must hold whenever rlEnabled === false):
     *   With RL off, active_weights and genome_weights are the SAME array
     *   reference, so mean absolute weight difference = mean(|active − genome|) is exactly 0 for every
     *   organism at all times. A non-zero value means the two are aliased apart
     *   somewhere and the "RL-off" pass is silently running RL.
     *
     * Call after an evaluate() with rlEnabled=false. Returns {mad, aliased_ok}.
     * Throws if the invariant is violated.
     */
    assertRlOffInvariant() {
        if (this.rlEnabled) {
            throw new Error('assertRlOffInvariant called on an RL-ON probe');
        }
        const founders = this.ga.founders;
        let maxMad = 0;
        let aliasOk = true;
        for (const a of founders) {
            const b = a.brain;
            if (!b || !b.genome_weights || !b.active_weights) continue;
            if (b.active_weights !== b.genome_weights) aliasOk = false;
            const g = b.genome_weights, w = b.active_weights;
            let s = 0;
            for (let i = 0; i < g.length; i++) s += Math.abs(w[i] - g[i]);
            const mad = s / g.length;
            if (mad > maxMad) maxMad = mad;
        }
        if (maxMad !== 0 || !aliasOk) {
            throw new Error(
                `RL-off invariant VIOLATED: max mean absolute weight difference=${maxMad}, ` +
                `active===genome for all founders: ${aliasOk}. ` +
                `active_weights is not aliased to genome_weights — RL is leaking into the RL-off pass.`);
        }
        return { mad: maxMad, aliased_ok: aliasOk };
    }
}

module.exports = { Probe };
