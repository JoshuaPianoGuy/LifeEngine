/**
 * Logger.js
 *
 * Structured experiment logger for the learning-vs-evolution experiment.
 *
 * Records three tiers of data:
 *
 *   1. Generation summary  — one row per generation, written to generation_log
 *      Fields: generation, condition, ticks, total_agents, peak_pop,
 *              avg_energy_early, avg_energy_end, top5_fitness, best_fitness,
 *              avg_lifetime, avg_learned_weight_diff, genome_variance
 *
 *   2. Top-5 organism detail — one row per top organism per generation
 *      Fields: generation, rank, fitness, lifetime, energy_at_death,
 *              energy_at_early_sample, foods_eaten, learned_weight_diff,
 *              network_weight_magnitude, day_night_cycle_count
 *
 *   3. Event log — timestamped freeform events (deaths, reproductions, etc.)
 *      Used for real-time console output and debugging.
 *
 * Download methods produce separate CSV files for each tier so they can be
 * analysed independently in R / Python / Excel.
 *
 * Usage:
 *   const logger = require('./Logger');
 *   logger.logGeneration(ga_manager, sorted_agents);
 *   logger.logEvent('GA', 'Generation ended', { gen: 5 });
 *   logger.downloadAll();          // downloads all three CSV files
 *   logger.downloadGenerations();  // just the generation summary
 */

'use strict';

class Logger {
    constructor() {
        this.generation_log = [];   // one entry per generation
        this.organism_log   = [];   // one entry per top-5 organism per generation
        this.event_log      = [];   // freeform timestamped events
        this._start_time    = Date.now();
    }

    // ── Generation summary ────────────────────────────────────────────────────

    /**
     * Record a full generation summary.
     * Call from GAManager._recordMetrics() after sorting agents by fitness.
     *
     * @param {object}   ga       GAManager instance
     * @param {Array}    sorted   agents sorted by fitness descending
     */
    logGeneration(ga, sorted) {
        if (!sorted || sorted.length === 0) return;

        const n          = sorted.length;
        const n_parents  = Math.min(5, n);
        const top5       = sorted.slice(0, n_parents);

        // Average energy at tick 1000 (fixed early-game sample)
        const early_samples = sorted.map(a => a.energy_at_early_sample).filter(e => e !== null && e !== undefined);
        const avg_energy_early = early_samples.length > 0
            ? early_samples.reduce((s, e) => s + e, 0) / early_samples.length
            : 0;

        // Average energy at end of life
        const avg_energy_end = sorted.reduce((s, a) => s + (a.energy || 0), 0) / n;

        // Top-5 fitness average
        const top5_fitness = top5.reduce((s, a) => s + a.getFitness(), 0) / top5.length;

        // Average lifetime
        const avg_lifetime = sorted.reduce((s, a) => s + (a.lifetime || 0), 0) / n;

        // Learned weight difference: mean absolute deviation between active_weights and
        // genome_weights across all top-5 agents. Only meaningful in Condition A (RL enabled).
        // Measures how much within-lifetime learning has modified the starting weights.
        const drifts = top5.map(a => this._calcDrift(a)).filter(d => d !== null);
        const avg_learned_weight_diff = drifts.length > 0
            ? drifts.reduce((s, d) => s + d, 0) / drifts.length
            : 0;

        // Genome variance across all agents — measures population diversity.
        // Computed as mean variance per weight position across all genomes.
        const genome_variance = this._calcGenomeVariance(sorted);

        // Mean network weight magnitude (RMS - Root Mean Square) across all agents.
        // Normalized by dividing L2 norm by sqrt(num_weights) to show typical weight magnitude.
        // With [-1, 1] bounds, this typically ranges [0, 1].
        const avg_network_weight_mag = sorted.reduce((s, a) => s + this._calcRMSWeight(a), 0) / n;

        const entry = {
            generation:              ga.generation,
            condition:               ga.rl_enabled ? 'learning' : 'natural_selection',
            wall_clock_s:            ((Date.now() - this._start_time) / 1000).toFixed(1),
            generation_ticks:        ga.tick_count,
            total_agents:            n,
            peak_population:         ga.peak_population,
            avg_energy_at_tick_1000: avg_energy_early.toFixed(3),
            avg_energy_end:          avg_energy_end.toFixed(3),
            top5_fitness:            top5_fitness.toFixed(4),
            best_fitness:            sorted[0].getFitness().toFixed(4),
            avg_lifetime:            avg_lifetime.toFixed(1),
            avg_learned_weight_diff: avg_learned_weight_diff.toFixed(6),
            genome_variance:         genome_variance.toFixed(6),
            avg_network_weight_mag:  avg_network_weight_mag.toFixed(4),
        };

        this.generation_log.push(entry);

        // Log top-5 organisms individually
        for (let i = 0; i < top5.length; i++) {
            this.logOrganism(ga.generation, i + 1, top5[i], ga.rl_enabled);
        }

        // Console summary
        console.log(
            `[Gen ${ga.generation}] ${ga.rl_enabled ? 'LEARN' : 'NSEL'} | ` +
            `ticks=${ga.tick_count} agents=${n} peak=${ga.peak_population} | ` +
            `E@tick1000=${entry.avg_energy_at_tick_1000} E_end=${entry.avg_energy_end} | ` +
            `top5_fit=${entry.top5_fitness} best=${entry.best_fitness} | ` +
            `learned_diff=${entry.avg_learned_weight_diff} var=${entry.genome_variance}`
        );
    }

    // ── Top-organism detail ───────────────────────────────────────────────────

    /**
     * Record detailed stats for one organism.
     * Called automatically by logGeneration() for the top-5.
     */
    logOrganism(generation, rank, agent, rl_enabled) {
        const learned_weight_diff = this._calcDrift(agent);
        const network_weight_mag  = this._calcRMSWeight(agent);

        // Count food eaten by type if tracked
        const food_counts = agent.food_by_type || {};

        const entry = {
            generation,
            rank,
            condition:                rl_enabled ? 'learning' : 'natural_selection',
            fitness:                  agent.getFitness().toFixed(4),
            lifetime:                 agent.lifetime || 0,
            energy_at_death:          (agent.energy || 0).toFixed(2),
            energy_at_tick_1000:      agent.energy_at_early_sample !== null && agent.energy_at_early_sample !== undefined
                ? agent.energy_at_early_sample.toFixed(2)
                : 'n/a',
            cumulative_food_score:    (agent.cumulative_food_score || 0).toFixed(4),
            food_low:                 food_counts['low food']      || 0,
            food_medium:              food_counts['medium food']   || 0,
            food_prestige:            food_counts['prestige food'] || 0,
            food_default:             food_counts['food']          || 0,
            cells_visited:            agent.visited_cells ? agent.visited_cells.size : 0,
            learned_weight_diff:      learned_weight_diff !== null ? learned_weight_diff.toFixed(6) : 'n/a',
            network_weight_magnitude: network_weight_mag.toFixed(4),
        };

        this.organism_log.push(entry);
    }

    // ── Freeform event log ────────────────────────────────────────────────────

    /**
     * Log a freeform event. Also prints to console.
     * @param {string} source   module name
     * @param {string} message
     * @param {object} data     optional extra data (will be JSON-stringified in CSV)
     */
    logEvent(source, message, data = null) {
        const entry = {
            timestamp_ms: Date.now() - this._start_time,
            source,
            message,
            data: data ? JSON.stringify(data) : '',
        };
        this.event_log.push(entry);
        console.log(`[${source}] ${message}`, data || '');
    }

    // ── Genetic analysis helpers ──────────────────────────────────────────────

    /**
     * Mean absolute difference between active_weights and genome_weights.
     * Measures within-lifetime learning-induced weight changes.
     * Returns null if RL is disabled (no learning) or brain unavailable.
     */
    _calcDrift(agent) {
        if (!agent.brain || !agent.brain.active_weights || !agent.brain.genome_weights) return null;
        if (agent.brain.active_weights === agent.brain.genome_weights) return 0;  // Condition B
        const aw = agent.brain.active_weights;
        const gw = agent.brain.genome_weights;
        let total = 0;
        for (let i = 0; i < gw.length; i++) total += Math.abs(aw[i] - gw[i]);
        return total / gw.length;
    }

    /**
     * RMS (Root Mean Square) weight magnitude of genome_weights.
     * Normalized L2 norm: sqrt(sum of squares) / sqrt(num_weights).
     * Shows typical/average weight magnitude. With [-1, 1] bounds, typically [0, 1].
     */
    _calcRMSWeight(agent) {
        if (!agent.brain || !agent.brain.genome_weights) return 0;
        const gw = agent.brain.genome_weights;
        let sum = 0;
        for (let i = 0; i < gw.length; i++) sum += gw[i] * gw[i];
        const l2_norm = Math.sqrt(sum);
        const rms = l2_norm / Math.sqrt(gw.length);
        return rms;
    }

    /**
     * Raw L2 norm (Euclidean magnitude) of genome_weights (for reference).
     * Returns the un-normalized L2 norm.
     */
    _calcL2Norm(agent) {
        if (!agent.brain || !agent.brain.genome_weights) return 0;
        const gw = agent.brain.genome_weights;
        let sum = 0;
        for (let i = 0; i < gw.length; i++) sum += gw[i] * gw[i];
        return Math.sqrt(sum);
    }

    /**
     * Mean per-position variance across all genomes in the population.
     * High variance = diverse population. Low = converged.
     * Expensive for large populations — samples up to 50 agents.
     */
    _calcGenomeVariance(agents) {
        const sample = agents.length > 50 ? agents.slice(0, 50) : agents;
        const genomes = sample
            .filter(a => a.brain && a.brain.genome_weights)
            .map(a => a.brain.genome_weights);
        if (genomes.length < 2) return 0;

        const len = genomes[0].length;
        let total_var = 0;
        for (let i = 0; i < len; i++) {
            const vals = genomes.map(g => g[i]);
            const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
            const var_ = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
            total_var += var_;
        }
        return total_var / len;
    }

    // ── CSV export ────────────────────────────────────────────────────────────

    _toCSV(rows) {
        if (rows.length === 0) return 'no data';
        const headers = Object.keys(rows[0]);
        const lines   = rows.map(row =>
            headers.map(h => {
                const v = row[h] === null || row[h] === undefined ? '' : String(row[h]);
                return v.includes(',') || v.includes('"') || v.includes('\n')
                    ? `"${v.replace(/"/g, '""')}"`
                    : v;
            }).join(',')
        );
        return headers.join(',') + '\n' + lines.join('\n');
    }

    generationsCSV()  { return this._toCSV(this.generation_log); }
    organismsCSV()    { return this._toCSV(this.organism_log); }
    eventsCSV()       { return this._toCSV(this.event_log); }

    // ── Download helpers ──────────────────────────────────────────────────────

    _download(content, filename) {
        const blob = new Blob([content], { type: 'text/csv' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    /** Download the generation summary CSV */
    downloadGenerations(filename) {
        const date = new Date().toISOString().split('T')[0];
        this._download(this.generationsCSV(), filename || `generations_${date}.csv`);
    }

    /** Download the per-organism detail CSV */
    downloadOrganisms(filename) {
        const date = new Date().toISOString().split('T')[0];
        this._download(this.organismsCSV(), filename || `organisms_${date}.csv`);
    }

    /** Download the freeform event log CSV */
    downloadEvents(filename) {
        const date = new Date().toISOString().split('T')[0];
        this._download(this.eventsCSV(), filename || `events_${date}.csv`);
    }

    /**
     * Download all three CSVs in sequence.
     * Small delay between each so browsers don't block multiple downloads.
     */
    downloadAll(prefix) {
        const date = new Date().toISOString().split('T')[0];
        const p    = prefix ? `${prefix}_` : '';
        this.downloadGenerations(`${p}generations_${date}.csv`);
        setTimeout(() => this.downloadOrganisms(`${p}organisms_${date}.csv`),   400);
        setTimeout(() => this.downloadEvents(`${p}events_${date}.csv`),         800);
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    clear() {
        this.generation_log = [];
        this.organism_log   = [];
        this.event_log      = [];
        this._start_time    = Date.now();
    }

    summary() {
        return {
            generations_recorded: this.generation_log.length,
            organisms_recorded:   this.organism_log.length,
            events_recorded:      this.event_log.length,
            elapsed_s:            ((Date.now() - this._start_time) / 1000).toFixed(1),
        };
    }
}

const logger = new Logger();
module.exports = logger;