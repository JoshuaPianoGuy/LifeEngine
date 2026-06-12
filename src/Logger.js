/**
 * Logger.js
 *
 * Structured experiment logger for the learning-vs-evolution experiment.
 *
 * Records three tiers of data:
 *
 *   1. Generation summary  — one row per generation, written to generation_log
 *      Fields: generation, condition, ticks, total_agents, peak_pop,
 *              avg_energy_early, avg_energy_end, top20percent_fitness, best_fitness,
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

const NNBrain = require('./Organism/Perception/NNBrain');
const WorldConfig = require('./WorldConfig');

let fs = null;
let path = null;
const isNodeRuntime = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
if (isNodeRuntime) {
    try {
        const req = eval('require');
        fs = req('fs');
        path = req('path');
    } catch (err) {
        fs = null;
        path = null;
    }
}

const WEIGHT_SNAPSHOT_PRECISION = 6;

// Fixed cap on organism rows logged per generation.
// Keeps organisms.csv size predictable regardless of population growth.
// 20 organisms is sufficient for PCA centroids, t-SNE, and MAD analysis.
// Increase if within-generation weight diversity analysis is needed.
const MAX_ORGANISM_LOG_PER_GEN = 20;
const LOG_W1_SNAPSHOT = true;
const LOG_ACTIVE_SNAPSHOT = true;
const LOG_FULL_GENOME_SNAPSHOT = false;

// Maximum number of organism rows to hold in RAM between flushes.
// At 100 organisms/gen × AUTO_SAVE_EVERY_N_GENS=5 = 500 rows per flush cycle.
// Cap at 2000 as a safety backstop (e.g. if the server is unreachable).
const MAX_ORGANISM_LOG_ROWS = 2000;

const AUTO_SAVE_ENABLED = true;
const AUTO_SAVE_EVERY_N_GENS = 5;
const AUTO_SAVE_DIR = 'logs';
const AUTO_SAVE_RUN_FOLDER = 'auto-run';
const AUTO_SAVE_APPEND = true;
const AUTO_DOWNLOAD_ENABLED = false;
const AUTO_DOWNLOAD_EVERY_TICKS = 50000;
const LOG_SERVER_ENABLED = true;
const LOG_SERVER_ENDPOINT = '/api/logs/append';

class Logger {
    constructor() {
        this.generation_log = [];   // one entry per generation
        this.organism_log   = [];   // one entry per top-5 organism per generation
        this.event_log      = [];   // freeform timestamped events
        this._start_time    = Date.now();
        this._auto_save_enabled = AUTO_SAVE_ENABLED;
        this._auto_save_fs_enabled = AUTO_SAVE_ENABLED && !!fs && !!path;
        // Save every window for tick-based conditions; every N gens otherwise.
        const tick_based = WorldConfig.experiment_mode === 'frozen_pg' ||
                           WorldConfig.experiment_mode === 'pure_rl';
        this._auto_save_every_n = tick_based ? 1 : AUTO_SAVE_EVERY_N_GENS;
        this._auto_save_dir = AUTO_SAVE_DIR;
        this._auto_save_tag = `autosave_${Date.now()}`;
        this._auto_save_run_dir = null;
        this._auto_save_append = AUTO_SAVE_APPEND;
        this._last_saved_gen_idx = 0;
        this._last_saved_org_idx = 0;
        this._last_saved_evt_idx = 0;
        this._auto_download_enabled = AUTO_DOWNLOAD_ENABLED;
        this._auto_download_every_ticks = AUTO_DOWNLOAD_EVERY_TICKS;
        this._last_auto_download_tick = 0;
        this._server_enabled = LOG_SERVER_ENABLED;
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
        // True top-20% — used for all population-level metrics in generations.csv
        // (top20percent_fitness, learned weight diff, weight variance).
        // Always reflects the real selection cohort regardless of population size.
        const selection_pct = (typeof ga.selection_percent === 'number')
            ? ga.selection_percent : 0.2;
        const num_top_pct  = Math.max(1, Math.floor(n * selection_pct));
        const top20pct     = sorted.slice(0, num_top_pct);

        // Capped slice for organism weight logging — fixed at MAX_ORGANISM_LOG_PER_GEN
        // to keep organisms.csv size predictable as population grows.
        // Only used for the per-organism rows written to organisms.csv.
        const top20pct_log = sorted.slice(0, Math.min(n, MAX_ORGANISM_LOG_PER_GEN));

        // Average energy at tick 1000 (fixed early-game sample)
        const early_samples = sorted.map(a => a.energy_at_early_sample).filter(e => e !== null && e !== undefined);
        const avg_energy_early = early_samples.length > 0
            ? early_samples.reduce((s, e) => s + e, 0) / early_samples.length
            : 0;

        // Average energy at end of life
        const avg_energy_end = sorted.reduce((s, a) => s + (a.energy || 0), 0) / n;

        // Top 20% fitness average
        const top20percent_fitness = top20pct.reduce((s, a) => s + a.getFitness(), 0) / top20pct.length;

        // Overall population average fitness
        const avg_fitness = sorted.reduce((s, a) => s + a.getFitness(), 0) / n;

        // Average lifetime
        const avg_lifetime = sorted.reduce((s, a) => s + (a.lifetime || 0), 0) / n;

        // Learned weight difference: mean absolute deviation between active_weights and
        // genome_weights across all top 20% agents. Only meaningful in Condition A (RL enabled).
        // Measures how much within-lifetime learning has modified the starting weights.
        const drifts = top20pct.map(a => this._calcDrift(a)).filter(d => d !== null);
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

        // How weights change between generations: difference from previous generation RMS weight
        let inter_gen_weight_change = 0;
        if (this.generation_log.length > 0) {
            const prev_mag = parseFloat(this.generation_log[this.generation_log.length - 1].avg_network_weight_mag);
            inter_gen_weight_change = avg_network_weight_mag - prev_mag;
        }

        const entry = {
            generation:              ga.generation,
            condition:               ga.condition_label || (ga.rl_enabled ? 'learning' : 'natural_selection'),
            wall_clock_s:            ((Date.now() - this._start_time) / 1000).toFixed(1),
            generation_ticks:        ga.tick_count,
            total_agents:            n,
            peak_population:         ga.peak_population,
            avg_energy_at_tick_1000: avg_energy_early.toFixed(3),
            avg_energy_end:          avg_energy_end.toFixed(3),
            avg_fitness:             avg_fitness.toFixed(4),
            top20percent_fitness:    top20percent_fitness.toFixed(4),
            best_fitness:            sorted[0].getFitness().toFixed(4),
            avg_lifetime:            avg_lifetime.toFixed(1),
            avg_learned_weight_diff: avg_learned_weight_diff.toFixed(6),
            inter_gen_weight_change: inter_gen_weight_change.toFixed(6),
            genome_variance:         genome_variance.toFixed(6),
            avg_network_weight_mag:  avg_network_weight_mag.toFixed(4),
        };

        this.generation_log.push(entry);

        // Track in FossilRecord for UI Charts
        const FossilRecord = require('./Stats/FossilRecord');
        FossilRecord.updateGenData(
            ga.generation, 
            ga.peak_population, 
            n, // total agents
            avg_learned_weight_diff, 
            inter_gen_weight_change, 
            genome_variance, 
            top20percent_fitness, 
            avg_fitness
        );

        // Log top-20% organisms — full population with weight snapshots is ~20 MB/flush
        for (let i = 0; i < top20pct_log.length; i++) {
            this.logOrganism(ga.generation, i + 1, top20pct_log[i], ga.rl_enabled);
        }

        this._autoSaveIfNeeded(ga.generation, ga.rl_enabled);
        this._autoDownloadIfNeeded(ga);

        // Console summary
        console.log(
            `[Gen ${ga.generation}] ${ga.rl_enabled ? 'LEARN' : 'NSEL'} | ` +
            `ticks=${ga.tick_count} agents=${n} peak=${ga.peak_population} | ` +
            `E@tick1000=${entry.avg_energy_at_tick_1000} E_end=${entry.avg_energy_end} | ` +
            `top20pct_fit=${entry.top20percent_fitness} best=${entry.best_fitness} | ` +
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
            condition:                (agent && agent.ga_manager && agent.ga_manager.condition_label)
                ? agent.ga_manager.condition_label
                : (rl_enabled ? 'learning' : 'natural_selection'),
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
            w1_weights:               LOG_W1_SNAPSHOT ? this._snapshotW1(agent) : '',
            active_weights:           LOG_ACTIVE_SNAPSHOT ? this._snapshotActive(agent) : '',
            genome_weights:           LOG_FULL_GENOME_SNAPSHOT ? this._snapshotGenome(agent) : '',
        };

        this.organism_log.push(entry);

        // Guard against unbounded growth between flushes (e.g. if the server
        // is unreachable and auto-save has silently disabled itself).
        if (this.organism_log.length > MAX_ORGANISM_LOG_ROWS) {
            // Drop the oldest half so we keep recent generations.
            const keep = Math.floor(MAX_ORGANISM_LOG_ROWS / 2);
            this.organism_log = this.organism_log.slice(-keep);
            this._last_saved_org_idx = 0;  // treat remainder as unsaved
            console.warn(`[Logger] organism_log trimmed to ${keep} rows (cap=${MAX_ORGANISM_LOG_ROWS})`);
        }
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

    _formatWeights(weights) {
        if (!weights || weights.length === 0) return '';
        return Array.from(weights)
            .map(v => Number(v).toFixed(WEIGHT_SNAPSHOT_PRECISION))
            .join(' ');
    }

    // ── Compact binary weight encoding ────────────────────────────────────────
    //
    // Space breakdown per organism row (GENOME_SIZE=1702, W1_SIZE=1472):
    //   Old text at 6 d.p.:  ~30 KB/row
    //   Float32 + Base64:    ~17 KB/row  (-43%)
    //   Int8   + Base64:      ~5 KB/row  (-85%, ~0.008 resolution, fine for genomes)
    //
    // w1_weights  → Int8+Base64:    genome weights don't need sub-0.01 precision
    // active_weights → Float32+Base64: preserves RL-learned drift (~0.009 mean)
    //
    // Decoding in Python:
    //   import base64, numpy as np
    //   w1 = np.frombuffer(base64.b64decode(field), dtype=np.int8).astype(np.float32) / 127
    //   aw = np.frombuffer(base64.b64decode(field), dtype=np.float32)

    _encodeFloat32B64(weights) {
        // Encode a Float32Array (or any array-like) to a Base64 string of its
        // raw IEEE-754 bytes.  Preserves full float32 precision.
        if (!weights || weights.length === 0) return '';
        const f32 = weights instanceof Float32Array ? weights : new Float32Array(weights);
        // In Node (no btoa): use Buffer. In browser: use Uint8Array + btoa.
        if (typeof Buffer !== 'undefined') {
            return Buffer.from(f32.buffer).toString('base64');
        }
        const bytes = new Uint8Array(f32.buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        return btoa(binary);
    }

    _encodeInt8B64(weights) {
        // Quantise weights (expected range [-1, 1]) to Int8 [-127, 127], then
        // Base64-encode the raw bytes.  Resolution: 1/127 ≈ 0.008.
        if (!weights || weights.length === 0) return '';
        const i8 = new Int8Array(weights.length);
        for (let i = 0; i < weights.length; i++) {
            // Clamp to [-1, 1] then scale; round to nearest integer.
            const v = Math.max(-1, Math.min(1, weights[i]));
            i8[i] = Math.round(v * 127);
        }
        if (typeof Buffer !== 'undefined') {
            return Buffer.from(i8.buffer).toString('base64');
        }
        const bytes = new Uint8Array(i8.buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        return btoa(binary);
    }

    _snapshotW1(agent) {
        // W1 layer of genome — Int8+Base64 (~2 KB vs ~14 KB text).
        if (!agent.brain || !agent.brain.genome_weights) return '';
        const gw = agent.brain.genome_weights;
        const w1_size = NNBrain.STATE_SIZE * NNBrain.HIDDEN_SIZE;
        if (gw.length < w1_size) return '';
        return this._encodeInt8B64(gw.subarray ? gw.subarray(0, w1_size) : Array.from(gw).slice(0, w1_size));
    }

    _snapshotActive(agent) {
        // Full active_weights — Float32+Base64 (~9 KB vs ~16 KB text).
        // Must preserve float32 precision to capture RL weight drift (~0.009 mean).
        if (!agent.brain || !agent.brain.active_weights) return '';
        return this._encodeFloat32B64(agent.brain.active_weights);
    }

    _snapshotGenome(agent) {
        if (!agent.brain || !agent.brain.genome_weights) return '';
        return this._encodeFloat32B64(agent.brain.genome_weights);
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

    // ── Auto-save (Node.js only) ────────────────────────────────────────────

    _autoSaveIfNeeded(generation, rl_enabled) {
        if (!this._auto_save_enabled) return;
        if (!this._auto_save_every_n || this._auto_save_every_n <= 0) return;
        if (generation % this._auto_save_every_n !== 0) return;

        try {
            if (this._auto_save_fs_enabled) {
                const out_dir = this._getRunDir(rl_enabled);
                if (!out_dir) return;
                fs.mkdirSync(out_dir, { recursive: true });

                const genPath = path.join(out_dir, 'generations.csv');
                const orgPath = path.join(out_dir, 'organisms.csv');
                const evtPath = path.join(out_dir, 'events.csv');

                if (this._auto_save_append) {
                    this._appendCSVRows(this.generation_log, this._last_saved_gen_idx, genPath);
                    this._appendCSVRows(this.organism_log, this._last_saved_org_idx, orgPath);
                    this._appendCSVRows(this.event_log, this._last_saved_evt_idx, evtPath);
                } else {
                    fs.writeFileSync(genPath, this.generationsCSV(), 'utf8');
                    fs.writeFileSync(orgPath, this.organismsCSV(), 'utf8');
                    fs.writeFileSync(evtPath, this.eventsCSV(), 'utf8');
                }
            } else if (this._server_enabled) {
                const condition = rl_enabled ? 'learning' : 'evolution';
                const mode = this._getModeLabel();
                this._postAppendRows(this.generation_log, this._last_saved_gen_idx, 'generations.csv', condition, mode);
                this._postAppendRows(this.organism_log, this._last_saved_org_idx, 'organisms.csv', condition, mode);
                this._postAppendRows(this.event_log, this._last_saved_evt_idx, 'events.csv', condition, mode);
            }

            this._last_saved_gen_idx = this.generation_log.length;
            this._last_saved_org_idx = this.organism_log.length;
            this._last_saved_evt_idx = this.event_log.length;

            // ── Trim in-memory arrays after a successful flush ─────────────────
            // Keep only the last generation entry so inter_gen_weight_change can
            // still read the previous generation's RMS magnitude.  Everything
            // before that has already been persisted.
            if (this.generation_log.length > 1) {
                this.generation_log = this.generation_log.slice(-1);
                // The one kept entry was already flushed — mark it saved so it
                // is NOT re-sent on the next flush cycle (fixes duplicate rows).
                this._last_saved_gen_idx = 1;
            }
            if (this.organism_log.length > 0) {
                this.organism_log = [];
                this._last_saved_org_idx = 0;
            }
            if (this.event_log.length > 0) {
                this.event_log = [];
                this._last_saved_evt_idx = 0;
            }
        } catch (err) {
            this._auto_save_enabled = false;
            console.warn('[Logger] Auto-save disabled:', err && err.message ? err.message : err);
        }
    }

    _autoDownloadIfNeeded(ga) {
        if (!this._auto_download_enabled) return;
        if (typeof document === 'undefined') return;
        if (!ga) return;
        const total_ticks = ga.env && typeof ga.env.total_ticks === 'number'
            ? ga.env.total_ticks
            : ga.tick_count;
        if (typeof total_ticks !== 'number') return;
        if (!this._auto_download_every_ticks || this._auto_download_every_ticks <= 0) return;
        if (total_ticks - this._last_auto_download_tick < this._auto_download_every_ticks) return;

        this._last_auto_download_tick = total_ticks;
        this._downloadAllImmediate(`autosave_${Date.now()}`);
        this.clear();
    }

    // ── CSV export ────────────────────────────────────────────────────────────

    _toCSV(rows) {
        if (rows.length === 0) return 'no data';
        return this._toCSVLines(rows, true);
    }

    _toCSVLines(rows, includeHeader) {
        if (!rows || rows.length === 0) return '';
        const headers = Object.keys(rows[0]);
        const lines = rows.map(row =>
            headers.map(h => {
                const v = row[h] === null || row[h] === undefined ? '' : String(row[h]);
                return v.includes(',') || v.includes('"') || v.includes('\n')
                    ? `"${v.replace(/"/g, '""')}"`
                    : v;
            }).join(',')
        );
        const headerLine = includeHeader ? headers.join(',') + '\n' : '';
        return headerLine + lines.join('\n');
    }

    _postAppendRows(rows, startIndex, filename, condition, mode) {
        if (!rows || rows.length <= startIndex) return;
        if (typeof fetch === 'undefined') return;
        const slice = rows.slice(startIndex);
        if (slice.length === 0) return;

        const header = Object.keys(slice[0]).join(',');

        // Organism rows contain two full weight snapshots (~25 KB each).
        // Browsers enforce a 64 KB hard limit on keepalive fetch bodies and
        // silently drop the request without any error if it is exceeded — so
        // keepalive must NOT be used for organism payloads.
        // generations/events rows are tiny and safe to send with keepalive.
        const isLargePayload = filename === 'organisms.csv';

        // Keep chunks small enough that even organism rows don't hit limits.
        // 2 rows × 25 KB = 50 KB < 64 KB keepalive ceiling (used for others).
        // For organisms (no keepalive) a larger chunk is fine — use 10 rows.
        const CHUNK_SIZE = isLargePayload ? 10 : 50;

        for (let i = 0; i < slice.length; i += CHUNK_SIZE) {
            const chunk = slice.slice(i, i + CHUNK_SIZE);
            const lines = this._toCSVLines(chunk, false);
            if (!lines) continue;

            fetch(LOG_SERVER_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    condition,
                    mode,
                    filename,
                    header,
                    rows: lines
                }),
                // keepalive must be false for large payloads (organism weight snapshots).
                // keepalive is only useful for page-unload scenarios anyway; mid-run
                // flushes don't need it.
                keepalive: !isLargePayload
            }).catch(() => {});
        }
    }

    _getModeLabel() {
        const mode = WorldConfig.experiment_mode;
        if (mode === 'frozen_pg' || mode === 'pure_rl') return mode;
        return 'standard';
    }

    _getRunDir(rl_enabled) {
        if (!this._auto_save_fs_enabled) return null;

        const base = path.resolve(process.cwd(), this._auto_save_dir);
        const condition_dir = rl_enabled ? 'learning' : 'evolution';
        const mode_dir = this._getModeLabel();
        const auto_dir = path.join(base, condition_dir, mode_dir, AUTO_SAVE_RUN_FOLDER);

        if (!this._auto_save_run_dir) {
            fs.mkdirSync(auto_dir, { recursive: true });
            const entries = fs.readdirSync(auto_dir, { withFileTypes: true })
                .filter(d => d.isDirectory() && /^run_\d+$/.test(d.name))
                .map(d => parseInt(d.name.replace('run_', ''), 10))
                .filter(n => Number.isFinite(n));
            const next = entries.length > 0 ? Math.max(...entries) + 1 : 1;
            this._auto_save_run_dir = path.join(auto_dir, `run_${next}`);
        }

        return this._auto_save_run_dir;
    }

    _appendCSVRows(rows, startIndex, filePath) {
        if (!rows || rows.length <= startIndex) return;
        const slice = rows.slice(startIndex);
        if (slice.length === 0) return;

        const hasFile = fs.existsSync(filePath);
        const hasContent = hasFile && fs.statSync(filePath).size > 0;
        const csv = this._toCSVLines(slice, !hasContent);
        if (!csv) return;

        const prefix = hasContent ? '\n' : '';
        fs.appendFileSync(filePath, prefix + csv, 'utf8');
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

    _downloadAllImmediate(prefix) {
        const date = new Date().toISOString().split('T')[0];
        const p    = prefix ? `${prefix}_` : '';
        this._download(this.generationsCSV(), `${p}generations_${date}.csv`);
        this._download(this.organismsCSV(), `${p}organisms_${date}.csv`);
        this._download(this.eventsCSV(), `${p}events_${date}.csv`);
    }

    /** Download the generation summary CSV */
    downloadGenerations(filename) {
        this._download(this.generationsCSV(), filename || 'generations.csv');
    }

    /** Download the per-organism detail CSV */
    downloadOrganisms(filename) {
        this._download(this.organismsCSV(), filename || 'organisms.csv');
    }

    /** Download the freeform event log CSV */
    downloadEvents(filename) {
        this._download(this.eventsCSV(), filename || 'events.csv');
    }

    /**
     * Download all three CSVs in sequence.
     * Small delay between each so browsers don't block multiple downloads.
     */
    downloadAll(prefix) {
        const p = prefix ? `${prefix}_` : '';
        this.downloadGenerations(`${p}generations.csv`);
        setTimeout(() => this.downloadOrganisms(`${p}organisms.csv`),   400);
        setTimeout(() => this.downloadEvents(`${p}events.csv`),         800);
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    clear() {
        this.generation_log = [];
        this.organism_log   = [];
        this.event_log      = [];
        this._start_time    = Date.now();
        this._last_saved_gen_idx = 0;
        this._last_saved_org_idx = 0;
        this._last_saved_evt_idx = 0;
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