/**
 * analyzeWeights.js
 *
 * Analyzes neural network weight magnitudes across generations.
 * Reads generation logs and shows how normalized RMS weights change per generation.
 *
 * NOTE: Log files record RMS-normalized weights (sqrt(sum_of_squares) / sqrt(num_weights)).
 *       With [-1, 1] bounded weights, typical RMS values are in [0, 1] range.
 *
 * Usage:
 *   node analysis/analyzeWeights.js <path-to-generation-log.csv>
 *
 * Output:
 *   - Console table showing RMS weight statistics per generation
 *   - Plots how average RMS weight magnitude changes over generations
 *   - Identifies generations with notable weight divergence
 */

'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');

class WeightAnalyzer {
    constructor() {
        this.generations = [];
        this.stats = {
            min_mag: Infinity,
            max_mag: -Infinity,
            avg_mag: 0,
            generations_analyzed: 0,
        };
    }

    /**
     * Parse CSV generation log file and extract weight data.
     * Expected headers include: generation, avg_network_weight_mag, avg_learned_weight_diff, etc.
     */
    async loadGenerationLog(filepath) {
        return new Promise((resolve, reject) => {
            const stream = fs.createReadStream(filepath);
            const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

            let headers = null;
            let lineNum = 0;

            rl.on('line', (line) => {
                lineNum++;
                if (lineNum === 1) {
                    headers = line.split(',').map(h => h.trim());
                    return;
                }

                const values = line.split(',').map(v => v.trim());
                const record = {};
                headers.forEach((h, i) => {
                    record[h] = isNaN(values[i]) ? values[i] : parseFloat(values[i]);
                });

                if (record.generation && record.avg_network_weight_mag !== undefined) {
                    this.generations.push({
                        generation: record.generation,
                        weight_mag: record.avg_network_weight_mag,
                        learned_diff: record.avg_learned_weight_diff || 0,
                        condition: record.condition || 'unknown',
                        ticks: record.generation_ticks || 0,
                        agents: record.total_agents || 0,
                        best_fitness: record.best_fitness || 0,
                    });
                }
            });

            rl.on('close', () => {
                this.computeStats();
                resolve(this.generations);
            });

            rl.on('error', reject);
        });
    }

    /**
     * Compute aggregate statistics across all generations.
     */
    computeStats() {
        if (this.generations.length === 0) return;

        const mags = this.generations.map(g => g.weight_mag);
        this.stats.min_mag = Math.min(...mags);
        this.stats.max_mag = Math.max(...mags);
        this.stats.avg_mag = mags.reduce((s, m) => s + m, 0) / mags.length;
        this.stats.generations_analyzed = mags.length;
    }

    /**
     * Normalize weight magnitude to [-1, 1] range for visualization.
     * Since logs now record RMS (normalized) weights, they're already in [0, 1] range.
     * This method is for reference - just clamps to ensure visualization bar works.
     */
    normalizeWeight(mag) {
        // RMS weights from logs are already normalized and should be in [0, 1]
        return Math.min(1, Math.max(0, mag));
    }

    /**
     * Print formatted table of weight changes per generation.
     */
    printTable() {
        console.log('\n' + '='.repeat(120));
        console.log('RMS WEIGHT MAGNITUDE ANALYSIS ACROSS GENERATIONS ([-1, 1] Bounded Weights)');
        console.log('='.repeat(120));
        console.log();

        if (this.generations.length === 0) {
            console.log('No generation data found.');
            return;
        }

        const headers = ['Gen', 'Condition', 'RMS Mag', 'Bar', 'Change', 'Learned Diff', 'Best Fitness', 'Agents', 'Ticks'];
        console.log(
            headers
                .map((h, i) => i === 2 ? h.padEnd(10) : i === 3 ? h.padEnd(12) : i === 4 ? h.padEnd(10) : h.padEnd(14))
                .join('│')
        );
        console.log('-'.repeat(120));

        for (let i = 0; i < this.generations.length; i++) {
            const gen = this.generations[i];
            const prev = i > 0 ? this.generations[i - 1] : null;
            const change = prev ? gen.weight_mag - prev.weight_mag : 0;
            const changeStr = change === 0 ? '→' : (change > 0 ? '↑' : '↓') + ' ' + Math.abs(change).toFixed(4);
            const normalized = this.normalizeWeight(gen.weight_mag);
            const bar = this._drawBar(normalized, 10);

            console.log(
                [
                    gen.generation.toString().padEnd(4),
                    gen.condition.padEnd(14),
                    gen.weight_mag.toFixed(5).padEnd(10),
                    bar.padEnd(12),
                    changeStr.padEnd(10),
                    gen.learned_diff.toFixed(6).padEnd(14),
                    gen.best_fitness.toFixed(4).padEnd(12),
                    gen.agents.toString().padEnd(7),
                    gen.ticks.toString().padEnd(6),
                ].join('│')
            );
        }

        console.log('-'.repeat(120));
        console.log();
    }

    /**
     * Draw a simple ASCII bar for visualization.
     */
    _drawBar(normalized, length) {
        const filled = Math.round(normalized * length);
        return '[' + '█'.repeat(filled) + '░'.repeat(length - filled) + ']';
    }

    /**
     * Print summary statistics.
     */
    printSummary() {
        const s = this.stats;
        console.log('SUMMARY STATISTICS (RMS-normalized weights):');
        console.log(`  Generations analyzed: ${s.generations_analyzed}`);
        console.log(`  Min weight magnitude: ${s.min_mag.toFixed(6)} (RMS)`);
        console.log(`  Max weight magnitude: ${s.max_mag.toFixed(6)} (RMS)`);
        console.log(`  Average magnitude:    ${s.avg_mag.toFixed(6)} (RMS)`);
        console.log(`  Note: RMS values are sqrt(L2_norm² / num_weights). With [-1, 1] bounds, range is typically [0, 1].`);
        console.log();

        // Trend analysis
        if (this.generations.length > 1) {
            const start = this.generations[0].weight_mag;
            const end = this.generations[this.generations.length - 1].weight_mag;
            const trend = end - start;
            console.log('TREND (RMS Weight Evolution):');
            if (Math.abs(trend) < 0.0001) {
                console.log(`  Stable: RMS weights unchanged (${trend.toFixed(6)} change over entire run)`);
            } else if (trend > 0) {
                console.log(`  ↑ INCREASING: RMS weights growing (${trend.toFixed(6)} net change)`);
                console.log(`    Interpretation: GA selecting for networks with larger typical weight magnitudes`);
            } else {
                console.log(`  ↓ DECREASING: RMS weights shrinking (${trend.toFixed(6)} net change)`);
                console.log(`    Interpretation: GA selecting for more conservative weight scales`);
            }
        }
        console.log();
    }

    /**
     * Identify generations with notable changes.
     */
    printNotableEvents() {
        console.log('NOTABLE EVENTS:');
        let found = false;

        for (let i = 1; i < this.generations.length; i++) {
            const prev = this.generations[i - 1];
            const curr = this.generations[i];
            const change = curr.weight_mag - prev.weight_mag;
            const pct_change = (change / prev.weight_mag) * 100;

            // Flag large changes (>5% per generation)
            if (Math.abs(pct_change) > 5) {
                found = true;
                const direction = change > 0 ? 'INCREASE' : 'DECREASE';
                console.log(
                    `  Gen ${curr.generation}: ${direction} of ${Math.abs(pct_change).toFixed(1)}% ` +
                    `(${prev.weight_mag.toFixed(4)} → ${curr.weight_mag.toFixed(4)})`
                );
            }
        }

        if (!found) {
            console.log('  No significant changes detected (>5% per generation)');
        }
        console.log();
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log('Usage: node analyzeWeights.js <path-to-generation-log.csv>');
        console.log();
        console.log('This script analyzes neural network weight magnitudes from generation logs.');
        console.log('It normalizes values to the [-1, 1] bounded range and shows how weights evolve.');
        process.exit(1);
    }

    const logPath = args[0];

    if (!fs.existsSync(logPath)) {
        console.error(`Error: File not found: ${logPath}`);
        process.exit(1);
    }

    try {
        const analyzer = new WeightAnalyzer();
        console.log(`Loading generation log: ${logPath}...`);
        await analyzer.loadGenerationLog(logPath);

        analyzer.printTable();
        analyzer.printSummary();
        analyzer.printNotableEvents();

        console.log('Analysis complete.');
    } catch (err) {
        console.error('Error:', err.message);
        process.exit(1);
    }
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
