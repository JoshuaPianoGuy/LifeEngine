const PopulationChart = require("./Charts/PopulationChart");
const SpeciesChart = require("./Charts/SpeciesChart");
const MutationChart = require("./Charts/MutationChart");
const CellsChart = require("./Charts/CellsChart");
const PeakPopChart = require("./Charts/PeakPopChart");
const TotalAgentsChart = require("./Charts/TotalAgentsChart");
const IntraGenWeightChart = require("./Charts/IntraGenWeightChart");
const InterGenWeightChart = require("./Charts/InterGenWeightChart");
const GenVarianceChart = require("./Charts/GenVarianceChart");
const Top20FitnessChart = require("./Charts/Top20FitnessChart");
const FossilRecord = require("./FossilRecord");
const NNBrain = require("../Organism/Perception/NNBrain");
const WorldConfig = require("../WorldConfig");

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

const STATS_APPEND_ENABLED = true;
const STATS_APPEND_CLEAR_MEMORY = true;
const STATS_APPEND_DIR = 'logs';
const STATS_APPEND_RUN_FOLDER = 'auto-run';
const STATS_SERVER_ENABLED = true;
const STATS_SERVER_ENDPOINT = '/api/logs/snapshot';
const STATS_LOG_FILENAME = 'life-engine-logs.csv';
const STATS_TICK_LOG_ENABLED = true;
const STATS_TICK_LOG_EVERY_TICKS = 50000;
const LOG_WEIGHTS_PRECISION = 6;

const ChartSelections = [PopulationChart, SpeciesChart, CellsChart, MutationChart, PeakPopChart, TotalAgentsChart, IntraGenWeightChart, InterGenWeightChart, GenVarianceChart, Top20FitnessChart];

class StatsPanel {
    constructor(env) {
        this.defineControls();
        this.chart_selection = 0;
        this.setChart();
        this.env = env;
        this.last_reset_count=env.reset_count;
        this.selected_organism = null;
        
        // Initialize organism history for detailed logging across entire run
        this.all_generations = [];  // Array of all generations across entire session
        this.current_run_generation = 0;  // Generation counter for current run (resets when environment resets)
        this.total_organisms_ever = 0;  // Track total organisms ever created
        this.ga_metrics = [];  // Store GA metrics from each generation

        this._append_enabled = STATS_APPEND_ENABLED && !!fs && !!path;
        this._append_clear_memory = STATS_APPEND_CLEAR_MEMORY;
        this._append_dir = STATS_APPEND_DIR;
        this._append_run_dir = null;
        this._server_enabled = STATS_SERVER_ENABLED;
        this._tick_log_enabled = STATS_TICK_LOG_ENABLED;
        this._tick_log_every = STATS_TICK_LOG_EVERY_TICKS;
        this._last_tick_log_tick = 0;
        this._log_start_time = Date.now();
        this._prev_avg_weight_mag = 0;
        this._frozen_pg_generation = 0;
    }

    setChart(selection=this.chart_selection) {
        this.chart_controller = new ChartSelections[selection]();
        this.chart_controller.setData();
        this.chart_controller.render();
    }

    startAutoRender() {
        this.setChart();
        this.render_loop = setInterval(function(){this.updateChart();}.bind(this), 1000);
    }

    stopAutoRender() {
        clearInterval(this.render_loop);
    }

    defineControls() {
        $('#chart-option').change ( function() {
            this.chart_selection = $("#chart-option")[0].selectedIndex;
            this.setChart();
        }.bind(this));
    }

    updateChart() {
        if (this.last_reset_count < this.env.reset_count){
            this.reset()
        }
        this.last_reset_count = this.env.reset_count;
        this.chart_controller.updateData();
        this.chart_controller.render();
    }

    setTickLogInterval(ticks) {
        const next = Number(ticks);
        if (!Number.isFinite(next) || next < 0) return;
        this._tick_log_every = next;
    }

    maybeLogTick(totalTicks = null) {
        // Emit tick snapshots for all experiment modes so automatic logs
        // match browser-download logs regardless of condition.
        if (!this._tick_log_enabled) return;
        if (!this._tick_log_every || this._tick_log_every <= 0) return;
        const ticks = typeof totalTicks === 'number' ? totalTicks : this.env.total_ticks;
        if (typeof ticks !== 'number') return;
        if (ticks - this._last_tick_log_tick < this._tick_log_every) return;

        this._last_tick_log_tick = ticks;
        const snapshot = this.captureGenerationSnapshot();
        if (this._append_enabled) {
            this._appendSnapshotToFile(snapshot);
            if (!this._append_clear_memory) {
                this.all_generations.push(snapshot);
            }
        } else if (this._server_enabled) {
            this._postSnapshotToServer(snapshot);
        } else {
            this.all_generations.push(snapshot);
        }

        // Emit GA-style detailed logs (generations.csv, organisms.csv, events.csv)
        // for all experiment modes so automatic logs always match browser-download logs.
        // Do NOT advance the generation counter here; generation increments
        // should be driven by environment reset/generation events only.
        this._appendDetailedLogs(snapshot);
    }

    // Calculate genetic drift for an organism
    calculateDrift(org) {
        if (!org.brain || !org.brain.active_weights || !org.brain.genome_weights) {
            return null;
        }
        let total_drift = 0;
        for (let j = 0; j < org.brain.genome_weights.length; j++) {
            total_drift += Math.abs(org.brain.active_weights[j] - org.brain.genome_weights[j]);
        }
        return (total_drift / org.brain.genome_weights.length).toFixed(4);
    }

    _calcWeightStats(weights) {
        if (!weights || weights.length === 0) return null;
        let min = Infinity;
        let max = -Infinity;
        let sum = 0;
        let sum_sq = 0;
        for (let i = 0; i < weights.length; i++) {
            const v = weights[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sum_sq += v * v;
        }
        const mean = sum / weights.length;
        const variance = (sum_sq / weights.length) - (mean * mean);
        const std = Math.sqrt(Math.max(0, variance));
        const rms = Math.sqrt(sum_sq / weights.length);
        return { mean, std, min, max, rms };
    }

    _formatWeightSample(weights, count = 12) {
        if (!weights || weights.length === 0) return '';
        const slice = Array.from(weights.slice(0, Math.min(count, weights.length)));
        return slice.map(v => v.toFixed(3)).join(' ');
    }

    _appendSnapshotToFile(snapshot) {
        if (!this._append_enabled || !snapshot) return;

        const out_dir = this._getRunDir();
        if (!out_dir) return;
        fs.mkdirSync(out_dir, { recursive: true });

        const filePath = path.join(out_dir, STATS_LOG_FILENAME);
        const hasFile = fs.existsSync(filePath);
        const hasContent = hasFile && fs.statSync(filePath).size > 0;
        const csv = this._formatSnapshotCSV(snapshot);
        if (!csv) return;

        const prefix = hasContent ? '\n' : '';
        fs.appendFileSync(filePath, prefix + csv, 'utf8');
    }

    _formatSnapshotCSV(snapshot) {
        if (!snapshot) return '';
        let csv = '';

        csv += '=== GENERATION SNAPSHOT ===\n';
        csv += 'generation,timestamp,ticks,organism_count,species_count,avg_lifetime,avg_food_collected,avg_energy,avg_cell_count,avg_genetic_drift\n';

        const orgs = snapshot.organisms || [];
        const avg_lifetime = orgs.length > 0
            ? (orgs.reduce((sum, o) => sum + (o.lifetime || 0), 0) / orgs.length).toFixed(2)
            : '0';
        const avg_food = orgs.length > 0
            ? (orgs.reduce((sum, o) => sum + (o.food_collected || 0), 0) / orgs.length).toFixed(2)
            : '0';
        const avg_energy = orgs.length > 0
            ? (orgs.reduce((sum, o) => sum + (o.energy || 0), 0) / orgs.length).toFixed(2)
            : '0';
        const avg_cells = orgs.length > 0
            ? (orgs.reduce((sum, o) => sum + (o.cell_count || 0), 0) / orgs.length).toFixed(2)
            : '0';
        const organisms_with_drift = orgs.filter(o => o.genetic_drift !== null);
        const avg_drift = organisms_with_drift.length > 0
            ? (organisms_with_drift.reduce((sum, o) => sum + parseFloat(o.genetic_drift), 0) / organisms_with_drift.length).toFixed(4)
            : 'N/A';

        csv += `${snapshot.generation_number},"${snapshot.timestamp}",${snapshot.tick_count},${snapshot.organism_count},${snapshot.species_count},${avg_lifetime},${avg_food},${avg_energy},${avg_cells},${avg_drift}\n`;

        csv += '=== TOP 20% ORGANISMS ===\n';
        csv += 'generation,rank,fitness,lifetime,species,cell_count,genetic_drift\n';
        (snapshot.top20pct || []).forEach(org => {
            csv += `${snapshot.generation_number},${org.rank},${org.fitness},${org.lifetime},"${org.species}",${org.cell_count},${org.genetic_drift || 'N/A'}\n`;
        });

        csv += '=== DETAILED ORGANISM DATA ===\n';
        csv += 'generation,organism_id,species,fitness,lifetime,food_collected,energy,max_energy,cell_count,cell_composition,genetic_drift,rl_enabled,position_x,position_y,direction\n';
        orgs.forEach(org => {
            let cell_comp = '';
            for (const [type, count] of Object.entries(org.cell_composition)) {
                if (cell_comp) cell_comp += '; ';
                cell_comp += `${type}:${count}`;
            }
            if (!cell_comp) cell_comp = 'none';

            csv += `${snapshot.generation_number},${org.id},"${org.species}",${org.fitness.toFixed(2)},${org.lifetime},${org.food_collected},${org.energy !== null ? org.energy : 'N/A'},${org.max_energy !== null ? org.max_energy : 'N/A'},${org.cell_count},"${cell_comp}",${org.genetic_drift || 'N/A'},${org.rl_enabled},${org.position_x},${org.position_y},${org.direction}\n`;
        });

        return csv.trimEnd();
    }

    _postSnapshotToServer(snapshot) {
        if (!this._server_enabled || !snapshot || typeof fetch === 'undefined') return;
        const csv = this._formatSnapshotCSV(snapshot);
        if (!csv) return;

        const condition = this.env && this.env.learning_enabled ? 'learning' : 'evolution';
        const mode = this._getModeLabel();
        fetch(STATS_SERVER_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition, mode, csv, filename: STATS_LOG_FILENAME }),
            keepalive: true
        }).catch(() => {});
    }

    _getModeLabel() {
        const mode = WorldConfig.experiment_mode;
        if (mode === 'frozen_pg' || mode === 'pure_rl') return mode;
        return 'standard';
    }

    _getRunDir() {
        if (!this._append_enabled) return null;

        const base = path.resolve(process.cwd(), this._append_dir);
        const condition_dir = this.env && this.env.learning_enabled ? 'learning' : 'evolution';
        const mode_dir = this._getModeLabel();
        const auto_dir = path.join(base, condition_dir, mode_dir, STATS_APPEND_RUN_FOLDER);

        if (!this._append_run_dir) {
            fs.mkdirSync(auto_dir, { recursive: true });
            const entries = fs.readdirSync(auto_dir, { withFileTypes: true })
                .filter(d => d.isDirectory() && /^run_\d+$/.test(d.name))
                .map(d => parseInt(d.name.replace('run_', ''), 10))
                .filter(n => Number.isFinite(n));
            const next = entries.length > 0 ? Math.max(...entries) + 1 : 1;
            this._append_run_dir = path.join(auto_dir, `run_${next}`);
        }

        return this._append_run_dir;
    }

    _getGenerationNumber() {
        // Always read from ga_manager when present — it is authoritative for all
        // modes (standard GA, frozen_pg, pure_rl).
        // current_run_generation is only incremented inside StatsPanel.reset(),
        // which is never called during GA-managed experiments, so it stays at 0.
        if (this.env && this.env.ga_manager && typeof this.env.ga_manager.generation === 'number') {
            return this.env.ga_manager.generation;
        }
        if (WorldConfig.experiment_mode === 'frozen_pg') {
            return this._frozen_pg_generation;
        }
        return this.current_run_generation;
    }

    _advanceGenerationCounter() {
        // Only advance the internal counter for frozen_pg where it is still
        // used as a fallback. pure_rl uses ga_manager.generation exclusively.
        if (WorldConfig.experiment_mode !== 'frozen_pg') return;
        this._frozen_pg_generation++;
    }

    // Capture organism data with ranking
    captureGenerationSnapshot() {
        const orgs = this.env.organisms;
        const generation_data = {
            generation_number: this._getGenerationNumber(),
            timestamp: new Date().toISOString(),
            tick_count: this.env.total_ticks,
            organism_count: orgs.length,
            species_count: FossilRecord.numExtantSpecies(),
            organisms: [],
            top20pct: []  // Add top 20% organisms
        };

        // Capture detailed data for each living organism
        const organisms_with_fitness = [];
        orgs.forEach((org, idx) => {
            const cell_types = {};
            for (const cell of org.anatomy.cells) {
                const cell_name = cell.state.name;
                cell_types[cell_name] = (cell_types[cell_name] || 0) + 1;
            }

            const drift = this.calculateDrift(org);
            const org_data = {
                id: idx,
                rank: 0,  // Will be set after sorting
                species: org.species ? org.species.name : 'Unknown',
                fitness: org.getFitness ? org.getFitness() : org.lifetime || 0,
                energy: org.energy !== undefined ? parseFloat(org.energy.toFixed(2)) : null,
                max_energy: org.max_energy || null,
                food_collected: org.food_collected || 0,
                lifetime: org.lifetime || 0,
                cell_count: org.anatomy.cells.length,
                cell_composition: cell_types,
                position_x: org.c || 0,
                position_y: org.r || 0,
                direction: org.direction || 0,
                rl_enabled: org.rl_enabled || false,
                genetic_drift: drift
            };

            generation_data.organisms.push(org_data);
            organisms_with_fitness.push({ org_data, original_org: org });
        });

        // Sort by fitness and capture top 20%
        organisms_with_fitness.sort((a, b) => b.org_data.fitness - a.org_data.fitness);
        let num_top = Math.floor(organisms_with_fitness.length * 0.2);
        num_top = Math.max(1, Math.min(num_top, organisms_with_fitness.length));

        for (let i = 0; i < num_top; i++) {
            const top_org = organisms_with_fitness[i].org_data;
            top_org.rank = i + 1;
            generation_data.top20pct.push({
                rank: i + 1,
                fitness: parseFloat(top_org.fitness.toFixed(2)),
                lifetime: top_org.lifetime,
                species: top_org.species,
                cell_count: top_org.cell_count,
                genetic_drift: top_org.genetic_drift
            });
        }

        return generation_data;
    }

    // Called when environment resets - saves current generation data
    reset() {
        // Capture current generation snapshot
        const snapshot = this.captureGenerationSnapshot();
        if (this._append_enabled) {
            this._appendSnapshotToFile(snapshot);
            if (!this._append_clear_memory) {
                this.all_generations.push(snapshot);
            } else {
                this.all_generations = [];
            }
        } else if (this._server_enabled) {
            this._postSnapshotToServer(snapshot);
        } else {
            this.all_generations.push(snapshot);
        }
        this._appendDetailedLogs(snapshot);
        this.total_organisms_ever += snapshot.organism_count;
        
        // Track GA metrics if available
        if (this.env.ga_manager) {
            const ga_data = {
                generation: this.current_run_generation,
                ticks: this.env.ga_manager.tick_count || 0,
                total_agents: snapshot.organism_count,
                peak_population: this.env.ga_manager.peak_population || 0,
                avg_lifetime: snapshot.organisms.length > 0 ? 
                    (snapshot.organisms.reduce((sum, o) => sum + o.lifetime, 0) / snapshot.organisms.length).toFixed(1) : 0
            };
            this.ga_metrics.push(ga_data);
        }
        
        // Do not advance the internal counter for tick-based conditions —
        // the GA manager is authoritative for both frozen_pg and pure_rl.
        const tick_based_mode = WorldConfig.experiment_mode === 'frozen_pg' ||
                                WorldConfig.experiment_mode === 'pure_rl';
        if (!tick_based_mode) {
            this.current_run_generation++;
        }
    }

    // Set the currently selected organism for display
    setSelectedOrganism(org) {
        this.selected_organism = org;
    }

    updateDetails() {
        var org_count = this.env.organisms.length;
        $('#org-count').text("Total Population: " + org_count);
        $('#species-count').text("Number of Species: " + FossilRecord.numExtantSpecies());
        let top_species = FossilRecord.getMostPopulousSpecies();
        if (top_species)
            $('#top-species').text("Most Populous Species: \"" + top_species.name + "\" (" + top_species.population + " organisms)");
        else    
            $('#top-species').text("Most Populous Species: None");
        $('#largest-org').text("Largest Organism Ever: " + this.env.largest_cell_count + " cells");
        $('#avg-mut').text("Average Mutation Rate: " + Math.round(this.env.averageMutability() * 100) / 100);

        // Add population-level statistics
        this.updatePopulationStats();

        // Update organism details if one is selected
        if (this.selected_organism && this.selected_organism.living) {
            this.updateSelectedOrganismInfo();
        }
    }

    // Display aggregate population statistics
    updatePopulationStats() {
        const orgs = this.env.organisms;
        if (orgs.length === 0) return;

        // Calculate average energy for living organisms
        const avg_energy = orgs.length > 0
            ? (orgs.reduce((sum, org) => sum + (org.energy || 0), 0) / orgs.length).toFixed(1)
            : 0;

        // Calculate average food collected
        const avg_food = orgs.length > 0
            ? (orgs.reduce((sum, org) => sum + org.food_collected, 0) / orgs.length).toFixed(1)
            : 0;

        // Calculate average lifetime
        const avg_lifetime = orgs.length > 0
            ? (orgs.reduce((sum, org) => sum + org.lifetime, 0) / orgs.length).toFixed(1)
            : 0;

        // Get generation info from GA manager
        const generation = this.env.ga_manager ? this.env.ga_manager.generation : 0;

        // Display population stats
        let statsHtml = `<strong>Population Avg Energy:</strong> ${avg_energy}<br>`;
        statsHtml += `<strong>Population Avg Food Collected:</strong> ${avg_food}<br>`;
        statsHtml += `<strong>Population Avg Lifetime:</strong> ${avg_lifetime} ticks<br>`;
        statsHtml += `<strong>Current Generation:</strong> ${generation}`;

        const statsContainer = $('#population-stats');
        if (statsContainer.length === 0) {
            // If container doesn't exist, add it to the About section
            $('#avg-mut').after(`<div id="population-stats" style="margin-top: 10px; font-size: 0.9em;"></div>`);
        }
        $('#population-stats').html(statsHtml);
    }

    // Display information about the selected organism in the Organism Info tab
    updateSelectedOrganismInfo() {
        const org = this.selected_organism;
        
        // Show the details section, hide the message
        $('#org-info-message').hide();
        $('#organism-details').show();
        
        // Position
        $('#org-position').html(`<strong>Position:</strong> (${org.c}, ${org.r})`);

        // Fitness + food score
        if (typeof org.getFitness === 'function') {
            $('#org-fitness').html(`<strong>Fitness:</strong> ${org.getFitness().toFixed(4)}`);
        } else {
            $('#org-fitness').html(`<strong>Fitness:</strong> n/a`);
        }
        $('#org-cumulative-food').html(`<strong>Cumulative Food Score:</strong> ${(org.cumulative_food_score || 0).toFixed(4)}`);
        
        // Lifetime
        $('#org-lifetime').html(`<strong>Lifetime:</strong> ${org.lifetime} ticks`);
        
        // Food collected
        $('#org-food-collected').html(`<strong>Food Collected:</strong> ${org.food_collected}`);

        // Energy info (for AdvancedOrganism)
        if (org.energy !== undefined) {
            const energy_pct = ((org.energy / org.max_energy) * 100).toFixed(1);
            $('#org-energy').html(`<strong>Energy:</strong> ${org.energy.toFixed(1)}/${org.max_energy} (${energy_pct}%)`).show();
        } else {
            $('#org-energy').hide();
        }

        // Movement direction
        const dir_names = ['↑ North', '→ East', '↓ South', '← West'];
        if (org.direction !== undefined) {
            $('#org-direction').html(`<strong>Current Direction:</strong> ${dir_names[org.direction]}`);
        }

        // Cell type inventory
        const cell_types = {};
        for (const cell of org.anatomy.cells) {
            const cell_name = cell.state.name;
            cell_types[cell_name] = (cell_types[cell_name] || 0) + 1;
        }
        let inventory_html = '';
        for (const [type, count] of Object.entries(cell_types)) {
            inventory_html += `<p>• ${type}: ${count}</p>`;
        }
        $('#org-cell-inventory').html(inventory_html);

        // Brain info
        if (org.brain && org.brain instanceof NNBrain) {
            $('#org-brain-rl').html(`<strong>RL Status:</strong> ${org.rl_enabled ? 'Enabled' : 'Disabled'}`);
            $('#org-brain-type').html('<strong>Type:</strong> Neural Network (46→32→6)');
            
            if (org.brain._last_action !== null) {
                const action_names = ['↑ North', '→ East', '↓ South', '← West', '⟲ Rotate Left', '⟳ Rotate Right'];
                $('#org-brain-last-action').html(`<strong>Last Action:</strong> ${action_names[org.brain._last_action]}`);
            } else {
                $('#org-brain-last-action').html(`<strong>Last Action:</strong> (none yet)`);
            }
            
            if (org.brain._last_probs) {
                const labels = ['↑', '→', '↓', '←', '⟲', '⟳'];
                const probs = org.brain._last_probs.map((p, i) => `${labels[i]}:${(p*100).toFixed(0)}%`).join('  |  ');
                $('#org-brain-action-probs').html(`<strong>Action Probabilities:</strong><br/>${probs}`);
            }

            // RL weight drift analysis (Condition A only)
            if (org.rl_enabled && org.brain.active_weights) {
                const genome = org.brain.genome_weights;
                const active = org.brain.active_weights;
                
                // Calculate drift statistics
                let total_drift = 0;
                let max_drift = 0;
                for (let i = 0; i < genome.length; i++) {
                    const diff = Math.abs(active[i] - genome[i]);
                    total_drift += diff;
                    max_drift = Math.max(max_drift, diff);
                }
                const avg_drift = total_drift / genome.length;
                
                // Weight statistics
                const active_stats = this._calcWeightStats(active);
                const genome_stats = this._calcWeightStats(genome);
                
                const learning_indicator = avg_drift < 0.01 ? '◯ Learning off' 
                                          : avg_drift < 0.1 ? '◐ Slow learning'
                                          : avg_drift < 0.5 ? '◑ Moderate learning'
                                          : '● Active learning';
                
                $('#org-brain-weights').html(
                    `<strong>Weight Status:</strong> ${learning_indicator}<br/>` +
                    `• Avg Drift: ${avg_drift.toFixed(4)}<br/>` +
                    `• Max Drift: ${max_drift.toFixed(4)}<br/>` +
                    `• Active RMS: ${active_stats ? active_stats.rms.toFixed(4) : 'n/a'}<br/>` +
                    `• Genome RMS: ${genome_stats ? genome_stats.rms.toFixed(4) : 'n/a'}`
                ).show();
                $('#org-weight-summary').html(
                    `<strong>Weight Summary:</strong> ` +
                    `mean=${active_stats ? active_stats.mean.toFixed(4) : 'n/a'} ` +
                    `std=${active_stats ? active_stats.std.toFixed(4) : 'n/a'} ` +
                    `min=${active_stats ? active_stats.min.toFixed(4) : 'n/a'} ` +
                    `max=${active_stats ? active_stats.max.toFixed(4) : 'n/a'}`
                ).show();
                $('#org-weight-sample').text(
                    this._formatWeightSample(active, 12)
                ).show();
            } else {
                $('#org-brain-weights').hide();
                const genome_stats = this._calcWeightStats(org.brain.genome_weights);
                $('#org-weight-summary').html(
                    `<strong>Weight Summary:</strong> ` +
                    `mean=${genome_stats ? genome_stats.mean.toFixed(4) : 'n/a'} ` +
                    `std=${genome_stats ? genome_stats.std.toFixed(4) : 'n/a'} ` +
                    `min=${genome_stats ? genome_stats.min.toFixed(4) : 'n/a'} ` +
                    `max=${genome_stats ? genome_stats.max.toFixed(4) : 'n/a'}`
                ).show();
                $('#org-weight-sample').text(
                    this._formatWeightSample(org.brain.genome_weights, 12)
                ).show();
            }
        } else {
            $('#org-brain-type').html('<strong>Type:</strong> None');
            $('#org-brain-rl').html('');
            $('#org-brain-last-action').html('');
            $('#org-brain-action-probs').html('');
            $('#org-brain-weights').hide();
            $('#org-weight-summary').hide();
            $('#org-weight-sample').hide();
        }
    }

    downloadLogs() {
        // Capture current generation before downloading
        const snapshot = this.captureGenerationSnapshot();
        if (this._append_enabled) {
            this._appendSnapshotToFile(snapshot);
            if (!this._append_clear_memory) {
                this.all_generations.push(snapshot);
            } else {
                this.all_generations = [];
            }
        } else if (this._server_enabled) {
            this._postSnapshotToServer(snapshot);
        } else {
            this.all_generations.push(snapshot);
        }
        this._appendDetailedLogs(snapshot);
        this.total_organisms_ever += snapshot.organism_count;
        // Do not advance frozen_pg generation counter here; GAManager is authoritative

        let csv = '';
        const timestamp = new Date().toISOString();
        
        // === EXPERIMENT SUMMARY ===
        csv += '=== EXPERIMENT SUMMARY ===\n';
        csv += `Export Timestamp,${timestamp}\n`;
        csv += `Total Organisms Ever,${this.total_organisms_ever}\n`;
        csv += `Total Generations,${this.all_generations.length}\n`;
        csv += `Total Simulation Ticks,${this.env.total_ticks}\n`;
        csv += `Extant Species Count,${FossilRecord.numExtantSpecies()}\n`;
        csv += `Extinct Species Count,${FossilRecord.numExtinctSpecies()}\n`;
        csv += `Largest Organism Ever,${this.env.largest_cell_count} cells\n`;
        csv += `Average Mutability,${Math.round(this.env.averageMutability() * 100) / 100}\n`;
        csv += '\n';

        // === GENERATION-BY-GENERATION SUMMARY ===
        csv += '=== GENERATION SUMMARY ===\n';
        csv += 'generation,timestamp,ticks,organism_count,species_count,avg_lifetime,avg_food_collected,avg_energy,avg_cell_count,avg_genetic_drift\n';
        
        this.all_generations.forEach(gen => {
            if (gen.organisms.length > 0) {
                const avg_lifetime = (gen.organisms.reduce((sum, o) => sum + (o.lifetime || 0), 0) / gen.organisms.length).toFixed(2);
                const avg_food = (gen.organisms.reduce((sum, o) => sum + (o.food_collected || 0), 0) / gen.organisms.length).toFixed(2);
                const avg_energy = (gen.organisms.reduce((sum, o) => sum + (o.energy || 0), 0) / gen.organisms.length).toFixed(2);
                const avg_cells = (gen.organisms.reduce((sum, o) => sum + (o.cell_count || 0), 0) / gen.organisms.length).toFixed(2);
                
                const organisms_with_drift = gen.organisms.filter(o => o.genetic_drift !== null);
                const avg_drift = organisms_with_drift.length > 0 ?
                    (organisms_with_drift.reduce((sum, o) => sum + parseFloat(o.genetic_drift), 0) / organisms_with_drift.length).toFixed(4) : 'N/A';
                
                csv += `${gen.generation_number},"${gen.timestamp}",${gen.tick_count},${gen.organism_count},${gen.species_count},${avg_lifetime},${avg_food},${avg_energy},${avg_cells},${avg_drift}\n`;
            }
        });
        csv += '\n';

        // === TOP 20% ORGANISMS PER GENERATION ===
        csv += '=== TOP 20% ORGANISMS BY GENERATION ===\n';
        csv += 'generation,rank,fitness,lifetime,species,cell_count,genetic_drift\n';
        
        this.all_generations.forEach(gen => {
            if (gen.top20pct && gen.top20pct.length > 0) {
                gen.top20pct.forEach(org => {
                    csv += `${gen.generation_number},${org.rank},${org.fitness},${org.lifetime},"${org.species}",${org.cell_count},${org.genetic_drift || 'N/A'}\n`;
                });
            }
        });
        csv += '\n';

        // === DETAILED ORGANISM DATA FOR EACH GENERATION ===
        csv += '=== DETAILED ORGANISM DATA ===\n';
        
        this.all_generations.forEach(gen => {
            csv += `\nGENERATION ${gen.generation_number}\n`;
            csv += `Timestamp,${gen.timestamp}\n`;
            csv += `Total Ticks,${gen.tick_count}\n`;
            csv += `Population Size,${gen.organism_count}\n`;
            csv += `Species Count,${gen.species_count}\n`;
            csv += '\n';
            
            // Headers for organism data
            csv += 'organism_id,species,fitness,lifetime,food_collected,energy,max_energy,cell_count,cell_composition,genetic_drift,rl_enabled,position_x,position_y,direction\n';
            
            gen.organisms.forEach(org => {
                // Build cell composition string
                let cell_comp = '';
                for (const [type, count] of Object.entries(org.cell_composition)) {
                    if (cell_comp) cell_comp += '; ';
                    cell_comp += `${type}:${count}`;
                }
                if (!cell_comp) cell_comp = 'none';
                
                csv += `${org.id},"${org.species}",${org.fitness.toFixed(2)},${org.lifetime},${org.food_collected},${org.energy !== null ? org.energy : 'N/A'},${org.max_energy !== null ? org.max_energy : 'N/A'},${org.cell_count},"${cell_comp}",${org.genetic_drift || 'N/A'},${org.rl_enabled},${org.position_x},${org.position_y},${org.direction}\n`;
            });
        });

        // === SPECIES-LEVEL SUMMARY ===
        csv += '\n=== SPECIES STATISTICS ===\n';
        csv += 'species_name,first_generation,last_generation,max_population,total_organisms,avg_fitness,avg_lifetime\n';
        
        const species_stats = {};
        this.all_generations.forEach((gen, gen_idx) => {
            gen.organisms.forEach(org => {
                const species = org.species;
                if (!species_stats[species]) {
                    species_stats[species] = {
                        first_generation: gen_idx,
                        last_generation: gen_idx,
                        max_population: 0,
                        total_organisms: 0,
                        total_fitness: 0,
                        total_lifetime: 0,
                        fitness_count: 0
                    };
                }
                species_stats[species].last_generation = gen_idx;
                species_stats[species].total_organisms++;
                species_stats[species].total_fitness += org.fitness;
                species_stats[species].total_lifetime += org.lifetime;
                species_stats[species].fitness_count++;
            });
            
            // Count populations per generation
            const species_pop = {};
            gen.organisms.forEach(org => {
                species_pop[org.species] = (species_pop[org.species] || 0) + 1;
            });
            
            for (const [species, pop] of Object.entries(species_pop)) {
                if (species_stats[species]) {
                    species_stats[species].max_population = Math.max(species_stats[species].max_population, pop);
                }
            }
        });
        
        for (const [species, stats] of Object.entries(species_stats)) {
            const avg_fitness = (stats.total_fitness / stats.fitness_count).toFixed(2);
            const avg_lifetime = (stats.total_lifetime / stats.total_organisms).toFixed(1);
            csv += `"${species}",${stats.first_generation},${stats.last_generation},${stats.max_population},${stats.total_organisms},${avg_fitness},${avg_lifetime}\n`;
        }

        // === CELL TYPE STATISTICS ===
        csv += '\n=== CELL TYPE FREQUENCY ANALYSIS ===\n';
        csv += 'generation,cell_type,total_count,average_per_organism,organisms_with_type,percentage_of_organisms\n';
        
        this.all_generations.forEach(gen => {
            if (gen.organisms.length === 0) return;
            
            const cell_analysis = {};
            gen.organisms.forEach(org => {
                for (const [type, count] of Object.entries(org.cell_composition)) {
                    if (!cell_analysis[type]) {
                        cell_analysis[type] = { total: 0, count: 0, organisms_with_type: 0 };
                    }
                    cell_analysis[type].total += count;
                    cell_analysis[type].organisms_with_type++;
                    cell_analysis[type].count = gen.organisms.length;
                }
            });
            
            for (const [type, data] of Object.entries(cell_analysis)) {
                const avg = (data.total / data.count).toFixed(2);
                const pct = ((data.organisms_with_type / gen.organism_count) * 100).toFixed(1);
                csv += `${gen.generation_number},"${type}",${data.total},${avg},${data.organisms_with_type},${pct}%\n`;
            }
        });

        // Create a Blob and download
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', 'life-engine-logs.csv');
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    _appendDetailedLogs(snapshot) {
        if (!snapshot) return;

        // In standard GA mode, Logger.js already writes generations.csv,
        // organisms.csv and events.csv once per completed generation using
        // GAManager.all_agents (dead + alive, correct fitness).  Writing here
        // too produces duplicate rows with zeroed stats, because env.organisms
        // only holds the *currently living* population at snapshot time.
        const is_standard_ga = WorldConfig.experiment_mode !== 'frozen_pg' &&
                                WorldConfig.experiment_mode !== 'pure_rl';
        if (is_standard_ga) return;

        const orgs = this.env.organisms || [];
        const sorted = this._sortOrganismsByFitness(orgs);
        const top20 = this._getTop20(sorted);
        const generationRow = this._buildGenerationRow(snapshot, sorted, top20);
        const organismRows = this._buildOrganismRows(snapshot, top20);
        const eventRow = this._buildEventRow(snapshot);

        if (this._append_enabled) {
            const out_dir = this._getRunDir();
            if (!out_dir) return;
            fs.mkdirSync(out_dir, { recursive: true });

            this._appendRowsWithHeader(
                path.join(out_dir, 'generations.csv'),
                this._generationHeader(),
                this._formatGenerationRow(generationRow)
            );
            if (organismRows.length > 0) {
                const orgLines = organismRows.map(r => this._formatOrganismRow(r)).join('\n');
                this._appendRowsWithHeader(
                    path.join(out_dir, 'organisms.csv'),
                    this._organismHeader(),
                    orgLines
                );
            } else {
                this._appendRowsWithHeader(
                    path.join(out_dir, 'organisms.csv'),
                    this._organismHeader(),
                    ''
                );
            }
            this._appendRowsWithHeader(
                path.join(out_dir, 'events.csv'),
                this._eventHeader(),
                this._formatEventRow(eventRow)
            );
        } else if (this._server_enabled) {
            const condition = this.env && this.env.learning_enabled ? 'learning' : 'evolution';
            const mode = this._getModeLabel();
            this._postAppendRows('generations.csv', condition, mode, this._generationHeader(), this._formatGenerationRow(generationRow));
            if (organismRows.length > 0) {
                const orgLines = organismRows.map(r => this._formatOrganismRow(r)).join('\n');
                this._postAppendRows('organisms.csv', condition, mode, this._organismHeader(), orgLines);
            } else {
                this._postAppendRows('organisms.csv', condition, mode, this._organismHeader(), '');
            }
            this._postAppendRows('events.csv', condition, mode, this._eventHeader(), this._formatEventRow(eventRow));
        }
    }

    _postAppendRows(filename, condition, mode, header, rows) {
        if (typeof fetch === 'undefined') return;
        fetch('/api/logs/append', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition, mode, filename, header, rows }),
            keepalive: true
        }).catch(() => {});
    }

    _appendRowsWithHeader(filePath, header, rows) {
        const hasFile = fs.existsSync(filePath);
        const hasContent = hasFile && fs.statSync(filePath).size > 0;
        const prefix = hasContent && rows ? '\n' : '';
        const headerText = !hasContent && header ? header + (rows ? '\n' : '') : '';
        fs.appendFileSync(filePath, prefix + headerText + (rows || ''), 'utf8');
    }

    _generationHeader() {
        return [
            'generation',
            'condition',
            'wall_clock_s',
            'generation_ticks',
            'total_agents',
            'peak_population',
            'avg_energy_at_tick_1000',
            'avg_energy_end',
            'avg_fitness',
            'top20percent_fitness',
            'best_fitness',
            'avg_lifetime',
            'avg_learned_weight_diff',
            'inter_gen_weight_change',
            'genome_variance',
            'avg_network_weight_mag'
        ].join(',');
    }

    _organismHeader() {
        return [
            'generation',
            'rank',
            'condition',
            'fitness',
            'lifetime',
            'energy_at_death',
            'energy_at_tick_1000',
            'cumulative_food_score',
            'food_low',
            'food_medium',
            'food_prestige',
            'food_default',
            'cells_visited',
            'learned_weight_diff',
            'network_weight_magnitude',
            'w1_weights',
            'active_weights',
            'genome_weights'
        ].join(',');
    }

    _eventHeader() {
        return 'timestamp_ms,source,message,data';
    }

    _buildGenerationRow(snapshot, sorted, top20) {
        const n = sorted.length;
        const condition = (this.env && this.env.ga_manager && this.env.ga_manager.condition_label)
            ? this.env.ga_manager.condition_label
            : (this.env && this.env.learning_enabled ? 'learning' : 'natural_selection');
        const wall_clock_s = ((Date.now() - this._log_start_time) / 1000).toFixed(1);
        // For tick-based managers, tick_count is ticks-this-window, not total_ticks.
        // peak_population comes from the manager, not the instantaneous living count.
        const ga = this.env && this.env.ga_manager;
        const generation_ticks = ga && typeof ga.tick_count === 'number' ? ga.tick_count : snapshot.tick_count;
        const total_agents = n;
        const peak_population = ga && typeof ga.peak_population === 'number' ? ga.peak_population : n;

        const early_samples = sorted
            .map(o => o.energy_at_early_sample)
            .filter(e => e !== null && e !== undefined);
        const avg_energy_early = early_samples.length > 0
            ? early_samples.reduce((s, e) => s + e, 0) / early_samples.length
            : 0;

        const avg_energy_end = n > 0
            ? sorted.reduce((s, o) => s + (o.energy || 0), 0) / n
            : 0;

        const avg_fitness = n > 0
            ? sorted.reduce((s, o) => s + (o.getFitness ? o.getFitness() : o.lifetime || 0), 0) / n
            : 0;

        const top20percent_fitness = top20.length > 0
            ? top20.reduce((s, o) => s + (o.getFitness ? o.getFitness() : o.lifetime || 0), 0) / top20.length
            : 0;

        const best_fitness = top20.length > 0
            ? (top20[0].getFitness ? top20[0].getFitness() : top20[0].lifetime || 0)
            : 0;

        const avg_lifetime = n > 0
            ? sorted.reduce((s, o) => s + (o.lifetime || 0), 0) / n
            : 0;

        const drifts = top20.map(o => this.calculateDrift(o)).filter(d => d !== null);
        const avg_learned_weight_diff = drifts.length > 0
            ? drifts.reduce((s, d) => s + parseFloat(d), 0) / drifts.length
            : 0;

        const avg_network_weight_mag = n > 0
            ? sorted.reduce((s, o) => s + this._calcRMSWeight(o), 0) / n
            : 0;

        const inter_gen_weight_change = avg_network_weight_mag - this._prev_avg_weight_mag;
        this._prev_avg_weight_mag = avg_network_weight_mag;

        const genome_variance = this._calcGenomeVariance(sorted);

        return {
            generation: snapshot.generation_number,
            condition,
            wall_clock_s,
            generation_ticks,
            total_agents,
            peak_population,
            avg_energy_at_tick_1000: avg_energy_early.toFixed(3),
            avg_energy_end: avg_energy_end.toFixed(3),
            avg_fitness: avg_fitness.toFixed(4),
            top20percent_fitness: top20percent_fitness.toFixed(4),
            best_fitness: best_fitness.toFixed(4),
            avg_lifetime: avg_lifetime.toFixed(1),
            avg_learned_weight_diff: avg_learned_weight_diff.toFixed(6),
            inter_gen_weight_change: inter_gen_weight_change.toFixed(6),
            genome_variance: genome_variance.toFixed(6),
            avg_network_weight_mag: avg_network_weight_mag.toFixed(4)
        };
    }

    _buildOrganismRows(snapshot, top20) {
        const rows = [];
        const condition = (this.env && this.env.ga_manager && this.env.ga_manager.condition_label)
            ? this.env.ga_manager.condition_label
            : (this.env && this.env.learning_enabled ? 'learning' : 'natural_selection');
        for (let i = 0; i < top20.length; i++) {
            const agent = top20[i];
            const learned_weight_diff = this.calculateDrift(agent);
            const network_weight_mag = this._calcRMSWeight(agent);
            const food_counts = agent.food_by_type || {};

            rows.push({
                generation: snapshot.generation_number,
                rank: i + 1,
                condition,
                fitness: (agent.getFitness ? agent.getFitness() : agent.lifetime || 0).toFixed(4),
                lifetime: agent.lifetime || 0,
                energy_at_death: (agent.energy || 0).toFixed(2),
                energy_at_tick_1000: agent.energy_at_early_sample !== null && agent.energy_at_early_sample !== undefined
                    ? agent.energy_at_early_sample.toFixed(2)
                    : 'n/a',
                cumulative_food_score: (agent.cumulative_food_score || 0).toFixed(4),
                food_low: food_counts['low food'] || 0,
                food_medium: food_counts['medium food'] || 0,
                food_prestige: food_counts['prestige food'] || 0,
                food_default: food_counts['food'] || 0,
                cells_visited: agent.visited_cells ? agent.visited_cells.size : 0,
                learned_weight_diff: learned_weight_diff !== null ? Number(learned_weight_diff).toFixed(6) : 'n/a',
                network_weight_magnitude: network_weight_mag.toFixed(4),
                w1_weights: this._snapshotW1(agent),
                active_weights: this._snapshotActive(agent),
                genome_weights: ''
            });
        }

        return rows;
    }

    _buildEventRow(snapshot) {
        const timestamp_ms = Date.now() - this._log_start_time;
        const data = JSON.stringify({ tick: snapshot.tick_count, generation: snapshot.generation_number });
        return {
            timestamp_ms,
            source: 'snapshot',
            message: 'tick_snapshot',
            data
        };
    }

    _formatGenerationRow(row) {
        const values = [
            row.generation,
            row.condition,
            row.wall_clock_s,
            row.generation_ticks,
            row.total_agents,
            row.peak_population,
            row.avg_energy_at_tick_1000,
            row.avg_energy_end,
            row.avg_fitness,
            row.top20percent_fitness,
            row.best_fitness,
            row.avg_lifetime,
            row.avg_learned_weight_diff,
            row.inter_gen_weight_change,
            row.genome_variance,
            row.avg_network_weight_mag
        ];
        return values.join(',');
    }

    _formatOrganismRow(row) {
        const values = [
            row.generation,
            row.rank,
            row.condition,
            row.fitness,
            row.lifetime,
            row.energy_at_death,
            row.energy_at_tick_1000,
            row.cumulative_food_score,
            row.food_low,
            row.food_medium,
            row.food_prestige,
            row.food_default,
            row.cells_visited,
            row.learned_weight_diff,
            row.network_weight_magnitude,
            this._csvEscape(row.w1_weights),
            this._csvEscape(row.active_weights),
            this._csvEscape(row.genome_weights)
        ];
        return values.join(',');
    }

    _formatEventRow(row) {
        const values = [
            row.timestamp_ms,
            row.source,
            row.message,
            this._csvEscape(row.data)
        ];
        return values.join(',');
    }

    _csvEscape(value) {
        const v = value === null || value === undefined ? '' : String(value);
        if (v.includes(',') || v.includes('"') || v.includes('\n')) {
            return `"${v.replace(/"/g, '""')}"`;
        }
        return v;
    }

    _sortOrganismsByFitness(orgs) {
        const copy = Array.from(orgs);
        copy.sort((a, b) => {
            const fa = a.getFitness ? a.getFitness() : a.lifetime || 0;
            const fb = b.getFitness ? b.getFitness() : b.lifetime || 0;
            return fb - fa;
        });
        return copy;
    }

    _getTop20(sorted) {
        const n = sorted.length;
        if (n === 0) return [];
        let num_top = Math.floor(n * 0.2);
        num_top = Math.max(1, Math.min(num_top, n));
        return sorted.slice(0, num_top);
    }

    _calcRMSWeight(agent) {
        if (!agent.brain || !agent.brain.genome_weights) return 0;
        const gw = agent.brain.genome_weights;
        let sum = 0;
        for (let i = 0; i < gw.length; i++) sum += gw[i] * gw[i];
        const l2_norm = Math.sqrt(sum);
        return l2_norm / Math.sqrt(gw.length);
    }

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

    _formatWeights(weights) {
        if (!weights || weights.length === 0) return '';
        return Array.from(weights)
            .map(v => Number(v).toFixed(LOG_WEIGHTS_PRECISION))
            .join(' ');
    }

    _snapshotW1(agent) {
        if (!agent.brain || !agent.brain.genome_weights) return '';
        const gw = agent.brain.genome_weights;
        const w1_size = NNBrain.STATE_SIZE * NNBrain.HIDDEN_SIZE;
        if (gw.length < w1_size) return '';
        return this._formatWeights(gw.slice(0, w1_size));
    }

    _snapshotActive(agent) {
        if (!agent.brain || !agent.brain.active_weights) return '';
        return this._formatWeights(agent.brain.active_weights);
    }
    
}

module.exports = StatsPanel;