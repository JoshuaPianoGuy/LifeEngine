const PopulationChart = require("./Charts/PopulationChart");
const SpeciesChart = require("./Charts/SpeciesChart");
const MutationChart = require("./Charts/MutationChart");
const CellsChart = require("./Charts/CellsChart");
const PeakPopChart = require("./Charts/PeakPopChart");
const IntraGenWeightChart = require("./Charts/IntraGenWeightChart");
const InterGenWeightChart = require("./Charts/InterGenWeightChart");
const GenVarianceChart = require("./Charts/GenVarianceChart");
const Top20FitnessChart = require("./Charts/Top20FitnessChart");
const FossilRecord = require("./FossilRecord");
const NNBrain = require("../Organism/Perception/NNBrain");


const ChartSelections = [PopulationChart, SpeciesChart, CellsChart, MutationChart, PeakPopChart, IntraGenWeightChart, InterGenWeightChart, GenVarianceChart, Top20FitnessChart];

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

    // Capture organism data with ranking
    captureGenerationSnapshot() {
        const orgs = this.env.organisms;
        const generation_data = {
            generation_number: this.current_run_generation,
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
        this.all_generations.push(snapshot);
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
        
        this.current_run_generation++;
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
            
            if (org.brain._last_action !== null) {
                const action_names = ['↑ North', '→ East', '↓ South', '← West'];
                $('#org-brain-last-action').html(`<strong>Last Action:</strong> ${action_names[org.brain._last_action]}`);
            } else {
                $('#org-brain-last-action').html(`<strong>Last Action:</strong> (none yet)`);
            }
            
            if (org.brain._last_probs) {
                const probs = org.brain._last_probs.map((p, i) => `${['↑', '→', '↓', '←'][i]}:${(p*100).toFixed(0)}%`).join('  |  ');
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
                let sum = 0, sum_sq = 0;
                for (let i = 0; i < active.length; i++) {
                    sum += active[i];
                    sum_sq += active[i] * active[i];
                }
                const mean = sum / active.length;
                const variance = (sum_sq / active.length) - (mean * mean);
                const std = Math.sqrt(Math.max(0, variance));
                
                const learning_indicator = avg_drift < 0.01 ? '◯ Learning off' 
                                          : avg_drift < 0.1 ? '◐ Slow learning'
                                          : avg_drift < 0.5 ? '◑ Moderate learning'
                                          : '● Active learning';
                
                // Log to console for debugging
                console.log(`[ORG] Org at (${org.c}, ${org.r}): lifetime=${org.lifetime}, drift=${avg_drift.toFixed(4)}, max=${max_drift.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}, rl_enabled=${org.env.learning_enabled}`);
                
                $('#org-brain-weights').html(
                    `<strong>Weight Status:</strong> ${learning_indicator}<br/>` +
                    `• Avg Drift: ${avg_drift.toFixed(4)}<br/>` +
                    `• Max Drift: ${max_drift.toFixed(4)}<br/>` +
                    `• Active Mean: ${mean.toFixed(4)}<br/>` +
                    `• Std Dev: ${std.toFixed(4)}`
                ).show();
            } else {
                $('#org-brain-weights').hide();
            }
        }
    }

    downloadLogs() {
        // Capture current generation before downloading
        const snapshot = this.captureGenerationSnapshot();
        this.all_generations.push(snapshot);
        this.total_organisms_ever += snapshot.organism_count;

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
        link.setAttribute('download', `life-engine-logs-${Date.now()}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
}

module.exports = StatsPanel;