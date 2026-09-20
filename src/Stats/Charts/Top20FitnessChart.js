const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * Fitness per generation, plotted as two series: the mean over the top 20% of
 * organisms and the mean over all of them. The top-20% line is the less noisy
 * read on what selection is actually acting on.
 *
 * Backing series: FossilRecord.gen_top20_fitnesses / gen_avg_fitnesses, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
class Top20FitnessChart extends GAChartController {
    constructor() {
        super("Organism Fitness per Generation", "Fitness");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'orange',
            showInLegend: true, 
            name: "top20_fitness",
            legendText: "Top 20% Avg Fitness",
            dataPoints: []
        });
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'darkgray',
            showInLegend: true, 
            name: "avg_fitness",
            legendText: "Overall Avg Fitness",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var f = FossilRecord.gen_top20_fitnesses[i];
        var af = FossilRecord.gen_avg_fitnesses[i];
        this.data[0].dataPoints.push({x: g, y: f});
        this.data[1].dataPoints.push({x: g, y: af});
    }
}

module.exports = Top20FitnessChart;
