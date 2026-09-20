const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * Mean pairwise variance across the founding genomes of each generation — the
 * EA's genetic-diversity trace. Falls as selection converges the population;
 * a collapse to near zero means the gene pool has lost its spread.
 *
 * Backing series: FossilRecord.gen_variances, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
class GenVarianceChart extends GAChartController {
    constructor() {
        super("Genetic Variance Between Genomes", "Variance");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'purple',
            showInLegend: true, 
            name: "gen_variance",
            legendText: "Genome Variance",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var v = FossilRecord.gen_variances[i];
        this.data[0].dataPoints.push({x: g, y: v});
    }
}

module.exports = GenVarianceChart;
