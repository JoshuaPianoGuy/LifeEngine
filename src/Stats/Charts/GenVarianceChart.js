const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

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
