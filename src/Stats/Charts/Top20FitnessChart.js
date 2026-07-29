const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

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
