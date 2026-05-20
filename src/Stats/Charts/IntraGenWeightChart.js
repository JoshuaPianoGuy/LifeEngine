const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

class IntraGenWeightChart extends GAChartController {
    constructor() {
        super("Lifetime Learning (Active Weight Updates)", "Avg Diff from Genome");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'green',
            showInLegend: true, 
            name: "intra_weight",
            legendText: "Lifetime Learning Diff",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var w = FossilRecord.gen_intra_weights[i];
        this.data[0].dataPoints.push({x: g, y: w});
    }
}

module.exports = IntraGenWeightChart;
