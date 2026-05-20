const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

class InterGenWeightChart extends GAChartController {
    constructor() {
        super("Evolutionary Shift (Inter-generational RMS Change)", "RMS Change from Prev Gen");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'red',
            showInLegend: true, 
            name: "inter_weight",
            legendText: "Evolutionary RMS Shift",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var w = FossilRecord.gen_inter_weights[i];
        this.data[0].dataPoints.push({x: g, y: w});
    }
}

module.exports = InterGenWeightChart;
