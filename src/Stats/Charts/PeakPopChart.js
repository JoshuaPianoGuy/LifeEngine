const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

class PeakPopChart extends GAChartController {
    constructor() {
        super("Peak Population per Generation", "Population");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'blue',
            showInLegend: true, 
            name: "peak_pop",
            legendText: "Peak Population",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var p = FossilRecord.gen_peak_pops[i];
        this.data[0].dataPoints.push({x: g, y: p});
    }
}

module.exports = PeakPopChart;
