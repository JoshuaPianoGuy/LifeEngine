const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

class TotalAgentsChart extends GAChartController {
    constructor() {
        super("Total Evaluated Agents per Generation", "Agents Born");
    }

    setData() {
        this.clear();
        this.data.push({
            type: "line",
            markerType: "circle",
            color: 'teal',
            showInLegend: true, 
            name: "total_agents",
            legendText: "Total Agents",
            dataPoints: []
        });
        this.addAllDataPoints();
    }

    addDataPoint(i) {
        var g = FossilRecord.gen_record[i];
        var p = FossilRecord.gen_total_agents[i];
        this.data[0].dataPoints.push({x: g, y: p});
    }
}

module.exports = TotalAgentsChart;
