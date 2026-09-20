const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * Total organisms born and evaluated in each generation — founders plus every
 * asexual descendant. Tracks reproductive throughput rather than fitness.
 *
 * Backing series: FossilRecord.gen_total_agents, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
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
