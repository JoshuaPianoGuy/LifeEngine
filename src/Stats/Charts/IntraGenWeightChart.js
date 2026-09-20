const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * Mean absolute drift of active_weights away from the inherited genome over a
 * lifetime — how much WITHIN-LIFE learning (REINFORCE) moved the policy.
 * Flat at zero for the evolution condition, which runs no RL.
 *
 * Backing series: FossilRecord.gen_intra_weights, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
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
