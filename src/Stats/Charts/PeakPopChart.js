const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * Peak simultaneous population reached during each generation. A proxy for how
 * well the cohort sustained itself: a healthy generation grows well past its
 * 100 founders, a failing one never exceeds them.
 *
 * Backing series: FossilRecord.gen_peak_pops, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
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
