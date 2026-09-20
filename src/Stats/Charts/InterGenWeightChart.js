const GAChartController = require("./GAChartController");
const FossilRecord = require("../FossilRecord");

/**
 * RMS change between one generation's founding genome and the previous
 * generation's — how far EVOLUTION moved the inherited weights per
 * generation. This is the between-generation counterpart to
 * IntraGenWeightChart's within-life measure.
 *
 * Backing series: FossilRecord.gen_inter_weights, indexed in step with
 * FossilRecord.gen_record (the generation number on the x-axis).
 *
 * Browser Stats-panel chart only; the headless/HPC runs log the same
 * quantities to generations.csv instead.
 */
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
