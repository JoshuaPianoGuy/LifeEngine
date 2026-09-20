const FossilRecord = require("../FossilRecord");

/**
 * Abstract base for every per-generation chart in the browser Stats panel.
 * Owns the CanvasJS chart object and the incremental-update bookkeeping;
 * subclasses supply only the series definitions and how to read one
 * generation out of FossilRecord.
 *
 * Subclasses MUST override setData() and addDataPoint(); the base versions
 * alert() rather than throw, which is the existing behaviour.
 *
 * All charts share the x-axis FossilRecord.gen_record (generation number).
 */
class GAChartController {
    /**
     * @param {string} title   chart title
     * @param {string} [y_axis] y-axis label
     * @param {string} [note]   caption rendered under the chart (#chart-note)
     */
    constructor(title, y_axis="", note="") {
        this.data = [];
        this.chart = new CanvasJS.Chart("chartContainer", {
            zoomEnabled: true,
            title:{
                text: title
            },
            axisX:{
                title: "Generation",
                minimum: 0,
            },
            axisY:{
                title: y_axis
            },
            data: this.data
        });
        this.chart.render();
        $('#chart-note').text(note);
    }

    /**
     * Define this chart's series and seed them from the full FossilRecord.
     * @abstract
     */
    setData() {
        alert("Must override updateData!");
    }

    setMinimum() {
        var min = 0;
        if (this.data[0].dataPoints.length > 0)
            min = this.data[0].dataPoints[0].x;
        this.chart.options.axisX.minimum = min;
    }

    addAllDataPoints(){
        for (var i in FossilRecord.gen_record){
            this.addDataPoint(i);
        }
    }

    render() {
        this.setMinimum();
        this.chart.render();
    }

    /**
     * Sync the chart to FossilRecord without redrawing everything: walk back
     * from the newest generation to find how many points are missing, append
     * those, then drop leading points if FossilRecord has rolled off older
     * generations (it keeps a bounded window).
     */
    updateData() {
        if (!FossilRecord.gen_record || FossilRecord.gen_record.length === 0) return;
        
        let record_size = FossilRecord.gen_record.length;
        let data_points = this.data[0].dataPoints;
        let newest_t = -1;
        if (data_points.length > 0) {
            newest_t = this.data[0].dataPoints[data_points.length-1].x;
        }
        let to_add = 0;
        let cur_t = FossilRecord.gen_record[record_size-1];
        
        // count new points
        while (cur_t !== newest_t && record_size - to_add - 1 >= 0) {
            to_add++;
            let idx = record_size - to_add - 1;
            if (idx >= 0) {
                cur_t = FossilRecord.gen_record[idx];
            } else {
                break;
            }
        }
        
        // add missing
        this.addNewest(to_add);

        while (data_points.length > FossilRecord.gen_record.length) {
            this.removeOldest();
        }
    }

    addNewest(to_add) {
        for (let i=to_add; i>0; i--) {
            let j = FossilRecord.gen_record.length-i;
            this.addDataPoint(j);
        }
    }

    removeOldest() {
        for (var dps of this.data) {
            dps.dataPoints.shift();
        }
    }

    /**
     * Append generation `i` of FossilRecord to this chart's series.
     * @abstract
     * @param {number} i index into FossilRecord.gen_record
     */
    addDataPoint(i) {
        alert("Must override addDataPoint");
    }

    /** Empty every series and redraw (used on reset / condition switch). */
    clear() {
        for (var item of this.data) {
            item.dataPoints.length = 0;
        }
        this.chart.render();
    }
}

module.exports = GAChartController;
