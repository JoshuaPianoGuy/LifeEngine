const FossilRecord = require("../FossilRecord");

class ChartController {
    constructor(title, y_axis="", note="") {
        this.data = [];
        this.chart = new CanvasJS.Chart("chartContainer", {
            zoomEnabled: true,
            title:{
                text: title
            },
            axisX:{
                title: "Ticks",
                minimum: 0,
            },
            axisY:{
                title: y_axis,
                minimum: 0,
            },
            data: this.data
        });
        this.chart.render();
        $('#chart-note').text(note);
    }

    setData() {
        alert("Must override updateData!");
    }

    setMinimum() {
        var min = 0;
        if (this.data[0].dataPoints != [])
            min = this.data[0].dataPoints[0].x;
        this.chart.options.axisX.minimum = min;
    }

    addAllDataPoints(){
        for (var i in FossilRecord.tick_record){
        this.addDataPoint(i)}
        // Limit data points to prevent UI freeze with large simulations
        // Show only the last 5000 ticks to keep chart responsive

    }

    render() {
        this.setMinimum();
        this.chart.render();
    }

    updateData() {
        const record_size = FossilRecord.tick_record.length;
        const data_points = this.data[0].dataPoints;
        if (record_size === 0) return;

        // Find where to resume from the existing chart tail. If the tail tick is
        // no longer present in the rolling record, rebuild from the current window.
        let start_index = 0;
        if (data_points.length > 0) {
            const newest_t = data_points[data_points.length - 1].x;
            const last_seen_idx = FossilRecord.tick_record.lastIndexOf(newest_t);
            start_index = last_seen_idx >= 0 ? last_seen_idx + 1 : 0;
        }

        for (let i = start_index; i < record_size; i++) {
            this.addDataPoint(i);
        }

        // remove oldest datapoints until the chart is the same size as the saved records
        while (data_points.length > FossilRecord.tick_record.length) {
            this.removeOldest();
        }
    }

    addNewest(to_add) {
        for (let i=to_add; i>0; i--) {
            let j = FossilRecord.tick_record.length-i;
            this.addDataPoint(j);
        }
    }

    removeOldest() {
        for (var dps of this.data) {
            dps.dataPoints.shift();
        }
    }

    addDataPoint(i) {
        alert("Must override addDataPoint")
    }

    clear() {
        this.data.length = 0;
        this.chart.render();
    }
}

module.exports = ChartController;