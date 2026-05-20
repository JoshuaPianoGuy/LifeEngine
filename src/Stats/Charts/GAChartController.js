const FossilRecord = require("../FossilRecord");

class GAChartController {
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

    addDataPoint(i) {
        alert("Must override addDataPoint");
    }

    clear() {
        for (var item of this.data) {
            item.dataPoints.length = 0;
        }
        this.chart.render();
    }
}

module.exports = GAChartController;
