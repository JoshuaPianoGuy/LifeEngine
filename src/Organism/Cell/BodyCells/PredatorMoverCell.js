const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");

class PredatorMoverCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorMover, org, loc_col, loc_row);
        this.org.anatomy.is_mover = true;
    }
}

module.exports = PredatorMoverCell;
