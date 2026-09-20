const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");

/**
 * Marks the predator's anatomy as mobile. Purely a flag-setting cell: it has
 * no per-tick behaviour, but its presence sets anatomy.is_mover, which is what
 * lets PredatorBrain issue movement for this organism.
 */
class PredatorMoverCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorMover, org, loc_col, loc_row);
        this.org.anatomy.is_mover = true;
    }
}

module.exports = PredatorMoverCell;
