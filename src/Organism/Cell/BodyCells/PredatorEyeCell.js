const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");
const Directions = require("../../Directions");

// Unlike prey EyeCell, this cell does not perform a 4-direction raycast.
// Predator detection is an omnidirectional radius scan (see PredatorBrain),
// which is simpler to tune via a single radius hyperparameter and doesn't
// depend on the predator facing the right way to notice prey. This cell
// exists mainly so the predator has a visually distinguishable "face" and
// marks the anatomy as having eyes, for parity/consistency with the prey
// cell-based body plan you described.
class PredatorEyeCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorEye, org, loc_col, loc_row);
        this.org.anatomy.has_eyes = true;
    }

    initInherit(parent) {
        super.initInherit(parent);
        this.direction = parent.direction;
    }

    initRandom() {
        this.direction = Directions.getRandomDirection();
    }

    initDefault() {
        this.direction = Directions.up;
    }

    getAbsoluteDirection() {
        var dir = this.org.rotation + this.direction;
        if (dir > 3) dir -= 4;
        return dir;
    }
}

module.exports = PredatorEyeCell;
