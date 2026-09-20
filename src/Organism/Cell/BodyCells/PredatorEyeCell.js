const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");
const Directions = require("../../Directions");

/**
 * Predator eye cell. Unlike the prey EyeCell, it does NOT perform a
 * 4-direction raycast: predator detection is an omnidirectional radius scan
 * (see PredatorBrain), which is simpler to tune through a single radius
 * hyperparameter and does not depend on the predator facing the right way to
 * notice prey.
 *
 * The cell exists mainly so the predator has a visually distinguishable
 * "face" and so the anatomy is marked as having eyes, keeping parity with the
 * prey's cell-based body plan. It still carries a `direction` so the sprite
 * orients correctly.
 */
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

    /**
     * This eye's facing in world space: its body-local direction rotated by
     * the organism's own rotation, wrapped to 0-3.
     * @returns {number} absolute direction index
     */
    getAbsoluteDirection() {
        var dir = this.org.rotation + this.direction;
        if (dir > 3) dir -= 4;
        return dir;
    }
}

module.exports = PredatorEyeCell;
