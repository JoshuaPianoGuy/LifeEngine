const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");

/**
 * Purely structural predator cell — no behaviour, just occupies a grid square
 * as part of the predator's fixed anatomy.
 *
 * It is the analogue of a prey's "center" cell, which needs no dedicated class
 * in the prey codebase because prey centre cells are usually mouths.
 * Predators get an explicit inert body cell instead, so the core of the body
 * is not forced to be a drain or an eye.
 */
class PredatorBodyCell extends BodyCell {
    /**
     * @param {PredatorOrganism} org      owning organism
     * @param {number}           loc_col  column offset within the body plan
     * @param {number}           loc_row  row offset within the body plan
     */
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorBody, org, loc_col, loc_row);
    }
}

module.exports = PredatorBodyCell;
