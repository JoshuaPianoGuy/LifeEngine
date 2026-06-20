const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");

// Purely structural predator cell — no behaviour, just occupies a grid square
// as part of the predator's fixed anatomy (the analogue of a prey's "center"
// cell, which doesn't need its own dedicated class in the prey codebase
// because prey center cells are usually mouths; predators get an explicit
// inert body cell instead so the core of the body isn't forced to be a
// drain or eye).
class PredatorBodyCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorBody, org, loc_col, loc_row);
    }
}

module.exports = PredatorBodyCell;
