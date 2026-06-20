const CellStates = require("../CellStates");
const BodyCell = require("./BodyCell");
const Hyperparams = require("../../../Hyperparameters");
const Directions = require("../../Directions");
const Observation = require("../../Perception/Observation")

class EyeCell extends BodyCell{
    constructor(org, loc_col, loc_row){
        super(CellStates.eye, org, loc_col, loc_row);
        this.org.anatomy.has_eyes = true;
        this._cached_obs = new Observation(null, 0, 0);
    }

    initInherit(parent) {
        // deep copy parent values
        super.initInherit(parent);
        this.direction = parent.direction;
    }
    
    initRandom() {
        // initialize values randomly
        this.direction = Directions.getRandomDirection();
    }

    initDefault() {
        // initialize to default values
        this.direction = Directions.up;
    }

    getAbsoluteDirection() {
        var dir = this.org.rotation + this.direction;
        if (dir > 3)
            dir -= 4;
        return dir;
    }

    performFunction() {
        if (this.org.brain && this.org.brain.is_nnbrain) return; // NNBrain does its own explicit raycasts, avoid double work
        var obs = this.look();
        this.org.brain.observe(obs);
    }

    look() {
        // Predator cells are a distinct percept (index 11, see
        // NNBrain.PERCEPT_INDEX) from prey organism cells (index 10) — a
        // raycast hitting a predator reports "predator here" specifically,
        // not just "something occupies this cell" or "nothing." This is
        // what makes direct, perception-based avoidance learnable/evolvable,
        // as opposed to only an indirect energy-loss consequence of contact
        // (see PredatorDrainCell).
        var env = this.org.env;
        var direction = this.getAbsoluteDirection();
        var addCol = 0;
        var addRow = 0;
        switch(direction) {
            case Directions.up:    addRow = -1; break;
            case Directions.down:  addRow = 1; break;
            case Directions.right: addCol = 1; break;
            case Directions.left:  addCol = -1; break;
        }
        var start_col = this.getRealCol();
        var start_row = this.getRealRow();
        var col = start_col;
        var row = start_row;
        var cell = null;
        for (var i=0; i<Hyperparams.lookRange; i++){
            col+=addCol;
            row+=addRow;
            cell = env.grid_map.cellAt(col, row);
            if (cell == null) {
                break;
            }
            if (cell.owner === this.org && Hyperparams.seeThroughSelf) {
                continue;
            }
            if (cell.state !== CellStates.empty) {
                this._cached_obs.cell = cell;
                this._cached_obs.distance = Math.abs(start_col-col) + Math.abs(start_row-row);
                this._cached_obs.direction = direction;
                return this._cached_obs;
            }
        }
        this._cached_obs.cell = cell;
        this._cached_obs.distance = Hyperparams.lookRange;
        this._cached_obs.direction = direction;
        return this._cached_obs;
    }
}

module.exports = EyeCell;