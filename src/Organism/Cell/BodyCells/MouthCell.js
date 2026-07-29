const CellStates  = require("../CellStates");
const BodyCell    = require("./BodyCell");
const Hyperparams = require("../../../Hyperparameters");

// Energy values for each edible cell state.
// Keyed by CellState name — O(1) lookup
// Default food is intentionally near-zero so agents are not incentivised
// to seek it; meaningful reward comes only from typed food.

//ONLY ACCESS TO FOOD ENERGY VALUES
const FOOD_ENERGY_VALUES = {
    'food':          0.01,   // base LifeEngine food — negligible score
    'low food':      0.5,
    'medium food':   1.0,
    'prestige food': 2.0,
};

class MouthCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.mouth, org, loc_col, loc_row);
    }

    performFunction() {
        const env    = this.org.env;
        const real_c = this.getRealCol();
        const real_r = this.getRealRow();
        for (const loc of Hyperparams.edibleNeighbors) {
            const cell = env.grid_map.cellAt(real_c + loc[0], real_r + loc[1]);
            this.eatNeighbor(cell, env);
        }
    }

    eatNeighbor(n_cell, env) {
        if (n_cell == null) return;

        // Check if this cell state is edible (exists in the energy map)
        const energy_value = FOOD_ENERGY_VALUES[n_cell.state.name];
        if (energy_value === undefined) return;

        // Tell the organism which food type was just eaten before incrementing
        // food_collected — AdvancedOrganism._foodValue() reads this to get the
        // correct energy reward.
        this.org.last_eaten_state = n_cell.state.name;

        env.changeCell(n_cell.col, n_cell.row, CellStates.empty, null);
        this.org.food_collected++;
    }
}

// Export the energy map so AdvancedOrganism and WorldEnvironment can reference it
MouthCell.FOOD_ENERGY_VALUES = FOOD_ENERGY_VALUES;

module.exports = MouthCell;