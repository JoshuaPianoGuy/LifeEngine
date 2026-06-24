const CellStates  = require("../CellStates");
const BodyCell    = require("./BodyCell");
const Hyperparams = require("../../../Hyperparameters");
const PredatorHyperparams = require("../../PredatorHyperparameters");

// Functional analogue of MouthCell, but drains prey energy directly on
// contact instead of eating food cells. Predators never gain energy from
// this — they don't need food, they only exist to impose an energy cost on
// prey (see PredatorOrganism / PredatorHyperparameters for rationale).
class PredatorDrainCell extends BodyCell {
    constructor(org, loc_col, loc_row) {
        super(CellStates.predatorDrain, org, loc_col, loc_row);
    }

    performFunction() {
        const env    = this.org.env;
        const real_c = this.getRealCol();
        const real_r = this.getRealRow();
        for (const loc of Hyperparams.edibleNeighbors) {
            const cell = env.grid_map.cellAt(real_c + loc[0], real_r + loc[1]);
            this.drainNeighbor(cell);
        }
    }

    drainNeighbor(n_cell) {
        if (n_cell == null || n_cell.owner == null) return;
        const target = n_cell.owner;
        if (target === this.org) return;
        if (!target.living) return;
        // Never drain other predators — only prey (AdvancedOrganism instances,
        // or any organism exposing an `energy` field driven by the energy
        // system). Checking for the predator flag rather than importing
        // PredatorOrganism here avoids a circular require.
        if (target.is_predator) return;
        if (typeof target.energy !== 'number') return;

        target.energy -= PredatorHyperparams.drainAmount;
        // Notify prey so it can inject a negative RL reward signal this tick.
        // The typeof guard keeps this safe if any non-AdvancedOrganism prey
        // type passes through (no import of AdvancedOrganism needed here).
        if (typeof target.notifyPredatorDrain === 'function') {
            target.notifyPredatorDrain(PredatorHyperparams.drainAmount);
        }
        // The prey's own update() loop only checks energy<=0 once per tick,
        // before its cells run — and predator drain happens on the
        // predator's tick, not the prey's. Without an explicit check here,
        // a drained-to-zero organism would keep acting (and could even eat
        // or reproduce) for up to one more full tick before its own
        // update() notices. Triggering die() immediately keeps "killed by a
        // predator" causally identical to "starved" from the prey's
        // perspective, just resolved at the moment of contact instead of
        // at the start of the prey's next tick.
        if (target.energy <= 0 && target.living) {
            target.die();
        }
    }
}

module.exports = PredatorDrainCell;