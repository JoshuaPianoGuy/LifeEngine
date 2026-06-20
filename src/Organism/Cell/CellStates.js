// A cell state is used to differentiate type and render the cell
class CellState{
    constructor(name) {
        this.name = name;
        this.color = 'black';
    }

    render(ctx, cell, size) {
        ctx.fillStyle = this.color;
        ctx.fillRect(cell.x, cell.y, size, size);
    }
}

class Empty extends CellState {
    constructor() {
        super('empty');
    }
}
class Food extends CellState {
    constructor() {
        super('food');
    }
}
class LowFood extends CellState{
    constructor(){
        super('low food');
    }
}
class MediumFood extends CellState{
    constructor(){
        super('medium food');
    }
}
class PrestigeFood extends CellState{
    constructor(){
        super('prestige food');
    }
}
class Wall extends CellState {
    constructor() {
        super('wall');
    }
}
class Cave extends CellState{
    constructor(){
        super('cave');
    }
}
class LowFoodLandmark extends CellState{
    constructor(){
        super('low food landmark')
    }
}
class MediumFoodLandmark extends CellState{
    constructor(){
        super('medium food landmark')
    }
}
class PrestigeFoodLandmark extends CellState{
    constructor(){
        super('prestige food landmark')
    }
}
class Mouth extends CellState {
    constructor() {
        super('mouth');
    }
}
class Producer extends CellState {
    constructor() {
        super('producer');
    }
}
class Mover extends CellState {
    constructor() {
        super('mover');
    }
}
class Killer extends CellState {
    constructor() {
        super('killer');
    }
}
class Armor extends CellState {
    constructor() {
        super('armor');
    }
}
class Eye extends CellState {
    constructor() {
        super('eye');
        this.slit_color = 'black';
    }
    render(ctx, cell, size) {
        ctx.fillStyle = this.color;
        ctx.fillRect(cell.x, cell.y, size, size);
        if(size <= 1)
            return;
        var half = size/2;
        var x = -(size)/8
        var y = -half;
        var h = size/2 + size/4;
        var w = size/4;
        ctx.translate(cell.x+half, cell.y+half);
        ctx.rotate((cell.cell_owner.getAbsoluteDirection() * 90) * Math.PI / 180);
        ctx.fillStyle = this.slit_color;
        ctx.fillRect(x, y, w, h);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
}

// ── Predator body cells ──────────────────────────────────────────────────────
// Distinct from the prey's 'mover'/'eye'/'mouth' states so the predator is
// visually and statefully separable. The predator's anatomy is fixed and
// never mutated/evolved, so these are intentionally excluded from
// CellStates.living (see defineLists below).

// Body/core cell — purely structural, no behaviour of its own.
class PredatorBody extends CellState {
    constructor() {
        super('predator body');
    }
}
// Movement cell — marks the organism as a mover, same role as prey MoverCell.
class PredatorMover extends CellState {
    constructor() {
        super('predator mover');
    }
}
// Sensing cell — used internally for radius-based detection (not a raycast
// eye like prey EyeCell). Rendered the same way as a prey eye for clarity.
class PredatorEye extends CellState {
    constructor() {
        super('predator eye');
        this.slit_color = 'black';
    }
    render(ctx, cell, size) {
        ctx.fillStyle = this.color;
        ctx.fillRect(cell.x, cell.y, size, size);
        if(size <= 1)
            return;
        var half = size/2;
        var x = -(size)/8
        var y = -half;
        var h = size/2 + size/4;
        var w = size/4;
        ctx.translate(cell.x+half, cell.y+half);
        ctx.rotate((cell.cell_owner.getAbsoluteDirection() * 90) * Math.PI / 180);
        ctx.fillStyle = this.slit_color;
        ctx.fillRect(x, y, w, h);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
}
// Drain cell — on contact, subtracts energy from prey. Functional analogue of
// MouthCell, but drains organism.energy directly instead of eating food cells.
class PredatorDrain extends CellState {
    constructor() {
        super('predator drain');
    }
}

const CellStates = {
    empty: new Empty(),
    food: new Food(),
    lowFood: new LowFood(),
    mediumFood: new MediumFood(),
    prestigeFood: new PrestigeFood(),
    wall: new Wall(),
    cave: new Cave(),
    lowFoodLandmark: new LowFoodLandmark(),
    mediumFoodLandmark: new MediumFoodLandmark(),
    prestigeFoodLandmark: new PrestigeFoodLandmark(),
    mouth: new Mouth(),
    producer: new Producer(),
    mover: new Mover(),
    killer: new Killer(),
    armor: new Armor(),
    eye: new Eye(),
    predatorBody: new PredatorBody(),
    predatorMover: new PredatorMover(),
    predatorEye: new PredatorEye(),
    predatorDrain: new PredatorDrain(),
    defineLists() {
        this.all = [this.empty, this.food, this.lowFood, this.mediumFood, this.prestigeFood, this.wall, this.cave, this.lowFoodLandmark, this.mediumFoodLandmark, this.prestigeFoodLandmark, this.mouth, this.producer, this.mover, this.killer, this.armor, this.eye, this.predatorBody, this.predatorMover, this.predatorEye, this.predatorDrain]
        this.living = [this.mouth, this.producer, this.mover, this.killer, this.armor, this.eye];
        // Predator cell states deliberately excluded from `living` — that list
        // drives base Organism.mutate()'s random cell-type selection for prey
        // anatomy mutation, and predator anatomy is fixed (never mutated).
        // Kept as a separate list for code that needs to identify predator cells
        // generically (e.g. "is this cell part of a predator's body").
        this.predator = [this.predatorBody, this.predatorMover, this.predatorEye, this.predatorDrain];
    },
    getRandomName: function() {
        return this.all[Math.floor(Math.random() * this.all.length)].name;
    },
    getRandomLivingType: function() {
        return this.living[Math.floor(Math.random() * this.living.length)];
    },
    isPredatorState: function(state) {
        return this.predator.indexOf(state) !== -1;
    }
}

CellStates.defineLists();

module.exports = CellStates;
