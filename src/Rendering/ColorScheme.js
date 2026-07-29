const CellStates = require("../Organism/Cell/CellStates");

// ── Scheme-invariant colours ────────────────────────────────────────────────
// Two groups are deliberately IDENTICAL in every scheme below, because they
// carry experimental meaning rather than aesthetics and must stay readable
// whichever scheme is selected:
//
//   * Food tiers + landmarks — low #ff0000 / medium #ff8000 / prestige #00ffff
//     (pre-existing convention).
//   * Predator cells — a bright pink/magenta ramp (hue ~325-330deg). Predators
//     were previously dark red, which was nearly indistinguishable from low
//     food (#ff0000, hue 0) at cell_size 2 on a 500x500 grid: red bodies on a
//     field of red food. Pink is far enough round the wheel to separate them at
//     a glance while staying visible on every scheme's dark background.
//     Note this sits near neon's `killer` (#F82380); killer cells never appear
//     in the experiments (organism anatomy is fixed to mover/eyes/mouths) but
//     can show up in the editor.
//
// `predator eye-slit` is the exception — it is the scheme's own background
// colour, so it stays per-scheme.
const color_schemes = {
    "neon":{
        "empty":"#0E1318",
        "food":"#2F7AB7",
        "low food": "#ff0000",
        "medium food": "#ff8000",
        "prestige food": "#00ffff",
        "wall":"white",
        "cave": "#3a0d00",
        "low food landmark": "#ff4000",
        "medium food landmark": "#ffbf00",
        "prestige food landmark": "#00bfff",
        "mouth":"#DEB14D",
        "producer":"#15DE59",
        "mover":"#60D4FF",
        "killer":"#F82380",
        "armor":"#7230DB",
        "eye":"#B6C1EA",
        "eye-slit": "#0E1318",
        "predator body":"#FF1F8F",
        "predator mover":"#FF63B5",
        "predator eye":"#FFD1EA",
        "predator eye-slit": "#0E1318",
        "predator drain":"#FF00C8"
    },
    "classic":{
        "empty":"#121D29",
        "food":"green",
        "low food": "#ff0000",
        "medium food": "#ff8000",
        "prestige food": "#00ffff",
        "wall":"gray",
        "cave": "#3a0d00",
        "low food landmark": "#ff4000",
        "medium food landmark": "#ffbf00",
        "prestige food landmark": "#00bfff",
        "mouth":"orange",
        "producer":"pink",
        "mover":"blue",
        "killer":"red",
        "armor":"purple",
        "eye":"yellow",
        "eye-slit": "#121D29",
        "predator body":"#FF1F8F",
        "predator mover":"#FF63B5",
        "predator eye":"#FFD1EA",
        "predator eye-slit": "#121D29",
        "predator drain":"#FF00C8"
    },
    "soft":{
        "empty":"#0B0E11",
        "food":"#4F86B2",
        "low food": "#ff0000",
        "medium food": "#ff8000",
        "prestige food": "#00ffff",
        "wall":"#5F6F78",
        "cave": "#3a0d00",
        "low food landmark": "#ff4000",
        "medium food landmark": "#ffbf00",
        "prestige food landmark": "#00bfff",
        "mouth":"#B89A6A",
        "producer":"#4EA17B",
        "mover":"#6BA2C4",
        "killer":"#B06B85",
        "armor":"#7C69B5",
        "eye":"#AEB4C2",
        "eye-slit": "#0B0E11",
        "predator body":"#FF1F8F",
        "predator mover":"#FF63B5",
        "predator eye":"#FFD1EA",
        "predator eye-slit": "#0B0E11",
        "predator drain":"#FF00C8"
    },
    "dark":{
        "empty":"black",
        "food":"#225986",
        "low food": "#ff0000",
        "medium food": "#ff8000",
        "prestige food": "#00ffff",
        "wall":"#56616E",
        "cave": "#3a0d00",
        "low food landmark": "#ff4000",
        "medium food landmark": "#ffbf00",
        "prestige food landmark": "#00bfff",
        "mouth":"#AD8A45",
        "producer":"#198D4F",
        "mover":"#278BB0",
        "killer":"#992E5E",
        "armor":"#5632B5",
        "eye":"#8892B3",
        "eye-slit": "black",
        "predator body":"#FF1F8F",
        "predator mover":"#FF63B5",
        "predator eye":"#FFD1EA",
        "predator eye-slit": "black",
        "predator drain":"#FF00C8"
    },
    "grayscale":{
        "empty":"black",
        "food":"#777777",
        "low food": "#ff0000",
        "medium food": "#ff8000",
        "prestige food": "#00ffff",
        "wall":"#EEEEEE",
        "cave": "#3a0d00",
        "low food landmark": "#ff4000",
        "medium food landmark": "#ffbf00",
        "prestige food landmark": "#00bfff",
        "mouth":"#FFFFFF",
        "producer":"#CCCCCC",
        "mover":"#BBBBBB",
        "killer":"#AAAAAA",
        "armor":"#999999",
        "eye":"#888888",
        "eye-slit": "black",
        "predator body":"#FF1F8F",
        "predator mover":"#FF63B5",
        "predator eye":"#FFD1EA",
        "predator eye-slit": "black",
        "predator drain":"#FF00C8"
    }
}
const color_scheme_names = Object.keys(color_schemes);

// Renderer controls access to a canvas. There is one renderer for each canvas
class ColorSchemeSingleton {
    constructor() {
        this.world_env = null;
        this.editor_env = null;
    }
    setEnvironment(world_env, editor_env) {
        this.world_env = world_env;
        this.editor_env = editor_env;
    }
    loadColorScheme(scheme_name='neon') {
        const color_scheme = color_schemes[scheme_name];
        for (var state of CellStates.all) {
            state.color = color_scheme[state.name];
        }
        CellStates.eye.slit_color=color_scheme['eye-slit']
        CellStates.predatorEye.slit_color=color_scheme['predator eye-slit']
        for (var cell_type in color_scheme) {
            // Set colors for cell-type and cell-legend-type with proper selector syntax
            const cell_type_elem = $('#' + cell_type.replace(/ /g, '\\ ') + '.cell-type');
            const legend_type_elem = $('#' + cell_type.replace(/ /g, '\\ ') + '.cell-legend-type');
            cell_type_elem.css('background-color', color_scheme[cell_type]);
            legend_type_elem.css('background-color', color_scheme[cell_type]);
        }
        this.world_env.renderer.renderFullGrid(this.world_env.grid_map.grid);
        this.editor_env.renderer.renderFullGrid(this.editor_env.grid_map.grid);
    }
}

const ColorScheme = new ColorSchemeSingleton();
module.exports = {
    ColorScheme,
    color_scheme_names
}