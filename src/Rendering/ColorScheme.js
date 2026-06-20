const CellStates = require("../Organism/Cell/CellStates");

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
        "predator body":"#3D0000",
        "predator mover":"#5C0011",
        "predator eye":"#FF003C",
        "predator eye-slit": "#0E1318",
        "predator drain":"#8C0030"
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
        "predator body":"#4A0000",
        "predator mover":"#660000",
        "predator eye":"#FF1A1A",
        "predator eye-slit": "#121D29",
        "predator drain":"#8B0000"
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
        "predator body":"#4D1A22",
        "predator mover":"#6B2531",
        "predator eye":"#C24B5E",
        "predator eye-slit": "#0B0E11",
        "predator drain":"#8C3344"
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
        "predator body":"#3D0A14",
        "predator mover":"#591020",
        "predator eye":"#C41E3A",
        "predator eye-slit": "black",
        "predator drain":"#7A1530"
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
        "predator body":"#222222",
        "predator mover":"#333333",
        "predator eye":"#444444",
        "predator eye-slit": "black",
        "predator drain":"#111111"
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