const WorldConfig = {
    headless: false,
    clear_walls_on_reset: false,
    auto_reset: true,
    auto_pause: false,
    brush_size: 2,
    learning_enabled: true,  // Set to true to run RL experiment with AdvancedOrganism
    experiment_mode: 'standard', // 'standard' | 'frozen_pg' | 'pure_rl'
}

module.exports = WorldConfig;