# Weight Analysis Tools

This directory contains utilities for analyzing neural network weight changes across generations in the LifeEngine learning-vs-evolution experiment.

## Scripts

### `analyzeWeights.js`

Analyzes and visualizes how neural network weight magnitudes change across generations.

**Features:**
- Reads generation log CSV files produced by the Logger
- Logs store RMS-normalized weights: sqrt(L2_norm² / num_weights)
- Shows per-generation weight statistics with ASCII visualizations
- With [-1, 1] bounded weights, typical RMS values are in [0, 1] range
- Identifies notable generational changes (>5% variation)
- Computes overall trends and statistics

**Usage:**

```bash
node analysis/analyzeWeights.js path/to/generation_log.csv
```

**Example output:**

```
==================================================
WEIGHT MAGNITUDE ANALYSIS ACROSS GENERATIONS
==================================================

Gen │ Condition        │ Weight Mag   │ Normalized   │ Change     │ Learned Diff    │ ...
────┼──────────────────┼──────────────┼──────────────┼────────────┼─────────────────┼────
1   │ learning         │ 0.5234       │ [████░░░░░░] │ →          │ 0.000000        │ ...
2   │ learning         │ 0.5189       │ [████░░░░░░] │ ↓ 0.0045   │ 0.000031        │ ...
3   │ learning         │ 0.5412       │ [█████░░░░░] │ ↑ 0.0223   │ 0.000058        │ ...
...

SUMMARY STATISTICS:
  Generations analyzed: 50
  Min weight magnitude: 0.512345
  Max weight magnitude: 0.623456
  Average magnitude:    0.562890
  Normalized range:     [0.013, 0.016]

TREND:
  ↑ Weights INCREASING (0.023456 net change over entire run)
  Risk: weights diverging from bounded range despite clipping
```

## Interpreting Results

### Weight Magnitude (RMS-normalized)

The **normalized weight magnitude** (RMS = Root Mean Square) indicates the typical/average weight magnitude:

- **Small magnitude** (< 0.1): Conservative network, underutilizing capacity
- **Medium magnitude** (0.1 - 0.3): Well-scaled weights, good network utilization
- **Large magnitude** (> 0.5): Weights near bounds, potential saturation

With weights bounded to [-1, 1], the RMS of a typical neural network should be in [0, 1] range.

The **RMS weight** is calculated as: $\text{RMS} = \frac{\sqrt{\sum w_i^2}}{\sqrt{n}}$

This is more interpretable than raw L2 norm because it shows the "typical" weight value independent of network size.

### Changes Per Generation

- **↑ Increasing trend**: GA selection is favoring networks with larger weight magnitudes
- **↓ Decreasing trend**: GA selection favors more conservative weight scales
- **→ Stable trend**: Weight magnitudes remain consistent across generations

### Learned Weight Difference

In Condition A (with RL enabled), this shows how much within-lifetime learning has modified weights compared to the starting genome. High values indicate significant RL-induced drift.

## Integration with Logger

The Logger class in `src/Logger.js` automatically tracks:
- `avg_network_weight_mag`: Average **RMS-normalized** weight magnitude per generation
  - Calculated as: sqrt(L2_norm² / num_weights)
  - Shows typical weight value, ranges [0, 1] for [-1, 1] bounded networks
- `avg_learned_weight_diff`: Average weight drift from RL (Condition A only)
  - Only meaningful when RL is enabled (Condition A)
- `genome_variance`: Population diversity in weight space

Generation logs are saved as CSV files via:
```javascript
logger.downloadGenerations();  // Downloads CSV with all generation summaries
```

## Future Enhancements

Potential additions:
- Weight distribution histograms per layer
- Individual weight tracking across generations
- Heat maps showing which weights change most
- Comparison between Condition A (learning) and Condition B (natural selection)
- Spectral analysis of weight evolution
