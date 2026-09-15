## Main-text table — the tests behind the four claims

| Environment | Comparison | Test | Statistic | p | p (Holm) |
|---|---|---|---:|---:|---:|
| Baseline | A vs B vs C, time to viability | Log-rank (seed-stratified) | chi^2(2) = 18.41 | 1e-4 | -- |
| Baseline | A vs B vs C, converged fitness variance | Brown-Forsythe | W = 23.85 | 2e-10 | -- |
| Baseline | B vs A, inherited genome fitness at generation 250 | Wilcoxon signed-rank (paired by seed) | W = 8 | 0.049 | 0.195 |
| Baseline | B vs A, inherited genome fitness at generation 500 | Wilcoxon signed-rank (paired by seed) | W = 18 | 0.375 | 1.000 |
| Baseline | B vs A, inherited genome fitness at generation 750 | Wilcoxon signed-rank (paired by seed) | W = 21 | 0.557 | 1.000 |
| Baseline | B vs A, inherited genome fitness at generation 1000 | Wilcoxon signed-rank (paired by seed) | W = 24 | 0.770 | 1.000 |
| Predator | A vs B vs C, time to viability | Log-rank (seed-stratified) | chi^2(2) = 176.7 | 4e-39 | -- |
| Predator | A vs B, runs below threshold | Fisher's exact | OR = inf | 0.002 | 0.003 |
| Predator | A vs C, runs below threshold | Fisher's exact | OR = 5.444 | 0.033 | 0.033 |
| Predator | A vs B vs C, converged fitness variance | Brown-Forsythe | W = 13.19 | 3e-6 | -- |
| Predator | B vs A, inherited genome fitness at generation 250 | Wilcoxon signed-rank (paired by seed) | W = 3 | 0.010 | 0.039 |
| Predator | B vs A, inherited genome fitness at generation 500 | Wilcoxon signed-rank (paired by seed) | W = 5 | 0.020 | 0.059 |
| Predator | B vs A, inherited genome fitness at generation 750 | Wilcoxon signed-rank (paired by seed) | W = 21 | 0.557 | 1.000 |
| Predator | B vs A, inherited genome fitness at generation 1000 | Wilcoxon signed-rank (paired by seed) | W = 25 | 0.846 | 1.000 |
