## Table A1 — tests reported in Section 5


**Log-rank, time to sustained viability.** Generations until mean fitness sits at or above the viability threshold of 4.375 for ten consecutive generations (Figure 4). Runs that never cross are censored at generation 1000. The stratified test compares runs only against other runs on the same terrain and is the one we quote; the unstratified statistic is given beside it so the effect of the clustering is visible. Both were implemented directly and verified against lifelines in the unstratified case.

| Contrast | Detail | Statistic | n | p | p (Holm) |
|---|---|---:|---:|---:|---:|
| Baseline: A vs B vs C, stratified by map seed | 2 df, censored at generation 1000 | 18.41 | 300 | 1e-4 | -- |
| Baseline: A vs B vs C, unstratified | 2 df, censored at generation 1000 | 20.94 | 300 | 3e-5 | -- |
| Predator: A vs B vs C, stratified by map seed | 2 df, censored at generation 1000 | 176.7 | 300 | 4e-39 | -- |
| Predator: A vs B vs C, unstratified | 2 df, censored at generation 1000 | 188.2 | 300 | 1e-41 | -- |

**Fisher's exact, runs below viability.** Number of runs whose converged fitness fell below 4.375, predator environment only — no run in either condition fell below it in the baseline (Figure 2). Fisher is exact, so the zero cell is not a problem; a chi-square approximation here would not be trustworthy. Holm-adjusted over the two contrasts.

| Contrast | Detail | Statistic | n | p | p (Holm) |
|---|---|---:|---:|---:|---:|
| Predator: A vs B | 10/100 vs 0/100 | inf | 200 | 0.002 | 0.003 |
| Predator: A vs C | 10/100 vs 2/100 | 5.444 | 200 | 0.033 | 0.033 |

**Brown-Forsythe, equality of variance.** Spread of converged fitness across the 100 runs of each condition (Figure 2, Table 4). Brown--Forsythe is Levene's test centred on the median, which is what makes it survive the bimodal distributions the predator environment produces. Pairwise contrasts are Holm-adjusted over the two.

| Contrast | Detail | Statistic | n | p | p (Holm) |
|---|---|---:|---:|---:|---:|
| Baseline: A vs B vs C (omnibus) | sd(A)=0.0551  sd(B)=0.0666  sd(C)=0.1150 | 23.85 | 300 | 2e-10 | -- |
| Baseline: B vs A | sd(B)=0.0666 vs sd(A)=0.0551 | 4.507 | 200 | 0.035 | 0.035 |
| Baseline: B vs C | sd(B)=0.0666 vs sd(C)=0.1150 | 20.14 | 200 | 1e-5 | 2e-5 |
| Baseline: A vs C | sd(A)=0.0551 vs sd(C)=0.1150 | 36.76 | 200 | 7e-9 | 2e-8 |
| Predator: A vs B vs C (omnibus) | sd(A)=1.0174  sd(B)=0.0614  sd(C)=0.4913 | 13.19 | 300 | 3e-6 | -- |
| Predator: B vs A | sd(B)=0.0614 vs sd(A)=1.0174 | 20.42 | 200 | 1e-5 | 2e-5 |
| Predator: B vs C | sd(B)=0.0614 vs sd(C)=0.4913 | 40.68 | 200 | 1e-9 | 4e-9 |
| Predator: A vs C | sd(A)=1.0174 vs sd(C)=0.4913 | 2.881 | 200 | 0.091 | 0.091 |

**Wilcoxon signed-rank, inherited genomes.** Inherited genomes re-evaluated with learning disabled, paired across the ten map seeds at each founder-dump checkpoint (Figure 3). Positive favours the learning condition. Rank-based because n=10 is too small to assume normality; Holm-adjusted over the four checkpoints within each environment.

| Contrast | Detail | Statistic | n | p | p (Holm) |
|---|---|---:|---:|---:|---:|
| Baseline: B vs A at generation 250 | mean difference -0.67 raw f_off, 3/10 seeds favour B | 8 | 10 | 0.049 | 0.195 |
| Baseline: B vs A at generation 500 | mean difference -0.38 raw f_off, 4/10 seeds favour B | 18 | 10 | 0.375 | 1.000 |
| Baseline: B vs A at generation 750 | mean difference +0.18 raw f_off, 6/10 seeds favour B | 21 | 10 | 0.557 | 1.000 |
| Baseline: B vs A at generation 1000 | mean difference +0.15 raw f_off, 6/10 seeds favour B | 24 | 10 | 0.770 | 1.000 |
| Predator: B vs A at generation 250 | mean difference +7.19 raw f_off, 8/10 seeds favour B | 3 | 10 | 0.010 | 0.039 |
| Predator: B vs A at generation 500 | mean difference +5.59 raw f_off, 8/10 seeds favour B | 5 | 10 | 0.020 | 0.059 |
| Predator: B vs A at generation 750 | mean difference +1.31 raw f_off, 4/10 seeds favour B | 21 | 10 | 0.557 | 1.000 |
| Predator: B vs A at generation 1000 | mean difference +0.12 raw f_off, 5/10 seeds favour B | 25 | 10 | 0.846 | 1.000 |

## Table A2 — effective sample size

| Environment | Condition | ICC | Design effect | n_eff (of 100) |
|---|---|---:|---:|---:|
| Baseline | Evolution | 0.493 | 5.44 | 18 |
| Baseline | Learning | 0.287 | 3.58 | 28 |
| Baseline | Pure RL | 0.033 | 1.30 | 77 |
| Predator | Evolution | 0.076 | 1.68 | 59 |
| Predator | Learning | 0.372 | 4.35 | 23 |
| Predator | Pure RL | 0.011 | 1.10 | 91 |
