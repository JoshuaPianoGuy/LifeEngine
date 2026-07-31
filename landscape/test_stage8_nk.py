"""Ground-truth recovery test for stage8_nk analyze/model/sim.

Builds a KNOWN second-order model on M loci, fabricates a results.csv exactly as
run_probe.js would write it (with per-map terrain offsets + per-probe noise), then
checks that `analyze` recovers a_i, eps_ij, K and the Walsh split.
"""
import itertools, json, os, subprocess, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'landscape'))
OUT = os.path.join(REPO, 'landscape/out/stage8_synth')
os.makedirs(OUT, exist_ok=True)

M_TOTAL = 70            # analyze always works in the full 70-locus frame
M = 14                  # loci actually used
R = 8
NOISE = 0.15            # per-probe sigma_fixed (dynamics noise, same map)
K_TRUE = 4              # each locus gets 4 real partners
rng = np.random.default_rng(7)

loci = list(range(M))
pairs = list(itertools.combinations(loci, 2))

# ── ground truth ─────────────────────────────────────────────────────────────
f0 = 3.0
a_true = rng.normal(0, 0.8, M)
eps_true = np.zeros((M, M))
# pick K_TRUE*M/2 edges
sel = rng.choice(len(pairs), size=K_TRUE * M // 2, replace=False)
for k in sel:
    i, j = pairs[k]
    v = rng.normal(0, 0.9)
    eps_true[i, j] = eps_true[j, i] = v

def f_true(bits):
    x = np.zeros(M)
    for i in bits:
        x[i] = 1.0
    return f0 + a_true @ x + sum(eps_true[i, j] * x[i] * x[j] for i, j in pairs)

# per-map terrain offset: HUGE (sigma_vary ~ 7) — must cancel in the contrast
terrain = rng.normal(0, 7.0, 2 * R)

rows = ['id,rl,map_index,ticks,mean_fitness,std_clones,n_clones']
def emit(job_id, r, tag, bits, map_off):
    m = map_off + r
    val = f_true(bits) + terrain[m] + rng.normal(0, NOISE)
    rows.append(f'{job_id},{1 if tag == "on" else 0},{m},2000,{val:.6f},1.0,100')

null_pairs = [pairs[k] for k in rng.choice(len(pairs), size=12, replace=False)]
null_loci = sorted({i for p in null_pairs for i in p})

for r in range(R):
    for tag in ('off', 'on'):
        emit(f'bg_r{r}_{tag}', r, tag, [], 0)
        for i in loci:
            emit(f's{i}_r{r}_{tag}', r, tag, [i], 0)
        for (i, j) in pairs:
            emit(f'd{i}_{j}_r{r}_{tag}', r, tag, [i, j], 0)
        # null replica on a disjoint map set
        emit(f'nbg_r{r}_{tag}', r, tag, [], R)
        for i in null_loci:
            emit(f'ns{i}_r{r}_{tag}', r, tag, [i], R)
        for (i, j) in null_pairs:
            emit(f'nd{i}_{j}_r{r}_{tag}', r, tag, [i, j], R)

open(os.path.join(OUT, 'results.csv'), 'w').write('\n'.join(rows) + '\n')

json.dump({'stage': 8, 'M': M_TOTAL, 'loci': loci,
           'n_pairs': len(pairs), 'n_pairs_total': len(pairs),
           'pairs_sampled': False,
           'background': 'synthetic', 'background_label': 'SYNTHETIC',
           'alpha': 0.25, 'allele_seed': 0, 'repeats': R, 'map_start': 0,
           'maps': list(range(R)), 'rl_passes': [False, True],
           'null_pairs': [[int(i), int(j)] for i, j in null_pairs],
           'config_overrides': None, 'mut_prob': 0.03, 'mut_sigma': 0.1,
           'block_touch_prob_mean': 0.817, 'theta0_norm': 1.0},
          open(os.path.join(OUT, 'meta.json'), 'w'), indent=2)

# ── run analyze ──────────────────────────────────────────────────────────────
py = sys.executable
subprocess.run([py, os.path.join(REPO, 'landscape/stage8_nk.py'), 'analyze',
                '--out-dir', OUT, '--z', '2'], check=True, cwd=REPO)

# ── check recovery ───────────────────────────────────────────────────────────
d = np.load(os.path.join(OUT, 'epistasis_off.npz'))
a_hat, eps_hat, sig = d['a'], d['eps'], d['sig']
K_i_hat = d['K_i']

a_err = np.abs(a_hat[:M] - a_true)
eu = np.triu_indices(M, 1)
e_hat_u = eps_hat[:M, :M][eu]
e_true_u = eps_true[eu]
e_err = np.abs(e_hat_u - e_true_u)

K_i_true = (eps_true != 0).sum(axis=1) * (M_TOTAL - 1) / (M - 1)  # same extrapolation

print('\n' + '=' * 68)
print('GROUND-TRUTH RECOVERY')
print('=' * 68)
print(f'  a_i     : max abs err {a_err.max():.4f}   (noise-implied SE '
      f'{NOISE*np.sqrt(2/R):.4f})')
print(f'  eps_ij  : max abs err {e_err.max():.4f}   (noise-implied SE '
      f'{NOISE*np.sqrt(4/R):.4f})')
print(f'  eps corr(true, hat) = {np.corrcoef(e_true_u, e_hat_u)[0,1]:.5f}')
real = e_true_u != 0
print(f'  detection: {sig[:M,:M][eu][real].sum()}/{real.sum()} real edges found, '
      f'{sig[:M,:M][eu][~real].sum()}/{(~real).sum()} false positives')
print(f'  K (extrapolated to 70 loci): true {K_i_true.mean():.2f}  '
      f'measured {np.nanmean(K_i_hat):.2f}')
print(f'  K in the {M}-locus frame:      true {(eps_true!=0).sum(axis=1).mean():.2f}  '
      f'measured {np.nanmean(K_i_hat)*(M-1)/(M_TOTAL-1):.2f}   (K_TRUE={K_TRUE})')

s = json.load(open(os.path.join(OUT, 'nk_summary.json')))
# true Walsh split in the M-locus frame
w1 = a_true / 2 + eps_true.sum(axis=1) / 4
w2 = eps_true[eu] / 4
V1t, V2t = (w1 ** 2).sum(), (w2 ** 2).sum()
print(f'  Walsh: true epistasis fraction {V2t/(V1t+V2t)*100:.1f}%  '
      f'measured {s["conditions"]["off"]["epistasis_fraction"]*100:.1f}%')
print(f'  empirical SE(eps) = {s["conditions"]["off"]["eps_se_empirical"]:.4f}  '
      f'(true {NOISE*np.sqrt(4/R):.4f})')
print(f'  terrain sigma was {terrain.std():.2f} — if the pairing failed these '
      f'errors would be O(10)')
