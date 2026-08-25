"""Ground-truth tests for the Stage-9 difficulty metrics (ll_difficulty.py).

Each metric is checked against a landscape whose answer is known analytically —
a plane, a cone, a two-basin surface, a pure AR(1) series — so a regression in
the metric shows up as a number that no longer matches its closed form, not as a
plot that looks slightly different.

    python landscape/test_ll_difficulty.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'landscape'))

import ll_difficulty as ld  # noqa: E402

FAILS = []


def check(name, got, want, tol=1e-9):
    ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
    print(f'  {"PASS" if ok else "FAIL"}  {name}: got {got!r} want {want!r} (tol {tol})')
    if not ok:
        FAILS.append(name)


def check_true(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name} {detail}')
    if not cond:
        FAILS.append(name)


N = 21
ax = np.linspace(-10, 10, N)
A, B = ld.grid_coords(ax, ax)

# ── A1 viable fraction ────────────────────────────────────────────────────────
print('\nA1 viable fraction')
check('threshold = 4.375 * k', ld.viability_threshold(2.0), 8.75)
check('VIABILITY_BASE = 3.5/0.8', ld.VIABILITY_BASE, 4.375)
# Half the cells above the threshold, by construction.
f = np.where(np.arange(N * N).reshape(N, N) % 2 == 0, 100.0, 0.0)
frac, n = ld.viable_fraction(f, k=1.0)
check('half the cells viable', frac, (N * N + 1) // 2 / (N * N), 1e-12)
check('n_cells counted', float(n), float(N * N))
# NaNs must be excluded, not counted as non-viable.
f2 = f.copy()
f2[0, :] = np.nan
frac2, n2 = ld.viable_fraction(f2, k=1.0)
check('NaN cells excluded from n', float(n2), float(N * N - N))
# Monotonicity in k: raising the bar can never raise the viable fraction.
sens = ld.viable_fraction_sensitivity(A + 12, A + 12)
evo = [s['evo'] for s in sens]
check_true('viable fraction is non-increasing in k', all(np.diff(evo) <= 1e-12))

# ── A2 gradient-to-noise ──────────────────────────────────────────────────────
print('\nA2 gradient-to-noise ratio')
# On a plane f = 2*i, an interior 3x3 window spans exactly 2 steps = 4.
ramp = np.tile(np.arange(N, dtype=float)[:, None] * 2.0, (1, N))
lr = ld.local_range(ramp)
check('interior local range on a plane', lr[5, 5], 4.0)
check('edge local range uses 2 rows only', lr[0, 5], 2.0)
gnr = ld.gradient_to_noise(ramp, np.full((N, N), 2.0))
check('GNR = range / sigma', gnr[5, 5], 2.0)
# sigma = 0 must not produce inf in the medians.
g0 = ld.gradient_to_noise(ramp, np.zeros((N, N)))
check_true('sigma=0 yields no finite GNR', not np.isfinite(g0).any())
summ = ld.gnr_summary(ramp, np.full((N, N), 2.0))
check_true('median GNR finite on a plane', np.isfinite(summ['gnr_median_all']))

# ── A3 fitness-distance correlation ───────────────────────────────────────────
print('\nA3 fitness-distance correlation')
# A perfect cone: f = -distance from the origin, so f and d are exactly
# anti-correlated and FDC must be -1 (the easiest possible landscape).
cone = -np.sqrt(A ** 2 + B ** 2)
r, n, ref = ld.fdc(A, B, cone, reference='best')
check('FDC on a perfect cone = -1', r, -1.0, 1e-12)
check_true('cone optimum found at the origin', ref == (0.0, 0.0), f'got {ref}')
# A plane tilted along alpha is perfectly informative about fitness, yet it
# CANNOT score -1: its optimum lies on the alpha=+10 edge, and distance to that
# corner mixes in beta, which carries no signal. ~-0.69 is the correct answer and
# is the reason FDC is only comparable between grids of the same shape.
r2, _, ref2 = ld.fdc(A, B, A.copy(), reference='best')
check_true('tilted plane FDC is negative but diluted by the corner optimum',
           -0.80 < r2 < -0.55, f'got {r2:.3f} (optimum at {ref2})')
check_true('a centred cone is more negative than a corner-optimum plane', r < r2,
           f'cone {r:.3f} < plane {r2:.3f}')
# Pure noise: no relationship between fitness and distance.
rng = np.random.default_rng(0)
r3, _, _ = ld.fdc(A, B, rng.normal(size=(N, N)), reference='top5')
check_true('FDC on white noise is near 0', abs(r3) < 0.25, f'got {r3:.3f}')

# ── A4 dispersion ─────────────────────────────────────────────────────────────
print('\nA4 dispersion')
# One tight cluster of high fitness -> dispersion << 1.
tight = -((A - 8) ** 2 + (B - 8) ** 2)
d_tight = ld.dispersion(A, B, tight, 0.05)
check_true('one tight basin -> dispersion << 1', d_tight < 0.35, f'got {d_tight:.3f}')
# Random fitness -> the top cells are as spread as any random subset -> ~1.
d_rand = np.mean([ld.dispersion(A, B, rng.normal(size=(N, N)), 0.10) for _ in range(20)])
check_true('random fitness -> dispersion ~ 1', abs(d_rand - 1.0) < 0.12, f'got {d_rand:.3f}')

# ── A5 meta-model R^2 ─────────────────────────────────────────────────────────
print('\nA5 meta-model R^2')
plane = 3.0 + 2.0 * A - 1.5 * B
m = ld.meta_model_r2(A, B, plane)
check('linear R2 on an exact plane', m['r2_linear'], 1.0, 1e-9)
check('quadratic R2 on an exact plane', m['r2_quadratic'], 1.0, 1e-9)
quad = 1.0 + A ** 2 + 0.5 * B ** 2 - 2 * A * B
m2 = ld.meta_model_r2(A, B, quad)
check('quadratic R2 on an exact quadratic', m2['r2_quadratic'], 1.0, 1e-9)
check_true('linear R2 on a quadratic is poor', m2['r2_linear'] < 0.2, f"got {m2['r2_linear']:.3f}")
check_true('adjusted R2 <= R2', m2['r2_quadratic_adj'] <= m2['r2_quadratic'] + 1e-12)

# ── A6 relative lift ──────────────────────────────────────────────────────────
print('\nA6 relative lift')
fe = np.full((N, N), 10.0)
fl = np.full((N, N), 11.0)
sig = np.full((N, N), 1.0)
lift = ld.relative_lift(fe, fl, sig, sig, n_repeats=100)
check('rel_lift mean = +10%', lift['rel_lift_mean'], 0.1, 1e-12)
check('rel_lift sd = 0 for a uniform lift', lift['rel_lift_sd'], 0.0, 1e-12)
check('no harmed cells when learning helps', lift['frac_delta_negative'], 0.0)
check('all cells helped at 2 SEM', lift['frac_delta_positive'], 1.0)
# A uniform HARM, well outside the noise, must be caught everywhere.
harm = ld.relative_lift(fe, np.full((N, N), 9.0), sig, sig, n_repeats=100)
check('uniform harm detected in every cell', harm['frac_delta_negative'], 1.0)
# Paired SEM is used when supplied: a tiny paired SEM resolves a tiny harm that
# the pooled SEM would miss.
tiny = ld.relative_lift(fe, fe - 0.05, sig, sig, n_repeats=100,
                        paired_sem=np.full((N, N), 0.001))
check('pooled SEM misses the small harm', tiny['frac_delta_negative'], 0.0)
check('paired SEM catches the small harm', tiny['frac_delta_negative_paired'], 1.0)

# ── A7 noise-aware peaks ──────────────────────────────────────────────────────
print('\nA7 noise-aware peak count')
flat = np.zeros((N, N))
flat[10, 10] = 5.0
n_raw, _ = ld.count_peaks_noise_aware(flat, None)
check('one planted peak found', float(n_raw), 1.0)
n_keep, _ = ld.count_peaks_noise_aware(flat, np.full((N, N), 1.0))
check('peak survives when 5 > 2*sigma(=2)', float(n_keep), 1.0)
n_drop, _ = ld.count_peaks_noise_aware(flat, np.full((N, N), 10.0))
check('peak dropped when 5 < 2*sigma(=20)', float(n_drop), 0.0)
rep = ld.peak_margin_report(flat, np.full((N, N), 10.0))
check('margin ratio = 5 / (2*10)', rep['best_ratio'], 0.25, 1e-12)
check_true('planted interior peak is not on the boundary', rep['n_on_boundary'] == 0)
# A monotone ramp has exactly one maximum, and it is on the boundary.
n_ramp, _ = ld.count_peaks_noise_aware(np.tile(np.arange(N, dtype=float)[:, None], (1, N)), None)
check_true('a ramp has few maxima', n_ramp <= N, f'got {n_ramp}')

# ── A8 lambda reanalysis ──────────────────────────────────────────────────────
print('\nA8 lambda at extended lags')
# A pure AR(1) with a known correlation length: rho(k) = phi^k, so
# lambda = -1/ln(phi). Recovering it from the estimator is the real test.
lam_true = 20.0
phi = np.exp(-1.0 / lam_true)
rng2 = np.random.default_rng(42)
W, T = 8, 4000
walks = np.empty((W, T))
for w in range(W):
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng2.normal(0, 1)
    walks[w] = x
out = ld.lambda_reanalysis(np.exp(walks * 0.02 + 2.0), kmax=100,
                           kfits=(10, 30, 60), transform='log')
for kf in (10, 30, 60):
    v = out[f'lambda_fit_k{kf}']
    check_true(f'AR(1) lambda recovered at kfit={kf}', abs(v - lam_true) < 3.0,
               f'got {v:.2f} want {lam_true}')
check_true('1/e crossing recovers lambda', abs(out['lambda_1e_crossing'] - lam_true) < 3.0,
           f"got {out['lambda_1e_crossing']:.2f}")
check_true('rho(1) ~ phi', abs(out['rho1'] - phi) < 0.02, f"got {out['rho1']:.4f} want {phi:.4f}")
check_true('a true exponential has small kfit spread',
           out['lambda_kfit_spread_frac'] < 0.25, f"got {out['lambda_kfit_spread_frac']:.3f}")
# crossing_lag on an exact geometric decay is exact by construction.
ks = np.arange(1, 101)
check_true('crossing_lag on exact exp(-k/25)',
           abs(ld.crossing_lag(np.exp(-ks / 25.0)) - 25.0) < 0.5,
           f'got {ld.crossing_lag(np.exp(-ks / 25.0)):.3f}')
check_true('crossing_lag returns nan when rho never crosses',
           np.isnan(ld.crossing_lag(np.full(100, 0.9))))

# ── A9 comparison ─────────────────────────────────────────────────────────────
print('\nA9 comparison')
c = ld.compare_metric(1.0, 0.5, 0.3, 0.4)
check('diff', c['diff'], 0.5)
check('se_diff = sqrt(.09+.16)', c['se_diff'], 0.5, 1e-12)
check('z = diff / se', c['z'], 1.0, 1e-12)
c2 = ld.compare_metric(1.0, 0.5, None, None)
check_true('no SE -> nan Z, diff still reported',
           np.isnan(c2['z']) and c2['diff'] == 0.5)

# ── A10 neutrality ────────────────────────────────────────────────────────────
print('\nA10 neutrality')
# A perfectly flat grid: every adjacent pair is neutral at any eps > 0, and the
# whole grid is one plateau.
flat = np.zeros((10, 10))
check('flat grid: neutral fraction = 1', ld.neutral_fraction(flat, 0.1)[0], 1.0)
pl = ld.neutral_plateaus(flat, 0.1)
check('flat grid: one plateau', pl['n_plateaus'], 1)
check('flat grid: plateau covers everything', pl['largest_frac'], 1.0)

# A plane f = i has every 4-neighbour step equal to exactly 0 (along beta) or 1
# (along alpha). On a 10x10 grid that is 90 steps of each, so eps just under 1
# leaves exactly the 90 beta steps neutral -> 0.5, and eps just over 1 catches
# all 180 -> 1.0. This pins the boundary behaviour of the < comparison.
ramp = np.repeat(np.arange(10.0)[:, None], 10, axis=1)
check('plane: neutral fraction at eps=0.99', ld.neutral_fraction(ramp, 0.99)[0], 0.5)
check('plane: neutral fraction at eps=1.01', ld.neutral_fraction(ramp, 1.01)[0], 1.0)
check('plane: pair count = 2*N*(N-1)', ld.neutral_fraction(ramp, 1.0)[1], 180)
# At eps < 1 each alpha row is its own plateau: 10 stripes of 10 cells.
pl = ld.neutral_plateaus(ramp, 0.99)
check('plane: 10 stripe plateaus', pl['n_plateaus'], 10)
check('plane: each stripe is 10 cells', pl['mean_size'], 10.0)

# Two flat terraces separated by one big jump: the neutral graph must NOT bridge
# them, i.e. plateaus follow the terraces, not the eps-blind bounding box.
terr = np.zeros((10, 10))
terr[5:, :] = 100.0
pl = ld.neutral_plateaus(terr, 1.0)
check('two terraces -> two plateaus', pl['n_plateaus'], 2)
check('two terraces -> each is half the grid', pl['largest_frac'], 0.5)

# Noise-based neutrality: identical fitness with real error bars is unresolved
# everywhere; a huge separation with tiny error bars is resolved nowhere-neutral.
sem = np.full((8, 8), 1.0)
check('noise neutrality: flat grid unresolved', ld.neutral_fraction_noise(np.zeros((8, 8)), sem)[0], 1.0)
steep = np.repeat(np.arange(8.0)[:, None], 8, axis=1) * 100.0
check_true('noise neutrality: steep grid resolves the alpha steps',
           abs(ld.neutral_fraction_noise(steep, sem)[0] - 0.5) < 1e-9,
           f'got {ld.neutral_fraction_noise(steep, sem)[0]:.3f}')

# The summary must scale eps by the grid's own range, so a landscape and the
# same landscape times 1000 report identical neutrality.
s1 = ld.neutrality_summary(ramp)
s2 = ld.neutrality_summary(ramp * 1000.0)
check_true('neutrality summary is scale-invariant',
           s1['neutral_frac'] == s2['neutral_frac'],
           f"{s1['neutral_frac']} vs {s2['neutral_frac']}")
check('summary reports the range', s1['range'], 9.0)

print('\n' + ('ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'))
sys.exit(1 if FAILS else 0)
