"""
landscape/ll_difficulty.py
======================================================================
Stage 9 metric kernel — how HARD is this landscape, not how HIGH is it.

Stages 3-8 measure where fitness IS (level, shape, ruggedness). These metrics
measure what a searcher would EXPERIENCE trying to climb it: how much of the
space is even survivable, whether the local gradient is readable through the
evaluation noise, whether "closer to the optimum" means "fitter", how scattered
the good regions are, and how much of the surface a quadratic already explains.

Every function here is pure numpy — no IO, no plotting, no argparse — so each
metric can be unit-tested against a hand-built grid. Grid arrays are shaped
(N, N) with i indexing alpha and j indexing beta, matching ll_grid.reconstruct_grid.

THE VIABILITY THRESHOLD IS NOT ARBITRARY.
    AdvancedOrganism.js:291 fires a reproduction attempt when
    energyGainedSinceReproduction >= 3.5, and the counter is reset "whether or
    not placement succeeded" (line 477-478) — so every 3.5 food banked buys one
    ATTEMPT, not one child. An attempt lands only if
    `isClear(...) && canAddOrganism() && Math.random() < 0.8`
    (REPRODUCTION_SUCCESS_RATE, ExperimentParams.js:87). Counting just the 0.8
    roll, the food needed to land one child is 3.5 / 0.8 = 4.375. That is the
    replacement point: below it a lineage shrinks, above it grows.
    VIABILITY_BASE is that number.

    It is a LOWER bound on the real cost, because the other two gates can also
    fail: in a crowded world `isClear` and `canAddOrganism` reject placements
    too, so the true food-per-child is >= 4.375 and the viable fraction reported
    here is correspondingly OPTIMISTIC. This does not affect BETWEEN-environment
    comparisons, which is what the metric is for — it shifts both surfaces the
    same way, and the k sweep covers the shift.

    Note this is a threshold on a COUNTER, not an energy debit: reproduction
    costs no energy in this simulator (see the README's correction of the
    original plan). 4.375 is "food banked per child landed", not "energy spent".

    The catch is that 4.375 is on the REPLAY scale (a real reproducing generation
    over 5 maps), while f(theta) is on the MONOMORPHIC probe scale (fixed cohort,
    1 map, reproduction off). The two are not the same unit, so the threshold on
    the probe scale is 4.375 * k, where k = f_monomorphic / f_replay is measured
    by the Stage-9 B1 calibration. Until that is run, k = K_DEFAULT is a stated
    assumption, which is exactly why viable_fraction_sensitivity() sweeps it.
"""

import numpy as np

# Reproduction energy threshold / success rate — see module docstring.
REPRODUCTION_ENERGY_THRESHOLD = 3.5
REPRODUCTION_SUCCESS_RATE = 0.8
VIABILITY_BASE = REPRODUCTION_ENERGY_THRESHOLD / REPRODUCTION_SUCCESS_RATE  # 4.375
K_DEFAULT = 2.5


# ── small shared helpers ──────────────────────────────────────────────────────
def _finite_pair(x, y):
    """Flatten two arrays to their common finite entries."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y):
    """Pearson r over the common finite entries. nan if degenerate.

    Local so the module has no scipy dependency (the pipeline never took one);
    identical to scipy.stats.pearsonr(...).statistic.
    """
    x, y = _finite_pair(x, y)
    if x.size < 3:
        return float('nan')
    xs = x - x.mean()
    ys = y - y.mean()
    den = np.sqrt((xs * xs).sum() * (ys * ys).sum())
    return float((xs * ys).sum() / den) if den > 0 else float('nan')


def corr_se(r, n):
    """SE of a Pearson r via the Fisher-z delta method: SE(r) = (1-r^2)/sqrt(n-3).

    Used only to give the Stage-9 comparison table an error bar on correlation
    differences. Cells are spatially autocorrelated, so n is an EFFECTIVE sample
    size at best and this SE is optimistic — it is a floor, not a guarantee.
    """
    if not np.isfinite(r) or n < 4:
        return float('nan')
    return float((1.0 - r ** 2) / np.sqrt(n - 3))


def _mean_pairwise_dist(pts):
    """Mean Euclidean distance over all unordered pairs of an (n,2) point set."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    if n < 2:
        return float('nan')
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def grid_coords(alphas, betas):
    """(N,N) alpha and beta coordinate grids matching grid[i, j] (i=alpha, j=beta)."""
    A, B = np.meshgrid(np.asarray(alphas, dtype=float),
                       np.asarray(betas, dtype=float), indexing='ij')
    return A, B


# ── A1. viable fraction ───────────────────────────────────────────────────────
def viability_threshold(k):
    """Viability cutoff on the monomorphic-probe scale. See module docstring."""
    return float(VIABILITY_BASE * k)


def viable_fraction(f, k=K_DEFAULT):
    """Share of grid cells whose fitness clears the replacement threshold.

    The most direct difficulty statement available: not "how high does it get"
    but "how much of this space is a population that does not die out".
    Returns (fraction, n_cells) so the caller can attach a binomial SE.
    """
    f = np.asarray(f, dtype=float).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        return float('nan'), 0
    return float(np.mean(f > viability_threshold(k))), int(f.size)


def viable_fraction_se(frac, n):
    """Binomial SE of a viable fraction. Optimistic — cells are autocorrelated."""
    if not np.isfinite(frac) or n < 1:
        return float('nan')
    return float(np.sqrt(max(frac * (1.0 - frac), 0.0) / n))


def viable_fraction_sensitivity(f_evo, f_learn, k_values=None):
    """viable_frac vs k for both RL passes.

    k is MEASURED by the B1 calibration, but it is measured on 3 genomes with
    spread, and the threshold moves linearly with it. Reporting the sweep means
    a reader can see whether the environment ranking survives the uncertainty in
    k, instead of trusting one point estimate.
    """
    if k_values is None:
        k_values = np.arange(1.5, 3.6, 0.1)
    out = []
    for k in k_values:
        ev, _ = viable_fraction(f_evo, k)
        ln, _ = viable_fraction(f_learn, k)
        out.append({'k': float(k), 'evo': ev, 'learn': ln})
    return out


# ── A2. gradient-to-noise ratio ───────────────────────────────────────────────
def local_range(f):
    """Max-min of f over each cell's 3x3 window (the cell included).

    Edge and corner cells use only the neighbours that exist, so the array has
    no NaN border — the alternative (dropping the border) would throw away the
    outer ring, which on a 25x25 grid is 15% of the cells.
    """
    f = np.asarray(f, dtype=float)
    N, M = f.shape
    out = np.full((N, M), np.nan)
    for i in range(N):
        for j in range(M):
            w = f[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            w = w[np.isfinite(w)]
            if w.size:
                out[i, j] = w.max() - w.min()
    return out


def gradient_to_noise(f, sigma):
    """GNR = local fitness range / per-cell evaluation noise.

    The difficulty question a hill-climber actually faces: is the signal between
    me and my neighbours bigger than the noise in measuring it? GNR < 1 means the
    local gradient is invisible to a searcher — it would be following noise. The
    contour at GNR = 1 is therefore the boundary of the searchable region.

    Cells with sigma = 0 (a single repeat, or an exactly-degenerate cell) give
    inf where the range is positive and nan where it is not; both are excluded
    from the medians by nanmedian/isfinite.
    """
    f = np.asarray(f, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lr = local_range(f)
    with np.errstate(divide='ignore', invalid='ignore'):
        gnr = lr / sigma
    gnr[~np.isfinite(gnr)] = np.nan
    return gnr


def gnr_summary(f, sigma):
    """Median GNR overall and split at the median of f.

    The split matters: noise scales with fitness in this simulator (Stage 7
    measured r(sigma, f) = +0.96..+0.99), so a single median would hide that the
    high-fitness region can be harder to read than the low one despite having a
    steeper gradient.
    """
    gnr = gradient_to_noise(f, sigma)
    fr = np.asarray(f, dtype=float)
    valid = np.isfinite(gnr) & np.isfinite(fr)
    if not valid.any():
        return {'gnr_median_all': float('nan'), 'gnr_median_lowf': float('nan'),
                'gnr_median_highf': float('nan'), 'gnr_frac_below_1': float('nan'),
                'gnr': gnr}
    fmed = float(np.nanmedian(fr[valid]))
    low = valid & (fr <= fmed)
    high = valid & (fr > fmed)
    return {
        'gnr_median_all': float(np.median(gnr[valid])),
        'gnr_median_lowf': float(np.median(gnr[low])) if low.any() else float('nan'),
        'gnr_median_highf': float(np.median(gnr[high])) if high.any() else float('nan'),
        'gnr_frac_below_1': float(np.mean(gnr[valid] < 1.0)),
        'gnr': gnr,
    }


# ── A3. fitness-distance correlation ──────────────────────────────────────────
def fdc(alpha, beta, f, reference='best', top_frac=0.05):
    """Correlation between fitness and distance to the optimum.

    Classic search-difficulty diagnostic (Jones & Forrest 1995). For a
    MAXIMISATION problem, strongly negative FDC = easy (fitness rises steadily as
    you approach the peak, so the gradient points home); FDC near 0 = the
    landscape gives a searcher no directional information; positive = deceptive,
    the gradient points AWAY from the optimum.

    FDC HAS NO ABSOLUTE SCALE — compare it BETWEEN environments on the same grid,
    never against a fixed "-1 = easy" yardstick. Only a landscape whose optimum
    is at the grid centre can reach -1: on this square domain a perfectly LINEAR
    surface f = alpha scores about -0.69, because its optimum sits at a corner
    and the distance to a corner mixes in the beta direction, which carries no
    fitness signal at all. A perfect cone centred in the grid does score -1. So
    the difference between -0.03 and -0.76 across two environments is meaningful;
    the distance of either from -1 is not.

    reference='best'    -> distance to the single argmax cell.
    reference='top5'    -> distance to the (alpha,beta) centroid of the top
                           `top_frac` of cells. The single best cell on a noisy
                           grid is partly a noise draw; the centroid of the top
                           5% is a far more stable target, so agreement between
                           the two is the robustness check.
    Returns (fdc, n_cells, reference_point).
    """
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    fv = np.asarray(f, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(fv)
    a, b, fv = a[m], b[m], fv[m]
    if fv.size < 4:
        return float('nan'), int(fv.size), (float('nan'), float('nan'))

    if reference == 'best':
        idx = int(np.argmax(fv))
        ra, rb = float(a[idx]), float(b[idx])
    elif reference == 'top5':
        n_top = max(2, int(round(top_frac * fv.size)))
        idx = np.argsort(fv)[-n_top:]
        ra, rb = float(a[idx].mean()), float(b[idx].mean())
    else:
        raise ValueError(f'unknown reference {reference!r}')

    d = np.sqrt((a - ra) ** 2 + (b - rb) ** 2)
    return pearson(fv, d), int(fv.size), (ra, rb)


# ── A4. dispersion ────────────────────────────────────────────────────────────
def dispersion(alpha, beta, f, p):
    """Mean pairwise spread of the top-p cells, relative to the whole grid.

    Dispersion (Lunacek & Whitley 2006). Ratio near 0 = the good cells are one
    tight cluster, so a searcher that finds any of them is in the right basin.
    Ratio near 1 = the good cells are scattered as widely as random points, i.e.
    a global-structure-free landscape where local search gives no guidance about
    where the other good regions are.

    Maximisation, so "top" = highest f (the original is a minimisation metric).
    """
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    fv = np.asarray(f, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(fv)
    a, b, fv = a[m], b[m], fv[m]
    if fv.size < 4:
        return float('nan')
    n_top = max(2, int(round(p * fv.size)))
    idx = np.argsort(fv)[-n_top:]
    pts_top = np.column_stack([a[idx], b[idx]])
    pts_all = np.column_stack([a, b])
    den = _mean_pairwise_dist(pts_all)
    if not np.isfinite(den) or den <= 0:
        return float('nan')
    return float(_mean_pairwise_dist(pts_top) / den)


def dispersion_profile(alpha, beta, f, ps=(0.02, 0.05, 0.10)):
    """Dispersion at several top-fractions. Rising with p = nested single basin;
    flat and high = genuinely scattered optima."""
    return {f'{p:g}': dispersion(alpha, beta, f, p) for p in ps}


# ── A5. meta-model R^2 ────────────────────────────────────────────────────────
def _ols_r2(X, y):
    """Return (R2, adjusted R2) for an OLS fit of y on design matrix X."""
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # errstate: numpy 2.0 on Apple Accelerate raises spurious divide/overflow/
    # invalid flags from matmul on well-conditioned inputs. Verified on this
    # grid: cond(X_quad) = 531, full rank, and X@beta agrees with the equivalent
    # einsum to 3.6e-15 with zero non-finite entries. The flags are noise, not a
    # numerical problem — so silence them here rather than globally.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float('nan'), float('nan')
    r2 = 1.0 - ss_res / ss_tot
    dof = n - p
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float('nan')
    return float(r2), float(r2_adj)


def meta_model_r2(alpha, beta, f):
    """How much of the surface a linear / quadratic surrogate already explains.

    A difficulty statement about MODELLABILITY: if a quadratic captures ~all the
    variance the landscape is effectively unimodal at this scale and any
    surrogate-assisted or gradient-following method will do well. The gap
    (1 - r2_quadratic) is the share that is genuinely non-quadratic structure —
    multiple basins, ridges, texture — which is the part that makes search hard.

    Adjusted R^2 is reported alongside because the quadratic spends 6 parameters
    against the linear model's 3; on a 625-cell grid the penalty is tiny, but
    quoting only raw R^2 for nested models invites the obvious objection.
    """
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    y = np.asarray(f, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[m], b[m], y[m]
    if y.size < 8:
        nan = float('nan')
        return {'r2_linear': nan, 'r2_quadratic': nan,
                'r2_linear_adj': nan, 'r2_quadratic_adj': nan}
    one = np.ones_like(a)
    X_lin = np.column_stack([one, a, b])
    X_quad = np.column_stack([one, a, b, a ** 2, b ** 2, a * b])
    r2l, r2la = _ols_r2(X_lin, y)
    r2q, r2qa = _ols_r2(X_quad, y)
    return {'r2_linear': r2l, 'r2_quadratic': r2q,
            'r2_linear_adj': r2la, 'r2_quadratic_adj': r2qa,
            'r2_nonquadratic_residual': float(1.0 - r2q) if np.isfinite(r2q) else float('nan')}


# ── A6. relative lift and the lifetime relation ───────────────────────────────
def relative_lift(f_evo, f_learn, sigma_evo, sigma_learn, n_repeats,
                  paired_sem=None):
    """What learning buys per cell, and where it costs.

    rel_lift = (f_learn - f_evo) / f_evo — the proportional gain from switching
    RL on at the same genome. Its correlation with f_evo is the difficulty-
    relevant part: a POSITIVE corr means learning helps most where the genome is
    already good (it sharpens winners and cannot rescue the bad regions), a
    negative one means learning flattens the landscape by lifting the floor.

    frac_delta_negative counts cells where learning measurably HURTS, at a 2-SEM
    margin. Two SEMs are reported:
      - `sem_pooled` = sqrt(sem_evo^2 + sem_learn^2), the spec's formula, correct
        if the two passes were independent.
      - `sem_paired` (if supplied) = SEM of the per-repeat difference. The grid
        evaluates both RL passes on IDENTICAL map indices precisely so terrain
        cancels in the difference, and Stage 7 measured that map identity is ~84%
        of the repeat variance. The pooled SEM throws that pairing away and is
        therefore roughly 2.5x too wide here — it UNDERCOUNTS harmed cells.
        Both are returned; the paired one is the honest number for this design.
    """
    fe = np.asarray(f_evo, dtype=float)
    fl = np.asarray(f_learn, dtype=float)
    delta = fl - fe
    with np.errstate(divide='ignore', invalid='ignore'):
        rl = np.where(fe > 0, delta / fe, np.nan)

    sem_e = np.asarray(sigma_evo, dtype=float) / np.sqrt(max(n_repeats, 1))
    sem_l = np.asarray(sigma_learn, dtype=float) / np.sqrt(max(n_repeats, 1))
    sem_pooled = np.sqrt(sem_e ** 2 + sem_l ** 2)

    ok = np.isfinite(delta) & np.isfinite(sem_pooled)
    frac_neg_pooled = float(np.mean(delta[ok] < -2 * sem_pooled[ok])) if ok.any() else float('nan')
    frac_pos_pooled = float(np.mean(delta[ok] > 2 * sem_pooled[ok])) if ok.any() else float('nan')

    frac_neg_paired = float('nan')
    if paired_sem is not None:
        ps = np.asarray(paired_sem, dtype=float)
        okp = np.isfinite(delta) & np.isfinite(ps)
        if okp.any():
            frac_neg_paired = float(np.mean(delta[okp] < -2 * ps[okp]))

    rlf = rl[np.isfinite(rl)]
    return {
        'rel_lift': rl,
        'delta': delta,
        'sem_pooled': sem_pooled,
        'rel_lift_mean': float(rlf.mean()) if rlf.size else float('nan'),
        'rel_lift_sd': float(rlf.std(ddof=1)) if rlf.size > 1 else float('nan'),
        'rel_lift_sem': float(rlf.std(ddof=1) / np.sqrt(rlf.size)) if rlf.size > 1 else float('nan'),
        'frac_delta_negative': frac_neg_pooled,
        'frac_delta_positive': frac_pos_pooled,
        'frac_delta_negative_paired': frac_neg_paired,
        'n_cells': int(rlf.size),
    }


# ── A7. noise-aware peak count ────────────────────────────────────────────────
def count_peaks_noise_aware(f, sem=None, margin=2.0):
    """Local maxima that survive the evaluation noise.

    ll_grid.count_peaks() counts strict 8-neighbour maxima, which on a noisy grid
    counts noise: any cell whose neighbours happen to draw low becomes a "peak",
    so the raw count is an upper bound that inflates with R decreasing. Requiring
    f[i,j] - f[neighbour] > margin * sem[i,j] for EVERY neighbour keeps only the
    maxima that are resolved by the data. The filtered count is the defensible
    multimodality claim; the raw count is reported beside it to show the gap.

    sem=None reproduces the raw (unfiltered) count.
    """
    g = np.asarray(f, dtype=float)
    N, M = g.shape
    s = None if sem is None else np.asarray(sem, dtype=float)
    peaks = []
    for i in range(N):
        for j in range(M):
            c = g[i, j]
            if not np.isfinite(c):
                continue
            thr = 0.0
            if s is not None:
                if not np.isfinite(s[i, j]):
                    continue
                thr = margin * s[i, j]
            is_peak = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M and np.isfinite(g[ni, nj]):
                        if not (c - g[ni, nj] > thr):
                            is_peak = False
                            break
                if not is_peak:
                    break
            if is_peak:
                peaks.append((i, j))
    return len(peaks), peaks


def peak_margin_report(f, sem, margin=2.0):
    """For each RAW local maximum: its height above its best neighbour, against
    the noise it would have to clear to count.

    A filtered count of 0 is ambiguous on its own — it could mean "every peak
    just missed" or "not one is remotely resolved". The ratio
    (min neighbour margin) / (margin * sem) settles it: >= 1 means the peak
    survives, and how far below 1 the largest ratio sits says how much more R
    (or how much finer a grid) it would take to resolve ANY peak in this plane.
    """
    g = np.asarray(f, dtype=float)
    s = np.asarray(sem, dtype=float)
    N, M = g.shape
    _, raw = count_peaks_noise_aware(g, None)
    rows = []
    for (i, j) in raw:
        margins = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and np.isfinite(g[ni, nj]):
                    margins.append(g[i, j] - g[ni, nj])
        if not margins:
            continue
        need = margin * float(s[i, j])
        rows.append({
            'i': int(i), 'j': int(j), 'f': float(g[i, j]),
            'min_margin': float(min(margins)),
            'need': need,
            'ratio': float(min(margins) / need) if need > 0 else float('inf'),
            'on_boundary': bool(i in (0, N - 1) or j in (0, M - 1)),
        })
    ratios = [r['ratio'] for r in rows if np.isfinite(r['ratio'])]
    return {
        'peaks': rows,
        'best_ratio': float(max(ratios)) if ratios else float('nan'),
        'n_on_boundary': int(sum(r['on_boundary'] for r in rows)),
    }


# ── A8. lambda reanalysis at extended lags ────────────────────────────────────
def autocorr(x, kmax):
    """Weinberger autocorrelation rho(k), population form. Mirrors stage6."""
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    denom = float(np.sum(d * d))
    out = np.empty(kmax)
    for k in range(1, kmax + 1):
        out[k - 1] = (np.sum(d[:n - k] * d[k:]) / denom) if denom > 0 else np.nan
    return out


def linear_detrend(y):
    """Subtract a least-squares linear trend; returns (residual, slope)."""
    x = np.arange(len(y), dtype=float)
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x), float(b[0])


def fit_lambda(rho, kfit):
    """LS fit of ln rho(k) = -k/lambda over k = 1..kfit where rho(k) > 0."""
    ks = np.arange(1, len(rho) + 1)
    m = (ks <= kfit) & (rho > 0) & np.isfinite(rho)
    if m.sum() < 2:
        return float('nan')
    slope = float(np.polyfit(ks[m], np.log(rho[m]), 1)[0])
    return (-1.0 / slope) if slope < 0 else float('inf')


def crossing_lag(rho, level=1.0 / np.e):
    """First lag where rho(k) falls below `level`, linearly interpolated.

    A FIT-FREE correlation length. lambda from an LS fit assumes rho decays as a
    clean exponential; if it does not (a shoulder, a plateau, curvature the
    detrend missed), the fitted lambda depends on where you stop fitting — which
    is exactly why kfit is swept. The 1/e crossing assumes nothing about the
    shape, so agreement between it and the fits is the evidence that the
    exponential model is appropriate at all.
    """
    rho = np.asarray(rho, dtype=float)
    ks = np.arange(1, rho.size + 1)
    below = np.where(np.isfinite(rho) & (rho < level))[0]
    if below.size == 0:
        return float('nan')
    i = int(below[0])
    if i == 0:
        return float(ks[0])
    r1, r0 = rho[i], rho[i - 1]
    if not np.isfinite(r0) or r0 == r1:
        return float(ks[i])
    # linear interpolation in k between the bracketing lags
    return float(ks[i - 1] + (r0 - level) / (r0 - r1))


def lambda_reanalysis(series, kmax=100, kfits=(10, 30, 60), transform='log'):
    """rho(k) to long lags, averaged across walks, with lambda at several kfit.

    `series` is [W, T]: one row per independent walk, per-step R-averaged fitness.

    Why extended lags: Stage 6 fits k = 1..10 by default. If the true decay is
    exponential, the fitted lambda must not depend on the fitting window; if it
    does, that dependence is itself the finding (the decay is not a single
    exponential, so one lambda does not summarise the landscape). Reporting
    kfit = 10, 30, 60 side by side makes that testable instead of assumed.

    transform='log' analyses ln f: Stage 7 measured r(sigma, f) = +0.96..+0.99,
    i.e. the noise is MULTIPLICATIVE, so ln stabilises the variance and turns an
    exponential relaxation toward a plateau into the straight line the linear
    detrender can actually remove.

    The curves are averaged across walks BEFORE fitting (spec A8); per-walk fits
    are also returned so the cross-walk SE is available.
    """
    s = np.asarray(series, dtype=float)
    if s.ndim != 2:
        raise ValueError('series must be [W, T]')
    if transform == 'log':
        if np.any(s <= 0):
            raise ValueError('lambda_reanalysis: non-positive fitness with transform=log')
        s = np.log(s)
    W, T = s.shape
    kmax = int(min(kmax, T - 2))
    if kmax < 2:
        raise ValueError(f'series too short for kmax>=2 (T={T})')

    proc = np.empty_like(s)
    slopes = []
    for i in range(W):
        proc[i], sl = linear_detrend(s[i])
        slopes.append(sl)

    rho_w = np.vstack([autocorr(proc[i], kmax) for i in range(W)])
    mean_rho = np.nanmean(rho_w, axis=0)
    se_rho = (np.nanstd(rho_w, axis=0, ddof=1) / np.sqrt(W)) if W > 1 else np.full(kmax, np.nan)

    out = {
        'kmax': kmax, 'walks': int(W), 'steps': int(T - 1),
        'transform': transform,
        'rho': mean_rho.tolist(),
        'rho_se': se_rho.tolist(),
        'rho1': float(mean_rho[0]),
        'rho1_se': float(se_rho[0]) if np.isfinite(se_rho[0]) else float('nan'),
        'lambda_1e_crossing': crossing_lag(mean_rho),
        'detrend_slopes': [float(x) for x in slopes],
    }
    for kf in kfits:
        if kf > kmax:
            out[f'lambda_fit_k{kf}'] = float('nan')
            out[f'lambda_fit_k{kf}_se'] = float('nan')
            continue
        out[f'lambda_fit_k{kf}'] = fit_lambda(mean_rho, kf)
        per_walk = np.array([fit_lambda(rho_w[i], kf) for i in range(W)])
        fin = per_walk[np.isfinite(per_walk)]
        out[f'lambda_fit_k{kf}_se'] = (
            float(fin.std(ddof=1) / np.sqrt(fin.size)) if fin.size > 1 else float('nan'))

    # Window-dependence diagnostic. If rho(k) really is exp(-k/lambda), the fitted
    # lambda cannot depend on where the fit stops, so the spread across kfit is a
    # DIRECT test of the model that produces lambda. Large spread means one lambda
    # does not summarise the decay and the fit-free 1/e crossing is the number to
    # quote. (Note rho(k) < 0 at long lag is expected: the autocorrelations of a
    # finite detrended series sum to about -1/2, so it is an artefact of the
    # estimator, not evidence of genuine anticorrelation.)
    lams = np.array([out[f'lambda_fit_k{kf}'] for kf in kfits], dtype=float)
    lams = lams[np.isfinite(lams)]
    if lams.size > 1:
        out['lambda_kfit_spread'] = float(lams.max() - lams.min())
        out['lambda_kfit_spread_frac'] = float((lams.max() - lams.min()) / lams.mean())
    else:
        out['lambda_kfit_spread'] = float('nan')
        out['lambda_kfit_spread_frac'] = float('nan')
    return out


# ── B1. locating the replacement crossing ─────────────────────────────────────
def replacement_crossing(f, cpf, lo=0.3, hi=3.0, n_boot=2000, seed=0):
    """Fitness at which a cohort exactly replaces itself (1 child per founder).

    Fits ln(children/founder) = a + b*f by least squares and solves for cpf = 1.
    The log scale is the right one: reproductive output is a multiplicative
    process bounded below by 0, and on these data ln(cpf) is close to linear in f
    while cpf itself is not (it is flat then steeply rising).

    Restricted to points with cpf in [lo, hi] — the neighbourhood of the
    crossing. Far above replacement the curve steepens and would drag the fitted
    line, moving an estimate that is only supposed to describe where the curve
    passes 1. This is a LOCAL estimator on purpose.

    Preferred over reading off the two cells that happen to straddle 1.0: the
    per-cell cpf is noisy (adjacent cells can invert), so a two-point reading
    inherits that noise entirely, while the fit averages over every nearby cell.

    Returns dict with f_cross, its bootstrap CI, the slope, n used, and r2.
    """
    f = np.asarray(f, dtype=float)
    c = np.asarray(cpf, dtype=float)
    m = np.isfinite(f) & np.isfinite(c) & (c >= lo) & (c <= hi) & (c > 0)
    if m.sum() < 3:
        return {'f_cross': float('nan'), 'n_used': int(m.sum()),
                'ci95': (float('nan'), float('nan')), 'slope': float('nan'),
                'r2': float('nan')}
    x, y = f[m], np.log(c[m])

    def solve(xx, yy):
        b, a = np.polyfit(xx, yy, 1)
        return (-a / b) if b > 0 else float('nan')

    b, a = np.polyfit(x, y, 1)
    f_cross = solve(x, y)
    pred = a + b * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float('nan')

    rng = np.random.default_rng(seed)
    boots = []
    n = x.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size < 2:
            continue
        v = solve(x[idx], y[idx])
        if np.isfinite(v):
            boots.append(v)
    ci = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
          if len(boots) > 20 else (float('nan'), float('nan')))
    return {'f_cross': float(f_cross), 'ci95': ci, 'slope': float(b),
            'n_used': int(n), 'r2': r2}


# ── A9. cross-environment comparison ──────────────────────────────────────────
def compare_metric(v_a, v_b, se_a=None, se_b=None):
    """One row of the environment comparison: difference, SE, Z.

    Z = (v_a - v_b) / sqrt(se_a^2 + se_b^2), reported only when both SEs exist.
    Where a metric has no defensible SE (a median, a single-fit R^2) the row
    still carries the difference, with SE and Z left as nan rather than invented.
    """
    d = float(v_a - v_b) if (np.isfinite(v_a) and np.isfinite(v_b)) else float('nan')
    if se_a is None or se_b is None or not np.isfinite(se_a) or not np.isfinite(se_b):
        return {'diff': d, 'se_diff': float('nan'), 'z': float('nan')}
    se = float(np.sqrt(se_a ** 2 + se_b ** 2))
    return {'diff': d, 'se_diff': se, 'z': float(d / se) if se > 0 else float('nan')}
