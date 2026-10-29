"""SHARED statistics of the startup fold-pathway framework.

Single source of truth: `simulation_data_image/plot.py` and
`exp_data_image/plot.py` both import from here.  Both sides observe their
system ONLY through images (the simulation is rendered first, see
simulation_data_image/render.py) and go through the same feature pipeline
(frame_features.frame_scalars) and the same statistics below.  The
ground-truth controls (reduced |Z3|, labeled endsum) run through exactly the
same code path as the image features.

Conventions (see label_free_second_transition.md):
  * fold coordinate  z3c = (xmax + xmin) - 2*cx  (cx = intensity centroid,
    crop-invariant), per-trajectory/per-video median-subtracted over ok
    frames -> `fold_coordinate`;
  * frame quality    ok = finite & no border contact & |jump| < JUMP_FRAC*Lmax
    -> `qc_mask`; applied identically on both sides;
  * ALL length constants are FRACTIONS of L_max; L_max is estimated from the
    data on each side the same way (`estimate_lmax`: p99 of ext over the
    highest-Wi / highest-voltage group);
  * first-passage truncation is DECOUPLED from the measuring band:
    t_end = first ok frame with ext > T_END_FRAC*Lmax (same for every band);
  * per-trajectory fold statistics over transit frames (ext in [lo, hi)
    before t_end): A = median |z3c| (`fold_amplitude`), and the full
    (med, q90, fold-frame fraction) triple (`fold_stats`) — the SAME
    statistics on both sides;
  * onset = first Wi bin whose bootstrap 2.5% of the median-A exceeds the
    97.5% of the low-Wi baseline (BASE_RNG), reported as the bracket
    (last non-significant, first significant].

Ground truth for the transition location: zero crossing of the conditional
Z3 drift on the reduced data, tight windows -> Wi* in GT_DRIFT.
(The learned-potential route in examples/.../z3_plateau is NOT used: its code
is known to be problematic.)
"""
import numpy as np

# ---- length fractions of L_max (historical px values at sim L_max ~278) ----
T_END_FRAC = 0.72          # first-passage cutoff        (was 200 px)
BARRIER_FRAC = 0.22        # two-lag barrier window      (was  60 px)
FOLD_THR_FRAC = 0.035      # folded-frame |z3c| threshold (was  8 px @ exp 230)
JUMP_FRAC = 0.25           # QC: max frame-to-frame ext jump
MIN_WIDTH_FRAC = 0.11      # onset2d: min band width     (was  30 px)
BANDS3_FRAC = [(0.18, 0.43), (0.22, 0.54), (0.29, 0.65)]   # near/mid/far bands

BASE_RNG = (0.3, 0.8)      # baseline Wi window: above 1st transition, below onset
GT_DRIFT = (0.84, 1.36)    # conditional-drift Wi* (tight windows), reduced data


# ----------------------------- per-frame QC -----------------------------

def estimate_lmax(ext_list, q=99):
    """L_max = q-th percentile of ext pooled over the given trajectories.

    Callers pass the highest-Wi (simulation) / highest-voltage (experiment)
    group so the pool contains fully stretched frames; identical on both sides.
    """
    v = np.concatenate([np.asarray(e, float).ravel() for e in ext_list])
    return float(np.percentile(v[np.isfinite(v)], q))


def qc_mask(ext, border=None, lmax=None):
    """Frame-quality mask: finite, no crop-border contact, no ext jump."""
    ext = np.asarray(ext, float)
    ok = np.isfinite(ext)
    if border is not None:
        ok &= (border == 0)
    if lmax is not None:
        d = np.abs(np.diff(ext, prepend=ext[:1]))
        ok &= (d < JUMP_FRAC * lmax)           # NaN diff (after a bad frame) -> drop
    return ok


def fold_coordinate(z3p, cx=None, ok=None):
    """z3c = z3p - 2*cx, median-subtracted over ok frames (both sides)."""
    z = np.asarray(z3p, float) - (2 * np.asarray(cx, float) if cx is not None else 0.0)
    m = ok if ok is not None else np.isfinite(z)
    return z - (np.median(z[m]) if m.sum() else 0.0)


# ----------------------------- per-trajectory -----------------------------

def first_passage_end(ext, t_end_ext, ok=None):
    """Index of the first ok frame with ext > t_end_ext (len(ext) if never)."""
    e = np.asarray(ext, float)
    m = (e > t_end_ext)
    if ok is not None:
        m &= ok
    idx = np.where(m)[0]
    return int(idx[0]) if len(idx) else len(e)


def truncated_pairs(ext_list, absz3c_list, t_end_ext, ok_list=None):
    """Per-trajectory (ext, |z3c|) of the FIRST stretch event, ok frames only.

    Truncate at first passage (ext > t_end_ext), then keep ok frames.
    Precompute once; `fold_amplitude`/`fold_stats` select band frames from it.
    """
    out = []
    for i, (e, a) in enumerate(zip(ext_list, absz3c_list)):
        ok = ok_list[i] if ok_list is not None else np.isfinite(np.asarray(e, float))
        t = first_passage_end(e, t_end_ext, ok)
        sel = np.zeros(len(e), bool); sel[:t] = ok[:t]
        out.append((np.asarray(e, float)[sel], np.asarray(a, float)[sel]))
    return out


def untruncated_pairs(ext_list, absz3c_list, ok_list=None):
    """(ext, |z3c|) over ALL ok frames (no first-passage truncation).

    Diagnostic only — the headline statistics use `truncated_pairs`.
    """
    out = []
    for i, (e, a) in enumerate(zip(ext_list, absz3c_list)):
        ok = ok_list[i] if ok_list is not None else np.isfinite(np.asarray(e, float))
        out.append((np.asarray(e, float)[ok], np.asarray(a, float)[ok]))
    return out


def fold_amplitude(pair, lo, hi, min_frames=3):
    """A = med|z3c| over transit frames of one truncated trajectory."""
    e, a = pair
    m = (e >= lo) & (e < hi)
    if m.sum() < min_frames:
        return np.nan
    return float(np.median(a[m]))


def fold_stats(pair, lo, hi, fold_thr, min_frames=5):
    """(med, q90, fold-frame %) over transit frames — same on both sides."""
    e, a = pair
    m = (e >= lo) & (e < hi)
    if m.sum() < min_frames:
        return np.nan, np.nan, np.nan
    v = a[m]
    return (float(np.median(v)), float(np.percentile(v, 90)),
            float(np.mean(v > fold_thr) * 100))


def fold_q90(pair, lo, hi, min_frames=5):
    """q90 |z3c| over the same transit frame set as `fold_amplitude`."""
    e, a = pair
    m = (e >= lo) & (e < hi)
    if m.sum() < min_frames:
        return np.nan
    return float(np.percentile(a[m], 90))


def make_fold_ff(fold_thr, min_frames=5):
    """Per-trajectory fold-frame fraction (%) with a FIXED px threshold.

    fold_thr is in px (FOLD_THR_FRAC * lmax of the calling side), so the
    returned statistic has the (pair, lo, hi) signature of `fold_amplitude`
    and can be passed as `stat=` to `onset_bracket`/`onset_map`.
    """
    def fold_ff(pair, lo, hi):
        e, a = pair
        m = (e >= lo) & (e < hi)
        if m.sum() < min_frames:
            return np.nan
        return float(np.mean(a[m] > fold_thr) * 100)
    return fold_ff


# ----------------------------- bootstrap -----------------------------

def boot_median_ci(v, rng, nb=400, pct=(2.5, 50, 97.5)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.full(len(pct), np.nan)
    bs = np.median(rng.choice(v, size=(nb, len(v))), axis=1)
    return np.percentile(bs, pct)


def boot_mean_ci(v, rng, nb=2000, pct=(2.5, 50, 97.5)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.full(len(pct), np.nan)
    bs = np.mean(rng.choice(v, size=(nb, len(v))), axis=1)
    return np.percentile(bs, pct)


# ----------------------------- onset -----------------------------

def onset_bracket(Wi, pairs, lo, hi, rng, base_rng=BASE_RNG, wmax=3.5,
                  nfine=16, min_group=10, nb=400, stat=None):
    """(last non-significant Wi, first significant Wi) for band [lo, hi).

    stat: per-trajectory statistic (pair, lo, hi) -> float; default
    `fold_amplitude` (med).  Same criterion for every stat: first Wi group
    whose bootstrap 2.5% exceeds the baseline 97.5%.
    """
    stat = stat or fold_amplitude
    base_tr = np.where((Wi > base_rng[0]) & (Wi < base_rng[1]))[0]
    bv = np.array([stat(pairs[t], lo, hi) for t in base_tr])
    bv = bv[np.isfinite(bv)]
    if len(bv) < 30:
        return np.nan, np.nan
    bhi = boot_median_ci(bv, rng, nb)[2]
    sel = np.where((Wi >= base_rng[1]) & (Wi <= wmax))[0]
    sel = sel[np.argsort(Wi[sel])]
    prev = None
    for b in np.array_split(sel, nfine):
        vals = np.array([stat(pairs[t], lo, hi) for t in b])
        v = vals[np.isfinite(vals)]
        if len(v) < min_group:
            continue
        w = float(np.median(Wi[b]))
        if boot_median_ci(v, rng, nb)[0] > bhi:
            return (prev if prev is not None else w), w
        prev = w
    return np.nan, np.nan


def onset_map(Wi, pairs, LOS, HIS, rng, min_width, **kw):
    """2D map of onset bracket midpoints over the (band_lo, band_hi) rectangle."""
    res = np.full((len(HIS), len(LOS)), np.nan)
    for il, lo in enumerate(LOS):
        for ih, hi in enumerate(HIS):
            if hi < lo + min_width:
                continue
            a, b = onset_bracket(Wi, pairs, lo, hi, rng, **kw)
            if np.isfinite(a):
                res[ih, il] = 0.5 * (a + b)
        print(f"onset_map lo={lo} done", flush=True)
    return res


# ----------------------------- two-lag contraction rate -----------------------------

def twolag_k(z3c_list, ext_list, idx, cap, barrier_ext, dt,
             lags=(2, 4, 6, 8), ok_list=None):
    """Two-lag conditional contraction rate on barrier frames (ext < barrier_ext).

    slopes b(D) = cov(x_{t+D}, x_t)/var(x_t) = alpha * exp(-k D); the ln-slope
    regression over D removes the measurement-noise attenuation alpha.
    k > 0: fold restored (saddle); k < 0: fold amplified (max).
    dt = time per frame; cap = amplitude window (coil scale).
    """
    lags = np.asarray(lags)
    Lm = int(lags.max())
    x0s = {L: [] for L in lags}; xls = {L: [] for L in lags}
    for tr in idx:
        e = np.asarray(ext_list[tr], float)
        f = np.asarray(z3c_list[tr], float)
        ok = (ok_list[tr] if ok_list is not None else np.isfinite(e))
        good = ok[:-Lm] & (e[:-Lm] < barrier_ext)
        for L in lags:                      # pair frames must both be ok
            good &= ok[L:len(ok) - Lm + L]
        sel = np.where(good)[0]
        if len(sel) == 0:
            continue
        for L in lags:
            x0s[L].append(f[sel]); xls[L].append(f[sel + L])
    if not x0s[lags[0]]:
        return np.nan, 0
    slopes = []
    for L in lags:
        x0 = np.concatenate(x0s[L]); xl = np.concatenate(xls[L])
        m = np.abs(x0 - x0.mean()) < cap
        x0c = x0[m] - x0[m].mean(); xlc = xl[m] - xl[m].mean()
        slopes.append((x0c * xlc).mean() / (x0c * x0c).mean())
    slopes = np.array(slopes)
    if np.any(slopes <= 0):
        return np.nan, int(m.sum())
    A = np.stack([lags * dt, np.ones_like(lags, float)], 1)
    coef, *_ = np.linalg.lstsq(A, np.log(slopes), rcond=None)
    return float(-coef[0]), int(m.sum())
