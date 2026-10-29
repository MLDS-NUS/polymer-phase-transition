"""SHARED image -> per-frame scalars (single source of truth).

Both extractors call `frame_scalars` on a 2D grayscale frame (molecule bright
on ~zero background, 8-bit scale):
  * experiment: masked/ PNG frames from the voltage-frames tars;
  * simulation: frames RENDERED from bead coordinates (render.py) — the
    simulation is only ever observed through this same image pipeline.

Per frame: theta (gyration long-axis), ext = xmax-xmin, z3p = xmax+xmin,
npts, aniso, border-contact fraction, intensity centroid (cx, cy), component
count.  The fold coordinate used downstream is z3c = z3p - 2*cx
(crop-invariant), per-trajectory median-subtracted (see indicators.py).

Noise handling: THR mask threshold (8-bit counts), connected components
>= CC_FRAC of the largest, frames with < MIN_PTS masked pixels -> NaN row.
"""
import numpy as np
from scipy import ndimage

THR, CC_FRAC, MIN_PTS = 20, 0.10, 40
NFEAT = 14
FIELDS = ("theta", "ext", "z3p", "npts", "aniso", "border", "cx", "cy", "ncomp",
          # --- fold observables along the long axis (all end-swap even, label-free) ---
          "xskew",   # standardised 3rd moment of intensity along the long axis
          "qfold",   # quantile version of z3p, normalised: (q95+q05)/2 - cx, / (q95-q05)
          "densasym",# (outer-third mass, high side - low side) / total
          "peakpos", # position of the densest decile, in units of the q05-q95 span, from centre
          "linden")  # linear density: total intensity / span  (fold doubles it)


def mask_scalars(a, m, w=None, ncomp=None):
    """Measurement core: the 9 scalars from a boolean mask m over frame a.

    w: per-pixel intensity weights over m in np.nonzero order (default: the
    raw pixel values — correct for zero-background images; the experiment
    side passes net signal a-bg).  ncomp: reported component count (default:
    label count of m).  This is the SINGLE implementation both sides use —
    segmentation (who decides m) lives with each side (sim: frame_scalars
    below; experiment: exp_data_image/mask.py).
    """
    H, W = a.shape
    ys, xs = np.nonzero(m)
    if len(xs) < MIN_PTS:
        return (np.nan,) * NFEAT
    w = (a[ys, xs] if w is None else w).astype(np.float64)
    sw = w.sum()
    cx = (xs * w).sum() / sw; cy = (ys * w).sum() / sw
    dx, dy = xs - cx, ys - cy
    Gxx = (w*dx*dx).sum()/sw; Gyy = (w*dy*dy).sum()/sw; Gxy = (w*dx*dy).sum()/sw
    theta = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
    tr = Gxx + Gyy; disc = max(tr * tr / 4 - (Gxx * Gyy - Gxy * Gxy), 0) ** 0.5
    aniso = (2 * disc) / (tr + 1e-12)
    xmax, xmin = int(xs.max()), int(xs.min())
    border = float(((xs <= 1) | (xs >= W - 2)).mean())
    if ncomp is None:
        ncomp = ndimage.label(m)[1]

    # --- fold observables measured ALONG THE LONG AXIS (u), not the image x ---
    # The fold mode is "both ends on one side, middle on the other".  z3p
    # (= xmax+xmin-2cx) vanishes for a symmetric hairpin, whose centroid and
    # geometric centre coincide, so it is blind to exactly the mode we want.
    # These candidates are signed, label-free and (except linden) dimensionless,
    # hence immune to pure dilation by construction.
    c, s_ = np.cos(theta), np.sin(theta)
    u = dx * c + dy * s_                      # coordinate along the long axis
    su = np.sqrt((w * u * u).sum() / sw) + 1e-9
    xskew = float((w * u ** 3).sum() / sw / su ** 3)
    order = np.argsort(u)
    cw = np.cumsum(w[order]) / sw
    q05, q50, q95 = (float(u[order][np.searchsorted(cw, q)])
                     for q in (0.05, 0.50, 0.95))
    span = max(q95 - q05, 1e-9)
    qfold = float(((q95 + q05) / 2.0) / span)          # centroid is at u=0 by construction
    lo_third = u < q05 + span / 3.0
    hi_third = u > q95 - span / 3.0
    densasym = float(((w * hi_third).sum() - (w * lo_third).sum()) / sw)
    nb = 12
    edges_u = np.linspace(q05, q95, nb + 1)
    prof = np.array([(w * ((u >= edges_u[i]) & (u < edges_u[i + 1]))).sum()
                     for i in range(nb)])
    peakpos = float((0.5 * (edges_u[int(np.argmax(prof))] + edges_u[int(np.argmax(prof)) + 1]))
                    / span)
    linden = float(sw / span)
    return (float(theta), float(xmax - xmin), float(xmax + xmin), float(len(xs)),
            float(aniso), border, float(cx), float(cy), float(ncomp),
            xskew, qfold, densasym, peakpos, linden)


def frame_scalars(a):
    """9 scalars from one zero-background frame (sim renders / masked frames):
    THR segmentation + CC filter + measurement core; (nan,)*9 if too few px."""
    m = a >= THR
    if int(m.sum()) < MIN_PTS:
        return (np.nan,) * NFEAT
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, np.arange(1, n + 1))
        m = np.isin(lab, np.where(sizes >= CC_FRAC * sizes.max())[0] + 1)
    return mask_scalars(a, m, ncomp=n)
