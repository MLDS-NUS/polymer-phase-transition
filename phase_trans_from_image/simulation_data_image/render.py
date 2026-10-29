"""Render simulation bead coordinates into experiment-like microscope frames.

The simulation is only ever OBSERVED through images: every trajectory frame is
rendered here (Gibson-Lanni PSF, camera noise, 8-bit) and then measured by the
same shared ../frame_features.frame_scalars as the experimental masked frames.
Bead coordinates are never used directly for the image-side features (the only
exceptions are the clearly-labeled truth controls: `endsum` and the reduced
|Z3| series).

Optics/geometry (conventions):
  * 1 simulation unit = 1 bond length = BOND_UM = 0.055 um = 1 camera pixel
    (so the feature px scale stays the historical one: L_max = 299 px);
  * frame = 100 x 500 tracking crop centered on the chain COM (the simulation
    keeps COM = 0), focal plane at z = 0 (chain z is centered by symmetry);
  * PSF: gLXYZParticleScan (Langevin_single_chain/microscPSF.py), particle
    depth PZ0_UM +- ZSPAN_UM above the coverslip, stack normalized so the
    in-focus slice sums to 1 -> per-bead photon budget = GAIN * 300/n_beads
    (ChangeBead: fewer beads, longer bonds, same light per contour length);
  * camera: Poisson shot noise on the expected counts, clipped to 8 bit.
    GAIN is auto-calibrated once (calibrate_gain) so the median on-molecule
    pixel matches the experimental masked frames (~55 counts); cached in
    data/sim_cache/render_calib.npz together with the PSF stack.

    python3 render.py demo    # render example frames -> render_demo.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "../../data/sim_cache"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../Langevin_single_chain")))

H, W = 100, 500            # camera crop (same as the experimental tracking crop)
BOND_UM = 0.055            # um per bond = um per pixel
WVL_UM = 0.51              # emission wavelength
PZ0_UM = 3.0               # nominal particle depth above the coverslip
ZSPAN_UM = 2.2             # rendered defocus range: |z| <= ZSPAN (beyond: last slice)
PSF_XY = 41                # PSF patch size (odd)
NB_REF = 300               # reference bead count (bead weight = NB_REF / n_beads)
TARGET_MED = 55.0          # calibration target: median on-molecule counts (exp ~40-70)
CALIB_FILE = f"{CACHE}/render_calib.npz"

M_PARAMS = {"M": 60.0, "NA": 1.4, "ng0": 1.515, "ng": 1.515, "ni0": 1.515,
            "ni": 1.515, "ns": 1.33, "ti0": 150, "tg": 170, "tg0": 170,
            "zd0": 200.0e3}


# ----------------------------- PSF stack -----------------------------

def build_psf_stack():
    """(nz, PSF_XY, PSF_XY) stack over z = PZ0 + [-ZSPAN, ZSPAN], step 1 px.

    The stage offset zv is first tuned (focal scan) so the microscope is
    focused AT depth PZ0 — with the ns/ni index mismatch the design focus
    (zv = 0) sits at the coverslip, not at the particle.  Normalized so the
    in-focus slice sums to 1; relative dimming with defocus is preserved.
    """
    from microscPSF import gLXYZFocalScan, gLXYZParticleScan
    zvs = np.arange(-8.0, 0.0 + 1e-9, 0.05)
    foc = gLXYZFocalScan(M_PARAMS, dxy=BOND_UM, xy_size=PSF_XY, zv=zvs,
                         normalize=False, pz=PZ0_UM, wvl=WVL_UM)
    zv_star = float(zvs[int(np.argmax(foc.max(axis=(1, 2))))])
    assert -7.9 < zv_star < -0.1, f"focal scan hit the zv range edge: {zv_star}"
    zrel = np.arange(-ZSPAN_UM, ZSPAN_UM + 1e-9, BOND_UM)
    stack = gLXYZParticleScan(M_PARAMS, dxy=BOND_UM, xy_size=PSF_XY,
                              pz=PZ0_UM + zrel, normalize=True, wvl=WVL_UM,
                              zv=zv_star)
    stack = np.asarray(stack, np.float32)
    ifocus = int(np.argmax(stack.max(axis=(1, 2))))
    nz = len(zrel)
    assert nz // 4 < ifocus < 3 * nz // 4, \
        f"in-focus slice {ifocus}/{nz} still at the stack edge (zv* = {zv_star})"
    stack /= stack[ifocus].sum()
    return stack, zrel.astype(np.float32), ifocus, zv_star


class Renderer:
    """Per-process renderer; loads the cached PSF stack + gain once."""

    def __init__(self):
        d = np.load(CALIB_FILE)
        self.stack, self.zrel, self.gain = d["stack"], d["zrel"], float(d["gain"])
        self.izmax = len(self.stack) - 1
        self.iz0 = int(d["ifocus"])   # sim z = 0 sits on the true focal slice
        self.half = PSF_XY // 2

    @staticmethod
    def _resample(coords, ds=1.0):
        """Emitters distributed along the CONTOUR at <= ds px arc spacing.

        Dye sits along the chain, not on beads: a coarse discretization
        (ChangeBead, 2 px bonds) must not render as a dotted line.  For the
        300-bead sets (1 px bonds) the sample points coincide with the beads,
        so their rendered frames are unchanged.
        """
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        L = float(seg.sum())
        if L < 1e-6 or seg.mean() <= 1.2 * ds:
            return coords          # bead spacing already <= px scale: unchanged
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(np.ceil(L / ds)) + 1, len(coords))
        t = np.linspace(0.0, L, n)
        return np.stack([np.interp(t, s, coords[:, k]) for k in range(3)], 1)

    def render_float(self, coords):
        """Expected counts image (float) from one frame of (nb, 3) sim coords.

        The virtual microscope tracks the molecule: crop centered on the
        contour centroid and focus on its z (like the experimental tracking
        crop + focus kept on the molecule).  Datasets whose simulation
        already keeps COM = 0 (the 300-bead sets) are rendered identically;
        z3c/ext are invariant to this centering by construction.
        """
        coords = self._resample(coords)
        coords = coords - coords.mean(0)
        img = np.zeros((H, W), np.float32)
        wgt = self.gain * NB_REF / len(coords)
        hp = self.half
        xp = np.rint(coords[:, 0] + (W - 1) / 2).astype(int)
        yp = np.rint((H - 1) / 2 - coords[:, 1]).astype(int)
        iz = np.clip(np.rint(coords[:, 2]).astype(int) + self.iz0, 0, self.izmax)
        for x, y, z in zip(xp, yp, iz):
            x0, x1 = x - hp, x + hp + 1
            y0, y1 = y - hp, y + hp + 1
            if x0 >= W or x1 <= 0 or y0 >= H or y1 <= 0:
                continue
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1 = PSF_XY - max(0, x1 - W); sy1 = PSF_XY - max(0, y1 - H)
            img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] += \
                wgt * self.stack[z, sy0:sy1, sx0:sx1]
        return img

    def render(self, coords, rng):
        """8-bit camera frame (Poisson shot noise, clipped).

        Shot noise is only sampled where the expected counts could plausibly
        cross the mask threshold (>5 expected counts: P(Poisson(5) >= THR=20)
        ~ 5e-7) — identical features, much cheaper than a full-frame draw.
        """
        img = self.render_float(coords)
        out = np.zeros(img.shape, np.uint8)
        m = img > 5.0
        out[m] = np.clip(rng.poisson(img[m]), 0, 255).astype(np.uint8)
        return out


# ----------------------------- calibration -----------------------------

def _sample_frames(n_traj=40, frames=(0, 200, 450, 700, 1000)):
    """Coordinate frames from the first shards of the main dataset (via
    extract_features helpers) for gain calibration."""
    import glob
    import pyarrow.parquet as pq
    from extract_features import HF_GLOB, DATASETS, SUB
    fs = sorted(glob.glob(HF_GLOB.format(repo=DATASETS["paramsweep_new"][0])))
    out = []
    for path in fs:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4):
            xcol = batch.column("x")
            for i in range(len(xcol)):
                x = np.array(xcol[i].as_py(), dtype=np.float32)
                nb = x.shape[1] // 3
                xs = x[::SUB].reshape(-1, nb, 3)
                out += [xs[t] for t in frames if t < len(xs)]
                if len(out) >= n_traj * len(frames):
                    return out
    return out


def calibrate(force=False):
    """Build the PSF stack and bisect GAIN so that the median on-molecule
    pixel (>= frame_features.THR) over sample frames hits TARGET_MED."""
    if os.path.exists(CALIB_FILE) and not force:
        return
    from frame_features import THR
    print("render calibration: building PSF stack ...", flush=True)
    stack, zrel, ifocus, zv_star = build_psf_stack()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(CALIB_FILE, stack=stack, zrel=zrel, ifocus=ifocus, zv=zv_star, gain=1.0)
    print(f"render calibration: zv* = {zv_star:.2f} um, in-focus slice "
          f"#{ifocus}/{len(zrel)}; sampling frames ...", flush=True)
    frames = _sample_frames()
    r = Renderer()
    base = [r.render_float(c) for c in frames]

    def med_at(gain):
        v = np.concatenate([b.ravel()[b.ravel() * gain >= THR] * gain for b in base])
        return np.median(v) if len(v) else 0.0

    lo, hi = 1.0, 1e6
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if med_at(mid) < TARGET_MED:
            lo = mid
        else:
            hi = mid
    gain = float(np.sqrt(lo * hi))
    np.savez(CALIB_FILE, stack=stack, zrel=zrel, ifocus=ifocus, zv=zv_star, gain=gain)
    print(f"render calibration: GAIN = {gain:.1f} "
          f"(median on-molecule counts {med_at(gain):.1f})", flush=True)


# ----------------------------- demo -----------------------------

def demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frame_features import frame_scalars
    calibrate()
    frames = _sample_frames(n_traj=1, frames=(0, 150, 300, 500, 750, 1000))
    r = Renderer(); rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(frames), 1, figsize=(9, 1.6 * len(frames)))
    for ax, c in zip(axes, frames):
        img = r.render(c, rng)
        s = frame_scalars(img.astype(np.float32))
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ext={s[1]:.0f}px z3c={s[2] - 2 * s[6]:+.1f} npts={s[3]:.0f} "
                     f"| coord ext={np.ptp(c[:, 0]):.0f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{HERE}/render_demo.png", dpi=130)
    print("saved render_demo.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calib":
        calibrate(force=True)
    else:
        demo()
