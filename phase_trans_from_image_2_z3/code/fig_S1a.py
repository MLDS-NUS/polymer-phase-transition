"""Fig. S1a — how the uncentered fold coordinate Z~_3 is measured on one frame.

Standalone version of Fig. S1(a) on a shorter, more clearly folded 40 V chain,
cropped to the molecule, with the retained mask, the axial extrema x_min / x_max, the span midpoint and
the intensity-weighted centroid x_com.  Below the image, the column-summed mask
intensity, whose mean is x_com.  The figure carries no text at all: labels are
added by hand afterwards, in a narrow blank margin kept above the guide lines.

    Z~_3 = x_max + x_min - 2 x_com    (= Z_3' of Fig. S1 = q of the paper)

The frame (40V_Tra_9, frame 67, L = 105 px) was picked by eye among the 40 V
frames that pass quality selection, lie in the analysis band
0.30 <= L/L_max < 0.55, form a single connected component and have the largest
|Z~_3|/L; the script asserts the first three.

Loads the raw frames of one 40 V trajectory (the pickle is a few GB; only the
selected trajectory is kept).

    python3 fig_S1a.py
"""
import os, sys
_BASE = os.path.dirname(os.path.abspath(__file__))     # .../phase_trans_from_image_2_z3/code
BASE = os.path.abspath(os.path.join(_BASE, ".."))      # the project folder
REPO = os.path.abspath(os.path.join(BASE, ".."))       # the repository root
FIG_SUPP = os.path.join(BASE, "figures", "supplement")
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(REPO, "phase_trans_from_image"))
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import frame_features as FF
import indicators as I

PKL = os.path.join(REPO, "data/exp_data/voltage-frames/40V_Feb25.pkl")
CACHE = os.path.join(REPO, "data/exp_data/cache_voltage_frames")
TRAJ, j = "40V_Tra_9", 67               # see the docstring
LMAX, BAND = 303.0, (0.30, 0.55)

# ------------------------------------------------------------------ data
print("loading raw frames ...", flush=True)
D = pickle.load(open(PKL, "rb"))
frames = np.asarray(D[TRAJ])
del D
d = np.load(os.path.join(CACHE, f"40V__Feb25__{TRAJ}.npz"))
e, z3p_raw, cx, bd = d["ext"], d["z3p"], d["cx"], d["border"]
ok = I.qc_mask(e, bd, LMAX)
z3c = I.fold_coordinate(z3p_raw, cx, ok)
assert ok[j] and BAND[0]*LMAX <= e[j] < BAND[1]*LMAX and d["ncomp"][j] == 1

img = np.asarray(frames[j], float)
if np.median(img) > 128:                 # pkl frames are stored white-background;
    img = 255.0 - img                    # flip as extract_pkl_batches.py does
m = img >= FF.THR                        # same segmentation as frame_scalars:
lab, n = FF.ndimage.label(m)             # threshold, then drop components smaller
if n > 1:                                # than 10 % of the largest one
    sz = FF.ndimage.sum(m, lab, np.arange(1, n + 1))
    m = np.isin(lab, np.where(sz >= FF.CC_FRAC*sz.max())[0] + 1)
ys, xs = np.nonzero(m)
w = img[ys, xs]
xmin, xmax = int(xs.min()), int(xs.max())
xc, yc = (xs*w).sum()/w.sum(), (ys*w).sum()/w.sum()
assert abs(xc - cx[j]) < 1e-6 and xmax + xmin == z3p_raw[j]       # = the cached features
mid = 0.5*(xmin + xmax)
Z3t = xmax + xmin - 2*xc
print(f"frame {j}:  L = {xmax-xmin} px,  x_com = {xc:.2f},  Z3~ = {Z3t:+.2f} px,  "
      f"Z3,c = {z3c[j]:+.2f} px")

# ------------------------------------------------------------------ style
C_EXT = "#1f5aa6"       # extrema and L
C_MID = "#c77dda"       # span midpoint (purple, not green: red-green colour blindness)
C_CEN = "#d62728"       # centroid
C_Z3 = "k"              # Z~_3/2 arrow
C_MSK = "#f28e2b"       # retained mask
LW = 2.2                # guide lines and arrows

PAD = 22                                         # px of background either side
X0, X1 = xmin - PAD, xmax + PAD
HY = max(11, int(np.ceil(max(yc - ys.min(), ys.max() - yc))) + 5)   # half-height
Y0, Y1 = int(round(yc)) - HY, int(round(yc)) + HY
crop = img[Y0:Y1 + 1, X0:X1 + 1]


# vertical layout, in image px (y grows downward).  The offsets were set on a
# 204 px wide crop; they scale with the crop width so that, at a fixed figure
# width, the strips and arrows keep their printed size whatever the chain length.
SV = (X1 - X0 + 1)/204
Y_TOP = Y0 - 0.5                                 # top edge of the image
Y_BOT = Y1 + 0.5                                 # bottom edge of the image
P_H = 17.0*SV                                    # height of the intensity strip
P_BASE = Y_BOT + 4.0*SV + P_H                    # its baseline
Y_L = P_BASE + 9.0*SV                            # L arrow
Y_ARR = Y_TOP - 5.5*SV                           # Z~_3/2 arrow
Y_TIP = Y_TOP - 11.0*SV                          # top of the guide lines
Y_MARGIN = 7.0*SV                                # blank above them, for hand labels

# the axes fill the whole figure and the figure has the aspect of the data
# window, so the blank margin survives in the saved file
XLIM = (X0 - 1.5, X1 + 1.5)
YLIM = (Y_L + 2.5*SV, Y_TIP - Y_MARGIN)
FIG_W = 7.05
fig = plt.figure(figsize=(FIG_W, FIG_W*(YLIM[0] - YLIM[1])/(XLIM[1] - XLIM[0])))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.imshow(crop, cmap="Greys", vmin=0, vmax=np.percentile(w, 99),
          extent=(X0 - 0.5, X1 + 0.5, Y_BOT, Y_TOP), interpolation="nearest", zorder=0)
ax.add_patch(Rectangle((X0 - 0.5, Y_TOP), X1 - X0 + 1, Y_BOT - Y_TOP,
                       fill=False, ec="0.6", lw=1.0, zorder=1))
ax.contour(np.arange(X0, X1 + 1), np.arange(Y0, Y1 + 1), m[Y0:Y1 + 1, X0:X1 + 1].astype(float),
           levels=[0.5], colors=[C_MSK], linewidths=1.6, zorder=2)

# column-summed mask intensity; x_com is its balance point
prof = np.bincount(xs - X0, weights=w, minlength=X1 - X0 + 1)
xx = np.arange(X0, X1 + 1)
yy = P_BASE - P_H*prof/prof.max()
ax.fill_between(xx, P_BASE, yy, step="mid", color="0.85", lw=0, zorder=1)
ax.step(xx, yy, where="mid", color="0.3", lw=1.3, zorder=2)
ax.plot([X0 - 0.5, X1 + 0.5], [P_BASE, P_BASE], color="0.3", lw=1.2,
        solid_capstyle="butt", zorder=2)
ax.plot(xc, P_BASE + 2.4*SV, "^", ms=10, mfc=C_CEN, mec="w", mew=1.0, zorder=5,
        clip_on=False)

# guide lines: the extrema run down to the L arrow, midpoint and centroid stop
# at the profile baseline
for xv, c, ls, y_lo in ((xmin, C_EXT, "-", Y_L), (xmax, C_EXT, "-", Y_L),
                        (mid, C_MID, (0, (3.5, 2.0)), P_BASE), (xc, C_CEN, "-", P_BASE)):
    ax.plot([xv, xv], [y_lo, Y_TIP], color=c, lw=LW, ls=ls, zorder=3,
            solid_capstyle="butt", dash_capstyle="butt")

# centroid marker on the frame
ax.plot(xc, yc, "o", ms=7.5, mfc=C_CEN, mec="w", mew=1.4, zorder=5)

# arrows
arr = dict(arrowstyle="<|-|>", lw=LW, shrinkA=0, shrinkB=0, mutation_scale=11)
ax.annotate("", xy=(xmax, Y_L), xytext=(xmin, Y_L),
            arrowprops=dict(color=C_EXT, **arr), zorder=4)
ax.annotate("", xy=(xc, Y_ARR), xytext=(mid, Y_ARR),
            arrowprops=dict(color=C_Z3, **arr), zorder=4)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.axis("off")

for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_SUPP, f"fig_S1a_Z3tilde.{ext_}"), dpi=200)
print("wrote fig_S1a_Z3tilde.png / .pdf")
