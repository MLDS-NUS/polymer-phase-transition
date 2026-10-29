"""The n -> infinity z2 optimum and z3 minimizer-set edges side by side.

Combines the two single-panel figures produced by
    1-trans-z2/wi_001_10/plot_argmax_vs_wi_001_10_ninf.py
    2-trans-z3/wi_001_10/plot_maximizer_vs_wi_001_10_ninf.py
using their own loaders, so the plotted numbers are identical.  Figure size,
DPI, fonts, Wi range, panel geometry and colours (z2 #1f77b4, z3 black, Wi_t
orange dashed) follow examples/UIDD_polymer_dynamics_wi/plot_transition_combined.py.

Sign convention: the stored F_all is maximised, so it is minus the free energy
of the reference figure; its maximizer set is labelled "minimizer set" here.

The orange line is the n -> infinity transition Wi_t:
  z2  where F''(0) changes sign, i.e. z = 0 stops being a maximum of F(z2);
      F'' from the 5-point stencil on F_all, crossing linearly interpolated.
      It lands on beta = pi^2 to ~1e-4.
  z3  the minimizer-set onset wi_plateau_onset stored by the solver
      (beta = 4 pi^2).

    python plot_argmax_ninf_combined.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "1-trans-z2" / "wi_001_10"))
sys.path.insert(0, str(HERE / "2-trans-z3" / "wi_001_10"))
import plot_argmax_vs_wi_001_10 as z2_plot      # noqa: E402
import plot_maximizer_vs_wi_001_10 as z3_plot   # noqa: E402

# --- style, copied from plot_transition_combined.py -------------------------
FS = dict(label=20, tick=14, annot=15)
FORMATS = ("png", "pdf")
DPI = 150
XLIM = (0.02, 10.0)               # wi_values range of both reference panels
# That figure is saved with bbox_inches="tight", and its colorbars set its
# width, so its geometry is reproduced here in pixels instead: the canvas,
# panel side, panel bottom and panel-to-panel gap measured on
# transition_combined.png (spines at x = 122, 1085; y = 14..722).  The pair is
# shifted right by 57 px, which centres the ink: the z2 tick labels (-0.5) are
# wider than the reference's (-2) and would otherwise clip the z2 label.
CANVAS_PX = (1927, 832)
AXES_PX = 708                     # square panel side
AXES_X0_PX = (179, 1142)          # panel left edges
AXES_Y0_PX = 110                  # panel bottom, from the canvas bottom
# ---------------------------------------------------------------------------


def z2_wi_c(path):
    """First Wi where F''(0) of the n -> infinity F(z2) turns positive."""
    with np.load(path, allow_pickle=False) as d:
        wi, z, F = d["wi_values"], d["z_grid"], d["F_all"]
    c = int(np.argmin(np.abs(z)))
    h = z[c + 1] - z[c]
    d2 = (-F[:, c + 2] + 16 * F[:, c + 1] - 30 * F[:, c]
          + 16 * F[:, c - 1] - F[:, c - 2]) / (12 * h**2)
    i = int(np.flatnonzero((d2[:-1] < 0) & (d2[1:] >= 0))[0])
    return float(wi[i] + (wi[i + 1] - wi[i]) * (-d2[i]) / (d2[i + 1] - d2[i]))


def load_z2():
    spec = next(s for s in z2_plot.SERIES if s[0] == "ninf")
    s = z2_plot.load_series(*spec)
    return dict(wi=s["wi"], lower=s["neg"], upper=s["pos"], wi_c=z2_wi_c(spec[2]),
                band=None, base="#1f77b4", markevery=1, ystep=0.5,
                ylabel=r"$z_2$")


def load_z3():
    s = z3_plot.load_ninf(z3_plot.DEFAULT_DATA_DIR, allow_partial=False)
    with np.load(s.source, allow_pickle=False) as d:
        wi_c = float(d["wi_plateau_onset"])
    return dict(wi=s.wi, lower=s.lower, upper=s.upper, wi_c=wi_c,
                band="minimizer set", base="k", markevery=12, ystep=0.2,
                ylabel=r"$z_3$")


def draw_panel(ax, p):
    style = dict(color=p["base"], lw=2.4, marker="D", ms=5.0,
                 markevery=p["markevery"], markerfacecolor=p["base"],
                 markeredgecolor=p["base"])
    if p["band"]:
        ax.fill_between(p["wi"], p["lower"], p["upper"], color=p["base"],
                        alpha=0.12, linewidth=0, zorder=1, label=p["band"])
    ax.axvline(p["wi_c"], ls="--", color="C1", lw=1.8, zorder=2,
               label=rf"$\mathrm{{Wi}}_t = {p['wi_c']:.2f}$")
    ax.plot(p["wi"], p["upper"], "-", **style)
    ax.plot(p["wi"], p["lower"], "-", **style)

    finite = np.concatenate([p["lower"][np.isfinite(p["lower"])],
                             p["upper"][np.isfinite(p["upper"])]])
    y_min, y_max = float(finite.min()), float(finite.max())
    y_pad = max(0.02, 0.08 * (y_max - y_min))
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.yaxis.set_major_locator(MultipleLocator(p["ystep"]))
    ax.set_xlabel(r"$\mathrm{Wi}$", fontsize=FS["label"])
    ax.set_ylabel(p["ylabel"], fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(False)
    ax.legend(fontsize=FS["annot"], loc="lower left", framealpha=0.92)


def main():
    W, H = CANVAS_PX
    # +0.01 px guards against Matplotlib rounding the raster one pixel short
    fig = plt.figure(figsize=((W + 0.01) / DPI, (H + 0.01) / DPI))
    panels = (load_z2(), load_z3())
    for x0, p in zip(AXES_X0_PX, panels):
        ax = fig.add_axes([x0 / W, AXES_Y0_PX / H, AXES_PX / W, AXES_PX / H])
        draw_panel(ax, p)
    for ext in FORMATS:
        path = HERE / f"argmax_ninf_combined.{ext}"
        plt.savefig(path, dpi=DPI)
        print(f"  -> {path}", flush=True)
    plt.close()
    for key, p in zip(("z2", "z3"), panels):
        print(f"     {key}: Wi_t = {p['wi_c']:.6f}   "
              f"beta_c = {p['wi_c'] * z2_plot.WI_TO_BETA:.6f}")


if __name__ == "__main__":
    main()
