"""The seven distinct plots of F_z2_z3_grid_seed<seed>.png, each on its own.

The grid's four panels are a curve with a surface inset each, and two of the
insets show the same surface (Wi = 0.66), so there are seven distinct plots:

    F2_Wi0.01      F_2(Z_2)              top-left curve
    F2_Wi0.66      F_2(Z_2)              top-right curve
    F23cut_Wi0.66  F_23(Z_2=0, Z_3)      bottom-left curve
    F23cut_Wi2.64  F_23(Z_2=0, Z_3)      bottom-right curve
    F23_Wi0.01     F_23(Z_2, Z_3)        top-left inset
    F23_Wi0.66     F_23(Z_2, Z_3)        top-right inset (the bottom-left inset is
                                         the same surface plus the Z_2 = 0 line)
    F23_Wi2.64     F_23(Z_2, Z_3)        bottom-right inset, without its Z_2 = 0 line

The grid is first laid out exactly as plot_surface_grid.py lays it out (and not
saved); every plot is then redrawn from the same saved data at the size, ranges,
ticks and fonts it has there, so the pieces fit back together.  The four curves
keep the grid's common y range; their dotted zero line, which in the grid stops
under the inset, runs the full height.  No axis labels and no "Wi = ..." titles
(added by hand); tick labels and colour bars are kept.

The curves are saved on one fixed canvas, frame at the same place, shared with
the four of FreeEnergybyIntegration/plot_F_grid_panels.py (which reads
CURVE_MARGIN from here), so all eight have the same pixel size and line up at
any inserted width.  The surfaces are cropped to their content.

    python plot_surface_grid_panels.py              # seed 1 by default
    MODEL_IDX=0 python plot_surface_grid_panels.py  # another seed (0 = seed 123)

Writes ../figs/<name>_seed<seed>.png / .pdf / .svg  (heatmaps embedded in the
pdf/svg at VECTOR_DPI, as in the grid).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

from plot_surface_grid import FIGSIZE, HERE, PNG_DPI, SPACING, VECTOR_DPI, load_script

OUT = HERE.parent / "figs"
# Inches of blank around a curve plot's frame, the same for all eight curve
# plots (FreeEnergybyIntegration/plot_F_grid_panels.py parses this assignment,
# so keep it a literal): the widest tick labels of the eight + ~0.1 in -- with
# the CURVE_LINES tick length they reach left 0.40 (FreeEnergy's "0.0"), right
# 0.06 (the "6" on the right edge of F23cut_Wi0.66), bottom 0.30, top 0.004.
# save() refuses a plot that overruns it.
CURVE_MARGIN = dict(left=0.49, right=0.16, bottom=0.38, top=0.11)
# Line widths (pt) of all eight curve plots, also parsed by
# FreeEnergybyIntegration/plot_F_grid_panels.py: the curves, the dotted z = 0
# line, frame and ticks (tick length in pt), grid, and FreeEnergy's thick
# maximizer-set bar.  Heavier than in the two grids, which stay as they are.
CURVE_LINES = dict(curve=3.6, zero=1.5, frame=1.6, tick_len=5.0, grid=1.2, set_bar=5.5)
# blank around a surface plot, cropped again to the content on saving
SURF_PAD = dict(left=0.6, right=1.0, bottom=0.6, top=0.2)


def build_grid(mi):
    """The 2x2 grid exactly as plot_surface_grid.py builds it.  Returns the seed,
    the figure, the four panels' items (top row first) and the two modules with
    their data."""
    z2 = load_script(HERE.parent / "1-trans-z2" / "plot_z2_z3_surface.py")
    z3 = load_script(HERE / "plot_z3_surface.py")
    seed, d2 = z2.get_data(mi)
    _, d3 = z3.get_data(mi)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.subplots_adjust(**SPACING)
    items2, top2 = z2.draw_panels(fig, axes[0], d2)
    items3, top3 = z3.draw_panels(fig, axes[1], d3)
    z3.clear_insets(fig, items2 + items3, z3.Y_LO, max(top2, top3))
    return seed, fig, items2 + items3, (z2, d2), (z3, d3)


def size_in(ax, fig):
    """(width, height) of an axes' frame in inches."""
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def new_axes(w, h, pad):
    """A figure holding one axes of exactly w x h inches."""
    W, H = w + pad["left"] + pad["right"], h + pad["bottom"] + pad["top"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([pad["left"] / W, pad["bottom"] / H, w / W, h / H])
    return fig, ax


def draw_curve(src, mod, curve_style):
    """One grid curve (main axes `src`) on its own axes of the same size, in its
    colour but with the CURVE_LINES widths."""
    fig, ax = new_axes(*size_in(src, src.figure), CURVE_MARGIN)
    line = src.get_lines()[0]                              # the curve, drawn first
    ax.plot(line.get_xdata(), line.get_ydata(), "-", color=curve_style["color"],
            lw=CURVE_LINES["curve"])
    ax.axvline(0, ls=":", color="k", lw=CURVE_LINES["zero"])
    ax.set_xlim(src.get_xlim())
    ax.set_ylim(src.get_ylim())
    ax.set_xticks(src.get_xticks())
    ax.set_yticks(src.get_yticks())
    ax.set_xlim(src.get_xlim())                            # set_xticks may widen it
    ax.set_ylim(src.get_ylim())
    ax.tick_params(labelsize=mod.FS["tick"], width=CURVE_LINES["frame"],
                   length=CURVE_LINES["tick_len"])
    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_LINES["frame"])
    ax.grid(alpha=0.3, lw=CURVE_LINES["grid"])
    return fig


def draw_surface(src, mod, z2, z3, F23, view2, view3):
    """One grid inset (inset axes `src`) on its own axes of the same size; the
    drawing is that of the modules' draw_panels."""
    fig, axi = new_axes(*size_in(src, src.figure), SURF_PAD)
    m2 = np.abs(z2) <= view2
    m3 = np.abs(z3) <= view3
    S = (F23 - F23.min())[np.ix_(m2, m3)].T                 # [z3, z2]
    pcm = axi.pcolormesh(z2[m2], z3[m3], np.maximum(S, 1e-1), shading="auto",
                         cmap="viridis", rasterized=True,
                         norm=LogNorm(vmin=1e-1, vmax=max(S.max(), 1.0)))
    axi.contour(z2[m2], z3[m3], S, levels=[l for l in mod.LEVELS if l < S.max()],
                colors="k", linewidths=0.4, alpha=0.35)
    axi.tick_params(labelsize=mod.FS["inset_tick"])
    axi.set_xlim(-view2, view2)
    axi.set_ylim(-view3, view3)
    axi.xaxis.set_major_locator(MultipleLocator(2))
    axi.yaxis.set_major_locator(MultipleLocator(2))
    cb = fig.colorbar(pcm, cax=axi.inset_axes([1.03, 0.0, 0.045, 1.0]))
    cb.ax.tick_params(labelsize=mod.FS["cbar_tick"])
    return fig


def check_fits(fig, name):
    """Raise if anything drawn lies outside the canvas (a fixed-canvas plot)."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    if min(tb.x0, tb.y0, W - tb.x1, H - tb.y1) < -1e-6:
        raise ValueError(f"{name}: labels overrun CURVE_MARGIN; widen it")


def save(fig, name, seed, fixed=False):
    """png/pdf/svg; `fixed` keeps the canvas as it is (curve plots), otherwise
    the file is cropped to the content."""
    if fixed:
        check_fits(fig, name)
    OUT.mkdir(exist_ok=True)
    for ext, dpi in (("png", PNG_DPI), ("pdf", VECTOR_DPI), ("svg", VECTOR_DPI)):
        out = OUT / f"{name}_seed{seed}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches=None if fixed else "tight")
        print(f"  -> {out}", flush=True)
    plt.close(fig)


def main():
    mi = int(os.environ.get("MODEL_IDX", "1"))
    seed, grid, items, (z2, d2), (z3, d3) = build_grid(mi)
    tl, tr, bl, br = items
    wi2, wi3 = [float(w) for w in d2["wi"]], [float(w) for w in d3["wi"]]
    assert wi2 == [0.01, 0.66] and wi3 == [0.66, 2.64], (wi2, wi3)

    # the four curves
    save(draw_curve(tl["ax"], z2, z2.CURVE), "F2_Wi0.01", seed, fixed=True)
    save(draw_curve(tr["ax"], z2, z2.CURVE), "F2_Wi0.66", seed, fixed=True)
    save(draw_curve(bl["ax"], z3, z3.CURVE), "F23cut_Wi0.66", seed, fixed=True)
    save(draw_curve(br["ax"], z3, z3.CURVE), "F23cut_Wi2.64", seed, fixed=True)

    # the three distinct surfaces: Wi = 0.01 and 0.66 from the top row,
    # Wi = 2.64 from the bottom row (drawn without its z2 = 0 line)
    for n, name in ((0, "F23_Wi0.01"), (1, "F23_Wi0.66")):
        src = (tl, tr)[n]["axi"]
        save(draw_surface(src, z2, d2["z2"], d2["z3"], d2["F23"][n],
                          z2.INSET_Z2, z2.INSET_Z3), name, seed)
    save(draw_surface(br["axi"], z3, d3["z2"], d3["z3"], d3["F23"][1],
                      z3.Z2_VIEW, z3.Z3_VIEW), "F23_Wi2.64", seed)
    plt.close(grid)


if __name__ == "__main__":
    main()
