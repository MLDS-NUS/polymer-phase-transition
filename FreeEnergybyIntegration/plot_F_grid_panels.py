#!/usr/bin/env python3
"""The four panels of F_z2_z3_grid.png, each on its own.

    F2_Wi0.02      F_{2,n}(z_2)            top left
    F2_Wi1.32      F_{2,n}(z_2)            top right
    F23cut_Wi1.32  F_{23,n}(z_2=0, z_3)    bottom left
    F23cut_Wi5.28  F_{23,n}(z_2=0, z_3)    bottom right

They are sized like the four curve plots split out of the learned-model grid
(../examples/UIDD_polymer_dynamics_wi/figs/{F2,F23cut}_Wi*_seed1, written by
../examples/UIDD_polymer_dynamics_wi/2-trans-z3/plot_surface_grid_panels.py):
the axes frame is one panel of that grid -- FIGSIZE and SPACING read at run time
by plot_F_grid_z2_z3.reference_style(), 5.24 x 3.34 in at present -- and the
font sizes are the same, so the eight plots line up.  (In F_z2_z3_grid.png
itself the panels are ~1% smaller: atomic_savefig() shrinks that canvas to fit
the pinned raster.)

The drawing is plot_F_grid_z2_z3.draw_row() itself, called with a row's two
panels on two separate figures, so each row keeps the grid's shared y range.
Its axis labels and "Wi = ..." titles are then hidden (added by hand); tick
labels and legends are kept; and its line widths are replaced by CURVE_LINES of
the learned-model splitter, so all eight curve plots share one line style.
Saved like the learned-model plots: png at PNG_DPI, pdf/svg at VECTOR_DPI, and
on the same fixed canvas -- the blank around the frame is CURVE_MARGIN, parsed
at run time from that folder's plot_surface_grid_panels.py -- so all eight
curve plots have the same pixel size and the frame at the same place.  Run it
in the env those were made in (UIDD-cuda12, matplotlib 3.10.8) for identical
text metrics.

    python plot_F_grid_panels.py

Writes figs/<name>.png / .pdf / .svg
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_F_grid_z2_z3 as G

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
# the learned-model splitter, whose CURVE_MARGIN (blank around the frame, in
# inches) and CURVE_LINES (line widths) all eight curve plots share
REF_PANELS = G.REF_DIR / "plot_surface_grid_panels.py"


def panel_size(style: dict) -> tuple[float, float]:
    """(width, height) in inches of one panel of the reference 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=style["figsize"])
    fig.subplots_adjust(**style["spacing"])
    box = axes[0, 0].get_position()
    plt.close(fig)
    return box.width * style["figsize"][0], box.height * style["figsize"][1]


def new_axes(w: float, h: float, pad: dict):
    """A figure holding one axes of exactly w x h inches, `pad` around it."""
    W, H = w + pad["left"] + pad["right"], h + pad["bottom"] + pad["top"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([pad["left"] / W, pad["bottom"] / H, w / W, h / H])
    return fig, ax


def restyle(ax, lines: dict, set_bar_lw: float) -> None:
    """Swap draw_row()'s line widths for the shared CURVE_LINES ones."""
    for line in ax.get_lines():
        if line.get_linestyle() == ":":                    # the z = 0 line
            line.set_linewidth(lines["zero"])
        elif not line.get_label().startswith("_"):         # a labelled curve
            line.set_linewidth(lines["curve"])
        elif line.get_linestyle() == "-" and line.get_linewidth() == set_bar_lw:
            line.set_linewidth(lines["set_bar"])           # the maximizer set
    for handle in ax.get_legend().legend_handles:          # copies, made earlier
        handle.set_linewidth(lines["curve"])
    ax.tick_params(width=lines["frame"], length=lines["tick_len"])
    for spine in ax.spines.values():
        spine.set_linewidth(lines["frame"])
    ax.grid(True, alpha=0.3, lw=lines["grid"])


def save(fig, name: str, style: dict) -> None:
    """png/pdf/svg on the fixed canvas; refuses a plot whose labels overrun it."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    if min(tb.x0, tb.y0, W - tb.x1, H - tb.y1) < -1e-6:
        raise ValueError(f"{name}: labels overrun CURVE_MARGIN of {REF_PANELS.name}")
    OUT.mkdir(exist_ok=True)
    for ext, dpi in (("png", style["png_dpi"]), ("pdf", style["vector_dpi"]),
                     ("svg", style["vector_dpi"])):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi)
        print(f"  -> {path}")
    plt.close(fig)


def main() -> None:
    z3mod = G.load_z3_module()
    z2_series = G.load_z2_series()
    G.check_z2_against_published(z2_series)
    z3_series = [z3mod.load_series(*spec) for spec in z3mod.SERIES]
    edges = next(item["edges"] for item in z3_series if item["key"] == "ninf")

    style = G.reference_style()
    fs = style["fs"]
    matplotlib.rcParams["pdf.fonttype"] = 42
    w, h = panel_size(style)
    shared = G.read_reference_constants(REF_PANELS, {"CURVE_MARGIN", "CURVE_LINES"})
    pad, lines = shared["CURVE_MARGIN"], shared["CURVE_LINES"]
    print(f"  panel {w:.4f} x {h:.4f} in; tick={fs['tick']}, label={fs['label']}, "
          f"title={fs['title']}, legend={fs['legend']}")

    rows = (
        (z2_series, G.Z2_WI, z2_series[0]["z"], r"$z_2$", r"$F_{2,n}(z_2)$",
         None, G.Z2_COLOR, ("F2_Wi0.02", "F2_Wi1.32")),
        (z3_series, G.Z3_WI, z3_series[0]["z3"], r"$z_3$",
         r"$F_{23,n}(z_2{=}0,z_3)$", edges, G.Z3_COLOR,
         ("F23cut_Wi1.32", "F23cut_Wi5.28")),
    )
    for series, targets, z, xlabel, ylabel, row_edges, color, names in rows:
        figs, axes = zip(*(new_axes(w, h, pad) for _ in targets))
        G.draw_row(list(axes), series, targets, z, xlabel, ylabel, row_edges,
                   z3mod, color, fs)
        for ax in axes:                     # hidden, not blanked: an empty but
            for text in (ax.xaxis.label, ax.yaxis.label, ax.title):  # visible label
                text.set_visible(False)     # would still pad the tight box
            restyle(ax, lines, z3mod.SET_BAR_LW)
        for fig, name in zip(figs, names):
            save(fig, name, style)


if __name__ == "__main__":
    main()
