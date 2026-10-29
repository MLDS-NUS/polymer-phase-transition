"""The panels of transition_2x2.png, each on its own, plus the two insets.

    transition_z2_learned        top left, without its inset (colourbar kept)
    transition_z2_learned_inset  the Wi_t(eta) inset of the top-left panel
    transition_z3_learned        top right, without its inset (colourbar kept)
    transition_z3_learned_inset  the Wi_t(eta) inset of the top-right panel
    transition_z2_ninf           bottom left
    transition_z3_ninf           bottom right

Every panel is drawn by its own script's draw_panel(), exactly as
plot_transition_2x2.py does it, on an axes of the same pixel size (the top row
gets the same slot, which its colourbar takes back), so contents, fonts and
sizes are those of the 2x2.  The inset is drawn inside its panel as usual; the
main plot is then saved with the inset removed, and the inset alone by hiding
everything else and cropping to it, so it keeps its size in the 2x2.  Cropped
like the 2x2 (bbox_inches="tight").  Run it in the env the 2x2 was made in
(base anaconda, matplotlib 3.9.2).

The two main learned-model plots are NOT square, though: their frame is the
frame of the F2_Wi*/F23cut_Wi* curve plots (FreeEnergybyIntegration/figs and
examples/.../figs, 5.24 x 3.34 in), at the same place on the same fixed canvas
(CURVE_MARGIN left, bottom and top, read from the learned-model splitter), so
the plots can be laid over one another.  Their colourbar sits to the right of
the frame at its full height, 30 px wide with the 2x2's gap, and the canvas
is as wide as that takes -- these two images are wider than the curve plots.
Both insets are square: the frame side is the learned-model frame height,
the canvas is cropped to INSET_CANVAS_FONT_SIZE tick labels.  The labels are
then drawn at INSET_FONT_SIZE and the square frame shrinks to fit that same
canvas; lines and markers are scaled by INSET_MARK_SCALE.
Their mean curves are solid, with individual seeds shown as faint dots
(alpha=0.2) in the same colour, without error bars.
The two n -> infinity panels have the size of the learned-model images: the
same frame at the same place on a canvas of the same width and height, and the
same fonts (tick 14 pt, legend 15 pt).  Only if their y tick labels ("-1.0")
do not fit the learned left margin is the frame moved right by the overflow.

No axis titles anywhere -- main axes, colourbars and insets -- and no titles
(added by hand); tick labels and legends are kept.

    python plot_transition_2x2_panels.py

Writes ../figs/<name>.png / .pdf / .svg

The n -> infinity data are stored in the flow convention
Wi = beta * 4321 / 45000 and are drawn unchanged, on the Wi range of the
learned-model panels.  Use --ninf-only to regenerate just these two panels
(all three formats).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

import plot_F_grid_panels as FP
import plot_transition_2x2 as T

OUT = Path(__file__).resolve().parent.parent / "figs"
DPI = T.DPI
MARGIN_PX = dict(left=220, right=260, bottom=160, top=60)   # cropped again


def panel_axes(w_px, h_px):
    """A figure holding one axes of w_px x h_px pixels at DPI."""
    W = w_px + MARGIN_PX["left"] + MARGIN_PX["right"]
    H = h_px + MARGIN_PX["bottom"] + MARGIN_PX["top"]
    fig = plt.figure(figsize=(W / DPI, H / DPI))
    ax = fig.add_axes([MARGIN_PX["left"] / W, MARGIN_PX["bottom"] / H,
                       w_px / W, h_px / H])
    return fig, ax


NINF_HEIGHT = 1.0                   # n -> infinity frame height / learned frame height
SMALL_PANEL_FONT_SIZE = 16          # tick and legend size of the 2x2 insets, points
INSET_CANVAS_FONT_SIZE = 40         # tick size that sets the standalone inset canvas, points
INSET_FONT_SIZE = 60                # tick size drawn in the standalone square insets, points
INSET_MARK_SCALE = 3.0              # line widths and marker diameters in those insets, x
CBAR_PX = 30                        # colourbar width, as in the 2x2
CBAR_GAP_PX = 11                    # frame to colourbar, as in the 2x2 (0.015 x 749 px)


def learned_rect_panel(spec):
    """A top-row panel with the frame of the curve plots instead of a square one.

    draw_panel() puts the colourbar next to a square axes; here both are put
    back by hand: the frame at the curve plots' size and place on their fixed
    canvas, the colourbar beside it at the same height.  Returns the figure,
    the canvas height in inches and the x (inches) where the content ends."""
    style = FP.G.reference_style()
    w, h = FP.panel_size(style)                      # the curve plots' frame, inches
    pad = FP.G.read_reference_constants(FP.REF_PANELS, {"CURVE_MARGIN"})["CURVE_MARGIN"]
    cb_w, gap = CBAR_PX / DPI, CBAR_GAP_PX / DPI
    W, H = pad["left"] + w + gap + cb_w + 1.5, pad["bottom"] + h + pad["top"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([pad["left"] / W, pad["bottom"] / H, w / W, h / H])
    T.learned.draw_panel(fig, ax, spec)
    strip_titles(fig)
    cax = next(a for a in fig.axes if a is not ax)
    for a in (ax, cax):
        a.set_box_aspect(None)                       # draw_panel made them square / 20:1
    # the shorter frame would make matplotlib pick other y ticks (z3: 2.5 steps,
    # "-7.5", ...); keep the ones of the square panel in the 2x2
    sq, sq_ax, _ = learned_panel(spec)
    lo, hi = sq_ax.get_ylim()
    ax.set_yticks([t for t in sq_ax.get_yticks() if lo <= t <= hi])
    plt.close(sq)
    ax.set_position([pad["left"] / W, pad["bottom"] / H, w / W, h / H])
    cax.set_position([(pad["left"] + w + gap) / W, pad["bottom"] / H, cb_w / W, h / H])
    ax.child_axes[0].remove()                        # the inset has its own file
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    if min(tb.x0, tb.y0, H - tb.y1) < -1e-6:
        raise ValueError(f"{spec['key']}: labels overrun CURVE_MARGIN")
    return fig, H, tb.x1 + 0.1


def ninf_rect_panel(p, x_end):
    """An n -> infinity panel on the learned-model panel's canvas: the same frame
    (width, NINF_HEIGHT of its height, same bottom margin), the same canvas width
    x_end and the same fonts (tick 14 pt, legend 15 pt, as drawn by
    plot_argmax_ninf_combined.draw_panel).  Returns the figure and canvas height."""
    style = FP.G.reference_style()
    w, h = FP.panel_size(style)
    h *= NINF_HEIGHT
    pad = FP.G.read_reference_constants(FP.REF_PANELS, {"CURVE_MARGIN"})["CURVE_MARGIN"]
    H = pad["bottom"] + h + pad["top"]

    def build(left):
        fig = plt.figure(figsize=(x_end, H))
        ax = fig.add_axes([left / x_end, pad["bottom"] / H, w / x_end, h / H])
        T.theory.draw_panel(ax, p)
        strip_titles(fig)
        fig.canvas.draw()
        return fig, fig.get_tightbbox(fig.canvas.get_renderer())

    probe, tb = build(pad["left"])                   # the learned panel's frame position
    plt.close(probe)
    left = pad["left"] if tb.x0 >= 0.0 else pad["left"] - tb.x0 + 0.02
    fig, tb = build(left)
    if min(tb.x0, tb.y0, H - tb.y1, x_end - tb.x1) < -1e-6:
        raise ValueError("n -> infinity panel: labels overrun its canvas")
    return fig, H


def strip_titles(fig):
    """Hide the axis titles and titles of every axes in `fig`, insets included.
    Hidden, not blanked: an empty but visible label still pads the tight box."""
    for ax in fig.axes:
        for a in (ax, *ax.child_axes):
            for text in (a.xaxis.label, a.yaxis.label, a.title):
                text.set_visible(False)


def learned_panel(spec):
    """A top-row panel as in the 2x2; returns the figure, its axes and inset."""
    fig, ax = panel_axes(T.SLOT_PX, T.AXES_PX)
    T.learned.draw_panel(fig, ax, spec, summarize_inset=True)
    strip_titles(fig)
    (axi,) = ax.child_axes
    enlarge_small_panel_text(axi)
    return fig, ax, axi


def enlarge_small_panel_text(ax):
    """Use the same readable text size in theory panels and standalone insets."""
    ax.tick_params(axis="both", which="both", labelsize=SMALL_PANEL_FONT_SIZE)
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_fontsize(SMALL_PANEL_FONT_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(SMALL_PANEL_FONT_SIZE)


def save(fig, name, bbox="tight"):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches=bbox)
        print(f"  -> {path}", flush=True)
    plt.close(fig)


def square_inset(fig, axi, side):
    """Give the inset a square frame of `side` inches on a canvas just large
    enough for its tick labels; returns the canvas box."""
    axi.set_box_aspect(None)
    axi.set_axes_locator(None)      # detach from the parent panel's inset locator
    pad = plt.rcParams["savefig.pad_inches"]
    # Resizing can change the chosen ticks and so the label widths; measure
    # again until the margins stop changing.
    for _ in range(5):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        frame = axi.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        tight = axi.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        margins = (frame.x0 - tight.x0 + pad, tight.x1 - frame.x1 + pad,
                   frame.y0 - tight.y0 + pad, tight.y1 - frame.y1 + pad)
        left, right, bottom, top = margins
        width, height = left + side + right, bottom + side + top
        fig.set_size_inches(width, height)
        axi.set_position([left / width, bottom / height, side / width, side / height])
        fig.canvas.draw()
        tight = axi.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        if tight.x0 >= -1e-6 and tight.y0 >= -1e-6 and tight.x1 <= width + 1e-6 and tight.y1 <= height + 1e-6:
            break
    return Bbox.from_bounds(0, 0, width, height)


def refit_square_inset(fig, axi, width, height):
    """Keep the canvas (inches) and fit the largest square frame that leaves
    room for the INSET_FONT_SIZE tick labels; the frame is centred."""
    axi.tick_params(axis="both", which="both", labelsize=INSET_FONT_SIZE)
    # Pin the tick rule first: with free locators the large labels change the
    # chosen ticks, hence the label widths, at every resize.
    axi.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5], integer=True))
    lo, hi = axi.get_ylim()
    axi.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5],
                                            integer=(hi - lo) >= 2.0))
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        frame = axi.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        tight = axi.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        left, right = frame.x0 - tight.x0 + pad, tight.x1 - frame.x1 + pad
        bottom, top = frame.y0 - tight.y0 + pad, tight.y1 - frame.y1 + pad
        side = min(width - left - right, height - bottom - top)
        x0 = left + 0.5 * (width - left - right - side)
        y0 = bottom + 0.5 * (height - bottom - top - side)
        axi.set_position([x0 / width, y0 / height, side / width, side / height])
        fig.canvas.draw()
        tight = axi.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        if tight.x0 >= -1e-6 and tight.y0 >= -1e-6 and tight.x1 <= width + 1e-6 and tight.y1 <= height + 1e-6:
            break


def scale_marks(axi, factor):
    """Multiply line widths and marker diameters of the inset's data artists."""
    for line in axi.lines:
        line.set_linewidth(line.get_linewidth() * factor)
        line.set_markersize(line.get_markersize() * factor)
    for coll in axi.collections:
        coll.set_sizes(np.asarray(coll.get_sizes()) * factor ** 2)   # areas
        coll.set_linewidths(np.asarray(coll.get_linewidths()) * factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ninf-only", action="store_true",
                        help="regenerate only the two theory panels")
    args = parser.parse_args()
    widths = {}
    for spec in T.learned.PANELS:
        # the main plot, inset removed, in the curve plots' frame
        fig, H, x_end = learned_rect_panel(spec)
        widths[spec["key"]] = x_end
        if args.ninf_only:
            plt.close(fig)
            continue
        save(fig, f"transition_{spec['key']}_learned", bbox=Bbox([[0, 0], [x_end, H]]))

        # the inset alone: hide the panel's own artists and its colourbar, and
        # crop to the inset with the margin bbox_inches="tight" would leave
        fig, ax, axi = learned_panel(spec)
        for artist in ax.get_children():
            if artist is not axi:
                artist.set_visible(False)
        for other in fig.axes:
            if other is not ax:
                other.set_visible(False)
        axi.set_visible(True)
        axi.tick_params(axis="both", which="both", labelsize=INSET_CANVAS_FONT_SIZE)
        box = square_inset(fig, axi, FP.panel_size(FP.G.reference_style())[1])
        scale_marks(axi, INSET_MARK_SCALE)
        refit_square_inset(fig, axi, box.width, box.height)
        save(fig, f"transition_{spec['key']}_learned_inset", bbox=box)

    for key, p in (("z2", T.theory.load_z2()), ("z3", T.theory.load_z3())):
        fig, H = ninf_rect_panel(p, widths[key])
        save(fig, f"transition_{key}_ninf", bbox=Bbox([[0, 0], [widths[key], H]]))
        print(f"     {key}: Wi_t = {p['wi_c']:.6f}")


if __name__ == "__main__":
    main()
