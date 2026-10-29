"""The four surface-inset panels of the two transitions in one 2x2 figure.

    top row    : ../1-trans-z2/F_z2_z3_surface_seed<seed>.png   F_2(z2)
    bottom row : F_z3_slice_surface_seed<seed>.png              F23(z2=0, z3)

The panels are drawn by those two scripts' own draw_panels(), from their saved
data (computed first if missing), so a style change made there shows up here
too.  All four panels share one y range, and one z1 integration window and
quadrature (checked below); the top row also integrates z3 over [-15, 15].

    python plot_surface_grid.py                  # seed 1 by default
    MODEL_IDX=0 python plot_surface_grid.py      # another seed (0 = seed 123)

Writes
    F_z2_z3_grid_seed<seed>.png / .pdf / .svg
The pdf and svg keep curves, axes and text as vectors; the inset heatmaps are
embedded as images at VECTOR_DPI.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# embed TrueType fonts in the pdf (the default Type 3 fonts are often refused
# by journal submission checks)
matplotlib.rcParams["pdf.fonttype"] = 42

FIGSIZE = (15, 10.5)                     # a little wider than tall
SPACING = dict(hspace=0.42, wspace=0.22)
PNG_DPI = 150
VECTOR_DPI = 300                         # resolution of the heatmaps in pdf/svg


def load_script(path):
    """Import a plot script by file name (its folder has a hyphen in it)."""
    sys.path.insert(0, str(path.parent))         # for its compute_*_curves
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    z2 = load_script(HERE.parent / "1-trans-z2" / "plot_z2_z3_surface.py")
    z3 = load_script(HERE / "plot_z3_surface.py")
    # one z1 window for all four panels, so the figure states a single one
    assert (z2.Z1_MIN, z2.Z1_MAX, z2.N_Z1) == (z3.Z1_MIN, z3.Z1_MAX, z3.N_Z1), \
        "the two rows integrate z1 over different windows or quadratures"
    mi = int(os.environ.get("MODEL_IDX", "1"))
    seed, d2 = z2.get_data(mi)
    _, d3 = z3.get_data(mi)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.subplots_adjust(**SPACING)
    items2, top2 = z2.draw_panels(fig, axes[0], d2)
    items3, top3 = z3.draw_panels(fig, axes[1], d3)
    z3.clear_insets(fig, items2 + items3, z3.Y_LO, max(top2, top3))

    for ext, dpi in (("png", PNG_DPI), ("pdf", VECTOR_DPI), ("svg", VECTOR_DPI)):
        out = HERE / f"F_z2_z3_grid_seed{seed}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  -> {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
