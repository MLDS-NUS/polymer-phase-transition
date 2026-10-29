"""2x2 figure: learned-model sublevel sets (top) over n -> infinity theory (bottom).

    top     examples/UIDD_polymer_dynamics_wi/plot_transition_combined.py
    bottom  plot_argmax_ninf_combined.py   (this directory)

Each panel is drawn by its own script's draw_panel, so contents, fonts and
colours are exactly those of the two single-row figures.  Columns are z2 (left)
and z3 (right), and panels are placed in pixels so both rows share the same
square side and left edges.  Top-row axes are given a slot 1/(1-0.040-0.015)
wider than the panel, which is what fig.colorbar(pad=0.015, fraction=0.040)
takes back, so the top row keeps the reference's 708 px panel and
30 x 600 px colorbar.  Saved with bbox_inches="tight", like the reference.

    python plot_transition_2x2.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "examples" / "UIDD_polymer_dynamics_wi"))
import plot_transition_combined as learned      # noqa: E402
import plot_argmax_ninf_combined as theory      # noqa: E402

DPI = 150
AXES_PX = 708                       # panel side, as in both source figures
SLOT_PX = AXES_PX / (1 - 0.040 - 0.015)   # top row: panel + colorbar
COL_X0_PX = (160, 1123)             # left edges; 963 px apart as in reference
ROW_Y0_PX = (958, 120)              # bottom edges of the top and bottom rows
CANVAS_PX = (2000, 1700)            # cropped by bbox_inches="tight"
FORMATS = ("png", "pdf")


def add_axes(fig, x0, y0, w, h):
    W, H = CANVAS_PX
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def main():
    W, H = CANVAS_PX
    fig = plt.figure(figsize=(W / DPI, H / DPI))
    top_y0, bottom_y0 = ROW_Y0_PX
    top = [learned.draw_panel(fig, add_axes(fig, x0, top_y0, SLOT_PX, AXES_PX), spec)
           for x0, spec in zip(COL_X0_PX, learned.PANELS)]
    bottom = (theory.load_z2(), theory.load_z3())
    for x0, p in zip(COL_X0_PX, bottom):
        theory.draw_panel(add_axes(fig, x0, bottom_y0, AXES_PX, AXES_PX), p)
    for ext in FORMATS:
        path = HERE / f"transition_2x2.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {path}", flush=True)
    plt.close(fig)
    for r in top:
        print(f"     learned {r['key']}: Wi_t = {r['mu']:.3f} "
              f"(range [{r['lo']:.3f}, {r['hi']:.3f}])")
    for key, p in zip(("z2", "z3"), bottom):
        print(f"     n->inf  {key}: Wi_t = {p['wi_c']:.6f}")


if __name__ == "__main__":
    main()
