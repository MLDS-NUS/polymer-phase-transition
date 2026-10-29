"""Combined supplement figure: conditional responses and their sensitivity to analysis
choices.  No figure title and no panel titles (the caption goes in the text).

  (a) 20 V   raw f and odd g conditional responses      (the 20 V panel of Fig. S3)
  (b) 40 V   raw f and odd g conditional responses      (the 40 V panel of Fig. S3)
  (c) s_0(40 V) against s_0(20 V) over the 72 admissible analysis settings
                                                        (Fig. S6a)

Reads results/stats_curves.npz and results/stats_robust40.csv; nothing is recomputed.

    python3 fig_S_combined.py
"""
import os
_BASE = os.path.dirname(os.path.abspath(__file__))     # .../phase_trans_from_image_2_z3/code
BASE = os.path.abspath(os.path.join(_BASE, ".."))      # the project folder
FIG_SUPP = os.path.join(BASE, "figures", "supplement")
RESULTS = os.path.join(BASE, "results")
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

NAME = "fig_S_combined"

# ------------------------------------------------------------------ data
stats = np.load(os.path.join(RESULTS, "stats_curves.npz"), allow_pickle=True)
rows = [r for r in csv.DictReader(open(os.path.join(RESULTS, "stats_robust40.csv")))
        if r["40V_slope"] != "" and r["20V_slope"] != ""]
S20 = np.array([float(r["20V_slope"]) for r in rows])
S40 = np.array([float(r["40V_slope"]) for r in rows])

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
RED, GREEN = "C3", "C2"

FIG_W = 7.05                    # inches: \linewidth of the one-column supplement (510 pt)
AX_L, AX_R = 0.68, FIG_W - 0.07
LETTER_X = 0.04                 # panel letters: left edge of the figure (and the same
                                # distance to the left of the right-hand axes)
TITLE_PAD = 4                   # pt: the "20 V" / "40 V" axes titles; the panel letters sit
                                # on the same baseline


def panel_b(ax, key, title, legend=False):
    x, f, g = stats[f"{key}_mids"], stats[f"{key}_f"], stats[f"{key}_g"]
    m = np.isfinite(f)
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.axvline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.plot(x[m], f[m], "o-", ms=4.8, lw=1.5, color="0.25", label=r"raw $f$", zorder=3)
    ax.plot(x[m], g[m], "s--", ms=4.2, lw=1.4, color=RED, label=r"odd $g$", zorder=4)
    ax.set_title(title, fontsize=9.5, pad=TITLE_PAD)
    ax.set_xlabel(r"$Z_{3,c}$ (px)")
    if legend:
        ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="0.8",
                  handlelength=2.2, borderpad=0.5, labelspacing=0.35, fontsize=8.5)


def panel_c(ax):
    ax.axhline(0, color="k", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color="k", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.scatter(S20, S40, s=20, c=GREEN, edgecolors="none", alpha=0.85, zorder=3)
    ax.set_xlim(-0.197, 0.010)
    ax.set_ylim(-0.003, 0.0475)
    minus = lambda fmt: matplotlib.ticker.FuncFormatter(            # a true minus sign, as in (a)(b)
        lambda v, _: (fmt % v).replace("-", "−"))
    ax.set_xticks(np.arange(-7, 1)*0.025)              # integer multiples: 0 stays +0.000
    ax.xaxis.set_major_formatter(minus("%.3f"))
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04])
    ax.yaxis.set_major_formatter(minus("%.2f"))
    ax.set_xlabel(r"$s_0$ at 20 V")
    ax.set_ylabel(r"$s_0$ at 40 V")


def make(figh=3.90, rows_in=None, save=True):
    """rows_in: top edge / height (inches from the top of the figure) of each row of axes."""
    rows_in = rows_in or dict(b=(0.30, 1.45), c=(2.55, 0.85))
    fig = plt.figure(figsize=(FIG_W, figh))

    def axes(x0, w, key):
        top, h = rows_in[key]
        return fig.add_axes([x0/FIG_W, 1 - (top + h)/figh, w/FIG_W, h/figh])

    def letter(s, x0, key):
        """Bold panel letter on the baseline of the axes titles."""
        top, _ = rows_in[key]
        fig.text(x0/FIG_W, 1 - top/figh + TITLE_PAD/72/figh, s, fontsize=10.5,
                 fontweight="bold", va="baseline", ha="left")

    GAP = 0.95                  # inches between the two response panels
    wb = (AX_R - AX_L - GAP)/2
    x_b2 = AX_R - wb
    ax_a, ax_b = axes(AX_L, wb, "b"), axes(x_b2, wb, "b")
    panel_b(ax_a, "20V2", "20 V", legend=True)
    panel_b(ax_b, "40V", "40 V")
    ax_a.set_ylabel("conditional response (px)", fontsize=8)
    ax_c = axes(AX_L, AX_R - AX_L, "c")
    panel_c(ax_c)
    # letters in the outer left margin of each panel, level with the axes titles
    letter("(a)", LETTER_X, "b")
    letter("(b)", x_b2 - (AX_L - LETTER_X), "b")
    letter("(c)", LETTER_X, "c")
    if save:
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG_SUPP, f"{NAME}.{ext}"), dpi=300)
        print(f"wrote {NAME}.png / .pdf")
    return fig


if __name__ == "__main__":
    make()
