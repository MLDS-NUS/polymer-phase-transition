"""fig_exp_normalform.{png,pdf,svg} — experimental drift curves for four voltage
groups, with trajectory-level bootstrap error bars.

The measured drift points are the result.  The cubic (pitchfork normal form) is drawn
lighter and pale: it is only a shape summary and must not outweigh the data.  The
apparent potential obtained by integrating that cubic is a small low-contrast inset —
a visual aid, not evidence.

No axis labels and no panel titles (voltages): they are added by hand.  Tick
labels are kept.

Also writes the 20 V and 40 V insets on their own, transparent background and no
frame, at the size they have in the main figure:
fig_exp_normalform_inset_{20V,40V}.{png,pdf,svg}.
The standalone 40 V curve is drawn over |Z| <= 1.5 z* instead of the data range
(1.21 z*), so that its outer walls rise above the central barrier (a W); beyond
1.21 z* it extrapolates the cubic fit.

Reads stats_curves.npz (written by analysis_stats.py); no recomputation here.

    python3 fig_exp_normalform.py
"""
import os, sys
_BASE = os.path.dirname(os.path.abspath(__file__))     # .../phase_trans_from_image_2_z3/code
BASE = os.path.abspath(os.path.join(_BASE, ".."))      # the project folder
REPO = os.path.abspath(os.path.join(BASE, ".."))       # the repository root
FIG_MAIN = os.path.join(BASE, "figures", "main")
FIG_SUPP = os.path.join(BASE, "figures", "supplement")
FIG_PANE = os.path.join(BASE, "figures", "panels")
RESULTS = os.path.join(BASE, "results")
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(REPO, "phase_trans_from_image"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# (key, name, x-tick step in px).  The steps are the ones the auto locator chose
# while the panels still had axis labels; pinned so that dropping the labels,
# which widens the axes, does not move the ticks.
GROUPS = [("10V", "10V", 10), ("20V2", "20V", 20), ("40V", "40V", 25), ("50V", "50V", 25)]
INSETS = ("20V", "40V")                  # also saved as standalone figures
W_ARMS = {"40V": 1.5}                    # standalone x half-range in units of z*

TAG = os.environ.get("TAG", "")
d = np.load(os.path.join(RESULTS, f"stats_curves{TAG}.npz"), allow_pickle=True)

# The 18 in figure is printed one PRL column wide (3.4 in, ~0.19x), so every line
# width here is about 5x its printed width.
plt.rcParams.update({
    "axes.linewidth": 3.0,
    "xtick.major.width": 3.0, "ytick.major.width": 3.0,
    "xtick.major.size": 9.0, "ytick.major.size": 9.0,
})



def draw_potential(axp, xs, U, col, frame=True):
    """the apparent-potential inset: curve, optionally in a pale frame; no ticks,
    no background"""
    axp.plot(xs, U, "-", lw=5.4, color=col, alpha=0.80)
    uspan = U.max() - U.min()
    axp.set_ylim(U.min() - 0.12*uspan, U.max() + 0.20*uspan)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_visible(frame); sp.set_color("0.6"); sp.set_linewidth(2.7)
    axp.patch.set_alpha(0)


fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.6))
spread, fits, insets = [], [], {}

for col_i, (key, title, xstep) in enumerate(GROUPS):
    x = d[f"{key}_mids"]
    g = d[f"{key}_g"]
    lo = d[f"{key}_g_lo"]
    hi = d[f"{key}_g_hi"]
    sc = float(d[f"{key}_sc"])
    slope = float(d[f"{key}_slope"])
    col = "C3" if slope > 0 else "C0"
    m = np.isfinite(g)
    xf, gf = x[m], g[m]

    # odd cubic fit:  g = c1*x + c3*x^3   (no intercept, no even terms)
    c1, c3 = np.linalg.lstsq(np.vstack([xf, xf**3]).T, gf, rcond=None)[0]
    xs = np.linspace(xf.min(), xf.max(), 400)
    g_fit = c1*xs + c3*xs**3
    U_fit = -(c1*xs**2/2 + c3*xs**4/4)
    U_fit -= U_fit.min()
    zstar = np.sqrt(-c1/c3) if (c1 > 0 and c3 < 0) else np.nan
    fits.append((title, c1, c3, zstar))

    ax = axes[col_i]
    ax.axhline(0, color="0.75", lw=3.2, zorder=0)     # g = 0 reference
    ax.axvline(0, color="0.75", lw=3.2, zorder=0)     # Z_{3,c} = 0 reference

    # shape summary: pale, so it does not outweigh the data
    ax.plot(xs, g_fit, "-", lw=5.0, color=col, alpha=0.40, zorder=2)

    # the data: trajectory-level bootstrap, 16-84 %
    ax.errorbar(xf, gf, yerr=[gf - lo[m], hi[m] - gf], fmt="o", ms=15,
                color=col, ecolor=col, elinewidth=6.0,
                capsize=9, capthick=6.0, zorder=4)
    spread.append(gf)

    # measured zero crossings of the odd part (both sides), from the data alone
    idx = np.where((x > 0) & np.isfinite(g))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if g[a] == 0 or np.sign(g[a]) == np.sign(g[b]):
            continue
        if g[a] > 0:                       # downward crossing = new stable state
            t = g[a]/(g[a] - g[b])
            zs = x[a] + t*(x[b] - x[a])
            for xx in (zs, -zs):
                ax.axvline(xx, color="k", lw=3.2, ls=(0, (4, 3)), alpha=0.55,
                           zorder=1)
                ax.plot(xx, 0, "v", ms=24, color="k", mfc="k",
                        mec="w", mew=2.6, zorder=6)
        break

    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.tick_params(labelsize=22)

    # apparent potential: small, low contrast, purely indicative
    # (axes fractions; sized to clear the 50 V point at (0.77, 0.60) on the left
    # placement and the 20 V point at (0.23, 0.76) on the right one)
    axp = ax.inset_axes([0.02, 0.60, 0.70, 0.38] if slope > 0
                        else [0.28, 0.60, 0.70, 0.38])
    draw_potential(axp, xs, U_fit, col)
    insets[title] = (axp, xs, U_fit, col, c1, c3, zstar)

dm = np.nanmax(np.abs(np.concatenate(spread)))
for j in range(4):
    axes[j].set_ylim(-dm*1.18, dm*2.15)

fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(FIG_MAIN, f"fig_exp_normalform{TAG}.{ext}"), dpi=160)
    print(f"wrote fig_exp_normalform{TAG}.{ext}")

# standalone insets, frameless: same physical size as in the main figure (so line
# widths match), plus the margin the frame had there
for title in INSETS:
    axp, xs, U_fit, col, c1, c3, zstar = insets[title]
    bb = axp.get_window_extent()
    w, h = bb.width/fig.dpi, bb.height/fig.dpi
    mg = 2.7/72                                          # inches
    fi = plt.figure(figsize=(w + 2*mg, h + 2*mg))
    fi.patch.set_alpha(0)
    ai = fi.add_axes([mg/(w + 2*mg), mg/(h + 2*mg), w/(w + 2*mg), h/(h + 2*mg)])
    if title in W_ARMS:                                  # widen to show the outer walls
        xs = np.linspace(-1, 1, 400)*W_ARMS[title]*zstar
        U_fit = -(c1*xs**2/2 + c3*xs**4/4)
        U_fit -= U_fit.min()
        draw_potential(ai, xs, U_fit, col, frame=False)
    else:
        draw_potential(ai, xs, U_fit, col, frame=False)
        ai.set_xlim(axp.get_xlim())
    for ext in ("png", "pdf", "svg"):
        name = f"fig_exp_normalform{TAG}_inset_{title}.{ext}"
        fi.savefig(os.path.join(FIG_MAIN, name), dpi=300, transparent=True)
        print(f"wrote {name}  ({w + 2*mg:.2f} x {h + 2*mg:.2f} in)")
    plt.close(fi)

print(f"\n{'group':>6} {'c1':>11} {'c3':>13} {'z*_fit':>9}   shape")
for t, c1, c3, zs in fits:
    shape = ("double well" if np.isfinite(zs)
             else ("single well" if c1 < 0 else "inverted"))
    print(f"{t:>6} {c1:+11.5f} {c3:+13.3e} "
          f"{zs if np.isfinite(zs) else float('nan'):9.1f}   {shape}")
