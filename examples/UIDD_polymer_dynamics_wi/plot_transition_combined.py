"""The z2 and z3 transitions side by side, as sublevel-set regions.

Combines the two single-panel figures produced by
    1-trans-z2/plot_z2_transition.py   (MODE="sublevel")
    2-trans-z3/plot_z3_transition.py   (MODE="sublevel")
into one square-panelled figure for the main text.  Writes
transition_combined.png here; nothing in either subdirectory is touched.

    python plot_transition_combined.py

WHAT IS DRAWN.  For each coordinate z (= z2 or z3) the shaded region is

    S_eta(Wi) = { z : F(z) - min_z F <= eta },

with the cell shaded by HOW MANY of the 5 seed models put it in S_eta -- 5/5
darkest, the pale fringe is where the models disagree.  Both F arrays are
already min-subtracted per (seed, Wi) in their npz and are even in z to ~1e-11.
z2 uses F(z2) with z1 and z3 marginalised; z3 uses the z2 = 0 slice F23(0, z3)
with only z1 marginalised -- the same quantities their own figures plot.

The vertical marker is Wi_t, where z = 0 leaves S_eta, i.e. F(0) - min F = eta,
per seed: dashed line at the mean, shaded span over the 5 seeds.  For z3 at
eta = 1 this reproduces the `wi_1kT` column of 2-trans-z3/wi_critical_z3.csv
exactly.

Colours: z2 the default matplotlib blue #1f77b4, z3 black.

MIN_STROKE is a RENDER-only floor on ribbon thickness; S_eta itself is never
modified and no printed number depends on it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

HERE = Path(__file__).resolve().parent

# --- knobs -----------------------------------------------------------------
PANELS = [
    dict(key="z2", npz="1-trans-z2/z2_argmin_data.npz", grid="z2_grid",
         eta=0.6, base="#1f77b4", ylim=(-2.6, 2.6), min_stroke=0.018,
         ylabel=r"$Z_2$",
         cbar=r"seeds with $F(Z_2)-\min F\leq\eta$",
         eta_scan=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2,
                   1.5, 2.0]),
    dict(key="z3", npz="2-trans-z3/z3_argmin_data.npz", grid="z3_grid",
         eta=1.0, base="k", ylim=(-9.0, 9.0), min_stroke=0.05,
         ylabel=r"$Z_3$",
         cbar=r"seeds with $F_{23}(0,Z_3)-\min F_{23}\leq\eta$",
         eta_scan=[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0,
                   3.0]),
]
SHADE_RANGE = (0.22, 1.00)       # tint fractions for the 1/5 .. 5/5 levels
FIGSIZE = (13.0, 6.2)
INSET_X0, INSET_W, INSET_H = 0.14, 0.32, 0.28   # Wi_t(eta) sensitivity inset
INSET_PAD = 0.11                 # axes fractions reserved BELOW the inset box
                                 # for its tick labels and x label; the inset
                                 # is then placed so that whole footprint
                                 # clears the shaded region underneath it
INSET_TOP = 0.99
ANNOTATE_WIC = False             # in-situ arrow for <Wi_t> (superseded by
                                 # the legend below)
SHOW_LEGEND = True               # the orange span and line, lower left
LEGEND_LOC = "lower left"
ANNOTATE_REGION = False          # label the shaded region (the colorbar
                                 # already states what it is)
MARKERS = ["o", "^", "s", "D", "v"]
SEED_COLORS = ["C0", "C2", "C3", "C4", "C5"]
# tick numbers keep their size; every other piece of text is larger
FS = dict(label=20, tick=14, annot=15, inset_label=15, inset_tick=11,
          cbar=16)
FORMATS = ("png", "pdf")         # pcolormesh is rasterized, so the PDF keeps
                                 # vector text and axes at a modest file size
DPI = 150
# ---------------------------------------------------------------------------


def tint_cmap(base, n):
    """White plus n tints of `base`, for the 0..n seed-count levels."""
    rgb = np.array(to_rgb(base))
    shades = [tuple(1.0 - t * (1.0 - rgb))
              for t in np.linspace(*SHADE_RANGE, n)]
    return ListedColormap(["white"] + shades)


def thicken(mask, z, min_half):
    """Render-only: widen every run in `mask` to at least 2*min_half in z."""
    if min_half <= 0:
        return mask, 0
    h = z[1] - z[0]
    need = max(1, int(np.ceil(2 * min_half / h)))
    out, bound = mask.copy(), 0
    for k in range(mask.shape[0]):
        m = mask[k]
        if not m.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        widened = False
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= need:
                continue
            c = (a + b) // 2
            lo = max(0, c - need // 2)
            out[k, lo:min(len(m), lo + need)] = True
            widened = True
        bound += widened
    return out, bound


def origin_exit(F0, wi, eta):
    """First Wi where F(0) - min F rises above eta, linearly interpolated."""
    y = F0 - eta
    s = np.where((y[:-1] <= 0) & (y[1:] > 0))[0]
    if not len(s):
        return np.nan
    i = int(s[0])
    return float(wi[i] + (wi[i + 1] - wi[i]) * (-y[i]) / (y[i + 1] - y[i]))


def inset_rect(ax, wi, z, masks, ylim):
    """Place the inset so its box AND its labels clear the shaded region.

    The region is thin and low at small Wi, so the inset goes upper-left; the
    only thing that ever collided was the strip below the box where the tick
    labels and the x label are drawn, which `INSET_PAD` reserves.
    """
    lg0, lg1 = np.log10(wi[0]), np.log10(wi[-1])
    x0 = 10 ** (lg0 + INSET_X0 * (lg1 - lg0))
    x1 = 10 ** (lg0 + (INSET_X0 + INSET_W) * (lg1 - lg0))
    m = (wi >= x0) & (wi <= x1)
    occ = masks[:, m].any(axis=0).any(axis=0)
    zmax = np.abs(z[occ]).max() if occ.any() else 0.0
    need = (zmax - ylim[0]) / (ylim[1] - ylim[0]) + INSET_PAD
    y0 = min(max(need, 0.0), INSET_TOP - INSET_H)
    return [INSET_X0, y0, INSET_W, min(INSET_H, INSET_TOP - y0)]


def draw_panel(fig, ax, spec, *, summarize_inset=False):
    d = np.load(HERE / spec["npz"], allow_pickle=True)
    wi, z, F = d["wi_values"], d[spec["grid"]], d["F"]
    n_seed = F.shape[0]
    F0 = F[:, :, int(np.argmin(np.abs(z)))]

    masks = F <= spec["eta"]
    bound = 0
    for i in range(n_seed):
        masks[i], b = thicken(masks[i], z, spec["min_stroke"])
        bound += b

    cmap = tint_cmap(spec["base"], n_seed)
    norm = BoundaryNorm(np.arange(-0.5, n_seed + 1.5, 1), cmap.N)
    pm = ax.pcolormesh(wi, z, masks.sum(axis=0).T, cmap=cmap, norm=norm,
                       shading="nearest", zorder=0, rasterized=True)
    cb = fig.colorbar(pm, ax=ax, pad=0.015, fraction=0.040,
                      ticks=range(n_seed + 1))
    cb.set_label(spec["cbar"], fontsize=FS["cbar"])
    cb.ax.tick_params(labelsize=FS["cbar"] - 2)

    wi_c = np.array([origin_exit(F0[i], wi, spec["eta"]) for i in range(n_seed)])
    lo, hi, mu = np.nanmin(wi_c), np.nanmax(wi_c), np.nanmean(wi_c)
    sd = np.nanstd(wi_c, ddof=1)
    ax.axvspan(lo, hi, color="C1", alpha=0.16, zorder=1,
               label=rf"$[{lo:.2f},\ {hi:.2f}]$")
    ax.axvline(mu, ls="--", color="C1", lw=1.8, zorder=2,
               label=rf"$\mathrm{{Wi}}_t = {mu:.2f}$")

    if ANNOTATE_WIC:
        ax.annotate(rf"$\langle \mathrm{{Wi}}_t\rangle={mu:.2f}\pm{sd:.2f}$",
                    xy=(mu, spec["ylim"][0] * 0.62),
                    xytext=(0.62, 0.13), textcoords="axes fraction",
                    fontsize=FS["annot"], color="C1", ha="left",
                    arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))
    if ANNOTATE_REGION:
        k = int(np.argmin(np.abs(wi - wi[-1] * 0.45)))
        zt = np.nanmax(np.where(masks[:, k].any(axis=0), z, np.nan))
        ax.annotate(rf"$S_\eta$,  $\eta={spec['eta']:g}$",
                    xy=(wi[k], zt), xytext=(0.60, 0.88),
                    textcoords="axes fraction", fontsize=FS["annot"],
                    ha="left", arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_xscale("log")
    ax.set_xlim(wi[0], wi[-1])
    ax.set_ylim(*spec["ylim"])
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$\mathrm{Wi}$", fontsize=FS["label"])
    ax.set_ylabel(spec["ylabel"], fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(False)
    if SHOW_LEGEND:
        ax.legend(fontsize=FS["annot"], loc=LEGEND_LOC, framealpha=0.92)

    axi = ax.inset_axes(inset_rect(ax, wi, z, masks, spec["ylim"]))
    means = []
    for e in spec["eta_scan"]:
        w = np.array([origin_exit(F0[i], wi, e) for i in range(n_seed)])
        if summarize_inset:
            axi.scatter(np.full(n_seed, e), w, s=16, color=spec["base"],
                        alpha=0.2, edgecolors="none", zorder=2)
            means.append(np.nanmean(w))
        else:
            for i in range(n_seed):
                axi.plot(e, w[i], MARKERS[i], ms=4.0, color=SEED_COLORS[i],
                         mec="k", mew=0.3, zorder=3)
            axi.errorbar(e, np.nanmean(w), yerr=np.nanstd(w, ddof=1), fmt="_",
                         ms=10, color="k", elinewidth=1.3, capsize=2.5, zorder=4)
    if summarize_inset:
        axi.plot(spec["eta_scan"], means, "o-", color=spec["base"],
                 ms=4.0, lw=1.5, zorder=4)
    axi.axvline(spec["eta"], color="C1", ls="--", lw=1.2, alpha=0.8)
    axi.set_xlabel(r"$\eta$", fontsize=FS["inset_label"])
    axi.set_ylabel(r"$\mathrm{Wi}_t$", fontsize=FS["inset_label"])
    axi.tick_params(labelsize=FS["inset_tick"])
    axi.grid(alpha=0.3)

    return dict(key=spec["key"], eta=spec["eta"], mu=mu, sd=sd, lo=lo, hi=hi,
                bound=bound, ncol=n_seed * len(wi))


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE)
    out = [draw_panel(fig, ax, spec) for ax, spec in zip(axes, PANELS)]
    plt.tight_layout()
    for ext in FORMATS:
        path = HERE / f"transition_combined.{ext}"
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {path}", flush=True)
    plt.close()
    for r in out:
        print(f"     {r['key']}: eta = {r['eta']:g}   Wi_t = {r['mu']:.3f} "
              f"+/- {r['sd']:.3f}  (rel {r['sd'] / r['mu']:.0%}, "
              f"range [{r['lo']:.3f}, {r['hi']:.3f}])"
              f"   stroke floor {r['bound']}/{r['ncol']} cols")


if __name__ == "__main__":
    main()
