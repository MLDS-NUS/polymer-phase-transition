"""Stage 2: figures that give Z2 and Z3 a physical meaning.

Needs cache/observables_stride<STRIDE>.npz from extract_observables.py.

    python plot_interpret.py             # all figures
    python plot_interpret.py scatter     # z2_z3_vs_physical.png  (notebook analogue)
    python plot_interpret.py wi          # z2_z3_vs_physical_by_wi.png
    python plot_interpret.py corr        # z2_z3_corr_vs_wi.png
    python plot_interpret.py loadings    # pca_loadings.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from extract_observables import CACHE, N_BEADS, load_pca

HERE = Path(__file__).resolve().parent
STRIDE = int(os.environ.get("STRIDE", "50"))
N_SCATTER = int(os.environ.get("N_SCATTER", "60000"))   # frames drawn per scatter panel
SEED = 0

# sequential blue ramp (steps 250 -> 700): Wi as a magnitude, lightest step still
# visible on white
WI_CMAP = LinearSegmentedColormap.from_list(
    "wi_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
WI_NORM = LogNorm(vmin=1e-2, vmax=10.8)          # Wi < 0.01 clipped to the lightest step
C_Z2 = "#2a78d6"   # categorical slot 1
C_Z3 = "#eb6834"   # categorical slot 2
C_REF = "#52514e"  # secondary ink

WI_RANGES = [(0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 11.0)]   # small-multiple columns
WI_EDGES = np.geomspace(0.01, 10.81, 16)                       # bins of the r(Wi) curves
ETE = r"x_1 - x_N"
FOLD = r"x_1 + x_N - 2x_{\rm com}"


def load():
    d = dict(np.load(os.environ.get("OBS_FILE", CACHE / f"observables_stride{STRIDE}.npz")))
    d["Wi_frame"] = np.broadcast_to(d["Wi"][:, None], d["Z2"].shape)
    return d


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def fig_scatter(d):
    """|Z2| vs end-to-end distance, |Z3| vs foldedness, all Wi pooled."""
    rng = np.random.default_rng(SEED)
    pick = rng.choice(d["Z2"].size, size=min(N_SCATTER, d["Z2"].size), replace=False)
    wi = d["Wi_frame"].ravel()[pick]
    pairs = [
        ("Z2", "ete_x", r"$|Z_2|$", rf"end-to-end distance $|{ETE}|$"),
        ("Z3", "fold_x", r"$|Z_3|$", rf"foldedness $|{FOLD}|$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (zk, ok, xl, yl) in zip(axes, pairs):
        z, o = np.abs(d[zk]), np.abs(d[ok])
        sc = ax.scatter(z.ravel()[pick], o.ravel()[pick], c=wi, cmap=WI_CMAP, norm=WI_NORM,
                        s=3, lw=0, alpha=0.6, rasterized=True)
        ax.text(0.04, 0.96, f"$r = {pearson(z, o):.3f}$", transform=ax.transAxes,
                va="top", fontsize=12)
        ax.set_xlabel(xl, fontsize=13)
        ax.set_ylabel(yl, fontsize=12)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    cb = fig.colorbar(sc, ax=axes, pad=0.01, aspect=30)
    cb.set_label(r"$Wi$", fontsize=13)
    cb.solids.set_alpha(1)
    n_traj, n_fr = d["Z2"].shape
    fig.suptitle(f"{n_traj} trajectories x {n_fr} frames (every {STRIDE}th of 5001); "
                 f"{N_SCATTER} of them drawn; $r$ = Pearson correlation over all of them",
                 fontsize=10, color=C_REF)
    out = HERE / "z2_z3_vs_physical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fit_through_origin(z, o):
    """c minimising |o - c z|^2."""
    return float((z * o).sum() / (z * z).sum())


def fig_by_wi(d):
    """Signed Z2 vs r_0 - r_N and Z3 vs foldedness, one column per Wi range."""
    rng = np.random.default_rng(SEED)
    ete_label = r"r_{0,x} - r_{N,x}"
    fold_label = r"(r_{0,x}+r_{N,x})/2 - r_{\rm com,x}"
    pairs = [
        ("Z2", "ete_x", "Z_2", ete_label, C_Z2, 1.0),
        ("Z3", "fold_x", "Z_3", fold_label, C_Z3, 0.5),
    ]
    fig, axes = plt.subplots(2, len(WI_RANGES), figsize=(7 * len(WI_RANGES), 15),
                             layout="constrained", sharex="row", sharey="row")
    for row, (zk, ok, zs, os_, color, scale) in enumerate(pairs):
        o_all = d[ok] * scale
        c_all = fit_through_origin(d[zk], o_all)
        for col, (lo, hi) in enumerate(WI_RANGES):
            ax = axes[row, col]
            last = col == len(WI_RANGES) - 1
            m = (d["Wi"] >= lo) & ((d["Wi"] <= hi) if last else (d["Wi"] < hi))
            z, o = d[zk][m], o_all[m]
            pick = rng.choice(z.size, size=min(N_SCATTER // 4, z.size), replace=False)
            ax.scatter(z.ravel()[pick], o.ravel()[pick], s=15, lw=0, alpha=0.35,
                       color=color, rasterized=True)
            zz = np.array([z.min(), z.max()])
            ax.plot(zz, c_all * zz, color=C_REF, lw=1.2, ls="--")
            # both relations are increasing: upper-left and lower-right stay empty
            ax.text(0.04, 0.96, f"$r = {pearson(z, o):.3f}$",
                    transform=ax.transAxes, va="top", fontsize=52)
            if row == 0:
                rel = r"\leq" if last else "<"
                ax.set_title(rf"${lo:g} \leq Wi {rel} {min(hi, float(d['Wi'].max())):.3g}$",
                             fontsize=48)
            ax.set_xlabel(f"${zs}$", fontsize=48)
            ax.tick_params(labelsize=40)
        axes[row, 0].set_ylabel(f"${os_}$", fontsize=48, labelpad=15)
    for ext in ("png", "pdf", "svg"):
        out = HERE / f"z2_z3_vs_physical_by_wi.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


def wi_bin_stats(d, frames):
    """Per Wi bin: r and fitted constant, on the given frame slice."""
    idx = np.clip(np.digitize(d["Wi"], WI_EDGES) - 1, 0, len(WI_EDGES) - 2)
    stats = {k: np.full(len(WI_EDGES) - 1, np.nan) for k in ("r2", "r3", "c2", "c3", "n")}
    for b in range(len(WI_EDGES) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        z2, z3 = d["Z2"][m][:, frames], d["Z3"][m][:, frames]
        e, f = d["ete_x"][m][:, frames], d["fold_x"][m][:, frames]
        stats["r2"][b], stats["r3"][b] = pearson(z2, e), pearson(z3, f)
        stats["c2"][b], stats["c3"][b] = fit_through_origin(z2, e), fit_through_origin(z3, f)
        stats["n"][b] = m.sum()
    return stats


def fig_corr_vs_wi(d):
    """How well the two identifications hold, bin by bin in Wi."""
    n_fr = d["Z2"].shape[1]
    t = d["t"][0]
    late = t >= 0.5 * t[-1]
    wi_c = np.sqrt(WI_EDGES[:-1] * WI_EDGES[1:])
    s_all = wi_bin_stats(d, slice(None))
    s_late = wi_bin_stats(d, late)
    c2_all = fit_through_origin(d["Z2"], d["ete_x"])
    c3_all = fit_through_origin(d["Z3"], d["fold_x"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    for key, color, lab in (("r2", C_Z2, rf"$Z_2$ vs ${ETE}$"), ("r3", C_Z3, rf"$Z_3$ vs ${FOLD}$")):
        ax.plot(wi_c, s_all[key], "-o", color=color, lw=2, ms=6, label=lab + " (all frames)")
        ax.plot(wi_c, s_late[key], "--s", color=color, lw=1.5, ms=5, mfc="white",
                label=lab + rf" (second half, $t \geq {0.5 * t[-1]:.2f}$)")
    ax.set_ylabel(r"Pearson $r$", fontsize=12)
    ax.set_ylim(min(0.5, np.nanmin(s_late["r3"]) - 0.05), 1.01)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("(a) correlation", fontsize=12, loc="left")

    ax = axes[1]
    ax.axhline(1, color=C_REF, lw=0.8)
    ax.plot(wi_c, s_all["c2"] / c2_all, "-o", color=C_Z2, lw=2, ms=6,
            label=rf"$c_2(Wi)/c_2$,  ${ETE} \approx c_2 Z_2$,  $c_2 = {c2_all:.1f}$")
    ax.plot(wi_c, s_all["c3"] / c3_all, "-o", color=C_Z3, lw=2, ms=6,
            label=rf"$c_3(Wi)/c_3$,  ${FOLD} \approx c_3 Z_3$,  $c_3 = {c3_all:.1f}$")
    ax.set_ylabel("proportionality constant / all-Wi value", fontsize=12)
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(b) conversion factor (least squares through 0)", fontsize=12, loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$Wi$  (bin centre)", fontsize=12)
    fig.suptitle(f"{int(np.nansum(s_all['n']))} trajectories in {len(wi_c)} log-spaced Wi bins "
                 f"({WI_EDGES[0]:g} to {WI_EDGES[-1]:g}; Wi < {WI_EDGES[0]:g} pooled into the "
                 f"first bin), {n_fr} frames each (every {STRIDE}th of 5001)",
                 fontsize=10, color=C_REF)
    out = HERE / "z2_z3_corr_vs_wi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    for b in range(len(wi_c)):
        print(f"  Wi~{wi_c[b]:6.3f} n={s_all['n'][b]:5.0f}  r2={s_all['r2'][b]:.3f}/{s_late['r2'][b]:.3f}"
              f"  r3={s_all['r3'][b]:.3f}/{s_late['r3'][b]:.3f}"
              f"  c2={s_all['c2'][b]:7.1f}  c3={s_all['c3'][b]:5.1f}")


def fig_loadings():
    """x-components of the two PCA directions against the first Rouse modes."""
    P, _ = load_pca()
    i = np.arange(1, N_BEADS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", sharey=True)
    for ax, row, p, color, name in ((axes[0], 0, 1, C_Z2, "Z_2"), (axes[1], 1, 2, C_Z3, "Z_3")):
        v = P[row, 0::3] / np.linalg.norm(P[row, 0::3])
        rouse = np.sqrt(2 / N_BEADS) * np.cos(p * np.pi * (i - 0.5) / N_BEADS)
        rouse *= np.sign(v @ rouse)
        ax.axhline(0, color=C_REF, lw=0.6)
        ax.plot(i, rouse, color=C_REF, lw=1.5, ls="--", label=f"Rouse mode {p}")
        ax.plot(i, v, color=color, lw=2, label=rf"$v_{{{row + 1}}}$  (direction of ${name}$)")
        ax.text(0.04, 0.05, rf"overlap $= {abs(v @ rouse):.3f}$", transform=ax.transAxes,
                fontsize=11)
        ax.set_xlabel(r"bead index $i$", fontsize=12)
        ax.set_xlim(1, N_BEADS)
        ax.legend(fontsize=10, loc="upper center", frameon=False)
    axes[0].set_ylabel(r"weight on $x_i$", fontsize=12)
    out = HERE / "pca_loadings.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main(which):
    if "loadings" in which:
        fig_loadings()
    if {"scatter", "wi", "corr"} & set(which):
        d = load()
        if "scatter" in which:
            fig_scatter(d)
        if "wi" in which:
            fig_by_wi(d)
        if "corr" in which:
            fig_corr_vs_wi(d)


if __name__ == "__main__":
    args = sys.argv[1:] or ["scatter", "wi", "corr", "loadings"]
    unknown = set(args) - {"scatter", "wi", "corr", "loadings"}
    if unknown:
        sys.exit(__doc__)
    main(args)
