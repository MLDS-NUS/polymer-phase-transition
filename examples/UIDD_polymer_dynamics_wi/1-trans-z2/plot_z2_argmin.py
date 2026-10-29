"""Stage 2: z2*(Wi) bifurcation diagrams from the cached F(z2) curves.

Layout follows FreeEnergybyIntegration/MC/results_z2/mc_n8_beta5_12_argmax.png:
the minimiser is tracked on the z2 >= 0 and z2 <= 0 branches (filled / open
markers) so the symmetry breaking is visible, with the same quantity plotted
against two x-axes -- here linear Wi and log Wi, since the transition sits near
Wi ~ 0.3 while the sweep runs to Wi = 10.

One difference from the MC reference matters.  There the symmetry breaking is a
continuous pitchfork.  Here F(z2) keeps a local minimum at z2 = 0 with a genuine
barrier (~0.5 in free energy units, converged under 8x quadrature refinement)
separating it from an outer minimum, so the *global* minimiser jumps
discontinuously: the transition is first-order-like, with a coexistence range.
Both branches are therefore tracked, and Wi_c is defined by the crossing
F(z2_outer) = F(0) rather than by a pitchfork fit.

    python plot_z2_argmin.py

Reads cache/curves_m*.npz (written by compute_z2_curves.py) and writes, in this
directory:
    z2_argmin_seed<seed>.png    one per model
    z2_argmin_combined.png      all 5 overlaid
    z2_argmin_ensemble.png      ensemble statistics, Wi_c, collapse
    z2_argmin_data.npz/.csv     the plotted numbers
    wi_critical.csv             Wi_c / spinodal per model
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE.parent))
from utils_polymer import local_extrema_1d  # noqa: E402

# F here comes from an analytic quadrature, not from samples, so no noise
# threshold is needed when calling the notebook's extrema finder.
TAU = 0.0


def parabolic_refine(F, z2, j):
    """Sub-grid minimum position from the parabola through j-1, j, j+1."""
    if j <= 0 or j >= len(z2) - 1:
        return z2[j]
    a, b, c = F[j - 1], F[j], F[j + 1]
    denom = a - 2.0 * b + c
    if denom <= 0:
        return z2[j]
    return z2[j] + 0.5 * (a - c) / denom * (z2[1] - z2[0])


def analyse_curve(F, z2):
    """Local minima of F(z2) on the z2 >= 0 side.

    Returns (origin_is_min, z2_outer, F_outer_minus_F0, barrier), where
    `z2_outer` is the deepest local minimum at z2 > 0 (nan if there is none)
    and `barrier` is the height of the ridge between z2 = 0 and it.
    """
    i0 = int(np.argmin(np.abs(z2)))
    _, min_idx = local_extrema_1d(F, x=z2, tau=TAU)
    origin_is_min = bool(np.any(np.abs(min_idx - i0) <= 1))
    outer = [j for j in min_idx if z2[j] > z2[i0] + 1e-12]
    if not outer:
        return origin_is_min, np.nan, np.nan, np.nan
    j = int(min(outer, key=lambda k: F[k]))
    barrier = float(F[i0:j + 1].max() - F[i0])
    return origin_is_min, parabolic_refine(F, z2, j), float(F[j] - F[i0]), barrier


def zero_crossing(x, y):
    """First x where y crosses from >0 to <=0, by linear interpolation."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    s = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if len(s) == 0:
        return np.nan
    i = int(s[0])
    return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))


def load_all():
    files = sorted(CACHE.glob("curves_m*.npz"),
                   key=lambda p: int(p.stem.split("m")[-1]))
    if not files:
        raise SystemExit(f"no cache in {CACHE}; run compute_z2_curves.py first")
    models = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        F, z2, wi = d["F"], d["z2_grid"], d["wi_values"]
        n = len(wi)
        i0 = int(np.argmin(np.abs(z2)))

        origin_min = np.zeros(n, dtype=bool)
        z2_outer = np.full(n, np.nan)
        dF = np.full(n, np.nan)          # F(z2_outer) - F(0)
        barrier = np.full(n, np.nan)
        for k in range(n):
            origin_min[k], z2_outer[k], dF[k], barrier[k] = analyse_curve(F[k], z2)

        # global minimiser: 0 while the origin still wins, the outer well after
        z2_star = np.where(np.isfinite(dF) & (dF < 0), z2_outer, 0.0)
        wi_c = zero_crossing(wi, dF)
        # jump in the minimiser at the transition (first-order order parameter)
        after = np.where(np.isfinite(wi_c) & (wi >= wi_c))[0]
        jump = float(z2_star[after[0]]) if len(after) else np.nan
        # first Wi at which any second local minimum is resolved. Below the
        # transition this is often a shallow wiggle at z2 ~ 0.1 rather than the
        # stretched branch, so it is recorded but not used as a spinodal.
        has_outer = np.isfinite(z2_outer)
        wi_2nd = float(wi[np.argmax(has_outer)]) if has_outer.any() else np.nan

        chk = d["check_idx"]
        dz_quad = max(
            abs(analyse_curve(d["F_fine"][c], z2)[1] - z2_outer[k])
            for c, k in enumerate(chk)
            if np.isfinite(z2_outer[k]))

        models.append(dict(
            idx=int(f.stem.split("m")[-1]), seed=int(d["seed"]),
            path=str(d["model_path"]), F=F, z2=z2, wi=wi,
            z2_star=z2_star, z2_outer=z2_outer, dF=dF, barrier=barrier,
            origin_min=origin_min, F0=F[:, i0],
            wi_c=wi_c, wi_2nd=wi_2nd, jump=jump,
            asym=float(np.abs(F - F[:, ::-1]).max()),
            quad=float(np.abs(F[chk] - d["F_fine"]).max()),
            quad_z2=float(dz_quad),
        ))
    return models


def _draw_branches(ax, m, color, marker, label, size=45, log=False,
                   metastable=True):
    """Global branch as filled/open markers; metastable branches in grey."""
    wi, z_out, dF = m["wi"], m["z2_outer"], m["dF"]

    if metastable:
        # the higher of the two wells: the outer one below Wi_c, the z2=0 one
        # above it -- i.e. the coexistence range of a first-order jump
        meta = np.isfinite(dF) & (dF >= 0)
        ax.plot(wi[meta], z_out[meta], ".", ms=4, color="0.55", zorder=2,
                label="secondary (metastable) local min")
        ax.plot(wi[meta], -z_out[meta], ".", ms=4, color="0.55", zorder=2)
        after = np.isfinite(dF) & (dF < 0) & m["origin_min"]
        ax.plot(wi[after], np.zeros(after.sum()), ".", ms=4, color="0.55",
                zorder=2)

    z = m["z2_star"]
    ax.scatter(wi, z, s=size, color=color, marker=marker, label=label, zorder=3)
    nz = z > 0
    ax.scatter(wi[nz], -z[nz], s=size, marker=marker, facecolors="none",
               edgecolors=color, zorder=3)
    # connect within each branch only -- never across the discontinuity
    for seg, sign in [(~nz, +1), (nz, +1), (nz, -1)]:
        if seg.sum() > 1:
            ax.plot(wi[seg], sign * z[seg], "-", color=color, lw=1.0, alpha=0.5)
    ax.axhline(0, ls=":", color="k", lw=0.8)
    if log:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)


def per_model_plot(m):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, log in [(axes[0], False), (axes[1], True)]:
        _draw_branches(ax, m, "C0", "o", f"seed {m['seed']} (global min)",
                       log=log)
        ax.axvline(m["wi_c"], ls="--", color="0.35", lw=1.1,
                   label=rf"$Wi_c={m['wi_c']:.3f}$")
        ax.set_xlabel(r"$Wi$" + (" (log)" if log else ""), fontsize=13)
        ax.set_ylabel(r"$z_2^{*} = \arg\min_{z_2}\,F(z_2)$", fontsize=12)
    axes[0].set_title(r"$z_2^{*}$ vs $Wi$", fontsize=13)
    axes[1].set_title(r"$z_2^{*}$ vs $Wi$ (log axis)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(rf"model {m['idx'] + 1} (seed {m['seed']}): minimiser of "
                 rf"$F(z_2)=-\log\iint e^{{-V(z_1,z_2,z_3;Wi)}}\,dz_1dz_3$",
                 fontsize=14)
    plt.tight_layout()
    out = HERE / f"z2_argmin_seed{m['seed']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


COLORS = ["C0", "C2", "C3", "C4", "C5"]
MARKERS = ["o", "^", "s", "D", "v"]


def combined_plot(models):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for i, m in enumerate(models):
        for ax, log in [(axes[0], False), (axes[1], True)]:
            _draw_branches(ax, m, COLORS[i], MARKERS[i], f"seed {m['seed']}",
                           size=26, log=log, metastable=False)
    for ax, log in [(axes[0], False), (axes[1], True)]:
        ax.set_xlabel(r"$Wi$" + (" (log)" if log else ""), fontsize=13)
        ax.set_ylabel(r"$z_2^{*}$", fontsize=12)
        ax.legend(fontsize=9, loc="center right", ncol=2)
    axes[0].set_title(r"$z_2^{*}$ vs $Wi$", fontsize=13)
    axes[1].set_title(r"$z_2^{*}$ vs $Wi$ (log axis)", fontsize=13)
    fig.suptitle(r"$z_2$ minimiser for the 5 seed-ensemble models "
                 r"(identical data and architecture, different init)",
                 fontsize=14)
    plt.tight_layout()
    out = HERE / "z2_argmin_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


def ensemble_plot(models):
    """Merge the 5 curves: spread as an error bar, Wi_c, and a collapse test."""
    wi = models[0]["wi"]
    Z = np.array([m["z2_star"] for m in models])           # [model, wi]
    mean, std = Z.mean(axis=0), Z.std(axis=0, ddof=1)
    wi_c = np.array([m["wi_c"] for m in models])
    jump = np.array([m["jump"] for m in models])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) ensemble mean +- std. F is exactly even in z2, so above Wi_c there are
    #     TWO degenerate global minima at +-z2*; both branches are drawn, the
    #     negative one being the exact mirror rather than a separate measurement.
    ax = axes[0, 0]
    for i, m in enumerate(models):
        ax.plot(wi, m["z2_star"], "-", color=COLORS[i], lw=0.9, alpha=0.55,
                label=f"seed {m['seed']}")
        ax.plot(wi, -m["z2_star"], "-", color=COLORS[i], lw=0.9, alpha=0.55)
    for sign, lab in [(+1, "ensemble mean"), (-1, None)]:
        ax.plot(wi, sign * mean, "-", color="k", lw=2.0, label=lab)
        ax.fill_between(wi, sign * (mean - std), sign * (mean + std),
                        color="k", alpha=0.18,
                        label=r"$\pm 1\sigma$ over seeds" if sign > 0 else None)
    ax.axhline(0, ls=":", color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Wi$ (log)", fontsize=12)
    ax.set_ylabel(r"$z_2^{*}$  (both degenerate minima)", fontsize=12)
    ax.set_title(r"(a) ensemble mean and seed spread", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # (b) the two wells exchange depth -- zero crossing defines Wi_c
    ax = axes[0, 1]
    for i, m in enumerate(models):
        ax.plot(m["wi"], m["dF"], "-", color=COLORS[i], lw=1.3,
                label=rf"seed {m['seed']}: $Wi_c$={m['wi_c']:.3f}")
        ax.plot(m["wi"], m["barrier"], "--", color=COLORS[i], lw=0.9, alpha=0.6)
    ax.axhline(0, ls=":", color="k", lw=0.9)
    ax.set_xlim(0.15, 0.65)
    ax.set_ylim(-2.0, 1.6)
    ax.set_xlabel(r"$Wi$", fontsize=12)
    ax.set_ylabel(r"$F(z_2^{\rm out}) - F(0)$   (solid)", fontsize=11)
    ax.set_title("(b) well depths cross at $Wi_c$; dashed = barrier height",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # (c) Wi_c and the size of the jump, per seed
    ax = axes[1, 0]
    xs = np.arange(len(models))
    ax.plot(xs, wi_c, "o", ms=10, color="C1", label=r"$Wi_c$ (wells equal)",
            zorder=3)
    ax.axhline(np.nanmean(wi_c), ls="--", color="C1", lw=1.2)
    ax.fill_between([-0.5, len(models) - 0.5],
                    np.nanmean(wi_c) - np.nanstd(wi_c, ddof=1),
                    np.nanmean(wi_c) + np.nanstd(wi_c, ddof=1),
                    color="C1", alpha=0.15)
    ax.plot(xs, jump, "s", ms=8, color="0.45",
            label=r"$\Delta z_2^{*}$ jump at $Wi_c$")
    ax.axhline(np.nanmean(jump), ls=":", color="0.45", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed\n{m['seed']}" for m in models], fontsize=9)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, max(np.nanmax(wi_c), np.nanmax(jump)) * 1.45)
    ax.set_ylabel(r"$Wi_c$   /   $\Delta z_2^{*}$", fontsize=12)
    ax.set_title(rf"(c) per seed: $Wi_c={np.nanmean(wi_c):.3f}"
                 rf"\pm{np.nanstd(wi_c, ddof=1):.3f}$,  "
                 rf"$\Delta z_2^{{*}}={np.nanmean(jump):.3f}"
                 rf"\pm{np.nanstd(jump, ddof=1):.3f}$", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # (d) does rescaling Wi by each seed's own Wi_c actually collapse the
    #     curves?  Compare the seed spread at fixed Wi with the spread at fixed
    #     u = Wi/Wi_c, both evaluated on the same portion of the curve.
    ax = axes[1, 1]
    u = np.geomspace(0.6, 25.0, 300)
    resc = np.array([np.interp(u, wi / wi_c[i], models[i]["z2_star"])
                     for i in range(len(models))])
    raw = np.array([np.interp(u * wi_c.mean(), wi, models[i]["z2_star"])
                    for i in range(len(models))])
    ax.semilogy(u, raw.std(axis=0, ddof=1), "-", color="C0", lw=1.8,
                label=r"at fixed $Wi$ (raw)")
    ax.semilogy(u, resc.std(axis=0, ddof=1), "-", color="C3", lw=1.8,
                label=r"at fixed $Wi/Wi_c^{(\mathrm{seed})}$ (rescaled)")
    ax.axvline(1.0, ls=":", color="k", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Wi / \langle Wi_c\rangle$   (rescaled: $Wi/Wi_c^{(\rm seed)}$)",
                  fontsize=11)
    ax.set_ylabel(r"seed-to-seed std of $z_2^{*}$", fontsize=12)
    ax.set_title(r"(d) rescaling by $Wi_c$ only helps at the jump",
                 fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Merging the 5 models: same first-order jump in $z_2^*$; the "
                 "seeds disagree only about where it happens", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "z2_argmin_ensemble.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


def save_data(models):
    np.savez_compressed(
        HERE / "z2_argmin_data.npz",
        wi_values=models[0]["wi"],
        z2_grid=models[0]["z2"],
        F=np.array([m["F"] for m in models]),
        z2_star=np.array([m["z2_star"] for m in models]),
        z2_outer=np.array([m["z2_outer"] for m in models]),
        dF_outer_minus_origin=np.array([m["dF"] for m in models]),
        barrier=np.array([m["barrier"] for m in models]),
        origin_is_min=np.array([m["origin_min"] for m in models]),
        seeds=np.array([m["seed"] for m in models]),
        model_paths=np.array([m["path"] for m in models]),
        wi_c=np.array([m["wi_c"] for m in models]),
        z2_jump_at_wi_c=np.array([m["jump"] for m in models]),
        wi_first_second_min=np.array([m["wi_2nd"] for m in models]),
    )
    print(f"  -> {HERE / 'z2_argmin_data.npz'}", flush=True)

    with open(HERE / "z2_argmin_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seed", "Wi", "z2_star_pos", "z2_star_neg",
                    "z2_outer_local_min", "F_outer_minus_F0", "barrier",
                    "origin_is_local_min"])
        for m in models:
            for k, x in enumerate(m["wi"]):
                w.writerow([m["idx"] + 1, m["seed"], f"{x:.6g}",
                            f"{m['z2_star'][k]:.6f}",
                            f"{-m['z2_star'][k]:.6f}",
                            f"{m['z2_outer'][k]:.6f}",
                            f"{m['dF'][k]:.6f}", f"{m['barrier'][k]:.6f}",
                            int(m["origin_min"][k])])
    print(f"  -> {HERE / 'z2_argmin_data.csv'}", flush=True)

    with open(HERE / "wi_critical.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seed", "model_path", "Wi_c", "z2_jump_at_Wi_c",
                    "Wi_first_second_min", "max_asym_F",
                    "max_quadrature_err_F", "max_quadrature_err_z2"])
        for m in models:
            w.writerow([m["idx"] + 1, m["seed"], m["path"],
                        f"{m['wi_c']:.6f}", f"{m['jump']:.6f}",
                        f"{m['wi_2nd']:.6f}", f"{m['asym']:.3e}",
                        f"{m['quad']:.3e}", f"{m['quad_z2']:.3e}"])
    print(f"  -> {HERE / 'wi_critical.csv'}", flush=True)


def main():
    models = load_all()
    print(f"loaded {len(models)} models, {len(models[0]['wi'])} Wi values\n")
    print(f"  {'seed':>6} {'Wi_c':>7} {'jump':>6} {'barrier@Wi_c':>13} "
          f"{'z2*(10)':>8} {'sym err':>9} {'quad dF':>9} {'quad dz2*':>10}")
    for m in models:
        k = int(np.argmin(np.abs(m["wi"] - m["wi_c"])))
        print(f"  {m['seed']:>6} {m['wi_c']:>7.4f} {m['jump']:>6.3f} "
              f"{m['barrier'][k]:>13.3f} {m['z2_star'][-1]:>8.4f} "
              f"{m['asym']:>9.1e} {m['quad']:>9.1e} {m['quad_z2']:>10.1e}")
    wi_c = np.array([m["wi_c"] for m in models])
    jump = np.array([m["jump"] for m in models])
    print(f"\n  Wi_c        = {wi_c.mean():.4f} +- {wi_c.std(ddof=1):.4f}"
          f"   (mean +- std over seeds)")
    print(f"  jump dz2*   = {jump.mean():.4f} +- {jump.std(ddof=1):.4f}\n")

    for m in models:
        per_model_plot(m)
    combined_plot(models)
    ensemble_plot(models)
    save_data(models)


if __name__ == "__main__":
    main()
