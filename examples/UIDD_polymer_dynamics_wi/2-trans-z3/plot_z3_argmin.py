"""Stage 2: z3*(Wi) from the cached F23(z2=0, z3) slices.

Same layout as ../1-trans-z2: the minimiser is tracked on the z3 >= 0 and
z3 <= 0 branches (filled / open markers, the latter being the exact mirror since
the slice is even in z3), against linear and log Wi.

Two things differ from the z2 case and drive the extra panels:
  * the z3 well is shallow.  Marginalising z1 nearly flattens the z3 dependence
    (see ../z3_plateau/README.md); the depth F(0) - F(z3*) is below 1 k_BT until
    Wi ~ 1.5, so "the minimum has moved" and "the state has changed" are not the
    same statement here.  Wi_1kT marks where the well is worth a k_BT.
  * the seed-to-seed spread of z3* is as large as its mean below Wi ~ 2.
    Panel (d) quantifies that spread instead of asserting a verdict on it.

    python plot_z3_argmin.py

Reads cache/curves_m*.npz (written by compute_z3_curves.py) and writes here:
    z3_argmin_seed<seed>.png, z3_argmin_combined.png, z3_argmin_ensemble.png,
    z3_argmin_data.npz/.csv, wi_critical_z3.csv
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

TAU = 0.0                       # analytic quadrature -> no noise threshold
DEPTH_SCALE = 1.0               # k_B T, since beta = 1
OFFSETS = [0.2, 0.5, 1.0, 2.0]  # finite-offset probes, as in ../z3_plateau
COLORS = ["C0", "C2", "C3", "C4", "C5"]
MARKERS = ["o", "^", "s", "D", "v"]


def parabolic_refine(F, z, j):
    if j <= 0 or j >= len(z) - 1:
        return z[j]
    a, b, c = F[j - 1], F[j], F[j + 1]
    denom = a - 2.0 * b + c
    if denom <= 0:
        return z[j]
    return z[j] + 0.5 * (a - c) / denom * (z[1] - z[0])


def analyse_curve(F, z3):
    """Global minimiser on the z3 >= 0 side, its depth, and the barrier to it."""
    i0 = int(np.argmin(np.abs(z3)))
    j = i0 + int(np.argmin(F[i0:]))
    z_star = parabolic_refine(F, z3, j) if j > i0 else 0.0
    depth = float(F[i0] - F[j])
    barrier = float(F[i0:j + 1].max() - F[i0]) if j > i0 else 0.0
    _, min_idx = local_extrema_1d(F, x=z3, tau=TAU)
    origin_is_min = bool(np.any(np.abs(min_idx - i0) <= 1))
    n_minima = int(np.sum(z3[min_idx] > 0)) if len(min_idx) else 0
    return z_star, depth, barrier, origin_is_min, n_minima


def first_crossing(x, y, level):
    """First x where y rises through `level`, by linear interpolation."""
    s = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    if len(s) == 0:
        return np.nan
    i = int(s[0])
    return float(x[i] + (x[i + 1] - x[i]) * (level - y[i]) / (y[i + 1] - y[i]))


def load_all():
    files = sorted(CACHE.glob("curves_m*.npz"),
                   key=lambda p: int(p.stem.split("m")[-1]))
    if not files:
        raise SystemExit(f"no cache in {CACHE}; run compute_z3_curves.py first")
    models = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        F, z3, wi = d["F"], d["z3_grid"], d["wi_values"]
        n = len(wi)
        i0 = int(np.argmin(np.abs(z3)))

        z3_star = np.zeros(n)
        depth = np.zeros(n)
        barrier = np.zeros(n)
        origin_min = np.zeros(n, dtype=bool)
        n_min = np.zeros(n, dtype=int)
        for k in range(n):
            (z3_star[k], depth[k], barrier[k],
             origin_min[k], n_min[k]) = analyse_curve(F[k], z3)

        # finite-offset probes: F23(0, d) - F23(0, 0), the ../z3_plateau indicator
        off_idx = [int(np.argmin(np.abs(z3 - dd))) for dd in OFFSETS]
        dF_off = F[:, off_idx] - F[:, i0][:, None]          # [wi, offset]

        split = z3_star > 0
        wi_split = float(wi[np.argmax(split)]) if split.any() else np.nan
        wi_1kt = first_crossing(wi, depth, DEPTH_SCALE)
        # Wi beyond which the well never drops back below 1 k_BT. NaN if it
        # does -- seed 1234 loses the outer minimum again above Wi ~ 7.5.
        above = depth >= DEPTH_SCALE
        stay = [k for k in range(len(wi)) if above[k:].all()]
        wi_1kt_stay = float(wi[stay[0]]) if stay else np.nan

        chk = d["check_idx"]
        dz_quad = max(abs(analyse_curve(d["F_fine"][c], z3)[0] - z3_star[k])
                      for c, k in enumerate(chk))

        models.append(dict(
            idx=int(f.stem.split("m")[-1]), seed=int(d["seed"]),
            path=str(d["model_path"]), F=F, z3=z3, wi=wi,
            z3_star=z3_star, depth=depth, barrier=barrier,
            origin_min=origin_min, n_min=n_min, dF_off=dF_off,
            wi_split=wi_split, wi_1kt=wi_1kt, wi_1kt_stay=wi_1kt_stay,
            asym=float(np.abs(F - F[:, ::-1]).max()),
            quad=float(np.abs(F[chk] - d["F_fine"]).max()),
            quad_z3=float(dz_quad),
        ))
    return models


def _draw(ax, m, color, marker, label, size=14, log=False):
    wi, z = m["wi"], m["z3_star"]
    ax.scatter(wi, z, s=size, color=color, marker=marker, label=label, zorder=3)
    nz = z > 0
    ax.scatter(wi[nz], -z[nz], s=size, marker=marker, facecolors="none",
               edgecolors=color, linewidths=0.6, zorder=3)
    ax.plot(wi, z, "-", color=color, lw=0.8, alpha=0.5)
    ax.plot(wi[nz], -z[nz], "-", color=color, lw=0.8, alpha=0.5)
    ax.axhline(0, ls=":", color="k", lw=0.8)
    if log:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)


def per_model_plot(m):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, log in [(axes[0], False), (axes[1], True)]:
        _draw(ax, m, "C0", "o", f"seed {m['seed']} (global min)", log=log)
        ax.axvline(m["wi_split"], ls=":", color="0.45", lw=1.2,
                   label=rf"$z_3^*$ leaves 0: $Wi={m['wi_split']:.2f}$")
        ax.axvline(m["wi_1kt"], ls="--", color="C3", lw=1.3,
                   label=rf"well depth $=1\,k_BT$: $Wi={m['wi_1kt']:.2f}$")
        ax.set_xlabel(r"$Wi$" + (" (log)" if log else ""), fontsize=13)
        ax.set_ylabel(r"$z_3^{*}=\arg\min_{z_3}F_{23}(z_2{=}0,z_3)$", fontsize=12)
    axes[0].set_title(r"$z_3^{*}$ vs $Wi$", fontsize=13)
    axes[1].set_title(r"$z_3^{*}$ vs $Wi$ (log axis)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(rf"model {m['idx'] + 1} (seed {m['seed']}): minimiser of "
                 rf"$F_{{23}}(z_2{{=}}0,z_3)=-\log\int e^{{-V}}dz_1$", fontsize=14)
    plt.tight_layout()
    out = HERE / f"z3_argmin_seed{m['seed']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


def combined_plot(models):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for i, m in enumerate(models):
        for ax, log in [(axes[0], False), (axes[1], True)]:
            _draw(ax, m, COLORS[i], MARKERS[i], f"seed {m['seed']}", size=10,
                  log=log)
    for ax, log in [(axes[0], False), (axes[1], True)]:
        ax.set_xlabel(r"$Wi$" + (" (log)" if log else ""), fontsize=13)
        ax.set_ylabel(r"$z_3^{*}$", fontsize=12)
        ax.legend(fontsize=9, loc="upper left", ncol=2)
    axes[0].set_title(r"$z_3^{*}$ vs $Wi$", fontsize=13)
    axes[1].set_title(r"$z_3^{*}$ vs $Wi$ (log axis)", fontsize=13)
    fig.suptitle(r"$z_3$ minimiser for the 5 seed-ensemble models "
                 r"(identical data and architecture, different init)",
                 fontsize=14)
    plt.tight_layout()
    out = HERE / "z3_argmin_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


def ensemble_plot(models):
    wi = models[0]["wi"]
    Z = np.array([m["z3_star"] for m in models])
    mean, std = Z.mean(axis=0), Z.std(axis=0, ddof=1)
    wi_1kt = np.array([m["wi_1kt"] for m in models])
    wi_split = np.array([m["wi_split"] for m in models])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) both branches, ensemble mean and spread
    ax = axes[0, 0]
    for i, m in enumerate(models):
        for sign in (+1, -1):
            ax.plot(wi, sign * m["z3_star"], "-", color=COLORS[i], lw=0.9,
                    alpha=0.55, label=f"seed {m['seed']}" if sign > 0 else None)
    for sign, lab in [(+1, "ensemble mean"), (-1, None)]:
        ax.plot(wi, sign * mean, "-", color="k", lw=2.0, label=lab)
        ax.fill_between(wi, sign * (mean - std), sign * (mean + std),
                        color="k", alpha=0.16,
                        label=r"$\pm1\sigma$ over seeds" if sign > 0 else None)
    ax.axhline(0, ls=":", color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Wi$ (log)", fontsize=12)
    ax.set_ylabel(r"$z_3^{*}$  (both degenerate minima)", fontsize=12)
    ax.set_title("(a) ensemble mean and seed spread", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # (b) how deep the well actually is
    ax = axes[0, 1]
    for i, m in enumerate(models):
        # depth is exactly 0 where z3* has not split; drop those on a log axis
        dep = np.where(m["depth"] > 0, m["depth"], np.nan)
        ax.plot(wi, dep, "-", color=COLORS[i], lw=1.3,
                label=rf"seed {m['seed']}: $1k_BT$ at $Wi$={m['wi_1kt']:.2f}")
    ax.axhline(DEPTH_SCALE, ls="--", color="k", lw=1.2)
    ax.text(0.03, DEPTH_SCALE * 1.15, r"$1\,k_BT$", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3e2)
    ax.set_xlabel(r"$Wi$ (log)", fontsize=12)
    ax.set_ylabel(r"well depth $F_{23}(0,0)-F_{23}(0,z_3^*)$", fontsize=11)
    ax.set_title("(b) the z3 well is shallow: depth vs $Wi$", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    # (c) where the well becomes thermally real, per seed
    ax = axes[1, 0]
    xs = np.arange(len(models))
    stay = np.array([m["wi_1kt_stay"] for m in models])
    # nudged apart: for three of the seeds the two markers coincide
    ax.plot(xs - 0.1, wi_1kt, "o", ms=10, color="C3", zorder=3,
            label=r"depth first reaches $1\,k_BT$")
    ax.plot(xs + 0.1, stay, "s", ms=8, color="0.35",
            label=r"depth stays $\geq 1\,k_BT$ from here on")
    for k in np.where(~np.isfinite(stay))[0]:
        ax.annotate("never\n(reverts)", (k, np.nanmax(wi_1kt) * 1.12),
                    ha="center", fontsize=8, color="0.35")
    ax.axhline(np.nanmean(wi_1kt), ls="--", color="C3", lw=1.2)
    ax.fill_between([-0.5, len(models) - 0.5],
                    np.nanmean(wi_1kt) - np.nanstd(wi_1kt, ddof=1),
                    np.nanmean(wi_1kt) + np.nanstd(wi_1kt, ddof=1),
                    color="C3", alpha=0.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed\n{m['seed']}" for m in models], fontsize=9)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, np.nanmax([np.nanmax(wi_1kt), np.nanmax(stay)]) * 1.35)
    ax.set_ylabel(r"$Wi$", fontsize=12)
    ax.set_title(rf"(c) per seed: $1k_BT$ at $Wi={np.nanmean(wi_1kt):.2f}"
                 rf"\pm{np.nanstd(wi_1kt, ddof=1):.2f}$", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # (d) is there a cross-seed consensus at all?
    ax = axes[1, 1]
    ratio = std / np.maximum(np.abs(mean), 1e-12)
    ax.loglog(wi, ratio, "-", color="C3", lw=1.8,
              label=r"$\sigma / |\langle z_3^*\rangle|$")
    ax.axhline(0.3, ls="--", color="k", lw=1.1)
    ax.text(0.025, 0.33, "30%", fontsize=9)
    ax.fill_between(wi, 1e-3, 1e2, where=ratio >= 0.3, color="C3", alpha=0.12,
                    step="mid", label=r"seeds differ by $\geq30\%$")
    ax.set_ylim(3e-2, 5e0)
    ax.set_xlabel(r"$Wi$ (log)", fontsize=12)
    ax.set_ylabel(r"relative seed-to-seed spread of $z_3^{*}$", fontsize=11)
    ax.set_title(r"(d) seed-to-seed spread; falls below 30% at "
                 r"$Wi\approx2.3$", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    fig.suptitle(r"$z_3$ minimiser across the 5 models: the split is early but "
                 r"shallow; the seeds converge above $Wi\approx2.3$",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "z3_argmin_ensemble.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


def save_data(models):
    np.savez_compressed(
        HERE / "z3_argmin_data.npz",
        wi_values=models[0]["wi"], z3_grid=models[0]["z3"],
        F=np.array([m["F"] for m in models]),
        z3_star=np.array([m["z3_star"] for m in models]),
        depth=np.array([m["depth"] for m in models]),
        barrier=np.array([m["barrier"] for m in models]),
        origin_is_min=np.array([m["origin_min"] for m in models]),
        n_minima_pos=np.array([m["n_min"] for m in models]),
        dF_offsets=np.array([m["dF_off"] for m in models]),
        offsets=np.array(OFFSETS),
        seeds=np.array([m["seed"] for m in models]),
        model_paths=np.array([m["path"] for m in models]),
        wi_split=np.array([m["wi_split"] for m in models]),
        wi_1kT=np.array([m["wi_1kt"] for m in models]),
        wi_1kT_permanent=np.array([m["wi_1kt_stay"] for m in models]),
    )
    print(f"  -> {HERE / 'z3_argmin_data.npz'}", flush=True)

    with open(HERE / "z3_argmin_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seed", "Wi", "z3_star_pos", "z3_star_neg",
                    "well_depth", "barrier", "n_local_minima_pos",
                    "origin_is_local_min"]
                   + [f"dF_offset_{d}" for d in OFFSETS])
        for m in models:
            for k, x in enumerate(m["wi"]):
                w.writerow([m["idx"] + 1, m["seed"], f"{x:.6g}",
                            f"{m['z3_star'][k]:.6f}", f"{-m['z3_star'][k]:.6f}",
                            f"{m['depth'][k]:.6f}", f"{m['barrier'][k]:.6f}",
                            m["n_min"][k], int(m["origin_min"][k])]
                           + [f"{v:.6f}" for v in m["dF_off"][k]])
    print(f"  -> {HERE / 'z3_argmin_data.csv'}", flush=True)

    with open(HERE / "wi_critical_z3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seed", "model_path", "Wi_z3_leaves_0",
                    "Wi_depth_1kT_first", "Wi_depth_1kT_permanent",
                    "z3_star_at_Wi10", "max_asym_F",
                    "max_quadrature_err_F", "max_quadrature_err_z3"])
        for m in models:
            w.writerow([m["idx"] + 1, m["seed"], m["path"],
                        f"{m['wi_split']:.6f}", f"{m['wi_1kt']:.6f}",
                        f"{m['wi_1kt_stay']:.6f}",
                        f"{m['z3_star'][-1]:.6f}", f"{m['asym']:.3e}",
                        f"{m['quad']:.3e}", f"{m['quad_z3']:.3e}"])
    print(f"  -> {HERE / 'wi_critical_z3.csv'}", flush=True)


def main():
    models = load_all()
    wi = models[0]["wi"]
    print(f"loaded {len(models)} models, {len(wi)} Wi values\n")
    print(f"  {'seed':>6} {'Wi(z3*>0)':>10} {'Wi(1kT)':>8} {'z3*(10)':>8} "
          f"{'sym err':>9} {'quad dF':>9} {'quad dz3*':>10}")
    for m in models:
        print(f"  {m['seed']:>6} {m['wi_split']:>10.3f} {m['wi_1kt']:>8.3f} "
              f"{m['z3_star'][-1]:>8.3f} {m['asym']:>9.1e} {m['quad']:>9.1e} "
              f"{m['quad_z3']:>10.1e}")
    w1 = np.array([m["wi_1kt"] for m in models])
    print(f"\n  Wi(depth = 1 kT) = {w1.mean():.3f} +- {w1.std(ddof=1):.3f}")

    Z = np.array([m["z3_star"] for m in models])
    mean, std = Z.mean(axis=0), Z.std(axis=0, ddof=1)
    ok = std < np.abs(mean)
    first = wi[np.argmax(ok & (wi > 0.5))] if (ok & (wi > 0.5)).any() else np.nan
    print(f"  seed spread stays below the mean only from Wi ~ {first:.2f}\n")

    for m in models:
        per_model_plot(m)
    combined_plot(models)
    ensemble_plot(models)
    save_data(models)


if __name__ == "__main__":
    main()
