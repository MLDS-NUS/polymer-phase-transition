"""F23(z2=0, z3) at two Wi, with the F23(z2,z3) surface each is cut from.

The counterpart of ../1-trans-z2/plot_z2_z3_surface.py, but the relation between
the main curve and the inset is DIFFERENT, and the figure says so:

    z2 folder : the curve is the surface *reduced* over z3 (integrated away)
    here      : the curve is the surface *sliced* at z2 = 0 (one column of it)

In both cases only z1 is marginalised inside F23.  The inset therefore carries a
vertical line at z2 = 0, in the main curve's colour, marking exactly which
column is plotted.

Wi = 0.66 is below every seed's 1 k_BT threshold (the z3 well exists but is
less than a k_BT deep); Wi = 2.64 is above it (two wells several k_BT deep).

The first figure is laid out like ../1-trans-z2/F_z2_z3_surface_seed<seed>.png
(same script structure there): curve shifted to min 0, inset surface on a log
colour scale, few labels, and one y range for both panels: the first panel
spans |z3| <= 6, the second is widened until its curve reaches the same
height.  The second output file shows the two wells again, zoomed in.

    GPU=0 python plot_z3_surface.py          # seed 1 by default
    MODEL_IDX=0 python plot_z3_surface.py    # another seed (0 = seed 123)

The first run computes the surfaces and saves them; later runs only replot from
the saved file (seconds), unless it was made with other WI_SHOW / grid settings
or RECOMPUTE=1 is set.

Writes
    F_z3_slice_surface_seed<seed>.png       the slices + surface insets
    F_z3_well_zoom_seed<seed>.png           the same two wells, each on its own scale
    F_z3_slice_surface_seed<seed>_data.npz  all plotted data:
        wi [n_wi]                  the Wi of each panel
        z2 [401], z3 [601]         surface grid
        F23 [n_wi, 401, 601]       F23(z2, z3), unshifted
        z3_slice [2401]            slice grid (the cached one)
        F_slice [n_wi, 2401]       F23(0, z3) - F23(0, 0)
        z3_star, depth [n_wi]      slice minimiser (z3 >= 0) and its depth
        min2d [n_wi, 2]            (z2, z3) of the surface's global minimum
                                   (saved for reference, not drawn)
        n_z1, z3_stride, seed, model_path
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO_ROOT = PARENT.parent.parent
for p in (str(PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from omegaconf import OmegaConf

jax.config.update("jax_enable_x64", True)

from I2_polymer_dynamics_wi import build_model  # noqa: E402
from compute_z3_curves import (  # noqa: E402
    MODEL_PATHS, N_Z1, T_FIXED, Z1_MAX, Z1_MIN)

WI_SHOW = [0.66, 2.64]
Z2_MIN, Z2_MAX = -5.0, 5.0      # notebook range for the surface
N_Z2 = 401
Z3_STRIDE = 4                   # surface z3 grid = every 4th cached z3 point
Z3_VIEW = 6.0                   # first panel's x range, and inset z3 range
Z2_VIEW = 3.0                   # inset z2 range
# inset box [x0, y0, w, h] in main-axes fractions, the same in both folders;
# x0 sits a little left of centre so the colourbar labels clear the steep
# Wi = 0.01 curve of the z2 figure (centred, that figure's top would be lifted)
INSET = [0.23, 0.40, 0.50, 0.56]
Y_LO = -0.5                     # bottom of the main axes
# contour levels, the same as in the z2 figure: over this window F23 - min
# runs to ~5e3, and at Wi = 2.64 the whole z2 = 0 column lies between 50 and 500
LEVELS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
# crimson: clear on viridis (which has no red) and distinct from the black of
# the z2 figure
CURVE = dict(color="k", lw=2.0)                 # the main curve
ZLINE = dict(color=CURVE["color"], lw=2.0)      # the z2 = 0 column in the inset
# font sizes of figure 1
FS = dict(
    tick=13,            # main-axes tick labels
    label=24,           # main-axes axis labels
    title=25.5,         # the "Wi = ..." panel titles
    inset_tick=11,
    inset_label=12,
    cbar_tick=10,
)


def fmt_depth(d):
    """0.33 -> '0.33', 12.09 -> '12.1', 0.00416 -> '0.0042' (not '0.00')."""
    return f"{d:.1f}" if d >= 1 else f"{d:.2g}"


def drawn_boxes(axi, cb, r):
    """Display-space boxes of what the inset draws: its frame, each tick label
    and axis label on its own, and the colourbar with its tick labels."""
    boxes = [axi.get_window_extent(r), cb.ax.get_window_extent(r),
             axi.xaxis.label.get_window_extent(r),
             axi.yaxis.label.get_window_extent(r)]
    for axis in (axi.xaxis, axi.yaxis, cb.ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        for t in axis.get_major_ticks() + axis.get_minor_ticks():
            if lo <= t.get_loc() <= hi:
                boxes += [lab.get_window_extent(r) for lab in (t.label1, t.label2)
                          if lab.get_visible() and lab.get_text()]
    return boxes


def lift_to_clear(ax, z, F, boxes, y_lo, y_hi, margin=0.02):
    """Smallest frame top >= y_hi that keeps the curve (z, F) at least `margin`
    below every box (Bbox in ax's axes fractions) it passes under."""
    x0, x1 = ax.get_xlim()
    xf = (z - x0) / (x1 - x0)
    for bb in boxes:
        under = (xf >= bb.x0) & (xf <= bb.x1)
        if under.any():
            y_hi = max(y_hi, y_lo + (F[under].max() - y_lo) / (bb.y0 - margin))
    return y_hi


def make_F23(model, n_z1=N_Z1):
    z1_grid = jnp.linspace(Z1_MIN, Z1_MAX, n_z1)
    dz1 = (Z1_MAX - Z1_MIN) / (n_z1 - 1)

    def F23(z2, z3, Wi):
        args = jnp.array([T_FIXED, Wi])
        v = jax.vmap(lambda z1: model.potential(jnp.array([z1, z2, z3]), args))(z1_grid)
        return -(logsumexp(-v) + jnp.log(dz1))

    @jax.jit
    def grid(z2_grid, z3_grid, Wi):
        # lax.map over z2, vmap over z3: vmapping all three axes at once
        # materialises n_z2 * n_z3 * n_z1 * 4 hidden rows and exhausts the GPU
        return jax.lax.map(
            lambda z2: jax.vmap(F23, (None, 0, None))(z2, z3_grid, Wi), z2_grid)

    return grid


def compute_data(mi, path, config):
    """Everything the two figures need; the slow part (~4 min on CPU)."""
    model = eqx.tree_deserialise_leaves(path / "model.eqx", build_model(config))
    grid_fn = make_F23(model)

    cached = np.load(HERE / "cache" / f"curves_m{mi}.npz", allow_pickle=True)
    z3_full = cached["z3_grid"]
    z2 = np.linspace(Z2_MIN, Z2_MAX, N_Z2)
    z3 = z3_full[::Z3_STRIDE]           # aligned with the cached grid
    i0_z2 = int(np.argmin(np.abs(z2)))
    i0_z3 = int(np.argmin(np.abs(z3_full)))

    d = {k: [] for k in ("F23", "F_slice", "z3_star", "depth", "min2d")}
    for wi in WI_SHOW:
        F23 = np.asarray(grid_fn(jnp.array(z2), jnp.array(z3), wi))   # [z2, z3]

        # the main curve IS the z2 = 0 column of this surface
        k = int(np.argmin(np.abs(cached["wi_values"] - wi)))
        F1 = cached["F"][k] - cached["F"][k][i0_z3]     # reference to z3 = 0
        col = F23[i0_z2] - F23[i0_z2, int(np.argmin(np.abs(z3)))]
        print(f"  Wi={wi}: max |surface column - cached slice| = "
              f"{np.abs(col - F1[::Z3_STRIDE]).max():.2e}", flush=True)

        j = i0_z3 + int(np.argmin(F1[i0_z3:]))
        g2, g3 = np.unravel_index(int(np.argmin(F23)), F23.shape)
        print(f"      2D minimum of the surface at (z2,z3) = "
              f"({z2[g2]:+.2f}, {z3[g3]:+.2f})", flush=True)
        d["F23"].append(F23)
        d["F_slice"].append(F1)
        d["z3_star"].append(z3_full[j])
        d["depth"].append(-F1[j])
        d["min2d"].append((z2[g2], z3[g3]))

    d = {k: np.array(v) for k, v in d.items()}
    d.update(wi=np.array(WI_SHOW), z2=z2, z3=z3, z3_slice=z3_full,
             n_z1=N_Z1, z3_stride=Z3_STRIDE, seed=int(config.model.seed),
             model_path=str(MODEL_PATHS[mi]))
    return d


def load_data(f):
    """The saved plotting data, or None if absent or made with other settings."""
    if not f.exists():
        return None
    with np.load(f) as z:
        d = dict(z)
    same = (d["wi"].shape == (len(WI_SHOW),) and np.allclose(d["wi"], WI_SHOW)
            and d["z2"].size == N_Z2
            and np.isclose(d["z2"][0], Z2_MIN) and np.isclose(d["z2"][-1], Z2_MAX)
            and int(d["n_z1"]) == N_Z1 and int(d["z3_stride"]) == Z3_STRIDE)
    return d if same else None


def get_data(mi):
    """(seed, plotting data) for model `mi`.  The data file is computed and
    saved first if it is missing or was made with other settings, or if
    RECOMPUTE=1; delete it to redo the surfaces."""
    path = PARENT / MODEL_PATHS[mi]
    config = OmegaConf.load(path / ".hydra/config.yaml")
    seed = int(config.model.seed)
    data_file = HERE / f"F_z3_slice_surface_seed{seed}_data.npz"
    d = None if os.environ.get("RECOMPUTE") else load_data(data_file)
    if d is None:
        d = compute_data(mi, path, config)
        np.savez(data_file, **d)
        print(f"  -> {data_file}", flush=True)
    else:
        print(f"  plotting from {data_file}", flush=True)
    return seed, d


def panels_from(d):
    return [dict(wi=float(wi), F23=d["F23"][n], F1=d["F_slice"][n],
                 zs=float(d["z3_star"][n]), depth=float(d["depth"][n]),
                 fmin=-float(d["depth"][n]))
            for n, wi in enumerate(d["wi"])]


def draw_panels(fig, axes, d):
    """The slice + surface-inset panels, one per Wi, on `axes`.  Returns the
    items clear_insets() needs and the frame top these panels ask for."""
    z2, z3, z3_full = d["z2"], d["z3"], d["z3_slice"]
    panels = panels_from(d)

    # the first panel spans |z3| <= Z3_VIEW, and every other panel is widened
    # to where its curve reaches the same height, so with a common y range
    # each curve leaves through the top corners of its frame
    shifted = [p["F1"] - p["fmin"] for p in panels]          # each to min 0
    f_top = float(shifted[0][np.abs(z3_full - Z3_VIEW).argmin()])
    for p, F1 in zip(panels, shifted):
        reach = np.flatnonzero((z3_full >= p["zs"]) & (F1 >= f_top))
        p["xview"] = (Z3_VIEW if p is panels[0] or not reach.size
                      else float(z3_full[reach[0]]))

    items = []
    for ax, p, F1 in zip(axes, panels, shifted):
        # ---- main axes: the z2 = 0 slice ----
        ax.plot(z3_full, F1, "-", **CURVE)
        vline = ax.axvline(0, ls=":", color="k", lw=0.8)
        ax.set_xlim(-p["xview"], p["xview"])
        ax.set_xlabel(r"$Z_3$", fontsize=FS["label"])
        ax.set_ylabel(r"$F_{23}(Z_2{=}0,Z_3)$", fontsize=FS["label"])
        ax.set_title(rf"$\mathrm{{Wi}}={p['wi']}$", fontsize=FS["title"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(alpha=0.3)

        # ---- inset: the parent 2D surface F23(z2,z3), same window in both ----
        axi = ax.inset_axes(INSET)
        m2 = np.abs(z2) <= Z2_VIEW
        m3 = np.abs(z3) <= Z3_VIEW
        S = (p["F23"] - p["F23"].min())[np.ix_(m2, m3)].T          # [z3, z2]
        # clipped at vmin: LogNorm would leave the exact 0 at the minimum blank
        # rasterized: in vector output (pdf/svg) the ~2e5 cells become one image
        pcm = axi.pcolormesh(z2[m2], z3[m3], np.maximum(S, 1e-1), shading="auto",
                             cmap="viridis", rasterized=True,
                             norm=LogNorm(vmin=1e-1, vmax=max(S.max(), 1.0)))
        axi.contour(z2[m2], z3[m3], S, levels=[l for l in LEVELS if l < S.max()],
                    colors="k", linewidths=0.4, alpha=0.35)
        # the sliced column z2 = 0
        axi.axvline(0.0, **ZLINE)
        axi.set_xlabel(r"$Z_2$", fontsize=FS["inset_label"], labelpad=1)
        axi.set_ylabel(r"$Z_3$", fontsize=FS["inset_label"], labelpad=1)
        axi.tick_params(labelsize=FS["inset_tick"])
        axi.set_xlim(-Z2_VIEW, Z2_VIEW)
        axi.set_ylim(-Z3_VIEW, Z3_VIEW)
        axi.xaxis.set_major_locator(MultipleLocator(2))
        axi.yaxis.set_major_locator(MultipleLocator(2))
        # colourbar as a child of the inset, so it cannot push past the frame
        cb = fig.colorbar(pcm, cax=axi.inset_axes([1.03, 0.0, 0.045, 1.0]))
        cb.ax.tick_params(labelsize=FS["cbar_tick"])
        items.append(dict(ax=ax, axi=axi, cb=cb, vline=vline, z=z3_full, F=F1))
    return items, 1.02 * f_top


def clear_insets(fig, items, y_lo, y_hi):
    """Give every panel in `items` the y range [y_lo, y_hi], raising y_hi where
    a curve would run into anything its inset draws, and stop each dotted
    zero line under its inset.  Measured after drawing, so this holds whatever
    FS, INSET and the layout are; call it once the layout is final."""
    y_want = y_hi
    for it in items:
        it["ax"].set_ylim(y_lo, y_hi)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for it in items:
        to_axes = it["ax"].transAxes.inverted()
        boxes = [b.transformed(to_axes)
                 for b in drawn_boxes(it["axi"], it["cb"], r)]
        y_hi = lift_to_clear(it["ax"], it["z"], it["F"], boxes, y_lo, y_hi)
        it["vline"].set_ydata([0.0, min(b.y0 for b in boxes) - 0.015])
    if y_hi > y_want:
        print(f"  note: y top raised from {y_want:.2f} to {y_hi:.2f} so the "
              f"curves clear the insets", flush=True)
    for it in items:
        it["ax"].set_ylim(y_lo, y_hi)
    return y_hi


def main():
    mi = int(os.environ.get("MODEL_IDX", "1"))
    seed, d = get_data(mi)

    # ================= figure 1: slices + surface insets =================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    items, y_hi = draw_panels(fig, axes, d)
    clear_insets(fig, items, Y_LO, y_hi)
    out = HERE / f"F_z3_slice_surface_seed{seed}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)

    # ============ figure 2: the wells, each on its own scale ==============
    z3_full = d["z3_slice"]
    panels = panels_from(d)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, p in zip(axes, panels):
        F1, zs, depth = p["F1"], p["zs"], p["depth"]
        zoom = max(abs(zs), 1.2) + 2.0
        ax.plot(z3_full, F1, "-", color="C0", lw=2.0)
        ax.plot([zs, -zs], [p["fmin"]] * 2, "o", ms=11, color="C3", zorder=4)
        ax.axvline(0, ls=":", color="k", lw=0.8)
        ax.axhline(0, ls=":", color="k", lw=0.8)
        ax.set_xlim(-zoom, zoom)
        win = np.abs(z3_full) <= zoom
        ax.set_ylim(F1[win].min() - 0.35 * max(depth, 0.3),
                    F1[win].min() + 3.2 * max(depth, 0.3))
        ax.set_xlabel(r"$z_3$", fontsize=13)
        ax.set_ylabel(r"$F_{23}(z_2{=}0,\,z_3) - F_{23}(0,0)$", fontsize=12)
        ax.set_title(rf"$Wi={p['wi']}$:   $z_3^*={{\pm}}{abs(zs):.2f}$,  "
                     rf"depth$={fmt_depth(depth)}\,k_BT$", fontsize=13)
        ax.grid(alpha=0.3)
    ratio = panels[1]["depth"] / panels[0]["depth"]
    fig.suptitle(rf"model {mi + 1} (seed {seed}): the same two wells, each on "
                 rf"its own scale (they differ by a factor {ratio:.0f} in depth)"
                 "\n"
                 rf"slice at $z_2=0$, only $z_1$ marginalised", fontsize=13)
    plt.tight_layout()
    out = HERE / f"F_z3_well_zoom_seed{seed}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
