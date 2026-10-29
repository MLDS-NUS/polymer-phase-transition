"""F(z2) at Wi = 0.01 and 0.66, each with its parent F(z2,z3) surface inset.

The 1D curve everywhere else in this folder marginalises both z1 and z3.  The
inset keeps z3 alive and marginalises z1 only,

    F23(z2,z3) = -log \\int e^{-V(z1,z2,z3;Wi)} dz1 ,

so the main curve is exactly the z3 reduction of the inset,
    F(z2) = -log \\int e^{-F23(z2,z3)} dz3 ,
which is checked against the cached curves when the surfaces are computed, at
the Wi the cache has (prints the max difference, ~1e-12; the cache starts at
Wi = 0.02).

Wi = 0.01 is below every seed's Wi_c (global minimum at z2 = 0; some seeds,
e.g. 123, already have shallow side minima ~0.3 k_BT up at z2 ~ +-0.12);
Wi = 0.66 is above it (two degenerate global minima at +-z2*).

Laid out like ../2-trans-z3/F_z3_slice_surface_seed<seed>.png (same inset
size and window, z2 in [-3, 3], z3 in [-6, 6]; log colour scale; few labels),
except that the curve is blue and the y axis is F_2(z2).  The y axis ends at
Y_TOP, the top of the seed-1 z3 figure (other seeds' z3 figures end elsewhere,
66.7-75.3), and each panel's x range ends where its curve reaches it, so both
curves leave through the top corners.  The surface is computed on the cached
curves' quadrature grids -- z1 in [-0.5, 2.5] (512 points, the window of
../2-trans-z3, so all four panels of ../2-trans-z3/F_z2_z3_grid_seed<seed>
integrate z1 over one window) and z3 in [-15, 15] (769 points) -- and F(z2)
integrates the whole surface; the inset shows its |z3| <= 6 part.

    GPU=0 python plot_z2_z3_surface.py          # seed 1 by default
    MODEL_IDX=0 python plot_z2_z3_surface.py    # another seed (0 = seed 123)

The first run computes the surfaces and saves them; later runs only replot from
the saved file (seconds), unless it was made with other WI_SHOW / grid settings
or RECOMPUTE=1 is set.

Writes
    F_z2_z3_surface_seed<seed>.png          the curves + surface insets
    F_z2_z3_surface_seed<seed>_data.npz     all plotted data:
        wi [n_wi]                  the Wi of each panel
        z2 [601], z3 [769]         grid (the cached curves' z2 grid and z3
                                   quadrature grid, [-15, 15])
        F23 [n_wi, 601, 769]       F23(z2, z3), unshifted
        F [n_wi, 601]              F(z2), shifted to min 0
        z2_star [n_wi]             minimiser of F on z2 >= 0
        min2d [n_wi, 2]            (z2, z3) of the surface's global minimum
                                   (saved for reference, not drawn)
        z1_window, n_z1, n_z3, seed, model_path
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
from compute_z2_curves import (  # noqa: E402
    MODEL_PATHS, N_Z1, N_Z3, T_FIXED, Z1_MAX, Z1_MIN, Z2_MAX, Z2_MIN, Z3_MAX,
    Z3_MIN)

WI_SHOW = [0.01, 0.66]
N_Z2 = 601          # same z2 grid as the cached curves
# the surface's z3 grid is the cached curves' z3 quadrature grid itself; it
# already covers the inset window
DZ3 = (Z3_MAX - Z3_MIN) / (N_Z3 - 1)
# top of the main axes, the same as in ../2-trans-z3/F_z3_slice_surface_seed1.png
# (1.02 x its Wi = 0.66 slice at |z3| = 6); each panel's x range ends where its
# curve reaches it
Y_TOP = 74.27
Y_LO = -0.5         # bottom of the main axes
INSET_Z2, INSET_Z3 = 3.0, 6.0     # inset window, as in the z3 figure
# inset box [x0, y0, w, h] in main-axes fractions, the same in both folders;
# x0 sits a little left of centre so the colourbar labels clear the steep
# Wi = 0.01 curve of the z2 figure (centred, that figure's top would be lifted)
INSET = [0.23, 0.40, 0.50, 0.56]
LEVELS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
CURVE = dict(color="#1f77b4", lw=2.0)           # the main curve (default C0)
# font sizes, as in ../2-trans-z3/plot_z3_surface.py
FS = dict(
    tick=13,            # main-axes tick labels
    label=24,           # main-axes axis labels
    title=25.5,         # the "Wi = ..." panel titles
    inset_tick=11,
    inset_label=12,
    cbar_tick=10,
)


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
    """F23(z2,z3) = -log \\int exp(-V) dz1, marginalising z1 only."""
    z1_grid = jnp.linspace(Z1_MIN, Z1_MAX, n_z1)
    dz1 = (Z1_MAX - Z1_MIN) / (n_z1 - 1)

    def F23(z2, z3, Wi):
        args = jnp.array([T_FIXED, Wi])
        v = jax.vmap(lambda z1: model.potential(jnp.array([z1, z2, z3]), args))(z1_grid)
        return -(logsumexp(-v) + jnp.log(dz1))

    @jax.jit
    def grid(z2_row, z3_grid, Wi):
        # lax.map (not vmap) over z2: vmapping all three axes at once would
        # materialise n_z2 * n_z3 * n_z1 * 4 hidden rows and blow up the GPU.
        return jax.lax.map(
            lambda z2: jax.vmap(F23, (None, 0, None))(z2, z3_grid, Wi), z2_row)

    return grid


def compute_data(mi, path, config):
    """Everything the figure needs; the slow part."""
    model = eqx.tree_deserialise_leaves(path / "model.eqx", build_model(config))
    grid_fn = make_F23(model)

    z2 = np.linspace(Z2_MIN, Z2_MAX, N_Z2)
    z3 = np.linspace(Z3_MIN, Z3_MAX, N_Z3)
    cached = np.load(HERE / "cache" / f"curves_m{mi}.npz", allow_pickle=True)
    i0 = int(np.argmin(np.abs(z2)))

    d = {k: [] for k in ("F23", "F", "z2_star", "min2d")}
    for wi in WI_SHOW:
        F23 = np.asarray(grid_fn(jnp.array(z2), jnp.array(z3), wi))  # [z2, z3]

        # the main curve is the z3 reduction of the whole surface. Shift by the
        # per-z2 row minimum: a single global shift underflows to 0 in the rows
        # where F23 is hundreds above the minimum.
        row = F23.min(axis=1)
        F1 = row - np.log(np.exp(-(F23 - row[:, None])).sum(axis=1)) - np.log(DZ3)
        F1 -= F1.min()

        # consistency with the cached curve computed by compute_z2_curves.py
        k = int(np.argmin(np.abs(cached["wi_values"] - wi)))
        if abs(cached["wi_values"][k] - wi) < 1e-9:
            print(f"  Wi={wi}: max |F(z2) from surface - cached| = "
                  f"{np.abs(F1 - cached['F'][k]).max():.2e}", flush=True)
        else:
            print(f"  Wi={wi}: no cached curve to check against", flush=True)

        j = i0 + int(np.argmin(F1[i0:]))
        g2, g3 = np.unravel_index(int(np.argmin(F23)), F23.shape)
        d["F23"].append(F23)
        d["F"].append(F1)
        d["z2_star"].append(z2[j])
        d["min2d"].append((z2[g2], z3[g3]))

    d = {k: np.array(v) for k, v in d.items()}
    d.update(wi=np.array(WI_SHOW), z2=z2, z3=z3,
             z1_window=np.array([Z1_MIN, Z1_MAX]), n_z1=N_Z1, n_z3=N_Z3,
             seed=int(config.model.seed), model_path=str(MODEL_PATHS[mi]))
    return d


def load_data(f):
    """The saved plotting data, or None if absent or made with other settings."""
    if not f.exists():
        return None
    with np.load(f) as z:
        d = dict(z)
    same = (d["wi"].shape == (len(WI_SHOW),) and np.allclose(d["wi"], WI_SHOW)
            and d["z2"].size == N_Z2 and d["z3"].size == N_Z3
            and np.isclose(d["z2"][0], Z2_MIN) and np.isclose(d["z2"][-1], Z2_MAX)
            and np.isclose(d["z3"][0], Z3_MIN) and np.isclose(d["z3"][-1], Z3_MAX)
            and "n_z1" in d and int(d["n_z1"]) == N_Z1
            and np.allclose(d["z1_window"], [Z1_MIN, Z1_MAX]))
    return d if same else None


def get_data(mi):
    """(seed, plotting data) for model `mi`.  The data file is computed and
    saved first if it is missing or was made with other settings, or if
    RECOMPUTE=1; delete it to redo the surfaces."""
    path = PARENT / MODEL_PATHS[mi]
    config = OmegaConf.load(path / ".hydra/config.yaml")
    seed = int(config.model.seed)
    data_file = HERE / f"F_z2_z3_surface_seed{seed}_data.npz"
    d = None if os.environ.get("RECOMPUTE") else load_data(data_file)
    if d is None:
        d = compute_data(mi, path, config)
        np.savez(data_file, **d)
        print(f"  -> {data_file}", flush=True)
    else:
        print(f"  plotting from {data_file}", flush=True)
    return seed, d


def draw_panels(fig, axes, d):
    """The F(z2) + surface-inset panels, one per Wi, on `axes`.  Returns the
    items clear_insets() needs and the frame top these panels ask for."""
    z2, z3 = d["z2"], d["z3"]
    panels = [dict(wi=float(wi), F23=d["F23"][n], F1=d["F"][n],
                   zs=float(d["z2_star"][n]))
              for n, wi in enumerate(d["wi"])]

    # each x range ends where the curve reaches 98% of Y_TOP, so with the y
    # range [Y_LO, Y_TOP] it leaves through the top corners
    for p in panels:
        reach = np.flatnonzero((z2 >= p["zs"]) & (p["F1"] >= Y_TOP / 1.02))
        p["xview"] = float(z2[reach[0]]) if reach.size else float(z2[-1])

    items = []
    for ax, p in zip(axes, panels):
        F1 = p["F1"]

        # ---- main axes: the 1D free energy ----
        ax.plot(z2, F1, "-", **CURVE)
        vline = ax.axvline(0, ls=":", color="k", lw=0.8)
        ax.set_xlim(-p["xview"], p["xview"])
        ax.set_xlabel(r"$Z_2$", fontsize=FS["label"])
        ax.set_ylabel(r"$F_2(Z_2)$", fontsize=FS["label"])
        ax.set_title(rf"$\mathrm{{Wi}}={p['wi']}$", fontsize=FS["title"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(alpha=0.3)

        # ---- inset: the parent 2D surface F23(z2,z3), same window in both ----
        axi = ax.inset_axes(INSET)
        m2 = np.abs(z2) <= INSET_Z2
        m3 = np.abs(z3) <= INSET_Z3
        S = (p["F23"] - p["F23"].min())[np.ix_(m2, m3)].T                # [z3, z2]
        # clipped at vmin: LogNorm would leave the exact 0 at the minimum blank
        # rasterized: in vector output (pdf/svg) the ~2e5 cells become one image
        pcm = axi.pcolormesh(z2[m2], z3[m3], np.maximum(S, 1e-1), shading="auto",
                             cmap="viridis", rasterized=True,
                             norm=LogNorm(vmin=1e-1, vmax=max(S.max(), 1.0)))
        axi.contour(z2[m2], z3[m3], S, levels=[l for l in LEVELS if l < S.max()],
                    colors="k", linewidths=0.4, alpha=0.35)
        axi.set_xlabel(r"$Z_2$", fontsize=FS["inset_label"], labelpad=1)
        axi.set_ylabel(r"$Z_3$", fontsize=FS["inset_label"], labelpad=1)
        axi.tick_params(labelsize=FS["inset_tick"])
        axi.set_xlim(-INSET_Z2, INSET_Z2)
        axi.set_ylim(-INSET_Z3, INSET_Z3)
        axi.xaxis.set_major_locator(MultipleLocator(2))
        axi.yaxis.set_major_locator(MultipleLocator(2))
        # colourbar as a child of the inset, so it cannot push past the frame
        cb = fig.colorbar(pcm, cax=axi.inset_axes([1.03, 0.0, 0.045, 1.0]))
        cb.ax.tick_params(labelsize=FS["cbar_tick"])
        items.append(dict(ax=ax, axi=axi, cb=cb, vline=vline, z=z2, F=F1))
    return items, Y_TOP


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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    items, y_hi = draw_panels(fig, axes, d)
    clear_insets(fig, items, Y_LO, y_hi)
    out = HERE / f"F_z2_z3_surface_seed{seed}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
