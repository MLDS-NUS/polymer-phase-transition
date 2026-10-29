"""Stage 1: cache F(z2) curves vs Wi for the 5 seed-ensemble models.

Reproduces the free energy of `phase_diagram_symm.ipynb` (cells 3 and 6):

    F(z2) = -(1/beta) log \\int\\int exp(-beta V(z1,z2,z3; T,Wi)) dz1 dz3

evaluated by the same streamed log-sum-exp double quadrature (T=1, beta=1), but
on wider windows than the notebook's z1 in [-0.5,1.5], z3 in [-5,5]:

    z1 in [-0.5, 2.5], 512 points   the window of ../2-trans-z3, so that every
                                    figure integrates z1 over the same range
    z3 in [-15, 15],   769 points   step 10/256, as before

z3 in [-5,5] cut off the folded wells near z2 = 0 once Wi >~ 3 (they sit at
z3 = +-6.1 at Wi = 5 and +-7.8 at Wi = 10), raising F there by up to ~100 k_BT.
On the new windows the weight left outside is below double precision for every
seed, up to Wi = 10 (README.md, "积分窗口").  The notebook only animates 10 Wi
values in [0.1,0.5] and marks the extrema; here the same curve is swept over a
Wi grid that brackets the z2 transition so that the minimiser z2*(Wi) can be
plotted as a bifurcation diagram.

The 5 checkpoints differ ONLY in `model.seed` (123 / 1 / 12 / 1234 / 12345) --
same data, same architecture -- so the spread across them measures the training
noise, not a physical effect.  This is the same ensemble used by ../z3_plateau.

    GPU=0 MODEL_IDX=0 python compute_z2_curves.py     # one model, one GPU
    python compute_z2_curves.py                       # all 5, sequentially

Writes cache/curves_m<idx>.npz with
    F[wi, z2]       -- F(z2), each curve shifted to min 0 (as in the notebook)
    F_fine[wi, z2]  -- same with a 2x finer (z1,z3) quadrature (convergence)
"""
import os
import sys
from pathlib import Path
from time import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent                     # examples/UIDD_polymer_dynamics_wi
REPO_ROOT = PARENT.parent.parent         # repo root (holds net/, examples/)
# I2_polymer_dynamics_wi does sys.path.append('..') / ('../..'), i.e. relative to
# the CWD, so it only resolves net when run from the notebook directory.
# Insert the absolute paths instead so this script works from anywhere.
for p in (str(PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from omegaconf import OmegaConf

jax.config.update("jax_enable_x64", True)

from I2_polymer_dynamics_wi import build_model  # noqa: E402

# the 5 seed-ensemble checkpoints (identical except model.seed), i.e. the
# `output_paths` list of phase_diagram_symm.ipynb cell 6
MODEL_PATHS = [
    "outputs/2026_02_14-15_31_11",   # seed 123
    "outputs/2026_02_18-20_41_29",   # seed 1
    "outputs/2026_02_18-20_41_31",   # seed 12
    "outputs/2026_02_18-20_41_33",   # seed 1234
    "outputs/2026_02_18-20_41_35",   # seed 12345
]

# grids: z2 as in the notebook; z1 and z3 windows wider than its, see docstring
Z1_MIN, Z1_MAX = -0.5, 2.5                      # = ../2-trans-z3
Z2_MIN, Z2_MAX = -3.0, 3.0
Z3_MIN, Z3_MAX = -15.0, 15.0
N_Z1 = int(os.environ.get("N_Z1", "512"))       # z1 quadrature, = ../2-trans-z3
N_Z3 = 769          # z3 quadrature, step 10/256 (no env override: ../2-trans-z3
                    # reads N_Z3 as its slice grid, and both load in one process)
N_Z2 = int(os.environ.get("N_Z2", "601"))       # odd -> the grid contains z2 = 0
T_FIXED = 1.0
BETA = 1.0

# Wi grid. The z2 minimiser jumps discontinuously (there is a barrier between
# z2=0 and the outer minimum), so the window around the jump -- which sits near
# Wi ~ 0.3 for every seed -- is refined to step 0.005.
WI_VALUES = np.unique(np.round(np.concatenate([
    np.arange(0.02, 1.0001, 0.02),      # 50 pts, step 0.02
    np.arange(0.15, 0.5501, 0.005),     # 81 pts, refined through the jump
    np.arange(1.05, 2.0001, 0.05),      # 20 pts, step 0.05
    np.geomspace(2.2, 10.0, 15),        # 15 pts, log-spaced tail
]), 6))

Z2_CHUNK = int(os.environ.get("Z2_CHUNK", "151"))   # z2 points per vmap batch


def make_F_curve(model, n_z1, n_z3):
    """F(z2) with z1, z3 marginalised -- the notebook's `F_z2_stream`."""
    z1_grid = jnp.linspace(Z1_MIN, Z1_MAX, n_z1)
    z3_grid = jnp.linspace(Z3_MIN, Z3_MAX, n_z3)
    dz1 = (Z1_MAX - Z1_MIN) / (n_z1 - 1)
    dz3 = (Z3_MAX - Z3_MIN) / (n_z3 - 1)

    def V(z1, z2, z3, T, Wi):
        return model.potential(jnp.array([z1, z2, z3]), jnp.array([T, Wi]))

    def F_z2(z2, T, Wi):
        # inner: log \int exp(-beta V) dz3 at fixed z1
        def logZ_over_z3(z1):
            logw3 = -BETA * jax.vmap(V, (None, None, 0, None, None))(
                z1, z2, z3_grid, T, Wi)
            return logsumexp(logw3 + jnp.log(dz3))

        # outer: streamed log-sum-exp over z1 (scan keeps memory flat)
        def step(carry, z1):
            return jnp.logaddexp(carry, logZ_over_z3(z1)), None

        logsum_z1, _ = jax.lax.scan(step, -jnp.inf, z1_grid)
        return -(1.0 / BETA) * (logsum_z1 + jnp.log(dz1))

    @jax.jit
    def F_curve(z2_chunk, T, Wi):
        return jax.vmap(F_z2, (0, None, None))(z2_chunk, T, Wi)

    def run(z2_grid, T, Wi):
        out = [np.asarray(F_curve(z2_grid[i:i + Z2_CHUNK], T, Wi))
               for i in range(0, z2_grid.shape[0], Z2_CHUNK)]
        return np.concatenate(out)

    return run


def run_model(mi, z2_grid):
    rel = MODEL_PATHS[mi]
    path = PARENT / rel
    config = OmegaConf.load(path / ".hydra/config.yaml")
    seed = int(config.model.seed)
    model = eqx.tree_deserialise_leaves(path / "model.eqx", build_model(config))

    curve = make_F_curve(model, N_Z1, N_Z3)
    curve_fine = make_F_curve(model, 2 * N_Z1 - 1, 2 * N_Z3 - 1)

    n_wi = len(WI_VALUES)
    F = np.zeros((n_wi, N_Z2))
    # the fine quadrature costs 4x, so run it only on a spot-check subset
    check_idx = np.arange(0, n_wi, max(1, n_wi // 8))
    F_fine = np.zeros((len(check_idx), N_Z2))

    t0 = time()
    for wi_i, wi in enumerate(WI_VALUES):
        f = curve(z2_grid, T_FIXED, float(wi))
        # the notebook reports F - min(F); the additive constant is meaningless
        F[wi_i] = f - f.min()
        if wi_i in check_idx:
            ff = curve_fine(z2_grid, T_FIXED, float(wi))
            F_fine[np.searchsorted(check_idx, wi_i)] = ff - ff.min()
        if (wi_i + 1) % 10 == 0 or wi_i == n_wi - 1:
            print(f"  [seed {seed:>5}] Wi {wi_i + 1:>3}/{n_wi}  "
                  f"({time() - t0:.0f}s)", flush=True)

    out_dir = HERE / "cache"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"curves_m{mi}.npz"
    np.savez_compressed(
        out, F=F, F_fine=F_fine, check_idx=check_idx,
        z2_grid=np.asarray(z2_grid),
        wi_values=WI_VALUES, seed=seed, model_path=rel,
        z1_window=(Z1_MIN, Z1_MAX), z3_window=(Z3_MIN, Z3_MAX),
        n_z1=N_Z1, n_z3=N_Z3, T_fixed=T_FIXED, beta=BETA)

    # sanity: SymmetricPotential averages over the z2 sign flip, so F must be
    # even in z2 to floating point. If it is not, the grid or model is wrong.
    asym = np.abs(F - F[:, ::-1]).max()
    quad = np.abs(F[check_idx] - F_fine).max()
    print(f"[{mi}] seed={seed:>5}  {rel}  -> {out}\n"
          f"      max |F(z2) - F(-z2)|            = {asym:.3e}  (should be ~0)\n"
          f"      max |F(n={N_Z1}x{N_Z3}) - F(n={2*N_Z1-1}x{2*N_Z3-1})| = "
          f"{quad:.3e}  (quadrature error)", flush=True)


def main():
    z2_grid = jnp.linspace(Z2_MIN, Z2_MAX, N_Z2)
    assert N_Z2 % 2 == 1, "N_Z2 must be odd so that z2 = 0 is on the grid"

    idx = os.environ.get("MODEL_IDX")
    todo = [int(idx)] if idx is not None else list(range(len(MODEL_PATHS)))
    print(f"models={todo}  Wi: {len(WI_VALUES)} pts in "
          f"[{WI_VALUES[0]:.2f},{WI_VALUES[-1]:.2f}]  z2 {N_Z2} pts  "
          f"z1 x z3 quadrature {N_Z1} x {N_Z3} on [{Z1_MIN},{Z1_MAX}] x "
          f"[{Z3_MIN},{Z3_MAX}] (+ {2*N_Z1-1} x {2*N_Z3-1} for convergence)",
          flush=True)

    for mi in todo:
        run_model(mi, z2_grid)


if __name__ == "__main__":
    main()
