"""Stage 1: cache F23(z2=0, z3) slices vs Wi for the 5 seed-ensemble models.

Reproduces the free energy of `../phase_diagram_symm_Z3.ipynb` (cells 3, 5, 6):

    F23(z2, z3) = -(1/beta) log \\int exp(-beta V(z1,z2,z3; T,Wi)) dz1

sliced at z2 = 0.  Note this marginalises z1 ONLY -- z3 stays a free coordinate,
unlike ../1-trans-z2 where both z1 and z3 are integrated out.

Grids follow that notebook (they differ from the z2 one!):
    z1 in [-0.5, 2.5],  z3 in [-15, 15],  T = 1, beta = 1
The notebook uses a 256-point z1 quadrature and 257 z3 points; here the z1
quadrature is 512 and z3 is refined to 2401 points so the minimiser can be
located.  Both are converged: see the printed 4x check, and README.md.

The 5 checkpoints differ ONLY in `model.seed` (123 / 1 / 12 / 1234 / 12345) --
same data, same architecture -- so the spread across them measures the training
noise.  Same ensemble as ../1-trans-z2 and ../z3_plateau.

    GPU=0 MODEL_IDX=0 python compute_z3_curves.py     # one model, one GPU
    python compute_z3_curves.py                       # all 5, sequentially

Writes cache/curves_m<idx>.npz with
    F[wi, z3]       -- F23(z2=0, z3), each curve shifted to min 0
    F_fine[wi, z3]  -- same with a 4x finer z1 quadrature (convergence check)
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

# the 5 seed-ensemble checkpoints, i.e. `output_paths` of the notebook's cell 6
MODEL_PATHS = [
    "outputs/2026_02_14-15_31_11",   # seed 123
    "outputs/2026_02_18-20_41_29",   # seed 1
    "outputs/2026_02_18-20_41_31",   # seed 12
    "outputs/2026_02_18-20_41_33",   # seed 1234
    "outputs/2026_02_18-20_41_35",   # seed 12345
]

# grids from phase_diagram_symm_Z3.ipynb cell 6
Z1_MIN, Z1_MAX = -0.5, 2.5
Z3_MIN, Z3_MAX = -15.0, 15.0
Z2_FIXED = 0.0
N_Z1 = int(os.environ.get("N_Z1", "512"))       # notebook uses 256
N_Z3 = int(os.environ.get("N_Z3", "2401"))      # odd -> the grid contains z3 = 0
T_FIXED = 1.0
BETA = 1.0

# Wi grid: the notebook animates [0.1, 2.5]; extended here to see both ends
WI_VALUES = np.unique(np.round(np.concatenate([
    np.arange(0.02, 3.0001, 0.01),      # 299 pts, step 0.01
    np.arange(3.05, 10.0001, 0.05),     # 140 pts, step 0.05
]), 6))


def make_F23_line(model, n_z1):
    """F23(z2=0, z3) = -(1/beta) log \\int exp(-beta V) dz1, over the z3 grid."""
    z1_grid = jnp.linspace(Z1_MIN, Z1_MAX, n_z1)
    dz1 = (Z1_MAX - Z1_MIN) / (n_z1 - 1)

    def F23(z3, Wi):
        args = jnp.array([T_FIXED, Wi])
        v = jax.vmap(lambda z1: model.potential(
            jnp.array([z1, Z2_FIXED, z3]), args))(z1_grid)
        return -(1.0 / BETA) * (logsumexp(-BETA * v) + jnp.log(dz1))

    @jax.jit
    def line(z3_grid, Wi):
        # lax.map over z3, not vmap: vmapping z3 and z1 together materialises
        # n_z3 * n_z1 * 4 hidden rows and exhausts the GPU at these sizes.
        return jax.lax.map(lambda z3: F23(z3, Wi), z3_grid)

    return line


def run_model(mi, z3_grid):
    rel = MODEL_PATHS[mi]
    path = PARENT / rel
    config = OmegaConf.load(path / ".hydra/config.yaml")
    seed = int(config.model.seed)
    model = eqx.tree_deserialise_leaves(path / "model.eqx", build_model(config))

    line = make_F23_line(model, N_Z1)
    line_fine = make_F23_line(model, 4 * N_Z1)

    n_wi = len(WI_VALUES)
    F = np.zeros((n_wi, N_Z3))
    check_idx = np.arange(0, n_wi, max(1, n_wi // 12))
    F_fine = np.zeros((len(check_idx), N_Z3))

    t0 = time()
    for wi_i, wi in enumerate(WI_VALUES):
        f = np.asarray(line(z3_grid, float(wi)))
        F[wi_i] = f - f.min()
        if wi_i in check_idx:
            ff = np.asarray(line_fine(z3_grid, float(wi)))
            F_fine[np.searchsorted(check_idx, wi_i)] = ff - ff.min()
        if (wi_i + 1) % 50 == 0 or wi_i == n_wi - 1:
            print(f"  [seed {seed:>5}] Wi {wi_i + 1:>4}/{n_wi}  "
                  f"({time() - t0:.0f}s)", flush=True)

    out_dir = HERE / "cache"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"curves_m{mi}.npz"
    np.savez_compressed(
        out, F=F, F_fine=F_fine, check_idx=check_idx,
        z3_grid=np.asarray(z3_grid), wi_values=WI_VALUES, seed=seed,
        model_path=rel, n_z1=N_Z1, z2_fixed=Z2_FIXED, T_fixed=T_FIXED, beta=BETA)

    # sanity: SymmetricPotential averages over the z3 sign flip, so the slice
    # must be even in z3 to floating point.
    asym = np.abs(F - F[:, ::-1]).max()
    quad = np.abs(F[check_idx] - F_fine).max()
    print(f"[{mi}] seed={seed:>5}  {rel}  -> {out}\n"
          f"      max |F(z3) - F(-z3)|             = {asym:.3e}  (should be ~0)\n"
          f"      max |F(n_z1={N_Z1}) - F(n_z1={4*N_Z1})| = {quad:.3e}  "
          f"(quadrature error)", flush=True)


def main():
    z3_grid = jnp.linspace(Z3_MIN, Z3_MAX, N_Z3)
    assert N_Z3 % 2 == 1, "N_Z3 must be odd so that z3 = 0 is on the grid"

    idx = os.environ.get("MODEL_IDX")
    todo = [int(idx)] if idx is not None else list(range(len(MODEL_PATHS)))
    print(f"models={todo}  Wi: {len(WI_VALUES)} pts in "
          f"[{WI_VALUES[0]:.2f},{WI_VALUES[-1]:.2f}]  z3 {N_Z3} pts  "
          f"z1 quadrature {N_Z1} (+ {4*N_Z1} for convergence)", flush=True)

    for mi in todo:
        run_model(mi, z3_grid)


if __name__ == "__main__":
    main()
