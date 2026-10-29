#!/usr/bin/env python3
"""Uniform-proposal MC of F_2,n(z2) at a few fixed Wi only.

The paper figure needs F_2,n only at a few Wi.  This runner computes exactly
those with the uniform-proposal estimator ``../wi_001_10/mc_core.py``, the z
grid, kernel width and sample count of the original Wi sweep, and publishes one
file per (n, Wi):

    mc/n<n>_wi_<Wi:.6f>.npz

A finished file is reused on a rerun only if it was made with the same
configuration; ``--force`` recomputes it.  The uniform proposal is reliable
only at moderate beta (for n = 32 it degrades from Wi ~ 1.2 on); the
runner records the worker spread ``max_relative_se`` as a diagnostic.

    python -u run_mc_fixed.py --n 8 --wi 0.02 1.32 --workers 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent / "wi_001_10"
sys.path.insert(0, str(CORE_DIR))
from mc_core import mc_logZ_soft  # noqa: E402

# Estimator settings of the original Wi sweep (z grid, Gaussian kernel width).
WI_TO_BETA = 45000.0 / 4321.0          # Wi = beta * 4321 / 45000
Z_GRID = np.linspace(-0.95, 0.95, 39)
SIGMA = 0.05

OUTPUT_DIR = HERE / "mc"
CONFIG_VERSION = 1
BASE_SEED = 271_828_183


def atomic_savez(path: Path, **items: object) -> None:
    """Write an NPZ fully before atomically making it visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **items)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_path(n: int, wi: float) -> Path:
    return OUTPUT_DIR / f"n{n}_wi_{wi:.6f}.npz"


def seed_for(n: int, wi: float) -> int:
    return BASE_SEED + 1_000_003 * n + int(round(wi * 1e6))


def is_done(path: Path, *, n: int, wi: float, M: int, workers: int, batch: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                bool(data["complete"])
                and int(data["config_version"]) == CONFIG_VERSION
                and int(data["n"]) == n
                and int(data["M_requested"]) == M
                and int(data["n_workers"]) == workers
                and int(data["batch"]) == batch
                and np.isclose(float(data["wi"]), wi, rtol=0.0, atol=5e-13)
                and np.isclose(float(data["beta"]), wi * WI_TO_BETA, rtol=0.0, atol=5e-11)
                and np.array_equal(data["z_grid"], Z_GRID)
                and np.isfinite(data["log_Z"]).all()
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def run_one(n: int, wi: float, args: argparse.Namespace) -> Path:
    path = output_path(n, wi)
    if not args.force and is_done(path, n=n, wi=wi, M=args.M, workers=args.workers, batch=args.batch):
        print(f"n={n} Wi={wi:.6f}: done, reusing {path.name}", flush=True)
        return path
    beta = wi * WI_TO_BETA
    seed = seed_for(n, wi)
    print(f"n={n} Wi={wi:.6f} beta={beta:.10g} f={beta / n**2:.8g}: M={args.M:,}, "
          f"workers={args.workers}, seed={seed}", flush=True)
    started = time.time()
    log_z, se = mc_logZ_soft(n, Z_GRID, np.array([beta / n**2]), sigma=SIGMA, M=args.M,
                             batch=args.batch, seed=seed, verbose=True,
                             n_workers=args.workers, return_se=True)
    row, row_se = np.asarray(log_z[0], float), np.asarray(se[0], float)
    if not np.isfinite(row).all() or not np.isfinite(row_se).all():
        raise FloatingPointError(f"non-finite MC output at n={n}, Wi={wi:.6f}")
    middle = int(np.argmin(np.abs(Z_GRID)))
    relative = (row - row[middle]) / n
    relative_se = np.sqrt(row_se**2 + row_se[middle]**2) / n
    z_pos = float(Z_GRID[Z_GRID >= 0][relative[Z_GRID >= 0].argmax()])
    z_neg = float(Z_GRID[Z_GRID <= 0][relative[Z_GRID <= 0].argmax()])
    atomic_savez(
        path,
        complete=np.array(True), config_version=np.array(CONFIG_VERSION),
        wi=np.array(wi), beta=np.array(beta), n=np.array(n), z_grid=Z_GRID,
        log_Z=row, se=row_se, relative_log_Z_per_n=relative, relative_se_per_n=relative_se,
        z_star_pos=np.array(z_pos), z_star_neg=np.array(z_neg),
        argmax_at_boundary=np.array(bool(np.isclose(abs(z_pos), Z_GRID[-1]) or
                                         np.isclose(abs(z_neg), Z_GRID[-1]))),
        max_relative_se=np.array(float(relative_se.max())), sigma=np.array(SIGMA),
        M_requested=np.array(args.M),
        M_actual=np.array(((args.M + args.workers - 1) // args.workers) * args.workers),
        batch=np.array(args.batch), n_workers=np.array(args.workers), seed=np.array(seed),
        elapsed_seconds=np.array(time.time() - started),
        estimator=np.array("uniform_proposal_mc"),
    )
    print(f"  z*={z_pos:+.3f}/{z_neg:+.3f}  max_rel_SE={relative_se.max():.3e}  -> {path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--wi", type=float, nargs="+", required=True, help="Wi values to compute.")
    parser.add_argument("--M", type=int, default=1_000_000_000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MC_WORKERS", "6")))
    parser.add_argument("--batch", type=int, default=100_000)
    parser.add_argument("--force", action="store_true", help="Recompute finished points.")
    args = parser.parse_args()
    if args.M <= 0 or args.batch <= 0 or args.workers < 2:
        parser.error("M and batch must be positive and --workers at least 2")
    return args


def main() -> None:
    args = parse_args()
    for wi in args.wi:
        run_one(args.n, float(wi), args)


if __name__ == "__main__":
    main()
