"""Shared, deterministic definitions for the joint-:math:`z_3` Wi sweep.

This directory is intentionally self-contained.  In particular, no consumer
needs to import the historical beta sweeps, whose grids and output contracts
are different.  Both the continuum and finite-:math:`n` runners import this
module so they cannot silently disagree about the Wi--beta conversion or the
340 requested sample locations.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# The conversion stated in the request: Wi = beta * 4321 / 45000.
WI_PER_BETA = 4321.0 / 45000.0
BETA_PER_WI = 45000.0 / 4321.0

# A fixed positive half of the z3 proposal anchors.  The z2=0 problem is even
# in z3; the MC mixture adds cyclic shifts and uses these six independently
# optimized continuum slope profiles.  Keeping the endpoint below 1/2 avoids
# the geometric boundary, while covering the requested MC z3 range.
ANCHOR_Z3 = np.linspace(0.0, 0.49, 6, dtype=float)


def make_wi_grid() -> np.ndarray:
    """Return exactly the 340-point Wi grid on [0.02, 10].

    It is built from the earlier 440-point grid, whose labels were half the
    present ones (Wi = beta*4321/45000, beta unchanged):

    * 0.01, then 0.02 .. 3.00 in increments of 0.01, then
    * 3.05 .. 10.00 in increments of 0.05,

    doubled and cut at Wi = 10, i.e. 0.02, 0.04 .. 6.00 (step 0.02) and
    6.10 .. 10.00 (step 0.1), rounded at six decimals.  Do not replace this
    with a geometrically spaced approximation: point indices are a stable
    resume/MC/plotting contract.
    """
    wi = np.concatenate(
        (
            np.array([0.01], dtype=float),
            np.arange(0.02, 3.0001, 0.01, dtype=float),
            np.arange(3.05, 10.0001, 0.05, dtype=float),
        )
    )
    wi = np.unique(np.round(2.0 * wi, 6))
    wi = wi[wi <= 10.0 + 1e-12]
    expected = 340
    if (
        wi.shape != (expected,)
        or not np.isclose(wi[0], 0.02, rtol=0.0, atol=0.0)
        or not np.isclose(wi[-1], 10.0, rtol=0.0, atol=0.0)
        or not np.all(np.diff(wi) > 0.0)
    ):
        raise RuntimeError("The fixed Wi grid no longer has its 340-point contract.")
    return wi


def beta_from_wi(wi: np.ndarray | float) -> np.ndarray | float:
    """Convert Wi to beta without rounding away the requested sample point."""
    values = np.asarray(wi, dtype=float) * BETA_PER_WI
    return float(values) if np.ndim(values) == 0 else values


def wi_from_beta(beta: np.ndarray | float) -> np.ndarray | float:
    """Convert beta to Wi without rounding away the requested sample point."""
    values = np.asarray(beta, dtype=float) * WI_PER_BETA
    return float(values) if np.ndim(values) == 0 else values


def canonical_json(payload: dict[str, Any]) -> str:
    """Encode a configuration deterministically for an on-disk fingerprint."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(payload: dict[str, Any]) -> str:
    """SHA-256 fingerprint used to reject incompatible resume checkpoints."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync after rename (useful on local/NFS filesystems)."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_savez(path: Path, **items: Any) -> None:
    """Publish a compressed NPZ only after its complete contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **items)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 status/progress/configuration text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

