"""
Monte Carlo integration of

    Z_n(z, f) = E_{theta ~ Unif[0,2pi)^n}[ exp(f * E(theta)) * delta_sigma(z - c_bar) ]

with a Gaussian soft delta of width sigma.  Reusable for any n >= 1.

E(theta) and c_bar(theta) follow the same convention as the grid quadrature:

    r_k = sum_{i=1..k} (cos theta_i, sin theta_i),  r_0 = 0,
    avg = (1/(n+1)) sum_{k=0..n} r_k,
    E   = sum_{k=0..n} [(r_kx - avg_x)^2 - (r_ky - avg_y)^2],
    c_bar = (1/n) sum_{i=1..n} cos theta_i.

Stable log-sum-exp accumulation across batches; multiprocessing across
workers; per-worker BLAS pinned to a single thread.
"""
from __future__ import annotations

import multiprocessing as mp
from time import time

import numpy as np

try:
    from threadpoolctl import threadpool_limits
    _HAS_TPCTL = True
except ImportError:
    _HAS_TPCTL = False


# ----------------------------------------------------------------------
# Worker chunk
# ----------------------------------------------------------------------

def _mc_chunk(args):
    """One worker chunk of M_chunk samples."""
    n, M_chunk, seed, f_arr, z_arr, sigma, batch = args

    if _HAS_TPCTL:
        threadpool_limits(limits=1)

    rng = np.random.default_rng(seed)
    inv2sig2 = 0.5 / (sigma * sigma)
    n_f, n_z = len(f_arr), len(z_arr)

    running_max = np.full((n_f, n_z), -np.inf)
    running_sum = np.zeros((n_f, n_z))

    done = 0
    while done < M_chunk:
        b = min(batch, M_chunk - done)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(b, n))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        rx = np.empty((b, n + 1))
        ry = np.empty((b, n + 1))
        rx[:, 0] = 0.0
        ry[:, 0] = 0.0
        np.cumsum(cos_t, axis=1, out=rx[:, 1:])
        np.cumsum(sin_t, axis=1, out=ry[:, 1:])

        avg_x = rx.mean(axis=1)
        avg_y = ry.mean(axis=1)

        dx = rx - avg_x[:, None]
        dy = ry - avg_y[:, None]
        E = (dx * dx - dy * dy).sum(axis=1)         # (b,)
        c_bar = cos_t.mean(axis=1)                  # (b,)

        diff = z_arr[None, :] - c_bar[:, None]
        gauss = -inv2sig2 * diff * diff             # (b, n_z)

        for fi in range(n_f):
            log_w = f_arr[fi] * E[:, None] + gauss
            batch_max = log_w.max(axis=0)
            batch_sum = np.exp(log_w - batch_max[None, :]).sum(axis=0)

            new_max = np.maximum(running_max[fi], batch_max)
            running_sum[fi] = (running_sum[fi]
                               * np.exp(running_max[fi] - new_max)
                               + batch_sum
                               * np.exp(batch_max - new_max))
            running_max[fi] = new_max

        done += b

    return running_max, running_sum, M_chunk


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def mc_logZ_soft(n, z_values, f_values, sigma=0.05, M=10_000_000,
                 batch=100_000, seed=0, verbose=True, n_workers=None,
                 return_se=False):
    """
    Monte Carlo estimate of log Z_n(z, f) with Gaussian soft-delta weighting.

    Parameters
    ----------
    n : int
        Number of theta variables.
    z_values, f_values : array-like
        Evaluation grid in (z, f).
    sigma : float
        Soft-delta width.
    M : int
        Total number of MC samples.
    batch : int
        Sample chunk size for the inner loop (memory vs overhead).
    seed : int
        Base RNG seed.  Each worker uses seed + 1009*worker_index.
    n_workers : int, optional
        Number of worker processes.  Defaults to min(8, max(1, M // batch)).
    return_se : bool
        If True, also return a worker-level standard-error estimate.

    Returns
    -------
    log_Z : array of shape (n_f, n_z)
        log Z_n(z, f) (not divided by n).
    se : array of shape (n_f, n_z), optional
        Worker-level standard error (only when return_se=True).
    """
    f_arr = np.asarray(f_values, dtype=float)
    z_arr = np.asarray(z_values, dtype=float)
    n_f, n_z = len(f_arr), len(z_arr)

    if n_workers is None:
        n_workers = min(8, max(1, M // max(batch, 1)))
    n_workers = max(1, n_workers)

    M_per = (M + n_workers - 1) // n_workers
    actual_M = M_per * n_workers

    tasks = [(n, M_per, seed + 1009 * k, f_arr, z_arr, sigma, batch)
             for k in range(n_workers)]

    if verbose:
        print(f"    MC n={n}: M={actual_M:,}  workers={n_workers}  "
              f"batch={batch}  sigma={sigma}", flush=True)

    t_start = time()
    if n_workers == 1:
        results = [_mc_chunk(tasks[0])]
    else:
        with mp.get_context('fork').Pool(n_workers) as pool:
            results = []
            for done, r in enumerate(
                    pool.imap_unordered(_mc_chunk, tasks), 1):
                results.append(r)
                if verbose:
                    el = time() - t_start
                    print(f"      worker {done}/{n_workers} done  "
                          f"elapsed={el:6.1f}s", flush=True)

    all_max = np.stack([r[0] for r in results], axis=0)
    all_sum = np.stack([r[1] for r in results], axis=0)
    all_M = np.array([r[2] for r in results], dtype=float)

    global_max = all_max.max(axis=0)
    total_sum = (all_sum
                 * np.exp(all_max - global_max[None])).sum(axis=0)

    const = -np.log(actual_M) - 0.5 * np.log(2.0 * np.pi) - np.log(sigma)
    log_Z = global_max + np.log(total_sum) + const

    if not return_se:
        return log_Z

    log_w_per = (all_max + np.log(all_sum)
                 - np.log(all_M)[:, None, None]
                 - 0.5 * np.log(2.0 * np.pi) - np.log(sigma))
    se = log_w_per.std(axis=0, ddof=1) / np.sqrt(len(results))
    return log_Z, se


# ----------------------------------------------------------------------
# Bessel reference (f = 0): exact density of c_bar via characteristic fn
# ----------------------------------------------------------------------

def analytical_logZ_f0(n, z_grid):
    """
    log p_z(z) at f=0, where p_z(z) = (n/pi) int_0^inf J_0(t)^n cos(n t z) dt.
    """
    from scipy.special import j0
    from scipy.integrate import quad

    log_pz = np.zeros(len(z_grid))
    for i, z in enumerate(z_grid):
        x = n * z
        val, _ = quad(lambda t: j0(t) ** n * np.cos(t * x),
                      0, np.inf, limit=500)
        p_S = val / np.pi
        p_z = n * max(p_S, 1e-300)
        log_pz[i] = np.log(p_z)
    return log_pz
