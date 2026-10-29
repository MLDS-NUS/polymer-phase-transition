"""Stage 1: pair the reduced coordinates (Z2, Z3) with physical observables.

Counterpart of the last cells of
`examples/polymer_dynamics/polymer_dynamics_analysis.ipynb`, which encode the
training frames and compare |Z2| with the end-to-end distance and |Z3| with the
foldedness.  Two differences here:

* the data span many Wi (11145 startup trajectories, one Wi each, Wi in
  [0, 10.8]) instead of a single Wi, so Wi is stored for every frame;
* the data are far larger (11145 x 5001 x 900 float32, ~200 GB), so the raw
  HF arrow shards are opened zero-copy and only every STRIDE-th frame is
  read from disk.

The encoder is the symmetry-constrained PCA of `../generate_reduced_data.py`
(`pca_components.pkl`): Z2 = v1.x, Z3 = v2.x, whitened.  It is exactly the map
that produced the training data of the OnsagerNet models (checked against
`reduced_symm_new_train_transformed`, max |diff| ~ 6e-7).

Per kept frame (bead positions r_i = (x_i, y_i, z_i), i = 1..300, COM = 0):
    z1       = (max_i x_i - min_i x_i) / 300           (same as the training data)
    Z2, Z3   = whitened PCA coordinates
    ete_x    = x_1 - x_300                             end-to-end distance along x
                                                       (sign chosen so that ete_x ~ +Z2)
    fold_x   = x_1 + x_300 - 2 x_com                   foldedness along x
    ete_3d   = |r_300 - r_1|

    python extract_observables.py                  # STRIDE=50 -> 101 frames/traj
    STRIDE=25 NPROC=16 python extract_observables.py

Writes cache/observables_stride<STRIDE>.npz.
"""
import glob
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from time import time

# pca_components.pkl stores jax Arrays; unpickling imports jax -> keep it on CPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

RAW_DIR = os.environ.get(
    "RAW_DIR",
    os.path.expanduser("~/nfs/hf_cache/datasets/MLDS-NUS___wis_param_sweep_new/default/0.0.0"),
)
PCA_FILE = os.environ.get(
    "PCA_FILE",
    "/home/aiqing/PhaseDiagram_save1/examples/UIDD_polymer_dynamics_wi/data_cache_nolog/pca_components.pkl",
)
SPLITS = ("train", "train_slow", "train_low")    # = config.data.reduction.train_splits
STRIDE = int(os.environ.get("STRIDE", "50"))
NPROC = int(os.environ.get("NPROC", "16"))
N_BEADS = 300
N_FRAMES = 5001
DIM = 3 * N_BEADS
FIELDS = ("z1", "Z2", "Z3", "ete_x", "fold_x", "ete_3d", "t")

_P = None   # (2, 900) PCA rows scaled by the whitening factors
_MU = None  # (900,)


def load_pca():
    with open(PCA_FILE, "rb") as f:
        comp = pickle.load(f)
    P = np.asarray(comp["P"], dtype=np.float64)
    ws = np.asarray(comp["whitening_scale"], dtype=np.float64)
    mu = np.asarray(comp["mu"], dtype=np.float64).ravel()
    return P * ws[:, None], mu


def _init_worker():
    global _P, _MU
    _P, _MU = load_pca()


def shard_files():
    snap = sorted(glob.glob(os.path.join(RAW_DIR, "*")))
    snap = [s for s in snap if os.path.isdir(s)]
    assert len(snap) == 1, f"expected one snapshot under {RAW_DIR}, got {snap}"
    jobs = []
    for s_idx, split in enumerate(SPLITS):
        fs = sorted(glob.glob(os.path.join(snap[0], f"wis_param_sweep_new-{split}-*.arrow")))
        assert fs, f"no shards for split {split}"
        jobs += [(s_idx, k, p) for k, p in enumerate(fs)]
    return jobs


def flat_values(batch, name):
    """Zero-copy flat float32 view of an Array2D column and its row offsets.

    (Do not combine_chunks(): that copies the whole ~500 MB column.)
    """
    col = batch.column(name)
    mid = col.values
    vals = mid.values.to_numpy(zero_copy_only=True)
    row_start = mid.offsets.to_numpy()[col.offsets.to_numpy()[:-1]]
    return vals, row_start


def process_row(x, t, out):
    x = x.astype(np.float64)
    bx = x[:, 0::3]
    z = (x - _MU) @ _P.T
    out["z1"].append((bx.max(1) - bx.min(1)) / N_BEADS)
    out["Z2"].append(z[:, 0])
    out["Z3"].append(z[:, 1])
    out["ete_x"].append(bx[:, 0] - bx[:, -1])
    out["fold_x"].append(bx[:, 0] + bx[:, -1] - 2.0 * bx.mean(1))
    ete = x[:, -3:] - x[:, :3]
    out["ete_3d"].append(np.linalg.norm(ete, axis=1))
    out["t"].append(t)


def do_shard(job):
    """Read every STRIDE-th frame of every trajectory in one arrow shard.

    The record batches are zero-copy views of the memory map, which gives the
    file offset of the x values; the kept frames are then fetched with pread
    (~2x faster on NFS than page-faulting the map, which triggers readahead).
    """
    s_idx, k, path = job
    frames = np.arange(0, N_FRAMES, STRIDE)
    nbytes = DIM * 4
    out = {f: [] for f in FIELDS}
    wi = []
    mm = pa.memory_map(path)
    base = mm.read_buffer(1).address
    mm.seek(0)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    try:
        for batch in pa.ipc.open_stream(mm):
            leaf = batch.column("x").values.values
            assert leaf.type == pa.float32() and leaf.null_count == 0
            x_off = leaf.buffers()[1].address - base + 4 * leaf.offset
            _, x0 = flat_values(batch, "x")
            av, a0 = flat_values(batch, "args")
            tv, t0 = flat_values(batch, "t")
            for i in range(batch.num_rows):
                offs = x_off + 4 * (x0[i] + frames * DIM)
                x = np.stack([np.frombuffer(os.pread(fd, nbytes, int(o)), np.float32)
                              for o in offs])
                process_row(x, tv[t0[i] + frames], out)
                wi.append(float(av[a0[i] + 1]))          # args = (T, Wi), constant in t
    finally:
        os.close(fd)
    res = {f: np.stack(v).astype(np.float32) for f, v in out.items()}   # (rows, frames)
    res["Wi"] = np.array(wi, dtype=np.float32)
    res["split"] = np.full(len(wi), s_idx, dtype=np.int8)
    res["shard"] = np.full(len(wi), k, dtype=np.int16)
    res["row"] = np.arange(len(wi), dtype=np.int16)
    return res


def main():
    CACHE.mkdir(exist_ok=True)
    out_file = CACHE / f"observables_stride{STRIDE}.npz"
    if out_file.exists():
        print(f"{out_file} exists, nothing to do")
        return
    jobs = shard_files()
    print(f"{len(jobs)} shards, stride {STRIDE} -> "
          f"{len(range(0, N_FRAMES, STRIDE))} frames/traj, {NPROC} workers", flush=True)
    t_start = time()
    parts = []
    with Pool(NPROC, initializer=_init_worker) as pool:
        for n, res in enumerate(pool.imap(do_shard, jobs), 1):
            parts.append(res)
            if n % 20 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} shards  {time() - t_start:.0f}s", flush=True)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    data["frame"] = np.arange(0, N_FRAMES, STRIDE, dtype=np.int16)
    data["splits"] = np.array(SPLITS)
    np.savez(out_file, **data)
    n_traj, n_fr = data["Z2"].shape
    print(f"saved {out_file}: {n_traj} traj x {n_fr} frames, "
          f"Wi in [{data['Wi'].min():.3g}, {data['Wi'].max():.3g}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    main()
