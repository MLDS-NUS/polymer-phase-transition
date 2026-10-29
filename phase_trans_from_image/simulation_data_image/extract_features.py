"""Extract IMAGE features from the SIMULATION datasets into data/sim_cache/.

The simulation is observed exactly like the experiment: every frame is
RENDERED to a microscope image (render.py: Gibson-Lanni PSF, shot noise,
8-bit, 100x500 tracking crop) and measured by the SHARED
../frame_features.frame_scalars.  Bead coordinates are never used for the
image-side features; the only labeled quantity kept is `endsum`
(x_first + x_last - 2*com, TRUTH CONTROL only).

Per trajectory (subsampled by SUB, dt = 5e-4*SUB): Wi + per-frame
theta/ext/z3p/npts/aniso/border/cx/cy/ncomp (image) + endsum (truth).

    python3 extract_features.py paramsweep_new        # main set  -> imgfeat_all/
    python3 extract_features.py holdout_new sweep_old changebead
    python3 extract_features.py truth                 # reduced (z1, Z3) series
                                                      # -> reduced_series.npz

`truth` needs the reduced dataset on disk (REDUCED_ROOT) and the `datasets`
package; the render modes need pyarrow + scipy (and calibrate the renderer
automatically on first use).
"""
import glob
import os
import sys
import zlib

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
CACHE = os.path.abspath(os.path.join(HERE, "../../data/sim_cache"))
HF_GLOB = os.path.expanduser("~/nfs/hf_cache/hub/datasets--MLDS-NUS--{repo}/snapshots/*/data/*.parquet")
REDUCED_ROOT = os.path.abspath(os.path.join(HERE, "../../data/reduced"))
SUB = 5
NPROC = 30

DATASETS = {
    "paramsweep_new": ("Wis_ParamSweep_New", "imgfeat_all"),
    "holdout_new":    ("Wis_Holdout_New", "imgfeat_holdout_new"),
    "sweep_old":      ("Wis_ParamSweep", "imgfeat_sweep_old"),
    "changebead":     ("Wis_ParamSweep_ChangeBead", "imgfeat_changebead"),
}

_R = None  # per-worker renderer


def _init_worker():
    global _R
    import render
    from frame_features import frame_scalars  # noqa: F401  (fail fast if missing)
    _R = render.Renderer()


def detect_layout(frame, nb):
    """'inter' for (x,y,z),(x,y,z),... rows, 'block' for [all x][all y][all z].

    The FENE bond length is (nearly) constant along the chain — the correct
    layout is the one whose consecutive-bead bond lengths have the smaller
    coefficient of variation (the wrong one scrambles coordinates).
    """
    inter = frame.reshape(nb, 3)
    block = frame.reshape(3, nb).T
    cv = []
    for c in (inter, block):
        b = np.linalg.norm(np.diff(c, axis=0), axis=1)
        cv.append(b.std() / (b.mean() + 1e-12))
    return "inter" if cv[0] <= cv[1] else "block"


def to_beads(x_sub, nb, layout):
    """(n_frames, nb, 3) bead coordinates from raw frame rows."""
    if layout == "inter":
        return x_sub.reshape(-1, nb, 3)
    return x_sub.reshape(-1, 3, nb).transpose(0, 2, 1)


def native_scale(name, nshard=12):
    """(scale px/unit, layout) for this dataset.

    Datasets are simulations of the SAME physical polymer, but native units
    and column layout differ (ChangeBead: doubled bonds, block layout).  The
    virtual microscope maps the physical chain to the historical px
    convention (L_max ~ 278 px): s = 278 / native L_max, with L_max estimated
    exactly like everywhere else (ext p99 pooled over the top-Wi decile).
    Datasets already in the px convention (s within 10% of 1) keep s = 1.
    """
    import pyarrow.parquet as pq
    repo, _ = DATASETS[name]
    fs = sorted(glob.glob(HF_GLOB.format(repo=repo)))
    rng = np.random.default_rng(0)
    pick = fs if len(fs) <= nshard else sorted(rng.choice(fs, nshard, replace=False))
    Wi, e99, layout = [], [], None
    for path in pick:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4):
            xcol = batch.column("x"); acol = batch.column("args")
            for i in range(len(xcol)):
                x = np.array(xcol[i].as_py(), dtype=np.float32)
                nb = x.shape[1] // 3
                if layout is None:
                    layout = detect_layout(x[len(x) // 2], nb)
                bx = to_beads(x[::SUB], nb, layout)[:, :, 0]
                Wi.append(float(acol[i][0][1].as_py()))
                e99.append(float(np.percentile(bx.max(1) - bx.min(1), 99)))
    Wi, e99 = np.array(Wi), np.array(e99)
    lmax = np.percentile(e99[Wi >= np.percentile(Wi, 90)], 99)
    s = 278.0 / lmax
    if 0.9 < s < 1.1:
        s = 1.0
    print(f"{name}: layout {layout}, native L_max ~ {lmax:.0f} units "
          f"-> scale {s:.3f} px/unit", flush=True)
    return s, layout


def do_shard(job):
    import pyarrow.parquet as pq
    from frame_features import frame_scalars, NFEAT, FIELDS
    path, outdir, scale, layout = job
    tag = os.path.basename(path).replace(".parquet", "")
    out = f"{outdir}/{tag}.npz"
    if os.path.exists(out):
        return tag
    pf = pq.ParquetFile(path)
    Wis, feats, ess = [], [], []
    row = 0
    for batch in pf.iter_batches(batch_size=4):
        xcol = batch.column("x"); acol = batch.column("args")
        for i in range(len(xcol)):
            x = np.array(xcol[i].as_py(), dtype=np.float32)
            nb = x.shape[1] // 3
            wi = float(acol[i][0][1].as_py())
            xs = to_beads(x[::SUB], nb, layout) * np.float32(scale)
            rng = np.random.default_rng(zlib.crc32(f"{tag}:{row}".encode()))
            rows = np.array([frame_scalars(_R.render(c, rng).astype(np.float32))
                             for c in xs], dtype=np.float32)
            assert rows.shape[1] == NFEAT
            bx = xs[:, :, 0]; com = bx.mean(1)
            Wis.append(wi)
            feats.append(rows)
            ess.append((bx[:, 0] + bx[:, -1] - 2 * com).astype(np.float32))
            row += 1
    feats = np.stack(feats)  # (n_traj, n_frames, NFEAT)
    tmp = f"{out}.tmp-{os.getpid()}.npz"
    np.savez_compressed(tmp, Wi=np.array(Wis, np.float32),
                        endsum=np.stack(ess),
                        **{k: feats[:, :, j] for j, k in enumerate(FIELDS)})
    os.replace(tmp, out)     # atomic: a killed worker never leaves a bad file
    return tag


def extract_parquet(name):
    import render
    render.calibrate()          # once, in the parent, before forking workers
    repo, sub = DATASETS[name]
    outdir = f"{CACHE}/{sub}"
    os.makedirs(outdir, exist_ok=True)
    fs = sorted(glob.glob(HF_GLOB.format(repo=repo)))
    todo = [f for f in fs
            if not os.path.exists(f"{outdir}/{os.path.basename(f)[:-8]}.npz")]
    scale, layout = native_scale(name) if todo else (1.0, "inter")
    print(f"{name}: {len(fs)} shards ({len(todo)} to do) -> {outdir}", flush=True)
    with Pool(NPROC, initializer=_init_worker) as p:
        for j, _ in enumerate(p.imap_unordered(do_shard,
                                               [(f, outdir, scale, layout) for f in fs])):
            if (j + 1) % 20 == 0:
                print(f"  {j+1}/{len(fs)}", flush=True)
    print(f"{name} DONE", flush=True)


def extract_truth():
    """Reduced (z1 -> ext px, Z2, Z3) series for the truth control."""
    from datasets import load_from_disk
    ds = load_from_disk(f"{REDUCED_ROOT}/reduced_symm_new_train_transformed").with_format("numpy")
    W, E, Z2, Z3 = [], [], [], []
    for batch in ds.iter(batch_size=64):
        x = np.asarray(batch["x"]); a = np.asarray(batch["args"])
        W.append(a[:, 0, 1].astype(np.float32))
        E.append((x[:, ::SUB, 0] * 300.0).astype(np.float32))
        Z2.append(x[:, ::SUB, 1].astype(np.float32))
        Z3.append(x[:, ::SUB, 2].astype(np.float32))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(f"{CACHE}/reduced_series.npz", Wi=np.concatenate(W),
                        ext=np.concatenate(E), Z2=np.concatenate(Z2),
                        Z3=np.concatenate(Z3))
    print("truth DONE ->", f"{CACHE}/reduced_series.npz", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["paramsweep_new"]
    for a in args:
        if a == "truth":
            extract_truth()
        elif a in DATASETS:
            extract_parquet(a)
        else:
            raise SystemExit(f"unknown dataset {a}; choose {list(DATASETS)+['truth']}")
