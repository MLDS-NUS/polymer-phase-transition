"""Extract per-frame features from the EXPERIMENTAL video tars.

Data: data/exp_data/voltage-frames/10V/<session>/VideoN_H500_W800.tar
(HF afgGshsdrh/voltage-frames).  Each tar holds masked/ no_mask/ raw_full/ PNG
frames; only masked/ is read (molecule bright on zero background, 100x500
tracking crop, ~1000 frames per video).

Per-frame scalars come from the SHARED ../frame_features.py (same function the
simulation renderer output goes through); the fold coordinate used downstream
is z3c = z3p - 2*cx (crop-invariant), per-video median-subtracted
(see ../indicators.py).

    python3 extract_features.py     # all complete tars -> data/exp_data/cache_voltage_frames/
"""
import glob
import io
import os
import re
import sys
import tarfile

import numpy as np
from PIL import Image
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from frame_features import frame_scalars, NFEAT

ROOT = os.path.abspath(os.path.join(HERE, "../../data/exp_data/voltage-frames"))
CACHE = os.path.abspath(os.path.join(HERE, "../../data/exp_data/cache_voltage_frames"))
NPROC = 8


def _natnum(name):
    m = re.search(r"(\d+)\.png$", name)
    return int(m.group(1)) if m else -1


def process_tar(path):
    key = os.path.relpath(path, ROOT).replace("/", "__").replace(".tar", "")
    out = f"{CACHE}/{key}.npz"
    if os.path.exists(out):
        return key, -1
    rows, nums = [], []
    with tarfile.open(path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and "/masked/" in m.name]
        members.sort(key=lambda m: _natnum(m.name))
        for m in members:
            try:
                a = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())),
                               dtype=np.float32)
                if a.ndim == 3:
                    a = a.mean(2)
                rows.append(frame_scalars(a)); nums.append(_natnum(m.name))
            except Exception:
                rows.append((np.nan,) * NFEAT); nums.append(_natnum(m.name))
    if not rows:
        return key, 0
    rows = np.array(rows, dtype=np.float64)
    np.savez_compressed(out, theta=rows[:, 0], ext=rows[:, 1], z3p=rows[:, 2],
                        npts=rows[:, 3], aniso=rows[:, 4], border=rows[:, 5],
                        cx=rows[:, 6], cy=rows[:, 7], ncomp=rows[:, 8],
                        frame=np.array(nums))
    return key, len(rows)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    fs = [f for f in sorted(glob.glob(f"{ROOT}/**/*.tar", recursive=True))
          if os.path.getsize(f) > 50e6]
    print(len(fs), "complete tars", flush=True)
    with Pool(NPROC) as pool:
        for key, n in pool.imap_unordered(process_tar, fs):
            print(key, n, flush=True)
    print("DONE", flush=True)
