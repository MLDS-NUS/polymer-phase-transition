"""Extract the Feb-2025 pkl batches (20V_2/40V/50V) with the SHARED extractor.

Same measurement function as every other dataset in this project
(../frame_features.frame_scalars, used for rendered simulation frames and for
the voltage-frames masked PNGs).  The only batch-specific step is the contrast
convention: these pkl frames are stored INVERTED (white background, median 255),
so each frame is flipped to "molecule bright on ~zero background" before the
shared function sees it -- exactly what the retired v1 extractor did.

    python3 extract_pkl_batches.py      -> data/exp_data/cache_voltage_frames/*.npz

Calibrated Wi for this batch (v1 documentation): 20V_2 1.5, 40V 3.5, 50V 3.8.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from frame_features import frame_scalars, NFEAT, FIELDS

DATA = os.path.abspath(os.path.join(HERE, "../../data/exp_data/voltage-frames"))
CACHE = os.path.abspath(os.path.join(HERE, "../../data/exp_data/cache_voltage_frames"))
BATCHES = {"20V_2_Feb25.pkl": ("20V2", 1.5), "40V_Feb25.pkl": ("40V", 3.5),
           "50V_Feb25.pkl": ("50V", 3.8)}


def main():
    os.makedirs(CACHE, exist_ok=True)
    for fn, (volt, wi) in BATCHES.items():
        d = pickle.load(open(os.path.join(DATA, fn), "rb"))
        n = 0
        for key in d:
            out = f"{CACHE}/{volt}__Feb25__{key}.npz"
            if os.path.exists(out):
                n += 1
                continue
            arr = np.asarray(d[key], dtype=np.float32)
            if np.median(arr) > 128:                 # inverted -> molecule bright
                arr = 255.0 - arr
            rows = np.array([frame_scalars(fr) for fr in arr], dtype=np.float64)
            if rows.shape[1] != NFEAT:
                raise SystemExit("feature count mismatch")
            np.savez_compressed(out, frame=np.arange(len(rows)), wi=wi,
                                **{k: rows[:, j] for j, k in enumerate(FIELDS)})
            n += 1
        print(f"{volt}: {n} trajectories (Wi {wi})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
