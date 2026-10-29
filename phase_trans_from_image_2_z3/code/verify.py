"""Experimental data loader used by analysis_stats.py.

Each experimental video goes through the same per-frame features
(../phase_trans_from_image/frame_features.py) and the same fold coordinate and
frame-quality mask (../phase_trans_from_image/indicators.py) as the simulation.
"""
import os, sys
_BASE = os.path.dirname(os.path.abspath(__file__))     # .../phase_trans_from_image_2_z3/code
BASE = os.path.abspath(os.path.join(_BASE, ".."))      # the project folder
REPO = os.path.abspath(os.path.join(BASE, ".."))       # the repository root
sys.path.insert(0, os.path.join(REPO, "phase_trans_from_image"))
import glob

import numpy as np

import indicators as I


EXP = os.path.abspath(os.path.join(REPO, "data/exp_data/cache_voltage_frames"))
# Two imaging batches; L_max is estimated INSIDE each batch (pooling across
# batches would be wrong: imaging differs).  It comes only from the groups the
# final result uses: tar from 10V itself (139 px), Feb-2025 from 50V (303 px).
EXP_BATCH = {
    # NOTE: the nominal voltage->Wi calibration is NOT reliable and must not be used
    # to label or order the experimental groups; refer to groups by voltage only.
    "tar":   dict(volts=["10V"], lmax_from="10V"),
    "feb25": dict(volts=["20V2", "40V", "50V"], lmax_from="50V"),
}


def load_exp(batch="tar"):
    """Experimental videos of one imaging batch, QC-rejected frames set NaN.

    Returns per-video voltage labels, extension and fold-coordinate series, and
    the batch L_max (99th percentile of ext over the `lmax_from` group).
    """
    cfg = EXP_BATCH[batch]
    lmax = I.estimate_lmax([np.load(f)["ext"] for f in
                            sorted(glob.glob(f"{EXP}/{cfg['lmax_from']}__*.npz"))])
    lab, ext, z3c = [], [], []
    for v in cfg["volts"]:
        for f in sorted(glob.glob(f"{EXP}/{v}__*.npz")):
            d = np.load(f)
            if "ext" not in d:
                continue
            e = d["ext"].astype(float).copy()
            ok = I.qc_mask(e, d["border"], lmax)
            z = I.fold_coordinate(d["z3p"], d["cx"], ok).astype(float)
            e[~ok] = np.nan; z[~ok] = np.nan
            lab.append(v); ext.append(e); z3c.append(z)
    return np.array(lab), ext, z3c, lmax
