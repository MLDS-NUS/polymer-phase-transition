"""Fig. S8 — image versus physical fold coordinate, styled after
examples/UIDD_polymer_dynamics_wi/z2-z3-Interpret/z2_z3_vs_physical_by_wi.png

  one row: x_1 + x_N - 2 x_com vs Z_{3,c} (image)
  columns: four Wi ranges; purple scatter points

Frames are paired EXACTLY, not by index: the reduced observables carry
(split, shard, row) and are stored every 50th of 5001 steps, while the image
features are stored every 5th, so cache frame j is image frame 10j.

    python3 fig_S8.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(BASE, ".."))
FIG_SUPP = os.path.join(BASE, "figures", "supplement")
RESULTS = os.path.join(BASE, "results")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = os.path.join(REPO, "data", "sim_cache", "imgfeat_all")
OBS = os.path.join(REPO, "examples", "UIDD_polymer_dynamics_wi", "z2-z3-Interpret",
                   "cache", "observables_stride50.npz")
NSHARD = {"train": 180, "train_slow": 180, "train_low": 46}
WI_BINS = [(0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.9)]
NDRAW = 15000
rng = np.random.default_rng(0)

d = np.load(OBS)
SPL = [str(x) for x in d["splits"]]
Z3, FOLD, Wi = d["Z3"], d["fold_x"], d["Wi"]
sp, sh, rw = d["split"], d["shard"], d["row"]

print("pairing image frames to reduced observables (exact, via split/shard/row) ...",
      flush=True)
cache = {}


def shard_of(split, shard):
    k = (split, shard)
    if k not in cache:
        f = os.path.join(IMG, f"{split}-{shard:05d}-of-{NSHARD[split]:05d}.npz")
        cache[k] = np.load(f) if os.path.exists(f) else None
        if len(cache) > 40:
            cache.pop(next(iter(cache)))
    return cache[k]


ZC, Z3o, FO, WW, TRAJ = [], [], [], [], []
idx = np.arange(Z3.shape[1])*10                 # cache frame j  ->  image frame 10j
for i in range(len(Wi)):
    a = shard_of(SPL[sp[i]], int(sh[i]))
    if a is None:
        continue
    r = int(rw[i])
    if r >= a["ext"].shape[0] or idx.max() >= a["ext"].shape[1]:
        continue
    zz = a["z3p"][r].astype(float) - 2*a["cx"][r].astype(float)
    g = np.isfinite(zz)
    if g.sum() < 50:
        continue
    zc = (zz - np.median(zz[g]))[idx]
    m = np.isfinite(zc) & np.isfinite(Z3[i]) & np.isfinite(FOLD[i])
    if m.sum() < 20:
        continue
    ZC.append(zc[m]); Z3o.append(Z3[i][m]); FO.append(FOLD[i][m])
    WW.append(np.full(int(m.sum()), Wi[i]))
    TRAJ.append(np.full(int(m.sum()), i))
ZC = np.concatenate(ZC); Z3o = np.concatenate(Z3o); FO = np.concatenate(FO)
WW = np.concatenate(WW); TRAJ = np.concatenate(TRAJ)
print(f"  {len(np.unique(TRAJ))} trajectories, {len(ZC)} frames", flush=True)

sl_z3 = np.polyfit(Z3o, ZC, 1)[0]               # global slopes, drawn in every panel
sl_fo = np.polyfit(FO, ZC, 1)[0]

# Keep both sets of statistics for downstream use; plot only the physical row.
summary = []
for ri, X in enumerate((Z3o, FO)):
    for wlo, whi in WI_BINS:
        mask = (WW >= wlo) & (WW < whi)
        summary.append((ri, wlo, whi, float(np.corrcoef(X[mask], ZC[mask])[0, 1]),
                        len(np.unique(TRAJ[mask])), int(mask.sum())))

C_PROXY = "#9467bd"
C_REF = "#52514e"
fig, axes = plt.subplots(1, len(WI_BINS), figsize=(28, 7.5),
                         layout="constrained", sharex=True, sharey=True)
xlo, xhi = np.percentile(FO, [0.2, 99.8])
ylo, yhi = np.percentile(ZC, [0.2, 99.8])
for ci, (wlo, whi) in enumerate(WI_BINS):
    ax = axes[ci]
    mask = (WW >= wlo) & (WW < whi)
    r_ = float(np.corrcoef(FO[mask], ZC[mask])[0, 1])
    j = np.flatnonzero(mask)
    if len(j) > NDRAW:
        j = rng.choice(j, NDRAW, replace=False)
    ax.scatter(FO[j], ZC[j], s=15, color=C_PROXY, alpha=0.35, lw=0,
               rasterized=True)
    xs = np.array([xlo, xhi])
    ax.plot(xs, sl_fo * xs, "--", color=C_REF, lw=1.2)
    ax.text(0.04, 0.96, f"$r = {r_:.3f}$", transform=ax.transAxes,
            fontsize=52, va="top")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    last = ci == len(WI_BINS) - 1
    rel = r"\leq" if last else "<"
    upper = 10.8 if last else whi
    ax.set_title(rf"${wlo:g} \leq Wi {rel} {upper:g}$", fontsize=48)
    ax.set_xlabel(r"$x_1 + x_N - 2x_{\rm com}$", fontsize=48)
    ax.tick_params(labelsize=40)
axes[0].set_ylabel(r"$\widetilde{Z}_3$ (image, px)", fontsize=48, labelpad=15)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_SUPP, f"fig_S8_proxy_fidelity.{e}"), dpi=150,
                bbox_inches="tight")
plt.close(fig)
print("wrote fig_S8_proxy_fidelity.png / .pdf", flush=True)

print()
print(f"  global: corr(Z3, Z3c) = {np.corrcoef(Z3o, ZC)[0, 1]:.3f}   "
      f"corr(fold_x, Z3c) = {np.corrcoef(FO, ZC)[0, 1]:.3f}   "
      f"corr(Z3, fold_x) = {np.corrcoef(Z3o, FO)[0, 1]:.3f}")
for ri, wlo, whi, r_, nt, n_ in summary:
    tag = "Z3    " if ri == 0 else "fold_x"
    print(f"  {tag} vs Z3c   Wi {wlo:4.1f}-{whi:4.1f}   r = {r_:+.3f}   "
          f"{nt:5d} traj  {n_:8d} frames")
np.savez(os.path.join(RESULTS, "stats_proxy.npz"),
         r_global_Z3=np.corrcoef(Z3o, ZC)[0, 1],
         r_global_fold=np.corrcoef(FO, ZC)[0, 1],
         r_Z3_fold=np.corrcoef(Z3o, FO)[0, 1],
         slope_Z3=sl_z3, slope_fold=sl_fo,
         summary=np.array(summary, float))
print("\nwrote results/stats_proxy.npz")
print("S8_DONE")
