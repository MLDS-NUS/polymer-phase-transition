"""Trajectory-level bootstrap statistics and robustness tests for the experimental
drift curves, plus the raw / odd / even decomposition.

Everything resamples TRAJECTORIES (not frame pairs): pairs inside one trajectory are
strongly correlated, so resampling pairs would badly understate the uncertainty.

Outputs
    stats_curves.npz      per-group: bin centres, raw f, odd g, even e, per-bin
                          bootstrap percentiles for each of them
    stats_slope40.npz     the full bootstrap distribution of the 40V origin slope
                          and of its outer zero position
    stats_robust40.csv    robustness grid for 40V (bin count, extension window,
                          lag-selection target)
    printed summary

    python3 analysis_stats.py
"""
import os, sys
_BASE = os.path.dirname(os.path.abspath(__file__))     # .../phase_trans_from_image_2_z3/code
BASE = os.path.abspath(os.path.join(_BASE, ".."))      # the project folder
REPO = os.path.abspath(os.path.join(BASE, ".."))       # the repository root
FIG_MAIN = os.path.join(BASE, "figures", "main")
FIG_SUPP = os.path.join(BASE, "figures", "supplement")
FIG_PANE = os.path.join(BASE, "figures", "panels")
RESULTS = os.path.join(BASE, "results")
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(REPO, "phase_trans_from_image"))
import csv
import itertools

import numpy as np

import verify as V
import indicators as I

GROUPS = [("tar", "10V"), ("feb25", "20V2"), ("feb25", "40V"), ("feb25", "50V")]
LABEL = {"10V": "10V", "20V2": "20V", "40V": "40V", "50V": "50V"}

BAND = tuple(float(x) for x in os.environ.get("BAND", "0.30,0.55").split(","))
TAG = os.environ.get("TAG", "")          # 输出文件后缀，便于并存多个带的结果
TARGET = 0.051
NBIN = 9
NBOOT = 2000
MIN_PAIRS_BIN = 150
MIN_PAIRS_TOT = 800
rng = np.random.default_rng(20250922)


# ----------------------------------------------------------------- extraction
def grab(ext, z3c, ks, lmax, band, l1, l2):
    """Per-trajectory (x0, dx) lists plus the mean in-lag dilation r."""
    lo, hi = band[0]*lmax, band[1]*lmax
    tend = I.T_END_FRAC*lmax
    P, R = [], []
    for k in ks:
        e, z = ext[k], z3c[k]
        te = np.where(e > tend)[0]
        te = te[0] if len(te) else len(e)
        sel = np.where((e[:max(te-l2, 0)] >= lo) & (e[:max(te-l2, 0)] < hi))[0]
        if not len(sel):
            continue
        a = z[sel]
        b = z[sel+l2] - z[sel+l1]
        r = (e[sel+l2] - e[sel+l1]) / np.maximum(e[sel], 1.0)
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(r)
        if m.any():
            P.append((a[m], b[m])); R.append(r[m])
    if not P:
        return None, np.nan
    return P, float(np.mean(np.concatenate(R)))


def three_parts(P, edges):
    """Binned mean displacement f, its odd part g and its even part e."""
    x0 = np.concatenate([q[0] for q in P])
    dx = np.concatenate([q[1] for q in P])
    if len(x0) < MIN_PAIRS_TOT:
        return None
    x0 = x0 - x0.mean()
    mids = 0.5*(edges[1:] + edges[:-1])
    f = np.full(len(edges)-1, np.nan)
    for i in range(len(edges)-1):
        m = (x0 >= edges[i]) & (x0 < edges[i+1])
        if m.sum() >= MIN_PAIRS_BIN:
            f[i] = dx[m].mean()
    g = 0.5*(f - f[::-1])          # odd  (the indicator)
    e = 0.5*(f + f[::-1])          # even (must be consistent with zero)
    return mids, f, g, e


def edges_for(P, nbin=NBIN, half=2.6):
    x0 = np.concatenate([q[0] for q in P])
    sc = np.nanstd(x0)
    return np.linspace(-half*sc, half*sc, nbin), sc


def origin_slope(mids, g, w):
    m = np.isfinite(g) & (np.abs(mids) <= w)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(mids[m], g[m], 1)[0]


def outer_zero(mids, g):
    """First zero crossing on x>0; returns (position, kind) or (nan, '')."""
    idx = np.where((mids > 0) & np.isfinite(g))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if g[a] == 0 or np.sign(g[a]) == np.sign(g[b]):
            continue
        t = g[a]/(g[a] - g[b])
        return mids[a] + t*(mids[b] - mids[a]), ("stable" if g[a] > 0 else "unstable")
    return np.nan, ""


def pick_lag(ext, z3c, ks, lmax, band, target):
    best = None
    for l2 in range(2, 60):
        l1 = max(1, int(round(l2/3)))
        if l1 >= l2:
            continue
        P, r = grab(ext, z3c, ks, lmax, band, l1, l2)
        if P is None or not np.isfinite(r):
            continue
        ed, _ = edges_for(P)
        if three_parts(P, ed) is None:
            continue
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (P, r, l1, l2)
    return best


# ----------------------------------------------------------------- 1. curves
print("=" * 74)
print(f"1.  per-bin trajectory bootstrap ({NBOOT} resamples)   BAND = {BAND}")
print("=" * 74)

store, cache = {}, {}
for batch in ("tar", "feb25"):
    lab, ext, z3c, lmax = V.load_exp(batch)
    cache[batch] = (lab, ext, z3c, lmax)

out = {}
for batch, v in GROUPS:
    lab, ext, z3c, lmax = cache[batch]
    ks = np.where(lab == v)[0]
    P, r, l1, l2 = pick_lag(ext, z3c, ks, lmax, BAND, TARGET)
    ed, sc = edges_for(P)
    mids, f, g, e = three_parts(P, ed)
    sl = origin_slope(mids, g, sc)

    F, G, E, S = [], [], [], []
    for _ in range(NBOOT):
        j = rng.integers(0, len(P), len(P))
        t = three_parts([P[q] for q in j], ed)
        if t is None:
            continue
        F.append(t[1]); G.append(t[2]); E.append(t[3])
        S.append(origin_slope(t[0], t[2], sc))
    F, G, E = np.array(F, float), np.array(G, float), np.array(E, float)
    S = np.array(S, float)

    pc = lambda A, q: np.nanpercentile(A, q, axis=0)
    store[v] = dict(mids=mids, f=f, g=g, e=e, sc=sc, r=r, l1=l1, l2=l2,
                    ntraj=len(P), npairs=int(np.isfinite(np.concatenate(
                        [q[0] for q in P])).sum()),
                    g_lo=pc(G, 16), g_hi=pc(G, 84),
                    g_lo95=pc(G, 2.5), g_hi95=pc(G, 97.5),
                    f_lo=pc(F, 16), f_hi=pc(F, 84),
                    e_lo=pc(E, 16), e_hi=pc(E, 84),
                    slope=sl, slope_boot=S)
    nfin = np.isfinite(S).sum()
    print(f"  {LABEL[v]:>4}  lag({l1},{l2})  r={r:+.3f}  sigma={sc:5.2f}px  "
          f"{len(P):3d} traj   slope {sl:+.4f}   P(slope>0)={np.mean(S[np.isfinite(S)]>0):.3f}"
          f"   ({nfin}/{NBOOT} resamples usable)")

for v, d in store.items():
    for k, arr in d.items():
        out[f"{v}_{k}"] = np.asarray(arr)      # slope_boot included
np.savez(os.path.join(RESULTS, f"stats_curves{TAG}.npz"), **out)
print("\n  wrote stats_curves.npz")


# ------------------------------------------------- 2. 40V slope + zero detail
print()
print("=" * 74)
print("2.  40V origin slope and outer zero — bootstrap detail")
print("=" * 74)

lab, ext, z3c, lmax = cache["feb25"]
ks40 = np.where(lab == "40V")[0]
P40, r40, l1_40, l2_40 = pick_lag(ext, z3c, ks40, lmax, BAND, TARGET)
ed40, sc40 = edges_for(P40)
mids40, f40, g40, e40 = three_parts(P40, ed40)
sl40 = origin_slope(mids40, g40, sc40)
z40, kind40 = outer_zero(mids40, g40)

S40, Z40, clean40 = [], [], 0
for _ in range(NBOOT):
    j = rng.integers(0, len(P40), len(P40))
    t = three_parts([P40[q] for q in j], ed40)
    if t is None:
        continue
    s = origin_slope(t[0], t[2], sc40)
    z, kd = outer_zero(t[0], t[2])
    S40.append(s)
    if np.isfinite(z) and kd == "stable":
        Z40.append(z)
        if s > 0:
            clean40 += 1
S40 = np.array(S40, float); Z40 = np.array(Z40, float)
fin = np.isfinite(S40)

print(f"  point estimate:  slope {sl40:+.4f}   outer zero {z40:.2f} px ({kind40})")
print(f"  slope bootstrap: median {np.nanmedian(S40):+.4f}"
      f"   68% [{np.nanpercentile(S40,16):+.4f}, {np.nanpercentile(S40,84):+.4f}]"
      f"   95% [{np.nanpercentile(S40,2.5):+.4f}, {np.nanpercentile(S40,97.5):+.4f}]")
print(f"  P(slope > 0)   = {np.mean(S40[fin] > 0):.3f}      "
      f"(n = {fin.sum()} usable resamples)")
print(f"  outer STABLE zero found in {len(Z40)}/{fin.sum()} = "
      f"{len(Z40)/fin.sum():.3f} of resamples")
if len(Z40):
    print(f"  z* bootstrap:    median {np.median(Z40):.2f} px"
          f"   68% [{np.percentile(Z40,16):.2f}, {np.percentile(Z40,84):.2f}]"
          f"   95% [{np.percentile(Z40,2.5):.2f}, {np.percentile(Z40,97.5):.2f}]")
else:
    print("  z* bootstrap:    no resample contains a stable outer zero")
print(f"  P(slope>0 AND one stable outer zero) = {clean40/fin.sum():.3f}")
np.savez(os.path.join(RESULTS, f"stats_slope40{TAG}.npz"), slope_boot=S40, zstar_boot=Z40,
         slope_point=sl40, zstar_point=z40, sigma=sc40,
         lag=np.array([l1_40, l2_40]), r=r40)
print("\n  wrote stats_slope40.npz")


# ----------------------------------------------------------- 3. robustness 40V
print()
print("=" * 74)
print("3.  40V robustness — bin count, extension window, lag target")
print("=" * 74)
print("    (20V shown alongside as the control: it must stay a stable origin)")

NBINS = [7, 9, 11, 13]
BANDS = [(0.28, 0.52), (0.30, 0.55), (0.32, 0.58)]
TARGETS = [0.040, 0.051, 0.060]
HALVES = [2.3, 2.6, 2.9]           # bin width via the half-range in sigma

ks20 = np.where(lab == "20V2")[0]
rows = []
for band, target, nbin, half in itertools.product(BANDS, TARGETS, NBINS, HALVES):
    rec = dict(band=f"{band[0]}-{band[1]}", target=target, nbin=nbin, half=half)
    for name, kk in (("40V", ks40), ("20V", ks20)):
        b = pick_lag(ext, z3c, kk, lmax, band, target)
        if b is None:
            rec[f"{name}_slope"] = ""; rec[f"{name}_zstar"] = ""; continue
        P, r, l1, l2 = b
        x0 = np.concatenate([q[0] for q in P]); sc = np.nanstd(x0)
        edg = np.linspace(-half*sc, half*sc, nbin)
        t = three_parts(P, edg)
        if t is None:
            rec[f"{name}_slope"] = ""; rec[f"{name}_zstar"] = ""; continue
        s = origin_slope(t[0], t[2], sc)
        z, kd = outer_zero(t[0], t[2])
        rec[f"{name}_lag"] = f"({l1},{l2})"
        rec[f"{name}_r"] = round(r, 4)
        rec[f"{name}_slope"] = round(float(s), 5) if np.isfinite(s) else ""
        rec[f"{name}_zstar"] = (round(float(z), 2) if (np.isfinite(z) and kd == "stable")
                                else "")
    rows.append(rec)

keys = ["band", "target", "nbin", "half",
        "40V_lag", "40V_r", "40V_slope", "40V_zstar",
        "20V_lag", "20V_r", "20V_slope", "20V_zstar"]
with open(os.path.join(RESULTS, f"stats_robust40{TAG}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

ok40 = [r for r in rows if r["40V_slope"] != ""]
s40 = [r["40V_slope"] for r in ok40]
z40s = [r["40V_zstar"] for r in ok40 if r["40V_zstar"] != ""]   # same denominator
s20 = [r["20V_slope"] for r in rows if r["20V_slope"] != ""]
print(f"\n  {len(rows)} settings tested (3 bands x 3 targets x 4 bin counts x 3 half-ranges)")
print(f"  40V origin slope : {len(s40)} evaluable, "
      f"{sum(1 for x in s40 if x > 0)} positive "
      f"({sum(1 for x in s40 if x > 0)/len(s40)*100:.0f}%), "
      f"range [{min(s40):+.4f}, {max(s40):+.4f}]")
if z40s:
    print(f"  40V stable outer zero found in {len(z40s)}/{len(s40)} of the evaluable "
          f"settings ({len(z40s)/len(s40)*100:.0f}%)"
          f"   z* range [{min(z40s):.1f}, {max(z40s):.1f}] px")
else:
    print("  40V stable outer zero: none")
print(f"  20V origin slope : {len(s20)} evaluable, "
      f"{sum(1 for x in s20 if x < 0)} negative "
      f"({sum(1 for x in s20 if x < 0)/len(s20)*100:.0f}%), "
      f"range [{min(s20):+.4f}, {max(s20):+.4f}]")
print("\n  wrote stats_robust40.csv")


# --------------------------------------------------------- 4. even-part check
print()
print("=" * 74)
print("4.  even part — is the odd projection creating the result?")
print("=" * 74)
print("    the even part carries the startup bias; it must be small compared with")
print("    the odd part where the odd part is claimed to be informative")
for v in ("20V2", "40V"):
    d = store[v]
    m = np.isfinite(d["g"]) & np.isfinite(d["e"])
    print(f"  {LABEL[v]:>4}: rms(odd) = {np.sqrt(np.nanmean(d['g'][m]**2)):.3f}   "
          f"rms(even) = {np.sqrt(np.nanmean(d['e'][m]**2)):.3f}   "
          f"ratio even/odd = {np.sqrt(np.nanmean(d['e'][m]**2))/np.sqrt(np.nanmean(d['g'][m]**2)):.3f}")
print("\nSTATS_DONE")
