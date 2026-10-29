"""Startup traces of the polymer extension Z1, three Wi groups, 20 fast + 5 slow each.

Z1 = (max_i x_i - min_i x_i) / N, N = 300 beads (the first column of the reduced
data), so Z1 = 1 is a fully stretched chain.  Everything is drawn at the finest
step the data has, dt = 5e-4.

TWO SOURCES, one per regime -- the panel subtitle says which:

* Wi < 5   -- the Wi sweep.  Which trajectories to draw is decided on
  ../z2-z3-Interpret/cache/observables_stride50.npz (11145 trajectories x 101
  frames, one Wi each); the traces themselves are then re-read frame by frame
  from `reduced_symm_new_train` (see read_sweep_full).  Only the
  equilibrium-coil ensembles are used (splits `train` + `train_low`,
  Z1(0) ~ 0.11); `train_slow` starts from a single folded configuration at
  Z1(0) = 0.254 and would show as a second start band.

* Wi >= 5 (and the slow traces of the 1 <= Wi < 5 panel) -- the test set
  `reduced_symm_new_test`, splits Wi{...}_T1_{fast,middle,slow}: 500
  trajectories per (Wi, class), dt = 5e-4.  It holds three FIXED initial
  configurations per Wi -- fast Z1(0)=0.187 (pre-stretched), middle 0.061
  (compact coil), slow 0.118 (folded, Z3=2.2) -- so the slow unfolders the
  coil-start sweep barely contains are actually present.  Its windows are
  shorter than the sweep's (t <= 0.5 at Wi >= 5), hence the per-panel xmax.

SELECTION.  Per panel, the onset time of a trajectory is

    tau = first t with Z1 >= THR          (tau = inf if it never gets there),

the trajectories that cross are split at their median tau, and the panel draws
N_FAST random traces from the early half plus N_SLOW from the late half + the
never-crossing ones.  The 20:5 ratio is a drawing choice, NOT the population
ratio -- the fraction of the group that ever reaches THR is in each subtitle.

    python plot_z1_time.py              # -> z1_vs_time.png
"""
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
OBS_FILE = Path(os.environ.get(
    "OBS_FILE", HERE.parent / "z2-z3-Interpret" / "cache" / "observables_stride50.npz"))
TEST_ROOT = Path(os.environ.get(
    "TEST_ROOT", "/home/aiqing/PhaseDiagram_save1/examples/UIDD_polymer_dynamics_wi"
                 "/data_cache_nolog/reduced_symm_new_test"))
SEED = 0

# Wi is carried by HUE, not by light/dark: plasma cut to [0.05, 0.85], i.e.
# dark blue-violet (low Wi) -> magenta -> orange (high Wi).  The cut drops
# plasma's pale yellow end, which thin lines cannot afford on white.  Same
# LogNorm as ../z2-z3-Interpret/plot_interpret.py; Wi < 0.01 clips to the end.
WI_CMAP = LinearSegmentedColormap.from_list(
    "wi_plasma", plt.get_cmap("plasma")(np.linspace(0.05, 0.85, 256)))
WI_NORM = LogNorm(vmin=1e-2, vmax=10.8)
INK = "#52514e"

SPLIT_NAMES = ("train", "train_slow", "train_low")     # = extract_observables.SPLITS
COIL_SPLITS = (0, 2)                                   # equilibrium-coil starts
THR = float(os.environ.get("THR", "0.3"))              # onset threshold on Z1
N_FAST = int(os.environ.get("N_FAST", "20"))
N_SLOW = int(os.environ.get("N_SLOW", "5"))
SWEEP_FULL_DIR = Path(os.environ.get(
    "SWEEP_FULL_DIR", "/home/aiqing/PhaseDiagram_save1/examples/UIDD_polymer_dynamics_wi"
                      "/data_cache_nolog/reduced_symm_new_train"))
SEGS_PER_TRAJ = 10          # reduced_symm_new_train stores 10 x 500 frames per trajectory
FULL_RES = os.environ.get("FULL_RES", "1") != "0"
SINGLE_XMAX = float(os.environ.get("SINGLE_XMAX", "1.0"))   # z1_traces_single.png
PANELS = [
    dict(wi=(0.0, 1.0), src="sweep", xmax=2.5, n_fast=3, n_slow=2),
    dict(wi=(1.0, 5.0), src="sweep", slow_src="test", xmax=1.25),
    dict(wi=(5.0, 10.81), src="test", xmax=0.5),
]

YLABEL = r"polymer extension  $Z_1$"
XLABEL = r"time  $t$"


# --- data ------------------------------------------------------------------
def load_sweep():
    """Coil-start trajectories of the sweep, plus their row ids in the FULL
    cache -- read_sweep_full indexes the unfiltered trajectory order."""
    d = dict(np.load(OBS_FILE))
    keep = np.isin(d["split"], COIL_SPLITS)
    return d["z1"][keep], d["t"][0], d["Wi"][keep], np.flatnonzero(keep)


def _col(batch, name):
    """(rows, frames, width) float32 view of a nested-list column, zero copy."""
    col = batch.column(name)
    mid = col.values
    n_fr = int(col.offsets[1].as_py() - col.offsets[0].as_py())
    width = int(mid.offsets[1].as_py() - mid.offsets[0].as_py())
    return mid.values.to_numpy(zero_copy_only=True).reshape(batch.num_rows, n_fr, width)


def read_test_split(name):
    t, z1 = [], []
    for f in sorted(glob.glob(str(TEST_ROOT / name / "*.arrow"))):
        for b in pa.ipc.open_stream(pa.memory_map(f)):
            t.append(_col(b, "t")[:, :, 0])
            z1.append(_col(b, "x")[:, :, 0])
    return np.concatenate(t), np.concatenate(z1)


def test_wi(name):
    """'Wi1_2_T1_slow' -> 1.2"""
    return float(name.split("_T1_")[0][2:].replace("_", "."))


def load_test(lo, hi):
    """All test splits with lo <= Wi < hi.  The splits share dt but not the
    window length (t <= 1.25 at Wi = 1, 0.5 at Wi = 2 fast), so the short ones
    are NaN-padded to the longest grid -- matplotlib just stops drawing there."""
    names = sorted(p.name for p in TEST_ROOT.iterdir()
                   if p.is_dir() and lo <= test_wi(p.name) < hi)
    grids, z1, wi, cls = [], [], [], []
    for n in names:
        t, z = read_test_split(n)
        if grids and not np.isclose(t[0, 1] - t[0, 0], grids[0][1] - grids[0][0]):
            raise ValueError(f"{n} has a different dt than the other splits")
        grids.append(t[0])
        z1.append(z)
        wi.append(np.full(len(z), test_wi(n)))
        cls.append(np.full(len(z), n.split("_T1_")[1]))
    t_ref = max(grids, key=len)
    z1 = [np.pad(z.astype(np.float64), ((0, 0), (0, len(t_ref) - z.shape[1])),
                 constant_values=np.nan) for z in z1]
    return np.concatenate(z1), t_ref, np.concatenate(wi), np.concatenate(cls)


# --- selection -------------------------------------------------------------
def onset_time(z1, t, thr=THR):
    """First t with Z1 >= thr, inf if the trajectory never reaches thr."""
    hit = z1 >= thr
    return np.where(hit.any(axis=1), t[np.argmax(hit, axis=1)], np.inf)


def pick_fast_slow(idx, tau, n_fast=N_FAST, n_slow=N_SLOW, seed=SEED):
    """n_fast traces from the early half, n_slow from the late half + never."""
    rng = np.random.default_rng(seed)
    crossed = idx[np.isfinite(tau[idx])]
    if crossed.size == 0:                       # nothing stretches: draw at random
        return np.array([], int), rng.choice(idx, min(n_slow, idx.size), replace=False)
    fast = crossed[tau[crossed] <= np.median(tau[crossed])]
    slow = np.setdiff1d(idx, fast)
    take = lambda pool, n: (pool if pool.size <= n
                            else np.sort(rng.choice(pool, n, replace=False)))
    return take(fast, n_fast), take(slow, n_slow)


def read_sweep_full(idx):
    """Full-resolution z1 (dt = 5e-4) for the given sweep trajectories.

    The stride-50 cache is every 50th frame; the frames themselves live in
    `reduced_symm_new_train`, which stores each trajectory as SEGS_PER_TRAJ
    consecutive 500-frame rows IN THE SAME ORDER as the cache (checked: Wi and
    the stride-50 z1 agree for trajectories 0, 4944, 5000, 9000, 11144), so
    trajectory i is rows 10i..10i+9.  The arrow files are memory-mapped and only
    the requested rows are touched, so 25 traces cost a few MB, not 2 GB."""
    want = {int(i) * SEGS_PER_TRAJ + k: (pos, k)
            for pos, i in enumerate(idx) for k in range(SEGS_PER_TRAJ)}
    z = t = None
    off = 0
    for f in sorted(glob.glob(str(SWEEP_FULL_DIR / "*.arrow"))):
        for b in pa.ipc.open_stream(pa.memory_map(f)):
            rows = [r for r in range(off, off + b.num_rows) if r in want]
            if rows:
                xcol, tcol = _col(b, "x"), _col(b, "t")
                n_fr = xcol.shape[1]
                if z is None:
                    z = np.empty((len(idx), SEGS_PER_TRAJ, n_fr), np.float32)
                    t = np.empty((SEGS_PER_TRAJ, n_fr))
                for r in rows:
                    pos, k = want[r]
                    z[pos, k] = xcol[r - off, :, 0]
                    t[k] = tcol[r - off, :, 0]
            off += b.num_rows
    if z is None:
        raise ValueError("no rows read -- is SWEEP_FULL_DIR right?")
    return t.ravel(), z.reshape(len(idx), -1)


def pick_slow(tau, n_slow=N_SLOW, seed=SEED):
    """n_slow traces from the late half of the trajectories that do cross THR.

    Used when the slow traces are borrowed from the test set: there the point is
    to show late stretchers, not chains that never stretch."""
    crossed = np.flatnonzero(np.isfinite(tau))
    late = crossed[tau[crossed] > np.median(tau[crossed])]
    if late.size <= n_slow:
        return late
    return np.sort(np.random.default_rng(seed).choice(late, n_slow, replace=False))


# --- figure ----------------------------------------------------------------
def draw(ax, t, z1, wi, idx, lw=1.1, alpha=0.85):
    if idx.size == 0:
        return
    segs = [np.column_stack([t, z1[i]]) for i in idx]
    ax.add_collection(LineCollection(segs, colors=WI_CMAP(WI_NORM(wi[idx])),
                                     linewidths=lw, alpha=alpha, capstyle="round"))


def panel_traces(panel, k, sweep, force_sweep=False):
    """The N_FAST fast + N_SLOW slow traces of one Wi group, as drawable layers.

    A layer is (t, z1, wi, idx): the slow traces may come from a different
    source than the fast ones, hence their own time grid and arrays.  Which
    trajectories are fast and which are slow is decided on the stride-50 cache
    (cheap, and tau does not need fine frames); the traces themselves are then
    re-read at dt = 5e-4 -- see read_sweep_full."""
    lo, hi = panel["wi"]
    z1_s, t_s, wi_s, tau_s, gidx_s = sweep
    from_sweep = panel["src"] == "sweep" or force_sweep
    if from_sweep:
        z1, t, wi, tau = z1_s, t_s, wi_s, tau_s
        idx = np.flatnonzero((wi >= lo) & (wi < hi))
        label = rf"${lo:g} \leq Wi < {hi:g}$"
    else:
        z1, t, wi, _cls = load_test(lo, hi)
        tau = onset_time(z1, t)
        idx = np.arange(len(z1))
        label = rf"${lo:g} \leq Wi \leq {wi.max():g}$"
    fast, slow = pick_fast_slow(idx, tau, seed=SEED + k,
                                n_fast=panel.get("n_fast", N_FAST),
                                n_slow=panel.get("n_slow", N_SLOW))

    def layer(sel):
        """Full-resolution frames for a sweep selection, as-is for a test one."""
        if from_sweep and FULL_RES and sel.size:
            t_f, z_f = read_sweep_full(gidx_s[sel])
            return (t_f, z_f, wi[sel], np.arange(sel.size))
        return (t, z1, wi, sel)

    layers = [layer(slow), layer(fast)]          # stragglers first
    drawn = f"{fast.size} fast + {slow.size} slow drawn"

    if panel.get("slow_src") == "test" and not force_sweep:
        z1_t, t_t, wi_t, _cls = load_test(lo, hi)
        slow = pick_slow(onset_time(z1_t, t_t), n_slow=panel.get("n_slow", N_SLOW),
                         seed=SEED + 100 + k)
        layers = [(t_t, z1_t, wi_t, slow), layers[1]]
        drawn = f"{fast.size} fast (sweep) + {slow.size} slow (test set)"

    dt = lambda tt: rf"{tt[1] - tt[0]:.0e}".replace("e-04", r"\times 10^{-4}")
    if from_sweep:
        src = rf"$Wi$ sweep, coil starts, $\Delta t = {dt(layers[1][0])}$"
        if panel.get("slow_src") == "test" and not force_sweep:
            src += (rf"; slow from test set, $Wi = ${np.unique(layers[0][2]).min():g}"
                    rf"$-${np.unique(layers[0][2]).max():g}")
    else:
        src = (rf"test set, $Wi = $ {', '.join(f'{v:g}' for v in np.unique(wi))}, "
               rf"3 starts $\times$ 500, $\Delta t = {dt(t)}$")

    return dict(layers=layers, idx=idx, tau=tau, fast=fast, slow=slow,
                src=src, label=label, drawn=drawn)


def fig_annotated(sweep):
    """The documented version: axes, titles, onset level, Wi colorbar."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True, layout="constrained")
    for k, (ax, panel) in enumerate(zip(axes, PANELS)):
        d = panel_traces(panel, k, sweep)
        ax.axhline(THR, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=0.5)
        for layer in d["layers"]:
            draw(ax, *layer)
        frac = np.isfinite(d["tau"][d["idx"]]).mean()
        ax.set_title(d["label"], fontsize=12, pad=25)
        ax.text(0.5, 1.055, rf"$n = {d['idx'].size}$   |   {100 * frac:.0f}% reach "
                            rf"$Z_1\!\geq\!{THR:g}$   |   {d['drawn']}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.text(0.5, 1.008, d["src"], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=INK, style="italic")
        ax.set_xlim(0.0, panel["xmax"])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(XLABEL, fontsize=11)
        ax.grid(True, lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(YLABEL, fontsize=11)
    axes[-1].text(0.98, THR + 0.025, rf"onset level $Z_1 = {THR:g}$", fontsize=8.5,
                  color=INK, va="bottom", ha="right",
                  transform=axes[-1].get_yaxis_transform(),
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    wi_colorbar(fig, axes)
    save(fig, "z1_vs_time.png")


def wi_colorbar(fig, ax, labelled=True, pad=0.012, shrink=1.0):
    """The Wi ramp.  labelled=False leaves a bare strip: no ticks, no label."""
    cb = fig.colorbar(ScalarMappable(norm=WI_NORM, cmap=WI_CMAP), ax=ax,
                      pad=pad, aspect=26 * shrink, shrink=shrink)
    if labelled:
        cb.set_label(r"$Wi$", fontsize=13)
    else:
        cb.set_ticks([])
        cb.ax.minorticks_off()          # LogNorm adds minor ticks of its own
        cb.outline.set_visible(False)
    return cb


def bare(ax, xmax):
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def fig_bare_panels(sweep):
    """Traces only: the same three groups, each keeping its own time axis."""
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 9.2), layout="constrained",
                             gridspec_kw=dict(hspace=0.03))
    for k, (ax, panel) in enumerate(zip(axes, PANELS)):
        d = panel_traces(panel, k, sweep)
        for layer in d["layers"]:
            draw(ax, *layer)
        bare(ax, panel["xmax"])
    wi_colorbar(fig, axes, labelled=False, pad=0.055, shrink=0.6)
    save(fig, "z1_traces.png")


def fig_bare_single(sweep):
    """Traces only, all three groups in ONE axes -> one common time base, so
    every trace comes from the Wi sweep, drawn out to SINGLE_XMAX."""
    fig, ax = plt.subplots(figsize=(5.4, 7.0), layout="constrained")
    for k, panel in enumerate(PANELS):
        d = panel_traces(panel, k, sweep, force_sweep=True)
        for layer in d["layers"]:
            draw(ax, *layer)
    bare(ax, SINGLE_XMAX)
    wi_colorbar(fig, ax, labelled=False, pad=0.055, shrink=0.6)
    save(fig, "z1_traces_single.png")


def save(fig, name, dpi=240):
    out = HERE / name
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    z1_s, t_s, wi_s, gidx_s = load_sweep()
    sweep = (z1_s, t_s, wi_s, onset_time(z1_s, t_s), gidx_s)
    fig_annotated(sweep)
    fig_bare_panels(sweep)
    fig_bare_single(sweep)
