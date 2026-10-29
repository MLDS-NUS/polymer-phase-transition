"""Create the combined n=8,16,32,infinity argmax plot after all jobs finish."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DPI = 150
TARGET_PIXELS = (1630, 955)
FIGSIZE = tuple((length + 0.01) / OUTPUT_DPI for length in TARGET_PIXELS)
BASE_COLOR = "C0"
WI_TO_BETA = 45000.0 / 4321.0

SERIES = (
    ("n8", r"$n=8$", DATA_DIR / "mc_n8_wi_001_10.npz", 0.20, "o", "mc"),
    ("n16", r"$n=16$", DATA_DIR / "mc_n16_wi_001_10.npz", 0.35, "^", "mc"),
    ("n32", r"$n=32$", DATA_DIR / "mc_n32_wi_001_10.npz", 0.50, "s", "mc"),
    ("ninf", r"$n\to\infty$", DATA_DIR / "ninf" / "ninf_wi_001_10.npz", 1.00, "D", "ninf"),
)


def atomic_csv(path: Path, array: np.ndarray, header: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            np.savetxt(handle, array, delimiter=",", header=header,
                       comments="", fmt="%.10g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def argmax_branches(values: np.ndarray, z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = int(np.argmin(np.abs(z_grid)))
    if not np.isclose(z_grid[center], 0.0):
        raise ValueError("z grid must contain zero")
    relative = values - values[:, center : center + 1]
    positive = z_grid >= 0.0
    negative = z_grid <= 0.0
    return (
        z_grid[positive][relative[:, positive].argmax(axis=1)],
        z_grid[negative][relative[:, negative].argmax(axis=1)],
    )


def load_series(key: str, label: str, path: Path, alpha: float, marker: str,
                kind: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if not np.asarray(data["completed"], dtype=bool).all():
            raise RuntimeError(f"{path} is incomplete; refusing to plot partial data")
        wi = np.asarray(data["wi_values"], dtype=float)
        beta = np.asarray(data["beta_values"], dtype=float)
        z_grid = np.asarray(data["z_grid"], dtype=float)
        if kind == "mc":
            n = int(data["n"])
            values = np.asarray(data["mc_logZ"], dtype=float) / n
            boundary = np.asarray(data["argmax_at_boundary"], dtype=bool)
            max_se = np.asarray(data["max_relative_se"], dtype=float)
        else:
            values = np.asarray(data["F_all"], dtype=float)
            boundary = np.asarray(data.get("argmax_at_boundary", np.zeros(len(wi))), dtype=bool)
            max_se = np.full(len(wi), np.nan)
    pos, neg = argmax_branches(values, z_grid)
    return dict(key=key, label=label, wi=wi, beta=beta, pos=pos, neg=neg,
                alpha=alpha, marker=marker, boundary=boundary, max_se=max_se)


def main() -> None:
    data = [load_series(*spec) for spec in SERIES]
    wi = data[0]["wi"]
    beta = data[0]["beta"]
    for series in data[1:]:
        if not np.allclose(series["wi"], wi) or not np.allclose(series["beta"], beta):
            raise ValueError("The four result files do not share the requested Wi grid")
    if not np.allclose(beta, wi * WI_TO_BETA):
        raise ValueError("Stored beta values do not match Wi*45000/4321")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for series in data:
        style = dict(color=BASE_COLOR, alpha=series["alpha"], lw=2.6,
                     marker=series["marker"], ms=7.5,
                     markerfacecolor=BASE_COLOR, markeredgecolor=BASE_COLOR)
        ax.plot(series["wi"], series["pos"], "-", label=series["label"], **style)
        ax.plot(series["wi"], series["neg"], "-", **style)

    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(wi[0], wi[-1])
    ax.set_xlabel(r"$Wi=\beta\,4321/45000$", fontsize=13)
    ax.set_ylabel(r"$z^*$", fontsize=13)
    ax.set_title(r"Argmax location vs $Wi$", fontsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    figure = HERE / "argmax_vs_wi_001_10.png"
    fig.savefig(figure, dpi=OUTPUT_DPI)
    plt.close(fig)

    columns = [beta, wi]
    names = ["beta", "Wi"]
    for series in data:
        columns.extend([series["pos"], series["neg"], series["boundary"].astype(int), series["max_se"]])
        names.extend([
            f"z_star_pos_{series['key']}", f"z_star_neg_{series['key']}",
            f"argmax_at_boundary_{series['key']}", f"max_relative_se_{series['key']}",
        ])
    csv = HERE / "argmax_vs_wi_001_10.csv"
    atomic_csv(csv, np.column_stack(columns), ",".join(names))
    print(f"  -> {figure}")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
