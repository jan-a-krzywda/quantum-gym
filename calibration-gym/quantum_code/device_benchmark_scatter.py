"""
2D scatter panels for device benchmarks (T1, T2R, readout error, readout QNDness).

Per-qubit colors match latent plots: ``tab20`` indexed by **logical** qubit id (0..16).

Numeric arrays are aligned to the bar-chart order used on the Tuna dashboard
(see ``QUBIT_BAR_ORDER``). Values are **manually transcribed** from screenshots when
API properties are unavailable; treat them as approximate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Logical qubit index order as shown left-to-right on 2026 benchmark bar charts.
QUBIT_BAR_ORDER: Tuple[int, ...] = (
    0,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
)

# --- Eyeball from screenshots (µs for T1 / T2R; % for errors) -----------------

# Latest T1 (µs)
_T1_US_BAR: Tuple[float, ...] = (
    22,
    26,
    22,
    18,
    26,
    29,
    19,
    23,
    25,
    25,
    20,
    19,
    22,
    24,
    24,
    15,
    25,
)

# Latest T2R (µs) — relaxation with refocus; dashboard label "T2R"
_T2R_US_BAR: Tuple[float, ...] = (
    18,
    26,
    5,
    7,
    28,
    7,
    16,
    8,
    8,
    19,
    7,
    8,
    3,
    0,
    3,
    6,
    8,
)

# Latest simultaneous readout error (%)
_RE_PCT_BAR: Tuple[float, ...] = (
    4.65,
    3.40,
    1.20,
    1.39,
    2.37,
    4.17,
    1.29,
    2.11,
    2.94,
    1.55,
    2.02,
    1.43,
    0.73,
    2.89,
    1.46,
    2.89,
    3.09,
)

# Latest simultaneous readout QNDness error (%)
_RQNDE_PCT_BAR: Tuple[float, ...] = (
    3.1,
    3.2,
    3.1,
    4.0,
    5.8,
    2.1,
    2.5,
    3.6,
    2.0,
    1.4,
    5.8,
    3.4,
    3.4,
    4.0,
    3.5,
    4.9,
    5.9,
)


def _get_tab20():
    try:
        return matplotlib.colormaps["tab20"]
    except AttributeError:
        return matplotlib.cm.get_cmap("tab20")


def _tab20_rgb(q: int, n_qubits: int) -> Tuple[float, float, float]:
    cmap = _get_tab20()
    norm = Normalize(vmin=0, vmax=max(n_qubits - 1, 1))
    r, g, b, _ = cmap(norm(float(q)))
    return float(r), float(g), float(b)


def _scatter_qubit_labels(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    logical_ids: Sequence[int],
    *,
    n_qubits: int,
) -> None:
    """Match ``plot_latent_from_checkpoint`` terminal-shot label style."""
    for i, q in enumerate(logical_ids):
        rgb = _tab20_rgb(int(q), n_qubits)
        ax.scatter(
            xs[i],
            ys[i],
            s=140,
            facecolors=rgb,
            edgecolors="none",
            zorder=5,
        )
        ax.annotate(
            str(int(q)),
            xy=(float(xs[i]), float(ys[i])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=rgb,
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.95,
            },
            clip_on=False,
        ).set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])


def _panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    logical_ids: Sequence[int],
    *,
    n_qubits: int,
) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28, linewidth=0.6)
    _scatter_qubit_labels(ax, x, y, logical_ids, n_qubits=n_qubits)
    ax.margins(0.12)


def plot_device_benchmark_planes(
    out_path: Path,
    *,
    dpi: int = 130,
    figsize: Tuple[float, float] = (12.5, 8.2),
) -> Path:
    """
    Write a single figure with six 2D planes: all pairs among T1, T2R, RE, RQNDE.

    Uses ``QUBIT_BAR_ORDER`` / transcribed arrays in this module (not the live backend).
    """
    n = len(QUBIT_BAR_ORDER)
    if not (
        len(_T1_US_BAR) == n
        == len(_T2R_US_BAR)
        == len(_RE_PCT_BAR)
        == len(_RQNDE_PCT_BAR)
    ):
        raise RuntimeError("benchmark array lengths must match QUBIT_BAR_ORDER")

    t1 = np.asarray(_T1_US_BAR, dtype=np.float64)
    t2 = np.asarray(_T2R_US_BAR, dtype=np.float64)
    re_ = np.asarray(_RE_PCT_BAR, dtype=np.float64)
    rq = np.asarray(_RQNDE_PCT_BAR, dtype=np.float64)
    q_ids = list(QUBIT_BAR_ORDER)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax_flat: List[plt.Axes] = list(axes.ravel())

    panels: Iterable[Tuple[int, str, np.ndarray, str, np.ndarray, str, str, str]] = (
        (0, r"$T_1$ (µs)", t1, r"$T_{2R}$ (µs)", t2, "T1 vs T2R", "t1", "t2r"),
        (1, r"$T_1$ (µs)", t1, "readout error (%)", re_, "T1 vs readout error", "t1", "re"),
        (2, r"$T_1$ (µs)", t1, "readout QNDness err. (%)", rq, "T1 vs RQNDE", "t1", "rqnde"),
        (3, r"$T_{2R}$ (µs)", t2, "readout error (%)", re_, "T2R vs readout error", "t2r", "re"),
        (4, r"$T_{2R}$ (µs)", t2, "readout QNDness err. (%)", rq, "T2R vs RQNDE", "t2r", "rqnde"),
        (5, "readout error (%)", re_, "readout QNDness err. (%)", rq, "readout err vs RQNDE", "re", "rqnde"),
    )

    for ax_i, xl, xv, yl, yv, ttl, _, _ in panels:
        _panel(ax_flat[ax_i], xv, yv, xl, yl, ttl, q_ids, n_qubits=n)

    fig.suptitle(
        "Device benchmarks (approx. from 2026 dashboard screenshots; tab20 = logical qubit)",
        fontsize=12,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
