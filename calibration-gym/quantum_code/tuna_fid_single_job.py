"""
Single-job Ramsey-style FID on all Tuna-17 qubits.

For each delay τ in a list (default τ_i = 100 * i nanoseconds), all qubits run in
parallel: H → idle(τ) → H → measure → reset. Every τ step is appended to the same
circuit so one `backend.run(...)` submits all delays at once (same pattern as
zebra_plot.ipynb: multiple measure rounds into distinct classical bits).

Requires: qiskit, qiskit-quantuminspire, and `qi login` for Quantum Inspire.

Memory plots: optional slow tilted PNG + 3D cumulative GIF; 2D slice GIF (Matplotlib
``imshow``, frame PNGs → GIF) with optional repetition binning. **Derived outputs:**
co-click probability, co-click excess (covariance), τ×repetition map, optional ``.npz``.
"""

from __future__ import annotations

import argparse
import io
import shutil
import warnings
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_quantuminspire.qi_provider import QIProvider
from zebra_plot_style import zebra_gif_2d_style


def tau_ns_from_indices(i_start: int, i_end: int, dt: float = 100.0) -> List[float]:
    """τ_i = 100 * i * ns for i in [i_start, i_end) (half-open)."""
    return [dt * i for i in range(i_start, i_end)]


def build_fid_circuit_all_qubits(
    num_qubits: int,
    tau_ns_list: Sequence[float],
    *,
    dt_seconds: Optional[float] = None, echo: bool = False, OTOC: bool = False
) -> QuantumCircuit:
    """
    One circuit: for each τ, all qubits get H — delay — H — measure — reset.

    Classical layout: for step k, qubit j is written to classical bit k * num_qubits + j
    (same indexing style as zebra_plot.ipynb).
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if not tau_ns_list:
        raise ValueError("tau_ns_list must be non-empty")

    qubits = list(range(num_qubits))
    n_tau = len(tau_ns_list)
    qc = QuantumCircuit(num_qubits, n_tau * num_qubits)

    for k, tau_ns in enumerate(tau_ns_list):
        qc.h(qubits)
        if dt_seconds is not None and dt_seconds > 0:
            print("tutaj")
            delay_dt = max(0, int(round(tau_ns * 1e-9 / dt_seconds)))
            delay_dt_8 = delay_dt - int(round(200*1e-9 / dt_seconds)    )
            if OTOC:
                for i in range(5):
                    qc.x(8)
                    qc.y(8)
                qubits_without_8 = [q for q in qubits if q != 8]
                qc.delay(delay_dt, qubits_without_8, unit="dt")
                qc.delay(delay_dt_8, 8, unit="dt")

            elif echo:
                qc.delay(delay_dt/2, qubits, unit="dt")
                qc.x(qubits)
                qc.delay(delay_dt/2, qubits, unit="dt")
                
            else:
                qc.delay(delay_dt, qubits, unit="dt")
        else:
            if echo:
                qc.delay(tau_ns/2, qubits, unit="ns")
                qc.x(qubits)
                qc.delay(tau_ns/2, qubits, unit="ns")
            else:
                qc.delay(tau_ns, qubits, unit="ns")
        qc.h(qubits)
        cbits = list(k * num_qubits + np.arange(num_qubits, dtype=int))
        print(cbits, qubits)
        qc.measure(qubits, cbits)

    return qc


def survival_prob_0_from_memory(
    memory: List[str], num_qubits: int, n_tau: int
) -> np.ndarray:
    """
    From shots with memory=True, shape (n_tau, num_qubits) probability of reading '0'.

    Classical bit index c = k * num_qubits + j is the j-th qubit at delay step k.
    In Qiskit bitstrings, c0 is the rightmost character.
    """
    n_shots = len(memory)
    if n_shots == 0:
        return np.zeros((n_tau, num_qubits))

    n_c = n_tau * num_qubits
    out = np.zeros((n_tau, num_qubits))
    for shot in memory:
        if len(shot) < n_c:
            shot = "0" * (n_c - len(shot)) + shot
        for k in range(n_tau):
            for j in range(num_qubits):
                cbit = k * num_qubits + j
                if shot[-1 - cbit] == "0":
                    out[k, j] += 1
    out /= n_shots
    return out


def bitstring_to_shot_matrix(shot: str, num_qubits: int, n_tau: int) -> np.ndarray:
    """
    One memory bitstring → matrix (num_qubits, n_tau), values 0/1.

    Same classical layout as ``survival_prob_0_from_memory``: cbit = k * num_qubits + j
    for delay step k and qubit j; Qiskit memory uses ``shot[-1 - cbit]``.
    Column k is delay step k (k=0 is the first τ); row j is qubit j.
    """
    n_c = n_tau * num_qubits
    if len(shot) < n_c:
        shot = "0" * (n_c - len(shot)) + shot
    mat = np.zeros((num_qubits, n_tau), dtype=np.uint8)
    for k in range(n_tau):
        for j in range(num_qubits):
            cbit = k * num_qubits + j
            mat[j, k] = 1 if shot[-1 - cbit] == "1" else 0
    return mat


def memory_to_stack(mem: List[str], num_qubits: int, n_tau: int) -> np.ndarray:
    """Shape (n_shots, num_qubits, n_tau), dtype uint8."""
    return np.stack([bitstring_to_shot_matrix(s, num_qubits, n_tau) for s in mem], axis=0)


def differential_readout_along_tau(
    stack: np.ndarray,
    *,
    implicit_prior: int = 0,
) -> np.ndarray:
    """
    Consecutive XOR along τ (column axis) for no-reset Ramsey-style readout.

    For each shot and qubit, output bit at delay index ``k`` is ``1`` iff the raw
    outcome differs from the previous delay: ``raw[k] ⊕ raw[k-1]``, with
    ``raw[-1] := implicit_prior & 1`` (default ``0``, i.e. compare the first τ bin
    to a nominal |0⟩). Equivalently: same as previous → ``0``, different → ``1``.

    ``stack`` is ``(n_shots, n_qubits, n_tau)`` from :func:`memory_to_stack`; the
    returned array has the same shape and ``dtype`` ``uint8``.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (n_shots, n_qubits, n_tau), got {stack.shape}")
    x = np.asarray(stack, dtype=np.uint8, order="C")
    n_s, n_q, n_t = x.shape
    if n_t == 0:
        return x.copy()
    prior = np.uint8(int(implicit_prior) & 1)
    out = np.empty_like(x)
    out[:, :, 0] = np.bitwise_xor(x[:, :, 0], prior)
    if n_t > 1:
        out[:, :, 1:] = np.bitwise_xor(x[:, :, 1:], x[:, :, :-1])
    return out


def qubit_persona_flatten(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Qubit Persona slicing: one row = τ sweep for one qubit on one shot.

    ``stack`` is ``(n_shots, n_qubits, n_tau)`` from :func:`memory_to_stack`.
    Returns ``(X, qubit_ids)`` with ``X`` shape ``(n_shots * n_qubits, n_tau)`` and
    ``qubit_ids`` length matching rows (each qubit index repeated ``n_shots`` times).
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (n_shots, n_qubits, n_tau), got {stack.shape}")
    n_shots, n_q, n_tau = stack.shape
    x = stack.reshape(-1, n_tau)
    qubit_ids = np.tile(np.arange(n_q, dtype=int), n_shots)
    return x, qubit_ids


def pca_qubit_persona(stack: np.ndarray, n_components: int = 2, **pca_kwargs):
    """
    PCA on Qubit Persona rows: features = τ samples, samples = (shot, qubit) pairs.

    Requires ``scikit-learn``. Interpreting PC1/PC2 as dominant decay-curve shapes
    applies when rows are independent τ sweeps (see :func:`qubit_persona_flatten`).
    Extra keyword arguments are forwarded to ``sklearn.decomposition.PCA``.

    Returns
    -------
    latent : ndarray, shape (n_shots * n_qubits, n_components)
    pca : sklearn.decomposition.PCA
    qubit_ids : ndarray, same length as rows of ``latent``
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "pca_qubit_persona requires scikit-learn (pip install scikit-learn)"
        ) from e

    x, qubit_ids = qubit_persona_flatten(stack)
    pca = PCA(n_components=n_components, **pca_kwargs)
    latent = pca.fit_transform(x)
    return latent, pca, qubit_ids


def tau_prefix_matrix(
    curve: np.ndarray, *, pad: str = "hold"
) -> np.ndarray:
    """
    Build length-``n_tau`` rows for each delay index ``k`` (0 .. n_tau-1).

    Row ``k`` is the measured values for indices ``0 .. k``, with the remainder
    filled so PCA receives a fixed-size vector: ``pad='hold'`` repeats ``curve[k]``
    (last observed value), ``pad='zero'`` uses zeros.

    Used to trace how a sweep moves in PC space as ``k`` increases.
    """
    x = np.asarray(curve, dtype=np.float64).ravel()
    n_tau = x.size
    if n_tau == 0:
        return np.zeros((0, 0))
    out = np.zeros((n_tau, n_tau), dtype=np.float64)
    for k in range(n_tau):
        out[k, : k + 1] = x[: k + 1]
        if pad == "hold":
            out[k, k + 1 :] = x[k]
        elif pad == "zero":
            pass
        else:
            raise ValueError("pad must be 'hold' or 'zero'")
    return out


def pca_latent_trajectory_along_tau(
    pca,
    stack: np.ndarray,
    shot: int,
    qubit: int,
    *,
    pad: str = "hold",
) -> np.ndarray:
    """
    PC coordinates for delay indices ``k = 0 .. n_tau-1`` for one (shot, qubit).

    Uses :func:`tau_prefix_matrix` and ``pca.transform``; row ``k`` corresponds
    to the τ-prefix ending at index ``k``. Shape ``(n_tau, n_components)``.
    """
    curve = stack[shot, qubit, :]
    pref = tau_prefix_matrix(curve, pad=pad)
    return pca.transform(pref)


def memory_cube_q_time_rep(stack: np.ndarray) -> np.ndarray:
    """
    Reorder to ``[qubits, time (τ), repetitions]``.

    ``stack`` is ``(n_shots, n_qubits, n_tau)`` from ``memory_to_stack``; result has
    ``cube[j, k, s] = stack[s, j, k]``.
    """
    return np.transpose(stack, (1, 2, 0))


def moving_average_along_repetitions(
    cube: np.ndarray, *, window: int = 5
) -> np.ndarray:
    """``cube`` is ``(n_q, n_tau, n_s)``; smooth along the last axis (centered, edges shrink)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    nq, nt, ns = cube.shape
    half = window // 2
    f = cube.astype(np.float64, copy=False)
    out = np.empty((nq, nt, ns), dtype=np.float64)
    for k in range(ns):
        i0 = max(0, k - half)
        i1 = min(ns, k + half + 1)
        out[:, :, k] = f[:, :, i0:i1].mean(axis=2)
    return out


def cube_repetition_block_mean(cube: np.ndarray, k: int) -> np.ndarray:
    """
    Average ``cube`` along the repetition axis in consecutive blocks of size ``k``.

    ``cube`` is ``(n_q, n_tau, n_rep)``. The last block may be shorter than ``k``.
    Returns shape ``(n_q, n_tau, n_blocks)`` with ``n_blocks = ceil(n_rep / k)``.
    """
    if k <= 1:
        return cube.astype(np.float64, copy=False)
    nq, nt, nr = cube.shape
    n_blocks = (nr + k - 1) // k
    f = cube.astype(np.float64, copy=False)
    out = np.empty((nq, nt, n_blocks), dtype=np.float64)
    for b in range(n_blocks):
        i0 = b * k
        i1 = min((b + 1) * k, nr)
        out[:, :, b] = f[:, :, i0:i1].mean(axis=2)
    return out


def inferred_qubits_from_memory(mem: List[str], n_tau: int) -> Optional[int]:
    """
    Infer qubit count from classical bitstring length: ``len(shot) / n_tau`` (must divide).

    If memory length does not match ``num_qubits * n_tau``, parsing was wrong or the job
    used a different register size — see ``warn_if_memory_qubit_mismatch``.
    """
    if not mem or n_tau <= 0:
        return None
    L = max(len(s) for s in mem)
    if L % n_tau != 0:
        return None
    return L // n_tau


def warn_if_memory_qubit_mismatch(
    mem: List[str], n_tau: int, num_qubits: int, *, context: str = ""
) -> None:
    """Emit a warning when bitstring length implies a qubit count ≠ ``num_qubits``."""
    inf = inferred_qubits_from_memory(mem, n_tau)
    if inf is None or inf == num_qubits:
        return
    extra = f" {context}" if context else ""
    warnings.warn(
        f"Memory bitstrings have length max {max(len(s) for s in mem) if mem else 0} "
        f"with n_tau={n_tau}, which implies {inf} qubits per round, but num_qubits={num_qubits} "
        f"was passed{extra}. The co-click matrix is therefore {num_qubits}×{num_qubits}. "
        f"Pass num_qubits={inf} if that matches your circuit, or use backend.num_qubits "
        f"for the full device (e.g. 17 on Tuna-17).",
        UserWarning,
        stacklevel=2,
    )


def co_click_probability_matrix(stack: np.ndarray) -> np.ndarray:
    """
    Qubit × qubit matrix ``C`` with ``C[i, j] = 𝔼[x_i x_j]`` over all shots and τ steps.

    For binary readout bits, **off-diagonal** ``i ≠ j``: empirical **P(both qubits read 1)**
    on the same (shot, τ) event. **Diagonal** ``C[i,i] = P(qubit i reads 1)** (a marginal,
    not a pair — often less interesting for correlation). Use
    ``plot_qubit_coclick_matrix(..., mask_diagonal=True)`` to hide diagonals in the figure.
    ``stack`` shape ``(n_shots, n_qubits, n_tau)``.

    Rows of the design matrix must be **one (shot, τ) event** with entries ``x_j =
    stack[shot, j, τ]``. A naive ``reshape(n_shots * n_tau, n_qubits)`` on C-ordered
    ``(n_shots, n_qubits, n_tau)`` is wrong: the fastest axis is τ, so it would string
    τ-slices of a **single** qubit into one row instead of all qubits at fixed τ.
    """
    n_s, n_q, n_t = stack.shape
    if n_s * n_t == 0:
        return np.zeros((n_q, n_q), dtype=np.float64)
    # (n_s, n_q, n_t) → (n_s, n_t, n_q) so each length-n_q row is stack[s, :, k]
    f = np.transpose(stack, (0, 2, 1)).reshape(n_s * n_t, n_q).astype(np.float64, copy=False)
    return (f.T @ f) / float(f.shape[0])


def co_click_excess_matrix(stack: np.ndarray) -> np.ndarray:
    """
    Bernoulli **covariance** (excess joint probability vs independent marginals):

    ``E[i,j] = C[i,j] - P(x_i=1) P(x_j=1)`` with ``C`` from ``co_click_probability_matrix``,
    and ``P(x_i=1) = C[i,i]``. Off-diagonal: ``P(both 1) - P(i=1)P(j=1)``. Diagonal:
    ``p_i(1-p_i)`` (readout variance).
    """
    c = co_click_probability_matrix(stack)
    p = np.clip(np.diag(c), 0.0, 1.0)
    return c - np.outer(p, p)


def mean_bit_tau_by_repetition(stack: np.ndarray) -> np.ndarray:
    """
    ``(n_tau, n_shots)``: ``M[t, s] = mean_q stack[s, q, t]`` — mean readout ``1`` rate
    across qubits at delay index ``t`` and repetition ``s``. Rows are τ, columns repetitions.
    """
    # stack (n_s, n_q, n_t) → mean over axis 1 → (n_s, n_t) → transpose to (n_tau, n_shots)
    return stack.astype(np.float64).mean(axis=1).T


def plot_qubit_coclick_matrix(
    matrix: np.ndarray,
    out_path: Path,
    *,
    title: str = "Pairwise P(both read 1); diagonal masked (single-qubit marginals)",
    dpi: int = 150,
    figsize: Tuple[float, float] = (5.5, 5.0),
    mask_diagonal: bool = True,
) -> None:
    """
    Grayscale heatmap: black = 0, white = 1.

    If ``mask_diagonal`` is True, diagonal entries are not shown (they are **P(qubit i = 1)**,
    not pairwise). Off-diagonals are **P(both i and j read 1)** on the same shot and τ.
    Colormap **vmax** is the **off-diagonal maximum** (black = 0, white = that max), so small
    correlations use the full grey range. If ``mask_diagonal`` is False, **vmax = 1**.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = matrix.shape[0]
    plot_m = matrix.astype(np.float64, copy=True)
    tmp = plot_m.copy()
    np.fill_diagonal(tmp, np.nan)
    off_max = float(np.nanmax(tmp))
    if mask_diagonal:
        np.fill_diagonal(plot_m, np.nan)
        if not np.isfinite(off_max) or off_max <= 0.0:
            vmax = 1.0
        else:
            vmax = off_max
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad((0.92, 0.92, 0.88))
    im = ax.imshow(
        plot_m,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xlabel("Qubit index j")
    ax.set_ylabel("Qubit index i")
    if n <= 24:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(12))
        ax.yaxis.set_major_locator(MaxNLocator(12))
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=f"P(both 1), i≠j (white = {vmax:.4g})",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_qubit_coclick_excess_matrix(
    excess: np.ndarray,
    out_path: Path,
    *,
    title: str = "Co-click excess: P(both 1) − P(i=1)P(j=1)",
    dpi: int = 150,
    figsize: Tuple[float, float] = (5.5, 5.0),
    mask_diagonal: bool = True,
) -> None:
    """Diverging heatmap: centered at 0, symmetric scale from max |off-diagonal|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = excess.shape[0]
    plot_m = excess.astype(np.float64, copy=True)
    tmp = plot_m.copy()
    np.fill_diagonal(tmp, np.nan)
    v = float(np.nanmax(np.abs(tmp)))
    if mask_diagonal:
        np.fill_diagonal(plot_m, np.nan)
    if not np.isfinite(v) or v <= 0.0:
        v = 0.25

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0.92, 0.92, 0.88))
    im = ax.imshow(
        plot_m,
        cmap=cmap,
        vmin=-v,
        vmax=v,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xlabel("Qubit index j")
    ax.set_ylabel("Qubit index i")
    if n <= 24:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(12))
        ax.yaxis.set_major_locator(MaxNLocator(12))
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="excess (covariance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_tau_vs_repetition_mean_qubits(
    grid: np.ndarray,
    out_path: Path,
    *,
    title: str = "Mean readout 1 (averaged over qubits)",
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    ``grid`` shape ``(n_tau, n_shots)``: rows = τ index (y), columns = repetition (x).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tau, n_rep = grid.shape
    if figsize is None:
        w = min(12.0, max(6.0, 0.12 * n_rep + 2.0))
        h = min(10.0, max(4.0, 0.22 * n_tau + 2.0))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        grid,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("Repetition (shot index)")
    ax.set_ylabel(r"$\tau$ index")
    ax.set_xticks(np.arange(0, n_rep, max(1, n_rep // 12)))
    ax.set_yticks(np.arange(0, n_tau, max(1, n_tau // 12)))
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="mean bit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _corner_x(
    s: int,
    j: float,
    k: float,
    *,
    plane_spacing: float,
    tilt_tau: float,
    tilt_qubit: float,
) -> float:
    """X (repetition axis): base position of shot ``s`` plus optional slant in τ / qubit."""
    return s * plane_spacing + tilt_tau * k + tilt_qubit * j


def _plot_memory_cells_3d(
    stack: np.ndarray,
    out_path: Union[Path, BinaryIO],
    *,
    tilt_tau: float,
    tilt_qubit: float,
    plane_spacing: float = 1.0,
    figsize: Tuple[float, float] = (5.0, 5.0),
    fill_alpha: float = 0.50,
    edge_color: str = "0.25",
    edge_linewidth: float = 0.2,
    dpi: int = 300,
    title: str,
    shot_end_exclusive: Optional[int] = None,
    n_shots_axis: Optional[int] = None,
) -> None:
    """
    3D layout: X = shot (repetition), Y = qubit, Z = τ index (depth “into” the sheet).

    Only cells with bit 1 get a translucent fill; bit 0 stays empty. Every cell
    gets a wire outline. Matplotlib axis grids are disabled.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    n_shots, n_q, n_tau = stack.shape
    draw_upto = n_shots if shot_end_exclusive is None else min(shot_end_exclusive, n_shots)
    n_axis = n_shots if n_shots_axis is None else n_shots_axis

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ~2:1 emphasis on repetition axis vs. the in-sheet axes
    span_x = max(1.0, (n_axis - 1) * plane_spacing + abs(tilt_tau) * max(n_tau - 1, 0) + abs(tilt_qubit) * max(n_q - 1, 0))
    span_y = float(max(1, n_q))
    span_z = float(max(1, n_tau))
    ax.set_box_aspect((4.0 * span_x, span_y, span_z))

    face_verts: List[np.ndarray] = []
    face_colors: List[Tuple[float, float, float, float]] = []
    edge_segments: List[Tuple[np.ndarray, np.ndarray]] = []

    black_fill = (0.0, 0.0, 0.0, float(fill_alpha))

    for s in range(draw_upto):
        for j in range(n_q):
            for k in range(n_tau):
                j0, j1 = float(j), float(j + 1)
                k0, k1 = float(k), float(k + 1)
                x00 = _corner_x(s, j0, k0, plane_spacing=plane_spacing, tilt_tau=tilt_tau, tilt_qubit=tilt_qubit)
                x01 = _corner_x(s, j0, k1, plane_spacing=plane_spacing, tilt_tau=tilt_tau, tilt_qubit=tilt_qubit)
                x11 = _corner_x(s, j1, k1, plane_spacing=plane_spacing, tilt_tau=tilt_tau, tilt_qubit=tilt_qubit)
                x10 = _corner_x(s, j1, k0, plane_spacing=plane_spacing, tilt_tau=tilt_tau, tilt_qubit=tilt_qubit)
                quad = np.array(
                    [
                        [x00, j0, k0],
                        [x01, j0, k1],
                        [x11, j1, k1],
                        [x10, j1, k0],
                    ],
                    dtype=float,
                )
                if stack[s, j, k]:
                    face_verts.append(quad)
                    face_colors.append(black_fill)
                p00, p01, p11, p10 = quad[0], quad[1], quad[2], quad[3]
                edge_segments.extend(
                    [
                        (p00, p01),
                        (p01, p11),
                        (p11, p10),
                        (p10, p00),
                    ]
                )

    if face_verts:
        faces = Poly3DCollection(
            face_verts,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0.0,
            zsort="average",
        )
        ax.add_collection3d(faces)

    if edge_segments:
        seg_array = np.array(edge_segments, dtype=float)
        lines = Line3DCollection(
            seg_array,
            colors=edge_color,
            linewidths=edge_linewidth,
            alpha=0.85,
        )
        ax.add_collection3d(lines)

    pad_x = 0.35 * plane_spacing + 0.15 * (abs(tilt_tau) * n_tau + abs(tilt_qubit) * n_q)
    ax.set_xlim(-pad_x, span_x + pad_x)
    ax.set_ylim(-0.5, span_y + 0.5)
    ax.set_zlim(-0.5, span_z + 0.5)

    ax.set_xlabel("Shot (repetition)")
    ax.set_ylabel("Qubit")
    ax.set_zlabel(r"$\tau$ index")
    ax.view_init(elev=20, azim=-80)
    if title:
        ax.set_title(title, fontsize=10)

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0

    fig.tight_layout()
    save_kw: dict = {"dpi": dpi}
    if isinstance(out_path, io.BytesIO):
        save_kw["format"] = "png"
    fig.savefig(out_path, **save_kw)
    plt.close(fig)


def plot_memory_3d_tilted_planes(
    stack: np.ndarray,
    out_path: Union[Path, BinaryIO],
    *,
    plane_spacing: float = 1.0,
    tilt_tau: float = 0.0,
    tilt_qubit: float = 0.00,
    fill_alpha: float = 0.22,
    dpi: int = 150,
    title: str = "Measurement bitstrings (tilted sheets)",
    shot_end_exclusive: Optional[int] = None,
    n_shots_axis: Optional[int] = None,
) -> None:
    """Sheets normal to X (shots), slight slant in X vs. τ / qubit for depth cue."""
    _plot_memory_cells_3d(
        stack,
        out_path,
        tilt_tau=tilt_tau,
        tilt_qubit=tilt_qubit,
        plane_spacing=plane_spacing,
        title=title,
        fill_alpha=fill_alpha,
        dpi=dpi,
        shot_end_exclusive=shot_end_exclusive,
        n_shots_axis=n_shots_axis,
    )


def save_memory_repetitions_gif(
    stack: np.ndarray,
    out_path: Path,
    *,
    plane_spacing: float = 1.0,
    tilt_tau: float = 0.0,
    tilt_qubit: float = 0.0,
    fill_alpha: float = 0.22,
    dpi: int = 120,
    fps: float = 2.5,
) -> None:
    """
    Animated GIF: cumulative tilted 3D view as repetitions (shots) increase.

    Axis span is fixed to the full ``stack`` so the camera scale stays stable; each
    frame adds the next shot along X.
    """
    from PIL import Image

    n_shots = stack.shape[0]
    if n_shots == 0:
        return

    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    frames: List[Image.Image] = []
    n_axis = n_shots
    for k in range(1, n_shots + 1):
        buf = io.BytesIO()
        plot_memory_3d_tilted_planes(
            stack,
            buf,
            plane_spacing=plane_spacing,
            tilt_tau=tilt_tau,
            tilt_qubit=tilt_qubit,
            fill_alpha=fill_alpha,
            dpi=dpi,
            title=f"Measurement bitstrings (tilted) — repetitions 1–{k} of {n_shots}",
            shot_end_exclusive=k,
            n_shots_axis=n_axis,
        )
        buf.seek(0)
        im = Image.open(buf)
        frames.append(im.copy())
        im.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _plot_memory_2d_q_time_slice(
    grid: np.ndarray,
    out_path: Union[Path, BinaryIO],
    *,
    dpi: int,
    title: str,
    figsize: Tuple[float, float],
) -> None:
    """``grid`` shape ``(n_qubits, n_tau)``: rows = qubit, columns = τ (time) index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n_q, n_tau = grid.shape
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("Greys").copy()
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\tau$ index (time)")
    ax.set_ylabel("Qubit")
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.yaxis.set_major_locator(MaxNLocator(10))
    if title:
        ax.set_title(title, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.15)
    save_kw: dict = {"dpi": dpi}
    if isinstance(out_path, io.BytesIO):
        save_kw["format"] = "png"
    fig.savefig(out_path, **save_kw)
    plt.close(fig)


def _format_lab_duration(seconds: float) -> str:
    """Duration label shown in display-seconds for blog readability."""
    s = float(seconds) * 1000.0
    if abs(s - round(s)) < 1e-9:
        return f"{int(round(s))} s"
    return f"{s:.3f} s"


def save_memory_repetitions_gif_2d(
    cube: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 72,
    fps: float = 1.5,
    figsize: Optional[Tuple[float, float]] = None,
    moving_average_window: Optional[int] = None,
    rep_bin_size: int = 1,
    lab_time_per_shot_s: float = 20 * 600e-9,
    keep_frame_pngs: bool = False,
    tau_ns: Optional[Sequence[float]] = None,
    marginal_history: int = 10,
) -> None:
    """
    2D GIF over repetition index ``k``: frame ``k`` shows ``data[:, :, k]`` (qubits × τ).

    ``cube`` must be ``(n_qubits, n_tau, n_repetitions)`` (see ``memory_cube_q_time_rep``).

    **Binning:** if ``rep_bin_size > 1``, consecutive shots are averaged so the GIF has
    ``ceil(n_rep / rep_bin_size)`` frames (each frame is the mean over that block along
    repetitions). Progress and lab-time labels use the **underlying** shot count.

    **Moving average (legacy):** if ``rep_bin_size <= 1`` and ``moving_average_window`` is set,
    each frame uses a centered mean along repetitions (edges use shorter windows). If
    ``rep_bin_size > 1``, the moving-average path is ignored in favor of block means.

    **Layout:** progress bar; **top** marginal = mean over qubits vs ``tau_ns`` [ns]; **main**
    heatmap; **right** marginal = mean over τ vs ``Q0…``. Recent frames are darker;
    older ones fade (``marginal_history`` traces).

    ``tau_ns`` length must match ``n_tau`` (delay in nanoseconds per column). If ``None``,
    uses ``100 * np.arange(n_tau)`` to match ``tau_ns_from_indices`` defaults.

    Uses one Matplotlib figure per frame saved as PNG, then stitched into a GIF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_q, n_tau, n_raw_rep = cube.shape
    if n_raw_rep == 0:
        return

    k_bin = max(1, int(rep_bin_size))
    if k_bin > 1 and moving_average_window is not None:
        warnings.warn(
            "save_memory_repetitions_gif_2d: rep_bin_size>1 ignores moving_average_window",
            UserWarning,
            stacklevel=2,
        )
    if k_bin > 1:
        data = cube_repetition_block_mean(cube, k_bin)
    elif moving_average_window is not None:
        data = moving_average_along_repetitions(cube, window=moving_average_window)
    else:
        data = cube.astype(np.float64, copy=False)

    n_q, n_tau, n_rep = data.shape

    if tau_ns is None:
        tau_arr = (100.0 * np.arange(n_tau, dtype=np.float64)).reshape(-1)
    else:
        tau_arr = np.asarray(tau_ns, dtype=np.float64).reshape(-1)
    if tau_arr.size != n_tau:
        raise ValueError(f"tau_ns length {tau_arr.size} != n_tau={n_tau}")

    tau_us = tau_arr / 1000.0
    if n_tau > 1:
        dtau = float(np.median(np.diff(tau_us)))
    else:
        dtau = max(0.05, 0.1 * float(abs(tau_us[0])))
    x_left = min(0.0, float(tau_us[0]) - dtau / 2.0)
    x_right = max(10.0, float(tau_us[-1]) + dtau / 2.0)

    style = zebra_gif_2d_style(n_q, n_tau)
    if figsize is None:
        style_figsize = style["figsize"]
        assert isinstance(style_figsize, tuple)
        figsize = style_figsize

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_path.parent / f"{out_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    t_total_s = n_raw_rep * lab_time_per_shot_s

    if k_bin > 1:
        ma_suffix = f"  bin{k_bin}"
    elif moving_average_window is not None:
        ma_suffix = f"  MA{moving_average_window}"
    else:
        ma_suffix = ""

    def cumulative_shots_after_frame(frame_idx: int) -> int:
        if k_bin > 1:
            return min((frame_idx + 1) * k_bin, n_raw_rep)
        return frame_idx + 1
    hist_n = max(1, int(marginal_history))

    # Fewer τ tick labels when many delays
    if n_tau <= 16:
        tick_ix = np.arange(n_tau, dtype=int)
    else:
        tick_ix = np.unique(
            np.linspace(0, n_tau - 1, num=min(14, n_tau), dtype=int)
        )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.36, 1.0],
        width_ratios=[1.0, 0.38],
        hspace=float(style["hspace_inner"]),
        wspace=float(style["wspace_inner"]),
        left=float(style["left"]),
        right=float(style["right"]),
        top=float(style["top"]),
        bottom=float(style["bottom"]),
    )
    ax_top = fig.add_subplot(gs[0, 0])            # [0,0]
    ax_info = fig.add_subplot(gs[0, 1])           # [0,1]
    ax_main = fig.add_subplot(gs[1, 0])           # [1,0]
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)  # [1,1]

    cmap = plt.get_cmap("Greys").copy()
    extent_xy = (x_left, x_right, float(n_q) - 0.5, -0.5)
    im = ax_main.imshow(
        data[:, :, 0],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=extent_xy,
    )
    label_fs = int(style["label_fs"])
    tick_fs_main_x = int(style["tick_fs_main_x"])
    tick_fs_main_y = int(style["tick_fs_main_y"])
    tick_fs_marg_x = int(style["tick_fs_marg_x"])
    marginal_label_fs = int(style["marginal_label_fs"])
    suptitle_fs = int(style["suptitle_fs"])
    tick_fs = min(tick_fs_main_x, tick_fs_main_y, tick_fs_marg_x)

    ax_main.set_xlabel(r"waiting time $\tau_i$ (us)", fontsize=label_fs, labelpad=4)
    ax_main.set_ylabel("Qubit", fontsize=label_fs)
    ax_main.set_xticks(np.arange(0.0, 11.0, 1.0))
    ax_main.set_xticklabels(
        [f"{int(v)}" for v in np.arange(0.0, 11.0, 1.0)],
        rotation=45,
        ha="right",
        fontsize=tick_fs,
    )
    ax_main.set_yticks(np.arange(n_q, dtype=float))
    ax_main.set_yticklabels([f"Q{i}" for i in range(n_q)], fontsize=tick_fs)
    ax_main.tick_params(axis="both", labelsize=tick_fs)
    ax_top.set_ylabel(r"fraction flipped $<\cdot>_q$", fontsize=marginal_label_fs)
    ax_top.set_xlim(x_left, x_right)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_xticks(tau_us[tick_ix])
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", which="both", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=tick_fs)
    ax_top.grid(True, alpha=0.25, linewidth=0.5)
    ax_right.set_xlabel(r"fraction flipped $<\cdot>_\tau$", fontsize=marginal_label_fs)
    ax_right.set_xlim(0.0, 1.0)
    ax_right.tick_params(axis="x", labelsize=tick_fs)
    ax_right.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax_right.yaxis.tick_left()
    plt.setp(ax_right.get_yticklabels(), visible=False)
    base_rgb = np.array([0.20, 0.45, 0.75])

    for k in range(n_rep):
        cum_shots = cumulative_shots_after_frame(k)
        frac = cum_shots / float(n_raw_rep)
        im.set_data(data[:, :, k])

        ax_info.clear()
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(-0.5, 0.8)
        ax_info.axis("off")
        ax_info.text(
            0.0,
            0.68,
            (
                f"lab-time {_format_lab_duration(cum_shots * lab_time_per_shot_s)}"
                f" / {_format_lab_duration(t_total_s)}"
            ),
            ha="left",
            va="center",
            fontsize=suptitle_fs,
        )
        ax_info.barh(
            0,
            1.0,
            height=0.45,
            left=0,
            color="0.92",
            edgecolor="0.35",
            linewidth=1.0,
        )
        ax_info.barh(
            0,
            frac,
            height=0.45,
            left=0,
            color="steelblue",
            linewidth=0,
        )

        k0 = max(0, k - hist_n + 1)
        ax_top.clear()
        ax_top.set_ylabel(r"fraction flipped $<\cdot>_q$", fontsize=marginal_label_fs)
        ax_top.set_xlim(x_left, x_right)
        ax_top.set_ylim(0.0, 1.0)
        ax_top.set_xticks(tau_us[tick_ix])
        ax_top.set_xticklabels([])
        ax_top.tick_params(axis="x", which="both", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=tick_fs)
        ax_top.grid(True, alpha=0.25, linewidth=0.5)
        n_lines = k - k0 + 1
        for j, kk in enumerate(range(k0, k + 1)):
            rel = (j + 1) / float(n_lines)
            alpha = 0.12 + 0.88 * rel
            lw = 0.7 + 1.6 * rel
            m_top = data[:, :, kk].mean(axis=0)
            ax_top.plot(
                tau_us,
                m_top,
                color=tuple(base_rgb),
                alpha=alpha,
                linewidth=lw,
                clip_on=True,
            )

        ax_right.clear()
        ax_right.set_xlabel(r"fraction flipped $<\cdot>_\tau$", fontsize=marginal_label_fs)
        ax_right.set_xlim(0.0, 1.0)
        ax_right.tick_params(axis="x", labelsize=tick_fs)
        ax_right.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        ax_right.yaxis.tick_left()
        plt.setp(ax_right.get_yticklabels(), visible=False)
        y_idx = np.arange(n_q, dtype=float)
        for j, kk in enumerate(range(k0, k + 1)):
            rel = (j + 1) / float(n_lines)
            alpha = 0.12 + 0.88 * rel
            lw = 0.7 + 1.6 * rel
            m_r = data[:, :, kk].mean(axis=1)
            ax_right.plot(
                m_r,
                y_idx,
                color=tuple(base_rgb),
                alpha=alpha,
                linewidth=lw,
                clip_on=True,
            )

        fp = frame_dir / f"frame_{k:04d}.png"
        fig.savefig(fp, dpi=dpi)

    plt.close(fig)

    frame_paths = [frame_dir / f"frame_{k:04d}.png" for k in range(n_rep)]
    frames_pil: List[Image.Image] = []
    for fp in frame_paths:
        im_p = Image.open(fp)
        frames_pil.append(im_p.copy())
        im_p.close()

    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
    )

    if not keep_frame_pngs:
        shutil.rmtree(frame_dir, ignore_errors=True)


def save_memory_3d_plots(
    mem: List[str],
    num_qubits: int,
    n_tau: int,
    out_dir: Path,
    *,
    max_shots: int,
    prefix: str = "tuna_fid_3d",
    include_3d: bool = False,
    gif_fps: float = 2.5,
    gif_dpi: int = 120,
    gif_2d_fps: float = 4,
    gif_2d_dpi: int = 100,
    gif_2d_rep_bin: int = 10,
    gif_2d_write_per_shot: bool = False,
    gif_2d_lab_time_per_shot_s: float = 20 * 600e-9,
    include_derived: bool = True,
    derived_dpi: int = 150,
    save_derived_npz: bool = True,
    gif_2d_keep_frames: bool = False,
    tau_ns: Optional[Sequence[float]] = None,
    gif_2d_marginal_history: int = 10,
    reset_qubits: bool = True,
    differential_implicit_prior: int = 0,
) -> Tuple[
    Optional[Path],
    Optional[Path],
    Path,
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Optional[Path],
]:
    """
    2D slice GIF (default): block-averaged along repetitions (``gif_2d_rep_bin`` shots per
    frame, Matplotlib ``imshow``, frame PNGs → GIF). Set ``gif_2d_write_per_shot`` to also
    write one frame per shot (can be very large).

    If ``include_3d`` is True, also writes the tilted PNG and cumulative 3D GIF (slow).

    If ``include_derived`` is True, also writes:

    - qubit×qubit **co-click** probability (raw joint P(both 1));
    - **co-click excess** (covariance) ``P(both 1) − P(i=1)P(j=1)``;
    - **τ × repetition** heatmap (mean over qubits);

    and if ``save_derived_npz`` is True, ``{prefix}_memory_derived.npz`` with ``co_click``,
    ``co_click_excess``, ``tau_vs_repetition_mean_q``.

    If ``reset_qubits`` is False, raw Z outcomes are mapped along τ with
    :func:`differential_readout_along_tau` before all plots and derived arrays (flip
    detection vs the previous τ bin; first bin vs ``differential_implicit_prior``,
    default ``0``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(mem) <= max_shots:
        use = mem
    else:
        idx = np.linspace(0, len(mem) - 1, max_shots, dtype=int)
        use = [mem[int(i)] for i in np.unique(idx)]
    warn_if_memory_qubit_mismatch(use, n_tau, num_qubits, context="(save_memory_3d_plots)")
    stack = memory_to_stack(use, num_qubits, n_tau)
    if not reset_qubits:
        stack = differential_readout_along_tau(
            stack, implicit_prior=differential_implicit_prior
        )
    cube = memory_cube_q_time_rep(stack)
    p_png: Optional[Path] = out_dir / f"{prefix}_tilted.png"
    p_gif_3d: Optional[Path] = out_dir / f"{prefix}_repetitions.gif"
    kb = max(1, int(gif_2d_rep_bin))
    if kb > 1:
        p_gif_2d = out_dir / f"{prefix}_repetitions_2d_bin{kb}.gif"
    else:
        p_gif_2d = out_dir / f"{prefix}_repetitions_2d.gif"
    p_gif_2d_per_shot: Optional[Path] = (
        (out_dir / f"{prefix}_repetitions_2d.gif") if gif_2d_write_per_shot and kb > 1 else None
    )
    p_coclick: Optional[Path] = out_dir / f"{prefix}_qubit_coclick.png"
    p_coclick_excess: Optional[Path] = out_dir / f"{prefix}_qubit_coclick_excess.png"
    p_tau_rep: Optional[Path] = out_dir / f"{prefix}_tau_vs_repetition_meanq.png"
    p_derived_npz: Optional[Path] = out_dir / f"{prefix}_memory_derived.npz"

    if include_3d:
        plot_memory_3d_tilted_planes(stack, p_png, dpi=150)
        save_memory_repetitions_gif(
            stack,
            p_gif_3d,
            fps=gif_fps,
            dpi=gif_dpi,
        )
    else:
        p_png = None
        p_gif_3d = None

    save_memory_repetitions_gif_2d(
        cube,
        p_gif_2d,
        fps=gif_2d_fps,
        dpi=gif_2d_dpi,
        moving_average_window=None,
        rep_bin_size=kb,
        lab_time_per_shot_s=gif_2d_lab_time_per_shot_s,
        keep_frame_pngs=gif_2d_keep_frames,
        tau_ns=tau_ns,
        marginal_history=gif_2d_marginal_history,
    )
    if p_gif_2d_per_shot is not None:
        save_memory_repetitions_gif_2d(
            cube,
            p_gif_2d_per_shot,
            fps=gif_2d_fps,
            dpi=gif_2d_dpi,
            moving_average_window=None,
            rep_bin_size=1,
            lab_time_per_shot_s=gif_2d_lab_time_per_shot_s,
            keep_frame_pngs=gif_2d_keep_frames,
            tau_ns=tau_ns,
            marginal_history=gif_2d_marginal_history,
        )

    if include_derived:
        cc = co_click_probability_matrix(stack)
        exc = co_click_excess_matrix(stack)
        tr = mean_bit_tau_by_repetition(stack)
        plot_qubit_coclick_matrix(cc, p_coclick, dpi=derived_dpi)
        plot_qubit_coclick_excess_matrix(exc, p_coclick_excess, dpi=derived_dpi)
        plot_tau_vs_repetition_mean_qubits(tr, p_tau_rep, dpi=derived_dpi)
        if save_derived_npz:
            np.savez_compressed(
                p_derived_npz,
                co_click=cc,
                co_click_excess=exc,
                tau_vs_repetition_mean_q=tr,
            )
        else:
            p_derived_npz = None
    else:
        p_coclick = None
        p_coclick_excess = None
        p_tau_rep = None
        p_derived_npz = None

    return (
        p_png,
        p_gif_3d,
        p_gif_2d,
        p_gif_2d_per_shot,
        p_coclick,
        p_coclick_excess,
        p_tau_rep,
        p_derived_npz,
    )


def display_image_path(path: Union[str, Path]) -> None:
    """
    Show a PNG or GIF in Jupyter.

    ``IPython.display.Image(filename=...)`` can raise *ValueError: Cannot embed the 'none'
    image format* for some GIFs because Pillow leaves ``Image.format`` unset; reading bytes
    and passing ``format='gif'`` avoids that.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    from IPython.display import Image, display

    if p.suffix.lower() == ".gif":
        display(Image(data=p.read_bytes(), format="gif"))
    else:
        display(Image(filename=str(p)))


def main() -> None:
    parser = argparse.ArgumentParser(description="FID / Ramsey echo, all taus in one job")
    parser.add_argument("--backend", default="Tuna-17", help="Quantum Inspire backend name")
    parser.add_argument("--i-start", type=int, default=1, help="first index i in τ_i = 100*i ns")
    parser.add_argument("--i-end", type=int, default=51, help="end index (exclusive) for i")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="only build & print circuit stats")
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=None,
        help="override qubit count (e.g. 17 for Tuna-17) for dry-run without API",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="after a real run, save 2D slice GIF (binned reps; see --gif-2d-rep-bin); use --plot-3d for slow 3D too",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="with --plot, also write tilted PNG + cumulative 3D GIF (much slower)",
    )
    parser.add_argument(
        "--plot-no-derived",
        action="store_true",
        help="with --plot, skip qubit co-click, τ×repetition heatmaps, and derived .npz",
    )
    parser.add_argument(
        "--derived-dpi",
        type=int,
        default=150,
        help="PNG resolution for co-click and τ×repetition heatmaps",
    )
    parser.add_argument(
        "--plot-max-shots",
        type=int,
        default=48,
        help="max number of shots to draw (subsampled if memory is larger)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=".",
        help="directory for 3D plot outputs",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=2.5,
        help="frames per second for the 3D tilted repetitions GIF",
    )
    parser.add_argument(
        "--gif-2d-fps",
        type=float,
        default=1.5,
        help="frames per second for the 2D qubit×τ slice GIFs",
    )
    parser.add_argument(
        "--gif-2d-rep-bin",
        type=int,
        default=10,
        help="average this many consecutive shots per 2D GIF frame (ceil(n_shots/k) frames)",
    )
    parser.add_argument(
        "--gif-2d-per-shot",
        action="store_true",
        help="also write a second 2D GIF with one frame per shot (only with --gif-2d-rep-bin > 1; can be huge)",
    )
    parser.add_argument(
        "--gif-dpi",
        type=int,
        default=120,
        help="raster resolution for the 3D cumulative GIF (if --plot-3d)",
    )
    parser.add_argument(
        "--gif-2d-dpi",
        type=int,
        default=72,
        help="raster resolution for 2D slice GIFs (lower = faster, default 72)",
    )
    parser.add_argument(
        "--gif-2d-lab-dt-ns",
        type=float,
        default=20.0 * 600.0,
        help="lab-time per shot for 2D GIF bar/label (ns); default 12000 (= 20×600 ns)",
    )
    parser.add_argument(
        "--gif-2d-keep-frames",
        action="store_true",
        help="keep PNG frame sequences in <gif_stem>_frames/ next to each 2D GIF",
    )
    parser.add_argument(
        "--gif-2d-marg-hist",
        type=int,
        default=10,
        help="number of past repetitions to show (fading) in 2D GIF marginals",
    )
    args = parser.parse_args()

    taus = tau_ns_from_indices(args.i_start, args.i_end)

    if args.dry_run and args.num_qubits is not None:
        n = args.num_qubits
        dt = 20e-9
        qc = build_fid_circuit_all_qubits(n, taus, dt_seconds=dt)
        tqc = transpile(qc, basis_gates=["id", "rz", "sx", "x", "cx", "reset", "delay"])
        print(f"[dry-run] qubits: {n}, tau steps: {len(taus)} (no Quantum Inspire API)")
        print(f"Tau (ns) first/last: {taus[0]} … {taus[-1]}")
        print(f"Circuit depth (transpiled): {tqc.depth()}, size: {tqc.size()}")
        return

    provider = QIProvider()
    backend = provider.get_backend(name=args.backend)
    n = backend.num_qubits

    dt = getattr(backend.configuration(), "dt", None)
    qc = build_fid_circuit_all_qubits(n, taus, dt_seconds=dt)

    tqc = transpile(
        qc,
        backend,
        optimization_level=args.optimization_level,
    )

    print(f"Backend: {args.backend}, qubits: {n}, tau steps: {len(taus)}")
    print(f"Tau (ns) first/last: {taus[0]} … {taus[-1]}")
    print(f"Circuit depth (transpiled): {tqc.depth()}, size: {tqc.size()}")

    if args.dry_run:
        return

    job = backend.run(tqc, shots=args.shots, memory=True)
    result = job.result()
    mem = result.get_memory()
    p0 = survival_prob_0_from_memory(mem, n, len(taus))
    print("P(|0>) shape (n_tau, n_qubits):", p0.shape)
    np.savez_compressed(
        "tuna_fid_single_job.npz",
        tau_ns=np.array(taus),
        p0=p0,
        shots=args.shots,
        backend=args.backend,
    )
    print("Wrote tuna_fid_single_job.npz")

    if args.plot:
        p_png, p_gif_3d, p_gif_2d, p_gif_2d_per_shot, p_cc, p_exc, p_tr, p_dnpz = save_memory_3d_plots(
            mem,
            n,
            len(taus),
            Path(args.plot_dir),
            max_shots=args.plot_max_shots,
            include_3d=args.plot_3d,
            gif_fps=args.gif_fps,
            gif_dpi=args.gif_dpi,
            gif_2d_fps=args.gif_2d_fps,
            gif_2d_dpi=args.gif_2d_dpi,
            gif_2d_rep_bin=args.gif_2d_rep_bin,
            gif_2d_write_per_shot=args.gif_2d_per_shot,
            gif_2d_lab_time_per_shot_s=args.gif_2d_lab_dt_ns * 1e-9,
            include_derived=not args.plot_no_derived,
            derived_dpi=args.derived_dpi,
            gif_2d_keep_frames=args.gif_2d_keep_frames,
            tau_ns=np.array(taus, dtype=float),
            gif_2d_marginal_history=args.gif_2d_marg_hist,
        )
        if p_png is not None:
            print(f"Wrote 3D plot (tilted planes): {p_png}")
        if p_gif_3d is not None:
            print(f"Wrote 3D repetitions GIF: {p_gif_3d}")
        print(f"Wrote 2D slice GIF (binned): {p_gif_2d}")
        if p_gif_2d_per_shot is not None:
            print(f"Wrote 2D slice GIF (per shot): {p_gif_2d_per_shot}")
        if p_cc is not None:
            print(f"Wrote qubit co-click heatmap: {p_cc}")
        if p_exc is not None:
            print(f"Wrote qubit co-click excess heatmap: {p_exc}")
        if p_tr is not None:
            print(f"Wrote τ vs repetition map: {p_tr}")
        if p_dnpz is not None:
            print(f"Wrote derived arrays: {p_dnpz}")


if __name__ == "__main__":
    main()
