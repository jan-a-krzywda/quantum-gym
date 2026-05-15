#!/usr/bin/env python3
"""
Per-qubit latent means (μ₁, μ₂) along the shot axis: trajectory means, correlated-increment
statistics, and optional correlation time from AR(1) / autocorrelation decay.

Uses the same data pipeline as ``train_vae.py`` (differential readout by default).

Writes:

- ``latent_dynamics.json`` / ``latent_dynamics_arrays.npz`` (μ(s,q), Σ, AR(1), …).
- ``latent_drift_covariance.png`` — Σ: **diagonal** cells = Var(Δ) in **YlOrRd**; **off-diagonal**
  shown as ``scale×Cov`` (default scale 10) on **RdBu_r** (NaN diagonal = grey).
- ``latent_drift_correlation.png`` — Pearson **correlation** (same index order).
- ``latent_drift_cov_corr_panel.png`` — side-by-side summary of both.
- ``latent_mu_bar_means_2d.png`` — scatter of per-qubit **(μ̄₁, μ̄₂)** (temporal means).
- ``latent_mu_bar_chip_spatial.png`` (17 qubits) — two panels: **μ̄₁** and **μ̄₂** as colors on the fixed chip layout.

For **latent trajectory GIFs** (means + fading history), run ``plot_latent_dynamics_gif.py``
on ``latent_dynamics_arrays.npz``.

After this, ``simulate_fitted_latent_zebra.py`` can use the saved ``μ̄`` and Σ (OU/RW path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import copy as copy_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch

if not torch.cuda.is_available():
    torch.backends.nnpack.enabled = False

from ML.analysis_processing.fid_data_io import build_stack_like_single_job, load_memory_pickle, stack_to_vae_tensors
from ML.plotting.style import apply_latent_zebra_style
from ML.vae_model import QubitConvVAE

# Tuna-17 style planar layout (arbitrary units; matches heavy-hex style topology illustration).
# Index = logical qubit id Q0…Q16.
TUNA17_QUBIT_XY: Dict[int, Tuple[float, float]] = {
    0: (2, 6),
    1: (1, 5),
    2: (3, 5),
    3: (5, 5),
    4: (2, 4),
    5: (4, 4),
    6: (6, 4),
    7: (1, 3),
    8: (3, 3),
    9: (5, 3),
    10: (0, 2),
    11: (2, 2),
    12: (4, 2),
    13: (1, 1),
    14: (3, 1),
    15: (5, 1),
    16: (4, 0),
}

TUNA17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 4),
    (2, 4),
    (2, 5),
    (3, 5),
    (3, 6),
    (4, 7),
    (4, 8),
    (5, 8),
    (5, 9),
    (6, 9),
    (7, 10),
    (7, 11),
    (8, 11),
    (8, 12),
    (9, 12),
    (10, 13),
    (11, 13),
    (11, 14),
    (12, 14),
    (12, 15),
    (14, 16),
    (15, 16),
)


def _pick_device(prefer: str) -> torch.device:
    prefer = prefer.lower().strip()
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if prefer == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer != "auto":
        raise ValueError("--device must be one of: auto, cpu, cuda, mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flat_rows_to_shot_qubit(mu_flat: np.ndarray, n_shots: int, n_qubits: int) -> np.ndarray:
    """``(n_shots * n_qubits, L)`` with row order ``tile(0..Q-1, n_shots)`` → ``(n_shots, n_qubits, L)``."""
    n_r, L = mu_flat.shape
    if n_r != n_shots * n_qubits:
        raise ValueError(f"flat rows {n_r} != n_shots * n_qubits = {n_shots * n_qubits}")
    out = np.empty((n_shots, n_qubits, L), dtype=mu_flat.dtype)
    for s in range(n_shots):
        out[s] = mu_flat[s * n_qubits : (s + 1) * n_qubits]
    return out


@torch.no_grad()
def encode_mu_batched(
    model: QubitConvVAE,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return encoder latent means ``μ``, shape ``(N, latent_dim)``."""
    model.eval()
    use_cuda = device.type == "cuda"
    outs: List[torch.Tensor] = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        batch = x[i : i + batch_size].to(device, non_blocking=use_cuda)
        enc = model.encoder(batch)
        mu = model.fc_mu(enc)
        outs.append(mu.cpu())
    return torch.cat(outs, dim=0).numpy()


def joint_increment_covariance(
    mu_sq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``mu_sq`` shape ``(n_shots, n_qubits, L)``.

    Returns
    -------
    delta : (S-1, n_qubits, L) shot-to-shot increments
    Delta : (S-1, n_qubits * L) stacked for joint covariance
    Sigma : (n_qubits * L, n_qubits * L) empirical covariance of increments
    """
    n_shots, n_q, L = mu_sq.shape
    delta = mu_sq[1:] - mu_sq[:-1]
    Delta = delta.reshape(n_shots - 1, n_q * L)
    Delta_c = Delta - Delta.mean(axis=0, keepdims=True)
    n_eff = max(1, Delta_c.shape[0] - 1)
    Sigma = (Delta_c.T @ Delta_c) / n_eff
    return delta, Delta, Sigma


def per_qubit_increment_cov(delta: np.ndarray) -> np.ndarray:
    """``delta`` (S-1, Q, L) → list-style stack of 2×2 blocks on diagonal would be separate;
    here return shape ``(Q, L, L)`` per-qubit increment covariance."""
    n_m1, n_q, L = delta.shape
    out = np.zeros((n_q, L, L), dtype=np.float64)
    dmean = delta.mean(axis=0)
    dc = delta - dmean
    n_eff = max(1, n_m1 - 1)
    for q in range(n_q):
        Dq = dc[:, q, :]
        out[q] = (Dq.T @ Dq) / n_eff
    return out


def ar1_rho_and_tau_steps(centered: np.ndarray) -> Tuple[float, float]:
    """
    OLS AR(1) fit X_{t+1} = ρ X_t + ε on centered series (1D): ρ = Σ x_t x_{t+1} / Σ x_t².
    Returns (ρ, τ_corr) where τ_corr = -1/log(|ρ|) steps if 0 < |ρ| < 1, else nan.
    """
    x = np.asarray(centered, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan"), float("nan")
    x0, x1 = x[:-1], x[1:]
    denom = float(np.dot(x0, x0))
    if denom < 1e-18:
        return float("nan"), float("nan")
    rho = float(np.dot(x0, x1) / denom)
    rho = max(min(rho, 0.9999), -0.9999)
    if abs(rho) < 1e-10 or abs(rho) >= 1.0:
        return rho, float("nan")
    tau = float(-1.0 / np.log(abs(rho)))
    return rho, tau


def plot_latent_mu_bar_2d(
    mu_bar_q: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Scatter of per-qubit temporal means (μ̄₁, μ̄₂) in the latent plane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_latent_zebra_style()

    n_q, ld = mu_bar_q.shape
    x = mu_bar_q[:, 0].astype(np.float64, copy=False)
    y = mu_bar_q[:, 1].astype(np.float64, copy=False) if ld > 1 else np.zeros(n_q, dtype=np.float64)

    try:
        cmap = matplotlib.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8.5, 6.8), dpi=dpi)
    for q in range(n_q):
        color = cmap(q / max(n_q - 1, 1))
        ax.scatter(
            [x[q]],
            [y[q]],
            c=[color],
            s=140,
            edgecolors="0.2",
            linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            str(q),
            (float(x[q]), float(y[q])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="0.15",
        )
    ax.set_xlabel(r"$\bar{\mu}_1$ (mean of encoder $\mu_1$ over shots)")
    ax.set_ylabel(r"$\bar{\mu}_2$")
    ax.set_title(
        "Per-qubit temporal mean of latent encoder outputs\n"
        r"(each point is one qubit's $\bar{\boldsymbol{\mu}}$ in the 2D latent plane)"
    )
    ax.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mu_bar_chip_spatial(
    mu_bar_q: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """
    Two-panel figure: ``μ̄₁`` and ``μ̄₂`` as node colors on the Tuna-17 planar layout
    (only if ``mu_bar_q.shape[0] == 17`` and latent dim ≥ 2).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_latent_zebra_style()

    n_q, ld = mu_bar_q.shape
    if n_q != len(TUNA17_QUBIT_XY) or ld < 2:
        return

    m1 = mu_bar_q[:, 0].astype(np.float64, copy=False)
    m2 = mu_bar_q[:, 1].astype(np.float64, copy=False)
    xs = np.array([TUNA17_QUBIT_XY[q][0] for q in range(n_q)], dtype=np.float64)
    ys = np.array([TUNA17_QUBIT_XY[q][1] for q in range(n_q)], dtype=np.float64)

    def _norm_for(vals: np.ndarray) -> Tuple[float, float]:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi - lo < 1e-12:
            lo -= 0.5
            hi += 0.5
        return lo, hi

    lo1, hi1 = _norm_for(m1)
    lo2, hi2 = _norm_for(m2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    cmap = _mpl_get_cmap("coolwarm")

    for ax, vals, lo, hi, title, cb_label in (
        (axes[0], m1, lo1, hi1, r"$\bar{\mu}_1$ on chip layout", r"$\bar{\mu}_1$"),
        (axes[1], m2, lo2, hi2, r"$\bar{\mu}_2$ on chip layout", r"$\bar{\mu}_2$"),
    ):
        ax.set_facecolor("white")
        for a, b in TUNA17_EDGES:
            xa, ya = TUNA17_QUBIT_XY[a]
            xb, yb = TUNA17_QUBIT_XY[b]
            ax.plot([xa, xb], [ya, yb], color="0.45", linewidth=1.8, alpha=0.65, zorder=1)
        sc = ax.scatter(
            xs,
            ys,
            c=vals,
            cmap=cmap,
            s=720,
            vmin=lo,
            vmax=hi,
            edgecolors="0.15",
            linewidths=1.1,
            zorder=5,
        )
        for q in range(n_q):
            ax.annotate(
                str(q),
                (xs[q], ys[q]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="0.05",
                zorder=6,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (layout units)")
        ax.set_ylabel("y (layout units)")
        ax.set_title(title, fontsize=12, color="0.1")
        ax.tick_params(colors="0.2", labelsize=9)
        ax.xaxis.label.set_color("0.15")
        ax.yaxis.label.set_color("0.15")
        ax.grid(False)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cb_label)

    fig.suptitle(
        "Encoder latent temporal means on Tuna-17 style geometry\n"
        "(colors: per-qubit mean of μ over shots)",
        fontsize=13,
        color="0.1",
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mpl_get_cmap(name: str):
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        return plt.get_cmap(name)


def _decorate_joint_axes(
    ax: Any,
    *,
    n_qubits: int,
    latent_dim: int,
    tick_fs: int = 11,
) -> None:
    centers = latent_dim * np.arange(n_qubits, dtype=float) + 0.5 * (latent_dim - 1)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels([str(q) for q in range(n_qubits)], fontsize=tick_fs)
    ax.set_yticklabels([str(q) for q in range(n_qubits)], fontsize=tick_fs)
    for q in range(1, n_qubits):
        ax.axhline(y=q * latent_dim - 0.5, color="k", linewidth=0.4, alpha=0.55)
        ax.axvline(x=q * latent_dim - 0.5, color="k", linewidth=0.4, alpha=0.55)


def _draw_sigma_hybrid(
    ax: Any,
    Sigma_joint: np.ndarray,
    *,
    d: int,
    offdiag_scale: float,
    n_qubits: int,
    latent_dim: int,
    title: str,
) -> Tuple[Any, Any, Any, float, np.ndarray]:
    """Off-diagonal: RdBu_r on ``scale × Σᵢⱼ``; diagonal: YlOrRd for Var(Δᵢ)."""
    import matplotlib.colors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    Z_off = np.full((d, d), np.nan, dtype=np.float64)
    mask_off = ~np.eye(d, dtype=bool)
    Z_off[mask_off] = Sigma_joint[mask_off] * float(offdiag_scale)
    vmax_off = float(np.nanmax(np.abs(Z_off)))
    if vmax_off < 1e-18:
        vmax_off = 1.0

    _base = _mpl_get_cmap("RdBu_r")
    try:
        cmap_off = _base.copy()
    except AttributeError:
        cmap_off = copy_module.copy(_base)
    cmap_off.set_bad(color=(0.9, 0.9, 0.9, 1.0))
    im = ax.imshow(
        Z_off,
        cmap=cmap_off,
        aspect="equal",
        interpolation="nearest",
        vmin=-vmax_off,
        vmax=vmax_off,
    )
    diag_vals = np.diag(Sigma_joint).astype(np.float64, copy=False)
    dmax = float(np.max(diag_vals))
    norm_d = matplotlib.colors.Normalize(vmin=0.0, vmax=max(dmax * 1.05, 1e-12))
    cmap_d = _mpl_get_cmap("YlOrRd")
    for i in range(d):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=cmap_d(norm_d(diag_vals[i])),
                edgecolor="0.25",
                linewidth=0.55,
                zorder=10,
            )
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(
        f"Qubit index j (latent_dim={latent_dim}: components per qubit)",
        fontsize=13,
    )
    ax.set_ylabel(
        f"Qubit index i (latent_dim={latent_dim}: components per qubit)",
        fontsize=13,
    )
    _decorate_joint_axes(ax, n_qubits=n_qubits, latent_dim=latent_dim, tick_fs=11)
    return im, norm_d, cmap_d, vmax_off, diag_vals


def plot_joint_drift_heatmaps(
    Sigma_joint: np.ndarray,
    corr_delta: np.ndarray,
    *,
    n_qubits: int,
    latent_dim: int,
    out_cov: Path,
    out_corr: Path,
    out_panel: Path,
    dpi: int = 150,
    offdiag_scale: float = 10.0,
) -> None:
    """Covariance (hybrid diag / scaled off-diag), correlation PNGs, and panel."""
    import matplotlib.colors
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    import matplotlib

    matplotlib.use("Agg")
    apply_latent_zebra_style()

    d = n_qubits * latent_dim
    if Sigma_joint.shape != (d, d):
        raise ValueError(f"Sigma_joint shape {Sigma_joint.shape} != ({d},{d})")
    if corr_delta.shape != (d, d):
        raise ValueError(f"corr_delta shape {corr_delta.shape} != ({d},{d})")

    frob = float(np.linalg.norm(Sigma_joint, ord="fro"))
    tr = float(np.trace(Sigma_joint))

    out_cov.parent.mkdir(parents=True, exist_ok=True)

    # Full-size covariance: off-diagonal × scale (RdBu), diagonal variances (YlOrRd)
    fig_c, ax_c = plt.subplots(figsize=(14.0, 8.0), dpi=dpi)
    im_off, norm_d, cmap_d, vmax_off, _diag = _draw_sigma_hybrid(
        ax_c,
        Sigma_joint,
        d=d,
        offdiag_scale=offdiag_scale,
        n_qubits=n_qubits,
        latent_dim=latent_dim,
        title=(
            f"Drift covariance Σ  ({d}×{d})  —  diagonal: Var(Δ); "
            f"off-diagonal: {offdiag_scale:g}×Cov\n"
            f"tr(Σ)={tr:.4g}   ‖Σ‖_F={frob:.4g}"
        ),
    )
    divider = make_axes_locatable(ax_c)
    cax_off = divider.append_axes("right", size="2.7%", pad=0.14)
    cb_off = fig_c.colorbar(im_off, cax=cax_off)
    cb_off.set_label(rf"Off-diagonal: ${offdiag_scale:g}\times$Cov$(\Delta_i,\Delta_j)$")
    cb_off.ax.tick_params(labelsize=11)

    sm = matplotlib.cm.ScalarMappable(norm=norm_d, cmap=cmap_d)
    sm.set_array([])
    cax_d = divider.append_axes("right", size="2.7%", pad=1.05)
    cb_d = fig_c.colorbar(sm, cax=cax_d)
    cb_d.set_label(r"Diagonal: Var$(\Delta_i)$")
    cb_d.ax.tick_params(labelsize=11)

    fig_c.tight_layout()
    fig_c.savefig(out_cov, dpi=dpi, bbox_inches="tight")
    plt.close(fig_c)

    # Full-size Pearson correlation
    fig_r, ax_r = plt.subplots(figsize=(14.0, 8.0), dpi=dpi)
    im_r = ax_r.imshow(
        corr_delta,
        cmap="RdBu_r",
        aspect="equal",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
    )
    ax_r.set_title(f"Pearson correlation of latent increments  ({d}×{d})", fontsize=14)
    ax_r.set_xlabel(
        f"Qubit index j (latent_dim={latent_dim}: components per qubit)",
        fontsize=13,
    )
    ax_r.set_ylabel(
        f"Qubit index i (latent_dim={latent_dim}: components per qubit)",
        fontsize=13,
    )
    _decorate_joint_axes(ax_r, n_qubits=n_qubits, latent_dim=latent_dim, tick_fs=11)
    cbr = plt.colorbar(im_r, ax=ax_r, fraction=0.035, pad=0.02, label="Corr(Δᵢ, Δⱼ)")
    cbr.ax.tick_params(labelsize=11)
    fig_r.tight_layout()
    fig_r.savefig(out_corr, dpi=dpi, bbox_inches="tight")
    plt.close(fig_r)

    # Side-by-side summary (left: same hybrid Σ; right: correlation)
    fig_p, axes = plt.subplots(1, 2, figsize=(17.5, 7.8), dpi=dpi)
    im0, norm0, cmap0, _, _ = _draw_sigma_hybrid(
        axes[0],
        Sigma_joint,
        d=d,
        offdiag_scale=offdiag_scale,
        n_qubits=n_qubits,
        latent_dim=latent_dim,
        title=f"Σ  (diag Var; off-diag {offdiag_scale:g}×)\ntr={tr:.4g}",
    )
    div0 = make_axes_locatable(axes[0])
    cax0a = div0.append_axes("right", size="2.2%", pad=0.08)
    fig_p.colorbar(im0, cax=cax0a, label=rf"${offdiag_scale:g}\times$off-diag")
    sm0 = matplotlib.cm.ScalarMappable(norm=norm0, cmap=cmap0)
    sm0.set_array([])
    cax0b = div0.append_axes("right", size="2.2%", pad=0.86)
    fig_p.colorbar(sm0, cax=cax0b, label="diag Var")

    im1 = axes[1].imshow(
        corr_delta,
        cmap="RdBu_r",
        aspect="equal",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[1].set_title(f"Correlation  ({d}×{d})", fontsize=13)
    axes[1].set_xlabel("Qubit index j", fontsize=12)
    axes[1].set_ylabel("Qubit index i", fontsize=12)
    _decorate_joint_axes(axes[1], n_qubits=n_qubits, latent_dim=latent_dim, tick_fs=10)
    cbp = plt.colorbar(im1, ax=axes[1], fraction=0.028, pad=0.14, label="Corr")
    cbp.ax.tick_params(labelsize=10)

    fig_p.suptitle(
        "Shot-to-shot increments Δμ: joint (q,μ₁/μ₂) ordering",
        fontsize=12,
        y=1.01,
    )
    fig_p.tight_layout()
    fig_p.savefig(out_panel, dpi=dpi, bbox_inches="tight")
    plt.close(fig_p)


def integrated_autocorr_time(
    x: np.ndarray, *, max_lag: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Normalized autocorrelation γ(k) for k=0..max_lag, and τ_int = 1/2 + sum_{k>=1} γ(k)
    (truncated where γ crosses zero or at max_lag).
    """
    z = np.asarray(x, dtype=np.float64).ravel()
    z = z - z.mean()
    var = np.var(z)
    if var < 1e-18:
        return np.array([1.0]), float("nan")
    z = z / np.sqrt(var)
    T = z.size
    if max_lag is None:
        max_lag = min(T - 1, 5000)
    gam = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        gam[k] = float(np.dot(z[: T - k], z[k:]) / T)
    tau_int = 0.5
    for k in range(1, max_lag + 1):
        if gam[k] <= 0:
            break
        tau_int += gam[k]
    return gam, float(tau_int)


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "inspire" / "fid_job_memory_noreset_large.pkl"


def _default_ckpt() -> Path:
    base = Path(__file__).resolve().parents[1] / "runs" / "fid_job_memory_noreset_large"
    new_path = base / "checkpoints" / "vae_checkpoint.pt"
    old_path = base / "vae_checkpoint.pt"
    return new_path if new_path.is_file() else old_path


def _run_root_from_ckpt(ckpt_path: Path) -> Path:
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlated random-walk statistics for per-qubit (μ₁,μ₂) along shots"
    )
    p.add_argument("--data", type=Path, default=None, help="FID memory pickle")
    p.add_argument("--source-file", type=Path, default=None, help="Optional source pickle for provenance")
    p.add_argument("--ckpt", type=Path, default=None, help="vae_checkpoint.pt")
    p.add_argument("--reset-qubits", action="store_true")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-json", type=Path, default=None, help="default: out-dir / latent_dynamics.json")
    p.add_argument("--out-dir", type=Path, default=None, help="default: ckpt parent")
    p.add_argument(
        "--max-lag",
        type=int,
        default=2000,
        help="max lag for integrated autocorrelation (optional block)",
    )
    p.add_argument(
        "--no-drift-plots",
        action="store_true",
        help="skip PNG heatmaps for Σ and correlation",
    )
    p.add_argument("--drift-plot-dpi", type=int, default=150)
    p.add_argument(
        "--offdiag-cov-scale",
        type=float,
        default=10.0,
        help="covariance heatmap: multiply off-diagonal entries by this for color scale (diagonal = Var, separate colormap)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data is not None else _default_data_path()
    ckpt_path = args.ckpt if args.ckpt is not None else _default_ckpt()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    run_root = Path(args.out_dir) if args.out_dir is not None else _run_root_from_ckpt(ckpt_path.resolve())
    out_fig = run_root / "figures"
    out_data = run_root / "data"
    out_reports = run_root / "reports"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json if args.out_json is not None else out_reports / "latent_dynamics.json"

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    seq_len = int(ckpt["seq_len"])
    latent_dim = int(ckpt["latent_dim"])

    mem = load_memory_pickle(data_path)
    built = build_stack_like_single_job(
        mem,
        num_qubits=ckpt.get("num_qubits"),
        n_tau=seq_len,
        reset_qubits=args.reset_qubits,
    )
    x_np, _ = stack_to_vae_tensors(built.stack)
    n_shots, n_qubits = built.n_shots, built.num_qubits

    device = _pick_device(args.device)
    model = QubitConvVAE(seq_len=seq_len, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])

    tensor_x = torch.from_numpy(x_np).unsqueeze(1)
    mu_flat = encode_mu_batched(
        model, tensor_x, device=device, batch_size=args.batch_size
    )
    mu_s_q = flat_rows_to_shot_qubit(mu_flat, n_shots, n_qubits)

    # Per-qubit temporal mean of μ (trajectory average)
    mu_bar_q = mu_s_q.mean(axis=0)

    # Center per qubit for AR(1) on latent trajectory (remove slow drift in mean)
    mu_centered = mu_s_q - mu_bar_q[np.newaxis, :, :]

    # Increments do not depend on subtracting a per-qubit constant in μ.
    delta, Delta, Sigma_joint = joint_increment_covariance(mu_s_q)

    per_q_cov = per_qubit_increment_cov(delta)

    # Amplitudes: RMS increment norm per qubit; global Frobenius / eigenvalues of joint Σ
    inc_rms_q = np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=0))
    eigvals_joint = np.linalg.eigvalsh(Sigma_joint)
    frob = float(np.linalg.norm(Sigma_joint, ord="fro"))
    trace_sigma = float(np.trace(Sigma_joint))

    # Cross-qubit correlation of scalar "step energy" ||Δμ^q|| (optional descriptive)
    step_norms = np.linalg.norm(delta, axis=2)
    cc_step = np.corrcoef(step_norms.T)

    # Pearson correlation between joint increment components (order: q0_d0, q0_d1, …).
    with np.errstate(invalid="ignore"):
        corr_delta = np.corrcoef(Delta.T)
    corr_delta = np.nan_to_num(corr_delta, nan=0.0)

    # AR(1) correlation time per (q, dim) on centered μ
    rho_qd = np.zeros((n_qubits, latent_dim), dtype=np.float64)
    tau_ar_qd = np.full((n_qubits, latent_dim), np.nan, dtype=np.float64)
    for q in range(n_qubits):
        for d in range(latent_dim):
            rho, tau = ar1_rho_and_tau_steps(mu_centered[:, q, d])
            rho_qd[q, d] = rho
            tau_ar_qd[q, d] = tau

    # Integrated autocorrelation time (optional): per (q,d) and median
    tau_int_grid = np.full((n_qubits, latent_dim), np.nan, dtype=np.float64)
    for q in range(n_qubits):
        for d in range(latent_dim):
            _, tau_i = integrated_autocorr_time(
                mu_centered[:, q, d], max_lag=min(args.max_lag, n_shots - 2)
            )
            tau_int_grid[q, d] = tau_i

    report: Dict[str, Any] = {
        "data_path": str(data_path.resolve()),
        "ckpt_path": str(ckpt_path.resolve()),
        "n_shots": n_shots,
        "n_qubits": n_qubits,
        "latent_dim": latent_dim,
        "differential": built.differential,
        "mu_bar_per_qubit": {
            str(q): {"mu_1": float(mu_bar_q[q, 0]), "mu_2": float(mu_bar_q[q, 1])}
            for q in range(n_qubits)
        },
        "increment_rms_l2_per_qubit": [float(inc_rms_q[q]) for q in range(n_qubits)],
        "joint_increment_covariance": Sigma_joint.tolist(),
        "joint_cov_frobenius": frob,
        "joint_cov_trace": trace_sigma,
        "joint_cov_eigenvalues_min_max": [
            float(eigvals_joint.min()),
            float(eigvals_joint.max()),
        ],
        "per_qubit_increment_cov_2x2": [
            per_q_cov[q].tolist() for q in range(n_qubits)
        ],
        "cross_qubit_correlation_step_l2": cc_step.tolist(),
        "increment_correlation_Pearson_full": corr_delta.tolist(),
        "rms_increment_l2_global": float(np.sqrt(np.mean(np.sum(delta**2, axis=(1, 2))))),
        "ar1_rho_per_qubit_dim": rho_qd.tolist(),
        "ar1_correlation_time_steps_per_qubit_dim": tau_ar_qd.tolist(),
        "ar1_correlation_time_steps_summary": {
            "median_over_q_dim": float(np.nanmedian(tau_ar_qd)),
            "mean_over_q_dim": float(np.nanmean(tau_ar_qd)),
        },
        "integrated_autocorr_time_steps_per_qubit_dim": tau_int_grid.tolist(),
        "integrated_autocorr_time_summary": {
            "median_over_q_dim": float(np.nanmedian(tau_int_grid)),
            "mean_over_q_dim": float(np.nanmean(tau_int_grid)),
        },
        "notes": {
            "joint_cov_rows": "Order [Δμ_1^0, Δμ_2^0, ..., Δμ_1^{Q-1}, Δμ_2^{Q-1}] per shot.",
            "drift_heatmaps": "latent_drift_covariance.png (diag YlOrRd, off-diag scaled), latent_drift_correlation.png, latent_drift_cov_corr_panel.png",
            "mu_bar_2d": "latent_mu_bar_means_2d.png — per-qubit (μ̄₁, μ̄₂)",
            "mu_bar_chip": "latent_mu_bar_chip_spatial.png — μ̄₁, μ̄₂ on Tuna-17 layout (17 qubits)",
            "ar1_tau": "-log|ρ|^{-1} in shot index units when AR(1) fit is valid.",
            "integrated_tau": "Geyer-style sum until first negative γ(k) or max_lag.",
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        out_data / "latent_dynamics_arrays.npz",
        mu_s_q=mu_s_q,
        mu_bar_q=mu_bar_q,
        delta=delta,
        Sigma_joint=Sigma_joint,
        increment_corr_pearson=corr_delta,
        rho_ar1=rho_qd,
        tau_ar1_steps=tau_ar_qd,
        tau_int_steps=tau_int_grid,
    )

    if not args.no_drift_plots:
        plot_joint_drift_heatmaps(
            Sigma_joint,
            corr_delta,
            n_qubits=n_qubits,
            latent_dim=latent_dim,
            out_cov=out_fig / "latent_drift_covariance.png",
            out_corr=out_fig / "latent_drift_correlation.png",
            out_panel=out_fig / "latent_drift_cov_corr_panel.png",
            dpi=int(args.drift_plot_dpi),
            offdiag_scale=float(args.offdiag_cov_scale),
        )
        plot_latent_mu_bar_2d(
            mu_bar_q,
            out_fig / "latent_mu_bar_means_2d.png",
            dpi=int(args.drift_plot_dpi),
        )
        plot_mu_bar_chip_spatial(
            mu_bar_q,
            out_fig / "latent_mu_bar_chip_spatial.png",
            dpi=int(args.drift_plot_dpi),
        )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_data / 'latent_dynamics_arrays.npz'}")
    if not args.no_drift_plots:
        print(f"Wrote {out_fig / 'latent_drift_covariance.png'}")
        print(f"Wrote {out_fig / 'latent_drift_correlation.png'}")
        print(f"Wrote {out_fig / 'latent_drift_cov_corr_panel.png'}")
        print(f"Wrote {out_fig / 'latent_mu_bar_means_2d.png'}")
        if n_qubits == len(TUNA17_QUBIT_XY) and latent_dim >= 2:
            print(f"Wrote {out_fig / 'latent_mu_bar_chip_spatial.png'}")
    print(
        "Per-qubit trajectory mean μ̄ (first 3 qubits):",
        mu_bar_q[: min(3, n_qubits)].tolist(),
    )
    print(
        "Joint increment cov eigenvalue range:",
        float(eigvals_joint.min()),
        "...",
        float(eigvals_joint.max()),
    )
    print(
        "AR(1) correlation time (steps): median",
        report["ar1_correlation_time_steps_summary"]["median_over_q_dim"],
        "mean",
        f'{report["ar1_correlation_time_steps_summary"]["mean_over_q_dim"]:.4f}',
    )
    print(
        "Integrated autocorr time (steps): median",
        report["integrated_autocorr_time_summary"]["median_over_q_dim"],
    )


if __name__ == "__main__":
    main()
