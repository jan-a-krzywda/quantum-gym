#!/usr/bin/env python3
"""
Build a 2D latent-space GIF from a trained VAE checkpoint (same ordering as ``latent_plot.png``).

Writes one file: ``latent_plot_avg<M>.gif`` where ``M`` is ``--avg-block`` (default 50).
Each frame is the mean latent over ``M`` consecutive shots per qubit; full trajectory history
is drawn with progressively stronger opacity (older points fade). A colorbar shows qubit index.
For each qubit, the mean of μ over **all shots** (full run) is drawn as a ``+`` in that qubit's
``tab20`` color, with the qubit index annotated beside it.

Requires the original memory pickle (path stored in ``vae_checkpoint.pt`` unless overridden).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
if not torch.cuda.is_available():
    torch.backends.nnpack.enabled = False

from ML.analysis_processing.fid_data_io import build_stack_like_single_job, load_memory_pickle, stack_to_vae_tensors
from ML.plotting.style import apply_latent_zebra_style
from ML.vae_model import QubitConvVAE

apply_latent_zebra_style()


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


def _latent_xy(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for plotting; pad Y with zeros if ``latent_dim == 1``."""
    x = mu[:, 0].astype(np.float64, copy=False)
    if mu.shape[1] > 1:
        y = mu[:, 1].astype(np.float64, copy=False)
    else:
        y = np.zeros_like(x)
    return x, y


def _block_mean_mu(z: np.ndarray, block: int) -> np.ndarray:
    """``z`` shape ``(n_shots, n_q, D)`` → ``(n_frames, n_q, D)`` non-overlapping block means."""
    n_shots = z.shape[0]
    b = max(1, int(block))
    n_frames = (n_shots + b - 1) // b
    out = np.empty((n_frames, z.shape[1], z.shape[2]), dtype=np.float64)
    for i in range(n_frames):
        lo = i * b
        hi = min(lo + b, n_shots)
        out[i] = z[lo:hi].mean(axis=0)
    return out


def _get_tab20():
    try:
        return matplotlib.colormaps["tab20"]
    except AttributeError:
        return matplotlib.cm.get_cmap("tab20")


def _tab20_rgba(q: int, n_qubits: int, alpha: float) -> Tuple[float, float, float, float]:
    cmap = _get_tab20()
    norm = Normalize(vmin=0, vmax=max(n_qubits - 1, 1))
    r, g, b, _ = cmap(norm(float(q)))
    return (float(r), float(g), float(b), float(alpha))


def _tab20_rgb(q: int, n_qubits: int) -> Tuple[float, float, float]:
    r, g, b, _ = _tab20_rgba(q, n_qubits, 1.0)
    return (r, g, b)


def _alphas_full_history(n_hist: int, *, alpha_old: float, alpha_new: float) -> np.ndarray:
    """Monotone fade from first frame (faint) to current (strong)."""
    if n_hist <= 0:
        return np.array([], dtype=np.float64)
    return np.linspace(alpha_old, alpha_new, n_hist, dtype=np.float64)


def _draw_latent_frame(
    ax: plt.Axes,
    z: np.ndarray,
    frame_idx: int,
    *,
    n_qubits: int,
    shot_mean_per_q: np.ndarray,
    title: str,
    drop_title_head_lines: int = 0,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    alpha_old: float,
    alpha_new: float,
) -> None:
    """
    ``z`` shape ``(n_frames, n_qubits, latent_dim)``; draw full history 0..frame_idx inclusive.
    """
    ax.clear()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.set_xlabel(r"latent dimension $\mu_1$")
    ax.set_ylabel(r"latent dimension $\mu_2$")
    title_lines = str(title).splitlines()
    if drop_title_head_lines > 0 and len(title_lines) > drop_title_head_lines:
        title = "\n".join(title_lines[drop_title_head_lines:])
    ax.set_title(title, fontsize=12)

    k = int(frame_idx)
    n_hist = k + 1
    if n_hist < 1:
        return

    alphas_pts = _alphas_full_history(n_hist, alpha_old=alpha_old, alpha_new=alpha_new)

    for q in range(n_qubits):
        seg = z[: k + 1, q, :].astype(np.float64, copy=False)
        xs, ys = _latent_xy(seg)

        for j in range(n_hist - 1):
            a_line = float(0.5 * (alphas_pts[j] + alphas_pts[j + 1]))
            c0 = _tab20_rgba(q, n_qubits, a_line)
            ax.plot(
                xs[j : j + 2],
                ys[j : j + 2],
                color=c0,
                linewidth=1.15,
                solid_capstyle="round",
                zorder=1 + j,
            )

        for j in range(n_hist):
            c = _tab20_rgba(q, n_qubits, float(alphas_pts[j]))
            ax.scatter(
                xs[j],
                ys[j],
                s=22 if j < n_hist - 1 else 36,
                facecolors=c,
                edgecolors="none",
                zorder=10 + j,
            )

    # Per-qubit mean over all raw shots (fixed); cross + label in qubit color
    for q in range(n_qubits):
        mx, my = _latent_xy(shot_mean_per_q[q : q + 1])
        rgb = _tab20_rgb(q, n_qubits)
        ax.plot(
            mx,
            my,
            linestyle="None",
            marker="+",
            markersize=13,
            markeredgewidth=2.2,
            color=rgb,
            zorder=99998,
        )
        ax.annotate(
            str(q),
            xy=(float(mx[0]), float(my[0])),
            xytext=(9, 9),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=rgb,
            zorder=99999,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.95,
            },
            clip_on=False,
        ).set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])


def _axis_limits(
    z: np.ndarray,
    *,
    extra: Optional[np.ndarray] = None,
    pad_frac: float = 0.06,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Global limits from ``z`` shaped ``(..., D)`` and optional extra rows (same D)."""
    parts: List[np.ndarray] = [z.reshape(-1, z.shape[-1])]
    if extra is not None and extra.size > 0:
        ex = np.asarray(extra, dtype=np.float64).reshape(-1, z.shape[-1])
        parts.append(ex)
    flat = np.concatenate(parts, axis=0)
    xs, ys = _latent_xy(flat)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax - xmin < 1e-9:
        xmin -= 0.05
        xmax += 0.05
    if ymax - ymin < 1e-9:
        ymin -= 0.05
        ymax += 0.05
    dx = (xmax - xmin) * pad_frac
    dy = (ymax - ymin) * pad_frac
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)


def _maybe_gifsicle_optimize(path: Path, *, colors: int) -> bool:
    """If ``gifsicle`` is on PATH, rewrite GIF in place with palette + optimization."""
    exe = shutil.which("gifsicle")
    if not exe:
        return False
    tmp = path.with_suffix(".opt.gif")
    try:
        subprocess.run(
            [
                exe,
                "-O3",
                "--colors",
                str(max(16, int(colors))),
                str(path),
                "-o",
                str(tmp),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        tmp.replace(path)
        return True
    except (OSError, subprocess.CalledProcessError):
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False


def _write_gif(
    out_path: Path,
    z: np.ndarray,
    *,
    n_qubits: int,
    shot_mean_per_q: np.ndarray,
    fps: float,
    dpi: int,
    figsize: Tuple[float, float],
    title_prefix: str,
    shot_progress_fn: Iterator[Tuple[int, int]],
    alpha_old: float,
    alpha_new: float,
    drop_title_head_lines: int = 0,
) -> None:
    """``z`` shape ``(n_frames, n_qubits, D)``; ``shot_mean_per_q`` is ``(n_qubits, D)`` over all shots."""
    xlim, ylim = _axis_limits(z, extra=shot_mean_per_q)
    n_frames = z.shape[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    norm = Normalize(vmin=0, vmax=max(n_qubits - 1, 1))
    sm = ScalarMappable(norm=norm, cmap=_get_tab20())
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label="Qubit index")
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.9, bottom=0.11)

    writer = PillowWriter(fps=max(float(fps), 1e-3))
    progress = iter(shot_progress_fn)

    with writer.saving(fig, str(out_path), dpi=dpi):
        for k in range(n_frames):
            _, cum_shots = next(progress)
            title = (
                f"{title_prefix}\nframe {k + 1}/{n_frames}  |  cumulative shots ≤ {cum_shots}"
            )
            _draw_latent_frame(
                ax,
                z,
                k,
                n_qubits=n_qubits,
                shot_mean_per_q=shot_mean_per_q,
                title=title,
                drop_title_head_lines=drop_title_head_lines,
                xlim=xlim,
                ylim=ylim,
                alpha_old=alpha_old,
                alpha_new=alpha_new,
            )
            writer.grab_frame()
    plt.close(fig)


def write_latent_gif_terminal_shots(
    out_path: Path,
    mu_s_q: np.ndarray,
    *,
    n_qubits: int,
    shot_mean_per_q: np.ndarray,
    terminal_shot_indices: List[int],
    fps: float,
    dpi: int,
    figsize: Tuple[float, float],
    title_prefix: str,
    alpha_old: float,
    alpha_new: float,
    drop_title_head_lines: int = 0,
) -> None:
    """
    Like ``_write_gif`` but each frame ``i`` draws the trajectory ``mu_s_q[: s + 1]`` where
    ``s = terminal_shot_indices[i]`` (full shot resolution, subsampled frames).

    ``mu_s_q`` shape ``(n_shots, n_qubits, D)``.
    """
    xlim, ylim = _axis_limits(mu_s_q, extra=shot_mean_per_q)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(terminal_shot_indices)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    norm = Normalize(vmin=0, vmax=max(n_qubits - 1, 1))
    sm = ScalarMappable(norm=norm, cmap=_get_tab20())
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label="Qubit index")
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.11)

    writer = PillowWriter(fps=max(float(fps), 1e-3))

    with writer.saving(fig, str(out_path), dpi=dpi):
        for k in range(n_frames):
            s = int(terminal_shot_indices[k])
            cum_shots = s + 1
            title = (
                f"{title_prefix}\nframe {k + 1}/{n_frames}  |  cumulative shots ≤ {cum_shots}"
            )
            _draw_latent_frame(
                ax,
                mu_s_q,
                s,
                n_qubits=n_qubits,
                shot_mean_per_q=shot_mean_per_q,
                title=title,
                drop_title_head_lines=drop_title_head_lines,
                xlim=xlim,
                ylim=ylim,
                alpha_old=alpha_old,
                alpha_new=alpha_new,
            )
            writer.grab_frame()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Block-averaged latent-space GIF (full faded history, colorbar, qubit labels)"
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory with vae_checkpoint.pt (default: parent of latent_plot.png if --latent-png set)",
    )
    p.add_argument(
        "--latent-png",
        type=Path,
        default=None,
        help="latent_plot.png path; run dir inferred as its parent if --run-dir omitted",
    )
    p.add_argument("--data", type=Path, default=None, help="Override memory pickle path")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=80, help="figure DPI for GIF frames (default 80)")
    p.add_argument("--figsize", type=float, nargs=2, default=(7.0, 5.4))
    p.add_argument(
        "--encode-batch",
        type=int,
        default=4096,
        help="forward batch size for the VAE encoder (default 4096)",
    )
    p.add_argument(
        "--no-gifsicle",
        action="store_true",
        help="skip ``gifsicle`` post-processing even when installed",
    )
    p.add_argument(
        "--gifsicle-colors",
        type=int,
        default=128,
        help="palette size for gifsicle (default 128; only if gifsicle is on PATH)",
    )
    p.add_argument("--avg-block", type=int, default=50, help="shots averaged per frame")
    p.add_argument(
        "--alpha-old",
        type=float,
        default=0.06,
        help="opacity of oldest trajectory points (default 0.06)",
    )
    p.add_argument(
        "--alpha-new",
        type=float,
        default=0.92,
        help="opacity of newest trajectory points (default 0.92)",
    )
    p.add_argument(
        "--reset-qubits",
        action="store_true",
        help="match a reset-style job (raw Z bits); default follows checkpoint differential flag",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        if args.latent_png is not None:
            run_dir = Path(args.latent_png).resolve().parent
        else:
            raise SystemExit("Pass --run-dir or --latent-png (to infer the run directory).")
    run_dir = Path(run_dir)
    ckpt_path_new = run_dir / "checkpoints" / "vae_checkpoint.pt"
    ckpt_path_old = run_dir / "vae_checkpoint.pt"
    ckpt_path = ckpt_path_new if ckpt_path_new.is_file() else ckpt_path_old
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    data_path = args.data if args.data is not None else Path(str(ckpt["data_path"]))
    if not data_path.is_file():
        raise FileNotFoundError(f"Memory pickle not found: {data_path} (use --data)")

    mem = load_memory_pickle(data_path)
    reset_qubits = bool(args.reset_qubits) or (not bool(ckpt.get("differential", True)))
    built = build_stack_like_single_job(
        mem,
        num_qubits=int(ckpt.get("num_qubits")) if ckpt.get("num_qubits") is not None else None,
        n_tau=int(ckpt["seq_len"]),
        reset_qubits=reset_qubits,
    )

    x_np, _qubit_ids = stack_to_vae_tensors(built.stack)
    n_shots, n_qubits = built.n_shots, built.num_qubits
    latent_dim = int(ckpt["latent_dim"])
    if x_np.shape != (n_shots * n_qubits, built.n_tau):
        raise ValueError(f"tensor shape mismatch: {x_np.shape} vs shots×qubits×tau")

    device = _pick_device(args.device)
    model = QubitConvVAE(seq_len=built.n_tau, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tensor_x = torch.from_numpy(x_np).unsqueeze(1).to(device)
    bs = max(256, int(args.encode_batch))
    enc_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, tensor_x.shape[0], bs):
            chunk = tensor_x[i : i + bs]
            enc_parts.append(model.fc_mu(model.encoder(chunk)))
        latent_mu = torch.cat(enc_parts, dim=0).cpu().numpy()

    z = latent_mu.reshape(n_shots, n_qubits, latent_dim).astype(np.float64, copy=False)
    shot_mean_per_q = z.mean(axis=0)

    blk = max(1, int(args.avg_block))
    zb = _block_mean_mu(z, blk)
    n_b = zb.shape[0]

    def binned_progress():
        for k in range(n_b):
            yield (k, min((k + 1) * blk, n_shots))

    fps = float(args.fps)
    dpi = int(args.dpi)
    figsize = (float(args.figsize[0]), float(args.figsize[1]))
    stem = data_path.stem

    out_bin = run_dir / "figures" / f"latent_plot_avg{blk}.gif"

    _write_gif(
        out_bin,
        zb,
        n_qubits=n_qubits,
        shot_mean_per_q=shot_mean_per_q,
        fps=fps,
        dpi=dpi,
        figsize=figsize,
        title_prefix=f"Latent μ — {stem}  (mean over {blk} shots; tab20 by qubit)",
        shot_progress_fn=binned_progress(),
        alpha_old=float(args.alpha_old),
        alpha_new=float(args.alpha_new),
    )
    if not args.no_gifsicle and _maybe_gifsicle_optimize(
        out_bin, colors=int(args.gifsicle_colors)
    ):
        print("Optimized GIF with gifsicle (smaller file).")
    print(f"Wrote {out_bin}")


if __name__ == "__main__":
    main()
