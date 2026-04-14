#!/usr/bin/env python3
"""
Compare latent-space dynamics between two FID memory datasets.

This script:
1) Encodes both datasets with the same VAE checkpoint.
2) Builds a side-by-side latent trajectory GIF (dataset A vs dataset B).
3) Computes and plots shot-wise delta in latent means:
      delta_mu = mu_B - mu_A
   and saves delta μ1 / μ2 summaries.

Outputs (under run-dir):
- figures/latent_dynamics_compare_side_by_side_avg<block>.gif
- figures/latent_dynamics_compare_delta_mu.png
- data/latent_dynamics_compare_arrays.npz
- reports/latent_dynamics_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

from ML.analysis_processing.fid_data_io import build_stack_like_single_job, load_memory_pickle, stack_to_vae_tensors
from ML.analysis_processing.generate_latent_dynamics import (
    _default_ckpt,
    _pick_device,
    _run_root_from_ckpt,
    encode_mu_batched,
    flat_rows_to_shot_qubit,
)
from ML.plotting.plot_latent_from_checkpoint import (
    _axis_limits,
    _block_mean_mu,
    _draw_latent_frame,
    _get_tab20,
    _maybe_gifsicle_optimize,
)
from ML.plotting.style import apply_latent_zebra_style
from ML.vae_model import QubitConvVAE


def _default_data_a() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "quantum_code"
        / "data"
        / "fid_job_memory_noreset_large.pkl"
    )


def _default_data_b() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "quantum_code"
        / "data"
        / "fid_job_memory_noreset_large_1404.pkl"
    )


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "quantum_code" / "data"


def _encode_dataset(
    *,
    data_path: Path,
    ckpt: Dict[str, object],
    model: QubitConvVAE,
    device: torch.device,
    batch_size: int,
    reset_qubits: bool,
) -> np.ndarray:
    seq_len = int(ckpt["seq_len"])
    mem = load_memory_pickle(data_path)
    built = build_stack_like_single_job(
        mem,
        num_qubits=ckpt.get("num_qubits"),
        n_tau=seq_len,
        reset_qubits=reset_qubits,
    )
    x_np, _ = stack_to_vae_tensors(built.stack)
    tensor_x = torch.from_numpy(x_np).unsqueeze(1)
    mu_flat = encode_mu_batched(
        model,
        tensor_x,
        device=device,
        batch_size=batch_size,
    )
    return flat_rows_to_shot_qubit(mu_flat, built.n_shots, built.num_qubits)


def _binned_progress(n_frames: int, block: int, n_shots: int) -> Iterator[Tuple[int, int]]:
    for k in range(n_frames):
        yield (k, min((k + 1) * block, n_shots))


def _write_side_by_side_gif(
    *,
    out_path: Path,
    z_a: np.ndarray,
    z_b: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    title_a: str,
    title_b: str,
    fps: float,
    dpi: int,
    figsize: Tuple[float, float],
    alpha_old: float,
    alpha_new: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_frames = min(z_a.shape[0], z_b.shape[0])
    if n_frames < 1:
        raise ValueError("No frames available for GIF generation.")
    n_q = int(z_a.shape[1])

    z_a = z_a[:n_frames]
    z_b = z_b[:n_frames]

    all_latent = np.concatenate([z_a.reshape(-1, z_a.shape[-1]), z_b.reshape(-1, z_b.shape[-1])], axis=0)
    all_means = np.concatenate([mean_a, mean_b], axis=0)
    xlim, ylim = _axis_limits(all_latent.reshape(-1, 1, all_latent.shape[-1]), extra=all_means)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.1)

    norm = Normalize(vmin=0, vmax=max(n_q - 1, 1))
    sm = ScalarMappable(norm=norm, cmap=_get_tab20())
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label="Qubit index")
    cbar.ax.tick_params(labelsize=10)

    writer = PillowWriter(fps=max(float(fps), 1e-3))
    progress_a = _binned_progress(n_frames, 1, n_frames)
    progress_b = _binned_progress(n_frames, 1, n_frames)

    with writer.saving(fig, str(out_path), dpi=dpi):
        for k in range(n_frames):
            _, cum_a = next(progress_a)
            _, cum_b = next(progress_b)
            _draw_latent_frame(
                ax_a,
                z_a,
                k,
                n_qubits=n_q,
                shot_mean_per_q=mean_a,
                title=f"{title_a}\nframe {k + 1}/{n_frames} | bins ≤ {cum_a}",
                xlim=xlim,
                ylim=ylim,
                alpha_old=alpha_old,
                alpha_new=alpha_new,
            )
            _draw_latent_frame(
                ax_b,
                z_b,
                k,
                n_qubits=n_q,
                shot_mean_per_q=mean_b,
                title=f"{title_b}\nframe {k + 1}/{n_frames} | bins ≤ {cum_b}",
                xlim=xlim,
                ylim=ylim,
                alpha_old=alpha_old,
                alpha_new=alpha_new,
            )
            writer.grab_frame()
    plt.close(fig)


def _plot_delta_mu(
    *,
    delta_mu: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_latent_zebra_style()
    n_shots, _n_q, latent_dim = delta_mu.shape
    x = np.arange(n_shots, dtype=np.int64)
    mean_q = delta_mu.mean(axis=1)
    std_q = delta_mu.std(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), dpi=dpi, sharex=True)
    labels = [r"$\Delta \mu_1$", r"$\Delta \mu_2$"]
    for d in range(min(2, latent_dim)):
        ax = axes[d]
        ax.plot(x, mean_q[:, d], color="tab:blue", linewidth=1.3, label="mean over qubits")
        ax.fill_between(
            x,
            mean_q[:, d] - std_q[:, d],
            mean_q[:, d] + std_q[:, d],
            color="tab:blue",
            alpha=0.2,
            linewidth=0.0,
            label="±1 std over qubits",
        )
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        ax.set_ylabel(labels[d])
        ax.grid(True, alpha=0.3, linewidth=0.6)
        if d == 0:
            ax.legend(loc="upper right", fontsize=9)

    if latent_dim < 2:
        axes[1].text(0.02, 0.5, "latent_dim < 2", transform=axes[1].transAxes, fontsize=10)
        axes[1].set_ylabel(r"$\Delta \mu_2$")
        axes[1].grid(True, alpha=0.3, linewidth=0.6)

    axes[-1].set_xlabel("Shot index")
    fig.suptitle("Shot-wise latent mean difference: dataset B - dataset A", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare latent dynamics between two datasets using one trained VAE checkpoint."
    )
    p.add_argument("--data-a", type=Path, default=None, help="Baseline dataset (default: fid_job_memory_noreset_large.pkl)")
    p.add_argument("--data-b", type=Path, default=None, help="Comparison dataset (default: fid_job_memory_noreset_large_1404.pkl)")
    p.add_argument(
        "--data-b-name",
        type=str,
        default=None,
        help="Comparison dataset filename under first_tests/quantum_code/data (e.g. large_otoc.pkl)",
    )
    p.add_argument("--ckpt", type=Path, default=None, help="VAE checkpoint path")
    p.add_argument("--out-dir", type=Path, default=None, help="Output run dir (default: checkpoint run root)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--reset-qubits", action="store_true")
    p.add_argument("--avg-block", type=int, default=50, help="Shots per frame for GIF")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--figsize", type=float, nargs=2, default=(13.0, 5.8))
    p.add_argument("--alpha-old", type=float, default=0.06)
    p.add_argument("--alpha-new", type=float, default=0.92)
    p.add_argument("--no-gifsicle", action="store_true")
    p.add_argument("--gifsicle-colors", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_a = args.data_a if args.data_a is not None else _default_data_a()
    if args.data_b is not None:
        data_b = args.data_b
    elif args.data_b_name is not None:
        data_b = _default_data_root() / args.data_b_name
    else:
        data_b = _default_data_b()
    ckpt_path = args.ckpt if args.ckpt is not None else _default_ckpt()

    if not data_a.is_file():
        raise FileNotFoundError(data_a)
    if not data_b.is_file():
        raise FileNotFoundError(data_b)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    run_root = Path(args.out_dir) if args.out_dir is not None else _run_root_from_ckpt(ckpt_path.resolve())
    out_fig = run_root / "figures"
    out_data = run_root / "data"
    out_reports = run_root / "reports"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    seq_len = int(ckpt["seq_len"])
    latent_dim = int(ckpt["latent_dim"])
    device = _pick_device(args.device)

    model = QubitConvVAE(seq_len=seq_len, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])

    mu_a = _encode_dataset(
        data_path=data_a,
        ckpt=ckpt,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        reset_qubits=bool(args.reset_qubits),
    )
    mu_b = _encode_dataset(
        data_path=data_b,
        ckpt=ckpt,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        reset_qubits=bool(args.reset_qubits),
    )

    if mu_a.shape[1:] != mu_b.shape[1:]:
        raise ValueError(
            f"Incompatible latent shapes after encoding: A {mu_a.shape}, B {mu_b.shape}."
        )

    n_common = min(mu_a.shape[0], mu_b.shape[0])
    if mu_a.shape[0] != mu_b.shape[0]:
        print(
            f"[warn] n_shots mismatch (A={mu_a.shape[0]}, B={mu_b.shape[0]}). "
            f"Using first {n_common} shots for direct delta."
        )
    mu_a_c = mu_a[:n_common]
    mu_b_c = mu_b[:n_common]
    n_q = mu_a_c.shape[1]

    blk = max(1, int(args.avg_block))
    z_a = _block_mean_mu(mu_a_c, blk)
    z_b = _block_mean_mu(mu_b_c, blk)
    mean_a = mu_a_c.mean(axis=0)
    mean_b = mu_b_c.mean(axis=0)
    delta_mu = mu_b_c - mu_a_c

    pair_tag = f"{data_a.stem}_vs_{data_b.stem}"
    side_by_side_gif = out_fig / f"latent_dynamics_compare_{pair_tag}_side_by_side_avg{blk}.gif"
    delta_png = out_fig / f"latent_dynamics_compare_{pair_tag}_delta_mu.png"
    npz_path = out_data / f"latent_dynamics_compare_{pair_tag}_arrays.npz"
    report_path = out_reports / f"latent_dynamics_compare_{pair_tag}.json"

    _write_side_by_side_gif(
        out_path=side_by_side_gif,
        z_a=z_a,
        z_b=z_b,
        mean_a=mean_a,
        mean_b=mean_b,
        title_a=f"A: {data_a.stem}",
        title_b=f"B: {data_b.stem}",
        fps=float(args.fps),
        dpi=int(args.dpi),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        alpha_old=float(args.alpha_old),
        alpha_new=float(args.alpha_new),
    )
    if not args.no_gifsicle and _maybe_gifsicle_optimize(side_by_side_gif, colors=int(args.gifsicle_colors)):
        print("Optimized side-by-side GIF with gifsicle.")

    _plot_delta_mu(delta_mu=delta_mu, out_path=delta_png, dpi=int(args.dpi))

    delta_mean_q = delta_mu.mean(axis=1)
    delta_std_q = delta_mu.std(axis=1)
    report = {
        "data_a": str(data_a.resolve()),
        "data_b": str(data_b.resolve()),
        "ckpt_path": str(ckpt_path.resolve()),
        "n_shots_common": int(n_common),
        "n_qubits": int(n_q),
        "latent_dim": int(latent_dim),
        "avg_block": int(blk),
        "delta_definition": "delta_mu = mu_B - mu_A",
        "delta_summary": {
            "delta_mu1_mean_over_all_shots_qubits": float(delta_mu[..., 0].mean()) if latent_dim >= 1 else float("nan"),
            "delta_mu2_mean_over_all_shots_qubits": float(delta_mu[..., 1].mean()) if latent_dim >= 2 else float("nan"),
            "delta_l2_rms_over_shots_qubits": float(np.sqrt(np.mean(np.sum(delta_mu**2, axis=2)))),
        },
        "outputs": {
            "gif_side_by_side": str(side_by_side_gif.resolve()),
            "delta_plot_png": str(delta_png.resolve()),
            "arrays_npz": str(npz_path.resolve()),
        },
    }

    np.savez_compressed(
        npz_path,
        mu_a=mu_a_c,
        mu_b=mu_b_c,
        mu_a_block=z_a,
        mu_b_block=z_b,
        mu_a_mean_q=mean_a,
        mu_b_mean_q=mean_b,
        delta_mu=delta_mu,
        delta_mu_mean_q=delta_mean_q,
        delta_mu_std_q=delta_std_q,
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {side_by_side_gif}")
    print(f"Wrote {delta_png}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
