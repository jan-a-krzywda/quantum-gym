#!/usr/bin/env python3
"""
GIF of (μ₁, μ₂) trajectories in latent space from saved ``mu_s_q`` (real encoder path or
simulated dynamics), with per-qubit temporal means as ``+`` markers and optional overlay of
increment-covariance statistics from ``latent_dynamics.json``.

Uses the same visual style as ``make_latent_gif.py`` (tab20, fading history).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

from ML.plotting.plot_latent_from_checkpoint import (  # noqa: E402
    _block_mean_mu,
    _maybe_gifsicle_optimize,
    _write_gif,
    write_latent_gif_terminal_shots,
)


def _load_mu_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    if "mu_s_q" not in z.files:
        raise KeyError(f"{path} must contain mu_s_q; has {z.files}")
    mu = np.asarray(z["mu_s_q"], dtype=np.float64)
    mu_bar = None
    if "mu_bar_q" in z.files:
        mu_bar = np.asarray(z["mu_bar_q"], dtype=np.float64)
    return mu, mu_bar


def _load_dynamics_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _title_suffix_from_json(data: Dict[str, Any]) -> str:
    tr = data.get("joint_cov_trace")
    fr = data.get("joint_cov_frobenius")
    parts = []
    if tr is not None:
        parts.append(f"tr(Σ_inc)={float(tr):.4g}")
    if fr is not None:
        parts.append(f"‖Σ_inc‖_F={float(fr):.4g}")
    return "  |  ".join(parts) if parts else ""


def _build_terminal_indices(n_shots: int, frame_step: int) -> List[int]:
    step = max(1, int(frame_step))
    inds = list(range(0, n_shots, step))
    if not inds:
        return [0]
    if inds[-1] != n_shots - 1:
        inds.append(n_shots - 1)
    return inds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GIF: latent (μ₁,μ₂) trajectories from mu_s_q npz + dynamics JSON"
    )
    p.add_argument(
        "--source-file",
        type=Path,
        default=None,
        help="Optional source pickle path for provenance in pipeline calls",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory with npz/json (default: fid_job_memory_noreset_large run)",
    )
    p.add_argument(
        "--mu-npz",
        type=Path,
        default=None,
        help="NPZ with mu_s_q (and optional mu_bar_q). Default: latent_dynamics_arrays.npz or sim npz",
    )
    p.add_argument(
        "--source",
        choices=("encoded", "sim"),
        default="encoded",
        help="encoded = latent_dynamics_arrays.npz (real μ); sim = sim_fitted_latent_mu_s_q.npz",
    )
    p.add_argument(
        "--dynamics-json",
        type=Path,
        default=None,
        help="latent_dynamics.json (Σ stats in title); default next to npz",
    )
    p.add_argument(
        "--sim-meta",
        type=Path,
        default=None,
        help="sim_fitted_latent_meta.json (if sim); default next to npz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output GIF path (default: run-dir / latent_dynamics_<source>.gif)",
    )
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Unused in plotting; accepted for pipeline consistency",
    )
    p.add_argument("--figsize", type=float, nargs=2, default=(7.2, 5.5))
    p.add_argument(
        "--avg-block",
        type=int,
        default=50,
        help="if >0 (default), each frame is mean μ over this many consecutive shots — fast, like "
        "make_latent_gif. Use 0 for full shot resolution (slow; pair with a large --frame-step).",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=100,
        help="if avg-block=0: one frame every this many shots (default 100). Ignored if avg-block>0.",
    )
    p.add_argument("--alpha-old", type=float, default=0.06)
    p.add_argument("--alpha-new", type=float, default=0.92)
    p.add_argument("--no-gifsicle", action="store_true")
    p.add_argument("--gifsicle-colors", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "fid_job_memory_noreset_large"
        )
    run_dir = Path(run_dir)

    if args.mu_npz is not None:
        mu_npz = Path(args.mu_npz)
    elif args.source == "sim":
        newp = run_dir / "data" / "sim_fitted_latent_mu_s_q.npz"
        oldp = run_dir / "sim_fitted_latent_mu_s_q.npz"
        mu_npz = newp if newp.is_file() else oldp
    else:
        newp = run_dir / "data" / "latent_dynamics_arrays.npz"
        oldp = run_dir / "latent_dynamics_arrays.npz"
        mu_npz = newp if newp.is_file() else oldp

    if not mu_npz.is_file():
        raise FileNotFoundError(
            f"{mu_npz} not found. Run analyze_latent_dynamics.py (encoded) or "
            "simulate_fitted_latent_zebra.py (sim), or pass --mu-npz."
        )

    mu_s_q, mu_bar_npz = _load_mu_npz(mu_npz)
    n_shots, n_q, latent_dim = mu_s_q.shape

    dyn_path = args.dynamics_json
    if dyn_path is None:
        newp = run_dir / "reports" / "latent_dynamics.json"
        oldp = run_dir / "latent_dynamics.json"
        dyn_path = newp if newp.is_file() else oldp
    sigma_line = ""
    if dyn_path.is_file():
        sigma_line = _title_suffix_from_json(_load_dynamics_json(dyn_path))

    sim_note = ""
    meta_path = args.sim_meta
    if args.source == "sim":
        if meta_path is None:
            newp = run_dir / "reports" / "sim_fitted_latent_meta.json"
            oldp = run_dir / "sim_fitted_latent_meta.json"
            meta_path = newp if newp.is_file() else oldp
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sim_note = (
                f"sim_mode={meta.get('sim_mode')}  σ_scale={meta.get('sigma_scale')}  "
                f"seed={meta.get('seed')}"
            )
            if meta.get("rho_median_used") is not None:
                sim_note += f"  ρ_med={meta.get('rho_median_used'):.4f}"

    if mu_bar_npz is not None and mu_bar_npz.shape == (n_q, latent_dim):
        shot_mean_per_q = mu_bar_npz
    else:
        shot_mean_per_q = mu_s_q.mean(axis=0)

    stem = mu_npz.stem
    title_base = f"Latent μ trajectories — {stem}  ({n_shots} shots × {n_q} qubits)"
    if sigma_line:
        title_base += f"\n{sigma_line}"
    if sim_note:
        title_base += f"\n{sim_note}"

    out = args.out
    if out is None:
        tag = "sim" if args.source == "sim" else "encoded"
        fig_dir = run_dir / "figures"
        if args.avg_block > 0:
            out = fig_dir / f"latent_dynamics_{tag}_avg{int(args.avg_block)}.gif"
        else:
            out = fig_dir / f"latent_dynamics_{tag}_step{int(args.frame_step)}.gif"

    fps = float(args.fps)
    dpi = int(args.dpi)
    figsize = (float(args.figsize[0]), float(args.figsize[1]))

    if args.avg_block > 0:
        blk = max(1, int(args.avg_block))
        zb = _block_mean_mu(mu_s_q, blk)
        n_b = zb.shape[0]

        def binned_progress():
            for k in range(n_b):
                yield (k, min((k + 1) * blk, n_shots))

        _write_gif(
            out,
            zb,
            n_qubits=n_q,
            shot_mean_per_q=shot_mean_per_q,
            fps=fps,
            dpi=dpi,
            figsize=figsize,
            title_prefix=f"{title_base}\n(mean μ over {blk} shots per frame)",
            shot_progress_fn=binned_progress(),
            alpha_old=float(args.alpha_old),
            alpha_new=float(args.alpha_new),
            drop_title_head_lines=2,
        )
    else:
        terminals = _build_terminal_indices(n_shots, args.frame_step)
        write_latent_gif_terminal_shots(
            out,
            mu_s_q,
            n_qubits=n_q,
            shot_mean_per_q=shot_mean_per_q,
            terminal_shot_indices=terminals,
            fps=fps,
            dpi=dpi,
            figsize=figsize,
            title_prefix=title_base,
            alpha_old=float(args.alpha_old),
            alpha_new=float(args.alpha_new),
            drop_title_head_lines=2,
        )

    if not args.no_gifsicle and _maybe_gifsicle_optimize(
        out, colors=int(args.gifsicle_colors)
    ):
        print("Optimized GIF with gifsicle.")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
