#!/usr/bin/env python3
"""
Train the qubit-persona Conv-VAE on a FID memory pickle (no-reset: differential τ readout).

Default data file: ``first_tests/inspire/fid_job_memory_noreset_large.pkl`` (same convention
as ``zebra_plot.ipynb``). Processing matches ``tuna_fid_single_job.save_memory_3d_plots``
with ``reset_qubits=False``.

**Defaults** favour a short diagnostic run (20 epochs, larger batches, KL annealing). Scale up
with ``--epochs`` and tune ``--lr`` if the loss is stable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_FIRST_TESTS = Path(__file__).resolve().parent.parent
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

import matplotlib.pyplot as plt
import numpy as np

# NNPACK prints a C++ [W] line on some CPUs before nnpack.enabled can take effect.
if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch

# Prefer native CPU conv when NNPACK is unsupported (avoids failed init paths).
if not torch.cuda.is_available():
    torch.backends.nnpack.enabled = False

from torch.utils.data import DataLoader, TensorDataset

from ML.fid_data_io import build_stack_like_single_job, load_memory_pickle, stack_to_vae_tensors
from ML.plotting.style import apply_latent_zebra_style
from ML.vae_model import QubitConvVAE, vae_loss


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


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "inspire" / "fid_job_memory_noreset_large.pkl"


def _resolve_run_dirs(run_root: Path) -> dict[str, Path]:
    return {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "figures": run_root / "figures",
        "data": run_root / "data",
        "reports": run_root / "reports",
    }


def _beta_at_epoch(epoch: int, *, beta: float, beta_start: float, beta_warmup_epochs: int) -> float:
    """Linear KL annealing: ``beta_start`` → ``beta`` over the first ``beta_warmup_epochs``."""
    if beta_warmup_epochs <= 0:
        return float(beta)
    t = min(1.0, float(epoch + 1) / float(beta_warmup_epochs))
    return float(beta_start + t * (beta - beta_start))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conv-VAE on single-shot FID memory (no-reset)")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Pickle from get_memory() (default: inspire/fid_job_memory_noreset_large.pkl)",
    )
    p.add_argument("--n-tau", type=int, default=None, help="delay steps (default: infer / 50)")
    p.add_argument("--num-qubits", type=int, default=None, help="override qubit count")
    p.add_argument(
        "--reset-qubits",
        action="store_true",
        help="skip differential readout (use raw Z bits; reset-style jobs)",
    )
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="default 20 for a quick run; increase (e.g. 100–200) once curves look good",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="larger batches = fewer steps per epoch (lower if you hit OOM)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Adam LR; reduce (e.g. 1e-3) if loss NaNs or oscillates wildly",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="final KL weight after warmup: loss = bce_mean + beta * kl_mean",
    )
    p.add_argument(
        "--beta-start",
        type=float,
        default=0.05,
        help="initial KL weight (ignored if --beta-warmup-epochs is 0)",
    )
    p.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=10,
        help="linearly ramp KL from beta-start to beta over this many epochs (0 = constant beta)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | mps",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for latent_plot.png and vae_checkpoint.pt (default: ML/runs/<stem>)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="print every N epochs (default: 1 if epochs<=30 else 5)",
    )
    ns = p.parse_args()
    if ns.log_every is None:
        ns.log_every = 1 if ns.epochs <= 30 else 5
    if ns.log_every < 1:
        p.error("--log-every must be >= 1")
    return ns


def main() -> None:
    args = parse_args()
    apply_latent_zebra_style()
    data_path = args.data if args.data is not None else _default_data_path()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Place fid_job_memory_noreset_large.pkl there "
            "or pass --data."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mem = load_memory_pickle(data_path)
    built = build_stack_like_single_job(
        mem,
        num_qubits=args.num_qubits,
        n_tau=args.n_tau,
        reset_qubits=args.reset_qubits,
    )
    x_np, qubit_ids = stack_to_vae_tensors(built.stack)
    n_tau = x_np.shape[1]

    tensor_x = torch.from_numpy(x_np).unsqueeze(1)
    ds = TensorDataset(tensor_x)
    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    device = _pick_device(args.device)
    print(f"Device: {device}")
    if args.beta_warmup_epochs <= 0:
        print(f"Train: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, β={args.beta} (fixed)")
    else:
        print(
            f"Train: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, "
            f"β {args.beta_start} → {args.beta} over {args.beta_warmup_epochs} epochs"
        )
    model = QubitConvVAE(seq_len=n_tau, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    n_batches = max(1, len(loader))
    log_every = int(args.log_every)
    for epoch in range(args.epochs):
        beta_e = _beta_at_epoch(
            epoch,
            beta=args.beta,
            beta_start=args.beta_start,
            beta_warmup_epochs=args.beta_warmup_epochs,
        )
        sum_loss = sum_rec = sum_kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=use_cuda)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss, rec, kld = vae_loss(recon, batch, mu, logvar, beta=beta_e)
            loss.backward()
            opt.step()
            sum_loss += float(loss.item())
            sum_rec += float(rec.detach().item())
            sum_kl += float(kld.detach().item())
        if epoch % log_every == 0 or epoch == args.epochs - 1:
            kl_term = beta_e * sum_kl / n_batches
            print(
                f"Epoch {epoch:4d} | β={beta_e:.4f} | loss {sum_loss / n_batches:.5f} "
                f"(bce {sum_rec / n_batches:.5f} + β·kl {kl_term:.5f})"
            )

    out_root = args.out_dir
    if out_root is None:
        out_root = Path(__file__).resolve().parent / "runs" / data_path.stem
    dirs = _resolve_run_dirs(Path(out_root))
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "seq_len": n_tau,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "beta_start": args.beta_start,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_qubits": built.num_qubits,
        "n_shots": built.n_shots,
        "differential": built.differential,
        "data_path": str(data_path.resolve()),
    }
    ckpt_path = dirs["checkpoints"] / "vae_checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    model.eval()
    with torch.no_grad():
        enc = model.encoder(tensor_x.to(device))
        latent_mu = model.fc_mu(enc).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        latent_mu[:, 0],
        latent_mu[:, 1] if latent_mu.shape[1] > 1 else np.zeros(len(latent_mu)),
        c=qubit_ids,
        cmap="tab20",
        alpha=0.5,
        s=15,
    )
    plt.colorbar(sc, label="Qubit ID")
    plt.xlabel("Latent 1 (μ)")
    plt.ylabel("Latent 2 (μ)" if args.latent_dim > 1 else "Latent 2 (unused)")
    bdesc = (
        f"β={args.beta}"
        if args.beta_warmup_epochs <= 0
        else f"β: {args.beta_start}→{args.beta} over {args.beta_warmup_epochs} ep"
    )
    plt.title(
        f"Conv-VAE latent means — {data_path.name}\n"
        f"{built.n_shots} shots × {built.num_qubits} qubits × {n_tau} τ  |  {bdesc}  |  lr={args.lr}"
    )
    plt.grid(True, alpha=0.3)
    fig_path = dirs["figures"] / "latent_plot.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Wrote {ckpt_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
