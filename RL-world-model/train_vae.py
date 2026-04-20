"""
Phase 3: Train MLP-VAE on BFS fingerprint dataset.

Each fingerprint row (one shot of one state) is a 120-dim binary vector.
MlpVAE (seq_len → latent_dim=2) is ~100x faster than Conv1D on CPU because
fingerprints are feature vectors, not temporal sequences needing conv filters.

After training:
  - Encodes all N states → μ_states (N, latent_dim)
  - Encodes GHZ target  → μ_target (latent_dim,)
  - Saves checkpoint + latent arrays to RL-world-model/runs/<name>/

Usage:
    python RL-world-model/train_vae.py --data RL-world-model/data/bfs_dataset.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ML.plotting.style import apply_latent_zebra_style

# Load MlpVAE from sibling file
def _load_mlp_vae_module():
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("mlp_vae", _HERE / "mlp_vae.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

_mlp_vae_mod = _load_mlp_vae_module()
MlpVAE = _mlp_vae_mod.MlpVAE
mlp_vae_loss = _mlp_vae_mod.mlp_vae_loss



def _pick_device(prefer: str) -> torch.device:
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _beta_schedule(epoch: int, beta: float, beta_start: float, warmup: int) -> float:
    if warmup <= 0:
        return beta
    t = min(1.0, (epoch + 1) / warmup)
    return beta_start + t * (beta - beta_start)


def encode_states(
    model: MlpVAE,
    fingerprints: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Encode N states → (N, latent_dim) by averaging μ across shots.

    fingerprints : (N, n_shots, seq_len)
    """
    N, n_shots, seq_len = fingerprints.shape
    model.eval()
    mu_states = np.zeros((N, model.latent_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(fingerprints[i].astype(np.float32)).to(device)
            # x: (n_shots, seq_len)
            mu = model.encode_mu(x)  # (n_shots, latent_dim)
            mu_states[i] = mu.mean(dim=0).cpu().numpy()

    return mu_states


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conv-VAE on BFS fingerprint dataset")
    p.add_argument("--data", type=Path, required=True,
                   help="BFS dataset .npz from generate_dataset.py")
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256,
                   help="MlpVAE hidden dim (default 256)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-shots", type=int, default=4,
                   help="Shots per state used for training (default 4). "
                        "Full n_shots used for encoding after training. "
                        "Keep small for CPU speed; VAE still sees 15k×4=60k diverse pairs.")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-start", type=float, default=0.05)
    p.add_argument("--beta-warmup", type=int, default=20)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run directory (default: RL-world-model/runs/vae)")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_latent_zebra_style()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load dataset ---
    raw = np.load(args.data, allow_pickle=True)
    fingerprints = raw["fingerprints"]          # (N, n_samples, n_features) float32
    transitions = raw["transitions"]            # (N, A) int32
    n_qubits = int(raw["n_qubits"])
    action_names = list(raw["action_names"])
    N, n_samples_data, n_features = fingerprints.shape

    print(f"Dataset: {N} states × {n_samples_data} samples × {n_features} features", flush=True)
    print(f"         n_qubits={n_qubits}, actions={len(action_names)}", flush=True)

    # Subsample shadow samples for training speed; all samples used for post-train encoding
    train_shots = min(args.train_shots, n_samples_data)
    seq_len = n_features  # alias kept for downstream compatibility
    x_train = fingerprints[:, :train_shots, :].reshape(N * train_shots, seq_len).astype(np.float32)
    n_train = len(x_train)
    print(f"Training samples: {N} states × {train_shots} shadow samples = {n_train:,}  "
          f"(encoding uses all {n_samples_data} samples)", flush=True)

    tensor_x = torch.from_numpy(x_train)  # (N*train_shots, seq_len) — MLP: no unsqueeze
    loader = DataLoader(
        TensorDataset(tensor_x),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    device = _pick_device(args.device)
    model = MlpVAE(seq_len=seq_len, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_batches = max(1, len(loader))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MlpVAE seq_len={seq_len}, latent_dim={args.latent_dim}, "
          f"hidden={args.hidden}, params={n_params:,}, device={device}, "
          f"{n_batches} batches/epoch", flush=True)

    # --- Train ---
    import time
    model.train()
    t_epoch_start = time.time()
    for epoch in range(args.epochs):
        beta_e = _beta_schedule(epoch, args.beta, args.beta_start, args.beta_warmup)
        sum_loss = sum_rec = sum_kl = 0.0
        for b_idx, (batch,) in enumerate(loader):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss, rec, kld = mlp_vae_loss(recon, batch, mu, logvar, beta=beta_e)
            loss.backward()
            opt.step()
            sum_loss += float(loss)
            sum_rec += float(rec)
            sum_kl += float(kld)

        t_now = time.time()
        epoch_secs = t_now - t_epoch_start
        t_epoch_start = t_now
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:4d} | β={beta_e:.3f} | loss={sum_loss/n_batches:.4f} "
                f"(bce={sum_rec/n_batches:.4f} + β·kl={beta_e*sum_kl/n_batches:.4f}) "
                f"| {epoch_secs:.1f}s",
                flush=True,
            )

    # --- Encode all states → μ_states (use full n_shots for accuracy) ---
    print("Encoding all states...", flush=True)
    mu_states = encode_states(model, fingerprints, device)  # (N, latent_dim)
    print(f"μ_states range: [{mu_states.min():.3f}, {mu_states.max():.3f}]", flush=True)

    # --- Encode GHZ + |000> targets via shadow fingerprints ---
    print("Encoding GHZ + |000> targets...", flush=True)
    import importlib.util as _ilu
    _sf_spec = _ilu.spec_from_file_location("shadow_fingerprint", _HERE / "shadow_fingerprint.py")
    _sf_mod = _ilu.module_from_spec(_sf_spec)
    _sf_spec.loader.exec_module(_sf_mod)
    rng = np.random.default_rng(args.seed)

    ghz_fp = _sf_mod.shadow_fingerprint_batch(
        _sf_mod.prepare_ghz_state(n_qubits),
        n_qubits, n_shots_per_sample=256, n_samples=256, rng=rng,
    )  # (256, n_features)
    ghz_tensor = torch.from_numpy(ghz_fp).to(device)
    model.eval()
    with torch.no_grad():
        mu_ghz = model.encode_mu(ghz_tensor).mean(dim=0).cpu().numpy()
    print(f"GHZ target μ: {mu_ghz}", flush=True)

    zero_fp = _sf_mod.shadow_fingerprint_batch(
        _sf_mod.prepare_zero_state(n_qubits),
        n_qubits, n_shots_per_sample=256, n_samples=256, rng=rng,
    )  # (256, n_features)
    zero_tensor = torch.from_numpy(zero_fp).to(device)
    with torch.no_grad():
        mu_zero = model.encode_mu(zero_tensor).mean(dim=0).cpu().numpy()
    print(f"|000> start μ: {mu_zero}", flush=True)

    # --- Save ---
    out_dir = args.out_dir if args.out_dir is not None else _HERE / "runs" / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "arch": "MlpVAE",
        "model": model.state_dict(),
        "seq_len": seq_len,        # = n_features = 36
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "n_qubits": n_qubits,
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "data_path": str(args.data.resolve()),
    }
    ckpt_path = out_dir / "vae_checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    np.savez_compressed(
        out_dir / "latents.npz",
        mu_states=mu_states,          # (N, latent_dim) — one per BFS state
        mu_ghz=mu_ghz,                # (latent_dim,)   — GHZ target
        mu_zero=mu_zero,              # (latent_dim,)   — |000> start
        transitions=transitions,      # (N, A) int32    — carried from dataset
        action_names=np.array(action_names),
    )

    # --- Load entanglement measures ---
    entanglement_measures = raw["entanglement_measures"]

    # --- Plot latent scatter ---
    import matplotlib.pyplot as plt

    def get_first_two_dimensions(mu_states, mu_ghz, mu_zero):
        if mu_states.shape[1] > 2:
            return mu_states[:, :2], mu_ghz[:2], mu_zero[:2]
        return mu_states, mu_ghz, mu_zero

    mu_states_2d, mu_ghz_2d, mu_zero_2d = get_first_two_dimensions(mu_states, mu_ghz, mu_zero)

    depths = raw["depths"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        mu_states_2d[:, 0], mu_states_2d[:, 1],
        c=depths, cmap="viridis", alpha=0.4, s=8, label="BFS states"
    )
    plt.colorbar(sc, ax=ax, label="BFS depth")
    ax.scatter(*mu_ghz_2d, color="red", s=120, marker="*", zorder=5, label="GHZ target")
    ax.scatter(*mu_zero_2d, color="cyan", s=120, marker="^", zorder=5, label="|000>")
    ax.set_xlabel("μ₁")
    ax.set_ylabel("μ₂")
    ax.set_title(f"VAE latent space — {N} BFS states  (latent_dim={args.latent_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "latent_scatter.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    # --- Plot latent scatter with entanglement ---
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        mu_states_2d[:, 0], mu_states_2d[:, 1],
        c=entanglement_measures, cmap="plasma", alpha=0.4, s=8, label="BFS states"
    )
    plt.colorbar(sc, ax=ax, label="Entanglement Measure")
    ax.scatter(*mu_ghz_2d, color="red", s=120, marker="*", zorder=5, label="GHZ target")
    ax.scatter(*mu_zero_2d, color="cyan", s=120, marker="^", zorder=5, label="|000>")
    ax.set_xlabel("μ₁")
    ax.set_ylabel("μ₂")
    ax.set_title(f"VAE latent space — {N} BFS states  (latent_dim={args.latent_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path_entanglement = out_dir / "latent_scatter_entanglement.png"
    fig.savefig(fig_path_entanglement, dpi=150)
    plt.close()

    # --- Plot latent scatter with PCA ---
    if mu_states.shape[1] > 2:
        pca = PCA(n_components=2)
        mu_states_2d = pca.fit_transform(mu_states)
    else:
        mu_states_2d = mu_states

    # Transform GHZ and |000> states to PCA space if latent_dim > 2
    if mu_states.shape[1] > 2:
        mu_ghz_pca = pca.transform([mu_ghz])[0]
        mu_zero_pca = pca.transform([mu_zero])[0]
    else:
        mu_ghz_pca = mu_ghz
        mu_zero_pca = mu_zero

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        mu_states_2d[:, 0], mu_states_2d[:, 1],
        c=entanglement_measures, cmap="plasma", alpha=0.4, s=8, label="BFS states"
    )
    plt.colorbar(sc, ax=ax, label="Entanglement Measure")
    ax.scatter(*mu_ghz_pca, color="red", s=120, marker="*", zorder=5, label="GHZ target")
    ax.scatter(*mu_zero_pca, color="cyan", s=120, marker="^", zorder=5, label="|000>")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"VAE latent space (PCA) — {N} BFS states  (latent_dim={args.latent_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "latent_scatter_pca.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nSaved:", flush=True)
    print(f"  {ckpt_path}", flush=True)
    print(f"  {out_dir / 'latents.npz'}", flush=True)
    print(f"  {fig_path}", flush=True)
    print(f"  {fig_path_entanglement}", flush=True)


if __name__ == "__main__":
    main()
