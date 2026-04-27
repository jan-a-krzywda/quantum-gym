"""
Phase 4: Train latent dynamics MLP (world model).

Uses the frozen VAE encoder latents and the transition dataset to train
    f_v: (μ_n, encode(a_n)) → μ_{n+1}

Action encoding: 6 independent one-hots of size 3  →  18-dim vector.
Each group of 3 = {-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT} for one rotation parameter.

Since transitions come from a noiseless statevector simulator, MSE loss is correct.

Usage:
    python RL-world-model/train_world_model.py \
        --dataset RL-world-model/data/transition_dataset.npz \
        --vae-ckpt RL-world-model/runs/vae/vae_checkpoint.pt
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

N_PARAMS = 6       # rotation parameters per round
N_CHOICES = 3      # {-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT}
ACTION_ENC_DIM = N_PARAMS * N_CHOICES  # 18


# ---------------------------------------------------------------------------
# Action encoding helpers
# ---------------------------------------------------------------------------

def action_indices_to_onehot(action_indices: np.ndarray) -> np.ndarray:
    """
    Convert (M, 6) int8 array of {-1, 0, +1} to (M, 18) float32 one-hot.

    Mapping per parameter: -1 → [1,0,0], 0 → [0,1,0], +1 → [0,0,1].
    """
    M = len(action_indices)
    onehot = np.zeros((M, ACTION_ENC_DIM), dtype=np.float32)
    choice_to_idx = {-1: 0, 0: 1, 1: 2}
    for m in range(M):
        for p in range(N_PARAMS):
            c = int(action_indices[m, p])
            onehot[m, p * N_CHOICES + choice_to_idx[c]] = 1.0
    return onehot


def action_multidiscrete_to_onehot(action: np.ndarray) -> np.ndarray:
    """
    Convert MultiDiscrete action (6,) with values in {0,1,2} to (18,) one-hot.
    Used by RL agent at inference time.
    """
    onehot = np.zeros(ACTION_ENC_DIM, dtype=np.float32)
    for p in range(N_PARAMS):
        onehot[p * N_CHOICES + int(action[p])] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentDynamicsMLP(nn.Module):
    """
    f_v(μ, a_enc) → μ'

    Input:  latent_dim + ACTION_ENC_DIM (18)
    Output: latent_dim
    """

    def __init__(self, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_enc_dim = ACTION_ENC_DIM
        self.net = nn.Sequential(
            nn.Linear(latent_dim + ACTION_ENC_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, mu: torch.Tensor, action_enc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([mu, action_enc], dim=-1))

    @torch.no_grad()
    def step(self, mu: torch.Tensor, action: np.ndarray) -> torch.Tensor:
        """Single step: action is (6,) MultiDiscrete indices {0,1,2}."""
        enc = torch.from_numpy(action_multidiscrete_to_onehot(action)).float().to(mu.device)
        return self.net(torch.cat([mu.unsqueeze(0), enc.unsqueeze(0)], dim=-1)).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def encode_fingerprints(
    vae_model,
    fingerprints: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode (M, 63) fingerprints → (M, latent_dim) latents using frozen VAE encoder."""
    vae_model.eval()
    M = len(fingerprints)
    latent_dim = vae_model.latent_dim
    mu_out = np.zeros((M, latent_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            x = torch.from_numpy(fingerprints[start:end]).to(device)
            mu_out[start:end] = vae_model.encode_mu(x).cpu().numpy()
    return mu_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: Train latent dynamics world model")
    p.add_argument("--dataset", type=Path, required=True,
                   help="transition_dataset.npz from generate_dataset.py")
    p.add_argument("--vae-ckpt", type=Path, required=True,
                   help="vae_checkpoint.pt from train_vae.py")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run dir (default: runs/world_model/)")
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


def _pick_device(prefer: str) -> torch.device:
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def _load_mlp_vae(ckpt_path: Path, device: torch.device):
    mlp_vae_spec = importlib.util.spec_from_file_location("mlp_vae", _HERE / "mlp_vae.py")
    mlp_vae_mod = importlib.util.module_from_spec(mlp_vae_spec)
    mlp_vae_spec.loader.exec_module(mlp_vae_mod)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seq_len = int(ckpt["seq_len"])
    latent_dim = int(ckpt["latent_dim"])
    hidden = int(ckpt["hidden"])
    model = mlp_vae_mod.MlpVAE(seq_len=seq_len, latent_dim=latent_dim, hidden=hidden)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, latent_dim


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _pick_device(args.device)

    # --- Load VAE ---
    print(f"Loading VAE from {args.vae_ckpt}...")
    vae, latent_dim = _load_mlp_vae(args.vae_ckpt, device)
    print(f"  latent_dim={latent_dim}")

    # --- Load transition dataset ---
    print(f"Loading dataset from {args.dataset}...")
    raw = np.load(args.dataset, allow_pickle=False)
    fps     = raw["fingerprints"].astype(np.float32)    # (M, 63)
    next_fps = raw["next_fps"].astype(np.float32)        # (M, 63)
    act_idx = raw["action_indices"]                      # (M, 6) int8

    M = len(fps)
    print(f"  {M} transitions, n_features={fps.shape[1]}")

    # --- Encode fingerprints → latents ---
    print("Encoding fingerprints to latents...")
    mu_src = encode_fingerprints(vae, fps, device)       # (M, latent_dim)
    mu_dst = encode_fingerprints(vae, next_fps, device)  # (M, latent_dim)
    print(f"  μ_src range: [{mu_src.min():.3f}, {mu_src.max():.3f}]")

    # --- Encode actions → 18-dim one-hots ---
    print("Encoding actions...")
    a_enc = action_indices_to_onehot(act_idx)  # (M, 18)

    # --- Train/val split ---
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(M)
    n_val = max(1, int(M * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"  train={len(train_idx)}, val={n_val}")

    def _make_loader(sel, shuffle):
        ds = TensorDataset(
            torch.from_numpy(mu_src[sel]),
            torch.from_numpy(a_enc[sel]),
            torch.from_numpy(mu_dst[sel]),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader   = _make_loader(val_idx,   shuffle=False)

    # --- Build model ---
    model = LatentDynamicsMLP(latent_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: in={latent_dim + ACTION_ENC_DIM}, hidden={args.hidden}, "
          f"out={latent_dim}, params={n_params:,}, device={device}")

    # --- Train ---
    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for mu_s, a_oh, mu_d in train_loader:
            mu_s, a_oh, mu_d = mu_s.to(device), a_oh.to(device), mu_d.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(mu_s, a_oh)
            loss = nn.functional.mse_loss(pred, mu_d)
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(mu_s)
        scheduler.step()
        train_loss /= len(train_idx)

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for mu_s, a_oh, mu_d in val_loader:
                    mu_s, a_oh, mu_d = mu_s.to(device), a_oh.to(device), mu_d.to(device)
                    val_loss += float(nn.functional.mse_loss(model(mu_s, a_oh), mu_d)) * len(mu_s)
            val_loss /= n_val
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val MSE: {best_val:.6f}")

    # --- Encode GHZ and |000> targets for downstream use ---
    sf_spec = importlib.util.spec_from_file_location("shadow_fingerprint", _HERE / "shadow_fingerprint.py")
    sf_mod = importlib.util.module_from_spec(sf_spec)
    sf_spec.loader.exec_module(sf_mod)

    rng_fp = np.random.default_rng(args.seed)
    n_qubits = 3
    ghz_fp = sf_mod.shadow_fingerprint_batch(
        sf_mod.prepare_ghz_state(n_qubits), n_qubits,
        n_shots_per_sample=256, n_samples=256, rng=rng_fp,
    )
    zero_fp = sf_mod.shadow_fingerprint_batch(
        sf_mod.prepare_zero_state(n_qubits), n_qubits,
        n_shots_per_sample=256, n_samples=256, rng=rng_fp,
    )
    vae.eval()
    with torch.no_grad():
        mu_ghz = vae.encode_mu(torch.from_numpy(ghz_fp).to(device)).mean(0).cpu().numpy()
        mu_zero = vae.encode_mu(torch.from_numpy(zero_fp).to(device)).mean(0).cpu().numpy()

    # --- Save ---
    out_dir = args.out_dir if args.out_dir is not None else _HERE / "runs" / "world_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "world_model.pt"
    torch.save({
        "model": model.state_dict(),
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "action_enc_dim": ACTION_ENC_DIM,
        "n_params": N_PARAMS,
        "n_choices": N_CHOICES,
        "mu_ghz": mu_ghz,
        "mu_zero": mu_zero,
        "best_val_mse": best_val,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
