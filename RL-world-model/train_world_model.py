"""
Phase 4: Train latent dynamics MLP (world model).

Uses the frozen VAE encoder latents (μ_states) and the BFS transition table
to train f_v: (μ_i, one_hot(a)) → μ_j.

Since all transitions are deterministic (statevector sim, no noise), MSE loss
is correct — no MDN needed.

Usage:
    python RL-world-model/train_world_model.py \
        --latents RL-world-model/runs/vae/latents.npz
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentDynamicsMLP(nn.Module):
    """
    f_v(μ, a) → μ'

    Input:  latent_dim + n_actions (concat of μ and one-hot action)
    Output: latent_dim (predicted next latent)
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, mu: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mu, action_onehot], dim=-1)
        return self.net(x)

    def predict(self, mu: torch.Tensor, action_idx: int) -> torch.Tensor:
        """Single-step predict from raw action index."""
        onehot = torch.zeros(1, self.n_actions, device=mu.device)
        onehot[0, action_idx] = 1.0
        mu_in = mu.unsqueeze(0) if mu.dim() == 1 else mu
        return self.net(torch.cat([mu_in, onehot], dim=-1)).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_transition_dataset(
    mu_states: np.ndarray,   # (N, latent_dim)
    transitions: np.ndarray, # (N, A) int32  — -1 = frontier (no transition)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat arrays of (mu_src, action_onehot, mu_dst) over valid transitions.

    Returns
    -------
    mu_src   : (M, latent_dim) float32
    a_onehot : (M, n_actions)  float32
    mu_dst   : (M, latent_dim) float32
    """
    N, A = transitions.shape
    latent_dim = mu_states.shape[1]

    src_list, act_list, dst_list = [], [], []
    eye = np.eye(A, dtype=np.float32)

    for i in range(N):
        for j in range(A):
            k = transitions[i, j]
            if k < 0:
                continue
            src_list.append(mu_states[i])
            act_list.append(eye[j])
            dst_list.append(mu_states[k])

    return (
        np.stack(src_list).astype(np.float32),
        np.stack(act_list).astype(np.float32),
        np.stack(dst_list).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent dynamics MLP (world model)")
    p.add_argument("--latents", type=Path, required=True,
                   help="latents.npz from train_vae.py")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of transitions held out for validation")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run dir (default: alongside latents.npz)")
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load latents ---
    raw = np.load(args.latents, allow_pickle=True)
    mu_states = raw["mu_states"].astype(np.float32)   # (N, latent_dim)
    transitions = raw["transitions"]                   # (N, A)
    mu_ghz = raw["mu_ghz"].astype(np.float32)
    mu_zero = raw["mu_zero"].astype(np.float32)
    action_names = list(raw["action_names"])

    N, latent_dim = mu_states.shape
    A = transitions.shape[1]
    print(f"Latents: {N} states, latent_dim={latent_dim}, {A} actions")
    print(f"μ_ghz = {mu_ghz},  μ_zero = {mu_zero}")

    # --- Build transition dataset ---
    mu_src, a_onehot, mu_dst = build_transition_dataset(mu_states, transitions)
    M = len(mu_src)
    print(f"Transitions: {M} valid pairs ({M/(N*A)*100:.1f}% of N×A)")

    # Train/val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(M)
    n_val = max(1, int(M * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(mu_src[idx]),
            torch.from_numpy(a_onehot[idx]),
            torch.from_numpy(mu_dst[idx]),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx, shuffle=False)

    device = _pick_device(args.device)
    model = LatentDynamicsMLP(latent_dim, A, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"Model: in={latent_dim+A}, hidden={args.hidden}, out={latent_dim}, device={device}")

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
                    pred = model(mu_s, a_oh)
                    val_loss += float(nn.functional.mse_loss(pred, mu_d)) * len(mu_s)
            val_loss /= len(val_idx)
            print(f"Epoch {epoch:4d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val MSE: {best_val:.6f}")

    # --- Save ---
    out_dir = args.out_dir if args.out_dir is not None else args.latents.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "world_model.pt"
    torch.save({
        "model": model.state_dict(),
        "latent_dim": latent_dim,
        "n_actions": A,
        "hidden": args.hidden,
        "action_names": action_names,
        "mu_ghz": mu_ghz,
        "mu_zero": mu_zero,
        "best_val_mse": best_val,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # --- Quick sanity: rollout GHZ path in latent space ---
    # H0→H1→CZ01→H1→H2→CZ12→H2
    ghz_actions = ["H0", "H1", "CZ01", "H1", "H2", "CZ12", "H2"]
    model.eval()
    with torch.no_grad():
        mu_t = torch.from_numpy(mu_zero).to(device)
        print(f"\nGHZ rollout in latent space:")
        print(f"  start |000>  μ={mu_t.cpu().numpy()}")
        for act in ghz_actions:
            if act not in action_names:
                print(f"  action {act!r} not in action set — skip")
                continue
            j = action_names.index(act)
            mu_t = model.predict(mu_t, j)
            dist = float(torch.norm(mu_t - torch.from_numpy(mu_ghz).to(device)))
            print(f"  → {act}  μ={mu_t.cpu().numpy()}  d_GHZ={dist:.4f}")


if __name__ == "__main__":
    main()
