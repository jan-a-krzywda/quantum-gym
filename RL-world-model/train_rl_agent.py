"""
Phase 5: Train DQN agent in latent "dream" space.

The agent never touches the quantum simulator. It uses:
  - World model  f_v(μ, a) → μ'   (trained LatentDynamicsMLP)
  - Reward       R = -||μ' - μ_ghz||₂
  - State        μ ∈ R^latent_dim (starting at μ_zero each episode)
  - Actions      discrete, index into ACTION_NAMES_3Q

Training is pure in-latent DQN with ε-greedy exploration.
After training, decodes the greedy policy path and prints the action sequence.

Usage:
    python RL-world-model/train_rl_agent.py \
        --world-model RL-world-model/runs/vae/world_model.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# World model (re-import definition so this file is self-contained)
# ---------------------------------------------------------------------------

class LatentDynamicsMLP(nn.Module):
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
        return self.net(torch.cat([mu, action_onehot], dim=-1))

    @torch.no_grad()
    def step(self, mu: torch.Tensor, action_idx: int) -> torch.Tensor:
        onehot = torch.zeros(1, self.n_actions, device=mu.device)
        onehot[0, action_idx] = 1.0
        return self.net(torch.cat([mu.unsqueeze(0), onehot], dim=-1)).squeeze(0)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, mu, action, reward, mu_next, done):
        self.buf.append((mu, action, reward, mu_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        mu, a, r, mu_next, done = zip(*batch)
        return (
            torch.stack(mu),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(mu_next),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Environment (pure latent)
# ---------------------------------------------------------------------------

class LatentEnv:
    """
    Deterministic latent-space MDP.

    State  : μ ∈ R^latent_dim
    Action : discrete index 0..n_actions-1
    Reward : -||μ' - μ_target||₂   (dense distance reward)
    Done   : dist < success_thresh  or  steps >= max_steps
    """

    def __init__(
        self,
        world_model: LatentDynamicsMLP,
        mu_start: torch.Tensor,
        mu_target: torch.Tensor,
        max_steps: int = 20,
        success_thresh: float = 0.01,
    ):
        self.wm = world_model
        self.mu_start = mu_start.clone()
        self.mu_target = mu_target.clone()
        self.max_steps = max_steps
        self.success_thresh = success_thresh
        self.mu = mu_start.clone()
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        self.mu = self.mu_start.clone()
        self.step_count = 0
        return self.mu.clone()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        self.mu = self.wm.step(self.mu, action)
        self.step_count += 1
        dist = float(torch.norm(self.mu - self.mu_target))
        reward = -dist
        done = dist < self.success_thresh or self.step_count >= self.max_steps
        return self.mu.clone(), reward, done


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dqn(
    env: LatentEnv,
    n_actions: int,
    latent_dim: int,
    device: torch.device,
    *,
    episodes: int = 5000,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 3000,  # episodes over which ε decays
    target_update: int = 50,
    buffer_size: int = 50_000,
    min_buffer: int = 500,
    hidden: int = 128,
    log_every: int = 200,
) -> tuple[QNetwork, list[float]]:
    q_net = QNetwork(latent_dim, n_actions, hidden=hidden).to(device)
    q_target = QNetwork(latent_dim, n_actions, hidden=hidden).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size)
    episode_returns = []
    successes = []

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / eps_decay)
        mu = env.reset().to(device)
        ep_return = 0.0

        while True:
            # ε-greedy
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    action = int(q_net(mu.unsqueeze(0)).argmax(dim=1).item())

            mu_next, reward, done = env.step(action)
            mu_next = mu_next.to(device)
            buf.push(mu.cpu(), action, reward, mu_next.cpu(), float(done))
            mu = mu_next
            ep_return += reward

            if len(buf) >= min_buffer:
                mu_b, a_b, r_b, mun_b, d_b = buf.sample(batch_size)
                mu_b = mu_b.to(device)
                a_b = a_b.to(device)
                r_b = r_b.to(device)
                mun_b = mun_b.to(device)
                d_b = d_b.to(device)

                with torch.no_grad():
                    q_next = q_target(mun_b).max(dim=1).values
                    q_tgt = r_b + gamma * q_next * (1 - d_b)

                q_pred = q_net(mu_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_pred, q_tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                opt.step()

            if done:
                dist = float(torch.norm(mu - env.mu_target.to(device)))
                successes.append(dist < env.success_thresh)
                break

        episode_returns.append(ep_return)

        if ep % target_update == 0:
            q_target.load_state_dict(q_net.state_dict())

        if ep % log_every == 0 or ep == episodes - 1:
            recent = episode_returns[-log_every:]
            succ_rate = np.mean(successes[-log_every:]) if successes else 0.0
            dist_now = float(torch.norm(mu - env.mu_target.to(device)))
            print(
                f"Episode {ep:5d} | ε={eps:.3f} | "
                f"avg_ret={np.mean(recent):.3f} | "
                f"success={succ_rate*100:.1f}% | "
                f"d_target={dist_now:.4f}"
            )

    return q_net, episode_returns


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_rollout(
    q_net: QNetwork,
    env: LatentEnv,
    action_names: list[str],
    device: torch.device,
    max_steps: int = 30,
) -> list[str]:
    """Follow greedy policy from start and print path."""
    mu = env.reset().to(device)
    path = []
    print("\nGreedy policy rollout:")
    dist0 = float(torch.norm(mu - env.mu_target.to(device)))
    print(f"  start  d={dist0:.4f}  μ={mu.cpu().numpy()}")
    for _ in range(max_steps):
        action = int(q_net(mu.unsqueeze(0)).argmax(dim=1).item())
        mu_next, _, done = env.step(action)
        mu = mu_next.to(device)
        dist = float(torch.norm(mu - env.mu_target.to(device)))
        act_name = action_names[action]
        path.append(act_name)
        print(f"  → {act_name:<6}  d={dist:.4f}  μ={mu.cpu().numpy()}")
        if done:
            break
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN agent in latent dream space")
    p.add_argument("--world-model", type=Path, required=True,
                   help="world_model.pt from train_world_model.py")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=20,
                   help="Max steps per episode (GHZ needs 7 min, pad for exploration)")
    p.add_argument("--success-thresh", type=float, default=0.02,
                   help="Latent distance to GHZ counted as success")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eps-decay", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
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
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Load world model ---
    ckpt = torch.load(args.world_model, map_location="cpu", weights_only=False)
    latent_dim = int(ckpt["latent_dim"])
    n_actions = int(ckpt["n_actions"])
    hidden = int(ckpt["hidden"])
    action_names = list(ckpt["action_names"])
    mu_ghz = torch.from_numpy(ckpt["mu_ghz"]).float()
    mu_zero = torch.from_numpy(ckpt["mu_zero"]).float()

    device = _pick_device(args.device)
    world_model = LatentDynamicsMLP(latent_dim, n_actions, hidden=hidden).to(device)
    world_model.load_state_dict(ckpt["model"])
    world_model.eval()

    print(f"World model: latent_dim={latent_dim}, n_actions={n_actions}")
    print(f"μ_zero  = {mu_zero.numpy()}")
    print(f"μ_GHZ   = {mu_ghz.numpy()}")
    print(f"dist(zero→GHZ) = {float(torch.norm(mu_zero - mu_ghz)):.4f}")
    print(f"Actions: {action_names}")

    # --- Build environment ---
    env = LatentEnv(
        world_model=world_model,
        mu_start=mu_zero.to(device),
        mu_target=mu_ghz.to(device),
        max_steps=args.max_steps,
        success_thresh=args.success_thresh,
    )

    # --- Train DQN ---
    print(f"\nTraining DQN: {args.episodes} episodes, max_steps={args.max_steps}, "
          f"success_thresh={args.success_thresh}, device={device}")
    q_net, returns = train_dqn(
        env, n_actions, latent_dim, device,
        episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        eps_decay=args.eps_decay,
        hidden=args.hidden,
        log_every=max(1, args.episodes // 25),
    )

    # --- Greedy rollout ---
    path = greedy_rollout(q_net, env, action_names, device)
    print(f"\nFound gate sequence: {' → '.join(path)}")

    # --- Save ---
    out_dir = args.out_dir if args.out_dir is not None else args.world_model.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "dqn_agent.pt"
    torch.save({
        "model": q_net.state_dict(),
        "latent_dim": latent_dim,
        "n_actions": n_actions,
        "hidden": args.hidden,
        "action_names": action_names,
        "gate_sequence": path,
        "mu_ghz": mu_ghz.numpy(),
        "mu_zero": mu_zero.numpy(),
    }, ckpt_path)

    # --- Training curve ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    window = max(1, len(returns) // 100)
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, lw=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("DQN latent-dream training curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "dqn_training_curve.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nSaved:")
    print(f"  {ckpt_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
