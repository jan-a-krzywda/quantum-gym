"""
Phase 5: Train PPO agent in latent "dream" space.

The agent never touches the quantum simulator. It uses:
  - World model  f_v(μ, action_enc) → μ'   (LatentDynamicsMLP)
  - VAE decoder  dec(μ) → shadow (63-dim)   (MlpVAE decoder)
  - Reward       R = cosine_similarity(dec(μ'), target_shadow)
  - State        μ ∈ R^latent_dim
  - Actions      MultiDiscrete([3,3,3,3,3,3])
                 each param: 0=−ANGLE_INCREMENT, 1=0, 2=+ANGLE_INCREMENT

Usage:
    python RL-world-model/train_rl_agent.py \
        --world-model RL-world-model/runs/world_model/world_model.pt \
        --vae-ckpt   RL-world-model/runs/vae/vae_checkpoint.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

N_PARAMS  = 6
N_CHOICES = 3
ACTION_ENC_DIM = N_PARAMS * N_CHOICES  # 18


# ---------------------------------------------------------------------------
# World model (re-import to keep file self-contained)
# ---------------------------------------------------------------------------

def _load_world_model_class():
    spec = importlib.util.spec_from_file_location("train_world_model", _HERE / "train_world_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LatentDynamicsMLP, mod.action_multidiscrete_to_onehot


def _load_mlp_vae(ckpt_path: Path, device: torch.device):
    spec = importlib.util.spec_from_file_location("mlp_vae", _HERE / "mlp_vae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mod.MlpVAE(
        seq_len=int(ckpt["seq_len"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# PPO actor-critic
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """
    Shared backbone → separate actor heads (one per rotation parameter) + critic.

    Actor: 6 independent Categorical(3) distributions.
    Critic: scalar value estimate.
    """

    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        # One linear head per parameter → logits over 3 choices
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden, N_CHOICES) for _ in range(N_PARAMS)
        ])
        self.critic = nn.Linear(hidden, 1)

    def forward(self, mu: torch.Tensor):
        h = self.backbone(mu)
        logits = [head(h) for head in self.actor_heads]   # list of (B, 3)
        value = self.critic(h).squeeze(-1)                 # (B,)
        return logits, value

    def get_action_and_logprob(self, mu: torch.Tensor):
        logits, value = self(mu)
        dists = [Categorical(logits=l) for l in logits]
        actions = torch.stack([d.sample() for d in dists], dim=-1)   # (B, 6)
        log_prob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))  # (B,)
        entropy  = sum(d.entropy() for d in dists)                    # (B,)
        return actions, log_prob, entropy, value

    def evaluate(self, mu: torch.Tensor, actions: torch.Tensor):
        """Evaluate log_prob and entropy of given actions."""
        logits, value = self(mu)
        dists = [Categorical(logits=l) for l in logits]
        log_prob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropy  = sum(d.entropy() for d in dists)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Latent environment
# ---------------------------------------------------------------------------

class LatentEnv:
    """
    Latent-space MDP driven by the world model.

    State   : μ ∈ R^latent_dim
    Action  : (6,) array with values in {0,1,2}
              where 0=−ANGLE_INCREMENT, 1=0, 2=+ANGLE_INCREMENT
    Reward  : cosine_similarity(decoder(μ'), ghz_shadow) mapped to [0,1]
    Done    : reward > success_thresh  or  steps >= max_steps
    """

    def __init__(
        self,
        world_model,
        vae_decoder,
        mu_start: torch.Tensor,
        ghz_shadow: torch.Tensor,     # (63,) float32 — target shadow vector
        action_enc_fn,
        max_steps: int = 20,
        success_thresh: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.wm = world_model
        self.decoder = vae_decoder
        self.mu_start = mu_start.clone().to(device)
        self.ghz_shadow = ghz_shadow.to(device)
        self.action_enc_fn = action_enc_fn
        self.max_steps = max_steps
        self.success_thresh = success_thresh
        self.device = device
        self.mu = self.mu_start.clone()
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        self.mu = self.mu_start.clone()
        self.step_count = 0
        return self.mu.clone()

    @torch.no_grad()
    def step(self, action: np.ndarray) -> tuple[torch.Tensor, float, bool]:
        enc = torch.from_numpy(self.action_enc_fn(action)).float().to(self.device)
        self.mu = self.wm(self.mu.unsqueeze(0), enc.unsqueeze(0)).squeeze(0)
        self.step_count += 1

        # Decode μ → shadow, compute cosine similarity with target
        shadow = self.decoder(self.mu.unsqueeze(0)).squeeze(0)
        reward = float(F.cosine_similarity(
            shadow.unsqueeze(0), self.ghz_shadow.unsqueeze(0)
        ).item())
        # Map from [-1,1] to [0,1]
        reward = (reward + 1.0) / 2.0

        done = reward >= self.success_thresh or self.step_count >= self.max_steps
        return self.mu.clone(), reward, done

    @torch.no_grad()
    def fidelity(self) -> float:
        shadow = self.decoder(self.mu.unsqueeze(0)).squeeze(0)
        return float(F.cosine_similarity(
            shadow.unsqueeze(0), self.ghz_shadow.unsqueeze(0)
        ).item() * 0.5 + 0.5)


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Generalised Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_val = last_value
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        next_val = 0.0 if d else next_val
        delta = r + gamma * next_val - v
        gae = delta + gamma * lam * (0.0 if d else gae)
        advantages.insert(0, gae)
        next_val = v
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def train_ppo(
    env: LatentEnv,
    latent_dim: int,
    device: torch.device,
    *,
    total_steps: int = 200_000,
    steps_per_update: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 256,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    hidden: int = 128,
    log_every: int = 5,
) -> PPOActorCritic:
    policy = PPOActorCritic(latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    updates = total_steps // steps_per_update
    mu = env.reset().to(device)
    ep_rewards: list[float] = []
    ep_successes: list[bool] = []
    cur_ep_reward = 0.0
    update_count = 0

    for update in range(1, updates + 1):
        # --- Collect rollout ---
        buf_mu, buf_actions, buf_logp, buf_vals = [], [], [], []
        buf_rewards, buf_dones = [], []

        for _ in range(steps_per_update):
            mu_t = mu.detach()
            with torch.no_grad():
                actions, log_prob, _, value = policy.get_action_and_logprob(mu_t.unsqueeze(0))
            action_np = actions[0].cpu().numpy()

            mu_next, reward, done = env.step(action_np)
            mu_next = mu_next.to(device)

            buf_mu.append(mu_t)
            buf_actions.append(actions[0])
            buf_logp.append(log_prob[0])
            buf_vals.append(value[0])
            buf_rewards.append(reward)
            buf_dones.append(done)

            cur_ep_reward += reward
            if done:
                ep_rewards.append(cur_ep_reward)
                ep_successes.append(env.fidelity() >= env.success_thresh)
                cur_ep_reward = 0.0
                mu = env.reset().to(device)
            else:
                mu = mu_next

        # Bootstrap value for last state
        with torch.no_grad():
            _, last_val = policy(mu.unsqueeze(0))
            last_val = float(last_val[0])

        advantages, returns = compute_gae(
            buf_rewards,
            [float(v) for v in buf_vals],
            buf_dones,
            last_val,
            gamma=gamma, lam=lam,
        )

        # Convert buffers to tensors
        t_mu      = torch.stack(buf_mu)                                     # (T, latent_dim)
        t_actions = torch.stack(buf_actions)                                # (T, 6)
        t_logp    = torch.stack(buf_logp)                                   # (T,)
        t_adv     = torch.tensor(advantages, dtype=torch.float32, device=device)
        t_ret     = torch.tensor(returns,    dtype=torch.float32, device=device)
        t_adv     = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        # --- PPO update ---
        T = len(t_mu)
        perm = torch.randperm(T, device=device)
        for _ in range(n_epochs):
            for start in range(0, T, batch_size):
                idx = perm[start:start + batch_size]
                b_mu      = t_mu[idx]
                b_actions = t_actions[idx]
                b_old_lp  = t_logp[idx].detach()
                b_adv     = t_adv[idx]
                b_ret     = t_ret[idx]

                new_lp, entropy, new_val = policy.evaluate(b_mu, b_actions)

                ratio = (new_lp - b_old_lp).exp()
                pg1 = ratio * b_adv
                pg2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * b_adv
                actor_loss = -torch.min(pg1, pg2).mean()

                critic_loss = F.mse_loss(new_val, b_ret)
                ent_loss    = -entropy.mean()

                loss = actor_loss + vf_coef * critic_loss + ent_coef * ent_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        update_count += 1
        if update_count % log_every == 0 or update == updates:
            recent_rewards = ep_rewards[-50:] if ep_rewards else [0.0]
            succ_rate = np.mean(ep_successes[-50:]) if ep_successes else 0.0
            print(
                f"Update {update:4d}/{updates} | "
                f"avg_reward={np.mean(recent_rewards):.3f} | "
                f"success={succ_rate*100:.1f}% | "
                f"fidelity={env.fidelity():.4f}"
            )

    return policy


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_rollout(
    policy: PPOActorCritic,
    env: LatentEnv,
    device: torch.device,
    max_steps: int = 30,
) -> list[np.ndarray]:
    mu = env.reset().to(device)
    path = []
    print("\nGreedy policy rollout:")
    print(f"  start  fid={env.fidelity():.4f}  μ={mu.cpu().numpy()}")
    for _ in range(max_steps):
        logits, _ = policy(mu.unsqueeze(0))
        action_np = np.array([int(l.argmax().item()) for l in logits])
        mu_next, reward, done = env.step(action_np)
        mu = mu_next.to(device)
        path.append(action_np)
        delta_names = ["−δ", " 0", "+δ"]
        act_str = " ".join(f"p{p}:{delta_names[action_np[p]]}" for p in range(N_PARAMS))
        print(f"  step {len(path):2d}: [{act_str}]  reward={reward:.4f}  fid={env.fidelity():.4f}")
        if done:
            break
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5: Train PPO agent in latent dream space")
    p.add_argument("--world-model", type=Path, required=True,
                   help="world_model.pt from train_world_model.py")
    p.add_argument("--vae-ckpt", type=Path, required=True,
                   help="vae_checkpoint.pt from train_vae.py")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--steps-per-update", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=20,
                   help="Max steps per episode")
    p.add_argument("--success-thresh", type=float, default=0.95,
                   help="Shadow cosine similarity [0,1] counted as success")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
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
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = _pick_device(args.device)

    # --- Load world model ---
    LatentDynamicsMLP, action_enc_fn = _load_world_model_class()
    wm_ckpt = torch.load(args.world_model, map_location="cpu", weights_only=False)
    latent_dim = int(wm_ckpt["latent_dim"])
    hidden_wm  = int(wm_ckpt["hidden"])

    world_model = LatentDynamicsMLP(latent_dim, hidden=hidden_wm).to(device)
    world_model.load_state_dict(wm_ckpt["model"])
    world_model.eval()

    mu_ghz  = torch.from_numpy(wm_ckpt["mu_ghz"].astype(np.float32))
    mu_zero = torch.from_numpy(wm_ckpt["mu_zero"].astype(np.float32))
    print(f"World model: latent_dim={latent_dim}")
    print(f"μ_zero = {mu_zero.numpy()}")
    print(f"μ_GHZ  = {mu_ghz.numpy()}")

    # --- Load VAE decoder ---
    vae = _load_mlp_vae(args.vae_ckpt, device)

    # --- Compute GHZ shadow target ---
    sf_spec = importlib.util.spec_from_file_location("shadow_fingerprint", _HERE / "shadow_fingerprint.py")
    sf_mod = importlib.util.module_from_spec(sf_spec)
    sf_spec.loader.exec_module(sf_mod)

    rng_fp = np.random.default_rng(args.seed)
    ghz_fp = sf_mod.shadow_fingerprint_batch(
        sf_mod.prepare_ghz_state(3), 3,
        n_shots_per_sample=1024, n_samples=64, rng=rng_fp,
    ).mean(axis=0)   # (63,) — averaged reference shadow
    ghz_shadow = torch.from_numpy(ghz_fp).float().to(device)

    print(f"GHZ shadow norm: {float(ghz_shadow.norm()):.4f}")

    # --- Build environment ---
    env = LatentEnv(
        world_model=world_model,
        vae_decoder=vae.decoder,
        mu_start=mu_zero.to(device),
        ghz_shadow=ghz_shadow,
        action_enc_fn=action_enc_fn,
        max_steps=args.max_steps,
        success_thresh=args.success_thresh,
        device=device,
    )

    # --- Train PPO ---
    print(f"\nTraining PPO: total_steps={args.total_steps}, "
          f"steps_per_update={args.steps_per_update}, "
          f"max_ep_steps={args.max_steps}, device={device}")
    policy = train_ppo(
        env, latent_dim, device,
        total_steps=args.total_steps,
        steps_per_update=args.steps_per_update,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        lr=args.lr,
        ent_coef=args.ent_coef,
        hidden=args.hidden,
        log_every=args.log_every,
    )

    # --- Greedy rollout ---
    path = greedy_rollout(policy, env, device)

    # --- Save ---
    out_dir = args.out_dir if args.out_dir is not None else _HERE / "runs" / "ppo_agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ppo_agent.pt"
    torch.save({
        "model": policy.state_dict(),
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "n_params": N_PARAMS,
        "n_choices": N_CHOICES,
        "action_sequence": [a.tolist() for a in path],
        "mu_ghz": mu_ghz.numpy(),
        "mu_zero": mu_zero.numpy(),
    }, ckpt_path)

    # --- Training curve ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Avg episode reward (last 50 eps)")
    ax.set_title("PPO latent-dream training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "ppo_training_curve.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nSaved:")
    print(f"  {ckpt_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
