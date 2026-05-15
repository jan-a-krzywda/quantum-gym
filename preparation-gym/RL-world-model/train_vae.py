"""
Phase 3: Train MLP-VAE on shadow fingerprint dataset.

Input: 63-dim float32 shadow Pauli expectations (1-local + 2-local + 3-local).
MlpVAE (seq_len=63 → latent_dim) compresses noisy physical fingerprints to a
smooth latent manifold, acting as a denoiser against SPAM errors.

After training:
  - Encodes all states → μ_states (N, latent_dim)
  - Encodes GHZ target → μ_target (latent_dim,)
  - Saves checkpoint + latent arrays to RL-world-model/runs/vae/

Usage:
    python RL-world-model/train_vae.py --data RL-world-model/data/transition_dataset.npz
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
    p = argparse.ArgumentParser(description="Phase 3: Train MLP-VAE on shadow fingerprint dataset")
    p.add_argument("--data", type=Path, required=True,
                   help="transition_dataset.npz from generate_dataset.py")
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
    p.add_argument("--beta-start", type=float, default=0.02)
    p.add_argument("--beta-warmup", type=int, default=30)
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
    fingerprints = raw["fingerprints"]          # (M, n_features) float32 — 2D transition dataset

    # Handle legacy 3D format (N, n_samples, n_features) from old BFS pipeline
    if fingerprints.ndim == 3:
        N, n_samples_data, n_features = fingerprints.shape
        train_shots = min(args.train_shots, n_samples_data)
        x_train = fingerprints[:, :train_shots, :].reshape(N * train_shots, n_features).astype(np.float32)
        print(f"Dataset (3D): {N} states × {n_samples_data} samples × {n_features} features", flush=True)
    else:
        n_features = fingerprints.shape[1]
        x_train = fingerprints.astype(np.float32)   # (M, 63)
        print(f"Dataset (2D): {len(x_train)} transitions × {n_features} features", flush=True)

    n_qubits = int(raw["n_qubits"]) if "n_qubits" in raw else 3
    n_train = len(x_train)
    seq_len = n_features
    print(f"Training samples: {n_train:,}  seq_len={seq_len}", flush=True)

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
    
    # Cosine annealing scheduler to speed up fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    
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
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:4d} | β={beta_e:.3f} | loss={sum_loss/n_batches:.4f} "
                f"Epoch {epoch:4d} | β={beta_e:.3f} | lr={current_lr:.1e} | loss={sum_loss/n_batches:.4f} "
                f"(bce={sum_rec/n_batches:.4f} + β·kl={beta_e*sum_kl/n_batches:.4f}) "
                f"| {epoch_secs:.1f}s",
                flush=True,
            )
            
        scheduler.step()

    # --- Encode training fingerprints → μ_states ---
    print("Encoding fingerprints to latents...", flush=True)
    model.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(x_train).to(device)
        mu_states_list = []
        for start in range(0, len(x_all), 2048):
            mu_states_list.append(model.encode_mu(x_all[start:start+2048]).cpu().numpy())
    mu_states = np.concatenate(mu_states_list, axis=0)   # (M, latent_dim)
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
    )   # (256, 63)
    ghz_tensor = torch.from_numpy(ghz_fp).to(device)
    with torch.no_grad():
        mu_ghz = model.encode_mu(ghz_tensor).mean(dim=0).cpu().numpy()
    print(f"GHZ target μ: {mu_ghz}", flush=True)

    zero_fp = _sf_mod.shadow_fingerprint_batch(
        _sf_mod.prepare_zero_state(n_qubits),
        n_qubits, n_shots_per_sample=256, n_samples=256, rng=rng,
    )   # (256, 63)
    zero_tensor = torch.from_numpy(zero_fp).to(device)
    with torch.no_grad():
        mu_zero = model.encode_mu(zero_tensor).mean(dim=0).cpu().numpy()
    print(f"|000> start μ: {mu_zero}", flush=True)

    # --- Save checkpoint ---
    out_dir = args.out_dir if args.out_dir is not None else _HERE / "runs" / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "arch": "MlpVAE",
        "model": model.state_dict(),
        "seq_len": seq_len,          # = n_features = 63
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
        mu_states=mu_states,    # (M, latent_dim)
        mu_ghz=mu_ghz,          # (latent_dim,)
        mu_zero=mu_zero,        # (latent_dim,)
    )

    # --- Plot latent scatter coloured by fidelity (if available) ---
    import matplotlib.pyplot as plt

    if "fidelities" in raw.files:
        color_vals = raw["fidelities"]
        color_label = "GHZ fidelity"
    elif "distances" in raw.files:
        color_vals = raw["distances"]
        color_label = "GHZ distance"
    else:
        color_vals = np.arange(len(mu_states))
        color_label = "sample index"

    if mu_states.shape[1] > 2:
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu_states)
        mu_ghz_2d  = pca.transform([mu_ghz])[0]
        mu_zero_2d = pca.transform([mu_zero])[0]
        xlabel, ylabel = "PCA 1", "PCA 2"
    elif mu_states.shape[1] == 1:
        mu_2d = np.column_stack((mu_states[:, 0], color_vals))
        
        if "distance" in color_label.lower():
            y_ghz, y_zero = 0.0, 0.7071  # sqrt(1 - 0.5) ≈ 0.7071
        elif "fidelity" in color_label.lower():
            y_ghz, y_zero = 1.0, 0.5
        else:
            y_ghz, y_zero = 0, len(mu_states) // 2
            
        mu_ghz_2d  = [mu_ghz[0], y_ghz]
        mu_zero_2d = [mu_zero[0], y_zero]
        xlabel, ylabel = "μ₁", color_label
    else:
        mu_2d = mu_states
        mu_ghz_2d  = mu_ghz
        mu_zero_2d = mu_zero
        xlabel, ylabel = "μ₁", "μ₂"

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(mu_2d[:, 0], mu_2d[:, 1],
                    c=color_vals, cmap="RdBu_r", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label=color_label)
    ax.scatter(*mu_ghz_2d,  color="red",  s=120, marker="*", zorder=5, label="GHZ")
    ax.scatter(*mu_zero_2d, color="black", s=120, marker="^", zorder=5, label="|000>")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"VAE latent space — {len(mu_states)} samples  (latent_dim={args.latent_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "latent_scatter.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    # --- Plot top-3 beam search trajectories in latent space ---
    traj_path = _HERE / "data" / "beam_trajectories.npz"
    if traj_path.exists():
        print("Plotting top-3 beam search trajectories in latent space...", flush=True)

        # Dynamically load load_trajectories from beam_search.py
        import importlib.util as _ilu
        _bs_spec = _ilu.spec_from_file_location("beam_search", _HERE / "beam_search.py")
        _bs_mod = _ilu.module_from_spec(_bs_spec)
        _bs_spec.loader.exec_module(_bs_mod)

        trajectories = _bs_mod.load_trajectories(traj_path)
        # Already sorted by fidelity descending
        top3 = trajectories[:3]

        def encode_trajectory(traj: dict) -> np.ndarray:
            """Encode each step of a trajectory's angle_schedule → latent μ.

            Optimizations (keep sampling):
            - evolve a single Statevector incrementally by applying only the new layer
            - sample fingerprints per-step but stack all samples and call the VAE encoder once
            - average encoder outputs per step to obtain μ for that step
            """
            sched = traj["angle_schedule"]  # (n_rounds, 6)
            n_rounds = len(sched)
            if n_rounds == 0:
                return np.zeros((0, model.latent_dim), dtype=np.float32)

            # sampling parameters (kept stochastic to match real-device fingerprints)
            n_shots_per_sample = 256
            n_samples_per_step = 256

            # prepare list to accumulate fingerprint arrays per step
            fp_list = []  # each element: (n_samples_per_step, seq_len)

            # build initial statevector for first step, then evolve incrementally
            sv = _bs_mod.ghz_target_sv().__class__(_bs_mod.build_ansatz_circuit(sched[:1]))

            # NOTE: using the RNG passed from main for reproducibility
            for step in range(1, n_rounds + 1):
                if step > 1:
                    # apply only the new layer (sched[step-1]) to evolve sv
                    layer_angles = sched[step - 1]
                    QC = _bs_mod.QuantumCircuit
                    layer = QC(_bs_mod.N_QUBITS)
                    # forward layer: Ry(theta) Rz(phi) per qubit, then CZs
                    for q in range(_bs_mod.N_QUBITS):
                        theta = float(layer_angles[2 * q])
                        phi = float(layer_angles[2 * q + 1])
                        layer.ry(theta, q)
                        layer.rz(phi, q)
                    for q0, q1 in _bs_mod.CONNECTIVITY:
                        layer.cz(q0, q1)
                    sv = sv.evolve(layer)

                # sample fingerprints for this step (keeps stochastic, matches real-device style)
                fp = _sf_mod.shadow_fingerprint_batch(
                    sv, n_qubits,
                    n_shots_per_sample=n_shots_per_sample,
                    n_samples=n_samples_per_step,
                    rng=rng,
                )
                fp_list.append(fp.astype(np.float32))

            # Batch-encode all fingerprints at once to amortize PyTorch overhead
            # fp_stack: (n_rounds * n_samples_per_step, seq_len)
            fp_stack = np.vstack(fp_list)
            t = torch.from_numpy(fp_stack).to(device)
            with torch.no_grad():
                mu_all = model.encode_mu(t)  # (total_samples, latent_dim)
            mu_all = mu_all.cpu().numpy()

            # reshape back to (n_rounds, n_samples_per_step, latent_dim) and average
            latent_dim = mu_all.shape[1]
            mu_all = mu_all.reshape(n_rounds, n_samples_per_step, latent_dim)
            mus = mu_all.mean(axis=1)  # (n_rounds, latent_dim)
            return mus

        traj_colors = ["orange", "lime", "magenta"]
        traj_latents = []
        for i, traj in enumerate(top3):
            print(f"  Encoding trajectory {i+1}/3  fidelity={traj['fidelity']:.4f}", flush=True)
            mus = encode_trajectory(traj)
            traj_latents.append(mus)

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        sc2 = ax2.scatter(mu_2d[:, 0], mu_2d[:, 1],
                          c=color_vals, cmap="RdBu_r", alpha=0.3, s=6)
        plt.colorbar(sc2, ax=ax2, label=color_label)
        ax2.scatter(*mu_ghz_2d,  color="red",  s=120, marker="*", zorder=5, label="GHZ")
        ax2.scatter(*mu_zero_2d, color="black", s=120, marker="^", zorder=5, label="|000>")

        for i, (mus, traj) in enumerate(zip(traj_latents, top3)):
            # Project trajectory to 2D if needed
            if mu_states.shape[1] > 2:
                mus_2d = pca.transform(mus)
            elif mu_states.shape[1] == 1:
                mus_2d = np.column_stack((mus[:, 0], np.zeros(len(mus))))
            else:
                mus_2d = mus
            color = traj_colors[i]
            ax2.plot(mus_2d[:, 0], mus_2d[:, 1], color=color, lw=1.5,
                     label=f"Traj {i+1} (F={traj['fidelity']:.3f})")
            ax2.scatter(mus_2d[:, 0], mus_2d[:, 1], color=color, s=30, zorder=4)
            # Mark start and end
            ax2.scatter(*mus_2d[0],  color=color, s=80, marker="o", zorder=5)
            ax2.scatter(*mus_2d[-1], color=color, s=80, marker="D", zorder=5)

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(f"VAE latent space — top-3 beam trajectories  (latent_dim={args.latent_dim})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        traj_fig_path = out_dir / "latent_scatter_trajectories.png"
        fig2.savefig(traj_fig_path, dpi=150)
        plt.close()
        print(f"  {traj_fig_path}", flush=True)
    else:
        print(f"No beam trajectories found at {traj_path}, skipping trajectory plot.", flush=True)

    print(f"\nSaved:", flush=True)
    print(f"  {ckpt_path}", flush=True)
    print(f"  {out_dir / 'latents.npz'}", flush=True)
    print(f"  {fig_path}", flush=True)


if __name__ == "__main__":
    main()
