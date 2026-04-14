#!/usr/bin/env python3
"""
Simulate synthetic FID memory using **fitted** per-qubit latent means μ̄ and **joint**
increment covariance Σ from ``analyze_latent_dynamics.py`` (``latent_dynamics.json``),
then decode through the trained VAE and run the same ``save_memory_3d_plots`` pipeline as
``zebra_plot.ipynb`` (GIFs, co-click matrices, τ×repetition, derived ``.npz``).

**Modes**
- ``rw``: μ[0]=μ̄, then μ[s+1]=μ[s]+ε with ε ~ N(0, σ²Σ) (correlated Gaussian random walk).
- ``ou``: vector OU-style mean reversion toward μ̄ using median AR(1) ρ from the dynamics
  JSON: μ[s+1]=μ̄+ρ(μ[s]−μ̄)+ε with ε ~ N(0,(1−ρ²)σ²Σ) (heuristic multivariate scale).

**Re-decode only:** pass ``--from-mu-npz path/to/sim_fitted_latent_mu_s_q.npz`` to skip
simulation and rebuild pickle + zebra outputs from saved ``mu_s_q`` (writes
``sim_zebra_decode_meta.json``; does not overwrite the NPZ). Use ``--sample-mode threshold``
for deterministic 0.5 rounding (often clearer than Bernoulli).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
if not torch.cuda.is_available():
    torch.backends.nnpack.enabled = False

from ML.vae_model import QubitConvVAE  # noqa: E402


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


def integrate_differential_along_tau(
    diff: np.ndarray, *, implicit_prior: int = 0
) -> np.ndarray:
    """Inverse of differential readout using properly vectorized accumulation."""
    prior = np.uint8(int(implicit_prior) & 1)
    diff = np.asarray(diff, dtype=np.uint8, order="C")

    # Cumulative XOR along the tau axis (axis=2)
    raw = np.bitwise_xor.accumulate(diff, axis=2)

    # Apply the implicit prior to the whole array if it is 1
    if prior != 0:
        raw = np.bitwise_xor(raw, prior)

    return raw


def stack_shot_to_bitstring(shot_stack_jk: np.ndarray) -> str:
    n_q, n_tau = shot_stack_jk.shape
    n_c = n_q * n_tau
    chars = ["0"] * n_c
    for k in range(n_tau):
        for j in range(n_q):
            cbit = k * n_q + j
            chars[n_c - 1 - cbit] = "1" if shot_stack_jk[j, k] else "0"
    return "".join(chars)


def stack_to_memory_list(stack_snjk: np.ndarray) -> list[str]:
    return [stack_shot_to_bitstring(stack_snjk[s]) for s in range(stack_snjk.shape[0])]


def _psd_cholesky(S: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    d = S.shape[0]
    j = jitter
    for _ in range(10):
        try:
            return np.linalg.cholesky(S + j * np.eye(d))
        except np.linalg.LinAlgError:
            j *= 10.0
    raise ValueError("Cholesky failed for increment covariance")


def load_dynamics(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    mu_bar : (n_q, L)
    Sigma : (2*n_q, 2*n_q)
    rho_median : median AR(1) ρ over qubits×dims (for OU), nan if missing
    """
    with path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    n_q = int(data["n_qubits"])
    L = int(data["latent_dim"])
    mb = data["mu_bar_per_qubit"]
    mu_bar = np.zeros((n_q, L), dtype=np.float64)
    for q in range(n_q):
        m = mb[str(q)]
        mu_bar[q, 0] = float(m["mu_1"])
        mu_bar[q, 1] = float(m["mu_2"])
    Sigma = np.asarray(data["joint_increment_covariance"], dtype=np.float64)
    rho_grid = np.asarray(data["ar1_rho_per_qubit_dim"], dtype=np.float64)
    rho_med = float(np.nanmedian(rho_grid))
    if not np.isfinite(rho_med):
        rho_med = 0.5
    rho_med = float(np.clip(rho_med, -0.999, 0.999))
    return mu_bar, Sigma, rho_med


def simulate_mu_rw(
    mu_bar: np.ndarray,
    Sigma: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
    *,
    sigma_scale: float,
) -> np.ndarray:
    """μ[0]=μ̄; increments ~ N(0, σ²Σ)."""
    n_q, L = mu_bar.shape
    d = n_q * L
    Lm = _psd_cholesky(Sigma * (sigma_scale**2))
    mu_sq = np.zeros((n_shots, n_q, L), dtype=np.float64)
    mu_sq[0] = mu_bar
    for s in range(n_shots - 1):
        xi = rng.standard_normal(d)
        mu_sq[s + 1] = mu_sq[s] + (Lm @ xi).reshape(n_q, L)
    return mu_sq


def simulate_mu_iid_mean(
    mu_bar: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
    *,
    sigma_white: float,
) -> np.ndarray:
    """
    Independent N(0,1) noise per (shot, qubit) plus ``mu_bar[q]`` — stays near the training
    manifold (like ``generate_vae_gif``) while separating qubits by mean. Does **not** use the
    fitted increment covariance (no correlated OU/RW drift that leaves the manifold).
    """
    n_q, L = mu_bar.shape
    out = np.empty((n_shots, n_q, L), dtype=np.float64)
    for s in range(n_shots):
        out[s] = mu_bar + sigma_white * rng.standard_normal((n_q, L))
    return out


def simulate_mu_ou(
    mu_bar: np.ndarray,
    Sigma: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
    *,
    rho: float,
    sigma_scale: float,
) -> np.ndarray:
    """μ[s+1]=μ̄+ρ(μ[s]−μ̄)+ε with ε ~ N(0, (1−ρ²)σ²Σ)."""
    n_q, L = mu_bar.shape
    d = n_q * L
    var_scale = max(1e-8, 1.0 - rho**2) * (sigma_scale**2)
    Lm = _psd_cholesky(Sigma * var_scale)
    mu_sq = np.zeros((n_shots, n_q, L), dtype=np.float64)
    # Start at stationary mean + small noise
    mu_sq[0] = mu_bar + 0.05 * (Lm @ rng.standard_normal(d)).reshape(n_q, L)
    for s in range(n_shots - 1):
        xi = rng.standard_normal(d)
        eps = (Lm @ xi).reshape(n_q, L)
        mu_sq[s + 1] = mu_bar + rho * (mu_sq[s] - mu_bar) + eps
    return mu_sq


@torch.no_grad()
def decode_mu_to_diff_stack(
    model: QubitConvVAE,
    mu_s_q: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    sample_mode: str,
) -> np.ndarray:
    """``mu_s_q`` (S, Q, L) → differential stack (S, Q, n_tau) uint8."""
    n_shots, n_q, L = mu_s_q.shape
    seq_len = model.seq_len
    flat = mu_s_q.reshape(-1, L)
    use_cuda = device.type == "cuda"
    outs: list[torch.Tensor] = []
    z_t = torch.from_numpy(flat.astype(np.float32))
    for i in range(0, flat.shape[0], batch_size):
        z = z_t[i : i + batch_size].to(device, non_blocking=use_cuda)
        h = model.decoder_input(z)
        recon = model.decoder(h).clamp(0.0, 1.0)
        if sample_mode == "bernoulli":
            bits = torch.bernoulli(recon)
        else:
            bits = (recon >= 0.5).float()
        outs.append(bits.squeeze(1).cpu())
    x = torch.cat(outs, dim=0).numpy().astype(np.uint8)
    return x.reshape(n_shots, n_q, seq_len)


def _default_ckpt() -> Path:
    base = Path(__file__).resolve().parents[1] / "runs" / "fid_job_memory_noreset_large"
    new_path = base / "checkpoints" / "vae_checkpoint.pt"
    old_path = base / "vae_checkpoint.pt"
    return new_path if new_path.is_file() else old_path


def _run_root_from_ckpt(ckpt_path: Path) -> Path:
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _ensure_tuna_module_on_path() -> None:
    """Add the directory containing ``tuna_fid_single_job.py`` to ``sys.path``."""
    candidates = (
        _FIRST_TESTS / "inspire",
        _FIRST_TESTS / "quantum_code",
    )
    for d in candidates:
        if (d / "tuna_fid_single_job.py").is_file():
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
    raise ModuleNotFoundError(
        "Could not locate tuna_fid_single_job.py in first_tests/inspire or first_tests/quantum_code"
    )


def _resolve_source_pickle_path(path_from_ckpt: Path, run_root: Path) -> Path | None:
    """Resolve moved acquisition pickle locations from checkpoint metadata."""
    candidates = []
    if str(path_from_ckpt):
        candidates.append(path_from_ckpt.expanduser())
    name = path_from_ckpt.name if path_from_ckpt.name else "fid_job_memory_noreset_large.pkl"
    candidates.extend(
        [
            _FIRST_TESTS / "inspire" / name,
            _FIRST_TESTS / "quantum_code" / name,
            _FIRST_TESTS / "quantum_code" / "data" / name,
            run_root / "data" / name,
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_taus_from_source_pickle(path: Path, *, expected_len: int) -> np.ndarray | None:
    """Read tau metadata from acquisition pickle payload if available."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    taus = obj.get("taus")
    if taus is None:
        return None
    try:
        tau_arr = np.asarray(taus, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if tau_arr.size != int(expected_len):
        return None
    return tau_arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zebra plots from VAE + fitted latent dynamics (μ̄, Σ)"
    )
    p.add_argument(
        "--dynamics-json",
        type=Path,
        default=None,
        help="latent_dynamics.json (default: next to --ckpt)",
    )
    p.add_argument("--ckpt", type=Path, default=None, help="vae_checkpoint.pt")
    p.add_argument(
        "--sim-mode",
        choices=("rw", "ou", "iid-mean"),
        default="rw",
        help="rw/ou = correlated increments from Σ (can leave decoder manifold). "
        "iid-mean = ε~N(0,I) per (s,q) + μ̄_q (decay-friendly; see generate_vae_gif --dynamics-json).",
    )
    p.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="rw/ou: scale applied to √Σ. iid-mean: std of Gaussian ε (default 1).",
    )
    p.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="override shot count (default: from dynamics JSON)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--prefix",
        type=str,
        default="sim_fitted_zebra_memory_3d",
        help="output prefix for zebra-style files",
    )
    p.add_argument("--max-shots", type=int, default=5000)
    p.add_argument("--gif-2d-rep-bin", type=int, default=100)
    p.add_argument("--include-3d", action="store_true")
    p.add_argument(
        "--pickle-name",
        type=str,
        default="sim_fitted_fid_memory.pkl",
    )
    p.add_argument(
        "--sample-mode",
        choices=("bernoulli", "threshold"),
        default="bernoulli",
        help="threshold = round decoder p at 0.5 (no extra Bernoulli noise; often clearer stripes)",
    )
    p.add_argument(
        "--from-mu-npz",
        type=Path,
        default=None,
        help="Load ``mu_s_q`` from this NPZ (e.g. sim_fitted_latent_mu_s_q.npz) and skip latent "
        "simulation; still decodes to pickle + zebra plots. Dynamics JSON is optional.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.ckpt if args.ckpt is not None else _default_ckpt()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    seq_len = int(ckpt["seq_len"])
    latent_dim = int(ckpt["latent_dim"])
    num_qubits = int(ckpt["num_qubits"])

    run_root = Path(args.out_dir) if args.out_dir is not None else _run_root_from_ckpt(ckpt_path.resolve())
    out_fig = run_root / "figures"
    out_data = run_root / "data"
    out_reports = run_root / "reports"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dyn_path: Path | None = args.dynamics_json
    if dyn_path is None:
        candidate_new = run_root / "reports" / "latent_dynamics.json"
        candidate_old = run_root / "latent_dynamics.json"
        dyn_path = candidate_new if candidate_new.is_file() else candidate_old

    from_npz = args.from_mu_npz
    if from_npz is not None:
        from_npz = Path(from_npz)
        if not from_npz.is_file():
            raise FileNotFoundError(from_npz)
        znp = np.load(from_npz, allow_pickle=False)
        if "mu_s_q" not in znp.files:
            raise KeyError(f"{from_npz} must contain mu_s_q")
        mu_s_q = np.asarray(znp["mu_s_q"], dtype=np.float64)
        n_shots, n_q, L = mu_s_q.shape
        if L != latent_dim or n_q != num_qubits:
            raise ValueError(
                f"mu_s_q shape ({n_shots},{n_q},{L}) incompatible with ckpt "
                f"(num_qubits={num_qubits}, latent_dim={latent_dim})"
            )
        if args.n_shots is not None and int(args.n_shots) != n_shots:
            raise ValueError(
                f"--n-shots {args.n_shots} does not match NPZ n_shots={n_shots}"
            )
        mu_bar = None
        if "mu_bar" in znp.files:
            mu_bar = np.asarray(znp["mu_bar"], dtype=np.float64)
        rho_med = float("nan")
        if dyn_path.is_file():
            _, _, rho_med = load_dynamics(dyn_path)
        n_shots = int(mu_s_q.shape[0])
    else:
        if not dyn_path.is_file():
            raise FileNotFoundError(
                f"{dyn_path} not found. Run analyze_latent_dynamics.py first, or pass "
                "--from-mu-npz with saved latents."
            )

        mu_bar, Sigma, rho_med = load_dynamics(dyn_path)
        n_q, L = mu_bar.shape
        if num_qubits != n_q or latent_dim != L:
            raise ValueError(
                f"Checkpoint qubits/latent ({num_qubits},{latent_dim}) "
                f"!= dynamics ({n_q},{L})"
            )

        with open(dyn_path, encoding="utf-8") as f:
            dyn_meta = json.load(f)
        n_shots_json = int(dyn_meta["n_shots"])
        n_shots = args.n_shots if args.n_shots is not None else n_shots_json

        if args.sim_mode == "rw":
            mu_s_q = simulate_mu_rw(
                mu_bar, Sigma, n_shots, rng, sigma_scale=args.sigma_scale
            )
        elif args.sim_mode == "ou":
            mu_s_q = simulate_mu_ou(
                mu_bar,
                Sigma,
                n_shots,
                rng,
                rho=rho_med,
                sigma_scale=args.sigma_scale,
            )
        else:
            mu_s_q = simulate_mu_iid_mean(
                mu_bar,
                n_shots,
                rng,
                sigma_white=float(args.sigma_scale),
            )

    device = _pick_device(args.device)
    model = QubitConvVAE(seq_len=seq_len, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diff_stack = decode_mu_to_diff_stack(
        model,
        mu_s_q,
        device=device,
        batch_size=args.batch_size,
        sample_mode=args.sample_mode,
    )
    raw_stack = integrate_differential_along_tau(diff_stack, implicit_prior=0)
    mem = stack_to_memory_list(raw_stack)

    decode_only = args.from_mu_npz is not None
    if decode_only:
        meta = {
            "from_mu_npz": str(Path(args.from_mu_npz).resolve()),
            "ckpt_path": str(ckpt_path.resolve()),
            "dynamics_json": str(dyn_path.resolve()) if dyn_path.is_file() else None,
            "rho_median_from_dynamics_json": float(rho_med)
            if np.isfinite(rho_med)
            else None,
            "sample_mode": args.sample_mode,
            "n_shots": n_shots,
            "seed": args.seed,
            "prefix": args.prefix,
        }
        meta_path = out_reports / "sim_zebra_decode_meta.json"
    else:
        meta = {
            "dynamics_json": str(dyn_path.resolve()),
            "ckpt_path": str(ckpt_path.resolve()),
            "sim_mode": args.sim_mode,
            "sigma_scale": args.sigma_scale,
            "rho_median_used": rho_med if args.sim_mode == "ou" else None,
            "n_shots": n_shots,
            "seed": args.seed,
        }
        if args.sim_mode == "iid-mean":
            meta["latent_note"] = (
                "iid-mean: ε~N(0,I) per (shot,qubit) + μ̄_q; increments not from Σ (decoder-friendly)."
            )
        meta_path = out_reports / "sim_fitted_latent_meta.json"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not decode_only:
        np.savez_compressed(
            out_data / "sim_fitted_latent_mu_s_q.npz",
            mu_s_q=mu_s_q.astype(np.float32),
            mu_bar=mu_bar.astype(np.float32),
        )

    pkl_path = out_data / args.pickle_name
    with pkl_path.open("wb") as f:
        pickle.dump(mem, f)
    print(f"Wrote {pkl_path}")
    print(f"Wrote {meta_path}")
    if not decode_only:
        print(f"Wrote {out_data / 'sim_fitted_latent_mu_s_q.npz'}")

    _ensure_tuna_module_on_path()
    from tuna_fid_single_job import save_memory_3d_plots, tau_ns_from_indices

    data_path_ckpt = Path(str(ckpt.get("data_path", "")))
    data_path = _resolve_source_pickle_path(data_path_ckpt, run_root=run_root)
    tau_arr = (
        _load_taus_from_source_pickle(data_path, expected_len=seq_len)
        if data_path is not None
        else None
    )
    if tau_arr is None:
        taus = tau_ns_from_indices(1, seq_len + 1, dt=100.0)
        tau_arr = np.asarray(taus, dtype=np.float64)
        print(
            "Warning: could not read taus from source pickle metadata "
            f"(ckpt data_path={data_path_ckpt}); falling back to dt=100 ns.",
            file=sys.stderr,
        )

    p_tilted, p_gif_3d, p_gif_2d, p_gif_2d_per_shot, p_cc, p_exc, p_tr, p_npz = save_memory_3d_plots(
        mem,
        num_qubits,
        seq_len,
        out_fig,
        max_shots=min(args.max_shots, len(mem)),
        prefix=args.prefix,
        tau_ns=tau_arr,
        gif_2d_marginal_history=10,
        gif_2d_rep_bin=args.gif_2d_rep_bin,
        reset_qubits=False,
        include_3d=args.include_3d,
    )
    if p_npz is not None and Path(p_npz).is_file():
        target_npz = out_data / Path(p_npz).name
        if target_npz != Path(p_npz):
            Path(p_npz).replace(target_npz)
            p_npz = target_npz
    if p_tilted is not None:
        print("Wrote:", p_tilted)
    if p_gif_3d is not None:
        print("Wrote:", p_gif_3d)
    print("Wrote:", p_gif_2d)
    if p_gif_2d_per_shot is not None:
        print("Wrote (per-shot 2D):", p_gif_2d_per_shot)
    if p_cc is not None:
        print("Wrote:", p_cc)
    if p_exc is not None:
        print("Wrote:", p_exc)
    if p_tr is not None:
        print("Wrote:", p_tr)
    if p_npz is not None:
        print("Wrote:", p_npz)


if __name__ == "__main__":
    main()
