#!/usr/bin/env python3
"""
Sample the trained Conv-VAE to produce synthetic FID-style memory bitstrings, then run the
same ``save_memory_3d_plots`` pipeline as ``zebra_plot.ipynb`` (pickle + 2D repetition GIF).

Training uses differential τ readout (``reset_qubits=False`` in ``build_stack_like_single_job``).
Samples are Bernoulli draws from the decoder; we integrate along τ to recover raw Z-style
bitstrings so ``save_memory_3d_plots(..., reset_qubits=False)`` applies the same differential
step as for hardware data.

**Default — uniform zebra (good decay-like structure, same latent law for every qubit):**
``z ~ N(0,I)`` independently for each (shot, qubit) row — no qubit conditioning. This matches
the original behaviour and tends to reproduce smooth FID-like stripes with **statistically
similar** rows across qubits.

**Optional — per-qubit latent shift:** ``--latent-mode qubit-mean-shift`` plus
``--dynamics-json`` uses ``z = ε + α μ̄_q`` (see ``--qubit-latent-scale``).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

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
    """
    One shot ``(num_qubits, n_tau)`` → Qiskit-style memory string (same layout as
    ``bitstring_to_shot_matrix`` in ``tuna_fid_single_job``).
    """
    n_q, n_tau = shot_stack_jk.shape
    n_c = n_q * n_tau
    chars = ["0"] * n_c
    for k in range(n_tau):
        for j in range(n_q):
            cbit = k * n_q + j
            chars[n_c - 1 - cbit] = "1" if shot_stack_jk[j, k] else "0"
    return "".join(chars)


def stack_to_memory_list(stack_snjk: np.ndarray) -> list[str]:
    """``(n_shots, n_qubits, n_tau)`` → list of bitstrings."""
    return [stack_shot_to_bitstring(stack_snjk[s]) for s in range(stack_snjk.shape[0])]


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


def load_mu_bar_from_dynamics(path: Path) -> np.ndarray:
    """``latent_dynamics.json`` → ``(n_qubits, latent_dim)`` array of per-qubit temporal means."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    n_q = int(data["n_qubits"])
    ld = int(data["latent_dim"])
    out = np.zeros((n_q, ld), dtype=np.float32)
    for q in range(n_q):
        m = data["mu_bar_per_qubit"][str(q)]
        out[q, 0] = float(m["mu_1"])
        if ld > 1:
            out[q, 1] = float(m["mu_2"])
    return out


def load_taus_from_source_pickle(path: Path, *, expected_len: int) -> np.ndarray | None:
    """
    Read tau metadata from acquisition pickle payload if available.

    Returns ``None`` when metadata is missing/incompatible.
    """
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
    p = argparse.ArgumentParser(description="VAE synthetic memory + zebra-style GIFs")
    p.add_argument(
        "--source-file",
        type=Path,
        default=None,
        help="Optional source pickle path for provenance in pipeline calls",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=_default_ckpt(),
        help="``vae_checkpoint.pt`` from train_vae.py",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="default: alongside ckpt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample-mode",
        choices=("bernoulli", "threshold"),
        default="bernoulli",
        help="draw bits from Bernoulli(p) or threshold at 0.5",
    )
    p.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="override shot count (default: from checkpoint ``n_shots``)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="vae_zebra_memory_3d",
        help="output filename prefix (cf. zebra_plot ``zebra_memory_3d``)",
    )
    p.add_argument("--max-shots", type=int, default=5000, help="cap for ``save_memory_3d_plots``")
    p.add_argument("--gif-2d-rep-bin", type=int, default=100, help="match zebra_plot notebook")
    p.add_argument(
        "--include-3d",
        action="store_true",
        help="also write tilted PNG + slow 3D cumulative GIF",
    )
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument(
        "--pickle-name",
        type=str,
        default="vae_synthetic_fid_memory.pkl",
        help="written under --out-dir",
    )
    p.add_argument(
        "--latent-mode",
        choices=("iid", "qubit-mean-shift"),
        default="iid",
        help="iid (default) = z ~ N(0,I) per row, uniform across qubits — zebra-like decay. "
        "qubit-mean-shift = add α·μ̄_q from --dynamics-json (requires that file).",
    )
    p.add_argument(
        "--dynamics-json",
        type=Path,
        default=None,
        help="Used only with --latent-mode qubit-mean-shift (latent_dynamics.json).",
    )
    p.add_argument(
        "--qubit-latent-scale",
        type=float,
        default=1.0,
        help="Only for qubit-mean-shift: z = ε + scale·μ̄_q.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    run_root = Path(args.out_dir) if args.out_dir is not None else _run_root_from_ckpt(args.ckpt.resolve())
    out_fig = run_root / "figures"
    out_data = run_root / "data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    seq_len = int(ckpt["seq_len"])
    latent_dim = int(ckpt["latent_dim"])
    num_qubits = int(ckpt["num_qubits"])
    n_shots_ckpt = int(ckpt["n_shots"])
    n_shots = args.n_shots if args.n_shots is not None else n_shots_ckpt
    n_rows = n_shots * num_qubits

    device = _pick_device(args.device)
    model = QubitConvVAE(seq_len=seq_len, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _ensure_tuna_module_on_path()
    from tuna_fid_single_job import save_memory_3d_plots, tau_ns_from_indices

    data_path_ckpt = Path(str(ckpt.get("data_path", "")))
    data_path = _resolve_source_pickle_path(data_path_ckpt, run_root=run_root)
    tau_arr = (
        load_taus_from_source_pickle(data_path, expected_len=seq_len)
        if data_path is not None
        else None
    )
    if tau_arr is None:
        # Backward-compatible fallback for old checkpoints/pickles without tau metadata.
        taus = tau_ns_from_indices(1, seq_len + 1, dt=100.0)
        tau_arr = np.asarray(taus, dtype=np.float64)
        print(
            "Warning: could not read taus from source pickle metadata "
            f"(ckpt data_path={data_path_ckpt}); falling back to dt=100 ns.",
            file=sys.stderr,
        )
    if tau_arr.size != seq_len:
        raise ValueError(f"tau list length {tau_arr.size} != seq_len={seq_len}")

    if args.latent_mode == "qubit-mean-shift":
        if args.dynamics_json is None:
            raise ValueError(
                "--latent-mode qubit-mean-shift requires --dynamics-json "
                "(latent_dynamics.json from analyze_latent_dynamics.py)."
            )
        dyn_path = Path(args.dynamics_json)
        if not dyn_path.is_file():
            raise FileNotFoundError(dyn_path)
    elif args.dynamics_json is not None:
        print(
            "Note: --dynamics-json is ignored unless --latent-mode qubit-mean-shift "
            "(default is iid / uniform qubits).",
            file=sys.stderr,
        )

    with torch.no_grad():
        z = torch.randn(n_rows, latent_dim, device=device)
        if (
            args.latent_mode == "qubit-mean-shift"
            and float(args.qubit_latent_scale) != 0.0
        ):
            mu_b = load_mu_bar_from_dynamics(Path(args.dynamics_json))
            if mu_b.shape != (num_qubits, latent_dim):
                raise ValueError(
                    f"mu_bar has shape {mu_b.shape}, "
                    f"expected ({num_qubits}, {latent_dim})"
                )
            mb = torch.from_numpy(mu_b).to(device=device, dtype=z.dtype)
            q_ix = torch.arange(n_rows, device=device) % num_qubits
            z = z + float(args.qubit_latent_scale) * mb[q_ix]
        h = model.decoder_input(z)
        recon = model.decoder(h)

    if args.sample_mode == "bernoulli":
        bits = torch.bernoulli(recon.clamp(0.0, 1.0))
    else:
        bits = (recon >= 0.5).float()
    x = bits.squeeze(1).cpu().numpy().astype(np.uint8)

    # Same row order as ``stack_to_vae_tensors`` / C-order reshape of (n_shots, n_q, n_tau).
    diff_stack = x.reshape(n_shots, num_qubits, seq_len)
    raw_stack = integrate_differential_along_tau(diff_stack, implicit_prior=0)
    mem = stack_to_memory_list(raw_stack)

    pkl_path = out_data / args.pickle_name
    with pkl_path.open("wb") as f:
        pickle.dump(mem, f)
    print(f"Wrote {pkl_path} ({len(mem)} shots × {num_qubits} qubits × {seq_len} τ)")

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
