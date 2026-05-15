"""
Phase 2: Dataset generator from beam-search trajectories.

Loads the beam-search trajectories (Phase 1), executes the intermediate circuits,
measures shadow fingerprints, and constructs the transition dataset:
    (x_{k,n}, a_{k,n}, x_{k,n+1})

where:
    x_{k,n}   = 63-dim shadow fingerprint of trajectory k after round n
    a_{k,n}   = 6-dim angle delta applied at round n+1  (values in {-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT})
    x_{k,n+1} = 63-dim shadow fingerprint after round n+1

Dataset format (saved as .npz):
    fingerprints   : (M, 63)  float32  — shadow features at each step
    next_fps       : (M, 63)  float32  — shadow features after action
    actions        : (M, 6)   float32  — angle deltas (raw float, multiples of ANGLE_INCREMENT)
    action_indices : (M, 6)   int8     — discretised actions: {-1, 0, +1}
    traj_ids       : (M,)     int32    — which trajectory each sample came from
    round_ids      : (M,)     int32    — which round (0-indexed) each sample came from
    distances      : (M,)     float32  — trace distance of next state with GHZ
    n_features     : int      — 63
    n_params       : int      — 6
    angle_increment: float

Usage:
    python generate_dataset.py --trajectories data/beam_trajectories.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load_module("beam_search", _HERE / "beam_search.py")
sf = _load_module("shadow_fingerprint", _HERE / "shadow_fingerprint.py")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_transition_dataset(
    trajectories: list,
    n_shots_per_sample: int = 1024,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict:
    """
    Build (x_kn, a_kn, x_kn+1) transition dataset from beam-search trajectories.

    Each trajectory has n_rounds rounds. We collect transitions for rounds 0..n_rounds-2.
    At each step:
      - fingerprint the state after round n     → x_kn
      - fingerprint the state after round n+1   → x_kn+1
      - action = angle delta for round n+1      → a_kn

    Parameters
    ----------
    trajectories : list of dicts from beam_search.load_trajectories
    n_shots_per_sample : Pauli shots per shadow sample
    n_samples : independent shadow samples per state (averaged for VAE training)
    rng : random generator
    verbose : print progress
    """
    if rng is None:
        rng = np.random.default_rng()

    fps_list = []
    next_fps_list = []
    actions_list = []
    action_idx_list = []
    traj_ids = []
    round_ids = []
    distances_list = []

    angle_increment = float(bs.ANGLE_INCREMENT)
    n_qubits = bs.N_QUBITS

    t0 = time.time()
    total_steps = sum(max(0, len(t["distance_history"]) - 1) for t in trajectories)
    done_steps = 0

    for k, traj in enumerate(trajectories):
        angle_schedule = traj["angle_schedule"]   # (n_rounds, 6)
        actions = traj["actions"]                  # (n_rounds, 6)
        distance_history = traj["distance_history"]
        n_rounds = angle_schedule.shape[0]

        for n in range(n_rounds - 1):
            # State after round n
            sched_n = angle_schedule[:n + 1]
            qc_n = bs.build_ansatz_circuit(sched_n)
            from qiskit.quantum_info import Statevector
            sv_n = Statevector(qc_n)

            # State after round n+1
            sched_n1 = angle_schedule[:n + 2]
            qc_n1 = bs.build_ansatz_circuit(sched_n1)
            sv_n1 = Statevector(qc_n1)

            # Shadow fingerprints: average over n_samples
            fp_n = sf.shadow_fingerprint_batch(
                sv_n, n_qubits, n_shots_per_sample=n_shots_per_sample,
                n_samples=n_samples, rng=rng,
            ).mean(axis=0)   # (63,)

            fp_n1 = sf.shadow_fingerprint_batch(
                sv_n1, n_qubits, n_shots_per_sample=n_shots_per_sample,
                n_samples=n_samples, rng=rng,
            ).mean(axis=0)   # (63,)

            # Action: delta at round n+1
            action_delta = actions[n + 1]   # (6,) floats in {-π/18, 0, +π/18}
            action_idx = np.round(action_delta / angle_increment).astype(np.int8)  # {-1, 0, +1}

            fps_list.append(fp_n)
            next_fps_list.append(fp_n1)
            actions_list.append(action_delta.astype(np.float32))
            action_idx_list.append(action_idx)
            traj_ids.append(k)
            round_ids.append(n)
            distances_list.append(float(distance_history[n + 1]))

            done_steps += 1
            if verbose and done_steps % 20 == 0:
                rate = done_steps / (time.time() - t0 + 1e-9)
                print(f"  Steps {done_steps}/{total_steps}  ({rate:.1f} steps/s)")

    if verbose:
        print(f"Dataset built: {len(fps_list)} transitions in {time.time()-t0:.1f}s")

    return {
        "fingerprints":    np.array(fps_list, dtype=np.float32),
        "next_fps":        np.array(next_fps_list, dtype=np.float32),
        "actions":         np.array(actions_list, dtype=np.float32),
        "action_indices":  np.array(action_idx_list, dtype=np.int8),
        "traj_ids":        np.array(traj_ids, dtype=np.int32),
        "round_ids":       np.array(round_ids, dtype=np.int32),
        "distances":       np.array(distances_list, dtype=np.float32),
        "n_features":      np.array(sf.N_FEATURES),
        "n_params":        np.array(bs.N_PARAMS),
        "angle_increment": np.array(angle_increment),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: Build transition dataset from beam trajectories")
    p.add_argument("--trajectories", type=Path, default=None,
                   help=".npz from beam_search.py (default: data/beam_trajectories.npz)")
    p.add_argument("--n-shots-per-sample", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=4,
                   help="Independent shadow samples per state (averaged)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .npz path (default: data/transition_dataset.npz)")
    p.add_argument("--quiet", action="store_true")
    # Optionally run beam search on-the-fly
    p.add_argument("--beam-width", type=int, default=None,
                   help="If set, run beam search instead of loading from file")
    p.add_argument("--max-rounds", type=int, default=bs.MAX_ROUNDS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    rng = np.random.default_rng(args.seed)
    data_dir = _HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or generate beam trajectories ---
    if args.beam_width is not None:
        if verbose:
            print(f"Running beam search (beam_width={args.beam_width}, max_rounds={args.max_rounds})...")
        trajectories = bs.beam_search(
            beam_width=args.beam_width,
            max_rounds=args.max_rounds,
            verbose=verbose,
        )
        traj_path = data_dir / "beam_trajectories.npz"
        bs.save_trajectories(traj_path, trajectories)
        if verbose:
            print(f"Trajectories saved to {traj_path}")
    else:
        traj_path = args.trajectories or data_dir / "beam_trajectories.npz"
        if not traj_path.exists():
            raise FileNotFoundError(
                f"Trajectories not found at {traj_path}. "
                "Run beam_search.py first or pass --beam-width."
            )
        if verbose:
            print(f"Loading trajectories from {traj_path}...")
        trajectories = bs.load_trajectories(traj_path)

    if verbose:
        print(f"Loaded {len(trajectories)} trajectories")
        total_rounds = sum(len(t["distance_history"]) for t in trajectories)
        print(f"  Total rounds: {total_rounds}  "
              f"  Total transitions: {total_rounds - len(trajectories)}")

    # --- Build dataset ---
    if verbose:
        print(f"Building transition dataset "
              f"(n_shots_per_sample={args.n_shots_per_sample}, n_samples={args.n_samples})...")
    dataset = build_transition_dataset(
        trajectories,
        n_shots_per_sample=args.n_shots_per_sample,
        n_samples=args.n_samples,
        rng=rng,
        verbose=verbose,
    )

    # --- Save ---
    out_path = args.out or data_dir / "transition_dataset.npz"
    np.savez_compressed(out_path, **dataset)
    size_mb = out_path.stat().st_size / 1e6

    if verbose:
        M = len(dataset["fingerprints"])
        print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
        print(f"  transitions    : {M}")
        print(f"  fingerprints   : {dataset['fingerprints'].shape}  "
              f"[{dataset['fingerprints'].min():.3f}, {dataset['fingerprints'].max():.3f}]")
        print(f"  distance range : [{dataset['distances'].min():.4f}, {dataset['distances'].max():.4f}]")


if __name__ == "__main__":
    main()
