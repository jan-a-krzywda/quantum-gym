"""
Phase 2: BFS transition-graph dataset generator (shadow fingerprints).

Explores quantum state space reachable from |00...0⟩ by BFS over the action set,
deduplicates statevectors (global-phase invariant), shadow-fingerprints each unique
state, and stores the full transition table.

Dataset format (saved as .npz):
    fingerprints  : (N, n_samples, 36) float32  — shadow features per unique state
    transitions   : (N, A) int32  — transitions[i, j] = next state index (-1 = frontier)
    n_qubits      : int
    n_shots_per_sample : int      — random Pauli shots per shadow sample
    n_samples     : int           — independent shadow samples per state
    n_features    : int           — 36 (9 one-local + 27 two-local Pauli expectations)
    action_names  : (A,) str
    depths        : (N,) int32    — BFS depth each state was discovered at

For RL:
    z[i] = encoder(fingerprints[i])          # latent of state i
    z_next = z[transitions[i, j]]            # latent after action j from state i
    reward = -||z_next - z_target||          # distance to target

Usage:
    python generate_dataset.py --max-states 5000 --max-depth 8
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Load multiqubit_fingerprint from sibling file (avoids package install)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def _load_mf():
    spec = importlib.util.spec_from_file_location(
        "multiqubit_fingerprint", _HERE / "multiqubit_fingerprint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mf = _load_mf()


def _load_sf():
    spec = importlib.util.spec_from_file_location(
        "shadow_fingerprint", _HERE / "shadow_fingerprint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sf = _load_sf()


# ---------------------------------------------------------------------------
# Statevector deduplication
# ---------------------------------------------------------------------------

_HASH_DECIMALS = 6  # round to 1e-6 before hashing — handles floating point drift


def _sv_hash(sv_data: np.ndarray) -> bytes:
    """
    Global-phase-invariant hash of a statevector.

    Fixes global phase by rotating so the first nonzero element is real positive,
    then rounds to _HASH_DECIMALS and hashes the byte representation.
    """
    arr = np.asarray(sv_data, dtype=np.complex128)
    # Find first nonzero element to fix global phase
    nz = np.flatnonzero(np.abs(arr) > 1e-9)
    if len(nz):
        phase = arr[nz[0]] / np.abs(arr[nz[0]])
        arr = arr / phase
    # Round real and imag separately, pack as float32 for compact hashing
    arr_r = np.round(arr.real, _HASH_DECIMALS).astype(np.float32)
    arr_i = np.round(arr.imag, _HASH_DECIMALS).astype(np.float32)
    return arr_r.tobytes() + arr_i.tobytes()


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs_state_graph(
    initial_sv,
    action_names: Sequence[str],
    *,
    max_depth: int = 8,
    max_states: int = 5000,
    verbose: bool = True,
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    BFS from initial_sv over action_names.

    Returns
    -------
    statevectors : list of Statevector objects, length N
    transitions  : (N, A) int32 — transitions[i, j] = index of a_j(state_i)
    depths       : (N,) int32
    """
    A = len(action_names)
    hash_to_idx = {}
    statevectors = []
    depths_list = []

    def _register(sv, depth: int) -> int:
        h = _sv_hash(sv.data)
        if h not in hash_to_idx:
            idx = len(statevectors)
            hash_to_idx[h] = idx
            statevectors.append(sv)
            depths_list.append(depth)
        return hash_to_idx[h]

    root_idx = _register(initial_sv, 0)
    # Queue: (sv, depth, state_idx)
    queue = deque([(initial_sv, 0, root_idx)])

    # transitions[i, j] = -1 means not yet computed
    # We grow this dynamically; preallocate with max_states rows
    _cap = max_states
    transitions = np.full((_cap, A), -1, dtype=np.int32)

    t0 = time.time()
    processed = 0

    while queue:
        sv, depth, i = queue.popleft()
        processed += 1

        if verbose and processed % 500 == 0:
            elapsed = time.time() - t0
            print(
                f"  BFS: processed={processed}, states={len(statevectors)}, "
                f"queue={len(queue)}, depth={depth}, elapsed={elapsed:.1f}s"
            )

        for j, action in enumerate(action_names):
            sv_next = mf.apply_action(sv, action)
            next_idx = _register(sv_next, depth + 1)

            if i < _cap:
                transitions[i, j] = next_idx

            # Expand next state only if within depth limit and not yet queued
            if depth + 1 < max_depth and len(statevectors) < max_states:
                # Only enqueue if newly discovered (depths_list just grew)
                if depths_list[next_idx] == depth + 1 and next_idx == len(statevectors) - 1:
                    queue.append((sv_next, depth + 1, next_idx))

        if len(statevectors) >= max_states:
            if verbose:
                print(f"  BFS: hit max_states={max_states}, stopping expansion.")
            break

    N = len(statevectors)
    if verbose:
        print(f"BFS done: {N} unique states, {processed} nodes processed, "
              f"{time.time() - t0:.1f}s")

    return statevectors, transitions[:N], np.array(depths_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Fingerprint all states
# ---------------------------------------------------------------------------


def fingerprint_all_states(
    statevectors: List,
    n_qubits: int,
    n_shots_per_sample: int,
    n_samples: int,
    rng: np.random.Generator,
    verbose: bool = True,
) -> np.ndarray:
    """
    Shadow-fingerprint every statevector.

    Returns
    -------
    fingerprints : (N, n_samples, 36) float32
    """
    N = len(statevectors)
    fingerprints = np.zeros((N, n_samples, sf.N_FEATURES), dtype=np.float32)

    t0 = time.time()
    for i, sv in enumerate(statevectors):
        fingerprints[i] = sf.shadow_fingerprint_batch(
            sv, n_qubits, n_shots_per_sample=n_shots_per_sample,
            n_samples=n_samples, rng=rng,
        )
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  Fingerprint {i+1}/{N}  ({rate:.1f} states/s, ETA {eta:.0f}s)")

    if verbose:
        print(f"Fingerprinting done: {N} states in {time.time()-t0:.1f}s")
    return fingerprints


# ---------------------------------------------------------------------------
# Compute entanglement entropy
# ---------------------------------------------------------------------------


def compute_entanglement_entropy(sv, subsystem: List[int]) -> float:
    """
    Compute the entanglement entropy of a quantum state.

    Parameters
    ----------
    sv : Statevector
        The quantum state.
    subsystem : list of int
        The qubits that form the subsystem (for bipartite entanglement).

    Returns
    -------
    float
        The entanglement entropy of the state.
    """
    from qiskit.quantum_info import partial_trace, entropy

    # Compute the reduced density matrix for the subsystem
    print(sv)
    reduced_density_matrix = partial_trace(sv, qargs=subsystem)

    # Compute the eigenvalues of the reduced density matrix
    eigenvalues = np.linalg.eigvalsh(reduced_density_matrix)

    # Compute the entanglement entropy: -Tr(rho_subsystem log(rho_subsystem))
    entropy_value = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

    return entropy_value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BFS transition-graph fingerprint dataset")
    p.add_argument("--n-qubits", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=10,
                   help="BFS depth limit (default 10)")
    p.add_argument("--max-states", type=int, default=1500,
                   help="Hard cap on unique states (default 15000)")
    p.add_argument("--n-shots-per-sample", type=int, default=128,
                   help="Random Pauli shots per shadow sample (default 256)")
    p.add_argument("--n-samples", type=int, default=1,
                   help="Independent shadow samples per state (default 16)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .npz path (default: RL-world-model/data/bfs_dataset.npz)")
    p.add_argument("--actions", nargs="+", default=None,
                   help="Override action set (default: all 9 for 3-qubit system)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    action_names = args.actions if args.actions is not None else mf.ACTION_NAMES_3Q
    rng = np.random.default_rng(args.seed)

    out_path = args.out
    if out_path is None:
        data_dir = _HERE / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / "bfs_dataset.npz"

 
    # --- BFS ---
    initial_sv = mf.prepare_zero_state(args.n_qubits)
    if verbose:
        print("Starting BFS from |00...0>...")
    statevectors, transitions, depths = bfs_state_graph(
        initial_sv,
        action_names,
        max_depth=args.max_depth,
        max_states=args.max_states,
        verbose=verbose,
    )
    N = len(statevectors)

    # --- Shadow fingerprint ---
    if verbose:
        print(f"Fingerprinting {N} states "
              f"(n_shots_per_sample={args.n_shots_per_sample}, n_samples={args.n_samples})...")
    fingerprints = fingerprint_all_states(
        statevectors,
        args.n_qubits,
        args.n_shots_per_sample,
        args.n_samples,
        rng=rng,
        verbose=verbose,
    )

    # --- Compute entanglement entropy ---
    if verbose:
        print(f"Computing entanglement entropy for {N} states...")
    entanglement_measures = [
        compute_entanglement_entropy(sv, subsystem=list(range(args.n_qubits // 2)))
        for sv in statevectors
    ]

    # --- Save ---
    np.savez_compressed(
        out_path,
        fingerprints=fingerprints,
        transitions=transitions,
        depths=depths,
        action_names=np.array(action_names),
        n_qubits=np.array(args.n_qubits),
        n_shots_per_sample=np.array(args.n_shots_per_sample),
        n_samples=np.array(args.n_samples),
        n_features=np.array(sf.N_FEATURES),
        max_depth=np.array(args.max_depth),
        entanglement_measures=np.array(entanglement_measures, dtype=np.float32),
    )

    size_mb = out_path.stat().st_size / 1e6
    if verbose:
        print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
        print(f"  fingerprints : {fingerprints.shape}  float32  "
              f"[{fingerprints.min():.3f}, {fingerprints.max():.3f}]")
        print(f"  transitions  : {transitions.shape}  int32")
        print(f"  depths       : depth 0..{depths.max()}  (histogram: "
              + ", ".join(f"d{d}={int((depths==d).sum())}"
                          for d in range(depths.max()+1)) + ")")
        print(f"  actions      : {action_names}")


if __name__ == "__main__":
    main()
