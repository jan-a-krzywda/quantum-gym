"""
Shadow-tomography fingerprint for multiqubit quantum states.

Uses random Pauli-basis measurements to estimate 1-local and 2-local Pauli
expectation values. Output: 36-dim float32 vector per state (n_qubits=3).

Feature layout:
  [0:9]   1-local:  ⟨X₀⟩,⟨Y₀⟩,⟨Z₀⟩, ⟨X₁⟩,⟨Y₁⟩,⟨Z₁⟩, ⟨X₂⟩,⟨Y₂⟩,⟨Z₂⟩
  [9:18]  2-local (pair 0,1): ⟨XX⟩,⟨XY⟩,⟨XZ⟩,⟨YX⟩,⟨YY⟩,⟨YZ⟩,⟨ZX⟩,⟨ZY⟩,⟨ZZ⟩
  [18:27] 2-local (pair 0,2): same order
  [27:36] 2-local (pair 1,2): same order

Inversion factors (Huang et al. 2020, random Pauli measurements):
  1-local: feature = (3/T) * Σ_{t: basis matches} b_q
  2-local: feature = (9/T) * Σ_{t: both bases match} b_q1 * b_q2
where T = n_shots total. This correctly gives features ∈ [-1, 1].

Note: dividing by n_matching (not T) then multiplying by 3/9 is WRONG —
that cancels the inversion and returns 3x/9x the true expectation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load_mf():
    spec = importlib.util.spec_from_file_location(
        "multiqubit_fingerprint", _HERE / "multiqubit_fingerprint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mf = _load_mf()
apply_action = _mf.apply_action
prepare_zero_state = _mf.prepare_zero_state
prepare_ghz_state = _mf.prepare_ghz_state
ACTION_NAMES_3Q = _mf.ACTION_NAMES_3Q

from qiskit import QuantumCircuit  # noqa: E402

_BASES = ["X", "Y", "Z"]
_BASIS_IDX = {"X": 0, "Y": 1, "Z": 2}

N_FEATURES_1Q = 9    # 3 qubits × 3 Paulis
N_FEATURES_2Q = 27   # 3 pairs × 9 Pauli combos
N_FEATURES = N_FEATURES_1Q + N_FEATURES_2Q  # 36

# Canonical qubit pairs (for n_qubits=3)
_PAIRS_3Q = [(0, 1), (0, 2), (1, 2)]


def _apply_basis_rotation(qc: QuantumCircuit, basis_str: str) -> None:
    """Rotate Pauli eigenbasis → Z-basis for measurement. Qiskit little-endian."""
    for q, b in enumerate(basis_str):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        # Z: no rotation needed


def shadow_fingerprint_from_statevector(
    sv,
    n_qubits: int = 3,
    n_shots: int = 256,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Single shadow fingerprint: (36,) float32, all features in [−1, 1].

    Groups shots by unique basis string to minimise sv.evolve() calls.
    Qiskit little-endian: qubit q occupies bit q of the sampled integer.
    """
    if rng is None:
        rng = np.random.default_rng()

    shot_bases = rng.choice(_BASES, size=(n_shots, n_qubits))          # (T, Q) str
    shot_bases_idx = np.vectorize(_BASIS_IDX.get)(shot_bases)           # (T, Q) int
    outcomes = np.zeros((n_shots, n_qubits), dtype=np.float32)          # ±1

    unique_rows, inverse = np.unique(shot_bases, axis=0, return_inverse=True)
    for bidx, basis_arr in enumerate(unique_rows):
        n_this = int((inverse == bidx).sum())
        if n_this == 0:
            continue

        rot_qc = QuantumCircuit(n_qubits)
        _apply_basis_rotation(rot_qc, "".join(basis_arr))
        rotated_sv = sv.evolve(rot_qc)

        probs = rotated_sv.probabilities()
        samples = rng.choice(len(probs), size=n_this, p=probs)

        mask = inverse == bidx
        for q in range(n_qubits):
            bits = (samples >> q) & 1          # little-endian: qubit q = bit q
            outcomes[mask, q] = 1.0 - 2.0 * bits   # 0→+1, 1→−1

    # basis one-hot: (T, Q, 3) — used for indicator sums
    basis_oh = np.eye(3, dtype=np.float32)[shot_bases_idx]

    fingerprint = np.empty(N_FEATURES, dtype=np.float32)
    idx = 0

    # 1-local: (3/T) * Σ_{t: basis[t,q]=P} b[t,q]  →  ⟨P_q⟩
    one_local = (outcomes[:, :, None] * basis_oh).sum(axis=0)   # (Q, 3)
    for q in range(n_qubits):
        for p in range(3):
            fingerprint[idx] = 3.0 * one_local[q, p] / n_shots
            idx += 1

    # 2-local: (9/T) * Σ_{t: B[t,q1]=P1 & B[t,q2]=P2} b[t,q1]*b[t,q2]  →  ⟨P_q1 P_q2⟩
    pairs = _PAIRS_3Q if n_qubits == 3 else [
        (q1, q2) for q1 in range(n_qubits) for q2 in range(q1 + 1, n_qubits)
    ]
    for q1, q2 in pairs:
        prod = outcomes[:, q1] * outcomes[:, q2]   # (T,)
        for p1 in range(3):
            for p2 in range(3):
                joint = basis_oh[:, q1, p1] * basis_oh[:, q2, p2]  # (T,) 0/1
                fingerprint[idx] = 9.0 * float((prod * joint).sum()) / n_shots
                idx += 1

    return fingerprint


def shadow_fingerprint_batch(
    sv,
    n_qubits: int = 3,
    n_shots_per_sample: int = 256,
    n_samples: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Multiple independent shadow fingerprints: (n_samples, 36) float32.
    Gives VAE multiple noisy views of the same state for training diversity.
    """
    if rng is None:
        rng = np.random.default_rng()
    return np.stack([
        shadow_fingerprint_from_statevector(sv, n_qubits, n_shots_per_sample, rng)
        for _ in range(n_samples)
    ])


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def test_states(n_shots: int = 4000) -> None:
    """
    Print key shadow features for |000⟩, |+++⟩, GHZ.

    Expected:
      |000⟩: ⟨Z_q⟩≈+1, ⟨ZZ⟩≈+1, ⟨XX⟩≈0
      |+++⟩: ⟨X_q⟩≈+1, ⟨ZZ⟩≈0, ⟨XX⟩≈+1
      GHZ:   ⟨Z_q⟩≈0,  ⟨ZZ⟩≈+1, ⟨XX⟩≈... (mixed sign depending on 3-local)
    """
    rng = np.random.default_rng(0)
    svs = {
        "|000⟩": prepare_zero_state(3),
        "|+++⟩": _prepare_plus_state(3),
        "GHZ  ": prepare_ghz_state(3),
    }

    pnames = ["X", "Y", "Z"]
    pairs = _PAIRS_3Q

    for label, sv in svs.items():
        fp = shadow_fingerprint_from_statevector(sv, n_shots=n_shots, rng=rng)
        print(f"\n{label}")
        print("  1-local:")
        for q in range(3):
            vals = "  ".join(f"⟨{p}{q}⟩={fp[q*3+pi]:+.3f}" for pi, p in enumerate(pnames))
            print(f"    {vals}")
        print("  2-local ZZ and XX:")
        for pair_i, (q1, q2) in enumerate(pairs):
            base = N_FEATURES_1Q + pair_i * 9
            # ZZ = p1=Z(2), p2=Z(2) → offset 2*3+2=8
            zz = fp[base + 8]
            xx = fp[base + 0]
            print(f"    ⟨Z{q1}Z{q2}⟩={zz:+.3f}  ⟨X{q1}X{q2}⟩={xx:+.3f}")


def _prepare_plus_state(n_qubits: int):
    qc = QuantumCircuit(n_qubits)
    qc.h(list(range(n_qubits)))
    from qiskit.quantum_info import Statevector
    return Statevector(qc)


if __name__ == "__main__":
    test_states()
