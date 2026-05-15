"""Small shadow-tomography helpers.

Currently implements estimation of Pauli-Z string expectations from computational-basis
measurement counts. This is a minimal first step; full classical shadows require
randomized Pauli measurements and inversion.
"""
from typing import Dict, List, Tuple
import numpy as np


def expectation_from_counts(counts: Dict[str, int], pauli: str) -> Tuple[float, float]:
    """Estimate expectation value and std for a Pauli-Z string `pauli` from counts.

    counts: dict mapping bitstring -> counts (bitstrings MSB..LSB as in Qiskit)
    pauli: string like 'ZIZ' or 'ZZZ' with length == n_qubits

    Returns (expectation, std_estimate)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0

    n_qubits = len(pauli)
    vals = []
    for bits, c in counts.items():
        # align bitstring length
        if len(bits) < n_qubits:
            bits = bits.zfill(n_qubits)
        # compute eigenvalue for this bitstring
        eig = 1
        for i, p in enumerate(pauli):
            if p == 'I':
                continue
            # Qiskit bitstring is q_{n-1}...q_0; we assume pauli[0] corresponds to qubit 0 (left)
            # match by aligning from left
            b = bits[i]
            if p == 'Z':
                if b == '1':
                    eig *= -1
        vals.append(eig * (c / total))

    expect = float(sum(vals))
    # Bernoulli variance approximation for Pauli measurement
    var = (1 - expect**2) / max(1, total)
    std = float(np.sqrt(var))
    return expect, std


def estimate_pauli_expectations_from_counts(counts: Dict[str, int], pauli_list: List[str]) -> Dict[str, Tuple[float, float]]:
    """Estimate multiple Pauli-Z string expectations from counts.

    Returns mapping pauli -> (expectation, std)
    """
    out = {}
    for p in pauli_list:
        out[p] = expectation_from_counts(counts, p)
    return out
