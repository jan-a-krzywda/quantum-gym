"""Quantum Inspire backend package for quantum-gym.

This package provides a runner (cloud submission template), an adapter that can
fall back to a local simulator, and basic shadow-tomography helpers.

High-level API:
- run_on_quantum_inspire(qc, shots=1024, backend_name=None)  # runner template
- run_simulator(qc, shots=1024)                              # adapter fallback
- estimate_pauli_expectations_from_counts(counts, pauli_list)
"""

from .runner import run_on_quantum_inspire, has_qi_sdk  # noqa: F401
from .adapter import run_simulator, qc_to_qasm          # noqa: F401
from .shadow_tomography import estimate_pauli_expectations_from_counts  # noqa: F401

__all__ = [
    "run_on_quantum_inspire",
    "has_qi_sdk",
    "run_simulator",
    "qc_to_qasm",
    "estimate_pauli_expectations_from_counts",
]
