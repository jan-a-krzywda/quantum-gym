"""shadow-gym

Thin compatibility layer for the shadow tomography component. This module re-exports
the shadow estimation helpers implemented in `backends.quantum_inspire.shadow_tomography`.

This file exists to make `shadow-gym` a visible subproject in the monorepo while
keeping the canonical implementation inside `backends/quantum_inspire` for easier
upstreaming later.
"""
try:
    from backends.quantum_inspire.shadow_tomography import (
        expectation_from_counts,
        estimate_pauli_expectations_from_counts,
    )
except Exception:
    # Provide fallbacks if the implementation is not importable
    def expectation_from_counts(counts, pauli):
        raise ImportError("backends.quantum_inspire.shadow_tomography not available")

    def estimate_pauli_expectations_from_counts(counts, pauli_list):
        raise ImportError("backends.quantum_inspire.shadow_tomography not available")

__all__ = [
    "expectation_from_counts",
    "estimate_pauli_expectations_from_counts",
]
