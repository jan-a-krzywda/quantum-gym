"""Adapter helpers: local simulator fallback and circuit conversion utilities."""
from typing import Dict
import numpy as np
try:
    from qiskit import Aer, transpile
    from qiskit.quantum_info import Statevector
    has_qiskit = True
except Exception:
    has_qiskit = False


def run_simulator(qc, shots=1024) -> Dict[str, int]:
    """Run a Qiskit QuantumCircuit locally and return counts.

    If statevector simulator is present, use sampling via Aer; otherwise return
    a deterministic statevector-derived counts (for expectation testing).
    """
    if not has_qiskit:
        raise RuntimeError("Qiskit not available in this environment")

    try:
        backend = Aer.get_backend('aer_simulator')
        qc2 = qc.copy()
        qc2 = transpile(qc2, backend)
        job = backend.run(qc2, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts
    except Exception:
        # last-resort deterministic counts from the statevector (single sample)
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data)**2
        idx = int(np.argmax(probs))
        bitstring = format(idx, f'0{qc.num_qubits}b')
        return {bitstring: 1}


def qc_to_qasm(qc) -> str:
    """Return OpenQASM string for a Qiskit circuit (adapter helper)."""
    return qc.qasm()
