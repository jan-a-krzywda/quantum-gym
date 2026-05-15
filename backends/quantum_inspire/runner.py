"""Runner helpers for Quantum Inspire.

This module attempts to import the Quantum Inspire SDK. If it's not available
the runner exposes a clear exception and a REST fallback placeholder. Replace
the REST code with real requests if you prefer not to use the SDK.
"""
import os
import time
from typing import Tuple

has_qi_sdk = False
try:
    # preferred: quantum-inspire SDK
    from quantuminspire.api import QuantumInspireAPI
    from quantuminspire.qiskit import QIProvider
    has_qi_sdk = True
except Exception:
    QuantumInspireAPI = None
    QIProvider = None


class QIRunnerError(RuntimeError):
    pass


def run_on_quantum_inspire(qc, shots=1024, backend_name=None, api_key=None, poll_interval=2.0, timeout=600.0) -> dict:
    """Submit a Qiskit QuantumCircuit to Quantum Inspire and return counts.

    Returns a dictionary of measurement counts {bitstring: counts}.

    Parameters
    - qc: qiskit.QuantumCircuit
    - shots: number of measurement shots
    - backend_name: optional backend target name
    - api_key: optional API key (falls back to environment var)
    - poll_interval: seconds between job status checks
    - timeout: maximum seconds to wait for job completion
    """
    if api_key is None:
        api_key = os.getenv("QUANTUM_INSPIRE_API_KEY")
    if not api_key:
        raise QIRunnerError("No Quantum Inspire API key provided (set QUANTUM_INSPIRE_API_KEY)")

    if has_qi_sdk:
        # SDK path (preferred)
        provider = QIProvider(api_key)
        try:
            if backend_name is None:
                backend = provider.get_backend()  # default
            else:
                backend = provider.get_backend(backend_name)
        except Exception as e:
            raise QIRunnerError(f"Failed selecting QI backend: {e}")

        # Submit job
        try:
            job = backend.run(qc, shots=shots)
        except Exception as e:
            raise QIRunnerError(f"QI job submission failed: {e}")

        start = time.time()
        while True:
            status = job.status()
            if status.is_finished or status == 'finished':
                break
            if time.time() - start > timeout:
                raise QIRunnerError("Job timed out")
            time.sleep(poll_interval)

        try:
            result = job.result()
            counts = result.get_counts()
            return counts
        except Exception as e:
            raise QIRunnerError(f"Failed retrieving QI job result: {e}")

    else:
        # REST fallback placeholder: raise a clear error so devs implement it only if needed
        raise QIRunnerError("Quantum Inspire SDK not installed. Install 'quantuminspire' or implement REST fallback.")


def get_counts(qc, shots=1024, backend_name=None, api_key=None, prefer_sdk=True):
    """Convenience wrapper: try QI SDK then fall back to local simulator adapter.

    Returns counts mapping bitstring -> int. This makes it safe to call from
    notebooks during development while keeping the option to run on real QI later.
    """
    # Try SDK path first
    if prefer_sdk and has_qi_sdk:
        try:
            return run_on_quantum_inspire(qc, shots=shots, backend_name=backend_name, api_key=api_key)
        except QIRunnerError:
            # Fall through to simulator fallback
            pass

    # Simulator fallback (local) — import here to avoid hard dependency at module import
    try:
        from .adapter import run_simulator
    except Exception as e:
        raise QIRunnerError(f"Simulator adapter unavailable: {e}")

    return run_simulator(qc, shots=shots)
