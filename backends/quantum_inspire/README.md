Quantum Inspire backend (scaffold)
=================================

This folder contains a minimal scaffold to integrate Quantum Inspire as a backend
for shadow-tomography and expectation estimation.

Files
- runner.py: template runner that prefers the `quantuminspire` SDK (if installed).
- adapter.py: local simulator fallback and QASM helper.
- shadow_tomography.py: minimal estimator to compute Pauli-Z expectations from counts.
- tests/: unit tests (pytest) that run offline without QI credentials.

Quick start
-----------
1. Install dev deps (qiskit, pytest) if you want to run the simulator tests.
2. Run tests: `pytest backends/quantum_inspire/tests`.

Notes
-----
- The runner currently raises an error if the QI SDK is not installed. Implement the
  REST fallback in `runner.py` if you prefer not to depend on the SDK.
- The shadow tomography implementation here is minimal (Z-basis only). For production
  use implement randomized classical shadows and inversion for general Pauli estimators.
