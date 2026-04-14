"""Load Quantum Inspire FID memory pickles and build stacks like ``tuna_fid_single_job``."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np


def _ensure_inspire_on_path() -> Path:
    """Ensure the directory containing ``tuna_fid_single_job`` is importable."""
    ml_dir = Path(__file__).resolve().parent
    first_tests = ml_dir.parent.parent
    candidates = (
        first_tests / "quantum_code",  # current layout
        first_tests / "inspire",       # legacy layout
    )
    for module_dir in candidates:
        if (module_dir / "tuna_fid_single_job.py").is_file():
            s = str(module_dir)
            if s not in sys.path:
                sys.path.insert(0, s)
            return module_dir
    raise FileNotFoundError(
        "Could not locate tuna_fid_single_job.py in first_tests/quantum_code or first_tests/inspire."
    )


def load_memory_pickle(path: Union[str, Path]) -> List[str]:
    """Load a job pickle produced from ``job.result().get_memory()``."""
    path = Path(path)
    with path.open("rb") as f:
        obj: Any = pickle.load(f)
    if isinstance(obj, dict):
        for key in ("memory", "mem", "raw_memory"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"Expected list of bitstrings from {path}, got {type(obj)}")
    return list(obj)


@dataclass(frozen=True)
class FidStackResult:
    """Binary stack for one qubit persona per row (shape ``n_shots * n_qubits``, ``n_tau``)."""

    stack: np.ndarray
    """``(n_shots, n_qubits, n_tau)``, uint8 in ``{0, 1}`` after optional differential readout."""

    num_qubits: int
    n_tau: int
    n_shots: int
    differential: bool


def infer_num_qubits(mem: List[str], n_tau: int) -> int:
    _ensure_inspire_on_path()
    from tuna_fid_single_job import inferred_qubits_from_memory

    n = inferred_qubits_from_memory(mem, n_tau)
    if n is None:
        raise ValueError(
            f"Could not infer num_qubits from memory with n_tau={n_tau}. "
            "Set num_qubits (and check n_tau) explicitly."
        )
    return int(n)


def build_stack_like_single_job(
    mem: List[str],
    *,
    num_qubits: Optional[int] = None,
    n_tau: Optional[int] = None,
    reset_qubits: bool = False,
    differential_implicit_prior: int = 0,
) -> FidStackResult:
    """
    Same pipeline as ``save_memory_3d_plots`` in ``tuna_fid_single_job`` for the stack:

    - ``memory_to_stack`` → ``(n_shots, n_qubits, n_tau)``
    - if ``not reset_qubits`` (no-reset Ramsey): ``differential_readout_along_tau``

    For pickles from circuits **without** per-τ reset, pass ``reset_qubits=False`` (default).
    """
    _ensure_inspire_on_path()
    from tuna_fid_single_job import (
        differential_readout_along_tau,
        memory_to_stack,
        warn_if_memory_qubit_mismatch,
    )

    if not mem:
        raise ValueError("memory list is empty")

    if n_tau is None:
        # Default matches ``tau_ns_from_indices(1, 51)`` in ``tuna_fid_single_job.main``
        n_tau = 50

    if num_qubits is None:
        num_qubits = infer_num_qubits(mem, n_tau)

    warn_if_memory_qubit_mismatch(mem, n_tau, num_qubits, context="(build_stack_like_single_job)")

    stack = memory_to_stack(mem, num_qubits, n_tau)
    differential = not reset_qubits
    if differential:
        stack = differential_readout_along_tau(
            stack, implicit_prior=differential_implicit_prior
        )

    n_shots, n_q, nt = stack.shape
    if nt != n_tau or n_q != num_qubits:
        raise ValueError(
            f"stack shape {stack.shape} inconsistent with n_tau={n_tau}, num_qubits={num_qubits}"
        )

    return FidStackResult(
        stack=stack,
        num_qubits=num_qubits,
        n_tau=n_tau,
        n_shots=n_shots,
        differential=differential,
    )


def stack_to_vae_tensors(
    stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Qubit-persona flattening (same idea as ``qubit_persona_flatten`` in ``tuna_fid_single_job``).

    Returns
    -------
    x : ndarray, shape ``(n_shots * n_qubits, n_tau)``, float32 in ``{0.0, 1.0}``
    qubit_ids : ndarray, shape ``(n_shots * n_qubits,)``
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D, got {stack.shape}")
    n_shots, n_qubits, n_tau = stack.shape
    x = stack.reshape(n_shots * n_qubits, n_tau).astype(np.float32, copy=False)
    qubit_ids = np.tile(np.arange(n_qubits, dtype=np.int64), n_shots)
    return x, qubit_ids
