"""Shared primitives: Pauli matrices, rotation unitaries, fidelity, Pauli string tools."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from itertools import product as iproduct

# --------------------------------------------------------------------------- #
# Pauli matrices
# --------------------------------------------------------------------------- #
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MAP = {"I": I2, "X": X, "Y": Y, "Z": Z}

# --------------------------------------------------------------------------- #
# Basis rotation unitaries U_b such that U_b maps basis-b eigenstates to |0>,|1>
#
# U_X = H:  H|+> = |0>, H|-> = |1>
# U_Y = HS†: (HS†)|+y> = |0>, (HS†)|-y> = |1>
# U_Z = I:  trivial
# --------------------------------------------------------------------------- #
H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sd = np.array([[1, 0], [0, -1j]], dtype=complex)   # S† (conjugate of S=diag(1,i))
HSd = H @ Sd                                        # U_Y = HS†

BASIS_UNITARIES: Dict[str, np.ndarray] = {"X": H, "Y": HSd, "Z": I2}

# Eigenvalues: U_b|0> = |+b> (eigenvalue +1), U_b|1> = |-b> (eigenvalue -1)
# Shadow formula for qubit i with basis b_i, outcome s_i:
#   rho_hat_i = U_b_i† (3|s_i><s_i| - I) U_b_i
# For Pauli P_i:
#   Tr(P_i @ rho_hat_i) = 3*(-1)^s_i  if P_i == b_i
#                       = 1            if P_i == 'I'
#                       = 0            otherwise (incompatible)


def kron_n(*ops: np.ndarray) -> np.ndarray:
    """Tensor product of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_string_matrix(s: str) -> np.ndarray:
    """'XYZZ' -> 16x16 matrix."""
    return kron_n(*[PAULI_MAP[c] for c in s])


def all_pauli_strings(n: int, min_weight: int = 1, max_weight: Optional[int] = None) -> List[str]:
    """All n-qubit Pauli strings (non-trivial) with weight in [min_weight, max_weight]."""
    if max_weight is None:
        max_weight = n
    result = []
    for combo in iproduct("IXYZ", repeat=n):
        s = "".join(combo)
        w = sum(c != "I" for c in s)
        if min_weight <= w <= max_weight:
            result.append(s)
    return result


def pauli_weight(s: str) -> int:
    return sum(c != "I" for c in s)


def is_compatible(pauli: str, basis: List[str]) -> bool:
    """True if every non-identity Pauli in `pauli` matches the corresponding basis."""
    return all(p == "I" or p == b for p, b in zip(pauli, basis))


def pauli_shadow_value(pauli: str, basis: List[str], outcome: List[int]) -> float:
    """
    Efficient single-shot shadow estimate of Tr(P @ rho_hat) using Pauli formula.
    Returns 3^w * prod((-1)^s_i for P_i != I) if compatible, else 0.
    """
    if not is_compatible(pauli, basis):
        return 0.0
    val = 1.0
    for p, b, s in zip(pauli, basis, outcome):
        if p == "I":
            pass  # factor 1
        else:
            val *= 3 * ((-1) ** s)
    return val


def shadow_snapshot_matrix(basis: List[str], outcome: List[int]) -> np.ndarray:
    """
    Full 2^n x 2^n shadow matrix for one shot.
    rho_hat = tensor_i (U_b_i† (3|s_i><s_i| - I) U_b_i)
    """
    ops = []
    for b, s in zip(basis, outcome):
        U = BASIS_UNITARIES[b]
        ket = np.zeros(2, dtype=complex)
        ket[s] = 1.0
        proj = np.outer(ket, ket)
        ops.append(U.conj().T @ (3 * proj - I2) @ U)
    return kron_n(*ops)


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Quantum fidelity F(rho1, rho2) = Tr(sqrt(sqrt(rho1) rho2 sqrt(rho1)))^2."""
    sqrt1 = _matrix_sqrt(rho1)
    M = sqrt1 @ rho2 @ sqrt1
    return float(np.real(np.trace(_matrix_sqrt(M))) ** 2)


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T


def cluster_state_stabilizers(n: int) -> List[str]:
    """1D cluster state stabilizers: K_0 = XZ, K_i = ZXZ, K_{n-1} = ZX."""
    stabs = []
    for i in range(n):
        ops = ["I"] * n
        ops[i] = "X"
        if i > 0:
            ops[i - 1] = "Z"
        if i < n - 1:
            ops[i + 1] = "Z"
        stabs.append("".join(ops))
    return stabs
