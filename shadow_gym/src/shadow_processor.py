"""
ShadowProcessor: reconstruct density matrices from classical shadow data.

Supports Pauli bases {X, Y, Z} and the QND diagonal basis 'D'.
Uses the standard inverse depolarizing channel M^{-1} for Pauli shadows.
"""
from typing import Optional, List, Tuple, Dict
import numpy as np

from .utils import (
    BASIS_UNITARIES,
    I2,
    kron_n,
    pauli_string_matrix,
    pauli_shadow_value,
    shadow_snapshot_matrix,
    fidelity,
    is_compatible,
    pauli_weight,
)

# Rotation for QND diagonal basis 'D': Rx(+pi/2) @ Ry(-pi/2)
_RY_NEG = np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2)
_RX_POS = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
_QND_D_UNITARY = _RX_POS @ _RY_NEG

# Extended basis unitary lookup including 'D'
_ALL_UNITARIES: Dict[str, np.ndarray] = {**BASIS_UNITARIES, "D": _QND_D_UNITARY}


def _project_dm(rho: np.ndarray) -> np.ndarray:
    """Project Hermitian matrix onto valid density matrix (PSD + trace-1)."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0.0)
    s = vals.sum()
    if s > 1e-12:
        vals /= s
    return vecs @ np.diag(vals) @ vecs.conj().T


class ShadowProcessor:
    """
    Convert (basis, outcome) shot data into density matrix estimates.

    Parameters
    ----------
    n_qubits : int
        Number of data qubits.
    """

    def __init__(self, n_qubits: int = 4):
        self.n = n_qubits
        self.dim = 2 ** n_qubits

    # ------------------------------------------------------------------ #
    # Dense density matrix reconstruction
    # ------------------------------------------------------------------ #

    def reconstruct(
        self,
        shots: List[Tuple[List[str], List[int]]],
        n_snapshots: Optional[int] = None,
        project: bool = True,
        regularization: float = 0.0,
    ) -> np.ndarray:
        """
        Reconstruct density matrix as average of shadow snapshots.

        Parameters
        ----------
        shots : list of (basis, outcome) tuples.
        n_snapshots : use only first n_snapshots shots (for convergence curves).
        project : project result onto valid density matrix (Hermitian, PSD, trace-1).
                  Required when sampling is non-uniform (e.g. adaptive AI).
        regularization : Tikhonov regularization strength (MLE-like, 0 = no reg).
                         Mixes estimate with maximally mixed: ρ_reg = (1-λ)ρ + λ·I/d.
                         Use small λ ~0.01 to suppress small eigenvalues.

        Returns
        -------
        rho_est : (dim x dim) complex numpy array.
        """
        data = shots[:n_snapshots] if n_snapshots else shots
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        for basis, outcome in data:
            rho += self._snapshot(basis, outcome)
        rho /= len(data)
        rho = (rho + rho.conj().T) / 2

        # Optional Tikhonov regularization (MLE-like shrinkage)
        if regularization > 0:
            mixed = np.eye(self.dim, dtype=complex) / self.dim
            rho = (1 - regularization) * rho + regularization * mixed

        if project:
            rho = _project_dm(rho)
        return rho

    def reconstruct_is(
        self,
        shots_probs: List[Tuple[List[str], List[int], float]],
        n_snapshots: Optional[int] = None,
        p_ref: Optional[float] = None,
    ) -> np.ndarray:
        """
        Importance-weighted shadow reconstruction for adaptive (non-iid) sampling.

        Corrects the bias introduced when the AI agent selects bases non-uniformly.
        Each shot is reweighted by w_t = P_ref(b_t) / P_agent_t(b_t).

        Parameters
        ----------
        shots_probs : list of (basis, outcome, p_agent) where p_agent is the
                      probability the agent assigned to this basis at selection time.
        p_ref : reference distribution weight. Default: 1/3^n (uniform Pauli).

        Returns
        -------
        rho_est : valid density matrix.
        """
        if p_ref is None:
            p_ref = 1.0 / (3 ** self.n)   # uniform over {X,Y,Z}^n

        data = shots_probs[:n_snapshots] if n_snapshots else shots_probs
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        total_w = 0.0
        for basis, outcome, p_agent in data:
            w = p_ref / max(p_agent, 1e-12)
            rho += w * self._snapshot(basis, outcome)
            total_w += w

        if total_w > 0:
            rho /= total_w
        rho = (rho + rho.conj().T) / 2
        return _project_dm(rho)

    def fidelity_curve_is(
        self,
        shots_probs: List[Tuple[List[str], List[int], float]],
        rho_true: np.ndarray,
        checkpoints: List[int],
    ) -> List[float]:
        """fidelity_curve for importance-weighted (AI) reconstruction."""
        return [
            fidelity(self.reconstruct_is(shots_probs, n), rho_true)
            for n in checkpoints
        ]

    def reconstruct_from_agent(
        self,
        paulis: List[str],
        shots: List[Tuple[List[str], List[int]]],
    ) -> np.ndarray:
        """
        Reconstruct density matrix from unbiased Pauli estimators.

        Bypasses shadow snapshots and importance-sampling weights entirely.
        For each tracked Pauli P, estimates ⟨P⟩ by averaging compatible shots,
        then builds ρ = Σ_P ⟨P⟩ · P / 2^n  (Pauli expansion).

        This is unbiased for ANY sampling distribution, including adaptive AI.

        Parameters
        ----------
        paulis : Pauli strings to include in the expansion (e.g. agent.paulis).
        shots  : list of (basis, outcome) tuples.

        Returns
        -------
        rho_est : valid density matrix (Hermitian, PSD, trace-1).
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        for p in paulis:
            exp_val = self.estimate_pauli(p, shots)
            rho += exp_val * pauli_string_matrix(p)
        rho /= self.dim
        return _project_dm((rho + rho.conj().T) / 2)

    def _snapshot(self, basis: List[str], outcome: List[int]) -> np.ndarray:
        """Single-shot shadow matrix using actual rotation unitaries."""
        ops = []
        for b, s in zip(basis, outcome):
            U = _ALL_UNITARIES[b]
            ket = np.zeros(2, dtype=complex)
            ket[s] = 1.0
            proj = np.outer(ket, ket)
            ops.append(U.conj().T @ (3 * proj - I2) @ U)
        return kron_n(*ops)

    # ------------------------------------------------------------------ #
    # QND-aware reconstruction (corrected non-uniform shadow coefficients)
    # ------------------------------------------------------------------ #

    def reconstruct_qnd(
        self,
        shots: List[Tuple[List[str], List[int]]],
        basis_probs: Optional[Dict[str, float]] = None,
        n_snapshots: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reconstruct from QND shots using per-basis shadow coefficients
        c_b = 1/P_eff(b), correcting for non-uniform sampling.

        For uniform ancilla init (theta=pi/2): P(X)=P(Z)=0.25, P(Y)=0.5
        so c_X=c_Z=4, c_Y=2.

        basis_probs : dict {'X': p_x, 'Y': p_y, 'Z': p_z}
            Effective per-qubit probabilities.  Default: uniform QND.
        """
        if basis_probs is None:
            basis_probs = {"X": 0.25, "Y": 0.50, "Z": 0.25}
        coeffs = {b: 1.0 / p for b, p in basis_probs.items()}

        data = shots[:n_snapshots] if n_snapshots else shots
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        for basis, outcome in data:
            rho += self._snapshot_coeffs(basis, outcome, coeffs)
        rho /= len(data)
        rho = (rho + rho.conj().T) / 2
        # Apply PSD projection to ensure valid density matrix
        rho = _project_dm(rho)
        return rho

    def _snapshot_coeffs(
        self,
        basis: List[str],
        outcome: List[int],
        coeffs: Dict[str, float],
    ) -> np.ndarray:
        """Shadow snapshot with per-basis scaling c_b * Pi - I per qubit."""
        ops = []
        for b, s in zip(basis, outcome):
            U = _ALL_UNITARIES[b]
            ket = np.zeros(2, dtype=complex)
            ket[s] = 1.0
            proj = np.outer(ket, ket)
            c = coeffs.get(b, 3.0)
            ops.append(U.conj().T @ (c * proj - I2) @ U)
        return kron_n(*ops)

    def fidelity_curve_qnd(
        self,
        shots: List[Tuple[List[str], List[int]]],
        rho_true: np.ndarray,
        checkpoints: List[int],
        basis_probs: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """fidelity_curve but using reconstruct_qnd."""
        return [
            fidelity(self.reconstruct_qnd(shots, basis_probs, n), rho_true)
            for n in checkpoints
        ]

    # ------------------------------------------------------------------ #
    # Efficient Pauli expectation estimation (no full matrix per shot)
    # ------------------------------------------------------------------ #

    def estimate_pauli(
        self,
        pauli: str,
        shots: List[Tuple[List[str], List[int]]],
    ) -> float:
        """
        Estimate Tr(P @ rho) — unbiased for ANY sampling distribution
        (uniform classical, adaptive AI, non-uniform QND).

        Method: average (-1)^parity over compatible shots only.
        E[(-1)^parity | compat] = ⟨P⟩ since outcomes depend only on state ρ
        (not on how basis was selected).  No 3^w rescaling needed.

        For 'D' basis shots, falls back to numeric trace.
        """
        compat_vals = []
        for basis, outcome in shots:
            if "D" in basis:
                # Use numeric formula and divide by 3^weight to get ⟨P⟩
                w = sum(1 for c in pauli if c != "I")
                if w == 0:
                    compat_vals.append(1.0)
                    continue
                v = self._pauli_value_numeric(pauli, basis, outcome)
                if v != 0.0:
                    compat_vals.append(v / (3 ** w))
                continue
            if not is_compatible(pauli, basis):
                continue
            parity = sum(o for o, p in zip(outcome, pauli) if p != "I") % 2
            compat_vals.append(1.0 if parity == 0 else -1.0)
        if not compat_vals:
            return 0.0
        return float(np.mean(compat_vals))

    def _pauli_value_numeric(
        self, pauli: str, basis: List[str], outcome: List[int]
    ) -> float:
        """Tr(P @ snapshot) via matrix multiply (used for 'D' basis)."""
        snap = self._snapshot(basis, outcome)
        P = pauli_string_matrix(pauli)
        return float(np.real(np.trace(P @ snap)))

    # ------------------------------------------------------------------ #
    # Convergence tracking
    # ------------------------------------------------------------------ #

    def fidelity_curve(
        self,
        shots: List[Tuple[List[str], List[int]]],
        rho_true: np.ndarray,
        checkpoints: List[int],
    ) -> List[float]:
        """
        Compute fidelity at each checkpoint n_shots.

        Parameters
        ----------
        shots : full shot list.
        rho_true : true density matrix.
        checkpoints : sorted list of shot counts at which to evaluate.

        Returns
        -------
        list of fidelity values (one per checkpoint).
        """
        fidelities = []
        for n in checkpoints:
            rho_est = self.reconstruct(shots, n_snapshots=n)
            fidelities.append(fidelity(rho_est, rho_true))
        return fidelities

    def pauli_errors(
        self,
        paulis: List[str],
        shots: List[Tuple[List[str], List[int]]],
        rho_true: np.ndarray,
        checkpoints: List[int],
    ) -> Dict[str, List[float]]:
        """
        Track |<P>_est - <P>_true| for a set of Pauli observables.

        Returns dict mapping each Pauli string to its error curve.
        """
        true_vals = {
            p: float(np.real(np.trace(pauli_string_matrix(p) @ rho_true)))
            for p in paulis
        }
        errors: Dict[str, List[float]] = {p: [] for p in paulis}
        for n in checkpoints:
            sub = shots[:n]
            for p in paulis:
                est = self.estimate_pauli(p, sub)
                errors[p].append(abs(est - true_vals[p]))
        return errors
