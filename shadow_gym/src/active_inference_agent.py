"""
ActiveInferenceAgent: Expected Free Energy (EFE) controller for adaptive tomography.

Maintains a running mean and observation count for each Pauli observable.
Uncertainty is estimated as 1 / (N_obs + 1) — no GP covariance matrix needed.

Selects measurement bases that maximize epistemic utility:

    U(B) = sum_{P compatible with B} w_P / (N_obs(P) + 1)

where the weight (from specification §3.2):

    w_P = (|mu_P| + 1) * alpha^{weight(P)} * chi_P

    alpha ~ 2.0   (locality boost: prefer high-weight operators)
    chi_P = 50.0  for stabilizers, 1.0 otherwise
"""
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from itertools import product as iproduct

from .utils import (
    all_pauli_strings,
    pauli_weight,
    pauli_shadow_value,
    pauli_string_matrix,
    cluster_state_stabilizers,
    is_compatible,
)


PAULI_BASES = ["X", "Y", "Z"]


class ActiveInferenceAgent:
    """
    Active Inference controller.

    Parameters
    ----------
    n_qubits : int
    alpha : float
        Locality weight exponent (spec: ~2.0).
    chi_stabilizer : float
        Relevance boost for stabilizer observables (spec: 50.0).
    max_weight : int
        Maximum Pauli weight to track (limits observable set size).
    temperature : float
        Softmax temperature for stochastic basis selection.
        0 → greedy argmax, large → uniform random.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        alpha: float = 2.0,
        chi_stabilizer: float = 1.0,
        max_weight: int = 4,
        temperature: float = 1.0,
        ucb_c: float = 1.0,
    ):
        self.n = n_qubits
        self.alpha = alpha
        self.chi_stabilizer = chi_stabilizer
        self.temperature = temperature
        self.ucb_c = ucb_c

        # All tracked Pauli observables
        self.paulis: list[str] = all_pauli_strings(n_qubits, min_weight=1, max_weight=max_weight)
        self.pauli_to_idx: dict[str, int] = {p: i for i, p in enumerate(self.paulis)}
        self.stabilizers: set[str] = set(cluster_state_stabilizers(n_qubits))

        self._n_paulis = len(self.paulis)

        # Belief state: running mean + observation count per Pauli.
        # Variance is approximated as 1 / (N_obs + 1) — no GP covariance matrix.
        self.mu: np.ndarray = np.zeros(self._n_paulis, dtype=np.float64)
        self._n_obs: np.ndarray = np.ones(self._n_paulis, dtype=np.int64)

        # All possible single-qubit basis combinations (3^n options)
        self._all_bases: list[list[str]] = [
            list(combo)
            for combo in iproduct(PAULI_BASES, repeat=n_qubits)
        ]

        # Precomputed static quantities for fast EFE
        self._n_bases  = len(self._all_bases)

        # Compatibility mask: shape (n_bases, n_paulis), True if basis can measure Pauli
        self._compat = np.zeros((self._n_bases, self._n_paulis), dtype=bool)
        for i, b in enumerate(self._all_bases):
            for j, p in enumerate(self.paulis):
                self._compat[i, j] = is_compatible(p, b)

        # Static weight factor α^weight × χ (does not depend on beliefs)
        self._w_static = np.array([
            (self.alpha ** pauli_weight(p)) * (self.chi_stabilizer if p in self.stabilizers else 1.0)
            for p in self.paulis
        ])

        # UCB1: per-basis pull counts and total step counter
        self._n_pulls = np.zeros(self._n_bases, dtype=np.int64)
        self._t_total = 0

    # ------------------------------------------------------------------ #
    # Belief update  (O(N) online mean — no covariance tracking)
    # ------------------------------------------------------------------ #

    def update(self, shots: list[tuple[list[str], list[int]]]) -> None:
        """O(N) update: track running mean per Pauli, no GP covariance matrix."""
        for basis, outcome in shots:
            if any(b == "D" for b in basis):
                continue  # skip QND diagonal shots

            for i, p in enumerate(self.paulis):
                val = pauli_shadow_value(p, basis, outcome)
                if val != 0.0:
                    n = self._n_obs[i]
                    self.mu[i] = (n * self.mu[i] + val) / (n + 1)
                    self._n_obs[i] += 1

    # ------------------------------------------------------------------ #
    # Basis selection
    # ------------------------------------------------------------------ #

    def _score_bases(self) -> np.ndarray:
        """EFE + UCB. Uncertainty = 1 / (N_obs + 1), no GP required."""
        vars_ = 1.0 / (self._n_obs + 1.0)

        # w = (|mu| + 1) * alpha^weight * chi
        w = (np.abs(self.mu) + 1.0) * self._w_static
        efe_scores = (self._compat * (w * vars_)).sum(axis=1)

        t = max(self._t_total, 1)
        ucb = self.ucb_c * np.sqrt(2.0 * np.log(t + 1) / (self._n_pulls + 1))
        ucb *= (efe_scores.max() if efe_scores.max() > 0 else 1.0)

        return efe_scores + ucb

    def _softmax_probs(self) -> np.ndarray:
        """Softmax over EFE+UCB scores."""
        scores = self._score_bases()
        if self.temperature == 0:
            probs = np.zeros(self._n_bases)
            probs[int(np.argmax(scores))] = 1.0
            return probs
        s = scores / (self.temperature + 1e-12)
        s -= s.max()
        probs = np.exp(s)
        probs /= probs.sum()
        return probs

    def efe(self, basis: list[str]) -> float:
        """EFE for a single basis (diagnostic use)."""
        idx = self._all_bases.index(basis)
        vars_ = 1.0 / (self._n_obs + 1.0)
        w = (np.abs(self.mu) + 1.0) * self._w_static
        return float((self._compat[idx] * w * vars_).sum())

    def _record_pulls(self, indices: np.ndarray) -> None:
        for i in indices:
            self._n_pulls[i] += 1
        self._t_total += len(indices)

    def select_basis(self) -> list[str]:
        """Sample next basis via softmax over EFE+UCB."""
        probs = self._softmax_probs()
        idx = np.random.choice(self._n_bases, p=probs)
        self._record_pulls(np.array([idx]))
        return list(self._all_bases[idx])

    def select_basis_with_prob(self) -> tuple[list[str], float]:
        """Select basis and return sampling probability (for IS reconstruction)."""
        probs = self._softmax_probs()
        idx = np.random.choice(self._n_bases, p=probs)
        self._record_pulls(np.array([idx]))
        return list(self._all_bases[idx]), float(probs[idx])

    def select_batch(self, batch_size: int) -> list[list[str]]:
        """Batch select. One softmax call per batch (beliefs static within batch)."""
        probs = self._softmax_probs()
        indices = np.random.choice(self._n_bases, size=batch_size, p=probs)
        self._record_pulls(indices)
        return [list(self._all_bases[i]) for i in indices]

    def select_batch_with_probs(self, batch_size: int) -> list[tuple[list[str], float]]:
        """Batch select with IS probabilities."""
        probs = self._softmax_probs()
        indices = np.random.choice(self._n_bases, size=batch_size, p=probs)
        self._record_pulls(indices)
        return [(list(self._all_bases[i]), float(probs[i])) for i in indices]

    # ------------------------------------------------------------------ #
    # Hardware Softmax angles  (for Act III / QND biasing)
    # ------------------------------------------------------------------ #

    def softmax_angles(self, qubit_idx: int) -> tuple[float, float]:
        """
        Compute (theta1, theta2) for ancilla initialization such that
        P(a1=1) ∝ marginal probability of measuring X on qubit_idx,
        P(a2=1) ∝ marginal probability of measuring Y on qubit_idx.

        This embeds the agent's exploration distribution into ancilla angles
        for use in the Hardware Softmax QND circuit.
        """
        probs = self._softmax_probs()
        # Marginal P(b_i = X) and P(b_i = Y) for qubit qubit_idx
        p_x = sum(probs[k] for k, b in enumerate(self._all_bases) if b[qubit_idx] == "X")
        p_y = sum(probs[k] for k, b in enumerate(self._all_bases) if b[qubit_idx] == "Y")

        # P(a1=1) = P(X), P(a2=1) = P(Y); theta = 2*arcsin(sqrt(P))
        theta1 = 2 * np.arcsin(np.sqrt(np.clip(p_x, 0, 1)))
        theta2 = 2 * np.arcsin(np.sqrt(np.clip(p_y, 0, 1)))
        return theta1, theta2

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def belief_summary(self, paulis: Optional[List[str]] = None) -> dict:
        """Return dict of {pauli: (mu, sigma)} where sigma = 1/sqrt(N_obs+1)."""
        targets = paulis or self.paulis
        sigmas = 1.0 / np.sqrt(self._n_obs + 1.0)
        return {
            p: (self.mu[self.pauli_to_idx[p]], sigmas[self.pauli_to_idx[p]])
            for p in targets
        }

    def n_observed(self, pauli: str) -> int:
        if pauli not in self.pauli_to_idx:
            return 0
        return int(self._n_obs[self.pauli_to_idx[pauli]])
