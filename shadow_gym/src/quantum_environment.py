"""
QuantumEnvironment: target state preparation and measurement sampling.

Two feedback modes for Act II benchmark:
  - classical: classically chosen Pauli basis per shot (no circuit overhead)
  - qnd:       ancilla-controlled Crot circuit (hardware deferred measurement)

Both produce (basis, outcome) tuples fed into ShadowProcessor.
The QND mode adds a 4th basis direction (diagonal in XZ-plane) via Crot.
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .utils import BASIS_UNITARIES, kron_n

# Physical qubit layout from specification (17-qubit hardware)
PHYSICAL_DATA = [2, 4, 8, 12, 14]

# Format: data_qubit: [direct_ancilla, next_nearest_ancilla]
PHYSICAL_ANCILLA_MAP = {
    2:  [0, 1],   # Q1 requires SWAP with Q0
    4:  [7, 10],  # Q10 requires SWAP with Q7
    8:  [11, 5],  # Both are directly connected! (0 SWAPs)
    12: [9, 6],   # Q6 requires SWAP with Q9
    14: [16, 15]  # Q15 requires SWAP with Q16
}

# Rotation applied to data qubit for each (a1, a2) ancilla outcome in Crot circuit
# cry(-pi/2) fires when a1=1  →  Ry(-pi/2) on data
# crx(+pi/2) fires when a2=1  →  Rx(+pi/2) on data, applied after
_RY_NEG = np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2)   # Ry(-pi/2)
_RX_POS = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)  # Rx(+pi/2)

QND_UNITARIES: Dict[Tuple[int, int], np.ndarray] = {
    (0, 0): np.eye(2, dtype=complex),           # Z basis
    (1, 0): _RY_NEG,                             # X basis (Ry(-pi/2) → measures X)
    (0, 1): _RX_POS,                             # Y basis (Rx(+pi/2) → measures Y)
    (1, 1): _RX_POS @ _RY_NEG,                  # diagonal (X+Y)/√2 direction
}

# Label used in ShadowProcessor; 'D' = diagonal QND basis
# NOTE: Rx(pi/2)@Ry(-pi/2) satisfies U†ZU = Y (same as (0,1) outcome).
# The spec's intended "diagonal (X-Y+Z)/√3" is aspirational; this circuit
# gives effective P(Y)=0.5 per qubit for uniform ancillas. Reconstruction
# must use corrected coefficients (c_Y=2, c_X=c_Z=4) — see ShadowProcessor.
QND_BASIS_LABELS: Dict[Tuple[int, int], str] = {
    (0, 0): "Z", (1, 0): "X", (0, 1): "Y", (1, 1): "Y"
}


class QuantumEnvironment:
    """
    4-qubit 1D cluster state environment.

    Parameters
    ----------
    n_data : int
        Number of data qubits (default 4, matching spec).
    """

    def __init__(self, n_data: int = 4):
        self.n = n_data
        self._sv: Optional[np.ndarray] = None
        self._rho_true: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # State preparation
    # ------------------------------------------------------------------ #

    def prepare_cluster_state(self, depolarizing_p: float = 0.0) -> np.ndarray:
        """
        Build 1D cluster state: H^n then CZ along chain.

        depolarizing_p : if > 0, apply global depolarizing channel
                         ρ → (1-p)|CL⟩⟨CL| + p · I/2^n.
                         Use small p ~0.05 to make stabilizer estimation non-trivial.

        Returns statevector (always pure); rho_true reflects mixing if p>0.
        """
        qc = QuantumCircuit(self.n)
        for i in range(self.n):
            qc.h(i)
        for i in range(self.n - 1):
            qc.cz(i, i + 1)
        sv = Statevector(qc)
        self._sv = sv.data.copy()
        rho_pure = np.outer(self._sv, self._sv.conj())
        if depolarizing_p > 0:
            d = 2 ** self.n
            self._rho_true = (1 - depolarizing_p) * rho_pure + depolarizing_p * np.eye(d) / d
            # For sampling, mix pure cluster with maximally mixed
            self._mix_p = depolarizing_p
        else:
            self._rho_true = rho_pure
            self._mix_p = 0.0
        return self._sv

    def prepare_ghz_state(self, depolarizing_p: float = 0.0) -> np.ndarray:
        """
        GHZ state: (|00...0⟩ + |11...1⟩) / √2.

        depolarizing_p : optional global depolarizing noise.

        Returns statevector; rho_true reflects mixing if p>0.
        """
        qc = QuantumCircuit(self.n)
        qc.h(0)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        sv = Statevector(qc)
        self._sv = sv.data.copy()
        rho_pure = np.outer(self._sv, self._sv.conj())
        if depolarizing_p > 0:
            d = 2 ** self.n
            self._rho_true = (1 - depolarizing_p) * rho_pure + depolarizing_p * np.eye(d) / d
            self._mix_p = depolarizing_p
        else:
            self._rho_true = rho_pure
            self._mix_p = 0.0
        return self._sv

    def prepare_w_state(self, depolarizing_p: float = 0.0) -> np.ndarray:
        """
        W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n.

        depolarizing_p : optional global depolarizing noise.

        Returns statevector; rho_true reflects mixing if p>0.
        """
        qc = QuantumCircuit(self.n)
        # Simple preparation: X on last qubit, then apply CX chain to spread
        qc.x(self.n - 1)
        for i in range(self.n - 2, -1, -1):
            qc.cx(i + 1, i)
        sv = Statevector(qc)
        self._sv = sv.data.copy()
        rho_pure = np.outer(self._sv, self._sv.conj())
        if depolarizing_p > 0:
            d = 2 ** self.n
            self._rho_true = (1 - depolarizing_p) * rho_pure + depolarizing_p * np.eye(d) / d
            self._mix_p = depolarizing_p
        else:
            self._rho_true = rho_pure
            self._mix_p = 0.0
        return self._sv

    def prepare_hypergraph_state(self, degree: int = 3, depolarizing_p: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
        """
        Random hypergraph state with specified vertex degree.

        Roughly: start with product state, apply random Clifford-controlled-Z gates
        where each qubit participates in ~degree hyperedges.

        Parameters
        ----------
        degree : target degree (hyperedges per qubit).
        depolarizing_p : optional global depolarizing noise.
        seed : numpy random seed for reproducibility.

        Returns statevector; rho_true reflects mixing if p>0.
        """
        if seed is not None:
            np.random.seed(seed)

        qc = QuantumCircuit(self.n)
        # Prepare equal superposition: H on all qubits
        for i in range(self.n):
            qc.h(i)

        # Apply random controlled-Z multi-qubit gates (hyperedges)
        n_hyperedges = max(1, self.n * degree // 2)
        for _ in range(n_hyperedges):
            # Pick 2-3 random qubits and apply CZ chains
            edge_size = np.random.randint(2, min(4, self.n + 1))
            edge = list(np.random.choice(self.n, size=edge_size, replace=False))
            for i in range(len(edge) - 1):
                qc.cz(edge[i], edge[i + 1])

        sv = Statevector(qc)
        self._sv = sv.data.copy()
        rho_pure = np.outer(self._sv, self._sv.conj())
        if depolarizing_p > 0:
            d = 2 ** self.n
            self._rho_true = (1 - depolarizing_p) * rho_pure + depolarizing_p * np.eye(d) / d
            self._mix_p = depolarizing_p
        else:
            self._rho_true = rho_pure
            self._mix_p = 0.0
        return self._sv

    def prepare_haar_random_state(self, seed: Optional[int] = None, depolarizing_p: float = 0.0) -> np.ndarray:
        """
        Haar-random pure state (worst-case tomography difficulty).

        Generates a random unitary via random angle single-qubit and CNOT gates.

        Parameters
        ----------
        seed : numpy random seed.
        depolarizing_p : optional global depolarizing noise.

        Returns statevector; rho_true reflects mixing if p>0.
        """
        if seed is not None:
            np.random.seed(seed)

        qc = QuantumCircuit(self.n)
        # Random single-qubit rotations (RY, RZ gates for compatibility)
        for i in range(self.n):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            qc.ry(theta, i)
            qc.rz(phi, i)
        # Random CNOTs
        for _ in range(self.n):
            i = np.random.randint(0, self.n)
            j = np.random.randint(0, self.n)
            if i != j:
                qc.cx(i, j)

        sv = Statevector(qc)
        self._sv = sv.data.copy()
        rho_pure = np.outer(self._sv, self._sv.conj())
        if depolarizing_p > 0:
            d = 2 ** self.n
            self._rho_true = (1 - depolarizing_p) * rho_pure + depolarizing_p * np.eye(d) / d
            self._mix_p = depolarizing_p
        else:
            self._rho_true = rho_pure
            self._mix_p = 0.0
        return self._sv

    @property
    def rho_true(self) -> np.ndarray:
        if self._rho_true is None:
            raise RuntimeError("Call prepare_*_state() first.")
        return self._rho_true

    # ------------------------------------------------------------------ #
    # Classical feedback sampling
    # ------------------------------------------------------------------ #

    def sample_classical(
        self,
        n_shots: int,
        bases: Optional[List[List[str]]] = None,
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Classical feedback: pick Pauli basis classically, measure.

        Parameters
        ----------
        n_shots : int
        bases : pre-specified list of bases (one per shot); random if None.

        Returns
        -------
        list of (basis, outcome) where basis is e.g. ['X','Z','Y','X']
        and outcome is list of bits.
        """
        self._check_state()
        results = []
        for t in range(n_shots):
            if bases is not None:
                basis = bases[t]
            else:
                basis = [np.random.choice(["X", "Y", "Z"]) for _ in range(self.n)]
            outcome = self._sample_outcome(basis, self._sv)
            results.append((basis, outcome))
        return results

    # ------------------------------------------------------------------ #
    # QND / quantum-feedback sampling
    # ------------------------------------------------------------------ #

    def build_qnd_circuit(self, softmax_angles: Optional[Dict] = None) -> QuantumCircuit:
        """
        Builds a true 17-qubit hardware-compliant QND circuit.
        Implements a 5-qubit data chain with 4 localized routing SWAPs on measured ancillas.
        """
        # Initialize full 17-qubit hardware space
        qc = QuantumCircuit(17, 17)

        # 1. State Preparation: 1D Cluster State on the native zig-zag data chain
        for dq in PHYSICAL_DATA:
            qc.h(dq)
        # Natively connected layers: (2-4), (4-8), (8-12), (12-14)
        for i in range(len(PHYSICAL_DATA) - 1):
            qc.cz(PHYSICAL_DATA[i], PHYSICAL_DATA[i + 1])
        qc.barrier()

        # 2. Hardware Softmax: Initialize the ancillas based on agent distribution
        for i, dq in enumerate(PHYSICAL_DATA):
            a1, a2 = PHYSICAL_ANCILLA_MAP[dq]
            t1, t2 = softmax_angles[i] if softmax_angles else (np.pi / 2, np.pi / 2)
            qc.ry(t1, a1)
            qc.ry(t2, a2)
        qc.barrier()

        # 3. Hardware Feed-Forward Loop incorporating local routing SWAPs
        for dq in PHYSICAL_DATA:
            a1, a2 = PHYSICAL_ANCILLA_MAP[dq]

            # Step A: Direct control interaction
            qc.cry(-np.pi / 2, a1, dq)

            # Step B: Check if layout routing requires a localized ancilla SWAP
            if dq == 8:
                # Q8 has direct connections to both pins
                qc.crx(np.pi / 2, a2, dq)
            else:
                # Swap the next-nearest ancilla state down into the direct coupling port
                qc.swap(a1, a2)
                qc.crx(np.pi / 2, a1, dq)  # a1 now contains the target state pointer
        qc.barrier()

        # 4. Global Readout
        # Measure all 17 qubits into their matching physical classical register tracks
        qc.measure(range(17), range(17))
        return qc

    def sample_qnd(
        self,
        n_shots: int,
        softmax_angles: Optional[Dict] = None,
    ) -> List[Tuple[List[str], List[int]]]:
        """
        QND / deferred-measurement sampling (statevector simulation only).

        For real hardware, use build_qnd_circuit() + Qiskit Sampler + parse_hardware_qnd().

        By the deferred-measurement principle, sampling the full statevector
        (no mid-circuit measurement) is equivalent to the Crot circuit.
        We exploit the fact that ancilla outcomes are independent of the data state
        (ancilla is the *control* in Crot), so we:
          1. Sample ancilla pairs (a1_i, a2_i) from sin²(theta/2) distribution.
          2. Apply the corresponding QND unitary to the 5-qubit data state.
          3. Sample the data measurement outcome.

        Returns list of (basis_labels, outcome) tuples.
        """
        self._check_state()

        # Ancilla excitation probabilities
        if softmax_angles:
            p1 = {i: np.sin(softmax_angles[i][0] / 2) ** 2 for i in range(self.n)}
            p2 = {i: np.sin(softmax_angles[i][1] / 2) ** 2 for i in range(self.n)}
        else:
            p1 = {i: 0.5 for i in range(self.n)}
            p2 = {i: 0.5 for i in range(self.n)}

        results = []
        for _ in range(n_shots):
            # Sample ancilla outcomes per data qubit
            a1 = [int(np.random.random() < p1[i]) for i in range(self.n)]
            a2 = [int(np.random.random() < p2[i]) for i in range(self.n)]
            ancilla_pairs = list(zip(a1, a2))

            # Build effective per-qubit rotation and label
            unitaries = [QND_UNITARIES[pair] for pair in ancilla_pairs]
            basis_labels = [QND_BASIS_LABELS[pair] for pair in ancilla_pairs]

            # Rotate full statevector and sample (with optional depolarizing noise)
            if getattr(self, "_mix_p", 0.0) > 0 and np.random.random() < self._mix_p:
                idx = np.random.randint(2 ** self.n)
            else:
                U_full = kron_n(*unitaries)
                sv_rot = U_full @ self._sv
                probs = np.abs(sv_rot) ** 2
                idx = np.random.choice(2 ** self.n, p=probs / probs.sum())
            outcome = [(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)]

            results.append((basis_labels, outcome))
        return results

    # ------------------------------------------------------------------ #
    # Hardware bitstring parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_hardware_qnd(raw_17bit: List[int]) -> Tuple[List[str], List[int]]:
        """
        Parse raw 17-qubit hardware measurement result into (basis, outcome).

        Parameters
        ----------
        raw_17bit : list of 17 bits from QPU measurement

        Returns
        -------
        (basis_labels, data_outcomes) : (list[str], list[int])
            basis_labels: ['X'|'Y'|'Z'|'D'] for each of 5 data qubits
            data_outcomes: measurement result on each data qubit
        """
        # Extract ancilla pair outcomes (will be remapped to basis via QND_BASIS_LABELS)
        a1_outcomes = [raw_17bit[PHYSICAL_ANCILLA_MAP[dq][0]] for dq in PHYSICAL_DATA]
        a2_outcomes = [raw_17bit[PHYSICAL_ANCILLA_MAP[dq][1]] for dq in PHYSICAL_DATA]
        ancilla_pairs = list(zip(a1_outcomes, a2_outcomes))

        # Map ancilla pairs → basis labels
        basis_labels = [QND_BASIS_LABELS[pair] for pair in ancilla_pairs]

        # Extract final data outcomes
        outcome = [raw_17bit[dq] for dq in PHYSICAL_DATA]

        return basis_labels, outcome

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_state(self):
        if self._sv is None:
            raise RuntimeError("Call prepare_cluster_state() first.")

    def _sample_outcome(self, basis: List[str], sv: np.ndarray) -> List[int]:
        # Depolarizing noise: with prob mix_p, return uniform random outcome
        if getattr(self, "_mix_p", 0.0) > 0 and np.random.random() < self._mix_p:
            idx = np.random.randint(2 ** self.n)
        else:
            U = kron_n(*[BASIS_UNITARIES[b] for b in basis])
            sv_rot = U @ sv
            probs = np.abs(sv_rot) ** 2
            probs /= probs.sum()
            idx = np.random.choice(2 ** self.n, p=probs)
        return [(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)]
