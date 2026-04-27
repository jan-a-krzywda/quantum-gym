# Architecture Update: Reverse Compiling via Targeted Disentanglement

**Context for Claude:** We are replacing the Forward Fidelity Beam Search with a Backward Disentanglement Beam Search to avoid barren plateaus. We will search for a path that destroys entanglement, then invert it to find the state-prep circuit.

## 1. Initialization (Backward Search)
* **Start State:** Initialize the classical simulator at the target state (e.g., `|GHZ> = (|000> + |111>) / sqrt(2)`).

## 2. The Scoring Metric (Purity)
* **Remove Fidelity:** Do not use state fidelity against the target as the pruning metric.
* **New Metric (Max Purity):** At each branch, calculate the reduced density matrix for Q0, Q1, and Q2 ($\rho_0, \rho_1, \rho_2$).
* **Score Formula:** `score = Tr(rho_0^2) + Tr(rho_1^2) + Tr(rho_2^2)`
* **Goal:** Maximize this score. The maximum possible score is `3.0`.

## 3. The Disentangling Ansatz Layer
* Continue using the discrete/incremental action space defined previously.
* At each step, apply the inverse of our forward layer: Apply $CZ(0,1)$ and $CZ(1,2)$, followed by single-qubit rotations $R_z(-\phi) R_y(-\theta)$.

## 4. Termination & Inversion
* **Early Stop:** If the Purity score reaches `> 2.99`, the state is fully separable into $|\psi_0\rangle \otimes |\psi_1\rangle \otimes |\psi_2\rangle$.
* **The Trivial Rotation:** Calculate the Bloch angles of the separable qubits and apply the trivial single-qubit rotations required to rotate them to $|000\rangle$.
* **The Forward Compilation:** Take the full history of backward actions, invert their sequence, and flip the signs of the angles. This sequence is now the guaranteed exact forward preparation circuit: $U_{GHZ}|000\rangle = |GHZ\rangle$.

## 5. Handoff to Phase 2
* Take this discovered forward sequence and execute it step-by-step to generate the hardware shadows (Phase 2), as originally planned.