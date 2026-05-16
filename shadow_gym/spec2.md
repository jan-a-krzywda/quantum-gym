# Master Specification: Active Inference for Scalable Quantum Tomography

**Target Platform:** 17-Qubit Semiconductor Spin Qubit Device  
**Narrative Flow:** 5-Qubit Exact Tracking -> 17-Qubit Neural Scaling -> QND Hardware Softmax

---

## 1. Hardware Topography & Target States

### 1.1 Act I: The 5-Qubit Ground Truth Chain
Used to prove the algorithmic sample-efficiency advantage using exact Bayesian tracking.
* **Target:** 1D Cluster State.
* **Data Qubits ($D$):** `Q2, Q4, Q8, Q12, Q14` (Native zig-zag chain, 0 SWAPs for state prep).
* **Ancilla Mapping (For Act III):** * `Q2` -> `[Q0, Q1]` (Requires 1 SWAP: Q0 <-> Q1)
    * `Q4` -> `[Q7, Q10]` (Requires 1 SWAP: Q7 <-> Q10)
    * `Q8` -> `[Q11, Q5]` (0 SWAPs, both directly connected)
    * `Q12` -> `[Q9, Q6]` (Requires 1 SWAP: Q9 <-> Q6)
    * `Q14` -> `[Q16, Q15]` (Requires 1 SWAP: Q16 <-> Q15)

### 1.2 Act II: The 17-Qubit 2D Lattice
Used to prove scalability utilizing a generative Neural Network Quantum State (NQS).
* **Target:** 2D Cluster State (Graph State) across the full device grid.
* **Stabilizers:** Bulk qubits target weight-5 operators $K_i = X_i \bigotimes_{n \in \text{neigh}(i)} Z_n$.

---

## 2. Software Architecture: Two-Tiered Active Inference

### 2.1 Tier 1: Fast Exact Tracker (5 Qubits)
* **Mechanism:** Tracks exact Gaussian beliefs $(\mu_i, \sigma_i^2)$ for Pauli strings.
* **Optimization:** BLAS-optimized matrix-vector multiplication (`np.dot`).
* **Utility (EFE):** $$U(B) = \sum_{i \in \text{diag}(B)} w_i \cdot \Delta \sigma_i^2$$
  $$w_i = |\mu_i| \cdot \alpha^{|w_i|} \cdot \chi_i$$
  *(Where $\alpha \approx 2.0$, and $\chi_i$ boosts target stabilizers).*

### 2.2 Tier 2: Generative RNN World Model (17 Qubits)
* **Mechanism:** Because $3^{17}$ bases cannot be explicitly tracked, the state is modeled by an Autoregressive Neural Network.
* **Training:** Supervised Maximum Likelihood Estimation (MLE) on the classical shadow replay buffer.
* **Utility (EFE):** Evaluated by asking the RNN to generate $K$ synthetic samples for a candidate basis. Bases that produce high variance in the target stabilizers across the synthetic samples are selected for the next hardware batch.

---

## 3. The Hardware Protocol: "Hardware Softmax" (Act III)

This protocol physically embeds the classical Active Inference distribution (from Tier 1 or Tier 2) into the quantum hardware to bypass classical feed-forward queue latency.

### Step-by-Step Execution
1. **Bias Ancillas:** Initialize physical ancillas $A_1, A_2$ with rotation angles $\theta_1, \theta_2$ derived from the AI's Softmax distribution: $P(X), P(Y), P(Z)$.
2. **QND Readout:** Mid-circuit measurement collapses the ancillas into $\{00, 01, 10, 11\}$.
3. **Routing SWAPs:** Execute the 4 required layout SWAPs (defined in 1.1) to move next-nearest-neighbor ancillas into coupling position. *(Note: Because ancillas are already measured, this moves classical pointers without degrading data coherence).*
4. **Deferred Feed-forward (\texttt{Crot}):**
    * Apply $CR_y(-\pi/2)$ (Control: $A_1$, Target: $D_i$).
    * Apply $CR_x(\pi/2)$ (Control: $A_2$, Target: $D_i$).
5. **Basis Mapping:** `00` -> $Z$ | `10` -> $X$ | `01` -> $Y$ | `11` -> Diagonal.