# Implementation Specification: Active Inference Tomography

**Target Platform:** 17-Qubit Semiconductor Spin Qubit Device  
**Target States:** 1D Cluster State (4-qubit) and 2D Lattice Cluster State (17-qubit)  
**Objective:** Hardware-efficient autonomous characterization using Deferred Measurement (QND + Crot), Active Reset Batching, and "Hardware Softmax."

---

## 1. Hardware Mapping & Topography

### 1.1 4-Qubit Sub-Chain (Ground Truth Benchmark)
Utilized for Act II benchmarking against standard randomized shadows.
* **Data Qubits ($D$):** `Q4, Q8, Q12, Q14` (Linear chain; no SWAPs required).
* **Ancilla Qubits ($A$):** 8 dedicated, non-shared neighbors.
    * $D_0$ (`Q4`) <- `[Q1, Q2]`
    * $D_1$ (`Q8`) <- `[Q5, Q11]`
    * $D_2$ (`Q12`) <- `[Q9, Q15]`
    * $D_3$ (`Q14`) <- `[Q13, Q16]`

### 1.2 17-Qubit 2D Lattice (Scalable Demo)
Utilized for Act I scalability demonstration.
* **Target:** 2D Cluster State (Graph State) defined by nearest-neighbor **CZ** bonds across the 17-qubit surface code grid.
* **Stabilizers ($K_i$):** Bulk qubits target weight-5 operators $K_i = X_i \bigotimes_{n \in \text{neigh}(i)} Z_n$.

---

## 2. Quantum Protocols & Engineering Tricks

### 2.1 Engineering Trick 1: Active Reset Batching ($N_{batch} \approx 20$)
To bypass classical compilation queues and network latency:
* **Protocol:** Multiple Active Inference basis choices are concatenated into a single execution job.
* **Mechanism:** High-fidelity **active resets** are implemented between state preparations.
* **Constraint:** The current control stack supports approximately **20 concatenated circuits** per shot. The classical agent must package its Softmax requests into batches of this size to optimize hardware throughput.

### 2.2 Engineering Trick 2: Hardware Softmax (QND + Crot)
The hardware "samples" the Active Inference distribution through on-chip entropy.
1. **Bias Ancillas:** Initialize $A_1, A_2$ with rotation angles $\theta_1, \theta_2$ representing the Softmax probabilities $P(X), P(Y), P(Z)$.
2. **QND Readout:** Mid-circuit measurement collapses ancillas into $\{00, 01, 10, 11\}$.
3. **Quantum Feed-forward:**
    * Apply $CR_y(-\pi/2)$ (Control: $A_1$, Target: $D_i$).
    * Apply $CR_x(\pi/2)$ (Control: $A_2$, Target: $D_i$).
4. **Basis Mapping:** `00` -> **Z** | `10` -> **X** | `01` -> **Y** | `11` -> **Diagonal** $(X-Y+Z)/\sqrt{3}$.

---

## 3. The Active Inference Engine

### 3.1 Software Architecture
* **`QuantumEnvironment`:** Handles circuit construction for 12-qubit (benchmarking) and 17-qubit (2D lattice) variants. Simulates/Executes **QND + Crot** logic.
* **`ShadowProcessor`:** Parses bitstrings and implements the inverse depolarizing channel $\mathcal{M}^{-1}$ (including the diagonal outcome).
* **`ActiveInferenceAgent`:** Maintains Gaussian beliefs $(\mu_i, \sigma_i^2)$ for target stabilizers.

### 3.2 Utility & Weighting Factor ($w_i$)
The agent maximizes Expected Free Energy (EFE) via:
$$U(B) = \sum_{i \in \text{diag}(B)} w_i \cdot \Delta \sigma_i^2$$
$$w_i = |\mu_i| \cdot \alpha^{|w_i|} \cdot \chi_i$$

* **Locality ($\alpha$):** $\alpha \approx 2.0$. Prioritizes high-weight operators (e.g., weight-5 lattice stabilizers).
* **Magnitude ($|\mu_i|$):** Focuses on confirmed signal (expectation values approaching 1 or -1).
* **Relevance ($\chi_i$):** $\chi_i = 50.0$ for stabilizers ($ZXZ$ or $X \cdot Z^{\otimes 4}$), $1.0$ for other Paulis.

---

## 4. Implementation Scaffold (Qiskit)

```python
import numpy as np
from qiskit import QuantumCircuit

def build_qnd_shadow_circuit(data_qubits, ancilla_map, softmax_angles=None):
    """
    Builds the QND + Crot circuit. 
    softmax_angles: dict mapping data_qubit index to (theta1, theta2)
    """
    qc = QuantumCircuit(17, 17)
    
    # 1. State Preparation (Cluster State)
    for dq in data_qubits:
        qc.h(dq)
    # CZ gates along the chain or lattice connectivity
    for i in range(len(data_qubits)-1):
        qc.cz(data_qubits[i], data_qubits[i+1])
    qc.barrier()
    
    # 2. Hardware Softmax Prep (Biased Ancilla initialization)
    for dq, ancillas in ancilla_map.items():
        # Softmax angles provided by ActiveInferenceAgent
        t1, t2 = softmax_angles[dq] if softmax_angles else (np.pi/2, np.pi/2)
        qc.ry(t1, ancillas[0])
        qc.ry(t2, ancillas[1])
    qc.barrier()
    
    # 3. Feed-forward Logic (Crot)
    # Maps ancilla Z-basis collapse to Data rotations
    for dq, ancillas in ancilla_map.items():
        qc.cry(-np.pi/2, ancillas[0], dq)
        qc.crx(np.pi/2, ancillas[1], dq)
    
    # 4. Readout
    for dq, ancillas in ancilla_map.items():
        # Ancillas store the basis choice; Data stores the outcome
        qc.measure([ancillas[0], ancillas[1], dq], [ancillas[0], ancillas[1], dq])
    
    return qc