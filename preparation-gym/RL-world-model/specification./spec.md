# Master Specification: Active Inference & Model-Based Planning for Quantum Control

## 1. Global Architecture & Hyperparameters
This project implements a Model Predictive Control (MPC) pipeline driven by Active Inference to prepare target quantum states (default: GHZ) on noisy quantum hardware. The system completely avoids sample-inefficient Reinforcement Learning (RL), opting instead for classical planning inside a learned, uncertainty-aware latent world model.

**Core Constants:**
* `N_QUBITS` = 3
* `CONNECTIVITY` = Linear `[(0, 1), (1, 2)]` (Mapped to native CZ or CNOT topology)
* `TARGET_STATE` = GHZ State `(|000> + |111>) / sqrt(2)` (Configurable)
* `MAX_ROUNDS` = 20 (Maximum depth of the circuit layers)
* `BEAM_WIDTH` = `L` (Number of parallel trajectories to keep during classical search)
* `ANGLE_INCREMENT` = `np.pi / 18` (10 degrees. This is the discrete step size for the agent. Must be configurable).

---

## Phase 1: The Simulator (Backward Disentanglement Beam Search)
We perform a classical Beam Search in the noiseless simulator to find the "golden trajectories" mathematically. To avoid barren plateaus, we search *backwards* by destroying entanglement.

1. **Initialization:** Start the classical simulator at the `TARGET_STATE` (GHZ).
2. **The Inverse Ansatz Layer:** At each step, apply the inverse of the forward layer (e.g., $CZ$ followed by $R_y(-\theta)$).
3. **Branching:** Action space is discrete `[-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT]` for each parameter.
4. **Scoring Metric (Purity):** Score branches by the purity of the reduced density matrices: `Tr(rho_0^2) + Tr(rho_1^2) + Tr(rho_2^2)`.
5. **Termination & Inversion:** Stop when purity > `2.99` (state is fully separable). Invert the action sequence and flip the angle signs to get the exact forward preparation circuit from `|000>`.

---

## Phase 2: Map the Maze (VAE Training on Simulated Data)
Train a Variational Autoencoder purely on the ideal, simulated states to establish a clean geometric map of the Hilbert space.

1. **Dataset:** Use intermediate statevectors from the Phase 1 simulated trajectories. Convert them into 63-parameter classical shadow fingerprints.
2. **Encoder:** Maps shadow $x \to \mu$ (Latent space). 
3. **Decoder:** Maps $\mu \to \hat{x}$ (Reconstructed shadow). 
4. *Crucial:* Because this is trained on noiseless data, the latent space $\mu$ learns the true, smooth manifold without hardware noise distorting it. The VAE is then frozen.

---

## Phase 3: Sparse Hardware Runs (The Reality Check)
Bridge the sim-to-real gap by running a small subset of trajectories on real quantum hardware.

1. Take the top 3-5 trajectories from Phase 1.
2. Execute them step-by-step on the real device (or noisy simulator).
3. Measure the noisy shadow fingerprints.
4. Pass them through the **frozen VAE encoder** to get $\mu_{real}$. 
5. Build a dataset of real hardware transitions: $(\mu_{t}, a_t, \mu_{t+1})$.

---

## Phase 4: Learn the Physics (Stochastic World Model)
Train a Latent Dynamics Model to learn the transition matrix of the quantum state, explicitly modeling the sim-to-real discrepancy via uncertainty.

* **Input:** Current latent state $\mu_t$ and discrete action $a_t$.
* **Output:** The predicted next latent state $\mu_{t+1}$ AND an epistemic uncertainty estimate ($\sigma^2_{t+1}$ or log-variance).
* **Training:** Train on the mixed dataset (simulated + real). The model learns to predict low variance in regions where sim and real align, and high variance where the hardware behavior deviates from simulation.

---

## Phase 5: Active Inference Planner (Hypothesis Testing)
Perform classical tree search / beam search over the World Model. The planner balances Pragmatic Value (reaching GHZ) with Epistemic Value (seeking/avoiding uncertainty) by minimizing Expected Free Energy (EFE).

1. **Simulation Phase:** Start at $\mu_0$. Evaluate branching discrete actions using the World Model.
2. **Scoring:** For each path, calculate the Active Inference Score:
   * `Score = Expected_Fidelity - (Penalty * Accumulated_Uncertainty)`
3. **Hypothesis Trigger:** If the predicted uncertainty $\sigma^2$ for a step crosses a safety threshold, the region is deemed "unknown fog".
   * The planner halts that branch.
   * It proposes the partial trajectory leading up to that point as a **"Hypothesis Test"** to be executed on real hardware.
4. **Output:** Returns either an exploitation trajectory (tag: `"execute"`) or an exploration trajectory (tag: `"verify"`).

---

## Phase 6: The Outer MPC Loop (Continuous Adaptation)
Deploy the agent in a closed loop with the quantum hardware.

1. Ask the Active Inference Planner for a trajectory.
2. **If `"execute"`:** The model is confident. Run the full trajectory on hardware to prepare the GHZ state. Loop finishes.
3. **If `"verify"`:** The model hit an uncertainty boundary. 
   * Run the short hypothesis trajectory on the real hardware.
   * Collect the resulting noisy shadow and encode it to $\mu_{real}$.
   * Add $(\mu_t, a_{verify}, \mu_{real})$ to the dataset.
   * Fine-tune the World Model. The uncertainty in this region instantly collapses.
   * Go back to Step 1 and re-plan.