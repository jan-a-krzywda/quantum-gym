# Master Specification: Model-Based Reinforcement Learning (MBRL) for Quantum Control

## 1. Global Architecture & Hyperparameters
This project implements a "Dreamer-style" MBRL pipeline to prepare target quantum states (default: GHZ) on noisy quantum hardware/simulators. The agent navigates a fixed-depth quantum circuit by making small, locally-linear incremental adjustments to the rotation angles.

**Core Constants:**
* `N_QUBITS` = 3
* `CONNECTIVITY` = Linear `[(0, 1), (1, 2)]` (Mapped to native CZ or CNOT topology)
* `TARGET_STATE` = GHZ State `(|000> + |111>) / sqrt(2)` (Configurable)
* `MAX_ROUNDS` = 10 (Maximum depth of the circuit layers)
* `BEAM_WIDTH` = `L` (Number of parallel trajectories to keep during classical search)
* `ANGLE_INCREMENT` = `np.pi / 18` (10 degrees. This is the discrete step size for the agent. Must be configurable).

---

## Phase 1: The Simulator (Classical Pathfinding)
We perform a classical Beam Search to find the "golden trajectories" in the mathematically ideal Hilbert space. This prevents the agent from getting lost in the exponentially large barren plateaus of the maze.

1. **Initialization:** Start strictly from the `|000>` state.
2. **The Ansatz Layer (U_a):** At each round, apply a parameterized layer:
   * Two independent rotations per qubit to allow full complex space exploration (e.g., $R_z(\phi) R_y(\theta)$).
   * Followed by the fixed entangling layer `CZ(0,1)` and `CZ(1,2)`.
3. **The Branching:** * The action space at each step is discrete: `[-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT]` for each of the 6 parameters.
   * Simulate all branches classically.
4. **Pruning:** Calculate exact fidelity against `TARGET_STATE`. Keep only the top `L` (Beam Width) trajectories.
5. **Termination:** Stop when `MAX_ROUNDS` (10) is reached or fidelity > 0.99.

---

## Phase 2: Hardware Fingerprinting & Data Collection
Transition from the noiseless classical math to the noisy physical reality. (Use a noisy simulator with T1/T2 and SPAM errors for V1, then deploy to Quantum Inspire).

1. **Unroll the Trajectories:** Take the best discrete trajectories from Phase 1. 
2. **Intermediate Execution:** For each trajectory $k$, execute the circuit up to round $n$ (where $n \le 10$). 
3. **Shadow Tomography:** Measure the intermediate state using randomized Pauli measurements.
4. **The Fingerprint ($x_{k,n}$):** Reconstruct the 63-parameter density matrix from the classical shadows. This is our raw, noisy observation state: $x_{k,n} = f[\psi(k,n)]$.
5. **Dataset Construction:** Build a dataset of sequential transitions: $(x_{k, n}, a_{k, n}, x_{k, n+1})$.

---

## Phase 3: Map the Maze (VAE Training)
Train a Variational Autoencoder to compress the noisy 63-parameter physical fingerprints into a smooth, low-dimensional manifold.

* **Encoder:** Maps noisy shadow $x_{k,n} \to \mu_{k,n}$ (Latent space). Acts as a denoiser against SPAM errors.
* **Decoder:** Maps $\mu_{k,n} \to \hat{x}_{k,n}$ (Reconstructed shadow) for the loss function.

---

## Phase 4: Learn the Physics (World Model)
Train a Latent Dynamics Model to learn the transition matrix of the quantum state over time.

* **Input:** Current latent state $\mu_n$ and the discrete action taken $a_n$.
* **Output/Prediction:** The next latent state $\mu_{n+1}$.
* **Significance:** Because the agent is taking small, incremental angle steps (`pi/18`), this space will be locally linear, making it highly stable to train.

---

## Phase 5: Train in the Dream (RL Agent)
Train a Reinforcement Learning agent entirely inside the World Model (without touching the quantum simulator).

* **Agent Type:** PPO or DQN with a MultiDiscrete action space.
* **Observation Space:** The continuous latent vector $\mu_n$.
* **Action Space:** Discrete vector of `[-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT]` for each qubit's rotation parameters.
* **Reward:** Decode the current $\mu$ to the 63-parameter shadow. Reward = Fidelity between the decoded shadow and the target GHZ shadow.

---

## Phase 6: Verification & Deployment
1. **Sim2Sim Verification:** Extract the fully trained RL policy. Run it sequentially on the Noisy Simulator to verify it can reliably hit the GHZ state.
2. **Sim2Real Deployment:** Run the trained policy directly on the real hardware (e.g., Quantum Inspire) to prove the pipeline's effectiveness on physical transmon qubits.