# quantum-gym

A monorepo of quantum ML experiments grouped into three self-contained **gyms**.  
Each gym targets a different stage of the quantum control pipeline.

---

## Gyms

### `calibration-gym/`
Builds a **digital twin** of a noisy quantum device using a Variational Autoencoder trained on raw readout data.  
The latent space captures the low-dimensional structure of qubit/tau responses, enabling state monitoring, drift detection, and generation of synthetic calibration sequences.

Key components: VAE training (`ML/training/`), latent-dynamics analysis (`ML/analysis_processing/`), and device experiment notebooks (`quantum_code/`).

---

### `preparation-gym/`
Proof-of-concept **RL + world-model** pipeline for adaptive state preparation.  
A VAE-based world model is trained on circuit outcomes; a learned planner (beam search / active inference) proposes the next circuit to reduce preparation error.

Key components: world-model training, RL agent, beam search planner, and fingerprint utilities (`RL-world-model/`).

---

### `shadow_gym/`
**Hardware-softmax shadow tomography** on Quantum Inspire (Tuna-17).  
Embeds the random-basis selection directly into the quantum circuit via ancilla mid-circuit measurements, eliminating the classical-quantum round-trip bottleneck.  
Includes simulation validation, density-matrix reconstruction via classical shadows, bootstrap convergence analysis, and an EFE-based active inference loop for adaptive basis selection.

Key components: quantum environment + shadow processor (`src/`), merged simulation + hardware notebook (`notebooks/shadow_tomography_full.ipynb`).

---

## Repository layout

```
quantum-gym/
├── calibration-gym/
│   ├── ML/                  # VAE, latent dynamics, plotting
│   └── quantum_code/        # Device notebooks and experiment helpers
├── preparation-gym/
│   └── RL-world-model/      # World model, RL agent, beam search
├── shadow_gym/
│   ├── src/                 # QuantumEnvironment, ShadowProcessor, utils
│   └── notebooks/           # shadow_tomography_full.ipynb
└── docs/                    # Repository-level documentation
```
