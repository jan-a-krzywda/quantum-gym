# quantum-gym (monorepo)# Quantum Gym



This repository is a collection of small "gyms" for quantum control, calibration,Quantum + ML workflow for creating an efficient low-dimensional representation of a noisy quantum system. This digital twin approach can be used for control, monitoring, and diagnostics of the system. 

and shadow tomography. The monorepo layout groups related experiments and

tools while keeping individual components easy to extract later.## What this repo does



Top-level structure- Trains a VAE model on qubit/tau readout data.

-------------------- Analyzes latent dynamics across shots and qubits.

- `calibration-gym/` — Primary ML and experiment pipelines. Contains:- Generates synthetic zebra outputs from sampled or simulated latents.

  - `ML/` — VAE training, latent-dynamics analysis, plotting and utilities.- Produces visual outputs (GIFs/plots) for quick inspection.

  - `quantum_code/` — Circuit construction, data acquisition notebooks and

    experiment helpers.## Demo GIFs

  - `README_ROOT.md` — original root README (kept for history).

- `preparation-gym/` — Ansatze, circuit preparation utilities and RL/world-modelAdd generated GIFs to `docs/gifs/` with the names below so they render in this README:

  experiments (including `RL-world-model`).

- `shadow-gym/` — Thin compatibility package re-exporting shadow tomography### Running average of qubit response

  helpers implemented under `backends/quantum_inspire`.![Synthetic zebra evolution](docs/gifs/real_data.gif)

- `backends/quantum_inspire/` — Backend adapter and shadow tomography helpers

  with a simulator fallback and unit tests.### Latent dynamics animation (comparison across two days)

- `RL-world-model/` — proof-of-concept pipelines and notebooks for using a![Latent dynamics animation](docs/gifs/two_days.gif)

  VAE + planner for active calibration.

- `docs/` — repository-level documentation and GIFs used by READMEs.### Generated artificial data

![Simulated fitted latent zebra](docs/gifs/sim_data.gif)

Quick overview of important folders

----------------------------------## How to run

- `calibration-gym/ML/`

  - Training: `training/train_vae_model.py` — train VAE on readout data.Run from repo root:

  - Analysis: `analysis_processing/*.py` — extract latent dynamics, generate

    synthetic zebra sequences, and produce visualizations.```bash

  - Model: `vae_model.py` — VAE model used across experiments.cd /Users/krzywdaja/Documents/quantum-gym

```

- `calibration-gym/quantum_code/`

  - Experiment notebooks and small scripts used for data collection andTrain model:

    device-specific helpers.

```bash

- `backends/quantum_inspire/`python ML/training/train_vae_model.py

  - `runner.py`, `adapter.py` — adapter to submit circuits to Quantum Inspire```

    (SDK preferred) and a local simulator fallback.

  - `shadow_tomography.py` — measurement-counts → Pauli expectation helpers.Analyze latent dynamics:

  - tests that mock responses so unit tests run offline.

```bash

- `preparation-gym/` and `RL-world-model/`python ML/analysis_processing/generate_latent_dynamics.py

  - Contain planning, beam search, and world-model experiments. Notebooks```

    here show PoC workflows and are useful references.

Generate synthetic zebra:

Using the repository

--------------------```bash

Clone and work from the repo root. A minimal example to run VAE training:python ML/analysis_processing/generate_synthetic_zebra.py --latent-mode iid

```

```bash

cd /path/to/quantum-gymSimulate fitted latent zebra:

python calibration-gym/ML/training/train_vae_model.py

``````bash

python ML/analysis_processing/simulate_latent_zebra.py --sim-mode ou

Run a small import smoke test (CI runs this automatically):```


```bash
python -c "import ML, quantum_code, shadow_gym; print('imports ok')"
```

Development notes
-----------------
- The repository was recently reorganized (see `MONOREPO_README.md`).
- Top-level shim packages `ML` and `quantum_code` exist to preserve
  backwards-compatibility. They are short-lived: prefer importing from
  `calibration-gym.*` in new code.
- A minimal CI workflow runs a lightweight import test. More CI jobs (lint,
  unit tests for heavy deps) can be added later.

Contributing
------------
- Open issues and PRs against `main`. For large refactors, prefer a feature
  branch and include tests.

License & contact
-----------------
See `LICENSE` for licensing information and the repo owner for contact.

If you'd like, I can also add a per-subproject README summary (short
examples for common tasks). Want me to add those next?
