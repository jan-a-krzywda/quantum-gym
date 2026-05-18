# preparation-gym

Proof-of-concept **RL + world-model** pipeline for adaptive quantum state preparation.  
A VAE-based world model is trained on circuit outcomes; a planner (beam search or active inference) proposes the next circuit to reduce preparation error.

## Structure

```
preparation-gym/
└── RL-world-model/
    ├── train_vae.py                # Train VAE world model on circuit outcome data
    ├── train_world_model.py        # Train the transition / latent dynamics model
    ├── train_rl_agent.py           # Train RL agent using the learned world model
    ├── active_inference_planner.py # EFE-based active inference planner
    ├── beam_search.py              # Beam search over circuit action sequences
    ├── generate_dataset.py         # Generate training data from the environment
    ├── mlp_vae.py                  # MLP encoder/decoder for the VAE
    ├── multiqubit_fingerprint.py   # Multi-qubit state fingerprinting utilities
    ├── shadow_fingerprint.py       # Shadow-based state fingerprinting
    └── pipeline_poc.ipynb          # End-to-end proof-of-concept notebook
```

## Quick start

From the repo root:

```bash
# Generate training data
python preparation-gym/RL-world-model/generate_dataset.py

# Train the VAE world model
python preparation-gym/RL-world-model/train_vae.py

# Train the transition model
python preparation-gym/RL-world-model/train_world_model.py

# Train the RL agent
python preparation-gym/RL-world-model/train_rl_agent.py
```

Or run the full pipeline interactively via `pipeline_poc.ipynb`.
