# ML Workflow Structure

## Folders

- `training/`: model training entrypoints.
- `analysis_processing/`: data processing, dynamics analysis, and simulation.
- `plotting/`: plotting and visualization entrypoints.
- `dynamics/`: deprecated compatibility wrappers to the new folders.

## Canonical Entrypoints

- `training/train_vae_model.py` -> forwards to `train_vae.py`
- `analysis_processing/generate_latent_dynamics.py` -> forwards to `analyze_latent_dynamics.py`
- `analysis_processing/generate_synthetic_zebra.py` -> forwards to `generate_vae_gif.py`
- `analysis_processing/simulate_latent_zebra.py` -> forwards to `simulate_fitted_latent_zebra.py`
- `plotting/plot_latent_dynamics.py` -> forwards to `plot_latent_dynamics_gif.py`
- `plotting/plot_latent_from_checkpoint.py` -> forwards to `make_latent_gif.py`

## Data IO Naming

- `fid_data_io.py` is the preferred data module name.
- `data.py` is kept as a compatibility shim.
