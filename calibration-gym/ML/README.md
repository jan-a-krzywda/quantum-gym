# Digital Twin ML Workflow

This folder contains training, analysis, simulation, and plotting scripts for the VAE-based digital twin.

## Recommended layout per run

Each run directory under `ML/runs/<run_name>/` should use:

- `checkpoints/` - model weights (`vae_checkpoint.pt`)
- `figures/` - PNG/GIF outputs
- `data/` - generated `.pkl` / `.npz` data artifacts
- `reports/` - JSON summaries/metadata

The scripts now default to this layout (with fallback to legacy paths).

## 1) Train VAE

```bash
cd /Users/krzywdaja/Documents/quantum-gym
PYTHONPATH=. python ML/train_vae.py
```

Outputs (default run `fid_job_memory_noreset_large`):

- `runs/fid_job_memory_noreset_large/checkpoints/vae_checkpoint.pt`
- `runs/fid_job_memory_noreset_large/figures/latent_plot.png`

## 2) Analyze latent dynamics (means, drift covariance, correlation time)

```bash
PYTHONPATH=. python ML/analyze_latent_dynamics.py
```

Outputs:

- `reports/latent_dynamics.json`
- `data/latent_dynamics_arrays.npz`
- `figures/latent_drift_covariance.png`
- `figures/latent_drift_correlation.png`
- `figures/latent_drift_cov_corr_panel.png`
- `figures/latent_mu_bar_means_2d.png`
- `figures/latent_mu_bar_chip_spatial.png` (for 17-qubit runs)

Useful flags:

- `--offdiag-cov-scale 10`
- `--drift-plot-dpi 150`
- `--no-drift-plots`

## 3) Generate synthetic zebra from iid latent sampling

```bash
PYTHONPATH=. python ML/generate_vae_gif.py --latent-mode iid
```

Outputs:

- `data/vae_synthetic_fid_memory.pkl`
- `figures/<prefix>_*.gif/.png/.npz`

## 4) Simulate from fitted latent dynamics then decode zebra

```bash
PYTHONPATH=. python ML/simulate_fitted_latent_zebra.py --sim-mode ou
```

Outputs:

- `reports/sim_fitted_latent_meta.json`
- `data/sim_fitted_latent_mu_s_q.npz`
- `data/sim_fitted_fid_memory.pkl`
- `figures/<prefix>_*.gif/.png/.npz`

## 5) Latent trajectory GIFs

```bash
PYTHONPATH=. python ML/make_latent_gif.py --run-dir ML/runs/fid_job_memory_noreset_large
PYTHONPATH=. python ML/plot_latent_dynamics_gif.py --source encoded
```

Outputs go to `figures/`.

## Notes

- Legacy runs (files directly in run root) are still supported by fallback path logic.
- For clean commits, keep only the artifacts you need and remove large intermediate GIFs.
