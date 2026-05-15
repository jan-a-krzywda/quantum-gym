quantum-gym (monorepo)
======================

This repository acts as a collection of subprojects ("gyms") for quantum control
experiments, calibration, and shadow tomography. The intention is to develop
features in this monorepo and split mature components into standalone repos
when appropriate.

Subprojects:
- `shadow-gym` - shadow tomography tooling (thin package re-exporting implementation)
- `calibration-gym` - ML and experimental code for calibration and VAE training
- `preparation-gym` - ansatz and circuit preparation utilities

See subproject READMEs for more details.

Note on recent reorganization
-----------------------------
The repository was reorganized into a monorepo layout. The primary ML and
experimental code previously at the project root has been moved under
`calibration-gym/` (subfolders `ML/` and `quantum_code/`). A small set of
top-level shims (`ML`, `quantum_code`) were added to preserve backwards
compatible imports. The original README content was migrated into
`calibration-gym/README.md`.
