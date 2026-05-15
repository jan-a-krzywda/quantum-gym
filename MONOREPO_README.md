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
