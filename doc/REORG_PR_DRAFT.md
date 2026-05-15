Title: chore: reorganize repository into monorepo structure

Summary:
This PR reorganizes the repository into a monorepo layout to better separate
concerns and make future extraction of mature components easier.

Changes:
- Move top-level experimental and ML code into `calibration-gym/ML` and
  `calibration-gym/quantum_code` (history preserved via git mv).
- Add `shadow-gym/` as a thin compatibility package that re-exports
  shadow tomography helpers implemented under `backends/quantum_inspire`.
- Add top-level compatibility shims:
  - `ML/__init__.py` -> re-exports `calibration-gym/ML`
  - `quantum_code/__init__.py` -> re-exports `calibration-gym/quantum_code`
- Add `backends/quantum_inspire/` scaffold (runner, adapter, shadow tomography
  helpers, tests).
- Add/update monorepo docs: `MONOREPO_README.md` and top-level `README.md`.

Rationale:
- Keeps historical layout working for forks and external scripts that still
  import `ML` or `quantum_code` at top-level, while placing the canonical code
  under `calibration-gym/` for clearer organization.

Notes and follow-ups:
- The top-level shims are intended as a short-term compatibility layer. Over
  time we should update imports across the codebase to reference
  `calibration-gym` directly and remove the shims.
- The Quantum Inspire backend scaffold includes a simulator fallback so local
  development doesn't require credentials. A REST SDK fallback and CI gating
  for real-device tests can be added in a follow-up PR.
- Tests for the new backend are included but require `pytest` and `qiskit` to
  run locally.

Do not merge yet
----------------
This branch contains structural changes intended to prepare the repository for
development of the shadow tomography features. Please do not merge into `main`
until:
- The team has reviewed the reorganization and is ready to proceed.
- Basic CI is in place to run unit tests (so we don't accidentally break other
  subprojects during follow-up feature work).

Once the branch is approved and merged, the next major task will be to start
implementing the shadow tomography algorithms inside `backends/quantum_inspire`
and add unit tests that mock the backend responses.

Testing:
- Ran a local smoke commit and verified files were moved with git history.
- Please run tests and lints in CI; unit tests require additional dev
  dependencies.

Reviewer notes:
- To test locally, checkout this branch and run `pytest backends/quantum_inspire/tests`.

