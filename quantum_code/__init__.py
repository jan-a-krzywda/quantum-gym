"""Compatibility shim for top-level `quantum_code` package.

This file keeps existing imports working after the folder was moved to
`calibration-gym/quantum_code` by adjusting the package search path.
"""
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
qc_path = str(repo_root / "calibration-gym" / "quantum_code")
if qc_path not in __path__:
    __path__.insert(0, qc_path)
