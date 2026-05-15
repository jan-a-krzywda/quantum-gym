"""Compatibility shim for top-level `ML` package.

This file makes `import ML` continue to work after the ML/ tree was moved into
`calibration-gym/ML` by adding that path to this package's __path__.
"""
from pathlib import Path
import sys

# Insert the moved ML package location at the front of the package search path
repo_root = Path(__file__).resolve().parents[1]
ml_path = str(repo_root / "calibration-gym" / "ML")
if ml_path not in __path__:
    __path__.insert(0, ml_path)
