#!/usr/bin/env python3
"""Descriptive training entrypoint for the VAE model."""

from __future__ import annotations

import sys
from pathlib import Path

_FIRST_TESTS = Path(__file__).resolve().parents[2]
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

from ML.train_vae import main


if __name__ == "__main__":
    main()
