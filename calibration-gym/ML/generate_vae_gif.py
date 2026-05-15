#!/usr/bin/env python3
"""Legacy wrapper; use `ML.analysis_processing.generate_synthetic_zebra` instead."""

from __future__ import annotations

import sys
from pathlib import Path

_FIRST_TESTS = Path(__file__).resolve().parent.parent
if str(_FIRST_TESTS) not in sys.path:
    sys.path.insert(0, str(_FIRST_TESTS))

from ML.analysis_processing.generate_synthetic_zebra import main


if __name__ == "__main__":
    main()
