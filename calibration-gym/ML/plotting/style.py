"""Shared Matplotlib style for latent and zebra-like figures."""

from __future__ import annotations

import matplotlib


def apply_latent_zebra_style() -> None:
    """Apply a readable, zebra-like style across latent figure outputs."""
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )
