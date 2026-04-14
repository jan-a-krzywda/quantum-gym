"""
Central style settings for zebra GIF readability.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union


Number = Union[int, float]


def zebra_gif_2d_style(n_q: int, n_tau: int) -> Dict[str, Union[Number, Tuple[float, float]]]:
    """
    Return plotting style tuned for notebook/blog visibility.

    Keeps figure dimensions moderate so rendered notebook frames do not shrink labels.
    """
    fig_w = min(9.8, max(7.8, 0.07 * n_tau + 5.0))
    fig_h = min(7.4, max(5.8, 0.15 * n_q + 3.4))
    return {
        "figsize": (fig_w, fig_h),
        "left": 0.08,
        "right": 0.98,
        "top": 0.96,
        "bottom": 0.11,
        "hspace_outer": 0.00,
        "hspace_inner": 0.10,
        "wspace_inner": 0.10,
        "label_fs": 14,
        "tick_fs_main_x": 10,
        "tick_fs_main_y": 10,
        "tick_fs_marg_x": 10,
        "marginal_label_fs": 11,
        "suptitle_fs": 12,
    }
