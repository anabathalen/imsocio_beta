"""Utility functions for IMSocio library.

This module contains general-purpose utilities used throughout the package.
"""

from .data_tools import (
    gaussian,
    r_squared,
    fit_gaussian_with_retries,
)

from .origami import (
    safe_float_conversion,
    remove_duplicate_values,
    interpolate_matrix,
    smooth_matrix_gaussian,
    smooth_matrix_savgol,
)

__all__ = [
    # data_tools
    'gaussian',
    'r_squared',
    'fit_gaussian_with_retries',
    # origami
    'safe_float_conversion',
    'remove_duplicate_values',
    'interpolate_matrix',
    'smooth_matrix_gaussian',
    'smooth_matrix_savgol',
]
