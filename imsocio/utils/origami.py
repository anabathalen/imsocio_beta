"""Core ORIGAMI/TWIM processing helpers.

Pure functions for ORIGAMI-style data processing that can be imported and unit tested.
"""
from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.signal import savgol_filter


def safe_float_conversion(value) -> float:
    """Safely convert a value to float.

    Rules:
    - numbers -> float
    - strings -> float if parseable else 0.0; treat "", "nan", "null", "none" (case-insensitive) as 0.0
    - iterables -> first successfully converted element, otherwise 0.0
    - others -> 0.0
    
    Args:
        value: Value to convert
        
    Returns:
        Converted float value or 0.0 if conversion fails
    """
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if s == "" or s.lower() in {"nan", "null", "none"}:
                return 0.0
            return float(s)
        if hasattr(value, "__iter__") and not isinstance(value, str):
            for item in value:
                try:
                    return safe_float_conversion(item)
                except Exception:
                    continue
            return 0.0
        return 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


def remove_duplicate_values(values: Iterable[float], tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Remove near-duplicate values, keeping the first occurrence.
    
    Args:
        values: Iterable of float values
        tolerance: Tolerance for considering values as duplicates
        
    Returns:
        Tuple of (clean_values, removed_indices)
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return arr, np.array([], dtype=int)

    unique_mask = np.ones(arr.shape[0], dtype=bool)
    for i in range(1, arr.shape[0]):
        # check against previous kept values
        for j in range(i):
            if unique_mask[j] and abs(arr[i] - arr[j]) < tolerance:
                unique_mask[i] = False
                break
    removed = np.where(~unique_mask)[0]
    return arr[unique_mask], removed


def interpolate_matrix(
    ccs_values: np.ndarray,
    trap_cv_values: np.ndarray,
    intensity_matrix: np.ndarray,
    method: str = "linear",
    multiplier: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a CCS×TrapCV matrix on a finer regular grid.

    Inputs must be strictly increasing 1D arrays and a 2D matrix with shape
    (len(ccs_values), len(trap_cv_values)).
    
    Args:
        ccs_values: 1D array of CCS values (strictly increasing)
        trap_cv_values: 1D array of trap CV values (strictly increasing)
        intensity_matrix: 2D intensity matrix
        method: Interpolation method ('linear' or 'cubic')
        multiplier: Factor to increase grid resolution (1 = no change)
        
    Returns:
        Tuple of (ccs_new, trapcv_new, matrix_new). If multiplier == 1, returns inputs.
        
    Raises:
        ValueError: If inputs are invalid or not strictly increasing
    """
    if multiplier <= 1:
        return ccs_values, trap_cv_values, intensity_matrix

    ccs_values = np.asarray(ccs_values, dtype=float)
    trap_cv_values = np.asarray(trap_cv_values, dtype=float)
    Z = np.asarray(intensity_matrix, dtype=float)

    if ccs_values.ndim != 1 or trap_cv_values.ndim != 1:
        raise ValueError("ccs_values and trap_cv_values must be 1D arrays")
    if Z.shape != (ccs_values.size, trap_cv_values.size):
        raise ValueError("intensity_matrix shape must match (len(ccs_values), len(trap_cv_values))")
    if np.any(np.diff(ccs_values) <= 0) or np.any(np.diff(trap_cv_values) <= 0):
        raise ValueError("ccs_values and trap_cv_values must be strictly increasing")

    n_ccs_new = int(ccs_values.size) * int(multiplier)
    n_trap_new = int(trap_cv_values.size) * int(multiplier)

    ccs_new = np.linspace(ccs_values.min(), ccs_values.max(), n_ccs_new)
    trap_new = np.linspace(trap_cv_values.min(), trap_cv_values.max(), n_trap_new)

    if method == "linear":
        interp_func = interpolate.RegularGridInterpolator(
            (ccs_values, trap_cv_values), Z, method="linear", bounds_error=False, fill_value=0.0
        )
        T, C = np.meshgrid(trap_new, ccs_new)
        pts = np.column_stack([C.ravel(), T.ravel()])
        Z_new = interp_func(pts).reshape(n_ccs_new, n_trap_new)
    elif method == "cubic":
        kx = min(3, max(1, ccs_values.size - 1))
        ky = min(3, max(1, trap_cv_values.size - 1))
        spline = interpolate.RectBivariateSpline(ccs_values, trap_cv_values, Z, kx=kx, ky=ky)
        Z_new = spline(ccs_new, trap_new)
    else:
        raise ValueError("method must be 'linear' or 'cubic'")

    return ccs_new, trap_new, Z_new


def smooth_matrix_gaussian(Z: np.ndarray, sigma: float = 1.0, truncate: float = 4.0) -> np.ndarray:
    """Apply Gaussian smoothing to a 2D matrix.
    
    Args:
        Z: 2D numpy array to smooth
        sigma: Standard deviation for Gaussian kernel
        truncate: Truncate filter at this many standard deviations
        
    Returns:
        Smoothed 2D array
    """
    return ndimage.gaussian_filter(np.asarray(Z, dtype=float), sigma=sigma, truncate=truncate)


def smooth_matrix_savgol(
    Z: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "mirror",
) -> np.ndarray:
    """Apply Savitzky–Golay smoothing to rows then columns.
    
    Args:
        Z: 2D numpy array to smooth
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial used to fit samples
        mode: Padding mode for edge handling
        
    Returns:
        Smoothed 2D array
    """
    Z = np.asarray(Z, dtype=float).copy()
    n_rows, n_cols = Z.shape

    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2 if polyorder is not None else 5)

    # Smooth rows
    if n_cols >= window_length:
        for i in range(n_rows):
            Z[i, :] = savgol_filter(Z[i, :], window_length=window_length, polyorder=polyorder, mode=mode)
    
    # Smooth columns
    if n_rows >= window_length:
        for j in range(n_cols):
            Z[:, j] = savgol_filter(Z[:, j], window_length=window_length, polyorder=polyorder, mode=mode)

    return Z
