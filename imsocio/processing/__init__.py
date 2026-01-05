"""
Processing module for calibrated and scaled drift time data.

This module handles:
- Baseline fitting and integration
- Mass spectrum processing
- ATD normalization and scaling
- Calibration data matching
- ORIGAMI-style aIMS/CIU processing and visualization
"""

from typing import Tuple
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import integrate
from sklearn.linear_model import LinearRegression

# Import ORIGAMI classes and helper functions
from .origami import (
    ORIGAMISettings,
    ORIGAMIDataProcessor,
    ORIGAMIVisualizer,
    safe_float_conversion,
    remove_duplicate_values,
    interpolate_matrix,
    smooth_matrix_gaussian,
    smooth_matrix_savgol,
)

# Import ESIProt classes
from .esiprot import (
    DeconvolutionResult,
    MZCalculation,
    ESIProtCalculator,
    ESIProtDataExporter
)


# Constants
PROTON_MASS = 1.007276


def fit_baseline_and_integrate(
    x: np.ndarray, 
    y: np.ndarray, 
    integration_range: Tuple[float, float]
) -> Tuple[float, np.ndarray]:
    """
    Fit a linear baseline and integrate the peak above the baseline.
    
    This function:
    1. Identifies baseline points from the edges of the integration range
    2. Fits a linear baseline model
    3. Integrates the area above the baseline
    
    Parameters:
        x: m/z values
        y: intensity values
        integration_range: (min_mz, max_mz) for integration
        
    Returns:
        Tuple of (area, baseline):
            - area: integrated area above baseline
            - baseline: fitted baseline values for plotting
            
    Example:
        >>> x = np.array([100, 101, 102, 103, 104])
        >>> y = np.array([10, 15, 20, 15, 10])
        >>> area, baseline = fit_baseline_and_integrate(x, y, (100, 104))
        >>> print(f"Area: {area:.2f}")
    """
    # Get data within integration range
    mask = (x >= integration_range[0]) & (x <= integration_range[1])
    if np.sum(mask) < 3:
        return 0.0, np.zeros_like(y)
    
    x_region = x[mask]
    y_region = y[mask]
    
    # Use the first and last 10% of points to fit baseline
    n_points = len(x_region)
    baseline_fraction = max(0.1, 3.0 / n_points)  # At least 10% or 3 points
    n_baseline = max(3, int(n_points * baseline_fraction))
    
    # Get baseline points from edges
    baseline_x = np.concatenate([x_region[:n_baseline], x_region[-n_baseline:]])
    baseline_y = np.concatenate([y_region[:n_baseline], y_region[-n_baseline:]])
    
    # Fit linear baseline
    try:
        baseline_model = LinearRegression()
        baseline_model.fit(baseline_x.reshape(-1, 1), baseline_y)
        
        # Calculate baseline for entire range
        baseline_region = baseline_model.predict(x_region.reshape(-1, 1))
        
        # Ensure baseline doesn't go above the signal
        baseline_region = np.minimum(baseline_region, y_region)
        
        # Calculate area above baseline using trapezoidal rule
        corrected_y = y_region - baseline_region
        corrected_y = np.maximum(corrected_y, 0)  # Remove negative values
        
        if len(corrected_y) > 1:
            area = np.trapz(corrected_y, x_region)
        else:
            area = 0.0
            
        # Create full baseline array for plotting
        full_baseline = np.zeros_like(y)
        full_baseline[mask] = baseline_region
        
        return max(0.0, area), full_baseline
        
    except Exception:
        # Fallback to simple integration
        area = np.trapz(y_region, x_region)
        return max(0.0, area), np.zeros_like(y)


def get_automatic_range(mz_center: float, percent: float) -> Tuple[float, float]:
    """
    Get automatic integration range based on percentage of m/z.
    
    Args:
        mz_center: Central m/z value
        percent: Percentage width (e.g., 5.0 for ±5%)
        
    Returns:
        Tuple of (min_mz, max_mz)
        
    Example:
        >>> range_min, range_max = get_automatic_range(1000.0, 5.0)
        >>> print(f"Range: {range_min:.1f} - {range_max:.1f}")
        Range: 950.0 - 1050.0
    """
    delta = mz_center * (percent / 100.0)
    return mz_center - delta, mz_center + delta


def calculate_theoretical_mz(mass: float, charge: int) -> float:
    """
    Calculate theoretical m/z for a given mass and charge state.
    
    Args:
        mass: Protein mass in Da
        charge: Charge state
        
    Returns:
        Theoretical m/z value
        
    Example:
        >>> mz = calculate_theoretical_mz(16952.3, 24)
        >>> print(f"m/z: {mz:.2f}")
    """
    # Use local constant to avoid namespace collision when both this function
    # and PROTON_MASS are imported into user code
    _PROTON_MASS = 1.007276
    return (mass + _PROTON_MASS * charge) / charge


# Import main classes
from .drift_calibration import DriftCalibrationProcessor, CalibratedDriftResult
from .visualization import (
    plot_spectrum_with_integration,
    plot_full_spectrum_with_charge_states
)

__all__ = [
    'PROTON_MASS',
    'fit_baseline_and_integrate',
    'get_automatic_range',
    'calculate_theoretical_mz',
    'DriftCalibrationProcessor',
    'CalibratedDriftResult',
    'plot_spectrum_with_integration',
    'plot_full_spectrum_with_charge_states',
    'ORIGAMISettings',
    'ORIGAMIDataProcessor',
    'ORIGAMIVisualizer',
    'safe_float_conversion',
    'remove_duplicate_values',
    'interpolate_matrix',
    'smooth_matrix_gaussian',
    'smooth_matrix_savgol',
    'DeconvolutionResult',
    'MZCalculation',
    'ESIProtCalculator',
    'ESIProtDataExporter'
]
