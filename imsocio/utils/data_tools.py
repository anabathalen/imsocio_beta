"""Data analysis tools for IMSocio.

Core fitting and statistical functions.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional


def gaussian(x: np.ndarray, amp: float, mean: float, stddev: float) -> np.ndarray:
    """Gaussian function for fitting.
    
    Args:
        x: Independent variable
        amp: Amplitude (height) of the Gaussian
        mean: Center position
        stddev: Standard deviation (width)
        
    Returns:
        Gaussian values at x
    """
    return amp * np.exp(-((x - mean)**2) / (2 * stddev**2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        R² value (0-1, where 1 is perfect fit)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def fit_gaussian_with_retries(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_attempts: int = 10,
    random_state: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """Fit a Gaussian to data with multiple random initial guesses.
    
    This function attempts multiple fits with different initial parameter guesses
    to avoid local minima and find the best fit.
    
    Args:
        x_values: Independent variable data
        y_values: Dependent variable data
        n_attempts: Number of fitting attempts with different initial guesses
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (best_params, best_r2, best_fitted_values)
            - best_params: Array of [amplitude, mean, stddev]
            - best_r2: R² value of the best fit
            - best_fitted_values: Fitted y values using best parameters
    """
    rng = np.random.default_rng(random_state)
    best_r2 = -np.inf
    best_params = None
    best_fitted_values = None

    # Better initial parameter estimates
    max_y = np.max(y_values)
    min_y = np.min(y_values)
    amp_guess = max_y - min_y
    
    # Find peak position (where y is maximum)
    peak_idx = np.argmax(y_values)
    mean_guess = x_values[peak_idx]
    
    # Estimate width from half-maximum
    half_max = min_y + (max_y - min_y) / 2
    above_half = y_values > half_max
    if np.sum(above_half) > 1:
        # Find width at half maximum
        indices_above = np.where(above_half)[0]
        x_width = x_values[indices_above[-1]] - x_values[indices_above[0]]
        # FWHM ≈ 2.355 * sigma for Gaussian
        stddev_guess = x_width / 2.355
    else:
        # Fallback if we can't estimate from half-maximum
        stddev_guess = (np.max(x_values) - np.min(x_values)) / 10

    # Loop through fitting attempts with different initial guesses
    for attempt in range(n_attempts):
        # Use informed guesses with some variation
        if attempt == 0:
            # First attempt: use best estimates
            initial_guess = [amp_guess, mean_guess, stddev_guess]
        else:
            # Subsequent attempts: add variation
            initial_guess = [
                amp_guess * rng.uniform(0.5, 1.5),
                mean_guess + rng.uniform(-stddev_guess, stddev_guess),
                stddev_guess * rng.uniform(0.5, 2.0)
            ]

        try:
            params, _ = curve_fit(gaussian, x_values, y_values, p0=initial_guess, method='trf')
            # Ensure stddev is positive (curve_fit can return negative values)
            params[2] = abs(params[2])
            fitted_values = gaussian(x_values, *params)
            r2 = r_squared(y_values, fitted_values)

            if r2 > best_r2:  # Keep only the best fit
                best_r2 = r2
                best_params = params
                best_fitted_values = fitted_values
        
        except RuntimeError:
            # Fit failed, try next attempt
            continue
    
    return best_params, best_r2, best_fitted_values
