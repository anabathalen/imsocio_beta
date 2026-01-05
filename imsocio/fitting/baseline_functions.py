"""Baseline correction functions for curve fitting.

This module provides baseline functions for modeling background signals
in spectroscopic and chromatographic data. These can be combined with
peak functions for comprehensive curve fitting.

Available baseline types:
- Linear: y = slope × x + intercept
- Polynomial: y = a_n×x^n + ... + a_1×x + a_0
- Exponential: y = a × exp(b×x) + c

All functions accept NumPy arrays and return NumPy arrays of the same shape.

Example:
    >>> import numpy as np
    >>> from imsocio.fitting.baseline_functions import linear_baseline
    >>> x = np.linspace(0, 10, 100)
    >>> baseline = linear_baseline(x, slope=0.5, intercept=10)
"""

import numpy as np
import warnings
from typing import Union


def linear_baseline(x: Union[np.ndarray, float], slope: float, 
                    intercept: float) -> Union[np.ndarray, float]:
    """Linear baseline: y = slope × x + intercept.
    
    Simplest baseline model, appropriate for slowly varying backgrounds.
    
    Args:
        x (array-like or float): Independent variable
        slope (float): Linear slope coefficient
        intercept (float): Y-intercept (baseline at x=0)
        
    Returns:
        array or float: Baseline values matching shape of x
        
    Raises:
        ValueError: If inputs contain non-finite values
        
    Notes:
        - All parameters can be positive or negative
        - No numerical stability issues
        - Suitable for most spectroscopic baselines
        
    Example:
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> baseline = linear_baseline(x, slope=0.5, intercept=10.0)
        >>> print(baseline)
        [10.  10.5 11.  11.5 12. ]
    """
    x = np.asarray(x, dtype=float)
    
    # Validate inputs
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.isfinite(slope):
        raise ValueError(f"slope must be finite, got {slope}")
    if not np.isfinite(intercept):
        raise ValueError(f"intercept must be finite, got {intercept}")
    
    return slope * x + intercept


def polynomial_baseline(x: Union[np.ndarray, float], 
                        *coeffs: float) -> Union[np.ndarray, float]:
    """Polynomial baseline: y = a_n×x^n + ... + a_1×x + a_0.
    
    Higher-order polynomial for complex baseline shapes. Uses NumPy's
    polyval for numerical stability.
    
    Args:
        x (array-like or float): Independent variable
        *coeffs (float): Polynomial coefficients in descending order
            - coeffs[0]: coefficient of x^n (highest degree)
            - coeffs[1]: coefficient of x^(n-1)
            - ...
            - coeffs[n]: constant term (x^0)
            
    Returns:
        array or float: Baseline values matching shape of x
        
    Raises:
        ValueError: If no coefficients provided or inputs are invalid
        
    Notes:
        - At least one coefficient required
        - Higher degrees may cause overfitting or numerical instability
        - Recommend degree ≤ 3 for most applications
        
    Example:
        >>> x = np.array([0, 1, 2, 3])
        >>> # Quadratic: y = 2x² + 3x + 1
        >>> baseline = polynomial_baseline(x, 2, 3, 1)
        >>> print(baseline)
        [ 1 6 15 28]
    """
    x = np.asarray(x, dtype=float)
    
    # Validate inputs
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if len(coeffs) == 0:
        raise ValueError("At least one coefficient required")
    
    coeffs_array = np.array(coeffs, dtype=float)
    if not np.all(np.isfinite(coeffs_array)):
        raise ValueError("All coefficients must be finite")
    
    # Warn for high-degree polynomials
    if len(coeffs) > 4:
        warnings.warn(
            f"Polynomial degree {len(coeffs)-1} may cause numerical instability. "
            f"Consider using degree ≤ 3.",
            UserWarning
        )
    
    return np.polyval(coeffs, x)


def exponential_baseline(x: Union[np.ndarray, float], a: float, 
                         b: float, c: float) -> Union[np.ndarray, float]:
    """Exponential baseline: y = a × exp(b×x) + c.
    
    Exponential baseline for decaying or growing backgrounds.
    Common in fluorescence, radioactive decay, and chemical kinetics.
    
    Args:
        x (array-like or float): Independent variable
        a (float): Amplitude coefficient (scaling factor)
        b (float): Exponential rate (growth if b>0, decay if b<0)
        c (float): Offset (asymptotic value as x→∞ for decay, x→-∞ for growth)
        
    Returns:
        array or float: Baseline values matching shape of x
        
    Raises:
        ValueError: If inputs are invalid or would cause overflow
        
    Notes:
        - Large |b×x| values can cause overflow/underflow
        - Function clips b×x to [-100, 100] to prevent overflow
        - For decay: b < 0 (typical in spectroscopy)
        - For growth: b > 0 (less common)
        
    Example:
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> # Exponential decay: y = 10×exp(-0.5×x) + 5
        >>> baseline = exponential_baseline(x, a=10, b=-0.5, c=5)
        >>> print(baseline)
        [15.         11.06530660  8.71306698  7.23066982  6.35304043]
    """
    x = np.asarray(x, dtype=float)
    
    # Validate inputs
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.isfinite(a):
        raise ValueError(f"a must be finite, got {a}")
    if not np.isfinite(b):
        raise ValueError(f"b must be finite, got {b}")
    if not np.isfinite(c):
        raise ValueError(f"c must be finite, got {c}")
    
    # Compute b*x with overflow protection
    exponent = b * x
    
    # Check for potential overflow
    if np.any(np.abs(exponent) > 100):
        warnings.warn(
            f"Large exponent values detected (max |b×x| = {np.max(np.abs(exponent)):.1f}). "
            f"Clipping to [-100, 100] to prevent overflow.",
            UserWarning
        )
        exponent = np.clip(exponent, -100, 100)
    
    result = a * np.exp(exponent) + c
    
    # Validate output
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Exponential baseline produced non-finite values. "
            f"Parameters: a={a}, b={b}, c={c}, x range=[{np.min(x):.3e}, {np.max(x):.3e}]"
        )
    
    return result
