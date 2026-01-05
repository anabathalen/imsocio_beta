"""Result Analyzer Module - Peak area and statistics calculation.

This module provides comprehensive peak analysis tools for fitted results,
including analytical and numerical area calculation and comprehensive
statistics generation in Origin-style format.

Key Features:
- Analytical area formulas for Gaussian and Lorentzian peaks
- Numerical integration for complex peak types (Voigt, BiGaussian, EMG)
- FWHM calculation with type-specific formulas
- Percentage-based statistics (area %, height %)
- Robust input validation and error handling

Classes:
    ResultAnalyzer: Static methods for peak analysis

Example:
    >>> from imsocio.fitting import ResultAnalyzer
    >>> parameters = [100, 5.0, 0.5, 80, 6.0, 0.6]  # 2 Gaussian peaks
    >>> areas = ResultAnalyzer.calculate_peak_areas(x_data, parameters, "Gaussian")
    >>> stats = ResultAnalyzer.calculate_peak_statistics(x, y, y_fit, parameters, "Gaussian")
"""

import numpy as np
import warnings
from scipy.integrate import trapezoid
from typing import List, Dict, Any

from .peak_functions import (
    gaussian_peak, voigt_peak, bigaussian_peak, 
    exponentially_modified_gaussian
)


class ResultAnalyzer:
    """
    Result analysis tools for fitted peaks.
    
    Provides static methods for:
    - Peak area calculation (analytical and numerical)
    - Comprehensive peak statistics (FWHM, area %, height %)
    - Origin-style result reporting
    
    All methods are static and can be called directly on the class.
    
    Methods
    -------
    calculate_peak_areas(x, parameters, peak_type)
        Calculate peak areas for all peaks in fit
    calculate_peak_statistics(x, y, fitted_curve, parameters, peak_type)
        Generate comprehensive statistics for all peaks
        
    Examples
    --------
    Calculate peak areas:
    
    >>> from imsocio.fitting import ResultAnalyzer
    >>> parameters = [100, 5.0, 0.5, 80, 6.0, 0.6]  # 2 Gaussian peaks
    >>> areas = ResultAnalyzer.calculate_peak_areas(x_data, parameters, "Gaussian")
    
    Get full peak statistics:
    
    >>> stats = ResultAnalyzer.calculate_peak_statistics(x_data, y_data, y_fit, 
    ...                                                   parameters, "Gaussian")
    >>> for peak in stats:
    ...     print(f"Peak {peak['peak_number']}: Area = {peak['area']:.2f}, "
    ...           f"FWHM = {peak['fwhm']:.3f}")
    """
    
    @staticmethod
    def calculate_peak_areas(x: np.ndarray, parameters: np.ndarray, 
                            peak_type: str) -> List[float]:
        """Calculate peak areas using analytical formulas or numerical integration.
        
        Uses analytical formulas for Gaussian and Lorentzian (faster, exact).
        Uses numerical integration for complex types (Voigt, BiGaussian, EMG).
        
        Args:
            x (array-like): X-axis data range (must be 1D, sorted)
            parameters (array-like): Fitted parameters for all peaks (flat array)
            peak_type (str): Peak function type:
                - "Gaussian": Area = A × σ × √(2π)
                - "Lorentzian": Area = A × γ × π
                - "Voigt", "BiGaussian", "EMG": Numerical integration
                
        Returns:
            list: Peak areas for each peak [area1, area2, ...]
            
        Raises:
            ValueError: If inputs are invalid, parameters length is wrong,
                or peak_type is unknown
                
        Notes:
            - Gaussian: Analytical integration of exp(-x²/(2σ²))
            - Lorentzian: Analytical integration of 1/(1+x²/γ²)
            - Complex types: 1000-point trapezoidal integration over x-range
            - Areas are always positive (absolute value of amplitude)
            
        Example:
            >>> x = np.linspace(0, 10, 100)
            >>> params = [100, 5.0, 0.5, 80, 6.0, 0.6]  # 2 Gaussian peaks
            >>> areas = ResultAnalyzer.calculate_peak_areas(x, params, "Gaussian")
            >>> print(f"Peak 1 area: {areas[0]:.2f}, Peak 2 area: {areas[1]:.2f}")
            Peak 1 area: 125.33, Peak 2 area: 120.13
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        parameters = np.asarray(parameters, dtype=float)
        
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")
        if parameters.ndim != 1:
            raise ValueError(f"parameters must be 1D array, got shape {parameters.shape}")
        if len(x) < 2:
            raise ValueError(f"x must have at least 2 points, got {len(x)}")
        if not isinstance(peak_type, str):
            raise ValueError(f"peak_type must be a string, got {type(peak_type)}")
        
        # Validate peak_type and get params_per_peak
        params_map = {
            "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, 
            "BiGaussian": 4, "EMG": 4
        }
        
        if peak_type not in params_map:
            raise ValueError(
                f"Unknown peak_type '{peak_type}'. "
                f"Valid types: {list(params_map.keys())}"
            )
        
        params_per_peak = params_map[peak_type]
        
        # Validate parameters length
        if len(parameters) % params_per_peak != 0:
            raise ValueError(
                f"parameters length ({len(parameters)}) must be multiple of "
                f"{params_per_peak} for {peak_type} peaks. "
                f"Got {len(parameters) // params_per_peak} complete peaks with "
                f"{len(parameters) % params_per_peak} leftover parameters."
            )
        
        n_peaks = len(parameters) // params_per_peak
        if n_peaks == 0:
            raise ValueError("parameters array is empty or too short for at least one peak")
        
        areas = []
        
        for i in range(n_peaks):
            base_idx = i * params_per_peak
            
            try:
                if peak_type == "Gaussian":
                    amplitude, center, sigma = parameters[base_idx:base_idx+3]
                    # Analytical area: ∫ A×exp(-x²/(2σ²)) dx = A×σ×√(2π)
                    if sigma <= 0:
                        raise ValueError(f"Peak {i}: sigma must be > 0, got {sigma:.3e}")
                    area = abs(amplitude) * abs(sigma) * np.sqrt(2 * np.pi)
                
                elif peak_type == "Lorentzian":
                    amplitude, center, gamma = parameters[base_idx:base_idx+3]
                    # Analytical area: ∫ A/(1+(x/γ)²) dx = A×π×γ
                    if gamma <= 0:
                        raise ValueError(f"Peak {i}: gamma must be > 0, got {gamma:.3e}")
                    area = abs(amplitude) * abs(gamma) * np.pi
                
                else:
                    # Numerical integration for complex peak types
                    # Use 1000 points for high accuracy
                    x_peak = np.linspace(x.min(), x.max(), 1000)
                    
                    if peak_type == "Voigt":
                        y_peak = voigt_peak(x_peak, *parameters[base_idx:base_idx+4])
                    elif peak_type == "BiGaussian":
                        y_peak = bigaussian_peak(x_peak, *parameters[base_idx:base_idx+4])
                    elif peak_type == "EMG":
                        y_peak = exponentially_modified_gaussian(x_peak, *parameters[base_idx:base_idx+4])
                    else:
                        # Fallback (should not reach here)
                        y_peak = gaussian_peak(x_peak, *parameters[base_idx:base_idx+3])
                    
                    # Trapezoidal integration
                    area = abs(trapezoid(y_peak, x_peak))
                    
                    if not np.isfinite(area):
                        raise ValueError(
                            f"Peak {i}: Integration produced non-finite area {area}"
                        )
            
            except Exception as e:
                # Catch errors from peak functions
                raise ValueError(
                    f"Error calculating area for peak {i} ({peak_type}): {str(e)}"
                ) from e
            
            areas.append(area)
        
        return areas
    
    @staticmethod
    def calculate_peak_statistics(x: np.ndarray, y: np.ndarray, fitted_curve: np.ndarray,
                                  parameters: np.ndarray, peak_type: str) -> List[Dict[str, Any]]:
        """Calculate comprehensive peak statistics in Origin-style format.
        
        Generates detailed statistics for each peak including position, size,
        and relative contributions. FWHM calculations use type-specific formulas.
        
        Args:
            x (array-like): X-axis data (must be 1D)
            y (array-like): Original y data (must match length of x)
            fitted_curve (array-like): Fitted y data (must match length of x)
            parameters (array-like): Fitted parameters for all peaks (flat array)
            peak_type (str): Peak function type
                
        Returns:
            list: List of statistics dicts, one per peak, containing:
                - peak_number (int): Peak index (1-based)
                - amplitude (float): Peak height
                - center (float): Peak center position
                - fwhm (float): Full Width at Half Maximum
                - area (float): Integrated peak area
                - area_percent (float): Percentage of total area across all peaks
                - height_percent (float): Percentage of maximum data intensity
                
        Raises:
            ValueError: If inputs are invalid or mismatched
            
        Notes:
            FWHM formulas by peak type:
            - Gaussian: FWHM = 2\u03c3\u221a(2 ln 2) \u2248 2.355\u03c3
            - Lorentzian: FWHM = 2\u03b3
            - Voigt: FWHM \u2248 0.5346(2\u03b3_L) + \u221a[0.2166(2\u03b3_L)\u00b2 + FWHM_G\u00b2]
            - BiGaussian/EMG: FWHM \u2248 2.355\u03c3 (approximation)
            
            Percentages are based on:
            - area_percent: Relative to sum of all peak areas
            - height_percent: Relative to max(y) data value
            
        Example:\n            >>> x = np.linspace(0, 10, 100)
            >>> y = 100*np.exp(-(x-5)**2/(2*0.5**2)) + 80*np.exp(-(x-6)**2/(2*0.6**2))
            >>> y_fit = y.copy()  # Assume perfect fit
            >>> params = [100, 5.0, 0.5, 80, 6.0, 0.6]
            >>> stats = ResultAnalyzer.calculate_peak_statistics(x, y, y_fit, params, "Gaussian")
            >>> print(f"Peak 1: Center={stats[0]['center']:.2f}, FWHM={stats[0]['fwhm']:.3f}")
            Peak 1: Center=5.00, FWHM=1.177
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fitted_curve = np.asarray(fitted_curve, dtype=float)
        parameters = np.asarray(parameters, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1 or fitted_curve.ndim != 1:
            raise ValueError(f"x, y, and fitted_curve must be 1D arrays, got shapes {x.shape}, {y.shape}, {fitted_curve.shape}")
        if len(x) != len(y) or len(x) != len(fitted_curve):
            raise ValueError(f"x, y, and fitted_curve must have same length, got {len(x)}, {len(y)}, {len(fitted_curve)}")
        if len(x) < 2:
            raise ValueError(f"Arrays must have at least 2 points, got {len(x)}")
        if not isinstance(peak_type, str):
            raise ValueError(f"ahhh")

