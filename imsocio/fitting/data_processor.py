"""Data Processor Module - Preprocessing tools for peak fitting.

This module provides data preprocessing tools commonly used in Origin software,
including smoothing algorithms and baseline subtraction methods.

Key Features:
- Savitzky-Golay smoothing with automatic parameter adjustment
- Moving average smoothing
- Linear baseline subtraction (using endpoints)
- Polynomial baseline subtraction (with region selection)

All methods include input validation and handle edge cases robustly.

Classes:
    DataProcessor: Static methods for data preprocessing

Example:
    >>> from imsocio.fitting import DataProcessor
    >>> y_smooth = DataProcessor.smooth_data(x, y, "Savitzky-Golay", window_size=11)
    >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, "Linear")
"""

import numpy as np
import warnings
from scipy.signal import savgol_filter
from typing import Tuple, Optional


class DataProcessor:
    """
    Data preprocessing tools for peak fitting.
    
    Provides static methods for common preprocessing operations:
    - Data smoothing (Savitzky-Golay, Moving Average)
    - Baseline subtraction (Linear, Polynomial)
    
    All methods are static and can be called directly on the class.
    
    Methods
    -------
    smooth_data(x, y, method, window_size, poly_order)
        Smooth data using various algorithms
    subtract_baseline(x, y, method, poly_degree, regions)
        Subtract baseline using various methods
        
    Examples
    --------
    Smooth data:
    
    >>> from imsocio.fitting import DataProcessor
    >>> y_smooth = DataProcessor.smooth_data(x, y, method="Savitzky-Golay", 
    ...                                       window_size=11, poly_order=3)
    
    Subtract linear baseline:
    
    >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Linear")
    
    Subtract polynomial baseline using specific regions:
    
    >>> regions = [(0, 2), (8, 10)]  # Use edges for baseline
    >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Polynomial",
    ...                                                          poly_degree=2, regions=regions)
    """
    
    @staticmethod
    def smooth_data(x: np.ndarray, y: np.ndarray, method: str = "Savitzky-Golay", 
                   window_size: int = 5, poly_order: int = 2) -> np.ndarray:
        """Smooth data using various algorithms (Origin-style options).
        
        Args:
            x (array-like): X-axis data (not used but kept for API consistency)
            y (array-like): Y-axis data to smooth (must be 1D)
            method (str): Smoothing method. Options:
                - "Savitzky-Golay": Polynomial smoothing (preserves peaks)
                - "Moving Average": Uniform filter (simple averaging)
                Default: "Savitzky-Golay"
            window_size (int): Window size for smoothing (default: 5)
                - Must be odd for Savitzky-Golay
                - Auto-adjusted if even or too large
            poly_order (int): Polynomial order for Savitzky-Golay (default: 2)
                - Must be < window_size
                - Auto-adjusted if too large
                
        Returns:
            ndarray: Smoothed y data (same shape as input)
            
        Raises:
            ValueError: If y is empty, not 1D, or contains non-finite values
            
        Notes:
            - Savitzky-Golay is preferred for peak data (preserves shape)
            - Moving Average is faster but can distort peaks
            - Edge effects are handled automatically
            
        Example:
            >>> x = np.linspace(0, 10, 100)
            >>> y_noisy = np.sin(x) + np.random.normal(0, 0.1, 100)
            >>> y_smooth = DataProcessor.smooth_data(x, y_noisy, "Savitzky-Golay", 11, 3)
        """
        # Validate inputs
        y = np.asarray(y, dtype=float)
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        if len(y) == 0:
            raise ValueError("y array is empty")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")
        if not isinstance(method, str):
            raise ValueError(f"method must be a string, got {type(method)}")
        
        # Validate window_size
        if not isinstance(window_size, (int, np.integer)):
            raise ValueError(f"window_size must be an integer, got {type(window_size)}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        
        if method == "Savitzky-Golay":
            # Auto-adjust window size
            original_window = window_size
            if window_size >= len(y):
                window_size = len(y) - 1 if len(y) > 1 else 1
                warnings.warn(
                    f"window_size ({original_window}) >= data length ({len(y)}). "
                    f"Adjusted to {window_size}.",
                    UserWarning
                )
            
            # Ensure window is odd
            if window_size % 2 == 0:
                window_size += 1
                if window_size != original_window:
                    warnings.warn(
                        f"window_size must be odd for Savitzky-Golay. "
                        f"Adjusted from {original_window} to {window_size}.",
                        UserWarning
                    )
            
            # Validate and adjust poly_order
            if not isinstance(poly_order, (int, np.integer)):
                raise ValueError(f"poly_order must be an integer, got {type(poly_order)}")
            
            original_poly = poly_order
            if poly_order >= window_size:
                poly_order = window_size - 1
                warnings.warn(
                    f"poly_order ({original_poly}) >= window_size ({window_size}). "
                    f"Adjusted to {poly_order}.",
                    UserWarning
                )
            if poly_order < 0:
                raise ValueError(f"poly_order must be >= 0, got {poly_order}")
            
            return savgol_filter(y, window_size, poly_order)
        
        elif method == "Moving Average":
            from scipy.ndimage import uniform_filter1d
            
            if window_size > len(y):
                warnings.warn(
                    f"window_size ({window_size}) > data length ({len(y)}). "
                    f"Using window_size = {len(y)}.",
                    UserWarning
                )
                window_size = len(y)
            
            return uniform_filter1d(y, size=window_size)
        
        else:
            warnings.warn(
                f"Unknown smoothing method '{method}'. "
                f"Valid options: 'Savitzky-Golay', 'Moving Average'. "
                f"Returning original data.",
                UserWarning
            )
            return y
    
    @staticmethod
    def subtract_baseline(x, y, method="Linear", poly_degree=2, regions=None):
        """
        Baseline subtraction (Origin-style methods).
        
        Parameters
        ----------
        x : array_like
            X-axis data
        y : array_like
            Y-axis data
        method : str, optional
            Baseline method: "None", "Linear", or "Polynomial" (default: "Linear")
        poly_degree : int, optional
            Polynomial degree for polynomial baseline (default: 2)
        regions : list of tuple, optional
            List of (start, end) x-ranges to use for baseline fitting.
            If None, uses automatic region selection (default: None)
            
        Returns
        -------
        y_corrected : ndarray
            Baseline-corrected y data
        baseline : ndarray
            The fitted baseline
            
        Notes
        -----
        For Linear method:
        - If regions=None, uses first and last 10% of data
        - Fits a line through specified or automatic baseline regions
        
        For Polynomial method:
        - If regions=None, uses entire dataset
        - Fits polynomial of specified degree
        
        Examples
        --------
        >>> # Linear baseline using default regions
        >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y)
        
        >>> # Polynomial baseline with custom regions
        >>> regions = [(0, 1), (9, 10)]
        >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Polynomial",
        ...                                                          poly_degree=3, regions=regions)
        """
        if method == "None":
            return y, np.zeros_like(y)
        
        elif method == "Linear":
            # Fit linear baseline to endpoints or specified regions
            if regions is None:
                # Use first and last 10% of data
                n_points = max(2, len(x) // 10)
                x_baseline = np.concatenate([x[:n_points], x[-n_points:]])
                y_baseline = np.concatenate([y[:n_points], y[-n_points:]])
            else:
                x_baseline = []
                y_baseline = []
                for start, end in regions:
                    mask = (x >= start) & (x <= end)
                    x_baseline.extend(x[mask])
                    y_baseline.extend(y[mask])
                x_baseline = np.array(x_baseline)
                y_baseline = np.array(y_baseline)
            
            if len(x_baseline) >= 2:
                slope, intercept = np.polyfit(x_baseline, y_baseline, 1)
                baseline = slope * x + intercept
            else:
                baseline = np.zeros_like(y)
            
        elif method == "Polynomial":
            # Fit polynomial baseline
            if regions is None:
                x_baseline = x
                y_baseline = y
            else:
                x_baseline = []
                y_baseline = []
                for start, end in regions:
                    mask = (x >= start) & (x <= end)
                    x_baseline.extend(x[mask])
                    y_baseline.extend(y[mask])
                x_baseline = np.array(x_baseline)
                y_baseline = np.array(y_baseline)
            
            if len(x_baseline) >= poly_degree + 1:
                coeffs = np.polyfit(x_baseline, y_baseline, poly_degree)
                baseline = np.polyval(coeffs, x)
            else:
                baseline = np.zeros_like(y)
        
        else:
            baseline = np.zeros_like(y)
        
        return y - baseline, baseline
