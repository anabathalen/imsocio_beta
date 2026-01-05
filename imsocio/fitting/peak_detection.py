"""Peak detection module for curve fitting.

Provides Origin-style peak detection with automatic parameter estimation.

This module implements peak detection algorithms compatible with Origin software,
using percentage-based parameters for height, prominence, and distance thresholds.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
import warnings


class PeakDetector:
    """
    Peak detection using Origin-style parameters.
    
    This class provides static methods for detecting peaks in spectroscopic data
    using percentage-based thresholds similar to those in Origin software. The
    implementation uses scipy's signal processing functions with automatic
    parameter scaling.
    
    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.exp(-((x - 5)**2) / 2)  # Gaussian peak
        >>> detector = PeakDetector()
        >>> peaks = detector.find_peaks_origin_style(x, y)
        >>> print(f"Found {len(peaks)} peaks")
    """
    
    @staticmethod
    def find_peaks_origin_style(x, y, min_height_percent=5, min_prominence_percent=2, 
                               min_distance_percent=5, smoothing_points=5):
        """
        Peak detection using Origin-style percentage-based parameters.
        
        This method detects peaks in spectroscopic data using percentage-based
        thresholds for height, prominence, and distance. It optionally applies
        Savitzky-Golay smoothing before peak detection.
        
        Args:
            x (array-like): X-axis data (e.g., drift time, m/z). Must be monotonic.
            y (array-like): Y-axis data (e.g., intensity). Must match length of x.
            min_height_percent (float, optional): Minimum peak height as percentage 
                of data range (0-100). Default: 5.
            min_prominence_percent (float, optional): Minimum peak prominence as 
                percentage of data range (0-100). Default: 2.
            min_distance_percent (float, optional): Minimum distance between peaks 
                as percentage of data length (0-100). Default: 5.
            smoothing_points (int, optional): Number of points for Savitzky-Golay 
                smoothing. Set to 0 to disable smoothing. Default: 5.
            
        Returns:
            list of dict: List of dictionaries containing peak information:
                - index (int): Peak index in array
                - x (float): Peak x position
                - y (float): Peak y value (height)
                - prominence (float): Peak prominence
                - width_half (float): Peak width at half maximum (FWHM)
                - width_base (float): Peak width at base (10% height)
                - area_estimate (float): Estimated peak area
                
        Raises:
            ValueError: If x and y have different lengths, or if data is too short.
            TypeError: If x or y are not array-like.
            
        Example:
            >>> x = np.linspace(0, 10, 100)
            >>> y = np.exp(-((x - 5)**2) / 2) + 0.1 * np.random.randn(100)
            >>> peaks = PeakDetector.find_peaks_origin_style(
            ...     x, y, 
            ...     min_height_percent=10,
            ...     smoothing_points=3
            ... )
            >>> for peak in peaks:
            ...     print(f"Peak at x={peak['x']:.2f}, height={peak['y']:.2f}")
                
        Notes:
            - Smoothing uses Savitzky-Golay filter with polynomial order 2
            - Width calculations use scipy's peak_widths function
            - Returns empty list if no peaks are found
            - Bare except clause is used for robustness in width calculations,
              falling back to default width estimates
        """
        # Input validation
        if not isinstance(x, (np.ndarray, list, tuple)):
            raise TypeError(f"x must be array-like, got {type(x)}")
        if not isinstance(y, (np.ndarray, list, tuple)):
            raise TypeError(f"y must be array-like, got {type(y)}")
            
        x = np.asarray(x)
        y = np.asarray(y)
        
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length: len(x)={len(x)}, len(y)={len(y)}")
            
        if len(x) < 3:
            raise ValueError(f"Data too short for peak detection: need at least 3 points, got {len(x)}")
            
        if len(y) == 0 or np.all(np.isnan(y)):
            warnings.warn("Input data is empty or all NaN, returning empty peak list")
            return []
            
        # Check for invalid parameter values
        if min_height_percent < 0 or min_height_percent > 100:
            raise ValueError(f"min_height_percent must be between 0 and 100, got {min_height_percent}")
        if min_prominence_percent < 0 or min_prominence_percent > 100:
            raise ValueError(f"min_prominence_percent must be between 0 and 100, got {min_prominence_percent}")
        if min_distance_percent < 0 or min_distance_percent > 100:
            raise ValueError(f"min_distance_percent must be between 0 and 100, got {min_distance_percent}")
        if smoothing_points < 0:
            raise ValueError(f"smoothing_points must be non-negative, got {smoothing_points}")
        # Smooth data if requested
        if smoothing_points > 0:
            # Calculate window length (must be odd and >= 3)
            window_length = smoothing_points * 2 + 1
            
            # Ensure window length doesn't exceed data length
            if window_length > len(y):
                window_length = len(y)
                # Make it odd
                if window_length % 2 == 0:
                    window_length -= 1
            
            # Minimum window length is 3
            if window_length < 3:
                warnings.warn(f"Data too short for smoothing (length={len(y)}), using unsmoothed data.")
                y_smooth = y.copy()
            else:
                # Ensure window_length is odd
                if window_length % 2 == 0:
                    window_length -= 1
                
                # Polynomial order must be less than window length
                polyorder = min(2, window_length - 1)
                
                try:
                    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(f"Savitzky-Golay smoothing failed: {e}. Using unsmoothed data.")
                    y_smooth = y.copy()
        else:
            y_smooth = y.copy()
        
        # Calculate thresholds
        y_range = np.max(y_smooth) - np.min(y_smooth)
        
        # Handle case where data is flat (no range)
        if y_range == 0 or np.isnan(y_range):
            warnings.warn("Data has no variation (flat signal or contains NaN), cannot detect peaks")
            return []
        
        # Threshold calculations relative to smoothed data range
        min_height = np.min(y_smooth) + y_range * (min_height_percent / 100.0)
        min_prominence = y_range * (min_prominence_percent / 100.0)
        
        # Distance threshold (minimum number of points between peaks)
        min_distance = max(1, int(len(x) * (min_distance_percent / 100.0)))
        
        # Validate threshold values are reasonable
        if min_prominence <= 0:
            warnings.warn(f"Prominence threshold is zero or negative ({min_prominence}), no peaks will be detected")
            return []
        
        # Find peaks
        try:
            peaks, properties = find_peaks(
                y_smooth,
                height=min_height,
                prominence=min_prominence,
                distance=int(min_distance)
            )
        except Exception as e:
            raise RuntimeError(f"Peak detection failed: {e}") from e
        
        # Return empty list if no peaks found
        if len(peaks) == 0:
            return []
        
        # Calculate peak widths at different heights (Origin-style)
        try:
            widths_half = peak_widths(y_smooth, peaks, rel_height=0.5)[0]
            widths_base = peak_widths(y_smooth, peaks, rel_height=0.1)[0]
            
            peak_info = []
            for i, peak_idx in enumerate(peaks):
                # Calculate dx (spacing between points)
                dx = x[1] - x[0] if len(x) > 1 else 1.0
                
                info = {
                    'index': int(peak_idx),
                    'x': float(x[peak_idx]),
                    'y': float(y_smooth[peak_idx]),
                    'prominence': float(properties['prominences'][i]),
                    'width_half': float(widths_half[i] * dx) if i < len(widths_half) else 0.0,
                    'width_base': float(widths_base[i] * dx) if i < len(widths_base) else 0.0,
                    'area_estimate': float(properties['prominences'][i] * widths_half[i] * dx) if i < len(widths_half) else 0.0
                }
                peak_info.append(info)
            
            return peak_info
            
        except Exception as e:
            # Fallback if width calculation fails
            warnings.warn(f"Peak width calculation failed: {e}. Using default width estimates.")
            peak_info = []
            x_span = x[-1] - x[0] if len(x) > 1 else 1.0
            
            for i, peak_idx in enumerate(peaks):
                info = {
                    'index': int(peak_idx),
                    'x': float(x[peak_idx]),
                    'y': float(y_smooth[peak_idx]),
                    'prominence': float(properties['prominences'][i]),
                    'width_half': float(x_span / 20),  # Default width estimate
                    'width_base': float(x_span / 10),
                    'area_estimate': float(properties['prominences'][i] * x_span / 20)
                }
                peak_info.append(info)
            return peak_info
