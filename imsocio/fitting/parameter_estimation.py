"""Parameter estimation module for curve fitting.

Provides Origin-style initial parameter estimation for various peak types.
This module implements intelligent parameter initialization based on peak
detection results, ensuring stable and efficient fitting convergence.

The estimators convert FWHM (Full Width at Half Maximum) from peak detection
to appropriate width parameters for each peak type:
- Gaussian: σ (standard deviation)
- Lorentzian: γ (HWHM - Half Width at Half Maximum)
- Voigt: σ_G (Gaussian) and γ_L (Lorentzian)
- BiGaussian: σ_L and σ_R (left and right widths)
- EMG: σ (Gaussian width) and τ (exponential decay constant)

Example:
    >>> from imsocio.fitting import ParameterEstimator
    >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 0.8}]
    >>> params = ParameterEstimator.estimate_parameters(x, y, peak_info, "Gaussian")
    >>> # Returns [amplitude, center, sigma] for the peak
"""

import numpy as np
import warnings
from typing import List, Dict, Any


class ParameterEstimator:
    """Estimates initial parameters for different peak types.
    
    This class provides static methods for estimating initial parameters
    from peak detection results. Proper initialization is critical for
    convergence of non-linear fitting algorithms.
    
    All methods follow Origin's parameter estimation strategy with
    safeguards against invalid values (negative widths, zero amplitudes, etc.).
    """
    
    @staticmethod
    def estimate_gaussian_parameters(x: np.ndarray, y: np.ndarray, 
                                    peak_info: List[Dict[str, Any]]) -> List[float]:
        """Estimate initial parameters for Gaussian peaks (Origin method).
        
        Converts FWHM from peak detection to Gaussian σ (standard deviation)
        using the relationship: FWHM = 2√(2 ln 2) σ ≈ 2.355 σ
        
        Args:
            x (array-like): X-axis data (must be 1D)
            y (array-like): Y-axis data (must match length of x)
            peak_info (list): List of dicts from peak detection, each containing:
                - 'x': peak center position
                - 'y': peak height (amplitude)
                - 'width_half': FWHM estimate (optional)
                
        Returns:
            list: Flat parameter list [amp1, center1, σ1, amp2, center2, σ2, ...]
            
        Raises:
            ValueError: If x and y have different lengths, are empty, or peak_info is invalid
            
        Notes:
            - Amplitudes are forced positive with minimum value of 1% of max(y)
            - Widths have minimum value of 0.1% of x-range for numerical stability
            - Missing FWHM defaults to 5% of x-range (Origin behavior)
            
        Example:
            >>> x = np.linspace(0, 10, 100)
            >>> y = 100 * np.exp(-(x-5)**2 / (2*0.5**2))
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.18}]
            >>> params = ParameterEstimator.estimate_gaussian_parameters(x, y, peak_info)
            >>> # Returns approximately [100, 5.0, 0.5]
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape}, {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
        if len(x) == 0:
            raise ValueError("x and y arrays are empty")
        if not isinstance(peak_info, list):
            raise ValueError(f"peak_info must be a list, got {type(peak_info)}")
        if len(peak_info) == 0:
            raise ValueError("peak_info list is empty - no peaks to estimate")
        
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        if x_span <= 0:
            raise ValueError(f"x must be increasing, got x[0]={x[0]}, x[-1]={x[-1]}")
        
        y_max = np.max(np.abs(y))
        if y_max == 0:
            y_max = 1.0  # Fallback for zero data
        
        # Gaussian FWHM to sigma conversion factor
        FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247
        
        for i, peak in enumerate(peak_info):
            # Validate peak dictionary
            if not isinstance(peak, dict):
                raise ValueError(f"peak_info[{i}] must be a dict, got {type(peak)}")
            if 'x' not in peak or 'y' not in peak:
                raise ValueError(f"peak_info[{i}] missing required keys 'x' and/or 'y'")
            
            # Estimate amplitude (force positive, minimum 1% of max)
            # Use actual y value at peak position for better initial guess
            peak_y = abs(peak['y'])
            
            # If peak amplitude is very small compared to max, might be noise
            # Set minimum amplitude to ensure numerical stability
            amplitude = max(peak_y, y_max * 0.01)
            
            # For single-peak fits, ensure amplitude makes sense
            # Peak should be close to the actual maximum in the data
            if len(peak_info) == 1 and peak_y < y_max * 0.5:
                warnings.warn(
                    f"Detected peak amplitude ({peak_y:.2e}) is much lower than "
                    f"data maximum ({y_max:.2e}). Peak detection may have missed "
                    f"the true peak maximum. Using data maximum as initial amplitude.",
                    UserWarning
                )
                amplitude = y_max
            
            # Center position
            center = float(peak['x'])
            if not (x[0] <= center <= x[-1]):
                warnings.warn(
                    f"Peak {i} center {center:.3f} is outside x-range "
                    f"[{x[0]:.3f}, {x[-1]:.3f}]. Using nearest boundary.",
                    UserWarning
                )
                center = np.clip(center, x[0], x[-1])
            
            # Convert FWHM to sigma, with fallback
            fwhm = peak.get('width_half', x_span / 20)  # Default: 5% of x-range
            if fwhm <= 0:
                warnings.warn(
                    f"Peak {i} has invalid FWHM {fwhm:.3e}. Using default {x_span/20:.3e}.",
                    UserWarning
                )
                fwhm = x_span / 20
            
            sigma = max(fwhm * FWHM_TO_SIGMA, x_span * 0.001)  # Minimum: 0.1% of x-range
            
            params.extend([amplitude, center, sigma])
        
        return params
    
    @staticmethod
    def estimate_lorentzian_parameters(x: np.ndarray, y: np.ndarray, 
                                      peak_info: List[Dict[str, Any]]) -> List[float]:
        """Estimate initial parameters for Lorentzian peaks.
        
        Converts FWHM from peak detection to Lorentzian γ (HWHM)
        using the relationship: FWHM = 2γ
        
        Args:
            x (array-like): X-axis data (must be 1D)
            y (array-like): Y-axis data (must match length of x)
            peak_info (list): List of dicts from peak detection
                
        Returns:
            list: Flat parameter list [amp1, center1, γ1, amp2, center2, γ2, ...]
            
        Raises:
            ValueError: If inputs are invalid
            
        Notes:
            Same validation and safeguards as estimate_gaussian_parameters.
            γ (gamma) is the Half Width at Half Maximum for Lorentzian.
            
        Example:
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.0}]
            >>> params = ParameterEstimator.estimate_lorentzian_parameters(x, y, peak_info)
            >>> # Returns [100, 5.0, 0.5] where 0.5 = FWHM/2
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape}, {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
        if len(x) == 0:
            raise ValueError("x and y arrays are empty")
        if not isinstance(peak_info, list) or len(peak_info) == 0:
            raise ValueError("peak_info must be a non-empty list")
        
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        if x_span <= 0:
            raise ValueError(f"x must be increasing, got x[0]={x[0]}, x[-1]={x[-1]}")
        
        y_max = np.max(np.abs(y))
        if y_max == 0:
            y_max = 1.0
        
        for i, peak in enumerate(peak_info):
            if not isinstance(peak, dict):
                raise ValueError(f"peak_info[{i}] must be a dict, got {type(peak)}")
            if 'x' not in peak or 'y' not in peak:
                raise ValueError(f"peak_info[{i}] missing required keys 'x' and/or 'y'")
            
            amplitude = max(abs(peak['y']), y_max * 0.01)
            center = float(peak['x'])
            
            if not (x[0] <= center <= x[-1]):
                warnings.warn(
                    f"Peak {i} center {center:.3f} outside x-range. Clipping.",
                    UserWarning
                )
                center = np.clip(center, x[0], x[-1])
            
            # For Lorentzian, FWHM = 2 * gamma
            fwhm = peak.get('width_half', x_span / 20)
            if fwhm <= 0:
                warnings.warn(
                    f"Peak {i} has invalid FWHM {fwhm:.3e}. Using default.",
                    UserWarning
                )
                fwhm = x_span / 20
            
            gamma = max(fwhm / 2.0, x_span * 0.001)
            
            params.extend([amplitude, center, gamma])
        
        return params
    
    @staticmethod
    def estimate_voigt_parameters(x: np.ndarray, y: np.ndarray, 
                                 peak_info: List[Dict[str, Any]]) -> List[float]:
        """Estimate initial parameters for Voigt peaks.
        
        Initializes Voigt parameters with equal Gaussian and Lorentzian contributions.
        A Voigt profile is a convolution of Gaussian and Lorentzian distributions.
        
        Args:
            x (array-like): X-axis data
            y (array-like): Y-axis data
            peak_info (list): List of peak detection results
                
        Returns:
            list: Flat parameter list [amp1, ctr1, σ_G1, γ_L1, amp2, ctr2, σ_G2, γ_L2, ...]
            
        Raises:
            ValueError: If inputs are invalid
            
        Notes:
            Initial strategy: Start with equal Gaussian (σ_G) and Lorentzian (γ_L) widths.
            Both are derived from the detected FWHM, allowing the fitter to determine
            the optimal Gaussian/Lorentzian balance.
            
        Example:
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.0}]
            >>> params = ParameterEstimator.estimate_voigt_parameters(x, y, peak_info)
            >>> # Returns [100, 5.0, sigma_g, gamma_l] with both widths from FWHM
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape}, {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
        if len(x) == 0:
            raise ValueError("x and y arrays are empty")
        if not isinstance(peak_info, list) or len(peak_info) == 0:
            raise ValueError("peak_info must be a non-empty list")
        
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        if x_span <= 0:
            raise ValueError(f"x must be increasing, got x[0]={x[0]}, x[-1]={x[-1]}")
        
        y_max = np.max(np.abs(y))
        if y_max == 0:
            y_max = 1.0
        
        # Conversion factors
        FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # For Gaussian component
        
        for i, peak in enumerate(peak_info):
            if not isinstance(peak, dict):
                raise ValueError(f"peak_info[{i}] must be a dict")
            if 'x' not in peak or 'y' not in peak:
                raise ValueError(f"peak_info[{i}] missing required keys")
            
            amplitude = max(abs(peak['y']), y_max * 0.01)
            center = float(peak['x'])
            
            if not (x[0] <= center <= x[-1]):
                warnings.warn(
                    f"Peak {i} center {center:.3f} outside x-range. Clipping.",
                    UserWarning
                )
                center = np.clip(center, x[0], x[-1])
            
            # Start with equal Gaussian and Lorentzian contributions
            base_width = max(peak.get('width_half', x_span / 20), x_span * 0.001)
            if base_width <= 0:
                warnings.warn(
                    f"Peak {i} has invalid width. Using default.",
                    UserWarning
                )
                base_width = x_span / 20
            
            width_g = base_width * FWHM_TO_SIGMA  # Gaussian sigma
            width_l = base_width / 2.0  # Lorentzian gamma
            
            params.extend([amplitude, center, width_g, width_l])
        
        return params
    
    @staticmethod
    def estimate_bigaussian_parameters(x: np.ndarray, y: np.ndarray, 
                                      peak_info: List[Dict[str, Any]]) -> List[float]:
        """Estimate initial parameters for BiGaussian peaks.
        
        BiGaussian uses separate left and right Gaussian widths for asymmetric peaks.
        Initial estimate starts with symmetric widths (width_L = width_R).
        
        Args:
            x (array-like): X-axis data
            y (array-like): Y-axis data
            peak_info (list): List of peak detection results
                
        Returns:
            list: Flat parameter list [amp1, ctr1, σ_L1, σ_R1, amp2, ctr2, σ_L2, σ_R2, ...]
            
        Raises:
            ValueError: If inputs are invalid
            
        Notes:
            Starts with symmetric widths (σ_L = σ_R). The optimizer will adjust
            them to match any asymmetry in the actual peak shape.
            Useful for tailing peaks in chromatography or mass spectrometry.
            
        Example:
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.18}]
            >>> params = ParameterEstimator.estimate_bigaussian_parameters(x, y, peak_info)
            >>> # Returns [100, 5.0, 0.5, 0.5] where both widths start equal
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape}, {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length")
        if len(x) == 0:
            raise ValueError("x and y arrays are empty")
        if not isinstance(peak_info, list) or len(peak_info) == 0:
            raise ValueError("peak_info must be a non-empty list")
        
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        if x_span <= 0:
            raise ValueError(f"x must be increasing")
        
        y_max = np.max(np.abs(y))
        if y_max == 0:
            y_max = 1.0
        
        FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        for i, peak in enumerate(peak_info):
            if not isinstance(peak, dict):
                raise ValueError(f"peak_info[{i}] must be a dict")
            if 'x' not in peak or 'y' not in peak:
                raise ValueError(f"peak_info[{i}] missing required keys")
            
            amplitude = max(abs(peak['y']), y_max * 0.01)
            center = float(peak['x'])
            
            if not (x[0] <= center <= x[-1]):
                warnings.warn(
                    f"Peak {i} center outside x-range. Clipping.",
                    UserWarning
                )
                center = np.clip(center, x[0], x[-1])
            
            # Start with symmetric widths
            fwhm = peak.get('width_half', x_span / 20)
            if fwhm <= 0:
                warnings.warn(
                    f"Peak {i} has invalid width. Using default.",
                    UserWarning
                )
                fwhm = x_span / 20
            
            sigma = max(fwhm * FWHM_TO_SIGMA, x_span * 0.001)
            
            params.extend([amplitude, center, sigma, sigma])  # Both widths equal initially
        
        return params
    
    @staticmethod
    def estimate_emg_parameters(x: np.ndarray, y: np.ndarray, 
                               peak_info: List[Dict[str, Any]]) -> List[float]:
        """Estimate initial parameters for EMG (Exponentially Modified Gaussian) peaks.
        
        EMG models peaks with exponential tailing, common in chromatography.
        The EMG is a convolution of a Gaussian with an exponential decay.
        
        Args:
            x (array-like): X-axis data
            y (array-like): Y-axis data
            peak_info (list): List of peak detection results
                
        Returns:
            list: Flat parameter list [amp1, ctr1, σ1, τ1, amp2, ctr2, σ2, τ2, ...]
                where τ (tau) is the exponential decay time constant
            
        Raises:
            ValueError: If inputs are invalid
            
        Notes:
            Initial strategy: Set τ (exponential decay constant) equal to σ.
            This provides a reasonable starting point for moderately tailed peaks.
            The optimizer will adjust τ to match the actual degree of tailing.
            
            Larger τ = more tailing; τ → 0 = pure Gaussian
            
        Example:
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.18}]
            >>> params = ParameterEstimator.estimate_emg_parameters(x, y, peak_info)
            >>> # Returns [100, 5.0, 0.5, 0.5] where sigma=tau=0.5
        """
        # Validate inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape}, {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length")
        if len(x) == 0:
            raise ValueError("x and y arrays are empty")
        if not isinstance(peak_info, list) or len(peak_info) == 0:
            raise ValueError("peak_info must be a non-empty list")
        
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        if x_span <= 0:
            raise ValueError(f"x must be increasing")
        
        y_max = np.max(np.abs(y))
        if y_max == 0:
            y_max = 1.0
        
        FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        for i, peak in enumerate(peak_info):
            if not isinstance(peak, dict):
                raise ValueError(f"peak_info[{i}] must be a dict")
            if 'x' not in peak or 'y' not in peak:
                raise ValueError(f"peak_info[{i}] missing required keys")
            
            amplitude = max(abs(peak['y']), y_max * 0.01)
            center = float(peak['x'])
            
            if not (x[0] <= center <= x[-1]):
                warnings.warn(
                    f"Peak {i} center outside x-range. Clipping.",
                    UserWarning
                )
                center = np.clip(center, x[0], x[-1])
            
            fwhm = peak.get('width_half', x_span / 20)
            if fwhm <= 0:
                warnings.warn(
                    f"Peak {i} has invalid width. Using default.",
                    UserWarning
                )
                fwhm = x_span / 20
            
            sigma = max(fwhm * FWHM_TO_SIGMA, x_span * 0.001)
            
            # Initial tau (exponential decay) set equal to sigma
            # This provides a moderate starting point for fitting
            tau = sigma
            
            params.extend([amplitude, center, sigma, tau])
        
        return params
    
    @staticmethod
    def estimate_parameters(x: np.ndarray, y: np.ndarray, 
                           peak_info: List[Dict[str, Any]], 
                           peak_type: str) -> List[float]:
        """Estimate parameters for any peak type (dispatcher method).
        
        This is the main entry point for parameter estimation. It validates
        the peak_type and dispatches to the appropriate estimator method.
        
        Args:
            x (array-like): X-axis data
            y (array-like): Y-axis data
            peak_info (list): List of peak detection results
            peak_type (str): Type of peak function. Options:
                - "Gaussian": 3 params/peak (amplitude, center, σ)
                - "Lorentzian": 3 params/peak (amplitude, center, γ)
                - "Voigt": 4 params/peak (amplitude, center, σ_G, γ_L)
                - "BiGaussian": 4 params/peak (amplitude, center, σ_L, σ_R)
                - "EMG": 4 params/peak (amplitude, center, σ, τ)
                
        Returns:
            list: Flat list of initial parameters for all peaks
            
        Raises:
            ValueError: If peak_type is invalid or inputs are malformed
            
        Notes:
            If peak_type is not recognized, defaults to Gaussian estimation
            with a warning. This maintains backward compatibility but alerts
            users to potential typos.
            
        Example:
            >>> x = np.linspace(0, 10, 100)
            >>> y = 100 * np.exp(-(x-5)**2 / (2*0.5**2))
            >>> peak_info = [{'x': 5.0, 'y': 100, 'width_half': 1.18}]
            >>> params = ParameterEstimator.estimate_parameters(x, y, peak_info, "Gaussian")
            >>> print(f"Estimated {len(params)} parameters")
            Estimated 3 parameters
        """
        # Validate peak_type
        valid_types = ["Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG"]
        if not isinstance(peak_type, str):
            raise ValueError(f"peak_type must be a string, got {type(peak_type)}")
        
        estimators = {
            "Gaussian": ParameterEstimator.estimate_gaussian_parameters,
            "Lorentzian": ParameterEstimator.estimate_lorentzian_parameters,
            "Voigt": ParameterEstimator.estimate_voigt_parameters,
            "BiGaussian": ParameterEstimator.estimate_bigaussian_parameters,
            "EMG": ParameterEstimator.estimate_emg_parameters
        }
        
        if peak_type not in estimators:
            warnings.warn(
                f"Unknown peak_type '{peak_type}'. Valid types: {valid_types}. "
                f"Defaulting to Gaussian estimation.",
                UserWarning
            )
            estimator = ParameterEstimator.estimate_gaussian_parameters
        else:
            estimator = estimators[peak_type]
        
        # Call the appropriate estimator (it will perform its own validation)
        return estimator(x, y, peak_info)
