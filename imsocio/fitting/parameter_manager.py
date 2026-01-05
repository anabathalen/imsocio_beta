"""Parameter manager module for constrained curve fitting.

This module provides the ParameterManager class for managing peak fitting
parameters with support for:
- Fixed parameters (user-specified constraints)
- Custom parameter bounds
- Automatic bound generation
- Parameter array reconstruction

The manager allows Origin-style parameter fixing where specific parameters
(e.g., peak centers) can be held constant during optimization.

Example:
    >>> from imsocio.fitting import ParameterManager
    >>> params = [100, 5.0, 0.5, 80, 7.0, 0.6]  # 2 Gaussian peaks
    >>> manager = ParameterManager("Gaussian", params, x_range=(0, 10))
    >>> manager.fix_parameter(0, 1)  # Fix first peak center
    >>> free_params, mapping = manager.get_fitting_parameters()
    >>> # free_params now excludes the fixed center
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional


class ParameterManager:
    """Manages parameters with bounds and constraints for peak fitting."""
    
    def __init__(self, peak_type: str, parameters: np.ndarray, x_range: Tuple[float, float]):
        """Initialize parameter manager with validation.
        
        Args:
            peak_type (str): Type of peak function. Options:
                - "Gaussian": 3 params per peak (amplitude, center, width)
                - "Lorentzian": 3 params per peak (amplitude, center, width)
                - "Voigt": 4 params per peak (amplitude, center, width_g, width_l)
                - "BiGaussian": 4 params per peak (amplitude, center, width1, width2)
                - "EMG": 4 params per peak (amplitude, center, width, tau)
            parameters (array-like): Flat array of initial parameters.
                Length must be multiple of params_per_peak.
            x_range (tuple): (x_min, x_max) for data range.
                Used for default center parameter bounds.
                
        Raises:
            ValueError: If peak_type is invalid, parameters length is wrong,
                or x_range is invalid.
                
        Example:
            >>> params = [100, 5.0, 0.5, 80, 7.0, 0.6]  # 2 Gaussian peaks
            >>> manager = ParameterManager("Gaussian", params, (0, 10))
            >>> print(f"Managing {manager.n_peaks} peaks")
            Managing 2 peaks
        """
        # Validate peak_type
        valid_types = ["Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG"]
        if peak_type not in valid_types:
            raise ValueError(
                f"Invalid peak_type '{peak_type}'. Valid options: {valid_types}"
            )
        
        # Convert and validate parameters
        parameters = np.asarray(parameters, dtype=float)
        if parameters.ndim != 1:
            raise ValueError(f"parameters must be 1D array, got shape {parameters.shape}")
        
        # Validate x_range
        if not isinstance(x_range, (tuple, list)) or len(x_range) != 2:
            raise ValueError(f"x_range must be tuple/list of length 2, got {x_range}")
        if x_range[1] <= x_range[0]:
            raise ValueError(
                f"x_range[1] must be > x_range[0], got {x_range}"
            )
        
        self.peak_type = peak_type
        self.parameters = parameters.copy()
        self.x_range = tuple(x_range)
        self.fixed_params: Dict[int, float] = {}  # Track which parameters are fixed
        self.param_bounds: Dict[int, Tuple[float, float]] = {}  # Track custom bounds
        self.params_per_peak = self.get_params_per_peak()
        
        # Validate parameters length
        if len(parameters) % self.params_per_peak != 0:
            raise ValueError(
                f"parameters length ({len(parameters)}) must be multiple of "
                f"{self.params_per_peak} for {peak_type} peaks. "
                f"Got {len(parameters) // self.params_per_peak} complete peaks "
                f"with {len(parameters) % self.params_per_peak} leftover parameters."
            )
        
        self.n_peaks = len(parameters) // self.params_per_peak
        
        if self.n_peaks == 0:
            raise ValueError("parameters array is empty or too short for at least one peak")
        
    def get_params_per_peak(self) -> int:
        """Get number of parameters per peak for different peak types.
        
        Returns:
            int: Number of parameters per peak:
                - Gaussian/Lorentzian: 3 (amplitude, center, width)
                - Voigt/BiGaussian/EMG: 4 (amplitude, center, width1, width2/tau)
                
        Raises:
            ValueError: If peak_type is not recognized.
        """
        params_dict = {
            "Gaussian": 3,
            "Lorentzian": 3,
            "Voigt": 4,
            "BiGaussian": 4,
            "EMG": 4
        }
        
        if self.peak_type not in params_dict:
            raise ValueError(
                f"Unknown peak_type '{self.peak_type}'. "
                f"Valid types: {list(params_dict.keys())}"
            )
        
        return params_dict[self.peak_type]
    
    def get_parameter_names(self) -> List[str]:
        """Get human-readable parameter names for each peak type.
        
        Returns:
            list: List of parameter name strings.
            
        Notes:
            - σ (sigma) = Gaussian standard deviation
            - γ (gamma) = Lorentzian HWHM
            - Width_G = Gaussian component width (Voigt)
            - Width_L = Lorentzian component width (Voigt)
            - Width_L/Width_R = left/right widths (BiGaussian)
            - Tau = exponential decay time constant (EMG)
            
        Example:
            >>> manager = ParameterManager("Voigt", [1, 5, 0.5, 0.3], (0, 10))
            >>> print(manager.get_parameter_names())
            ['Amplitude', 'Center', 'Width_G', 'Width_L']
        """
        names_dict = {
            "Gaussian": ["Amplitude", "Center", "Width (σ)"],
            "Lorentzian": ["Amplitude", "Center", "Width (γ)"],
            "Voigt": ["Amplitude", "Center", "Width_G", "Width_L"],
            "BiGaussian": ["Amplitude", "Center", "Width_L", "Width_R"],
            "EMG": ["Amplitude", "Center", "Width", "Tau"]
        }
        
        return names_dict.get(self.peak_type, ["Amplitude", "Center", "Width"])
    
    def update_parameter(self, peak_idx: int, param_idx: int, value: float) -> None:
        """Update a specific parameter value.
        
        Args:
            peak_idx (int): Peak index (0-based, 0 to n_peaks-1)
            param_idx (int): Parameter index within peak (0-based, 0 to params_per_peak-1)
            value (float): New parameter value
            
        Raises:
            ValueError: If indices are out of range
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.update_parameter(0, 0, 150)  # Update amplitude to 150
            >>> print(manager.parameters[0])
            150.0
        """
        if not 0 <= peak_idx < self.n_peaks:
            raise ValueError(
                f"peak_idx must be 0 to {self.n_peaks-1}, got {peak_idx}"
            )
        if not 0 <= param_idx < self.params_per_peak:
            raise ValueError(
                f"param_idx must be 0 to {self.params_per_peak-1}, got {param_idx}"
            )
        
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        self.parameters[global_param_idx] = float(value)
    
    def fix_parameter(self, peak_idx: int, param_idx: int, fixed: bool = True) -> None:
        """Fix or unfix a parameter at its current value during fitting.
        
        Args:
            peak_idx (int): Peak index (0-based, 0 to n_peaks-1)
            param_idx (int): Parameter index within peak (0-based, 0 to params_per_peak-1)
            fixed (bool): True to fix parameter, False to unfix. Default True.
            
        Raises:
            ValueError: If indices are out of range
            
        Notes:
            Fixed parameters are excluded from optimization and held constant.
            Common use case: fixing peak centers after initial detection.
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.fix_parameter(0, 1)  # Fix center
            >>> manager.is_parameter_fixed(0, 1)
            True
            >>> manager.fix_parameter(0, 1, fixed=False)  # Unfix center
            >>> manager.is_parameter_fixed(0, 1)
            False
        """
        if not 0 <= peak_idx < self.n_peaks:
            raise ValueError(
                f"peak_idx must be 0 to {self.n_peaks-1}, got {peak_idx}"
            )
        if not 0 <= param_idx < self.params_per_peak:
            raise ValueError(
                f"param_idx must be 0 to {self.params_per_peak-1}, got {param_idx}"
            )
        
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        if fixed:
            self.fixed_params[global_param_idx] = self.parameters[global_param_idx]
        else:
            self.fixed_params.pop(global_param_idx, None)
    
    def is_parameter_fixed(self, peak_idx: int, param_idx: int) -> bool:
        """Check if a parameter is fixed.
        
        Args:
            peak_idx (int): Peak index (0-based, 0 to n_peaks-1)
            param_idx (int): Parameter index within peak (0-based, 0 to params_per_peak-1)
            
        Returns:
            bool: True if parameter is fixed, False if free to vary
            
        Raises:
            ValueError: If indices are out of range
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.fix_parameter(0, 1)
            >>> manager.is_parameter_fixed(0, 1)
            True
            >>> manager.is_parameter_fixed(0, 0)
            False
        """
        if not 0 <= peak_idx < self.n_peaks:
            raise ValueError(
                f"peak_idx must be 0 to {self.n_peaks-1}, got {peak_idx}"
            )
        if not 0 <= param_idx < self.params_per_peak:
            raise ValueError(
                f"param_idx must be 0 to {self.params_per_peak-1}, got {param_idx}"
            )
        
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        return global_param_idx in self.fixed_params
    
    def set_parameter_bounds(self, peak_idx: int, param_idx: int, 
                            lower: float, upper: float) -> None:
        """Set custom bounds for a specific parameter.
        
        Args:
            peak_idx (int): Peak index (0-based, 0 to n_peaks-1)
            param_idx (int): Parameter index within peak (0-based, 0 to params_per_peak-1)
            lower (float): Lower bound for parameter
            upper (float): Upper bound for parameter
            
        Raises:
            ValueError: If indices are out of range, bounds are invalid,
                or current parameter value is outside bounds
                
        Notes:
            - lower_bound must be < upper_bound
            - Current parameter value should be within [lower_bound, upper_bound]
            - Bounds override default bounds for this parameter
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.set_parameter_bounds(0, 0, 50, 200)  # Amplitude: [50, 200]
            >>> manager.set_parameter_bounds(0, 2, 0.1, 2.0)  # Width: [0.1, 2.0]
        """
        if not 0 <= peak_idx < self.n_peaks:
            raise ValueError(
                f"peak_idx must be 0 to {self.n_peaks-1}, got {peak_idx}"
            )
        if not 0 <= param_idx < self.params_per_peak:
            raise ValueError(
                f"param_idx must be 0 to {self.params_per_peak-1}, got {param_idx}"
            )
        
        # Validate bounds
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"bounds must be finite, got lower={lower}, upper={upper}")
        if lower >= upper:
            raise ValueError(
                f"lower bound must be < upper bound, got lower={lower}, upper={upper}"
            )
        
        # Check current value is within bounds
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        current_value = self.parameters[global_param_idx]
        if not (lower <= current_value <= upper):
            warnings.warn(
                f"Current parameter value {current_value:.3e} is outside "
                f"specified bounds ({lower:.3e}, {upper:.3e}). "
                f"Consider adjusting initial value or bounds.",
                UserWarning
            )
        
        self.param_bounds[global_param_idx] = (float(lower), float(upper))
    
    def get_fitting_parameters(self) -> Tuple[np.ndarray, List[int]]:
        """Get free parameters for optimization, excluding fixed parameters.
        
        Returns:
            tuple: (free_params, param_mapping)
                - free_params (ndarray): Array of parameters to optimize
                - param_mapping (list): Indices mapping free params to full array
                    [global_idx_0, global_idx_1, ...]
                    
        Notes:
            Fixed parameters are excluded from optimization. After fitting,
            use reconstruct_full_parameters() to restore the complete array.
            
        Example:
            >>> params = [100, 5.0, 0.5, 80, 7.0, 0.6]  # 2 Gaussian peaks
            >>> manager = ParameterManager("Gaussian", params, (0, 10))
            >>> manager.fix_parameter(0, 1)  # Fix first peak center
            >>> free_params, mapping = manager.get_fitting_parameters()
            >>> print(len(free_params))  # 5 free params (6 total - 1 fixed)
            5
            >>> print(mapping)  # [0, 2, 3, 4, 5] - index 1 missing (fixed)
            [0, 2, 3, 4, 5]
        """
        free_params = []
        param_mapping = []
        
        for i, param in enumerate(self.parameters):
            if i not in self.fixed_params:
                free_params.append(param)
                param_mapping.append(i)
        
        return np.array(free_params), param_mapping
    
    def reconstruct_full_parameters(self, fitted_params: np.ndarray, 
                                    param_mapping: List[int]) -> np.ndarray:
        """Reconstruct full parameter array from optimized free parameters.
        
        Args:
            fitted_params (array-like): Optimized free parameter values from fitting
            param_mapping (list): Mapping from free param indices to global indices
                (obtained from get_fitting_parameters)
                
        Returns:
            ndarray: Complete parameter array with fixed values restored
            
        Raises:
            ValueError: If fitted_params length doesn't match param_mapping,
                or if mapping indices are invalid
                
        Notes:
            This method merges optimized free parameters with fixed parameter
            values to create a complete parameter array for peak functions.
            Fixed parameters are restored from self.fixed_params dict.
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.fix_parameter(0, 1)  # Fix center at 5.0
            >>> free_params, mapping = manager.get_fitting_parameters()
            >>> # After optimization, amplitude=120, width=0.6
            >>> optimized_free = np.array([120, 0.6])
            >>> full_params = manager.reconstruct_full_parameters(optimized_free, mapping)
            >>> print(full_params)  # [120, 5.0, 0.6] - center unchanged
            [120.   5.   0.6]
        """
        fitted_params = np.asarray(fitted_params, dtype=float)
        
        if len(fitted_params) != len(param_mapping):
            raise ValueError(
                f"fitted_params length ({len(fitted_params)}) must match "
                f"param_mapping length ({len(param_mapping)})"
            )
        
        # Validate mapping indices
        max_global_idx = len(self.parameters) - 1
        for mapped_idx in param_mapping:
            if not 0 <= mapped_idx <= max_global_idx:
                raise ValueError(
                    f"Invalid mapping: index {mapped_idx} out of range "
                    f"[0, {max_global_idx}]"
                )
        
        full_params = self.parameters.copy()
        
        for i, mapped_idx in enumerate(param_mapping):
            full_params[mapped_idx] = fitted_params[i]
        
        # Update fixed parameters with their fixed values
        for idx, value in self.fixed_params.items():
            full_params[idx] = value
        
        return full_params
    
    def get_bounds_for_fitting(self, param_mapping: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for free parameters (excluding fixed ones).
        
        Args:
            param_mapping (list): Indices mapping free params to full array
                (obtained from get_fitting_parameters)
                
        Returns:
            tuple: (bounds_lower, bounds_upper)
                - bounds_lower (ndarray): Lower bounds for each free parameter
                - bounds_upper (ndarray): Upper bounds for each free parameter
                
        Raises:
            ValueError: If param_mapping contains invalid indices
                
        Notes:
            Default bounds by parameter type:
            - Amplitude (param_idx=0): [0, max(10*amplitude, 1.0)]
            - Center (param_idx=1): [x_min, x_max] from x_range
            - Width parameters (param_idx>=2): [max(0.1*width, 1e-6), 10*width]
            
            Custom bounds override defaults if set via set_parameter_bounds().
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.set_parameter_bounds(0, 0, 50, 200)  # Custom amplitude bounds
            >>> free_params, mapping = manager.get_fitting_parameters()
            >>> lower, upper = manager.get_bounds_for_fitting(mapping)
            >>> print(f"Amplitude bounds: [{lower[0]}, {upper[0]}]")
            Amplitude bounds: [50, 200]
        """
        # Validate param_mapping
        if not param_mapping:
            warnings.warn(
                "param_mapping is empty - all parameters may be fixed. "
                "Returning empty bound arrays.",
                UserWarning
            )
            return np.array([]), np.array([])
        
        max_global_idx = len(self.parameters) - 1
        for mapped_idx in param_mapping:
            if not 0 <= mapped_idx <= max_global_idx:
                raise ValueError(
                    f"Invalid mapping: index {mapped_idx} out of range "
                    f"[0, {max_global_idx}]"
                )
        
        bounds_lower = []
        bounds_upper = []
        
        for mapped_idx in param_mapping:
            peak_idx = mapped_idx // self.params_per_peak
            param_idx = mapped_idx % self.params_per_peak
            
            # Check if custom bounds are set
            if mapped_idx in self.param_bounds:
                lower, upper = self.param_bounds[mapped_idx]
                bounds_lower.append(lower)
                bounds_upper.append(upper)
            else:
                # Use default bounds based on parameter type
                current_value = self.parameters[mapped_idx]
                
                if param_idx == 0:  # Amplitude
                    amp = abs(current_value)
                    bounds_lower.append(0.0)
                    bounds_upper.append(max(amp * 10, 1.0))
                elif param_idx == 1:  # Center
                    bounds_lower.append(self.x_range[0])
                    bounds_upper.append(self.x_range[1])
                else:  # Width parameters (sigma, gamma, tau, etc.)
                    width = abs(current_value)
                    if width < 1e-10:
                        # Handle near-zero widths
                        warnings.warn(
                            f"Parameter {mapped_idx} (peak {peak_idx}, param {param_idx}) "
                            f"has very small value {current_value:.3e}. "
                            f"Using default bounds [1e-6, 1.0].",
                            UserWarning
                        )
                        bounds_lower.append(1e-6)
                        bounds_upper.append(1.0)
                    else:
                        bounds_lower.append(max(width * 0.1, 1e-6))
                        bounds_upper.append(width * 10)
        
        return np.array(bounds_lower), np.array(bounds_upper)    
    def add_peak(self, peak_params: List[float]) -> None:
        """Add a new peak with specified parameters.
        
        Args:
            peak_params (list): Parameters for the new peak.
                Must have exactly params_per_peak elements.
                
        Raises:
            ValueError: If peak_params has wrong length
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> print(f"Initial peaks: {manager.n_peaks}")
            Initial peaks: 1
            >>> manager.add_peak([80, 7.0, 0.6])  # Add second peak
            >>> print(f"After adding: {manager.n_peaks}")
            After adding: 2
        """
        peak_params = np.asarray(peak_params, dtype=float)
        if len(peak_params) != self.params_per_peak:
            raise ValueError(
                f"peak_params must have {self.params_per_peak} elements "
                f"for {self.peak_type} peaks, got {len(peak_params)}"
            )
        
        self.parameters = np.concatenate([self.parameters, peak_params])
        self.n_peaks += 1
    
    def set_tight_bounds(self, peak_idx: int, param_idx: int, 
                        tolerance_percent: float = 1.0) -> None:
        """Set tight bounds around current value (useful for constraining parameters).
        
        Args:
            peak_idx (int): Peak index (0-based)
            param_idx (int): Parameter index within peak (0-based)
            tolerance_percent (float): Allowed variation as percentage of current value.
                Default 1.0 means ±1% bounds.
                
        Raises:
            ValueError: If indices are out of range or tolerance is invalid
            
        Example:
            >>> manager = ParameterManager("Gaussian", [100, 5, 0.5], (0, 10))
            >>> manager.set_tight_bounds(0, 1, tolerance_percent=1.0)  # ±1% on center
            >>> # Center can now only vary between 4.95 and 5.05
        """
        if not 0 <= peak_idx < self.n_peaks:
            raise ValueError(
                f"peak_idx must be 0 to {self.n_peaks-1}, got {peak_idx}"
            )
        if not 0 <= param_idx < self.params_per_peak:
            raise ValueError(
                f"param_idx must be 0 to {self.params_per_peak-1}, got {param_idx}"
            )
        if tolerance_percent <= 0:
            raise ValueError(
                f"tolerance_percent must be positive, got {tolerance_percent}"
            )
        
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        current_value = self.parameters[global_param_idx]
        
        # Calculate bounds
        delta = abs(current_value) * tolerance_percent / 100.0
        lower = current_value - delta
        upper = current_value + delta
        
        # Ensure bounds are in correct order
        if lower > upper:
            lower, upper = upper, lower
        
        # For parameters that must be positive (amplitude, width), enforce minimum
        if param_idx != 1:  # Not center (which can be any value in x_range)
            lower = max(lower, 1e-10)
            upper = max(upper, 1e-10)
        
        self.param_bounds[global_param_idx] = (float(lower), float(upper))