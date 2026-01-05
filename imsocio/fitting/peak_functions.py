"""Peak function definitions for curve fitting.

This module provides various peak shape functions commonly used in
spectroscopic data analysis, similar to Origin Pro's peak fitting.

Supported peak types:
    - Gaussian: Symmetric bell-shaped curve
    - Lorentzian: Broader tails than Gaussian
    - Voigt: Convolution of Gaussian and Lorentzian (pseudo-Voigt approximation)
    - BiGaussian: Asymmetric peak with different left/right widths
    - EMG: Exponentially Modified Gaussian (for tailing peaks)
    - Asymmetric Gaussian: Gaussian with exponential asymmetry

All functions are vectorized and work with numpy arrays.
"""

import numpy as np
from scipy.special import erfc
import warnings
from typing import Union


def gaussian_peak(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Gaussian peak function.
    
    The Gaussian (normal distribution) is the most common peak shape in
    spectroscopy. The function is:
    
        y = A * exp(-0.5 * ((x - x0) / σ)²)
    
    where A is amplitude, x0 is center, and σ is the standard deviation (width).
    
    Args:
        x (array-like): Independent variable (e.g., m/z, drift time, wavelength)
        amplitude (float): Peak height (maximum y value)
        center (float): Peak center position (x value at maximum)
        width (float): Peak width parameter (standard deviation σ)
            FWHM ≈ 2.355 * width
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If width is zero or negative
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = gaussian_peak(x, amplitude=1.0, center=0.0, width=1.0)
        >>> print(f"Peak height: {y.max():.3f}")
        Peak height: 1.000
    
    Notes:
        - Width parameter is standard deviation (σ), not FWHM
        - FWHM = 2.355 * width (for reference)
        - Function is normalized to have maximum = amplitude
    """
    x = np.asarray(x)
    
    if width <= 0:
        raise ValueError(f"Gaussian width must be positive, got {width}")
    
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def lorentzian_peak(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Lorentzian (Cauchy) peak function.
    
    The Lorentzian has broader tails than Gaussian and is common in
    spectroscopy (e.g., natural line broadening). The function is:
    
        y = A / (1 + ((x - x0) / γ)²)
    
    where A is amplitude, x0 is center, and γ is the half-width at half-maximum (HWHM).
    
    Args:
        x (array-like): Independent variable
        amplitude (float): Peak height (maximum y value)
        center (float): Peak center position
        width (float): Half-width at half-maximum (HWHM, denoted γ)
            FWHM = 2 * width
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If width is zero or negative
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = lorentzian_peak(x, amplitude=1.0, center=0.0, width=0.5)
        >>> print(f"Peak height: {y.max():.3f}")
        Peak height: 1.000
    
    Notes:
        - Width parameter is HWHM (γ), not standard deviation
        - FWHM = 2 * width
        - Lorentzian has heavier tails than Gaussian
    """
    x = np.asarray(x)
    
    if width <= 0:
        raise ValueError(f"Lorentzian width must be positive, got {width}")
    
    return amplitude / (1 + ((x - center) / width) ** 2)


def voigt_peak(x: np.ndarray, amplitude: float, center: float, width_g: float, width_l: float) -> np.ndarray:
    """Pseudo-Voigt approximation.
    
    The Voigt profile is a convolution of Gaussian and Lorentzian functions,
    common in spectroscopy when both Gaussian (instrumental) and Lorentzian
    (natural) broadening occur. This uses the pseudo-Voigt approximation:
    
        V(x) = η * L(x) + (1 - η) * G(x)
    
    where η is a mixing parameter calculated from width ratio.
    
    Args:
        x (array-like): Independent variable
        amplitude (float): Peak height (maximum y value)
        center (float): Peak center position
        width_g (float): Gaussian component width (FWHM)
        width_l (float): Lorentzian component width (FWHM)
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If either width is zero or negative
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = voigt_peak(x, amplitude=1.0, center=0.0, width_g=0.5, width_l=0.5)
    
    Notes:
        - This is the pseudo-Voigt approximation, not true Voigt profile
        - Mixing parameter η calculated using polynomial approximation
        - Both width parameters are in FWHM units for consistency
        - η is clipped to [0, 1] range for stability
    
    References:
        Thompson, Cox & Hastings (1987), J. Appl. Cryst. 20, 79-83
    """
    x = np.asarray(x)
    
    if width_g <= 0:
        raise ValueError(f"Gaussian width must be positive, got {width_g}")
    if width_l <= 0:
        raise ValueError(f"Lorentzian width must be positive, got {width_l}")
    
    # Calculate mixing parameter η using polynomial approximation
    ratio = width_l / width_g
    eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    eta = np.clip(eta, 0, 1)
    
    # Calculate normalized Gaussian and Lorentzian components (both FWHM-based)
    gaussian = np.exp(-np.log(2) * ((x - center) / (width_g / 2)) ** 2)  # -ln(2) = -0.693147
    lorentzian = 1 / (1 + ((x - center) / (width_l / 2)) ** 2)
    
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)


def bigaussian_peak(x: np.ndarray, amplitude: float, center: float, width1: float, width2: float) -> np.ndarray:
    """Bi-Gaussian (split Gaussian) function with different widths on each side.
    
    Useful for asymmetric peaks with Gaussian-like shapes on both sides
    but different widths. The function is:
    
        y = A * exp(-0.5 * ((x - x0) / σ1)²)  for x ≤ x0
        y = A * exp(-0.5 * ((x - x0) / σ2)²)  for x > x0
    
    Args:
        x (array-like): Independent variable
        amplitude (float): Peak height (maximum y value)
        center (float): Peak center position (boundary between two Gaussians)
        width1 (float): Left-side width (standard deviation for x ≤ center)
        width2 (float): Right-side width (standard deviation for x > center)
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If either width is zero or negative
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = bigaussian_peak(x, amplitude=1.0, center=0.0, width1=0.5, width2=1.5)
        >>> # Peak is sharper on left (width1=0.5) and broader on right (width2=1.5)
    
    Notes:
        - Function is continuous at center point
        - width1 controls left tail, width2 controls right tail
        - Useful for chromatographic peaks with tailing
    """
    x = np.asarray(x)
    
    if width1 <= 0:
        raise ValueError(f"Left width (width1) must be positive, got {width1}")
    if width2 <= 0:
        raise ValueError(f"Right width (width2) must be positive, got {width2}")
    
    result = np.zeros_like(x, dtype=float)
    left_mask = x <= center
    right_mask = x > center
    
    if np.any(left_mask):
        result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / width1) ** 2)
    if np.any(right_mask):
        result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / width2) ** 2)
    
    return result


def exponentially_modified_gaussian(x: np.ndarray, amplitude: float, center: float, 
                                    width: float, tau: float) -> np.ndarray:
    """Exponentially Modified Gaussian (EMG) peak function.
    
    The EMG is a convolution of a Gaussian with an exponential decay,
    commonly used for chromatographic peaks with tailing. The analytical
    form is:
    
        EMG(x) = (Aλ/2) * exp(λ/2 * (2x0 + λσ² - 2x)) * erfc((x0 + λσ² - x) / (σ√2))
    
    where λ = 1/τ, σ = width/√2, x0 = center, A = amplitude.
    
    Args:
        x (array-like): Independent variable
        amplitude (float): Peak height (approximate, actual maximum depends on tau)
        center (float): Gaussian component center position
        width (float): Gaussian component width (standard deviation of Gaussian)
        tau (float): Exponential time constant (decay time)
            tau > 0: right-tailing (common in chromatography)
            tau < 0: left-tailing
            tau → 0: approaches pure Gaussian
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If width is zero or negative
        
    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = exponentially_modified_gaussian(x, amplitude=1.0, center=5.0, 
        ...                                     width=0.5, tau=0.5)
        >>> # Creates a Gaussian peak at x=5 with exponential tailing
    
    Notes:
        - Peak maximum shifts from 'center' due to exponential component
        - Larger |tau| creates more tailing
        - Width is Gaussian σ parameter (not FWHM)
        - Uses complementary error function (erfc) from scipy
        
    References:
        Grushka (1972), Anal. Chem. 44, 1733-1738
    """
    x = np.asarray(x)
    
    if width <= 0:
        raise ValueError(f"EMG width must be positive, got {width}")
    
    # Convert to standard EMG parameters
    sigma = width / np.sqrt(2)
    
    # Handle tau = 0 case (pure Gaussian)
    if abs(tau) < 1e-10:
        warnings.warn("tau is very close to zero, returning Gaussian peak")
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    
    lambda_param = 1.0 / tau
    
    # Calculate EMG with numerical stability improvements
    with np.errstate(over='ignore', under='ignore'):  # Handle overflow/underflow gracefully
        exponent = (lambda_param / 2) * (2 * center + lambda_param * sigma**2 - 2 * x)
        term1 = (lambda_param / 2) * np.exp(exponent)
        
        erfc_arg = (center + lambda_param * sigma**2 - x) / (sigma * np.sqrt(2))
        term2 = erfc(erfc_arg)
        
        result = amplitude * term1 * term2
        
        # Replace inf/nan with 0 for numerical stability
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result


def asymmetric_gaussian(x: np.ndarray, amplitude: float, center: float, 
                       width: float, asymmetry: float) -> np.ndarray:
    """Asymmetric Gaussian function with exponential modification.
    
    Combines Gaussian shape with exponential asymmetry term:
    
        y = A * exp(-0.5 * ((x - x0) / σ)²) * exp((x - x0) / a)
    
    where a is the asymmetry parameter.
    
    Args:
        x (array-like): Independent variable
        amplitude (float): Peak height (approximate)
        center (float): Peak center position
        width (float): Gaussian width (standard deviation)
        asymmetry (float): Asymmetry parameter
            asymmetry > 0: right-skewed (tailing to right)
            asymmetry < 0: left-skewed (tailing to left)
            asymmetry → ∞: approaches pure Gaussian
        
    Returns:
        np.ndarray: Array of y values with same shape as x
        
    Raises:
        ValueError: If width is zero or negative, or asymmetry is zero
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = asymmetric_gaussian(x, amplitude=1.0, center=0.0, 
        ...                         width=1.0, asymmetry=2.0)
        >>> # Creates right-skewed peak
    
    Notes:
        - Peak maximum shifts from 'center' due to asymmetry
        - Smaller |asymmetry| creates more skewness
        - Can produce very large values if asymmetry is small
    """
    x = np.asarray(x)
    
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if asymmetry == 0:
        raise ValueError("Asymmetry parameter cannot be zero")
    
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    exponential = np.exp((x - center) / asymmetry)
    
    with np.errstate(over='warn'):  # Warn on overflow
        result = amplitude * gaussian * exponential
        result = np.nan_to_num(result, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    
    return result


def multi_peak_function(x: np.ndarray, peak_type: str, *params) -> np.ndarray:
    """Multi-peak function supporting different peak types.
    
    Combines multiple peaks of the same type into a single function.
    Used for fitting overlapping peaks in spectroscopic data.
    
    Args:
        x (array-like): Independent variable
        peak_type (str): Type of peak function. Options:
            - "Gaussian": Symmetric Gaussian peaks (3 params each)
            - "Lorentzian": Lorentzian peaks (3 params each)
            - "Voigt": Voigt (pseudo-Voigt) peaks (4 params each)
            - "BiGaussian": Bi-Gaussian peaks (4 params each)
            - "EMG": Exponentially modified Gaussian (4 params each)
        *params: Flattened array of parameters for all peaks.
            For example, with 2 Gaussian peaks:
            (amp1, center1, width1, amp2, center2, width2)
        
    Returns:
        np.ndarray: Sum of all peak contributions
        
    Raises:
        ValueError: If peak_type is not recognized or params length is invalid
        
    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> # Two Gaussian peaks: peak1 at x=3, peak2 at x=7
        >>> params = [1.0, 3.0, 0.5,  # Peak 1: amp, center, width
        ...           0.8, 7.0, 0.7]  # Peak 2: amp, center, width
        >>> y = multi_peak_function(x, "Gaussian", *params)
    
    Notes:
        - All peaks must be of the same type
        - Parameters are passed in sequence: peak1_params, peak2_params, ...
        - Incomplete parameter sets (not evenly divisible) are ignored with warning
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    
    params_per_peak_map = {
        "Gaussian": 3,
        "Lorentzian": 3,
        "Voigt": 4,
        "BiGaussian": 4,
        "EMG": 4
    }
    
    if peak_type not in params_per_peak_map:
        raise ValueError(
            f"Unknown peak type '{peak_type}'. "
            f"Valid types: {list(params_per_peak_map.keys())}"
        )
    
    params_per_peak = params_per_peak_map[peak_type]
    
    if len(params) == 0:
        warnings.warn("No parameters provided, returning zero")
        return y
    
    if len(params) % params_per_peak != 0:
        warnings.warn(
            f"Parameter count ({len(params)}) not evenly divisible by "
            f"{params_per_peak} (params per {peak_type} peak). "
            f"Last {len(params) % params_per_peak} parameters will be ignored."
        )
    
    peak_functions = {
        "Gaussian": gaussian_peak,
        "Lorentzian": lorentzian_peak,
        "Voigt": voigt_peak,
        "BiGaussian": bigaussian_peak,
        "EMG": exponentially_modified_gaussian
    }
    
    peak_func = peak_functions[peak_type]
    
    # Sum contributions from all peaks
    num_peaks = len(params) // params_per_peak
    for i in range(num_peaks):
        start_idx = i * params_per_peak
        end_idx = start_idx + params_per_peak
        peak_params = params[start_idx:end_idx]
        
        try:
            y += peak_func(x, *peak_params)
        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"Peak {i+1} failed: {e}. Skipping this peak.")
            continue
    
    return y


def get_params_per_peak(peak_type: str) -> int:
    """Get number of parameters required for a peak type.
    
    Args:
        peak_type (str): Type of peak function (case-sensitive)
        
    Returns:
        int: Number of parameters per peak
            - Gaussian/Lorentzian: 3 (amplitude, center, width)
            - Voigt/BiGaussian/EMG: 4 (amplitude, center, width1, width2/tau)
        
    Raises:
        ValueError: If peak_type is not recognized
        
    Example:
        >>> n = get_params_per_peak("Gaussian")
        >>> print(n)
        3
        >>> n = get_params_per_peak("Voigt")
        >>> print(n)
        4
    """
    params_map = {
        "Gaussian": 3,
        "Lorentzian": 3,
        "Voigt": 4,
        "BiGaussian": 4,
        "EMG": 4
    }
    
    if peak_type not in params_map:
        raise ValueError(
            f"Unknown peak type '{peak_type}'. "
            f"Valid types: {list(params_map.keys())}"
        )
    
    return params_map[peak_type]


def get_parameter_names(peak_type: str) -> list:
    """Get parameter names for a peak type.
    
    Returns human-readable parameter names for a given peak function type.
    Useful for displaying results and creating parameter labels.
    
    Args:
        peak_type (str): Type of peak function (case-sensitive)
        
    Returns:
        list: List of parameter name strings
        
    Raises:
        ValueError: If peak_type is not recognized
        
    Example:
        >>> names = get_parameter_names("Gaussian")
        >>> print(names)
        ['Amplitude', 'Center', 'Width']
        >>> names = get_parameter_names("Voigt")
        >>> print(names)
        ['Amplitude', 'Center', 'Width_G', 'Width_L']
    
    Notes:
        - Width_G = Gaussian component width
        - Width_L = Lorentzian component width
        - Width1/Width2 = left/right widths for BiGaussian
        - Tau = exponential time constant for EMG
    """
    names_map = {
        "Gaussian": ["Amplitude", "Center", "Width"],
        "Lorentzian": ["Amplitude", "Center", "Width"],
        "Voigt": ["Amplitude", "Center", "Width_G", "Width_L"],
        "BiGaussian": ["Amplitude", "Center", "Width1", "Width2"],
        "EMG": ["Amplitude", "Center", "Width", "Tau"]
    }
    
    if peak_type not in names_map:
        raise ValueError(
            f"Unknown peak type '{peak_type}'. "
            f"Valid types: {list(names_map.keys())}"
        )
    
    return names_map[peak_type]
