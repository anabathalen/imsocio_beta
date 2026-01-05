"""Peak fitting module for IMSocio.

This module provides comprehensive peak fitting functionality including:
- Multiple peak functions (Gaussian, Lorentzian, Voigt, BiGaussian, EMG)
- Baseline correction options
- Parameter management and constraint handling
- Peak detection and parameter estimation
- Complete fitting engine with multiple optimization methods
- Data preprocessing and result analysis tools
"""

from .peak_functions import (
    gaussian_peak,
    lorentzian_peak,
    voigt_peak,
    bigaussian_peak,
    exponentially_modified_gaussian,
    asymmetric_gaussian,
    multi_peak_function,
    get_params_per_peak,
    get_parameter_names
)

from .baseline_functions import (
    linear_baseline,
    polynomial_baseline,
    exponential_baseline
)

from .peak_detection import PeakDetector
from .parameter_estimation import ParameterEstimator
from .parameter_manager import ParameterManager
from .fitting_engine import FittingEngine
from .data_processor import DataProcessor
from .result_analyzer import ResultAnalyzer
from .ccsd_processor import CCSDDataProcessor

__all__ = [
    # Peak functions
    'gaussian_peak',
    'lorentzian_peak',
    'voigt_peak',
    'bigaussian_peak',
    'exponentially_modified_gaussian',
    'asymmetric_gaussian',
    'multi_peak_function',
    'get_params_per_peak',
    'get_parameter_names',
    # Baseline functions
    'linear_baseline',
    'polynomial_baseline',
    'exponential_baseline',
    # Classes
    'PeakDetector',
    'ParameterEstimator',
    'ParameterManager',
    'FittingEngine',
    'DataProcessor',
    'ResultAnalyzer',
    'CCSDDataProcessor'
]

