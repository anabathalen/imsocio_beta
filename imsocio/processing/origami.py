"""
ORIGAMI-style aIMS/CIU Data Processing and Visualization
=========================================================

This module provides ORIGAMI-style data processing and visualization for
activation IMS (aIMS) and Collision-Induced Unfolding (CIU) experiments.

Includes helper functions for TWIM data processing, interpolation, and smoothing.

Classes
-------
ORIGAMISettings
    Configuration settings for ORIGAMI-style plots
ORIGAMIDataProcessor
    Data processing methods matching ORIGAMI software
ORIGAMIVisualizer
    2D heatmap visualization for aIMS/CIU data

Functions
---------
safe_float_conversion
    Safely convert values to float
remove_duplicate_values
    Remove duplicate values from arrays
interpolate_matrix
    Interpolate 2D matrix on finer grid
smooth_matrix_gaussian
    Apply Gaussian smoothing to matrix
smooth_matrix_savgol
    Apply Savitzky-Golay smoothing to matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Iterable
from io import BytesIO
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# Helper Functions for TWIM Data Processing
# ============================================================================

def safe_float_conversion(value) -> float:
    """Safely convert a value to float.
    
    Rules:
    - numbers -> float
    - strings -> float if parseable else 0.0; treat "", "nan", "null", "none" as 0.0
    - iterables -> first successfully converted element, otherwise 0.0
    - others -> 0.0
    
    Parameters
    ----------
    value : any
        Value to convert
        
    Returns
    -------
    float
        Converted float value or 0.0
    """
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if s == "" or s.lower() in {"nan", "null", "none"}:
                return 0.0
            return float(s)
        if hasattr(value, "__iter__") and not isinstance(value, str):
            for item in value:
                try:
                    return safe_float_conversion(item)
                except Exception:
                    continue
            return 0.0
        return 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


def remove_duplicate_values(
    values: Iterable[float], 
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove near-duplicate values, keeping the first occurrence.
    
    Parameters
    ----------
    values : iterable of float
        Values to check for duplicates
    tolerance : float, optional
        Tolerance for considering values as duplicates (default: 1e-6)
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (clean_values, removed_indices)
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return arr, np.array([], dtype=int)

    unique_mask = np.ones(arr.shape[0], dtype=bool)
    for i in range(1, arr.shape[0]):
        # check against previous kept values
        for j in range(i):
            if unique_mask[j] and abs(arr[i] - arr[j]) < tolerance:
                unique_mask[i] = False
                break
    removed = np.where(~unique_mask)[0]
    return arr[unique_mask], removed


def interpolate_matrix(
    ccs_values: np.ndarray,
    trap_cv_values: np.ndarray,
    intensity_matrix: np.ndarray,
    method: str = "linear",
    multiplier: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a CCS×TrapCV matrix on a finer regular grid.

    Inputs must be strictly increasing 1D arrays and a 2D matrix with shape
    (len(ccs_values), len(trap_cv_values)).
    
    Parameters
    ----------
    ccs_values : np.ndarray
        1D array of CCS values (strictly increasing)
    trap_cv_values : np.ndarray
        1D array of TrapCV values (strictly increasing)
    intensity_matrix : np.ndarray
        2D intensity matrix
    method : str, optional
        Interpolation method: 'linear' or 'cubic' (default: 'linear')
    multiplier : int, optional
        Grid resolution multiplier (default: 1)
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (ccs_new, trapcv_new, matrix_new). If multiplier == 1, returns inputs.
    """
    if multiplier <= 1:
        return ccs_values, trap_cv_values, intensity_matrix

    ccs_values = np.asarray(ccs_values, dtype=float)
    trap_cv_values = np.asarray(trap_cv_values, dtype=float)
    Z = np.asarray(intensity_matrix, dtype=float)

    if ccs_values.ndim != 1 or trap_cv_values.ndim != 1:
        raise ValueError("ccs_values and trap_cv_values must be 1D arrays")
    if Z.shape != (ccs_values.size, trap_cv_values.size):
        raise ValueError("intensity_matrix shape must match (len(ccs_values), len(trap_cv_values))")
    if np.any(np.diff(ccs_values) <= 0) or np.any(np.diff(trap_cv_values) <= 0):
        raise ValueError("ccs_values and trap_cv_values must be strictly increasing")

    n_ccs_new = int(ccs_values.size) * int(multiplier)
    n_trap_new = int(trap_cv_values.size) * int(multiplier)

    ccs_new = np.linspace(ccs_values.min(), ccs_values.max(), n_ccs_new)
    trap_new = np.linspace(trap_cv_values.min(), trap_cv_values.max(), n_trap_new)

    if method == "linear":
        interp_func = RegularGridInterpolator(
            (ccs_values, trap_cv_values), Z, method="linear", bounds_error=False, fill_value=0.0
        )
        T, C = np.meshgrid(trap_new, ccs_new)
        pts = np.column_stack([C.ravel(), T.ravel()])
        Z_new = interp_func(pts).reshape(n_ccs_new, n_trap_new)
    elif method == "cubic":
        kx = min(3, max(1, ccs_values.size - 1))
        ky = min(3, max(1, trap_cv_values.size - 1))
        spline = RectBivariateSpline(ccs_values, trap_cv_values, Z, kx=kx, ky=ky)
        Z_new = spline(ccs_new, trap_new)
    else:
        raise ValueError("method must be 'linear' or 'cubic'")

    return ccs_new, trap_new, Z_new


def smooth_matrix_gaussian(
    Z: np.ndarray, 
    sigma: float = 1.0, 
    truncate: float = 4.0
) -> np.ndarray:
    """Apply Gaussian smoothing to a 2D matrix.
    
    Parameters
    ----------
    Z : np.ndarray
        2D matrix to smooth
    sigma : float, optional
        Standard deviation for Gaussian kernel (default: 1.0)
    truncate : float, optional
        Truncate filter at this many standard deviations (default: 4.0)
        
    Returns
    -------
    np.ndarray
        Smoothed matrix
    """
    return ndimage.gaussian_filter(np.asarray(Z, dtype=float), sigma=sigma, truncate=truncate)


def smooth_matrix_savgol(
    Z: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "mirror",
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to rows then columns when valid.
    
    Parameters
    ----------
    Z : np.ndarray
        2D matrix to smooth
    window_length : int, optional
        Length of filter window (default: 11)
    polyorder : int, optional
        Order of polynomial (default: 3)
    mode : str, optional
        Mode for handling boundaries (default: 'mirror')
        
    Returns
    -------
    np.ndarray
        Smoothed matrix
    """
    Z = np.asarray(Z, dtype=float).copy()
    n_rows, n_cols = Z.shape

    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2 if polyorder is not None else 5)

    # rows
    if n_cols >= window_length:
        for i in range(n_rows):
            Z[i, :] = savgol_filter(Z[i, :], window_length=window_length, polyorder=polyorder, mode=mode)
    # cols
    if n_rows >= window_length:
        for j in range(n_cols):
            Z[:, j] = savgol_filter(Z[:, j], window_length=window_length, polyorder=polyorder, mode=mode)

    return Z


# ============================================================================
# ORIGAMI Classes
# ============================================================================


@dataclass
class ORIGAMISettings:
    """ORIGAMI-style configuration settings for aIMS/CIU plots.
    
    Attributes
    ----------
    grid_resolution : int
        Resolution of interpolation grid (default: 200)
    interpolation_method : str
        Interpolation method: 'cubic', 'linear', 'nearest' (default: 'cubic')
    normalize_data : bool
        Whether to normalize intensity data (default: True)
    normalization_mode : str
        Normalization mode: 'Maximum', 'Logarithmic', 'Natural log', 'Square root'
    apply_smoothing : bool
        Whether to apply smoothing to data (default: False)
    smoothing_type : str
        Smoothing type: 'gaussian' or 'savgol' (default: 'gaussian')
    gaussian_sigma : float
        Sigma parameter for Gaussian smoothing (default: 2.0)
    savgol_window : int
        Window size for Savitzky-Golay smoothing (default: 11)
    savgol_polyorder : int
        Polynomial order for Savitzky-Golay smoothing (default: 3)
    noise_threshold : float
        Threshold for noise reduction (default: 0.0)
    apply_intensity_threshold : bool
        Whether to apply intensity thresholds (default: False)
    intensity_min_threshold : float
        Minimum intensity threshold (default: 0.0)
    intensity_max_threshold : float
        Maximum intensity threshold (default: 1.0)
    colormap : str
        Matplotlib colormap name (default: 'viridis')
    show_colorbar : bool
        Whether to show colorbar (default: True)
    colorbar_position : str
        Colorbar position: 'right', 'left', 'top', 'bottom' (default: 'right')
    colorbar_width : float
        Colorbar width percentage (default: 5.0)
    colorbar_pad : float
        Padding between plot and colorbar (default: 0.05)
    font_size : int
        Font size for labels (default: 12)
    label_weight : bool
        Whether to use bold labels (default: False)
    tick_size : int
        Tick label size (default: 10)
    figure_size : float
        Figure size in inches (default: 8.0)
    dpi : int
        Figure DPI (default: 300)
    """
    
    # Grid and interpolation
    grid_resolution: int = 200
    interpolation_method: str = 'cubic'
    
    # Normalization and processing
    normalize_data: bool = True
    normalization_mode: str = 'Maximum'
    
    # Smoothing
    apply_smoothing: bool = False
    smoothing_type: str = 'gaussian'
    gaussian_sigma: float = 2.0
    savgol_window: int = 11
    savgol_polyorder: int = 3
    
    # Noise reduction
    noise_threshold: float = 0.0
    apply_intensity_threshold: bool = False
    intensity_min_threshold: float = 0.0
    intensity_max_threshold: float = 1.0
    
    # Plot appearance
    colormap: str = 'viridis'
    show_colorbar: bool = True
    colorbar_position: str = 'right'
    colorbar_width: float = 5.0
    colorbar_pad: float = 0.05
    
    # Labels and fonts
    font_size: int = 12
    label_weight: bool = False
    tick_size: int = 10
    
    # Figure
    figure_size: float = 8.0
    dpi: int = 300


class ORIGAMIDataProcessor:
    """ORIGAMI-style data processing methods.
    
    Provides static methods for loading, processing, and preparing
    aIMS/CIU data for visualization.
    
    Methods
    -------
    load_twim_extract(file)
        Load TWIMExtract file format
    normalize_intensities(intensities, mode)
        Normalize intensity array using specified mode
    apply_smoothing(grid_z, smoothing_type, **kwargs)
        Apply smoothing to 2D intensity grid
    create_interpolated_grid(drift_times, cvs, intensities, resolution, method)
        Create interpolated 2D grid from scattered data
    """
    
    @staticmethod
    def load_twim_extract(file) -> pd.DataFrame:
        """Load TWIMExtract file with ORIGAMI-style parsing.
        
        Parameters
        ----------
        file : file-like object
            Uploaded file containing TWIMExtract data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'Drift_Time', 'CV', 'Intensity'
            
        Raises
        ------
        ValueError
            If file format is invalid or $TrapCV: line not found
        """
        try:
            file.seek(0)
            content = file.read().decode('utf-8').strip()
            lines = content.split('\n')
            
            # Find collision voltage line
            cv_line_idx = None
            for i, line in enumerate(lines):
                if line.startswith('$TrapCV:'):
                    cv_line_idx = i
                    break
            
            if cv_line_idx is None:
                raise ValueError("Could not find $TrapCV: line in TWIMExtract file")
            
            # Extract collision voltages
            cv_line = lines[cv_line_idx]
            cv_values = [float(x.strip()) for x in cv_line.split(',')[1:] if x.strip()]
            
            # Parse data starting after CV line
            data_lines = lines[cv_line_idx + 1:]
            data_rows = []
            
            for line in data_lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= len(cv_values) + 1:
                        try:
                            drift_time = float(parts[0])
                            intensities = []
                            for val_str in parts[1:len(cv_values)+1]:
                                try:
                                    intensities.append(float(val_str))
                                except (ValueError, TypeError):
                                    intensities.append(0.0)
                            
                            # Create row for each CV value
                            for cv, intensity in zip(cv_values, intensities):
                                data_rows.append({
                                    'Drift_Time': drift_time,
                                    'CV': cv,
                                    'Intensity': intensity
                                })
                        except (ValueError, IndexError):
                            continue
            
            if not data_rows:
                raise ValueError("No valid data rows found in file")
            
            return pd.DataFrame(data_rows)
            
        except Exception as e:
            raise ValueError(f"Error loading TWIMExtract file: {str(e)}")
    
    @staticmethod
    def normalize_intensities(intensities: np.ndarray, mode: str = 'Maximum') -> np.ndarray:
        """Normalize intensity array using specified mode.
        
        Parameters
        ----------
        intensities : np.ndarray
            Input intensity array
        mode : str
            Normalization mode:
            - 'Maximum': Divide by maximum value
            - 'Logarithmic': Log10 transformation
            - 'Natural log': Natural log transformation
            - 'Square root': Square root transformation
            
        Returns
        -------
        np.ndarray
            Normalized intensity array
        """
        intensities = np.asarray(intensities, dtype=float)
        
        if mode == 'Maximum':
            max_val = np.max(intensities)
            if max_val > 0:
                return intensities / max_val
            return intensities
        
        elif mode == 'Logarithmic':
            # Add small epsilon to avoid log(0)
            return np.log10(intensities + 1e-10)
        
        elif mode == 'Natural log':
            return np.log(intensities + 1e-10)
        
        elif mode == 'Square root':
            return np.sqrt(np.maximum(intensities, 0))
        
        else:
            return intensities
    
    @staticmethod
    def apply_smoothing(
        grid_z: np.ndarray,
        smoothing_type: str = 'gaussian',
        gaussian_sigma: float = 2.0,
        savgol_window: int = 11,
        savgol_polyorder: int = 3
    ) -> np.ndarray:
        """Apply smoothing to 2D intensity grid.
        
        Parameters
        ----------
        grid_z : np.ndarray
            2D intensity grid
        smoothing_type : str
            'gaussian' or 'savgol'
        gaussian_sigma : float
            Sigma for Gaussian filter (default: 2.0)
        savgol_window : int
            Window size for Savitzky-Golay filter (default: 11)
        savgol_polyorder : int
            Polynomial order for Savitzky-Golay filter (default: 3)
            
        Returns
        -------
        np.ndarray
            Smoothed 2D intensity grid
        """
        if smoothing_type == 'gaussian':
            return gaussian_filter(grid_z, sigma=gaussian_sigma)
        
        elif smoothing_type == 'savgol':
            # Apply Savitzky-Golay filter to each axis
            smoothed = savgol_filter(grid_z, window_length=savgol_window, 
                                    polyorder=savgol_polyorder, axis=0)
            smoothed = savgol_filter(smoothed, window_length=savgol_window, 
                                   polyorder=savgol_polyorder, axis=1)
            return smoothed
        
        return grid_z
    
    @staticmethod
    def create_interpolated_grid(
        drift_times: np.ndarray,
        cvs: np.ndarray,
        intensities: np.ndarray,
        resolution: int = 200,
        method: str = 'cubic'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create interpolated 2D grid from scattered data.
        
        Parameters
        ----------
        drift_times : np.ndarray
            Drift time values
        cvs : np.ndarray
            Collision voltage values
        intensities : np.ndarray
            Intensity values
        resolution : int
            Grid resolution (default: 200)
        method : str
            Interpolation method: 'cubic', 'linear', 'nearest' (default: 'cubic')
            
        Returns
        -------
        tuple of (grid_x, grid_y, grid_z)
            grid_x : 2D array of drift times
            grid_y : 2D array of CVs
            grid_z : 2D array of interpolated intensities
        """
        # Create grid
        drift_min, drift_max = drift_times.min(), drift_times.max()
        cv_min, cv_max = cvs.min(), cvs.max()
        
        grid_drift = np.linspace(drift_min, drift_max, resolution)
        grid_cv = np.linspace(cv_min, cv_max, resolution)
        grid_x, grid_y = np.meshgrid(grid_drift, grid_cv)
        
        # Interpolate
        points = np.column_stack((drift_times, cvs))
        grid_z = griddata(points, intensities, (grid_x, grid_y), method=method, fill_value=0)
        
        return grid_x, grid_y, grid_z


class ORIGAMIVisualizer:
    """ORIGAMI-style 2D heatmap visualization for aIMS/CIU data.
    
    Methods
    -------
    create_plot(df, settings, **kwargs)
        Create complete ORIGAMI-style 2D heatmap plot
    """
    
    @staticmethod
    def create_plot(
        df: pd.DataFrame,
        settings: ORIGAMISettings,
        x_label: str = "Drift Time (ms)",
        y_label: str = "Collision Voltage (V)",
        title: Optional[str] = None,
        cmap_label: str = "Intensity",
        transparent_bg: bool = False
    ) -> Tuple[plt.Figure, BytesIO]:
        """Create ORIGAMI-style 2D heatmap plot.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: 'Drift_Time', 'CV', 'Intensity'
        settings : ORIGAMISettings
            Plot configuration settings
        x_label : str
            X-axis label (default: "Drift Time (ms)")
        y_label : str
            Y-axis label (default: "Collision Voltage (V)")
        title : str, optional
            Plot title
        cmap_label : str
            Colorbar label (default: "Intensity")
        transparent_bg : bool
            Whether to use transparent background (default: False)
            
        Returns
        -------
        tuple of (fig, buffer)
            fig : matplotlib.figure.Figure
                The created figure
            buffer : BytesIO
                PNG image buffer for download
        """
        # Extract data
        drift_times = df['Drift_Time'].values
        cvs = df['CV'].values
        intensities = df['Intensity'].values
        
        # Apply noise threshold
        if settings.noise_threshold > 0:
            intensities = np.where(intensities < settings.noise_threshold, 0, intensities)
        
        # Normalize
        if settings.normalize_data:
            intensities = ORIGAMIDataProcessor.normalize_intensities(
                intensities, settings.normalization_mode
            )
        
        # Apply intensity thresholds
        if settings.apply_intensity_threshold:
            intensities = np.clip(intensities, 
                                settings.intensity_min_threshold,
                                settings.intensity_max_threshold)
        
        # Create interpolated grid
        grid_x, grid_y, grid_z = ORIGAMIDataProcessor.create_interpolated_grid(
            drift_times, cvs, intensities,
            resolution=settings.grid_resolution,
            method=settings.interpolation_method
        )
        
        # Apply smoothing
        if settings.apply_smoothing:
            grid_z = ORIGAMIDataProcessor.apply_smoothing(
                grid_z,
                smoothing_type=settings.smoothing_type,
                gaussian_sigma=settings.gaussian_sigma,
                savgol_window=settings.savgol_window,
                savgol_polyorder=settings.savgol_polyorder
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(settings.figure_size, settings.figure_size),
                              dpi=settings.dpi)
        
        # Create heatmap
        im = ax.pcolormesh(grid_x, grid_y, grid_z, 
                          cmap=settings.colormap,
                          shading='auto')
        
        # Set labels
        weight = 'bold' if settings.label_weight else 'normal'
        ax.set_xlabel(x_label, fontsize=settings.font_size, weight=weight)
        ax.set_ylabel(y_label, fontsize=settings.font_size, weight=weight)
        
        if title:
            ax.set_title(title, fontsize=settings.font_size + 2, weight='bold')
        
        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=settings.tick_size)
        
        # Add colorbar
        if settings.show_colorbar:
            cbar = plt.colorbar(
                im, ax=ax,
                location=settings.colorbar_position,
                pad=settings.colorbar_pad,
                fraction=settings.colorbar_width / 100
            )
            cbar.set_label(cmap_label, fontsize=settings.font_size, weight=weight)
            cbar.ax.tick_params(labelsize=settings.tick_size)
        
        # Set background
        if transparent_bg:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=settings.dpi,
                   bbox_inches='tight', transparent=transparent_bg)
        buffer.seek(0)
        
        return fig, buffer
