"""Mass spectrum visualization and processing module.

This module provides classes for handling mass spectrometry data,
including data validation, processing, annotation, and plotting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D
import os
import warnings


def validate_font(font_name: str, fallback: str = 'sans-serif') -> tuple[str, str]:
    """Validate if a font is available and return font name with warning message.
    
    Args:
        font_name: Requested font name
        fallback: Fallback font if requested font not found
        
    Returns:
        Tuple of (font_to_use, warning_message)
    """
    # Get all available font names
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    # Check if requested font is available
    if font_name in available_fonts:
        return font_name, ""
    
    # Check if it's a generic family (always works)
    generic_families = ['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy']
    if font_name.lower() in generic_families:
        return font_name, ""
    
    # Font not found, use fallback
    warning_msg = f"Font '{font_name}' not available on this system. Using '{fallback}' instead."
    return fallback, warning_msg


@dataclass
class SpectrumData:
    """Container for mass spectrum data with built-in validation."""
    
    mz: np.ndarray
    intensity: np.ndarray
    name: str = ""
    
    def __post_init__(self):
        """Validate and clean data after initialization."""
        self.mz = np.array(self.mz, dtype=np.float64)
        self.intensity = np.array(self.intensity, dtype=np.float64)
        self._validate_and_clean()
    
    def _validate_and_clean(self):
        """Clean and validate data to prevent astronomical values."""
        # Remove NaN and infinite values
        valid_mask = np.isfinite(self.mz) & np.isfinite(self.intensity)
        self.mz = self.mz[valid_mask]
        self.intensity = self.intensity[valid_mask]
        
        # Remove negative intensities
        positive_mask = self.intensity >= 0
        self.mz = self.mz[positive_mask]
        self.intensity = self.intensity[positive_mask]
        
        # Cap maximum intensity to prevent astronomical plots
        MAX_INTENSITY = 1e6
        if len(self.intensity) > 0:
            max_int = np.max(self.intensity)
            if max_int > MAX_INTENSITY:
                scale_factor = MAX_INTENSITY / max_int
                self.intensity = self.intensity * scale_factor
        
        # Sort by m/z
        if len(self.mz) > 0:
            sort_idx = np.argsort(self.mz)
            self.mz = self.mz[sort_idx]
            self.intensity = self.intensity[sort_idx]
    
    def filter_mz_range(self, mz_min: float, mz_max: float) -> 'SpectrumData':
        """Return filtered data within m/z range."""
        if len(self.mz) == 0:
            return SpectrumData([], [], self.name)
        
        mask = (self.mz >= mz_min) & (self.mz <= mz_max)
        return SpectrumData(self.mz[mask], self.intensity[mask], self.name)
    
    def get_max_intensity(self) -> float:
        """Get maximum intensity safely."""
        return np.max(self.intensity) if len(self.intensity) > 0 else 0.0
    
    def normalize(self, method: str = 'max', mz_range: Optional[tuple] = None):
        """Normalize spectrum data in-place.
        
        Args:
            method: Normalization method ('max' or 'sum')
            mz_range: Optional tuple (min_mz, max_mz) to normalize only to visible range.
                     If None, normalizes to entire spectrum.
        """
        if len(self.intensity) == 0:
            return
        
        # If mz_range provided, calculate normalization factor from visible range only
        if mz_range is not None:
            min_mz, max_mz = mz_range
            mask = (self.mz >= min_mz) & (self.mz <= max_mz)
            
            if not np.any(mask):
                # If no data in range, fall back to full spectrum
                visible_intensity = self.intensity
            else:
                visible_intensity = self.intensity[mask]
        else:
            visible_intensity = self.intensity
        
        if method == 'max':
            max_val = np.max(visible_intensity)
            if max_val > 0:
                self.intensity = self.intensity / max_val
        elif method == 'sum':
            sum_val = np.sum(visible_intensity)
            if sum_val > 0:
                self.intensity = self.intensity / sum_val


class SpectrumReader:
    """Read mass spectrum files with validation."""
    
    @staticmethod
    def _extract_charge_from_filename(filename: str) -> Optional[str]:
        """Extract charge state from filename to use as label.
        
        Args:
            filename: Name of the file
            
        Returns:
            String label with charge state if found, otherwise filename
        """
        import re
        
        # Try to extract charge state from various filename patterns
        # Pattern 1: Just a number like "14.csv" or "14.txt"
        stem = filename.replace('.csv', '').replace('.txt', '').replace('.TXT', '').replace('.CSV', '')
        try:
            charge = int(stem)
            if 1 <= charge <= 200:
                return f"z = {charge}"
        except ValueError:
            pass
        
        # Pattern 2: Contains _charge or _z followed by number
        patterns = [
            r'_charge[_-]?(\d+)',
            r'_z[_-]?(\d+)',
            r'charge[_-]?(\d+)',
            r'z[_-]?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"z = {match.group(1)}"
        
        # Pattern 3: Number at the end before extension
        match = re.search(r'_(\d+)(?:\.\w+)?$', filename)
        if match:
            charge = int(match.group(1))
            if 1 <= charge <= 200:
                return f"z = {charge}"
        
        # If no charge state found, return filename without extension
        return stem
    
    @staticmethod
    def read_file(file_or_path) -> SpectrumData:
        """Read spectrum file (file-like object or path) and return SpectrumData.
        
        Args:
            file_or_path: File-like object or string path to spectrum file
            
        Returns:
            SpectrumData object with validated m/z and intensity data
        """
        try:
            # Read CSV with flexible delimiter detection
            def _read_csv(src):
                return pd.read_csv(
                    src, sep=None, engine="python", header=None, 
                    on_bad_lines='skip', encoding='utf-8'
                )

            if hasattr(file_or_path, "read"):  # uploaded file-like
                df = _read_csv(file_or_path)
                filename = os.path.basename(getattr(file_or_path, 'name', 'spectrum'))
            else:  # path string
                df = _read_csv(file_or_path)
                filename = os.path.basename(file_or_path)

            # Extract charge state label from filename
            name = SpectrumReader._extract_charge_from_filename(filename)

            if len(df.columns) < 2:
                raise ValueError(f"File must have at least 2 columns, found {len(df.columns)}")

            mz_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
            intensity_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values

            min_len = min(len(mz_data), len(intensity_data))
            mz_data = mz_data[:min_len]
            intensity_data = intensity_data[:min_len]

            return SpectrumData(mz_data, intensity_data, name)
            
        except Exception as e:
            # Return empty spectrum on error
            filename = os.path.basename(getattr(file_or_path, 'name', str(file_or_path)))
            name = SpectrumReader._extract_charge_from_filename(filename)
            return SpectrumData([], [], name)


class SpectrumProcessor:
    """Process mass spectrum data with smoothing, baseline correction, etc."""
    
    @staticmethod
    def smooth(spectrum: SpectrumData, window: int = 5, order: int = 2) -> SpectrumData:
        """Apply Savitzky-Golay smoothing to spectrum.
        
        Args:
            spectrum: Input spectrum data
            window: Smoothing window length (must be odd, supports up to 101+)
            order: Polynomial order (must be less than window)
            
        Returns:
            New SpectrumData with smoothed intensities
        """
        if len(spectrum.intensity) <= 5:
            return SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
        
        # Ensure window is valid: odd, >= 3, <= data length
        window = min(window, len(spectrum.intensity) - 1)
        if window % 2 == 0:
            window -= 1  # Make it odd
        if window < 3:
            return SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
        
        # Ensure polynomial order is less than window
        order = min(order, window - 1)
        
        try:
            smoothed_intensity = savgol_filter(spectrum.intensity, window, order)
            result = SpectrumData(spectrum.mz.copy(), smoothed_intensity, spectrum.name)
            result._validate_and_clean()
            return result
        except:
            return SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
    
    @staticmethod
    def baseline_correct(spectrum: SpectrumData, percentile: float = 5) -> SpectrumData:
        """Apply baseline correction using percentile method.
        
        Args:
            spectrum: Input spectrum data
            percentile: Percentile value for baseline estimation
            
        Returns:
            New SpectrumData with baseline corrected
        """
        if len(spectrum.intensity) == 0:
            return SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
        
        try:
            baseline = np.percentile(spectrum.intensity, percentile)
            corrected_intensity = np.maximum(spectrum.intensity - baseline, 0)
            return SpectrumData(spectrum.mz.copy(), corrected_intensity, spectrum.name)
        except:
            return SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
    
    @staticmethod
    def process(spectrum: SpectrumData, options: Dict[str, Any]) -> SpectrumData:
        """Apply multiple processing steps based on options dictionary.
        
        Args:
            spectrum: Input spectrum data
            options: Dictionary with processing options. Should include 'x_min' and 'x_max' 
                    if normalizing to visible range.
            
        Returns:
            Processed SpectrumData
        """
        if len(spectrum.intensity) == 0:
            return spectrum
        
        processed = SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
        
        # Smoothing
        if options.get('smoothing', False):
            window = options.get('smooth_window', 5)
            order = options.get('smooth_order', 2)
            processed = SpectrumProcessor.smooth(processed, window, order)
        
        # Baseline correction
        if options.get('baseline_correction', False):
            percentile = options.get('baseline_percentile', 5)
            processed = SpectrumProcessor.baseline_correct(processed, percentile)
        
        # Normalization - always normalize to selected range
        if options.get('normalize', False):
            # Get mz range from options (x_min, x_max)
            x_min = options.get('x_min')
            x_max = options.get('x_max')
            
            if x_min is not None and x_max is not None:
                mz_range = (x_min, x_max)
            else:
                mz_range = None
            
            processed.normalize(options.get('normalize_type', 'max'), mz_range=mz_range)
        
        return processed


class PlotStyler:
    """Handle plot styling and axis configuration."""
    
    @staticmethod
    def apply_styling(ax: plt.Axes, settings: Dict[str, Any], is_bottom: bool = True):
        """Apply comprehensive styling to a matplotlib axes.
        
        Args:
            ax: Matplotlib axes to style
            settings: Dictionary with styling settings
            is_bottom: Whether this is a bottom axes (affects axis display)
        """
        # Grid
        if settings.get('show_grid', False):
            ax.grid(
                True,
                alpha=settings.get('grid_alpha', 0.3),
                linestyle=settings.get('grid_style', '--'),
                color=settings.get('grid_color', 'gray'),
                linewidth=0.5
            )
        
        # Axis labels
        if settings.get('show_x_label', True) and is_bottom:
            ax.set_xlabel(
                settings.get('x_label_text', 'm/z'),
                fontsize=settings.get('axis_label_size', 14),
                weight=settings.get('axis_label_weight', 'bold'),
                fontfamily=settings.get('font_family', 'Arial'),
                color=settings.get('axis_label_color', 'black')
            )
        
        if settings.get('show_y_label', True):
            ax.set_ylabel(
                settings.get('y_label_text', 'Intensity'),
                fontsize=settings.get('axis_label_size', 14),
                weight=settings.get('axis_label_weight', 'bold'),
                fontfamily=settings.get('font_family', 'Arial'),
                color=settings.get('axis_label_color', 'black')
            )
        
        # Tick labels
        if not settings.get('show_x_tick_labels', True):
            ax.set_xticklabels([])
        if not settings.get('show_y_tick_labels', True):
            ax.set_yticklabels([])
        
        # Tick parameters
        tick_color = settings.get('tick_color', 'black')
        tick_label_size = settings.get('tick_label_size', 12)
        ax.tick_params(
            labelsize=tick_label_size,
            bottom=settings.get('show_bottom_axis', True),
            top=settings.get('show_top_axis', False),
            left=settings.get('show_left_axis', True),
            right=settings.get('show_right_axis', False),
            color=tick_color,
            labelcolor=tick_color
        )
        
        # Apply font family and size explicitly to tick labels
        font_family = settings.get('font_family', 'sans-serif')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(tick_label_size)
        
        # Hide y-ticks if requested
        if settings.get('hide_y_ticks', False):
            ax.set_yticks([])
        elif settings.get('y_tick_count'):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=settings['y_tick_count']))
        
        # Custom x-ticks
        if settings.get('custom_x_ticks', False):
            spacing = settings.get('x_tick_spacing', 100)
            x_min, x_max = ax.get_xlim()
            ax.xaxis.set_major_locator(MultipleLocator(spacing))
        
        # Spines
        spine_color = settings.get('tick_color', settings.get('spine_color', 'black'))
        for spine in ax.spines.values():
            spine.set_linewidth(settings.get('spine_width', 1.5))
            spine.set_color(spine_color)
    
    @staticmethod
    def get_color_palette(palette_name: str, n_colors: int) -> List[str]:
        """Get color palette for multiple spectra.
        
        Args:
            palette_name: Name of seaborn palette
            n_colors: Number of colors needed
            
        Returns:
            List of color hex codes
        """
        try:
            import seaborn as sns
            return sns.color_palette(palette_name, n_colors).as_hex()
        except:
            # Fallback colors
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            return (default_colors * ((n_colors // len(default_colors)) + 1))[:n_colors]


class SpectrumAnnotator:
    """Handle spectrum annotations and labels."""
    
    @staticmethod
    def add_annotations(
        ax: plt.Axes,
        spectrum: SpectrumData,
        annotations: List[Dict[str, Any]],
        y_max: float,
        baseline: float = 0.0,
        global_max_intensity: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        zoom: float = 1.0,
        baseline_offset: float = 0.0,
        settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[List, List]:
        """Add annotations to spectrum plot and return legend entries.
        
        Args:
            ax: Matplotlib axes
            spectrum: Spectrum data
            annotations: List of annotation dictionaries
            y_max: Maximum y-axis value
            baseline: Baseline offset for the spectrum (for stacked plots)
            global_max_intensity: Maximum intensity across all spectra (for consistent spacing)
            x_min: Minimum x-axis value (for calculating max in visible range)
            x_max: Maximum x-axis value (for calculating max in visible range)
            zoom: Zoom factor applied to intensities
            baseline_offset: Baseline offset value (negative for downward offset)
            settings: Plot settings dictionary (for font family and weight)
            
        Returns:
            Tuple of (legend_handles, legend_labels)
        """
        if not annotations or len(spectrum.intensity) == 0:
            return [], []

        legend_handles = []
        legend_labels = []
        seen_labels = set()
        
        # Use global max if provided, otherwise use spectrum's max
        if global_max_intensity is not None:
            max_intensity = global_max_intensity
        else:
            # Calculate max intensity in visible range if x limits provided
            if x_min is not None and x_max is not None:
                visible_mask = (spectrum.mz >= x_min) & (spectrum.mz <= x_max)
                if np.any(visible_mask):
                    max_intensity = np.max(spectrum.intensity[visible_mask])
                else:
                    max_intensity = spectrum.get_max_intensity()
            else:
                max_intensity = spectrum.get_max_intensity()
        
        # Apply zoom to max intensity for consistent spacing
        max_intensity = max_intensity * zoom
        
        for annotation in annotations:
            if annotation.get('mode') == 'manual':
                target_mz = annotation['mz']
                tolerance = annotation['tolerance']
                
                # Find peak in range
                mask = (spectrum.mz >= target_mz - tolerance) & (spectrum.mz <= target_mz + tolerance)
                if not np.any(mask):
                    continue
                
                peak_intensities = spectrum.intensity[mask]
                peak_mzs = spectrum.mz[mask]
                
                if len(peak_intensities) == 0:
                    continue
                
                max_idx = np.argmax(peak_intensities)
                peak_mz = peak_mzs[max_idx]
                peak_intensity = peak_intensities[max_idx]
                
                # Find maximum intensity in ±50 m/z region around the peak
                region_mask = (spectrum.mz >= peak_mz - 50) & (spectrum.mz <= peak_mz + 50)
                if np.any(region_mask):
                    region_max_intensity = np.max(spectrum.intensity[region_mask])
                else:
                    region_max_intensity = peak_intensity
                
                # Apply zoom and baseline offset to the intensities
                peak_intensity = (peak_intensity * zoom) + baseline_offset + baseline
                region_max_intensity = (region_max_intensity * zoom) + baseline_offset + baseline
                
                # Position annotations relative to the maximum intensity
                # Use larger spacing (20% and 25%) to ensure visibility, especially for normalized data
                spacing = 0.2 * max_intensity  # 20% of max for marker offset
                marker_y = region_max_intensity + spacing
                label_y = marker_y + (0.05 * max_intensity)  # Additional 5% for label
                
                # Cap at reasonable positions
                marker_y = min(marker_y, y_max * 0.9)
                label_y = min(label_y, y_max * 0.95)
                
                # Draw annotation line
                if annotation.get('show_line', True):
                    ax.plot([peak_mz, peak_mz], [peak_intensity, marker_y], 
                           color=annotation['color'], linewidth=1, alpha=0.7)
                
                # Draw marker
                ax.scatter([peak_mz], [marker_y], color=annotation['color'], 
                          marker=annotation['shape'], s=annotation.get('size', 80), 
                          edgecolor='black', linewidth=0.5, zorder=5)
                
                # Label configuration
                font_family = settings.get('font_family', 'Arial') if settings else 'Arial'
                annotation_weight = settings.get('annotation_weight', 'bold') if settings else 'bold'
                annotation_font_size = annotation.get('font_size', 12)
                
                label_kwargs = {
                    'ha': 'center', 
                    'va': 'bottom', 
                    'fontsize': annotation_font_size,
                    'color': annotation['color'], 
                    'weight': annotation_weight,
                    'fontfamily': font_family,
                    'fontname': font_family  # Ensure fontname is also set
                }
                
                if annotation.get('show_label_border', False):
                    label_kwargs['bbox'] = dict(
                        boxstyle="round,pad=0.3",
                        facecolor=annotation.get('label_bg_color', 'white'),
                        edgecolor=annotation['color'],
                        linewidth=0.5,
                        alpha=annotation.get('label_alpha', 0.8)
                    )
                
                # Add label text
                label_text = annotation.get('label', f"{peak_mz:.2f}")
                if annotation.get('show_mz_value', False):
                    label_text = f"{peak_mz:.2f}"
                
                ax.text(peak_mz, label_y, label_text, **label_kwargs)
                
                # Add to legend if enabled
                if annotation.get('add_to_legend', False):
                    legend_label = annotation.get('legend_label', label_text)
                    if legend_label not in seen_labels:
                        seen_labels.add(legend_label)
                        handle = Line2D(
                            [0], [0],
                            marker=annotation['shape'],
                            color='w',
                            markerfacecolor=annotation['color'],
                            markersize=8,
                            markeredgecolor='black',
                            markeredgewidth=0.5
                        )
                        legend_handles.append(handle)
                        legend_labels.append(legend_label)
        
        return legend_handles, legend_labels
    
    @staticmethod
    def add_vertical_lines(
        ax: plt.Axes,
        vertical_lines: List[Dict[str, Any]],
        plot_settings: Dict[str, Any],
        plot_type: str = "single",
        spectrum_index: int = 0,
        stack_offset: float = 1.2
    ):
        """Add vertical reference lines to plot.
        
        Args:
            ax: Matplotlib axes
            vertical_lines: List of vertical line configurations
            plot_settings: Plot settings dictionary
            plot_type: Type of plot (single, stacked, etc.)
            spectrum_index: Index of spectrum for stacked plots
            stack_offset: Offset between stacked spectra
        """
        if not vertical_lines:
            return
        
        for vline in vertical_lines:
            if not vline.get('enabled', True):
                continue
            
            x_pos = vline['position']
            
            # Determine y-range based on plot type
            if plot_type == "stacked":
                y_start = spectrum_index * stack_offset
                y_end = y_start + 1.0
            else:
                y_start, y_end = ax.get_ylim()
            
            ax.axvline(
                x=x_pos,
                ymin=(y_start - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                ymax=(y_end - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                color=vline.get('color', 'red'),
                linestyle=vline.get('style', '--'),
                linewidth=vline.get('width', 1.5),
                alpha=vline.get('alpha', 0.7),
                label=vline.get('label')
            )


class MassSpectrumPlotter:
    """Create publication-quality mass spectrum plots."""
    
    @staticmethod
    def calculate_y_limits(
        spectra: List[SpectrumData],
        plot_type: str,
        custom_y_min: Optional[float] = None,
        custom_y_max: Optional[float] = None,
        zoom: float = 1.4,
        stack_offset: float = 1.2,
        preserve_baseline: bool = False,
        baseline_offset: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate safe Y-axis limits for plot.
        
        Args:
            spectra: List of SpectrumData objects
            plot_type: Type of plot (single, stacked, overlay, mirror)
            custom_y_min: Custom minimum Y value
            custom_y_max: Custom maximum Y value
            zoom: Zoom factor for Y-axis (can be up to 50x or more)
            stack_offset: Offset between stacked spectra
            preserve_baseline: Whether to preserve zero baseline
            baseline_offset: Fixed amount to offset baseline below zero (e.g., 100 for 100 units below)
            
        Returns:
            Tuple of (y_min, y_max)
        """
        if not spectra or all(len(s.intensity) == 0 for s in spectra):
            return 0, 1000
        
        max_intensity = max([s.get_max_intensity() for s in spectra if len(s.intensity) > 0])
        
        if max_intensity <= 0:
            return 0, 1000
        
        # Calculate baseline offset (negative moves down)
        offset = -baseline_offset if not preserve_baseline else 0
        
        # Calculate limits based on plot type
        if plot_type == "single":
            if custom_y_min is not None and custom_y_max is not None:
                # Validate custom range
                if (np.isfinite(custom_y_min) and np.isfinite(custom_y_max) and 
                    custom_y_max > custom_y_min and 
                    abs(custom_y_max - custom_y_min) < 1e6 and
                    abs(custom_y_max) < 1e6 and abs(custom_y_min) < 1e6):
                    return float(custom_y_min), float(custom_y_max)
            
            y_min = offset
            y_max = max_intensity
            
        elif plot_type == "stacked":
            num_spectra = len([s for s in spectra if len(s.intensity) > 0])
            # Stacked plots always need baseline at zero for proper visualization
            y_min = -0.1
            y_max = 1.0 + (num_spectra - 1) * min(stack_offset, 2.0) + 0.5
            
        elif plot_type == "overlay":
            y_min = offset
            y_max = max_intensity
            
        elif plot_type == "mirror":
            y_min = -max_intensity
            y_max = max_intensity
        
        else:
            y_min = offset
            y_max = max_intensity
        
        # Final safety checks
        y_min = np.clip(y_min, -1e6, 1e6)
        y_max = np.clip(y_max, -1e6, 1e6)
        
        if y_max <= y_min:
            y_max = y_min + 1
        
        if abs(y_max - y_min) > 1e6:
            y_max = y_min + 1e6
        
        return float(y_min), float(y_max)
    
    @staticmethod
    def create_plot(
        spectra: List[SpectrumData],
        plot_settings: Dict[str, Any],
        annotations: Optional[List[Dict[str, Any]]] = None,
        plot_type: str = "single",
        spectrum_labels: Optional[List[str]] = None,
        vertical_lines: Optional[List[Dict[str, Any]]] = None,
        spectrum_colors: Optional[List[str]] = None
    ) -> Optional[plt.Figure]:
        """Create a complete mass spectrum plot.
        
        Args:
            spectra: List of SpectrumData objects
            plot_settings: Dictionary with all plot settings
            annotations: List of annotation configurations
            plot_type: Type of plot (single, stacked, overlay, mirror)
            spectrum_labels: Optional labels for each spectrum
            vertical_lines: Optional vertical reference lines
            spectrum_colors: Optional custom colors for each spectrum
            
        Returns:
            Tuple of (Matplotlib Figure object or None on error, font warning message)
        """
        if not spectra:
            return None, ""
        
        # Validate font availability
        requested_font = plot_settings.get('font_family', 'sans-serif')
        font_to_use, font_warning = validate_font(requested_font, fallback='sans-serif')
        plot_settings['font_family'] = font_to_use
        
        annotations = annotations or []
        spectrum_labels = spectrum_labels or [s.name for s in spectra]
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(plot_settings.get('width', 10), plot_settings.get('height', 6)),
            dpi=plot_settings.get('dpi', 300)
        )
        
        # Set background
        bg_color = plot_settings.get('background', 'white')
        if bg_color != 'transparent':
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
        
        # Filter spectra by m/z range
        x_min = plot_settings.get('x_min')
        x_max = plot_settings.get('x_max')
        if x_min is not None and x_max is not None:
            spectra = [s.filter_mz_range(x_min, x_max) for s in spectra]
        
        # Get colors
        if spectrum_colors is None:
            if len(spectra) > 1:
                spectrum_colors = PlotStyler.get_color_palette(
                    plot_settings.get('palette', 'Set2'), len(spectra)
                )
            else:
                spectrum_colors = [plot_settings.get('line_color', 'blue')]
        
        # Calculate y-limits
        y_min, y_max = MassSpectrumPlotter.calculate_y_limits(
            spectra,
            plot_type,
            plot_settings.get('y_min'),
            plot_settings.get('y_max'),
            plot_settings.get('zoom', 1.4),
            plot_settings.get('stack_offset', 1.2),
            plot_settings.get('preserve_baseline', False),
            plot_settings.get('baseline_offset', 0.0)
        )
        
        # Calculate baseline offset for annotations
        max_intensity = max([s.get_max_intensity() for s in spectra if len(s.intensity) > 0])
        preserve_baseline = plot_settings.get('preserve_baseline', False)
        baseline_offset_value = plot_settings.get('baseline_offset', 0.0)
        baseline_offset = -baseline_offset_value if not preserve_baseline else 0
        zoom = plot_settings.get('zoom', 1.0)
        
        # Track legend handles and labels from annotations
        annotation_legend_handles = []
        annotation_legend_labels = []
        
        # Plot based on type
        if plot_type == "single":
            MassSpectrumPlotter._plot_single(ax, spectra[0], plot_settings, spectrum_colors[0])
            if annotations:
                handles, labels = SpectrumAnnotator.add_annotations(
                    ax, spectra[0], annotations, y_max,
                    x_min=x_min, x_max=x_max,
                    zoom=zoom,
                    baseline_offset=baseline_offset,
                    settings=plot_settings
                )
                annotation_legend_handles.extend(handles)
                annotation_legend_labels.extend(labels)
        
        elif plot_type == "stacked":
            handles, labels = MassSpectrumPlotter._plot_stacked(
                ax, spectra, plot_settings, spectrum_labels, spectrum_colors, annotations
            )
            annotation_legend_handles.extend(handles)
            annotation_legend_labels.extend(labels)
        
        elif plot_type == "overlay":
            MassSpectrumPlotter._plot_overlay(
                ax, spectra, plot_settings, spectrum_labels, spectrum_colors
            )
            # Add annotations for overlay plots (use first spectrum for reference)
            if annotations and len(spectra) > 0:
                handles, labels = SpectrumAnnotator.add_annotations(
                    ax, spectra[0], annotations, y_max,
                    x_min=x_min, x_max=x_max,
                    zoom=zoom,
                    baseline_offset=baseline_offset,
                    settings=plot_settings
                )
                annotation_legend_handles.extend(handles)
                annotation_legend_labels.extend(labels)
        
        elif plot_type == "mirror":
            MassSpectrumPlotter._plot_mirror(
                ax, spectra, plot_settings, spectrum_labels, spectrum_colors
            )
            # Add annotations for mirror plots (use first spectrum for reference)
            if annotations and len(spectra) > 0:
                handles, labels = SpectrumAnnotator.add_annotations(
                    ax, spectra[0], annotations, y_max,
                    zoom=zoom,
                    baseline_offset=baseline_offset,
                    settings=plot_settings
                )
                annotation_legend_handles.extend(handles)
                annotation_legend_labels.extend(labels)
        
        # Add vertical lines
        if vertical_lines:
            SpectrumAnnotator.add_vertical_lines(
                ax, vertical_lines, plot_settings, plot_type, 0, 
                plot_settings.get('stack_offset', 1.2)
            )
        
        # Set limits
        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Apply styling
        PlotStyler.apply_styling(ax, plot_settings)
        
        # Title
        if plot_settings.get('title'):
            ax.set_title(
                plot_settings['title'],
                fontsize=plot_settings.get('title_font_size', 16),
                weight=plot_settings.get('title_weight', 'bold'),
                fontfamily=plot_settings.get('font_family', 'Arial')
            )
        
        # Legend
        if plot_settings.get('show_legend', False):
            # Only show annotation legend entries (protein masses), not file names
            # If there are annotation entries, show only those
            if annotation_legend_handles:
                legend_font_size = plot_settings.get('tick_label_size', 12)
                ax.legend(
                    annotation_legend_handles, 
                    annotation_legend_labels,
                    loc=plot_settings.get('legend_pos', 'best'),
                    frameon=plot_settings.get('legend_frame', True),
                    prop={
                        'family': plot_settings.get('font_family', 'Arial'),
                        'size': legend_font_size
                    }
                )
        
        plt.tight_layout()
        return fig, font_warning
    
    @staticmethod
    def _plot_single(ax: plt.Axes, spectrum: SpectrumData, settings: Dict, color: str):
        """Plot a single spectrum."""
        zoom = settings.get('zoom', 1.0)
        ax.plot(
            spectrum.mz,
            spectrum.intensity * zoom,
            color=color,
            linewidth=settings.get('line_width', 1.5),
            linestyle=settings.get('line_style', '-'),
            alpha=settings.get('alpha', 1.0),
            label=spectrum.name
        )
        
        if settings.get('fill_under', False):
            ax.fill_between(
                spectrum.mz,
                spectrum.intensity * zoom,
                alpha=settings.get('fill_alpha', 0.3),
                color=color
            )
    
    @staticmethod
    def _plot_stacked(
        ax: plt.Axes,
        spectra: List[SpectrumData],
        settings: Dict,
        labels: List[str],
        colors: List[str],
        annotations: List
    ) -> Tuple[List, List]:
        """Plot multiple spectra in stacked format.
        
        Returns:
            Tuple of (legend_handles, legend_labels) from annotations
        """
        stack_offset = settings.get('stack_offset', 1.2)
        
        for i, (spectrum, label, color) in enumerate(zip(spectra, labels, colors)):
            if len(spectrum.intensity) == 0:
                continue
            
            # Normalize to 0-1
            norm_spectrum = SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
            norm_spectrum.normalize('max')
            
            # Apply offset
            offset_intensity = norm_spectrum.intensity + (i * stack_offset)
            
            ax.plot(
                norm_spectrum.mz,
                offset_intensity,
                color=color,
                linewidth=settings.get('line_width', 1.5),
                alpha=settings.get('alpha', 1.0),
                label=label
            )
            
            if settings.get('fill_under', False):
                ax.fill_between(
                    norm_spectrum.mz,
                    i * stack_offset,
                    offset_intensity,
                    alpha=settings.get('fill_alpha', 0.2),
                    color=color
                )
        
        # Add annotations for each spectrum separately
        all_legend_handles = []
        all_legend_labels = []
        
        if annotations and len(spectra) > 0:
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            
            # Calculate global maximum intensity in visible range (before normalization)
            global_max = 0.0
            for spectrum in spectra:
                if len(spectrum.intensity) > 0:
                    visible_mask = (spectrum.mz >= x_min) & (spectrum.mz <= x_max)
                    if np.any(visible_mask):
                        spec_max = np.max(spectrum.intensity[visible_mask])
                        global_max = max(global_max, spec_max)
            
            # Since all spectra are normalized to 1.0, use 1.0 as the reference
            global_max_for_spacing = 1.0
            
            for i, spectrum in enumerate(spectra):
                if len(spectrum.intensity) == 0:
                    continue
                
                # Create offset spectrum for annotation positioning
                offset_spectrum = SpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
                offset_spectrum.normalize('max')
                # Apply the same offset used for plotting
                offset_spectrum.intensity = offset_spectrum.intensity + (i * stack_offset)
                
                # Add annotations for this spectrum with baseline at the offset position
                baseline = i * stack_offset
                handles, labels = SpectrumAnnotator.add_annotations(
                    ax, offset_spectrum, annotations, y_max, baseline, 
                    global_max_intensity=global_max_for_spacing,
                    x_min=x_min, x_max=x_max,
                    zoom=1.0,
                    baseline_offset=0.0,
                    settings=settings
                )
                all_legend_handles.extend(handles)
                all_legend_labels.extend(labels)
        
        return all_legend_handles, all_legend_labels
    
    @staticmethod
    def _plot_overlay(
        ax: plt.Axes,
        spectra: List[SpectrumData],
        settings: Dict,
        labels: List[str],
        colors: List[str]
    ):
        """Plot multiple spectra overlaid."""
        zoom = settings.get('zoom', 1.0)
        for spectrum, label, color in zip(spectra, labels, colors):
            if len(spectrum.intensity) == 0:
                continue
            
            ax.plot(
                spectrum.mz,
                spectrum.intensity * zoom,
                color=color,
                linewidth=settings.get('line_width', 1.5),
                alpha=settings.get('alpha', 0.7),
                label=label
            )
    
    @staticmethod
    def _plot_mirror(
        ax: plt.Axes,
        spectra: List[SpectrumData],
        settings: Dict,
        labels: List[str],
        colors: List[str]
    ):
        """Plot two spectra in mirror format."""
        if len(spectra) < 2:
            return
        
        zoom = settings.get('zoom', 1.0)
        
        # Top spectrum
        ax.plot(
            spectra[0].mz,
            spectra[0].intensity * zoom,
            color=colors[0],
            linewidth=settings.get('line_width', 1.5),
            label=labels[0]
        )
        
        # Bottom spectrum (mirrored)
        ax.plot(
            spectra[1].mz,
            -spectra[1].intensity * zoom,
            color=colors[1] if len(colors) > 1 else settings.get('line_color_2', 'red'),
            linewidth=settings.get('line_width', 1.5),
            label=labels[1]
        )
        
        if settings.get('fill_under', False):
            ax.fill_between(
                spectra[0].mz,
                spectra[0].intensity * zoom,
                alpha=settings.get('fill_alpha', 0.3),
                color=colors[0]
            )
            ax.fill_between(
                spectra[1].mz,
                -spectra[1].intensity * zoom,
                alpha=settings.get('fill_alpha', 0.3),
                color=colors[1] if len(colors) > 1 else settings.get('line_color_2', 'red')
            )
