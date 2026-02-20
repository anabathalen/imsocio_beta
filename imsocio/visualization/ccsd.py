"""CCSD (Collision Cross Section Distribution) plotting module.

This module provides tools for visualizing calibrated CCS data with various
display modes (stacked, summed), Gaussian fit overlays, and extensive
customization options.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from io import BytesIO
from .mass_spectrum import validate_font


@dataclass
class CCSDData:
    """Container for calibrated CCSD data.
    
    Attributes:
        df: DataFrame with columns: Charge, CCS, CCS Std.Dev., 
            Scaled_Intensity, Normalized_Intensity, m/z
        filter_threshold: Filter out points where CCS Std.Dev. > threshold * CCS
    """
    df: pd.DataFrame
    filter_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate and filter the dataframe."""
        required_cols = {
            "Charge", "CCS", "CCS Std.Dev.", 
            "Scaled_Intensity", "Normalized_Intensity", "m/z"
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter based on standard deviation threshold
        self.df = self.df[
            self.df["CCS Std.Dev."] < self.filter_threshold * self.df["CCS"]
        ].copy()
    
    def get_charge_states(self) -> list[int]:
        """Get sorted list of unique charge states."""
        return sorted(self.df["Charge"].unique())
    
    def filter_charges(self, charges: list[int]) -> pd.DataFrame:
        """Return dataframe filtered to specific charge states."""
        return self.df[self.df["Charge"].isin(charges)].copy()
    
    def get_ccs_range(self) -> tuple[float, float]:
        """Get min and max CCS values (floored and ceiled)."""
        return (
            float(np.floor(self.df["CCS"].min())),
            float(np.ceil(self.df["CCS"].max()))
        )


@dataclass
class GaussianFitData:
    """Container for Gaussian fit parameters.
    
    Attributes:
        df: DataFrame with columns: Charge, Peak_Number, Peak_Type,
            Amplitude, Center_CCS, Sigma
    """
    df: pd.DataFrame
    
    def __post_init__(self):
        """Validate and clean the dataframe."""
        required = {
            "Charge", "Peak_Number", "Peak_Type",
            "Amplitude", "Center_CCS", "Sigma"
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Keep only Gaussian peaks
        self.df = self.df[
            self.df["Peak_Type"].str.lower() == "gaussian"
        ].copy()
        
        # Coerce numeric columns
        for col in ["Charge", "Amplitude", "Center_CCS", "Sigma"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        
        # Drop rows with missing values
        self.df = self.df.dropna(
            subset=["Charge", "Amplitude", "Center_CCS", "Sigma"]
        )
        
        # Ensure integer charge
        self.df["Charge"] = self.df["Charge"].astype(int)
    
    def filter_charges(self, charges: list[int]) -> pd.DataFrame:
        """Return dataframe filtered to specific charge states."""
        return self.df[self.df["Charge"].isin(charges)].copy()
    
    def get_max_components(self) -> int:
        """Get maximum number of components across all charges."""
        try:
            return int(
                max(self.df.groupby('Charge')['Peak_Number'].nunique().max(), 1)
            )
        except Exception:
            return int(max(self.df['Peak_Number'].nunique(), 1))


@dataclass
class PlotSettings:
    """Settings for CCSD plot customization.
    
    Attributes:
        fig_width: Figure width in inches
        fig_height: Figure height in inches
        fig_dpi: Figure DPI resolution
        font_size: Font size for labels and text
        font_family: Font family name
        line_thickness: Line thickness for traces
        plot_mode: "Summed" or "Stacked" display mode
        use_scaled: If True, use Scaled_Intensity; else Normalized_Intensity
        ccs_min: Minimum CCS value for x-axis
        ccs_max: Maximum CCS value for x-axis
        ccs_label_values: List of CCS values to mark with vertical lines
        show_dashed_lines: Show dashed vertical lines at label positions
        show_ccs_labels: Show CCS value labels on plot
        label_vertical_pos: Vertical position of labels (0=bottom, 1=top)
        label_horizontal_offset: Horizontal offset for labels in CCS units
        label_orientation: "Vertical" or "Horizontal" label orientation
        shade_under: Fill area under curves
        black_lines: Use black lines for all traces
        bg_transparent: Transparent background for saved figure
        trace_colors: List of colors for charge state traces
        gaussian_colors: Optional list of colors for Gaussian components
        shade_gaussians: Fill area under Gaussian components
        title_fontsize: Font size for subplot titles
        title_fontweight: Font weight for subplot titles ("normal" or "bold")
        show_charge_labels: Show charge state labels (e.g., "15+") on plot
        charge_labels_to_show: List of specific charge states to label (None = all)
    """
    fig_width: float = 6.0
    fig_height: float = 4.0
    fig_dpi: int = 300
    font_size: int = 12
    font_family: str = "DejaVu Sans"
    line_thickness: float = 1.0
    plot_mode: Literal["Summed", "Stacked"] = "Summed"
    use_scaled: bool = True
    ccs_min: float = 0.0
    ccs_max: float = 3000.0
    ccs_label_values: list[float] = None
    show_dashed_lines: bool = True
    show_ccs_labels: bool = True
    label_vertical_pos: float = 0.95
    label_horizontal_offset: float = 0.0
    label_orientation: Literal["Vertical", "Horizontal"] = "Vertical"
    shade_under: bool = True
    black_lines: bool = False
    bg_transparent: bool = False
    trace_colors: list = None
    gaussian_colors: list = None
    shade_gaussians: bool = False
    title_fontsize: int = 14
    title_fontweight: str = "bold"
    show_charge_labels: bool = True
    charge_labels_to_show: list[int] = None
    
    def __post_init__(self):
        """Initialize default lists."""
        if self.ccs_label_values is None:
            self.ccs_label_values = []
        if self.trace_colors is None:
            self.trace_colors = ["black"]
        if self.gaussian_colors is None:
            self.gaussian_colors = []
    
    def should_show_charge_label(self, charge: int) -> bool:
        """Check if a specific charge state should be labeled.
        
        Args:
            charge: Charge state to check
            
        Returns:
            True if the charge label should be shown
        """
        if not self.show_charge_labels:
            return False
        if self.charge_labels_to_show is None:
            return True
        return charge in self.charge_labels_to_show
    
    def get_intensity_column(self) -> str:
        """Get the appropriate intensity column name based on settings.
        
        Returns:
            "Scaled_Intensity" or "Normalized_Intensity"
        """
        return "Scaled_Intensity" if self.use_scaled else "Normalized_Intensity"


class CCSDPlotter:
    """Plotter for collision cross section distributions."""
    
    @staticmethod
    def plot_multiple_ccs_traces(
        datasets: list[tuple[str, CCSDData, list[int]]],
        settings: PlotSettings,
        gaussian_datasets: Optional[list[tuple[str, GaussianFitData]]] = None,
        show_gaussian_fits: bool = False,
        layout: Literal["horizontal", "vertical", "grid"] = "vertical"
    ) -> tuple[plt.Figure, dict, str]:
        """Plot multiple CCSD datasets as subplots.
        
        Args:
            datasets: List of (name, CCSDData, selected_charges) tuples
            settings: Plot customization settings
            gaussian_datasets: Optional list of (name, GaussianFitData) tuples
            show_gaussian_fits: Whether to overlay Gaussian fits
            layout: Subplot layout - "horizontal", "vertical", or "grid"
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
            maxima_info maps dataset names to trace-maxima mappings
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        n_datasets = len(datasets)
        
        # Determine subplot layout
        if layout == "horizontal":
            nrows, ncols = 1, n_datasets
        elif layout == "vertical":
            nrows, ncols = n_datasets, 1
        else:  # grid
            ncols = int(np.ceil(np.sqrt(n_datasets)))
            nrows = int(np.ceil(n_datasets / ncols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(settings.fig_width * ncols, settings.fig_height * nrows),
            dpi=settings.fig_dpi,
            squeeze=False
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Storage for all maxima info
        all_maxima_info = {}
        
        # Match Gaussian data by name
        gaussian_dict = {}
        if gaussian_datasets:
            gaussian_dict = {name: gdata for name, gdata in gaussian_datasets}
        
        # Plot each dataset
        for idx, (name, ccsd_data, selected_charges) in enumerate(datasets):
            ax = axes_flat[idx]
            
            # Get corresponding Gaussian data if available
            gaussian_data = gaussian_dict.get(name, None)
            
            # Create temporary figure for single plot, then transfer to subplot
            maxima_info = CCSDPlotter._plot_single_ccs_trace(
                ax=ax,
                ccsd_data=ccsd_data,
                selected_charges=selected_charges,
                settings=settings,
                gaussian_data=gaussian_data,
                show_gaussian_fits=show_gaussian_fits,
                subplot_title=name
            )
            
            all_maxima_info[name] = maxima_info
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig, all_maxima_info, font_warning
    
    @staticmethod
    def _plot_single_ccs_trace(
        ax: plt.Axes,
        ccsd_data: CCSDData,
        selected_charges: list[int],
        settings: PlotSettings,
        gaussian_data: Optional[GaussianFitData] = None,
        show_gaussian_fits: bool = False,
        subplot_title: Optional[str] = None
    ) -> dict:
        """Plot CCS traces on a single axes object.
        
        Args:
            ax: Matplotlib axes to plot on
            ccsd_data: Calibrated CCSD data
            selected_charges: List of charge states to plot
            settings: Plot customization settings
            gaussian_data: Optional Gaussian fit data for overlay
            show_gaussian_fits: Whether to overlay Gaussian fits
            subplot_title: Optional title for subplot
            
        Returns:
            Dictionary mapping trace labels to list of (CCS, intensity) tuples
        """
        # Filter data to selected charges
        cal_df = ccsd_data.filter_charges(selected_charges)
        
        # Create CCS grid for interpolation
        ccs_grid = np.arange(settings.ccs_min, settings.ccs_max + 1, 1.0)
        
        # Storage for traces and maxima
        interpolated_traces = []
        maxima_info = {}
        
        if settings.plot_mode == "Summed":
            # Plot each charge state
            for i, (charge, group) in enumerate(cal_df.groupby("Charge")):
                group_sorted = group.sort_values("CCS")
                
                # Select intensity column
                y_col = settings.get_intensity_column()
                y_values = group_sorted[y_col]
                
                # Interpolate onto grid
                interp = np.interp(
                    ccs_grid, 
                    group_sorted["CCS"], 
                    y_values, 
                    left=0, 
                    right=0
                )
                interpolated_traces.append(interp)
                
                # Determine line color
                line_color = "black" if settings.black_lines else settings.trace_colors[i]
                
                # Plot trace
                ax.plot(
                    ccs_grid, 
                    interp, 
                    color=line_color,
                    label=f"{int(charge)}+",
                    linewidth=settings.line_thickness
                )
                
                # Optional shading
                if settings.shade_under:
                    ax.fill_between(
                        ccs_grid, 0, interp,
                        color=settings.trace_colors[i],
                        alpha=0.3
                    )
                
                # Overlay Gaussian components
                if show_gaussian_fits and gaussian_data is not None:
                    components = CCSDPlotter._compute_gaussian_components(
                        charge, ccs_grid, gaussian_data, settings.use_scaled
                    )
                    
                    for j, comp in enumerate(components):
                        # Use same color as the trace for this charge state
                        comp_color = line_color
                        
                        ax.plot(
                            ccs_grid, comp,
                            color=comp_color,
                            linestyle="--",
                            linewidth=max(settings.line_thickness, 1.0),
                            alpha=0.9
                        )
                        
                        if settings.shade_gaussians:
                            ax.fill_between(
                                ccs_grid, 0, comp,
                                color=comp_color,
                                alpha=0.15
                            )
            
            # Compute and plot summed trace
            if interpolated_traces:
                total_trace = np.sum(interpolated_traces, axis=0)
                
                ax.plot(
                    ccs_grid, total_trace,
                    color="black",
                    linewidth=settings.line_thickness,
                    label="Summed"
                )
                
                # Find local maxima
                maxima_idx = argrelextrema(total_trace, np.greater)[0]
                maxima_ccs = ccs_grid[maxima_idx]
                maxima_vals = total_trace[maxima_idx]
                maxima_info["Summed"] = list(zip(maxima_ccs, maxima_vals))
            
            ax.legend(fontsize=settings.font_size, frameon=False)
        
        elif settings.plot_mode == "Stacked":
            # Interpolate all traces first
            interpolated = {}
            base_max = 0.0
            
            for charge, group in cal_df.groupby("Charge"):
                group_sorted = group.sort_values("CCS")
                
                y_col = "Scaled_Intensity" if settings.use_scaled else "Normalized_Intensity"
                y_values = group_sorted[y_col]
                
                interp = np.interp(
                    ccs_grid,
                    group_sorted["CCS"],
                    y_values,
                    left=0,
                    right=0
                )
                
                interpolated[int(charge)] = interp
                if interp.size:
                    base_max = max(base_max, float(interp.max()))
            
            # Calculate offsets
            if not settings.use_scaled:
                gap = 0.10 * base_max if base_max > 0 else 0.10
                offset_step = (base_max if base_max > 0 else 1.0) + gap
                def offset_for(i):
                    return i * offset_step
            else:
                offset_unit = 1.0 / max(len(interpolated), 1)
                def offset_for(i):
                    return i * offset_unit * base_max
            
            # Plot each charge with offset
            for i, charge in enumerate(sorted(interpolated.keys())):
                interp = interpolated[charge]
                offset = offset_for(i)
                offset_interp = interp + offset
                
                # Determine color
                line_color = "black" if settings.black_lines else settings.trace_colors[i]
                
                # Plot trace
                ax.plot(
                    ccs_grid, offset_interp,
                    color=line_color,
                    linewidth=settings.line_thickness
                )
                
                # Optional shading
                if settings.shade_under:
                    ax.fill_between(
                        ccs_grid, offset, offset_interp,
                        color=settings.trace_colors[i],
                        alpha=0.3
                    )
                
                # Overlay Gaussian components at same offset
                if show_gaussian_fits and gaussian_data is not None:
                    components = CCSDPlotter._compute_gaussian_components(
                        charge, ccs_grid, gaussian_data, settings.use_scaled
                    )
                    
                    for j, comp in enumerate(components):
                        # Use same color as the trace for this charge state
                        comp_color = line_color
                        
                        ax.plot(
                            ccs_grid, comp + offset,
                            color=comp_color,
                            linestyle="--",
                            linewidth=max(settings.line_thickness, 1.0),
                            alpha=0.9
                        )
                        
                        if settings.shade_gaussians:
                            ax.fill_between(
                                ccs_grid, offset, comp + offset,
                                color=comp_color,
                                alpha=0.15
                            )
                
                # Add charge label if enabled for this charge state
                if settings.should_show_charge_label(charge):
                    label_x = settings.ccs_min + (settings.ccs_max - settings.ccs_min) * 0.05
                    label_y = offset + (base_max * 0.05 if base_max > 0 else 0.05)
                    ax.text(
                        label_x, label_y,
                        f"{int(charge)}+",
                        fontsize=settings.font_size,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        color=settings.trace_colors[i]
                    )
                
                # Find local maxima for this trace
                maxima_idx = argrelextrema(interp, np.greater)[0]
                maxima_ccs = ccs_grid[maxima_idx]
                maxima_vals = interp[maxima_idx]
                maxima_info[f"{int(charge)}+"] = list(zip(maxima_ccs, maxima_vals))
        
        # Add CCS label lines and annotations
        ylim = ax.get_ylim()
        y_label = ylim[0] + settings.label_vertical_pos * (ylim[1] - ylim[0])
        
        for ccs_value in settings.ccs_label_values:
            if ccs_value > 0:
                # Draw dashed line
                if settings.show_dashed_lines:
                    ax.axvline(
                        ccs_value,
                        color="black",
                        linewidth=1.0,
                        linestyle="--"
                    )
                
                # Add label
                if settings.show_ccs_labels:
                    ax.text(
                        ccs_value + settings.label_horizontal_offset,
                        y_label,
                        f"{ccs_value:.1f}",
                        rotation=90 if settings.label_orientation == "Vertical" else 0,
                        verticalalignment='top' if settings.label_orientation == "Vertical" else 'center',
                        horizontalalignment='right' if settings.label_orientation == "Vertical" else 'center',
                        fontsize=settings.font_size,
                        color="black",
                        backgroundcolor="white" if not settings.bg_transparent else "none"
                    )
        
        # Set axes properties
        ax.set_xlim([settings.ccs_min, settings.ccs_max])
        ax.set_xlabel("CCS (Å²)", fontsize=settings.font_size)
        ax.set_yticks([])
        ax.grid(False)
        
        for label in ax.get_xticklabels():
            label.set_fontsize(settings.font_size)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        
        # Add subplot title if provided
        if subplot_title:
            ax.set_title(
                subplot_title,
                fontsize=settings.title_fontsize,
                fontweight=settings.title_fontweight
            )
        
        return maxima_info
    
    @staticmethod
    def _compute_gaussian_components(
        charge: int,
        ccs_grid: np.ndarray,
        gaussian_data: Optional[GaussianFitData],
        use_scaled: bool
    ) -> list[np.ndarray]:
        """Compute individual Gaussian component curves for a charge state.
        
        Args:
            charge: Charge state to compute components for
            ccs_grid: CCS values to evaluate Gaussians at
            gaussian_data: Gaussian fit parameters
            use_scaled: If True, don't normalize components
            
        Returns:
            List of arrays, each representing one Gaussian component
        """
        if gaussian_data is None or gaussian_data.df.empty:
            return []
        
        # Get components for this charge
        sub = gaussian_data.df[gaussian_data.df["Charge"] == int(charge)]
        if sub.empty:
            return []
        
        components = []
        
        # Sort by Peak_Number for consistent ordering
        for _, row in sub.sort_values(["Peak_Number"]).iterrows():
            amp = float(row["Amplitude"])
            center = float(row["Center_CCS"])
            sigma = float(row["Sigma"])
            
            # Validate parameters
            if sigma <= 0 or not np.isfinite([amp, center, sigma]).all():
                continue
            
            # Compute Gaussian
            gaussian = amp * np.exp(-0.5 * ((ccs_grid - center) / sigma) ** 2)
            components.append(gaussian)
        
        if not components:
            return []
        
        # Normalize if using normalized intensities
        if not use_scaled:
            summed = np.sum(components, axis=0)
            max_val = float(np.max(summed)) if np.isfinite(summed).any() and np.max(summed) > 0 else 0.0
            if max_val > 0:
                components = [c / max_val for c in components]
        
        return components
    
    @staticmethod
    def plot_ccs_traces(
        ccsd_data: CCSDData,
        selected_charges: list[int],
        settings: PlotSettings,
        gaussian_data: Optional[GaussianFitData] = None,
        show_gaussian_fits: bool = False
    ) -> tuple[plt.Figure, dict, str]:
        """Plot CCS traces for selected charge states.
        
        Args:
            ccsd_data: Calibrated CCSD data
            selected_charges: List of charge states to plot
            settings: Plot customization settings
            gaussian_data: Optional Gaussian fit data for overlay
            show_gaussian_fits: Whether to overlay Gaussian fits
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
            maxima_info maps trace labels to list of (CCS, intensity) tuples
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(settings.fig_width, settings.fig_height),
            dpi=settings.fig_dpi
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Use the internal plotting method
        maxima_info = CCSDPlotter._plot_single_ccs_trace(
            ax=ax,
            ccsd_data=ccsd_data,
            selected_charges=selected_charges,
            settings=settings,
            gaussian_data=gaussian_data,
            show_gaussian_fits=show_gaussian_fits,
            subplot_title=None
        )
        
        return fig, maxima_info, font_warning
    
    @staticmethod
    def plot_total_ccsd(
        ccsd_data: CCSDData,
        selected_charges: list[int],
        settings: PlotSettings,
        dataset_name: str = "Total CCSD"
    ) -> tuple[plt.Figure, dict, str]:
        """Plot total CCSD (sum of all charge states).
        
        Args:
            ccsd_data: Calibrated CCSD data
            selected_charges: List of charge states to include in sum
            settings: Plot customization settings
            dataset_name: Name for the dataset (for labels)
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(settings.fig_width, settings.fig_height),
            dpi=settings.fig_dpi
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Filter data to selected charges
        cal_df = ccsd_data.filter_charges(selected_charges)
        
        # Create CCS grid for interpolation (match original behavior)
        ccs_grid = np.arange(settings.ccs_min, settings.ccs_max + 1, 1.0)
        
        # Interpolate and sum all charge states
        interpolated_traces = []
        y_col = settings.get_intensity_column()
        
        for charge, group in cal_df.groupby("Charge"):
            group_sorted = group.sort_values("CCS")
            y_values = group_sorted[y_col]
            
            # Interpolate onto grid
            interp = np.interp(
                ccs_grid, 
                group_sorted["CCS"], 
                y_values, 
                left=0, 
                right=0
            )
            interpolated_traces.append(interp)
        
        # Compute total trace
        if interpolated_traces:
            total_trace = np.sum(interpolated_traces, axis=0)
            
            # Plot total trace
            ax.plot(
                ccs_grid, 
                total_trace,
                color="black",
                linewidth=settings.line_thickness,
                label="Summed"
            )
            
            # Find local maxima
            maxima_idx = argrelextrema(total_trace, np.greater)[0]
            maxima_ccs = ccs_grid[maxima_idx]
            maxima_vals = total_trace[maxima_idx]
            maxima_info = {"Summed": list(zip(maxima_ccs, maxima_vals))}
        else:
            maxima_info = {}
        
        # Add CCS label lines and annotations
        ylim = ax.get_ylim()
        y_label = ylim[0] + settings.label_vertical_pos * (ylim[1] - ylim[0])
        
        for ccs_value in settings.ccs_label_values:
            if ccs_value > 0:
                # Draw dashed line
                if settings.show_dashed_lines:
                    ax.axvline(
                        ccs_value,
                        color="black",
                        linewidth=1.0,
                        linestyle="--"
                    )
                
                # Add label
                if settings.show_ccs_labels:
                    ax.text(
                        ccs_value + settings.label_horizontal_offset,
                        y_label,
                        f"{ccs_value:.1f}",
                        rotation=90 if settings.label_orientation == "Vertical" else 0,
                        verticalalignment='top' if settings.label_orientation == "Vertical" else 'center',
                        horizontalalignment='right' if settings.label_orientation == "Vertical" else 'center',
                        fontsize=settings.font_size,
                        color="black",
                        backgroundcolor="white" if not settings.bg_transparent else "none"
                    )
        
        # Set axes properties (match original styling)
        ax.set_xlim([settings.ccs_min, settings.ccs_max])
        ax.set_xlabel("CCS (Å²)", fontsize=settings.font_size)
        ax.set_yticks([])
        ax.grid(False)
        
        for label in ax.get_xticklabels():
            label.set_fontsize(settings.font_size)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        
        ax.legend(fontsize=settings.font_size, frameon=False)
        
        # Apply background transparency
        if settings.bg_transparent:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
        
        plt.tight_layout()
        
        return fig, maxima_info, font_warning
    
    @staticmethod
    def plot_combined_datasets(
        datasets: list[tuple[str, CCSDData, list[int]]],
        settings: PlotSettings,
        show_total_only: bool = False
    ) -> tuple[plt.Figure, dict, str]:
        """Plot multiple datasets on same axes.
        
        Args:
            datasets: List of (name, CCSDData, selected_charges) tuples
            settings: Plot customization settings
            show_total_only: If True, show only total for each dataset
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(settings.fig_width, settings.fig_height),
            dpi=settings.fig_dpi
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Create CCS grid for interpolation
        ccs_grid = np.arange(settings.ccs_min, settings.ccs_max + 1, 1.0)
        
        # Storage for maxima
        all_maxima_info = {}
        
        if show_total_only:
            # Plot total CCSD for each dataset
            for i, (name, ccsd_data, selected_charges) in enumerate(datasets):
                # Filter data to selected charges
                cal_df = ccsd_data.filter_charges(selected_charges)
                
                # Select intensity column
                y_col = settings.get_intensity_column()
                
                # Interpolate and sum all charge states for this dataset
                interpolated_traces = []
                for charge, group in cal_df.groupby("Charge"):
                    group_sorted = group.sort_values("CCS")
                    y_values = group_sorted[y_col]
                    
                    # Interpolate onto grid
                    interp = np.interp(
                        ccs_grid, 
                        group_sorted["CCS"], 
                        y_values, 
                        left=0, 
                        right=0
                    )
                    interpolated_traces.append(interp)
                
                if interpolated_traces:
                    total_trace = np.sum(interpolated_traces, axis=0)
                    
                    # Determine color
                    color = settings.trace_colors[i % len(settings.trace_colors)] if settings.trace_colors else "black"
                    
                    # Plot
                    ax.plot(
                        ccs_grid,
                        total_trace,
                        color=color,
                        linewidth=settings.line_thickness,
                        label=name
                    )
                    
                    # Find maxima
                    maxima_idx = argrelextrema(total_trace, np.greater)[0]
                    maxima_ccs = ccs_grid[maxima_idx]
                    maxima_vals = total_trace[maxima_idx]
                    all_maxima_info[name] = {"Summed": list(zip(maxima_ccs, maxima_vals))}
        else:
            # Plot individual charge states for all datasets
            color_idx = 0
            for name, ccsd_data, selected_charges in datasets:
                cal_df = ccsd_data.filter_charges(selected_charges)
                y_col = settings.get_intensity_column()
                
                for charge in sorted(selected_charges):
                    charge_data = cal_df[cal_df["Charge"] == charge]
                    if charge_data.empty:
                        continue
                    
                    charge_data = charge_data.sort_values("CCS")
                    
                    # Interpolate onto grid
                    interp = np.interp(
                        ccs_grid,
                        charge_data["CCS"],
                        charge_data[y_col],
                        left=0,
                        right=0
                    )
                    
                    # Determine color
                    line_color = "black" if settings.black_lines else settings.trace_colors[color_idx % len(settings.trace_colors)]
                    color_idx += 1
                    
                    # Plot
                    label = f"{name} - {charge}+"
                    ax.plot(
                        ccs_grid,
                        interp,
                        color=line_color,
                        linewidth=settings.line_thickness,
                        label=label
                    )
                    
                    # Optional shading
                    if settings.shade_under:
                        ax.fill_between(
                            ccs_grid,
                            0,
                            interp,
                            color=settings.trace_colors[color_idx % len(settings.trace_colors)] if settings.trace_colors else line_color,
                            alpha=0.3
                        )
                
                # Store empty maxima info for now
                all_maxima_info[name] = {}
        
        # Add CCS label lines and annotations
        ylim = ax.get_ylim()
        y_label = ylim[0] + settings.label_vertical_pos * (ylim[1] - ylim[0])
        
        for ccs_value in settings.ccs_label_values:
            if ccs_value > 0:
                # Draw dashed line
                if settings.show_dashed_lines:
                    ax.axvline(
                        ccs_value,
                        color="black",
                        linewidth=1.0,
                        linestyle="--"
                    )
                
                # Add label
                if settings.show_ccs_labels:
                    ax.text(
                        ccs_value + settings.label_horizontal_offset,
                        y_label,
                        f"{ccs_value:.1f}",
                        rotation=90 if settings.label_orientation == "Vertical" else 0,
                        verticalalignment='top' if settings.label_orientation == "Vertical" else 'center',
                        horizontalalignment='right' if settings.label_orientation == "Vertical" else 'center',
                        fontsize=settings.font_size,
                        color="black",
                        backgroundcolor="white" if not settings.bg_transparent else "none"
                    )
        
        # Set axes properties (match original styling)
        ax.set_xlim([settings.ccs_min, settings.ccs_max])
        ax.set_xlabel("CCS (Å²)", fontsize=settings.font_size)
        ax.set_yticks([])
        ax.grid(False)
        
        for label in ax.get_xticklabels():
            label.set_fontsize(settings.font_size)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        
        ax.legend(fontsize=settings.font_size, frameon=False)
        
        # Apply background transparency
        if settings.bg_transparent:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
        
        plt.tight_layout()
        
        return fig, all_maxima_info, font_warning
    
    @staticmethod
    @staticmethod
    def plot_multiple_total_ccsds(
        datasets: list[tuple[str, CCSDData, list[int]]],
        settings: PlotSettings,
        layout: Literal["horizontal", "vertical", "grid"] = "vertical"
    ) -> tuple[plt.Figure, dict, str]:
        """Plot multiple total CCSD datasets as subplots.
        
        Args:
            datasets: List of (name, CCSDData, selected_charges) tuples
            settings: Plot customization settings
            layout: Subplot layout - "horizontal", "vertical", or "grid"
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        n_datasets = len(datasets)
        
        # Determine subplot layout
        if layout == "horizontal":
            nrows, ncols = 1, n_datasets
        elif layout == "vertical":
            nrows, ncols = n_datasets, 1
        else:  # grid
            ncols = int(np.ceil(np.sqrt(n_datasets)))
            nrows = int(np.ceil(n_datasets / ncols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(settings.fig_width * ncols, settings.fig_height * nrows),
            dpi=settings.fig_dpi,
            squeeze=False
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        all_maxima_info = {}
        
        # Create CCS grid for interpolation
        ccs_grid = np.arange(settings.ccs_min, settings.ccs_max + 1, 1.0)
        y_col = settings.get_intensity_column()
        
        # Plot each dataset
        for idx, (name, ccsd_data, selected_charges) in enumerate(datasets):
            ax = axes_flat[idx]
            
            # Filter data to selected charges
            cal_df = ccsd_data.filter_charges(selected_charges)
            
            # Interpolate and sum all charge states
            interpolated_traces = []
            for charge, group in cal_df.groupby("Charge"):
                group_sorted = group.sort_values("CCS")
                y_values = group_sorted[y_col]
                
                # Interpolate onto grid
                interp = np.interp(
                    ccs_grid, 
                    group_sorted["CCS"], 
                    y_values, 
                    left=0, 
                    right=0
                )
                interpolated_traces.append(interp)
            
            # Compute and plot total trace
            if interpolated_traces:
                total_trace = np.sum(interpolated_traces, axis=0)
                
                # Determine color
                color = settings.trace_colors[0] if settings.trace_colors else "black"
                
                # Plot
                ax.plot(
                    ccs_grid,
                    total_trace,
                    color=color,
                    linewidth=settings.line_thickness,
                    label='Summed'
                )
                
                if settings.shade_under:
                    ax.fill_between(
                        ccs_grid,
                        0,
                        total_trace,
                        color=color,
                        alpha=0.3
                    )
                
                # Find local maxima
                maxima_idx = argrelextrema(total_trace, np.greater)[0]
                maxima_ccs = ccs_grid[maxima_idx]
                maxima_vals = total_trace[maxima_idx]
                maxima_info = {"Summed": list(zip(maxima_ccs, maxima_vals))}
                
                # Add CCS labels if requested
                if settings.show_ccs_labels and len(settings.ccs_label_values) > 0:
                    ylim = ax.get_ylim()
                    y_label = ylim[0] + settings.label_vertical_pos * (ylim[1] - ylim[0])
                    
                    for ccs_value in settings.ccs_label_values:
                        if ccs_value > 0:
                            # Draw dashed line
                            if settings.show_dashed_lines:
                                ax.axvline(
                                    ccs_value,
                                    color="gray",
                                    linewidth=0.8,
                                    linestyle="--",
                                    alpha=0.6
                                )
                            
                            # Add label
                            rotation = 90 if settings.label_orientation == "Vertical" else 0
                            ax.text(
                                ccs_value + settings.label_horizontal_offset,
                                y_label,
                                f'{ccs_value:.0f}',
                                rotation=rotation,
                                fontsize=settings.font_size * 0.8,
                                color='black',
                                ha='center',
                                va='bottom' if settings.label_orientation == "Vertical" else 'center'
                            )
            else:
                maxima_info = {}
            
            # Set title
            ax.set_title(name, fontsize=settings.title_fontsize, fontweight=settings.title_fontweight)
            
            # Set axis properties to match original plots
            ax.set_xlabel('CCS (Å²)', fontsize=settings.font_size)
            ax.set_ylabel('', fontsize=settings.font_size)
            ax.set_xlim(settings.ccs_min, settings.ccs_max)
            ax.set_yticks([])
            ax.legend(fontsize=settings.font_size, frameon=False)
            ax.grid(False)
            
            for label in ax.get_xticklabels():
                label.set_fontsize(settings.font_size)
            
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)
            
            all_maxima_info[name] = maxima_info
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Apply background transparency
        if settings.bg_transparent:
            fig.patch.set_alpha(0)
            for ax in axes_flat:
                ax.patch.set_alpha(0)
        
        plt.tight_layout()
        
        return fig, all_maxima_info, font_warning
    
    @staticmethod
    def plot_replicate_ccsds(
        datasets: list[tuple[str, CCSDData, list[int]]],
        settings: PlotSettings
    ) -> tuple[plt.Figure, dict, str]:
        """Plot mean ± standard deviation of replicate CCSD measurements.
        
        Normalizes each replicate so that the maximum intensity across all charge
        states equals 1, then computes mean and standard deviation at each CCS point.
        Plots mean as a line with ±1 std deviation as a shaded region.
        
        Args:
            datasets: List of tuples (name, ccsd_data, selected_charges)
            settings: Plot customization settings
            
        Returns:
            Tuple of (matplotlib figure, maxima_info dictionary, font_warning)
        """
        # Validate font
        font_to_use, font_warning = validate_font(settings.font_family, fallback='DejaVu Sans')
        
        if len(datasets) < 2:
            raise ValueError("Replicate analysis requires at least 2 datasets")
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(settings.fig_width, settings.fig_height),
            dpi=settings.fig_dpi
        )
        plt.rcParams.update({'font.family': font_to_use})
        
        # Create CCS grid for interpolation
        ccs_grid = np.arange(settings.ccs_min, settings.ccs_max + 1, 1.0)
        y_col = settings.get_intensity_column()
        
        # Get all unique charge states from first dataset
        # (assuming all replicates have the same charge states)
        _, first_ccsd_data, selected_charges = datasets[0]
        
        # Store normalized traces for each replicate and charge state
        # Structure: {charge: [replicate1_trace, replicate2_trace, ...]}
        charge_replicate_traces = {charge: [] for charge in selected_charges}
        
        # Process each replicate
        for name, ccsd_data, charges in datasets:
            cal_df = ccsd_data.filter_charges(charges)
            
            # Interpolate all charge states first, then normalize by total intensity
            replicate_traces = {}
            
            for charge in charges:
                charge_df = cal_df[cal_df["Charge"] == charge].sort_values("CCS")
                
                if len(charge_df) == 0:
                    # No data for this charge state, use zeros
                    replicate_traces[charge] = np.zeros_like(ccs_grid)
                    continue
                
                # Interpolate onto grid (not normalized yet)
                y_values = charge_df[y_col]
                interp = np.interp(
                    ccs_grid,
                    charge_df["CCS"],
                    y_values,
                    left=0,
                    right=0
                )
                replicate_traces[charge] = interp
            
            # Calculate total scaled intensity (sum across all charge states and CCS points)
            # This is equivalent to the integral under the summed CCSD curve
            total_intensity = sum(replicate_traces.values()).sum()
            
            if total_intensity == 0:
                raise ValueError(f"Dataset '{name}' has zero total intensity")
            
            # Normalize all traces by the total intensity
            for charge in charges:
                normalized_trace = replicate_traces[charge] / total_intensity
                charge_replicate_traces[charge].append(normalized_trace)
        
        # Calculate mean and std for each charge state and plot
        maxima_info = {}
        
        for i, charge in enumerate(selected_charges):
            # Stack all replicate traces for this charge
            traces = np.array(charge_replicate_traces[charge])
            
            if len(traces) == 0:
                continue
            
            # Calculate mean and std across replicates (axis 0)
            mean_trace = np.mean(traces, axis=0)
            std_trace = np.std(traces, axis=0)
            
            # Get color for this charge state
            color_idx = i % len(settings.trace_colors)
            color = settings.trace_colors[color_idx]
            if settings.black_lines:
                color = 'black'
            
            # Plot mean line
            label = f"Charge {charge}+ (n={len(datasets)})"
            ax.plot(
                ccs_grid,
                mean_trace,
                color=color,
                linewidth=settings.line_thickness,
                label=label
            )
            
            # Shade ±1 std deviation (clipped at 0 to prevent negative values)
            ax.fill_between(
                ccs_grid,
                np.maximum(0, mean_trace - std_trace),
                mean_trace + std_trace,
                color=color,
                alpha=0.2,
                linewidth=0
            )
            
            # Find local maxima in mean trace
            # Use window of 5 points for maxima detection
            maxima_indices = argrelextrema(mean_trace, np.greater, order=5)[0]
            maxima = [(ccs_grid[idx], mean_trace[idx]) for idx in maxima_indices]
            maxima_info[label] = maxima
        
        # Add vertical dashed lines and labels for CCS values
        if settings.ccs_label_values:
            y_max = ax.get_ylim()[1]
            y_label = settings.label_vertical_pos * y_max
            
            for ccs_value in settings.ccs_label_values:
                # Draw dashed line
                if settings.show_dashed_lines:
                    ax.axvline(
                        ccs_value,
                        color="black",
                        linewidth=1.0,
                        linestyle="--"
                    )
                
                # Add label
                if settings.show_ccs_labels:
                    ax.text(
                        ccs_value + settings.label_horizontal_offset,
                        y_label,
                        f"{ccs_value:.1f}",
                        rotation=90 if settings.label_orientation == "Vertical" else 0,
                        verticalalignment='top' if settings.label_orientation == "Vertical" else 'center',
                        horizontalalignment='right' if settings.label_orientation == "Vertical" else 'center',
                        fontsize=settings.font_size,
                        color="black",
                        backgroundcolor="white" if not settings.bg_transparent else "none"
                    )
        
        # Set axes properties to match other plots
        ax.set_xlim([settings.ccs_min, settings.ccs_max])
        ax.set_xlabel("CCS (Å²)", fontsize=settings.font_size)
        ax.set_yticks([])
        ax.grid(False)
        
        for label in ax.get_xticklabels():
            label.set_fontsize(settings.font_size)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        
        # Add legend
        ax.legend(fontsize=settings.font_size, frameon=False)
        
        # Apply background transparency
        if settings.bg_transparent:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
        
        plt.tight_layout()
        
        return fig, maxima_info, font_warning
    
    @staticmethod
    def save_figure_to_buffer(
        fig: plt.Figure,
        dpi: int = 300,
        transparent: bool = False
    ) -> BytesIO:
        """Save matplotlib figure to PNG buffer.
        
        Args:
            fig: Matplotlib figure to save
            dpi: Resolution in dots per inch
            transparent: Whether to use transparent background
            
        Returns:
            BytesIO buffer containing PNG data
        """
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format='png',
            dpi=dpi,
            bbox_inches='tight',
            transparent=transparent
        )
        buffer.seek(0)
        return buffer
