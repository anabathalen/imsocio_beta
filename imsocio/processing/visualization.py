"""
Visualization functions for mass spectra and integration.

This module provides plotting utilities for:
- Mass spectrum integration with baseline fitting
- Full mass spectrum overview with charge states
"""

from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import fit_baseline_and_integrate, calculate_theoretical_mz, PROTON_MASS


def plot_spectrum_with_integration(
    ms_df: pd.DataFrame,
    mz_theoretical: float,
    integration_range: Tuple[float, float],
    smoothing_window: int = 51,
    show_zoomed: bool = True
) -> Tuple[Optional[float], bool, plt.Figure]:
    """
    Plot mass spectrum with baseline fitting and integration.
    
    Args:
        ms_df: DataFrame with 'm/z' and 'Intensity' columns
        mz_theoretical: Theoretical m/z for vertical line
        integration_range: (min_mz, max_mz) for integration
        smoothing_window: Window size for smoothing
        show_zoomed: If True, zoom to ±10% of theoretical m/z
        
    Returns:
        Tuple of (area, range_outside_view, figure)
    """
    # Determine plot window
    if show_zoomed:
        mz_window_min = mz_theoretical * 0.90
        mz_window_max = mz_theoretical * 1.10
        ms_df_window = ms_df[
            (ms_df["m/z"] >= mz_window_min) & 
            (ms_df["m/z"] <= mz_window_max)
        ].copy()
        title_suffix = " (Zoomed)"
    else:
        ms_df_window = ms_df.copy()
        title_suffix = " (Full Spectrum)"
    
    if len(ms_df_window) == 0:
        return None, False, None
    
    # Apply smoothing
    ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
        window=max(1, smoothing_window), 
        center=True, 
        min_periods=1
    ).mean()
    
    # Get data for integration
    integration_mask = (
        (ms_df_window["m/z"] >= integration_range[0]) & 
        (ms_df_window["m/z"] <= integration_range[1])
    )
    integration_df = ms_df_window[integration_mask]
    
    # Check if range extends beyond current view
    range_outside_view = (
        integration_range[0] < ms_df_window["m/z"].min() or 
        integration_range[1] > ms_df_window["m/z"].max()
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot spectrum
    ax.plot(
        ms_df_window["m/z"], 
        ms_df_window["Smoothed"], 
        color="blue", 
        linewidth=1.5, 
        label="Smoothed spectrum"
    )
    ax.axvline(
        mz_theoretical, 
        color="red", 
        linestyle="--", 
        alpha=0.7, 
        label=f"Theoretical m/z: {mz_theoretical:.3f}"
    )
    
    area = None
    if len(integration_df) >= 3:
        # Fit baseline and integrate
        area, baseline = fit_baseline_and_integrate(
            ms_df_window["m/z"].values, 
            ms_df_window["Smoothed"].values, 
            integration_range
        )
        
        # Plot baseline in integration region
        baseline_mask = (
            (ms_df_window["m/z"] >= integration_range[0]) & 
            (ms_df_window["m/z"] <= integration_range[1])
        )
        if np.any(baseline_mask):
            baseline_x = ms_df_window["m/z"].values[baseline_mask]
            baseline_y = baseline[baseline_mask]
            ax.plot(
                baseline_x, 
                baseline_y, 
                'g--', 
                linewidth=2, 
                label="Fitted baseline"
            )
            
            # Fill area above baseline
            spectrum_y = ms_df_window["Smoothed"].values[baseline_mask]
            ax.fill_between(
                baseline_x, 
                baseline_y, 
                spectrum_y, 
                where=(spectrum_y >= baseline_y), 
                color="orange", 
                alpha=0.4, 
                label="Integrated area"
            )
        
        # Add integration bounds
        ax.axvline(
            integration_range[0], 
            color="green", 
            linestyle="-", 
            alpha=0.8, 
            linewidth=2
        )
        ax.axvline(
            integration_range[1], 
            color="green", 
            linestyle="-", 
            alpha=0.8, 
            linewidth=2
        )
    
    # Labels and formatting
    ax.set_xlabel("m/z")
    ax.set_ylabel("Smoothed Intensity")
    ax.set_title(
        f"Integration region: {integration_range[0]:.3f} - "
        f"{integration_range[1]:.3f} m/z{title_suffix}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add area annotation if calculated
    if area is not None:
        ax.text(
            0.02, 0.98, 
            f"Area: {area:.2e}", 
            transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    return area, range_outside_view, fig


def plot_full_spectrum_with_charge_states(
    ms_df: pd.DataFrame,
    protein_name: str,
    protein_mass: float,
    charge_range: Tuple[int, int],
    selected_charge: int,
    scale_ranges: Optional[Dict[Tuple[str, int], Tuple[float, float]]] = None
) -> plt.Figure:
    """
    Plot full mass spectrum with vertical lines for all charge states.
    
    Args:
        ms_df: DataFrame with 'm/z' and 'Intensity' columns
        protein_name: Name of the protein
        protein_mass: Protein mass in Da
        charge_range: (min_charge, max_charge)
        selected_charge: Currently selected charge state
        scale_ranges: Optional dict of integration ranges
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        ms_df["m/z"], 
        ms_df["Intensity"], 
        color="gray", 
        linewidth=1, 
        alpha=0.7, 
        label="Mass spectrum"
    )
    
    # Add vertical lines for each charge state
    min_charge, max_charge = charge_range
    for charge in range(min_charge, max_charge + 1):
        mz = calculate_theoretical_mz(protein_mass, charge)
        
        color = "red" if charge == selected_charge else "blue"
        alpha = 0.9 if charge == selected_charge else 0.5
        linestyle = "-" if charge == selected_charge else "--"
        linewidth = 2 if charge == selected_charge else 1
        
        ax.axvline(
            mz, 
            color=color, 
            linestyle=linestyle, 
            alpha=alpha, 
            linewidth=linewidth
        )
        
        # Add charge state label
        label_height = ax.get_ylim()[1] * 0.9
        ax.text(
            mz, 
            label_height, 
            f"{charge}+", 
            color=color, 
            ha="center", 
            va="top", 
            fontsize=10, 
            fontweight='bold' if charge == selected_charge else 'normal',
            bbox=dict(facecolor='white', alpha=0.8, pad=2)
        )
        
        # Show integration range if defined
        if scale_ranges and (protein_name, charge) in scale_ranges:
            range_min, range_max = scale_ranges[(protein_name, charge)]
            ax.axvspan(
                range_min, 
                range_max, 
                alpha=0.2, 
                color=color, 
                label=f"Integration range {charge}+" if charge == selected_charge else ""
            )
    
    # Add legend
    ax.plot(
        [], [], 
        color="red", 
        linestyle="-", 
        linewidth=2, 
        label=f"Selected: {selected_charge}+"
    )
    ax.plot(
        [], [], 
        color="blue", 
        linestyle="--", 
        alpha=0.5, 
        label="Other charge states"
    )
    
    # Set labels and title
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(
        f"Mass Spectrum for {protein_name} - "
        f"All Charge States and Integration Ranges"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return fig
