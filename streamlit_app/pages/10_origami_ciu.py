"""
ORIGAMI 2.0
Streamlit app applying ORIGAMI ANALYZE to TWIMExtract data.
"""

import sys
from pathlib import Path

# Add parent directory to path to import imsocio package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import json
from typing import Tuple, List, Optional, Dict, Any

# Import from imsocio package
from imsocio.processing import (
    safe_float_conversion,
    remove_duplicate_values,
    interpolate_matrix,
    smooth_matrix_gaussian,
    smooth_matrix_savgol,
)

# Import Streamlit UI helpers
from streamlit_app import styling


class ORIGAMIInterface:
    """Streamlit interface for ORIGAMI 2.0."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown(
            '<div class="main-header">'
            '<h1>📎 ORIGAMI 2.0</h1>'
            '<p>Implementing core ORIGAMI analyze<sup>1</sup> functionality for TWIMExtract<sup>2</sup> data.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Streamlit implementation of core ORIGAMI functionality for activated ion mobility spectrometry (aIMS) visualization.</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li><strong>CCS Calibration:</strong> Convert drift times to CCSs using calibration files</li>
                <li><strong>Replicate Averaging:</strong> Upload multiple replicates to compute mean ± std deviation</li>
                <li><strong>RMSD Analysis:</strong> Calculate overall RMSD and RMSD<sub>CV</sub> for fingerprint comparison (CIUSuite/ORIGAMI style)</li>
                <li><strong>2D Interpolation:</strong> Increase data point density with linear or cubic interpolation</li>
                <li><strong>Smoothing:</strong> Gaussian orSavitzky-Golay smoothing</li>
                <li><strong>Normalisation:</strong> Normalise each collision voltage slice</li>
                <li><strong>CIU₅₀ Analysis:</strong> Extract modal CCS, fit sigmoidal curves, and identify conformational transition voltages</li>
                <li><strong>Figure Generation:</strong> Both static <code>matplotlib</code> graphs and interactive <code>plotly</code> graphs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def save_settings_to_dict() -> Dict[str, Any]:
        """Save current settings to a dictionary.
        
        Returns:
            Dictionary containing all current settings
        """
        settings = {}
        
        # Data settings
        for key in ['is_cyclic', 'inject_time', 'interp_multiplier', 'interp_method', 
                    'normalize_cv']:
            if key in st.session_state:
                settings[key] = st.session_state[key]
        
        # Smoothing settings
        for key in ['apply_smoothing', 'smoothing_method', 'gaussian_sigma', 
                    'gaussian_truncate', 'sg_window_length', 'sg_polyorder', 'sg_mode']:
            if key in st.session_state:
                settings[key] = st.session_state[key]
        
        # Figure customization
        for key in ['use_custom_color', 'hex_color', 'color_scheme', 'reverse_colors',
                    'font_family', 'font_size', 'figure_width_inches', 'figure_height_inches',
                    'figure_dpi', 'show_colorbar']:
            if key in st.session_state:
                settings[key] = st.session_state[key]
        
        # Axis limits
        for key in ['auto_x_limits', 'x_min', 'x_max', 'auto_y_limits', 'y_min', 'y_max']:
            if key in st.session_state:
                settings[key] = st.session_state[key]
        
        # Other settings
        for key in ['colorbar_title', 'custom_title']:
            if key in st.session_state:
                settings[key] = st.session_state[key]
        
        return settings
    
    @staticmethod
    def load_settings_from_dict(settings: Dict[str, Any]):
        """Load settings from dictionary into session state.
        
        Args:
            settings: Dictionary containing settings to load
        """
        for key, value in settings.items():
            st.session_state[key] = value
    
    @staticmethod
    def show_settings_management():
        """Show settings save/load interface."""
        with st.expander("⚙️ Save/Import Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Save Settings")
                if st.button("Save Current Settings"):
                    settings = ORIGAMIInterface.save_settings_to_dict()
                    settings_json = json.dumps(settings, indent=2)
                    
                    st.download_button(
                        label="Download Settings File",
                        data=settings_json,
                        file_name="settings.json",
                        mime="application/json",
                        help="Download your current settings to reuse later"
                    )
                    st.success("Settings ready for download!")
            
            with col2:
                st.subheader("Load Settings")
                
                if st.button("Clear Settings File"):
                    # Clear all settings from session state
                    keys_to_clear = ['interp_multiplier', 'interp_method', 'normalize_cv', 
                                    'apply_smoothing', 'show_colorbar', 'color_scheme',
                                    'use_custom_color', 'reverse_colors', 'font_size',
                                    'figure_width_inches', 'figure_height_inches', 'figure_dpi',
                                    'auto_x_limits', 'auto_y_limits', 'custom_title', 'colorbar_title']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                    
                settings_file = st.file_uploader(
                    "Upload Settings File",
                    type=['json'],
                    help="Upload a previously saved settings file"
                )
                
                if settings_file is not None:
                    try:
                        settings = json.loads(settings_file.read().decode('utf-8'))
                        ORIGAMIInterface.load_settings_from_dict(settings)
                        st.success("✅ Settings loaded! The widgets below now use your saved values.")
                    except Exception as e:
                        st.error(f"Error loading settings: {str(e)}")
    
    @staticmethod
    def show_file_upload() -> Tuple[Optional[Any], Optional[list], str, bool, bool]:
        """Show file upload widgets for calibration and TWIMExtract files.
        
        Returns:
            Tuple of (calibration_files, twim_files, data_mode, replicate_mode, ciu50_analysis)
            Note: calibration_files is a single file or list of files depending on replicate_mode
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📁 Upload Files</h3>', unsafe_allow_html=True)
        
        # Data mode selection
        data_mode = st.radio(
            "Data Acquisition Mode:",
            ["Standard TWIMExtract", "ORIGAMI"],
            help="Standard: Voltages in $TrapCV: row. ORIGAMI: Voltages extracted from range file names in first row."
        )
        
        # Replicate mode and CIU50 analysis options
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            replicate_mode = st.checkbox(
                "Replicate averaging mode",
                value=False,
                help="Upload multiple replicates to plot mean ± std deviation"
            )
        
        with col_opt2:
            ciu50_analysis = st.checkbox(
                "Enable CIU50 analysis",
                value=False,
                help="Extract modal CCS at each voltage and fit sigmoidal curves to transitions"
            )
        
        # Different UI based on replicate mode
        if replicate_mode:
            st.markdown("---")
            st.markdown("#### 📊 Replicate Files")
            st.info("💡 **Upload files for each replicate**: Each replicate needs its own calibration file (accounting for daily calibration differences)")
            
            # Number of replicates selector
            n_replicates = st.number_input(
                "Number of replicates:",
                min_value=2,
                max_value=10,
                value=st.session_state.get('n_replicates', 3),
                step=1,
                help="Specify how many replicates you want to upload",
                key='n_replicates'
            )
            
            # Initialize lists to store files
            calibration_files = []
            twim_files = []
            
            # Create upload sections for each replicate
            for i in range(n_replicates):
                with st.expander(f"📁 Replicate {i+1}", expanded=(i<2)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cal_file = st.file_uploader(
                            f"Calibration File",
                            type=['csv', 'txt'],
                            help="CSV file with columns: Z, Drift, CCS, CCS Std.Dev.",
                            key=f"cal_file_{i}"
                        )
                        if cal_file:
                            st.success(f"✓ {cal_file.name}")
                            calibration_files.append(cal_file)
                    
                    with col2:
                        twim_file = st.file_uploader(
                            f"TWIMExtract File",
                            type=['csv', 'txt'],
                            help="CSV file from TWIMExtract with drift times and intensities",
                            key=f"twim_file_{i}"
                        )
                        if twim_file:
                            st.success(f"✓ {twim_file.name}")
                            twim_files.append(twim_file)
            
            # Validation
            if len(calibration_files) != len(twim_files):
                st.warning(f"⚠️ Upload status: {len(calibration_files)} calibration files, {len(twim_files)} TWIMExtract files. Each replicate needs both files.")
            elif len(calibration_files) < 2:
                st.warning(f"⚠️ Replicate mode requires at least 2 complete replicate sets. Currently: {len(calibration_files)} sets uploaded.")
            else:
                st.success(f"✅ {len(calibration_files)} replicate sets ready for processing")
        
        else:
            # Single file mode
            col1, col2 = st.columns(2)

            with col1:
                calibration_file = st.file_uploader(
                    "Upload Calibration File",
                    type=['csv', 'txt'],
                    help="CSV file with columns: Z, Drift, CCS, CCS Std.Dev."
                )
                calibration_files = calibration_file  # Keep as single file for non-replicate mode

            with col2:
                twim_file = st.file_uploader(
                    "Upload TWIMExtract File",
                    type=['csv', 'txt'],
                    help="CSV file from TWIMExtract with drift times and intensities"
                )
                twim_files = [twim_file] if twim_file else []
        
        st.markdown('</div>', unsafe_allow_html=True)
        return calibration_files, twim_files, data_mode, replicate_mode, ciu50_analysis


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Load custom styling
    styling.load_custom_css()
    
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    # Show header
    ORIGAMIInterface.show_header()
    
    # Show settings management
    ORIGAMIInterface.show_settings_management()
    
    # Show file upload
    calibration_files, twim_files, data_mode, replicate_mode, ciu50_analysis = ORIGAMIInterface.show_file_upload()
    
    # Return uploaded files and options
    return calibration_files, twim_files, data_mode, replicate_mode, ciu50_analysis


# Run main application
calibration_files, twim_files, data_mode, replicate_mode, ciu50_analysis = main()


# ============================================================================
# Data Processing
# ============================================================================

def parse_origami_format(twim_content):
    """Parse ORIGAMI format where voltages are in the first row (range file names).
    
    Args:
        twim_content: List of lines from the file
        
    Returns:
        Tuple of (trap_cv_values, data_start_idx, removed_trapcv_indices)
    """
    import re
    
    # First row contains range file names
    first_row = twim_content[0]
    st.write(f"🔍 DEBUG parse_origami: first_row = {first_row[0:200]}...")  # First 200 chars
    range_files = first_row.split(',')[1:]  # Skip first column (which is empty or header)
    st.write(f"🔍 DEBUG parse_origami: Found {len(range_files)} range files")
    st.write(f"🔍 DEBUG parse_origami: First 5 range files: {range_files[:5]}")
    st.write(f"🔍 DEBUG parse_origami: Last 5 range files: {range_files[-5:]}")
    
    trap_cv_values = []
    for range_file in range_files:
        range_file = range_file.strip()
        if range_file == '':
            continue
        
        # Extract voltage from filename (e.g., "20V.txt" -> 20.0)
        voltage_match = re.search(r'(\d+(?:\.\d+)?)V', range_file)
        
        if voltage_match:
            voltage = float(voltage_match.group(1))
            trap_cv_values.append(voltage)
        else:
            st.warning(f"Could not extract voltage from: {range_file}")
    
    st.write(f"🔍 DEBUG parse_origami: Extracted {len(trap_cv_values)} voltages")
    st.write(f"🔍 DEBUG parse_origami: trap_cv_values[:10] = {trap_cv_values[:10]}")
    st.write(f"🔍 DEBUG parse_origami: trap_cv_values[-10:] = {trap_cv_values[-10:]}")
    
    if not trap_cv_values:
        st.error("Could not extract any voltages from range file row")
        return None, None, None
    
    # Data starts from second row
    data_start_idx = 1
    
    # Remove duplicates
    original_trap_cv_count = len(trap_cv_values)
    trap_cv_values_clean, removed_trapcv_indices = remove_duplicate_values(trap_cv_values)
    
    st.write(f"🔍 DEBUG parse_origami: After remove_duplicate_values:")
    st.write(f"  trap_cv_values_clean type={type(trap_cv_values_clean)}, len={len(trap_cv_values_clean)}")
    st.write(f"  trap_cv_values_clean[:10] = {trap_cv_values_clean[:10]}")
    st.write(f"  trap_cv_values_clean[-10:] = {trap_cv_values_clean[-10:]}")
    
    if len(removed_trapcv_indices) > 0:
        st.warning(f"Removed {len(removed_trapcv_indices)} duplicate voltages: {[trap_cv_values[i] for i in removed_trapcv_indices]}")
        st.info(f"Original voltage count: {original_trap_cv_count} → Clean voltage count: {len(trap_cv_values_clean)}")
    
    return trap_cv_values_clean, data_start_idx, removed_trapcv_indices


def parse_standard_format(twim_content):
    """Parse standard TWIMExtract format where voltages are in $TrapCV: row.
    
    Args:
        twim_content: List of lines from the file
        
    Returns:
        Tuple of (trap_cv_values, data_start_idx, removed_trapcv_indices)
    """
    # Find the TrapCV row
    trap_cv_values = None
    data_start_idx = None
    
    for i, line in enumerate(twim_content):
        if line.startswith('$TrapCV:'):
            trap_cv_str = line.split(',')[1:]
            trap_cv_values = []
            for x in trap_cv_str:
                cleaned = x.strip()
                if cleaned != '':
                    try:
                        trap_cv_values.append(float(cleaned))
                    except ValueError:
                        st.warning(f"Could not parse TrapCV value: {cleaned}")
                        continue
            data_start_idx = i + 1
            break
    
    if trap_cv_values is None or len(trap_cv_values) == 0:
        st.error("Could not find valid $TrapCV: row in the TWIMExtract file")
        return None, None, None
    
    # Remove duplicate TrapCV values
    original_trap_cv_count = len(trap_cv_values)
    trap_cv_values_clean, removed_trapcv_indices = remove_duplicate_values(trap_cv_values)
    
    if len(removed_trapcv_indices) > 0:
        st.warning(f"Removed {len(removed_trapcv_indices)} duplicate TrapCV values: {[trap_cv_values[i] for i in removed_trapcv_indices]}")
        st.info(f"Original TrapCV count: {original_trap_cv_count} → Clean TrapCV count: {len(trap_cv_values_clean)}")
    
    return trap_cv_values_clean, data_start_idx, removed_trapcv_indices


# ============================================================================
# CIU50 Analysis Functions
# ============================================================================

def extract_modal_ccs(ccs_values, trap_cv_values, intensity_matrix, ccs_ranges=None):
    """Extract modal CCS value at each collision voltage.
    
    Args:
        ccs_values: Array of CCS values
        trap_cv_values: Array of collision voltage values
        intensity_matrix: 2D intensity matrix (CCS x Voltage)
        ccs_ranges: Optional list of (ccs_min, ccs_max) tuples to restrict search
        
    Returns:
        Dictionary mapping voltage to modal CCS (or list of modal CCS if ranges provided)
    """
    modal_ccs_data = {}
    
    if ccs_ranges is None:
        # Single modal CCS per voltage (global maximum)
        for j, voltage in enumerate(trap_cv_values):
            intensity_slice = intensity_matrix[:, j]
            max_idx = np.argmax(intensity_slice)
            modal_ccs_data[voltage] = ccs_values[max_idx]
    else:
        # Multiple modal CCS per voltage (one per range)
        for j, voltage in enumerate(trap_cv_values):
            intensity_slice = intensity_matrix[:, j]
            modal_ccs_list = []
            
            for ccs_min, ccs_max in ccs_ranges:
                # Find indices within range
                mask = (ccs_values >= ccs_min) & (ccs_values <= ccs_max)
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    # Find maximum within this range
                    intensities_in_range = intensity_slice[mask]
                    local_max_idx = np.argmax(intensities_in_range)
                    global_idx = indices[local_max_idx]
                    modal_ccs_list.append(ccs_values[global_idx])
                else:
                    modal_ccs_list.append(np.nan)
            
            modal_ccs_data[voltage] = modal_ccs_list
    
    return modal_ccs_data


def sigmoid(x, L, x0, k, b):
    """Sigmoid function for CIU50 fitting.
    
    Args:
        x: Independent variable (voltage)
        L: Maximum value (upper asymptote)
        x0: Midpoint (CIU50 value)
        k: Steepness
        b: Baseline (lower asymptote)
        
    Returns:
        Sigmoid function value
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid_and_find_ciu50(voltages, modal_ccs_values, initial_ccs, final_ccs):
    """Fit sigmoid curve to CCS vs voltage data and find CIU50.
    
    Args:
        voltages: Array of collision voltages
        modal_ccs_values: Array of modal CCS values
        initial_ccs: Starting CCS value (for initial guess)
        final_ccs: Final CCS value (for initial guess)
        
    Returns:
        Tuple of (fit_params, ciu50, fitted_curve, r_squared)
    """
    from scipy.optimize import curve_fit
    
    # Remove NaN values
    valid_mask = ~np.isnan(modal_ccs_values)
    voltages_clean = voltages[valid_mask]
    ccs_clean = modal_ccs_values[valid_mask]
    
    if len(voltages_clean) < 4:
        return None, None, None, None
    
    # Initial parameter guesses
    L_guess = abs(final_ccs - initial_ccs)
    x0_guess = np.mean(voltages_clean)
    k_guess = 0.5
    b_guess = min(initial_ccs, final_ccs)
    
    try:
        # Fit sigmoid
        popt, _ = curve_fit(
            sigmoid,
            voltages_clean,
            ccs_clean,
            p0=[L_guess, x0_guess, k_guess, b_guess],
            maxfev=10000
        )
        
        # Extract CIU50 (midpoint x0)
        ciu50 = popt[1]
        
        # Generate fitted curve
        voltage_fine = np.linspace(voltages_clean.min(), voltages_clean.max(), 200)
        fitted_curve = sigmoid(voltage_fine, *popt)
        
        # Calculate R²
        residuals = ccs_clean - sigmoid(voltages_clean, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ccs_clean - np.mean(ccs_clean))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return popt, ciu50, (voltage_fine, fitted_curve), r_squared
        
    except Exception as e:
        st.warning(f"Sigmoid fitting failed: {str(e)}")
        return None, None, None, None


def plot_ciu50_analysis(modal_ccs_data, trap_cv_values, conformer_ranges, 
                        conformer_labels, font_family='Arial', font_size=12):
    """Plot modal CCS vs voltage with sigmoid fits and CIU50 labels.
    
    Args:
        modal_ccs_data: Dictionary mapping voltage to list of modal CCS values
        trap_cv_values: Array of collision voltages
        conformer_ranges: List of (ccs_min, ccs_max) tuples
        conformer_labels: List of labels for each conformer
        font_family: Font family for plot
        font_size: Font size for plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plt.rcParams.update({'font.family': font_family, 'font.size': font_size})
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(conformer_ranges)))
    
    # Extract modal CCS for each conformer
    for i, (label, color) in enumerate(zip(conformer_labels, colors)):
        voltages = []
        modal_ccs_values = []
        
        for voltage in sorted(modal_ccs_data.keys()):
            voltages.append(voltage)
            modal_ccs_values.append(modal_ccs_data[voltage][i])
        
        voltages = np.array(voltages)
        modal_ccs_values = np.array(modal_ccs_values)
        
        # Plot data points
        ax.scatter(voltages, modal_ccs_values, color=color, s=50, 
                   label=f"{label}", zorder=3, edgecolors='black', linewidth=0.5)
        
        # Fit sigmoid if transition is detected
        if not np.all(np.isnan(modal_ccs_values)):
            # Estimate initial and final CCS
            valid_ccs = modal_ccs_values[~np.isnan(modal_ccs_values)]
            if len(valid_ccs) > 3:
                initial_ccs = valid_ccs[0]
                final_ccs = valid_ccs[-1]
                
                # Only fit if there's significant change
                if abs(final_ccs - initial_ccs) > 10:  # At least 10 Ų change
                    popt, ciu50, fitted_curve, r_squared = fit_sigmoid_and_find_ciu50(
                        voltages, modal_ccs_values, initial_ccs, final_ccs
                    )
                    
                    if fitted_curve is not None:
                        voltage_fine, ccs_fitted = fitted_curve
                        ax.plot(voltage_fine, ccs_fitted, color=color, 
                               linestyle='--', linewidth=2, alpha=0.7, zorder=2)
                        
                        # Add CIU50 label
                        if ciu50 is not None:
                            ccs_at_ciu50 = sigmoid(ciu50, *popt)
                            ax.axvline(ciu50, color=color, linestyle=':', alpha=0.5, zorder=1)
                            ax.annotate(
                                f'CIU₅₀ = {ciu50:.1f}V\nR² = {r_squared:.3f}',
                                xy=(ciu50, ccs_at_ciu50),
                                xytext=(10, 10),
                                textcoords='offset points',
                                fontsize=font_size-2,
                                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
                            )
    
    ax.set_xlabel('Collision Voltage (V)', fontsize=font_size+2, fontweight='bold')
    ax.set_ylabel('Modal CCS (Ų)', fontsize=font_size+2, fontweight='bold')
    ax.set_title('CIU50 Analysis: Conformational Transitions', 
                 fontsize=font_size+4, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=font_size, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# Handle calibration files - convert to list format for consistent processing
if replicate_mode:
    cal_files_list = calibration_files if calibration_files else []
else:
    cal_files_list = [calibration_files] if calibration_files else []

if cal_files_list and twim_files and len(twim_files) > 0:
    # Validate file counts in replicate mode
    if replicate_mode and len(cal_files_list) != len(twim_files):
        st.error(f"⚠️ Number of calibration files ({len(cal_files_list)}) must match number of TWIMExtract files ({len(twim_files)}) in replicate mode.")
        st.stop()
    
    try:
        # Read calibration files
        cal_dfs = []
        for idx, cal_file in enumerate(cal_files_list):
            cal_df = pd.read_csv(cal_file)
            
            # Remove rows where error is larger than CCS value
            initial_rows = len(cal_df)
            cal_df = cal_df[cal_df['CCS Std.Dev.'] <= cal_df['CCS']]
            removed_rows = initial_rows - len(cal_df)
            
            cal_dfs.append(cal_df)
            
            if replicate_mode:
                st.success(f"Calibration file {idx+1} loaded: {len(cal_df)} calibration points")
            else:
                st.success(f"Calibration file loaded: {len(cal_df)} calibration points")
            
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} points where error ≥ CCS value")
        
        # For non-replicate mode, use the first (and only) calibration file
        if not replicate_mode:
            cal_df = cal_dfs[0]
        
        # Display mode information
        if replicate_mode:
            st.info(f"📊 Replicate Mode: Processing {len(twim_files)} files for averaging")
            if len(twim_files) < 2:
                st.warning("⚠️ Replicate mode requires at least 2 files. Please upload more files or disable replicate mode.")
                st.stop()
            
            if data_mode == "ORIGAMI":
                st.warning("⚠️ ORIGAMI format detected: Each file will have its voltages extracted independently from range filenames. All data will be interpolated to a common voltage grid.")
        
        # Read first TWIMExtract file to extract trap CV values and data format
        twim_file = twim_files[0]
        twim_content = twim_file.read().decode('utf-8').split('\n')
        
        # Parse data based on mode
        if data_mode == "ORIGAMI":
            # ORIGAMI format: voltages in first row (range file names)
            trap_cv_values, data_start_idx, removed_trapcv_indices = parse_origami_format(twim_content)
        else:
            # Standard format: voltages in $TrapCV: row
            trap_cv_values, data_start_idx, removed_trapcv_indices = parse_standard_format(twim_content)
        
        st.write(f"🔍 DEBUG: After parsing first file:")
        st.write(f"  trap_cv_values type = {type(trap_cv_values)}")
        st.write(f"  trap_cv_values shape/len = {trap_cv_values.shape if hasattr(trap_cv_values, 'shape') else len(trap_cv_values) if trap_cv_values is not None else 'None'}")
        st.write(f"  trap_cv_values[:10] = {trap_cv_values[:10] if trap_cv_values is not None else 'None'}")
        st.write(f"  trap_cv_values[-10:] = {trap_cv_values[-10:] if trap_cv_values is not None else 'None'}")
        st.write(f"  trap_cv_values min/max = {np.min(trap_cv_values) if trap_cv_values is not None else 'None'} / {np.max(trap_cv_values) if trap_cv_values is not None else 'None'}")
        st.write(f"  removed_trapcv_indices = {removed_trapcv_indices}")
        
        if trap_cv_values is None or data_start_idx is None:
            st.stop()
        
        # Keep voltages in file order for now (will sort after DataFrame creation)
        trap_cv_values_original = trap_cv_values.copy()
        
        # Reset file pointer by reopening
        twim_file.seek(0)
        
        # Parse data rows
        data_rows = []
        invalid_rows = 0
        
        for line_num, line in enumerate(twim_content[data_start_idx:], start=data_start_idx+1):
            if line.strip() == '':
                continue
            
            values = line.split(',')
            if len(values) <= 1:
                continue
            
            try:
                drift_time = safe_float_conversion(values[0])
                
                intensities = []
                for i in range(1, len(values)):
                    original_trapcv_idx = i - 1
                    
                    if original_trapcv_idx in removed_trapcv_indices:
                        continue
                    
                    if len(intensities) < len(trap_cv_values):
                        intensity_val = safe_float_conversion(values[i])
                        intensities.append(intensity_val)
                
                while len(intensities) < len(trap_cv_values):
                    intensities.append(0.0)
                
                intensities = intensities[:len(trap_cv_values)]
                data_rows.append([drift_time] + intensities)
                
            except Exception as e:
                invalid_rows += 1
                if invalid_rows <= 5:
                    st.warning(f"Error parsing line {line_num}: {str(e)}")
                continue
        
        if invalid_rows > 5:
            st.warning(f"... and {invalid_rows - 5} more parsing errors")
        
        if len(data_rows) == 0:
            st.error("No valid data rows found in TWIMExtract file")
            st.stop()
        
        # Create DataFrame with columns in file order (matching data_rows)
        columns = ['Drift_Time'] + [f'TrapCV_{cv}' for cv in trap_cv_values_original]
        twim_df = pd.DataFrame(data_rows, columns=columns)
        
        # Sort trap_cv_values and reorder DataFrame columns if needed
        if not np.all(np.diff(trap_cv_values_original) > 0):
            st.info(f"⚠️ Detected non-monotonic TrapCV values (ORIGAMI alphabetical sort) - reordering to numerical")
            trap_cv_sort_idx = np.argsort(trap_cv_values_original)
            trap_cv_values = np.array(trap_cv_values_original)[trap_cv_sort_idx]
            
            # Reorder DataFrame columns to match sorted voltages
            sorted_intensity_cols = [f'TrapCV_{trap_cv_values_original[i]}' for i in trap_cv_sort_idx]
            twim_df = twim_df[['Drift_Time'] + sorted_intensity_cols]
            
            # Rename columns to match new sorted voltage values
            rename_dict = {old_col: f'TrapCV_{new_val}' 
                          for old_col, new_val in zip(sorted_intensity_cols, trap_cv_values)}
            twim_df =twim_df.rename(columns=rename_dict)
            
            st.success(f"✓ Reordered {len(trap_cv_values)} voltage columns: {trap_cv_values[0]:.0f}V to {trap_cv_values[-1]:.0f}V")
        else:
            trap_cv_values = trap_cv_values_original
        
        # Ensure all columns are numeric
        for col in twim_df.columns:
            if col != 'Drift_Time':
                twim_df[col] = twim_df[col].apply(safe_float_conversion)
        
        st.success(f"TWIMExtract file loaded: {len(twim_df)} drift time points, {len(trap_cv_values)} TrapCV values")
        if invalid_rows > 0:
            st.info(f"Skipped {invalid_rows} invalid rows during parsing")
        
        # ====================================================================
        # Process additional files in replicate mode
        # ====================================================================
        
        if replicate_mode and len(twim_files) > 1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("**Processing replicate files for averaging...**", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store all dataframes with their corresponding voltages
            st.write(f"🔍 DEBUG: First file trap_cv_values type={type(trap_cv_values)}, len={len(trap_cv_values)}")
            st.write(f"🔍 DEBUG: First file trap_cv_values[:10] = {trap_cv_values[:10]}")
            st.write(f"🔍 DEBUG: First file trap_cv_values[-10:] = {trap_cv_values[-10:]}")
            st.write(f"🔍 DEBUG: First file trap_cv_values min={np.min(trap_cv_values)}, max={np.max(trap_cv_values)}")
            twim_dfs = [(twim_df.copy(), trap_cv_values)]
            
            # Process remaining files
            for file_idx, additional_file in enumerate(twim_files[1:], start=2):
                with st.spinner(f"Processing file {file_idx}/{len(twim_files)}..."):
                    try:
                        # Read file content
                        file_content = additional_file.read().decode('utf-8').split('\n')
                        additional_file.seek(0)  # Reset for potential reuse
                        
                        # Parse data based on mode - extract voltages for THIS file
                        if data_mode == "ORIGAMI":
                            # ORIGAMI format: each file has its own voltages in first row
                            trap_cv_values_rep, data_start_idx_rep, removed_trapcv_indices_rep = parse_origami_format(file_content)
                        else:
                            # Standard format: voltages in $TrapCV: row
                            trap_cv_values_rep, data_start_idx_rep, removed_trapcv_indices_rep = parse_standard_format(file_content)
                        
                        st.write(f"🔍 DEBUG: File {file_idx} trap_cv_values_rep type={type(trap_cv_values_rep)}")
                        st.write(f"🔍 DEBUG: File {file_idx} trap_cv_values_rep[:10] = {trap_cv_values_rep[:10]}")
                        st.write(f"🔍 DEBUG: File {file_idx} trap_cv_values_rep[-10:] = {trap_cv_values_rep[-10:]}")
                        st.write(f"🔍 DEBUG: File {file_idx} trap_cv_values_rep min={np.min(trap_cv_values_rep)}, max={np.max(trap_cv_values_rep)}")
                        
                        if trap_cv_values_rep is None:
                            st.warning(f"⚠️ Failed to parse voltages from file {file_idx}")
                            continue
                        
                        # Keep original order for parsing
                        trap_cv_values_rep_original = trap_cv_values_rep.copy()
                        
                        # Parse data rows
                        data_rows_rep = []
                        invalid_rows_rep = 0
                        
                        for line_num, line in enumerate(file_content[data_start_idx_rep:], start=data_start_idx_rep):
                            if not line.strip():
                                continue
                            
                            try:
                                values = line.split(',')
                                if len(values) < 2:
                                    continue
                                
                                drift_time = safe_float_conversion(values[0])
                                if drift_time is None:
                                    invalid_rows_rep += 1
                                    continue
                                
                                intensities = []
                                for i in range(1, len(values)):
                                    original_trapcv_idx = i - 1
                                    
                                    if original_trapcv_idx in removed_trapcv_indices_rep:
                                        continue
                                    
                                    if len(intensities) < len(trap_cv_values_rep):
                                        intensity_val = safe_float_conversion(values[i])
                                        intensities.append(intensity_val)
                                
                                while len(intensities) < len(trap_cv_values_rep):
                                    intensities.append(0.0)
                                
                                intensities = intensities[:len(trap_cv_values_rep)]
                                data_rows_rep.append([drift_time] + intensities)
                            
                            except Exception:
                                invalid_rows_rep += 1
                                continue
                        
                        # Create DataFrame for this replicate with columns in file order
                        columns_rep = ['Drift_Time'] + [f'TrapCV_{cv}' for cv in trap_cv_values_rep_original]
                        twim_df_rep = pd.DataFrame(data_rows_rep, columns=columns_rep)
                        
                        # Sort trap_cv_values and reorder DataFrame columns if needed
                        if not np.all(np.diff(trap_cv_values_rep_original) > 0):
                            st.info(f"File {file_idx}: Reordering non-monotonic voltages to numerical order")
                            trap_cv_sort_idx_rep = np.argsort(trap_cv_values_rep_original)
                            trap_cv_values_rep = np.array(trap_cv_values_rep_original)[trap_cv_sort_idx_rep]
                            
                            # Reorder DataFrame columns to match sorted voltages
                            sorted_intensity_cols_rep = [f'TrapCV_{trap_cv_values_rep_original[i]}' for i in trap_cv_sort_idx_rep]
                            twim_df_rep = twim_df_rep[['Drift_Time'] + sorted_intensity_cols_rep]
                            
                            # Rename columns to match new sorted voltage values
                            rename_dict_rep = {old_col: f'TrapCV_{new_val}' 
                                              for old_col, new_val in zip(sorted_intensity_cols_rep, trap_cv_values_rep)}
                            twim_df_rep = twim_df_rep.rename(columns=rename_dict_rep)
                        else:
                            trap_cv_values_rep = trap_cv_values_rep_original
                        
                        # Ensure numeric columns
                        for col in twim_df_rep.columns:
                            if col != 'Drift_Time':
                                twim_df_rep[col] = twim_df_rep[col].apply(safe_float_conversion)
                        
                        twim_dfs.append((twim_df_rep, trap_cv_values_rep))
                        st.success(f"✓ File {file_idx} loaded: {len(twim_df_rep)} drift points, {len(trap_cv_values_rep)} voltages ({trap_cv_values_rep[0]:.0f}-{trap_cv_values_rep[-1]:.0f}V)")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load file {file_idx}: {str(e)}")
                        continue
            
            st.info(f"✅ Loaded {len(twim_dfs)} replicate files successfully")
            
            # Create common voltage grid from all replicates
            all_voltages = []
            st.write(f"🔍 DEBUG: twim_dfs has {len(twim_dfs)} files")
            for idx, (df, voltages) in enumerate(twim_dfs):
                st.write(f"  File {idx+1}: voltages type={type(voltages)}, len={len(voltages) if hasattr(voltages, '__len__') else 'N/A'}")
                st.write(f"  File {idx+1}: voltages[:10] = {voltages[:10] if len(voltages) > 10 else voltages}")
                st.write(f"  File {idx+1}: voltages[-10:] = {voltages[-10:] if len(voltages) > 10 else voltages}")
                st.write(f"  File {idx+1}: min/max = {np.min(voltages)} / {np.max(voltages)}")
                all_voltages.extend(voltages)
            
            st.write(f"🔍 DEBUG: all_voltages has {len(all_voltages)} values total")
            st.write(f"🔍 DEBUG: all_voltages[:10] = {all_voltages[:10]}")
            st.write(f"🔍 DEBUG: all_voltages[-10:] = {all_voltages[-10:]}")
            
            # Use unique voltages sorted
            common_voltages = sorted(list(set(all_voltages)))
            st.write(f"🔍 DEBUG: After set/sort, common_voltages[:10] = {common_voltages[:10]}")
            st.write(f"🔍 DEBUG: After set/sort, common_voltages[-10:] = {common_voltages[-10:]}")
            st.write(f"🔍 DEBUG: After set/sort, common_voltages min/max = {np.min(common_voltages)} / {np.max(common_voltages)}")
            
            if data_mode == "ORIGAMI":
                st.info(f"📊 Common voltage grid created: {len(common_voltages)} unique voltages from {common_voltages[0]:.1f}V to {common_voltages[-1]:.1f}V")
            
            # Store twim_dfs with voltages in session state for access in processing
            st.session_state['twim_dfs_replicates'] = twim_dfs
            st.session_state['common_voltages'] = common_voltages
        else:
            st.session_state['twim_dfs_replicates'] = None
            st.session_state['common_voltages'] = None
        
        # Charge state selection
        charge_states = cal_df['Z'].unique()
        
        if len(charge_states) > 1:
            selected_charge = st.selectbox(
                "Select charge state for CCS conversion:",
                charge_states,
                help="Multiple charge states found in calibration file"
            )
        else:
            selected_charge = charge_states[0]
            st.info(f"Using charge state: {selected_charge}")
        
        # ====================================================================
        # Data Settings
        # ====================================================================
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚙️ Data Settings</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            is_cyclic = st.checkbox(
                "Is this cyclic data?",
                key="is_cyclic"
            )
            inject_time = None
            if is_cyclic:
                inject_time = st.number_input(
                    "Inject time (ms)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help="Inject time in milliseconds to subtract from drift times",
                    key="inject_time"
                )

        with col2:
            interp_multiplier = st.number_input(
                "Interpolation multiplier",
                min_value=1,
                max_value=20,
                value=st.session_state.get('interp_multiplier', 1),
                step=1,
                help="Multiply the number of data points in both dimensions",
                key="interp_multiplier"
            )
            
            interp_method = st.selectbox(
                "Interpolation method",
                ["linear", "cubic"],
                help="Choose interpolation method for adding data points",
                key="interp_method"
            )

        with col3:
            normalize_cv = st.checkbox(
                "Normalise CV slices",
                value=st.session_state.get('normalize_cv', False),
                help="Normalise each TrapCV column to its maximum value",
                key="normalize_cv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # Smoothing Settings
        # ====================================================================
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎨 Smoothing Settings</h3>', unsafe_allow_html=True)
        
        apply_smoothing = st.checkbox(
            "Apply smoothing to data",
            value=st.session_state.get('apply_smoothing', False),
            help="Apply smoothing after CCS conversion and optional interpolation",
            key="apply_smoothing"
        )
        
        if apply_smoothing:
            smoothing_method = st.selectbox(
                "Smoothing method",
                ["Gaussian", "Savitzky-Golay"],
                help="Choose smoothing algorithm",
                key="smoothing_method"
            )
            
            if smoothing_method == "Gaussian":
                col1, col2 = st.columns(2)
                with col1:
                    gaussian_sigma = st.number_input(
                        "Sigma",
                        min_value=0.1,
                        max_value=10.0,
                        value=st.session_state.get('gaussian_sigma', 1.0),
                        step=0.1,
                        help="Standard deviation for Gaussian",
                        key="gaussian_sigma"
                    )
                with col2:
                    gaussian_truncate = st.number_input(
                        "Truncate",
                        min_value=1.0,
                        max_value=10.0,
                        value=st.session_state.get('gaussian_truncate', 4.0),
                        step=0.5,
                        help="Truncate filter at this many standard deviations",
                        key="gaussian_truncate"
                    )
            
            elif smoothing_method == "Savitzky-Golay":
                col1, col2, col3 = st.columns(3)
                with col1:
                    sg_window_length = st.number_input(
                        "Window length",
                        min_value=3,
                        max_value=51,
                        value=st.session_state.get('sg_window_length', 11),
                        step=2,
                        help="Length of filter window (must be odd)",
                        key="sg_window_length"
                    )
                with col2:
                    sg_polyorder = st.number_input(
                        "Polynomial order",
                        min_value=1,
                        max_value=5,
                        value=st.session_state.get('sg_polyorder', 3),
                        step=1,
                        help="Order of polynomial used to fit samples",
                        key="sg_polyorder"
                    )
                with col3:
                    sg_mode = st.selectbox(
                        "Mode",
                        ["mirror", "nearest", "wrap", "interp"],
                        help="How to handle boundaries",
                        key="sg_mode"
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # Figure Customization
        # ====================================================================
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎨 Figure Customization</h3>', unsafe_allow_html=True)
        
        # Color settings
        st.markdown("**Color Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            use_custom_color = st.checkbox(
                "Use custom color",
                value=st.session_state.get('use_custom_color', False),
                help="Use a custom hex color instead of predefined schemes",
                key="use_custom_color"
            )
            
            if use_custom_color:
                hex_color = st.color_picker(
                    "Select color",
                    value="#FF0000",
                    help="Pick a color for the heatmap gradient",
                    key="hex_color"
                )
            else:
                color_scheme = st.selectbox(
                    "Color scheme",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", 
                     "Blues", "Reds", "Greens", "YlOrRd", "YlGnBu", "RdYlBu",
                     "Spectral", "Coolwarm", "Jet", "Hot", "Cool"],
                    help="Choose a predefined color scheme",
                    key="color_scheme"
                )
        
        with col2:
            reverse_colors = st.checkbox(
                "Reverse color scale",
                value=st.session_state.get('reverse_colors', False),
                help="Reverse the direction of the color gradient",
                key="reverse_colors"
            )
            
            show_colorbar = st.checkbox(
                "Show colorbar",
                value=st.session_state.get('show_colorbar', True),
                help="Display colorbar on the figure",
                key="show_colorbar"
            )
            
            if show_colorbar:
                colorbar_title = st.text_input(
                    "Colorbar title",
                    value=st.session_state.get('colorbar_title', 'Intensity'),
                    help="Label for the colorbar",
                    key="colorbar_title"
                )
        
        st.markdown("---")
        
        # Typography and size
        st.markdown("**Font and Figure Size**")
        col1, col2 = st.columns(2)
        
        with col1:
            font_family = st.selectbox(
                "Font family",
                ["Arial", "Times New Roman", "Courier New", "Helvetica", "Georgia"],
                help="Font for all text in the figure",
                key="font_family"
            )
            
            font_size = st.number_input(
                "Font size",
                min_value=8,
                max_value=24,
                value=st.session_state.get('font_size', 14),
                step=1,
                help="Font size for all text elements",
                key="font_size"
            )
        
        with col2:
            figure_width_inches = st.number_input(
                "Figure width (inches)",
                min_value=4.0,
                max_value=20.0,
                value=st.session_state.get('figure_width_inches', 10.0),
                step=0.5,
                help="Width of static figure in inches",
                key="figure_width_inches"
            )
            
            figure_height_inches = st.number_input(
                "Figure height (inches)",
                min_value=4.0,
                max_value=20.0,
                value=st.session_state.get('figure_height_inches', 8.0),
                step=0.5,
                help="Height of static figure in inches",
                key="figure_height_inches"
            )
        
        # Additional settings
        col1, col2 = st.columns(2)
        
        with col1:
            figure_dpi = st.number_input(
                "Figure DPI",
                min_value=72,
                max_value=1000,
                value=st.session_state.get('figure_dpi', 300),
                step=50,
                help="Resolution for static figure export",
                key="figure_dpi"
            )
        
        with col2:
            custom_title = st.text_input(
                "Custom title (optional)",
                value=st.session_state.get('custom_title', ''),
                help="Override automatic title generation",
                key="custom_title"
            )
        
        st.markdown("---")
        
        # Axis limits
        st.markdown("**Axis Limits**")
        col1, col2 = st.columns(2)
        
        with col1:
            auto_x_limits = st.checkbox(
                "Auto X-axis limits",
                value=st.session_state.get('auto_x_limits', True),
                help="Automatically determine X-axis range",
                key="auto_x_limits"
            )
            
            if not auto_x_limits:
                x_min = st.number_input(
                    "X min",
                    value=st.session_state.get('x_min', 0.0),
                    key="x_min"
                )
                x_max = st.number_input(
                    "X max",
                    value=st.session_state.get('x_max', 100.0),
                    key="x_max"
                )
        
        with col2:
            auto_y_limits = st.checkbox(
                "Auto Y-axis limits",
                value=st.session_state.get('auto_y_limits', True),
                help="Automatically determine Y-axis range",
                key="auto_y_limits"
            )
            
            if not auto_y_limits:
                y_min = st.number_input(
                    "Y min",
                    value=st.session_state.get('y_min', 0.0),
                    key="y_min"
                )
                y_max = st.number_input(
                    "Y max",
                    value=st.session_state.get('y_max', 1000.0),
                    key="y_max"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # Data Processing and Visualization
        # ====================================================================
        
        if st.button("🎯 Generate Fingerprint", type="primary"):
            with st.spinner("Processing data..."):
                # Filter calibration data for selected charge state
                cal_z_df = cal_df[cal_df['Z'] == selected_charge].copy()
                
                if len(cal_z_df) < 2:
                    st.error(f"Insufficient calibration points for charge state {selected_charge}. Need at least 2 points.")
                    st.stop()
                
                # Convert calibration drift times from seconds to milliseconds
                cal_z_df['Drift_ms'] = cal_z_df['Drift'] * 1000.0
                
                # Subtract inject time if cyclic
                if is_cyclic and inject_time is not None and inject_time > 0:
                    drift_times = twim_df['Drift_Time'].values - inject_time
                    st.info(f"Subtracted inject time ({inject_time} ms) from drift times")
                else:
                    drift_times = twim_df['Drift_Time'].values
                
                # Interpolate to get CCS for each drift time
                ccs_for_drift = np.interp(
                    drift_times,
                    cal_z_df['Drift_ms'].values,
                    cal_z_df['CCS'].values
                )
                
                # Build intensity matrix
                intensity_columns = [col for col in twim_df.columns if col.startswith('TrapCV_')]
                intensity_matrix_original = twim_df[intensity_columns].values
                
                st.write(f"🔍 Initial first file processing: Shape: {intensity_matrix_original.shape}, Non-zero: {np.count_nonzero(intensity_matrix_original)}, Max: {np.max(intensity_matrix_original):.2e}")
                
                # Create CCS-indexed dataframe
                ccs_df = pd.DataFrame({
                    'CCS': ccs_for_drift,
                })
                
                for i, col in enumerate(intensity_columns):
                    ccs_df[col] = intensity_matrix_original[:, i]
                
                # Sort by CCS and remove duplicates
                ccs_df = ccs_df.sort_values('CCS')
                original_ccs_count = len(ccs_df)
                
                ccs_values_raw = ccs_df['CCS'].values
                ccs_values_clean, removed_ccs_indices = remove_duplicate_values(ccs_values_raw)
                
                if len(removed_ccs_indices) > 0:
                    st.warning(f"Removed {len(removed_ccs_indices)} duplicate CCS values")
                
                # Filter dataframe to remove duplicates
                keep_mask = np.ones(len(ccs_df), dtype=bool)
                keep_mask[removed_ccs_indices] = False
                ccs_df = ccs_df[keep_mask].reset_index(drop=True)
                
                ccs_values = ccs_df['CCS'].values
                intensity_matrix_original = ccs_df[intensity_columns].values
                
                # Verify TrapCV values are monotonic
                if not np.all(np.diff(trap_cv_values) > 0):
                    trap_cv_sort_idx = np.argsort(trap_cv_values)
                    trap_cv_values = np.array(trap_cv_values)[trap_cv_sort_idx]
                    intensity_matrix_original = intensity_matrix_original[:, trap_cv_sort_idx]
                    st.info("Sorted TrapCV values for interpolation compatibility")
                
                # ================================================================
                # Process replicates if in replicate mode
                # ================================================================
                
                intensity_matrix_std = None  # Will store std dev if replicates exist
                overall_rmsd = None  # Overall RMSD across entire matrix
                rmsd_cv = None  # RMSD at each collision voltage
                stacked_matrices = None  # Store stacked replicate matrices for later cropping
                
                if replicate_mode and st.session_state.get('twim_dfs_replicates') is not None:
                    twim_dfs_list = st.session_state['twim_dfs_replicates']
                    common_voltages_grid = st.session_state.get('common_voltages')
                    
                    st.write(f"🔍 Replicate mode check: {len(twim_dfs_list)} files loaded from session")
                    
                    if len(twim_dfs_list) > 1:
                        st.info(f"🔄 Averaging {len(twim_dfs_list)} replicate files...")
                        
                        # When in replicate mode, we need to reprocess ALL files including the first one
                        # Use the first replicate to establish the common CCS grid
                        # Don't use the ccs_values/trap_cv_values from above as they may have been sorted
                        
                        # Get the first replicate's data to establish common grids
                        first_df, first_voltages = twim_dfs_list[0]
                        
                        # Get calibration for first replicate
                        if replicate_mode and len(cal_dfs) > 0:
                            cal_z_df_first = cal_dfs[0][cal_dfs[0]['Z'] == selected_charge].copy()
                            cal_z_df_first['Drift_ms'] = cal_z_df_first['Drift'] * 1000.0
                        else:
                            cal_z_df_first = cal_z_df
                        
                        if len(cal_z_df_first) < 2:
                            st.error(f"Insufficient calibration points for charge state {selected_charge} in first replicate. Need at least 2 points.")
                            st.stop()
                        
                        # Process first replicate to get CCS grid
                        if is_cyclic and inject_time is not None and inject_time > 0:
                            drift_times_first = first_df['Drift_Time'].values - inject_time
                        else:
                            drift_times_first = first_df['Drift_Time'].values
                        
                        ccs_for_drift_first = np.interp(
                            drift_times_first,
                            cal_z_df_first['Drift_ms'].values,
                            cal_z_df_first['CCS'].values
                        )
                        
                        # Build CCS dataframe for first replicate
                        intensity_columns_first = [f'TrapCV_{cv}' for cv in first_voltages]
                        intensity_matrix_first = first_df[intensity_columns_first].values
                        
                        ccs_df_first = pd.DataFrame({'CCS': ccs_for_drift_first})
                        for i, col in enumerate(intensity_columns_first):
                            ccs_df_first[col] = intensity_matrix_first[:, i]
                        
                        # Sort and remove duplicates
                        ccs_df_first = ccs_df_first.sort_values('CCS')
                        ccs_values_first_raw = ccs_df_first['CCS'].values
                        _, removed_indices_first = remove_duplicate_values(ccs_values_first_raw)
                        
                        keep_mask_first = np.ones(len(ccs_df_first), dtype=bool)
                        keep_mask_first[removed_indices_first] = False
                        ccs_df_first = ccs_df_first[keep_mask_first].reset_index(drop=True)
                        
                        # Use this as the common CCS grid
                        ccs_values = ccs_df_first['CCS'].values
                        
                        st.write(f"📊 Established common CCS grid from first replicate: {len(ccs_values)} points, range [{np.min(ccs_values):.1f}, {np.max(ccs_values):.1f}]")
                        
                        # Use common voltage grid if available (for ORIGAMI mode with different voltages)
                        if common_voltages_grid is not None and len(common_voltages_grid) > 0:
                            target_voltages = np.array(common_voltages_grid)
                            st.info(f"Using common voltage grid: {len(target_voltages)} voltages")
                            st.write(f"  Target voltage range: [{target_voltages[0]:.1f}, {target_voltages[-1]:.1f}]")
                        else:
                            # All files have same voltages (standard mode) - use first replicate's voltages
                            target_voltages = np.array(first_voltages)
                            st.write(f"  Using first replicate voltages: {len(target_voltages)} voltages, range [{target_voltages[0]:.1f}, {target_voltages[-1]:.1f}]")
                        
                        # Store all CCS-converted and voltage-interpolated matrices
                        all_matrices = []
                        
                        # Process each replicate
                        for rep_idx, (twim_df_rep, voltages_rep) in enumerate(twim_dfs_list):
                            st.write(f"🔄 Processing replicate {rep_idx + 1}/{len(twim_dfs_list)}...")
                            
                            # Get calibration for this replicate
                            if replicate_mode and len(cal_dfs) > rep_idx:
                                cal_z_df_rep = cal_dfs[rep_idx][cal_dfs[rep_idx]['Z'] == selected_charge].copy()
                                cal_z_df_rep['Drift_ms'] = cal_z_df_rep['Drift'] * 1000.0
                                
                                if len(cal_z_df_rep) < 2:
                                    st.error(f"⚠️ Insufficient calibration points for charge state {selected_charge} in replicate {rep_idx + 1}. Skipping this replicate.")
                                    continue
                            else:
                                cal_z_df_rep = cal_z_df
                            
                            # Apply same processing as first file
                            if is_cyclic and inject_time is not None and inject_time > 0:
                                drift_times_rep = twim_df_rep['Drift_Time'].values - inject_time
                            else:
                                drift_times_rep = twim_df_rep['Drift_Time'].values
                            
                            # Convert to CCS using this replicate's calibration
                            ccs_for_drift_rep = np.interp(
                                drift_times_rep,
                                cal_z_df_rep['Drift_ms'].values,
                                cal_z_df_rep['CCS'].values
                            )
                            
                            # Build intensity matrix from this replicate's columns
                            intensity_columns_rep = [f'TrapCV_{cv}' for cv in voltages_rep]
                            
                            # Check if columns exist
                            missing_cols = [col for col in intensity_columns_rep if col not in twim_df_rep.columns]
                            if missing_cols:
                                st.error(f"⚠️ Missing columns in replicate {rep_idx + 1}: {missing_cols[:5]}")
                                st.write(f"Available columns: {twim_df_rep.columns.tolist()[:10]}")
                                continue
                            
                            intensity_matrix_rep = twim_df_rep[intensity_columns_rep].values
                            
                            # Debug info
                            st.write(f"  - Shape: {intensity_matrix_rep.shape}, Non-zero: {np.count_nonzero(intensity_matrix_rep)}, Max: {np.max(intensity_matrix_rep):.2e}")
                            
                            # Create CCS-indexed dataframe
                            ccs_df_rep = pd.DataFrame({'CCS': ccs_for_drift_rep})
                            for i, col in enumerate(intensity_columns_rep):
                                ccs_df_rep[col] = intensity_matrix_rep[:, i]
                            
                            # Sort and remove duplicates
                            ccs_df_rep = ccs_df_rep.sort_values('CCS')
                            ccs_values_rep = ccs_df_rep['CCS'].values
                            _, removed_indices_rep = remove_duplicate_values(ccs_values_rep)
                            
                            keep_mask_rep = np.ones(len(ccs_df_rep), dtype=bool)
                            keep_mask_rep[removed_indices_rep] = False
                            ccs_df_rep = ccs_df_rep[keep_mask_rep].reset_index(drop=True)
                            
                            intensity_matrix_rep_clean = ccs_df_rep[intensity_columns_rep].values
                            
                            st.write(f"  - After duplicate removal: Non-zero: {np.count_nonzero(intensity_matrix_rep_clean)}, Max: {np.max(intensity_matrix_rep_clean):.2e}")
                            st.write(f"  - CCS range replicate: [{np.min(ccs_df_rep['CCS']):.1f}, {np.max(ccs_df_rep['CCS']):.1f}] (n={len(ccs_df_rep)})")
                            st.write(f"  - CCS range target: [{np.min(ccs_values):.1f}, {np.max(ccs_values):.1f}] (n={len(ccs_values)})")
                            st.write(f"  - Voltage range replicate: {len(voltages_rep)} voltages")
                            st.write(f"  - Voltage range target: {len(target_voltages)} voltages")
                            
                            # Now interpolate to common voltage grid if needed
                            # Check if voltage grids are actually different (not just array comparison)
                            voltages_match = (len(voltages_rep) == len(target_voltages) and 
                                            np.allclose(voltages_rep, target_voltages, rtol=1e-5))
                            
                            if common_voltages_grid is not None and not voltages_match:
                                # Need to interpolate to common voltage grid
                                st.write(f"  - Voltage interpolation needed: {len(voltages_rep)} -> {len(target_voltages)}")
                                st.write(f"  - Rep voltages: {voltages_rep[:5]} ... {voltages_rep[-5:]}")
                                st.write(f"  - Target voltages: {target_voltages[:5]} ... {target_voltages[-5:]}")
                                
                                intensity_matrix_voltage_interp = np.zeros((len(ccs_df_rep), len(target_voltages)))
                                
                                for ccs_idx in range(intensity_matrix_rep_clean.shape[0]):
                                    intensity_matrix_voltage_interp[ccs_idx, :] = np.interp(
                                        target_voltages,
                                        voltages_rep,
                                        intensity_matrix_rep_clean[ccs_idx, :],
                                        left=0,
                                        right=0
                                    )
                                
                                intensity_matrix_rep_clean = intensity_matrix_voltage_interp
                                st.write(f"  - After voltage interp: Non-zero: {np.count_nonzero(intensity_matrix_rep_clean)}, Max: {np.max(intensity_matrix_rep_clean):.2e}")
                            else:
                                st.write(f"  - No voltage interpolation needed (voltages match)")
                            
                            # Interpolate to common CCS grid (using first replicate's CCS values)
                            st.write(f"  - Starting CCS interpolation to common grid...")
                            
                            # Check if CCS grids are actually identical
                            ccs_match = (len(ccs_df_rep) == len(ccs_values) and 
                                       np.allclose(ccs_df_rep['CCS'].values, ccs_values, rtol=1e-5))
                            
                            if ccs_match:
                                st.write(f"  - CCS grids are identical, skipping interpolation")
                                intensity_matrix_interp = intensity_matrix_rep_clean
                            else:
                                intensity_matrix_interp = np.zeros((len(ccs_values), intensity_matrix_rep_clean.shape[1]))
                                
                                # Check for overlap
                                ccs_rep_min, ccs_rep_max = np.min(ccs_df_rep['CCS']), np.max(ccs_df_rep['CCS'])
                                ccs_target_min, ccs_target_max = np.min(ccs_values), np.max(ccs_values)
                                st.write(f"  - Interpolating: Rep [{ccs_rep_min:.1f}, {ccs_rep_max:.1f}] to Target [{ccs_target_min:.1f}, {ccs_target_max:.1f}]")
                                
                                for j in range(intensity_matrix_rep_clean.shape[1]):
                                    intensity_matrix_interp[:, j] = np.interp(
                                        ccs_values,  # Common CCS grid from first replicate
                                        ccs_df_rep['CCS'].values,
                                        intensity_matrix_rep_clean[:, j],
                                        left=0,
                                        right=0
                                    )
                            
                            st.write(f"  - After CCS interp: Non-zero: {np.count_nonzero(intensity_matrix_interp)}, Max: {np.max(intensity_matrix_interp):.2e}")
                            
                            # Normalize this replicate if CV normalization is enabled
                            # ORIGAMI normalizes each replicate BEFORE averaging
                            if normalize_cv:
                                intensity_matrix_normalized = intensity_matrix_interp.copy()
                                for j in range(intensity_matrix_normalized.shape[1]):
                                    col_max = np.max(intensity_matrix_normalized[:, j])
                                    if col_max > 0:
                                        intensity_matrix_normalized[:, j] = intensity_matrix_normalized[:, j] / col_max
                                all_matrices.append(intensity_matrix_normalized)
                                st.write(f"  - After CV normalization: Max per slice = 1.0")
                            else:
                                all_matrices.append(intensity_matrix_interp)
                        
                        # Debug: check all_matrices before stacking
                        st.write(f"📊 All matrices collected: {len(all_matrices)} matrices")
                        for idx, mat in enumerate(all_matrices):
                            st.write(f"  Matrix {idx + 1}: Shape {mat.shape}, Non-zero: {np.count_nonzero(mat)}, Max: {np.max(mat):.2e}")
                        
                        # Stack all matrices and compute mean and std
                        # Note: If normalize_cv is True, these are already normalized replicates
                        stacked_matrices = np.stack(all_matrices, axis=0)
                        intensity_matrix_original = np.mean(stacked_matrices, axis=0)
                        intensity_matrix_std = np.std(stacked_matrices, axis=0)
                        
                        # Calculate RMSD metrics (similar to CIUSuite/ORIGAMI)
                        # RMSD: Root Mean Square Deviation of replicates from mean
                        # Overall RMSD (global difference indicator)
                        differences_squared = (stacked_matrices - intensity_matrix_original) ** 2
                        overall_rmsd = np.sqrt(np.mean(differences_squared))
                        
                        # RMSD_CV: RMSD at each collision voltage (local differences)
                        # For each voltage column, compute RMSD across all CCS points and replicates
                        n_voltages = intensity_matrix_original.shape[1]
                        rmsd_cv = np.zeros(n_voltages)
                        for j in range(n_voltages):
                            # Get all replicates for this voltage column
                            replicate_columns = stacked_matrices[:, :, j]  # shape: (n_replicates, n_ccs)
                            mean_column = intensity_matrix_original[:, j]  # shape: (n_ccs,)
                            
                            # Compute RMSD for this voltage
                            diff_sq = (replicate_columns - mean_column) ** 2
                            rmsd_cv[j] = np.sqrt(np.mean(diff_sq))
                        
                        st.write(f"✅ After averaging: Non-zero: {np.count_nonzero(intensity_matrix_original)}, Max: {np.max(intensity_matrix_original):.2e}")
                        st.write(f"📊 RMSD Metrics:")
                        st.write(f"  - Overall RMSD: {overall_rmsd:.4f}")
                        st.write(f"  - RMSD_CV range: [{np.min(rmsd_cv):.4f}, {np.max(rmsd_cv):.4f}]")
                        
                        if normalize_cv:
                            st.info("CV slice normalization was applied to each replicate before averaging (ORIGAMI standard)")
                        
                        # Update trap_cv_values to use common grid
                        trap_cv_values = target_voltages
                        
                        if common_voltages_grid is not None:
                            st.info(f"📊 Using common voltage grid: {len(target_voltages)} voltages from {target_voltages[0]:.1f}V to {target_voltages[-1]:.1f}V")
                        
                        st.success(f"✅ Averaged {len(all_matrices)} replicates on common grid")
                
                # Store dimensions for reporting
                original_ccs_points = len(ccs_values)
                original_trapcv_points = len(trap_cv_values)
                
                # Store original grids before interpolation
                ccs_values_original = ccs_values.copy()
                trap_cv_values_original = trap_cv_values.copy()
                
                # Apply CV normalization if requested (for single file mode)
                # Note: In replicate mode with normalize_cv=True, normalization already happened per-replicate
                if normalize_cv and (not replicate_mode or len(twim_dfs_list) <= 1):
                    for j in range(intensity_matrix_original.shape[1]):
                        col_max = np.max(intensity_matrix_original[:, j])
                        if col_max > 0:
                            intensity_matrix_original[:, j] = intensity_matrix_original[:, j] / col_max
                    st.info("Applied CV slice normalisation")
                
                # Apply interpolation if requested
                if interp_multiplier > 1:
                    try:
                        ccs_values, trap_cv_values, intensity_matrix_final = interpolate_matrix(
                            ccs_values_original,
                            trap_cv_values_original,
                            intensity_matrix_original,
                            method=interp_method,
                            multiplier=interp_multiplier,
                        )
                        
                        # Also interpolate std dev if it exists
                        if intensity_matrix_std is not None:
                            _, _, intensity_matrix_std_final = interpolate_matrix(
                                ccs_values_original,
                                trap_cv_values_original,
                                intensity_matrix_std,
                                method=interp_method,
                                multiplier=interp_multiplier,
                            )
                            intensity_matrix_std = intensity_matrix_std_final
                        
                        st.info(
                            f"2D interpolation: {original_ccs_points}×{original_trapcv_points} → "
                            f"{len(ccs_values)}×{len(trap_cv_values)} points using {interp_method} method"
                        )
                    except Exception as interp_error:
                        st.error(f"Interpolation failed: {str(interp_error)}")
                        st.info("Using original data without interpolation")
                        intensity_matrix_final = intensity_matrix_original
                else:
                    intensity_matrix_final = intensity_matrix_original
                
                # Apply smoothing if requested
                if apply_smoothing:
                    if smoothing_method == "Gaussian":
                        intensity_matrix_final = smooth_matrix_gaussian(
                            intensity_matrix_final, 
                            sigma=gaussian_sigma, 
                            truncate=gaussian_truncate
                        )
                        
                        # Also smooth std dev if it exists
                        if intensity_matrix_std is not None:
                            intensity_matrix_std = smooth_matrix_gaussian(
                                intensity_matrix_std,
                                sigma=gaussian_sigma,
                                truncate=gaussian_truncate
                            )
                        
                        st.info(f"Applied Gaussian smoothing (σ={gaussian_sigma}, truncate={gaussian_truncate})")
                    elif smoothing_method == "Savitzky-Golay":
                        intensity_matrix_final = smooth_matrix_savgol(
                            intensity_matrix_final,
                            window_length=sg_window_length,
                            polyorder=sg_polyorder,
                            mode=sg_mode,
                        )
                        
                        # Also smooth std dev if it exists
                        if intensity_matrix_std is not None:
                            intensity_matrix_std = smooth_matrix_savgol(
                                intensity_matrix_std,
                                window_length=sg_window_length,
                                polyorder=sg_polyorder,
                                mode=sg_mode,
                            )
                        
                        st.info(f"Applied Savitzky-Golay smoothing (window={sg_window_length}, poly_order={sg_polyorder}, mode={sg_mode})")
                
                # ============================================================
                # Prepare color scheme for Plotly
                # ============================================================
                
                if use_custom_color:
                    hex_clean = hex_color.lstrip('#')
                    rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
                    
                    if reverse_colors:
                        colorscale = [
                            [0, f'rgb{rgb}'],
                            [1, 'rgb(255, 255, 255)']
                        ]
                    else:
                        colorscale = [
                            [0, 'rgb(255, 255, 255)'],
                            [1, f'rgb{rgb}']
                        ]
                else:
                    colorscale = color_scheme.lower()
                    if reverse_colors:
                        colorscale += "_r"
                
                # ============================================================
                # Create Plotly Interactive Figure
                # ============================================================
                
                colorbar_config = dict(
                    title=dict(
                        text=colorbar_title,
                        font=dict(size=font_size, family=font_family)
                    ),
                    tickfont=dict(size=font_size, family=font_family)
                ) if show_colorbar else None
                
                fig_matrix = go.Figure(data=go.Heatmap(
                    z=intensity_matrix_final,
                    x=trap_cv_values,
                    y=ccs_values,
                    colorscale=colorscale,
                    colorbar=colorbar_config,
                    showscale=show_colorbar,
                    zmin=0,
                    zauto=True
                ))
                
                # Build title
                if custom_title:
                    title = custom_title
                else:
                    title_parts = [f'CCS Fingerprint Matrix (Charge State {selected_charge})']
                    if interp_multiplier > 1:
                        title_parts.append(f'{interp_method.capitalize()} {interp_multiplier}x Interpolation')
                    if normalize_cv:
                        title_parts.append('CV Normalised')
                    if apply_smoothing:
                        title_parts.append(f'{smoothing_method} Smoothing')
                    title = ' - '.join(title_parts)
                
                figure_width = int(figure_width_inches * 96)
                figure_height = int(figure_height_inches * 96)
                
                fig_matrix.update_layout(
                    title=dict(
                        text=title,
                        font=dict(size=font_size, family=font_family),
                        x=0.5
                    ),
                    xaxis=dict(
                        title=dict(
                            text='Trap CV (V)',
                            font=dict(size=font_size, family=font_family)
                        ),
                        tickfont=dict(size=font_size, family=font_family),
                        showline=True,
                        linewidth=2,
                        linecolor='black',
                        mirror=True
                    ),
                    yaxis=dict(
                        title=dict(
                            text='Collision Cross Section (Å²)',
                            font=dict(size=font_size, family=font_family)
                        ),
                        tickfont=dict(size=font_size, family=font_family),
                        showline=True,
                        linewidth=2,
                        linecolor='black',
                        mirror=True
                    ),
                    width=figure_width,
                    height=figure_height,
                    font=dict(size=font_size, family=font_family),
                    plot_bgcolor='white',
                    margin=dict(l=80, r=150, t=100, b=80)
                )
                
                # Apply axis limits if specified
                if not auto_x_limits:
                    fig_matrix.update_xaxes(range=[x_min, x_max])
                if not auto_y_limits:
                    fig_matrix.update_yaxes(range=[y_min, y_max])
                
                # ============================================================
                # Crop data and recalculate RMSD if axis limits are applied
                # ============================================================
                
                # Store original values for RMSD plotting
                trap_cv_values_for_rmsd = trap_cv_values.copy()
                
                if (not auto_x_limits or not auto_y_limits) and replicate_mode and intensity_matrix_std is not None:
                    # Find indices for cropping
                    if not auto_x_limits:
                        cv_mask = (trap_cv_values >= x_min) & (trap_cv_values <= x_max)
                        cv_indices = np.where(cv_mask)[0]
                    else:
                        cv_indices = np.arange(len(trap_cv_values))
                    
                    if not auto_y_limits:
                        ccs_mask = (ccs_values >= y_min) & (ccs_values <= y_max)
                        ccs_indices = np.where(ccs_mask)[0]
                    else:
                        ccs_indices = np.arange(len(ccs_values))
                    
                    # Crop the data
                    if len(cv_indices) > 0 and len(ccs_indices) > 0:
                        # Create cropped index grids
                        ccs_grid_crop = np.ix_(ccs_indices, cv_indices)
                        
                        # Crop intensity matrices
                        intensity_matrix_final_crop = intensity_matrix_final[ccs_grid_crop]
                        intensity_matrix_std_crop = intensity_matrix_std[ccs_grid_crop]
                        
                        # Crop coordinate arrays for consistency
                        trap_cv_values_crop = trap_cv_values[cv_indices]
                        ccs_values_crop = ccs_values[ccs_indices]
                        
                        # Update main arrays with cropped versions for CIU50 analysis
                        intensity_matrix_final = intensity_matrix_final_crop
                        trap_cv_values = trap_cv_values_crop
                        ccs_values = ccs_values_crop
                        
                        # Crop stacked matrices if available
                        if stacked_matrices is not None:
                            stacked_matrices_crop = stacked_matrices[:, ccs_indices, :][:, :, cv_indices]
                            mean_crop = intensity_matrix_final_crop
                            
                            # Recalculate RMSD on cropped data
                            differences_squared_crop = (stacked_matrices_crop - mean_crop) ** 2
                            overall_rmsd_crop = np.sqrt(np.mean(differences_squared_crop))
                            
                            # Recalculate RMSD_CV on cropped data
                            n_voltages_crop = mean_crop.shape[1]
                            rmsd_cv_crop = np.zeros(n_voltages_crop)
                            for j in range(n_voltages_crop):
                                replicate_columns = stacked_matrices_crop[:, :, j]
                                mean_column = mean_crop[:, j]
                                diff_sq = (replicate_columns - mean_column) ** 2
                                rmsd_cv_crop[j] = np.sqrt(np.mean(diff_sq))
                            
                            # Update RMSD values with cropped versions
                            overall_rmsd = overall_rmsd_crop
                            rmsd_cv = rmsd_cv_crop
                            
                            # Update stacked matrices with cropped version for CIU50 analysis
                            stacked_matrices = stacked_matrices_crop
                            
                            # Update voltage values for RMSD_CV plotting
                            trap_cv_values_for_rmsd = trap_cv_values_crop
                            
                            st.info(f"🔍 RMSD recalculated on cropped data: CV range [{trap_cv_values_crop[0]:.1f}, {trap_cv_values_crop[-1]:.1f}], CCS range [{ccs_values_crop[0]:.1f}, {ccs_values_crop[-1]:.1f}]")
                
                # ============================================================
                # Display Results
                # ============================================================
                
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-header">📈 Result</h3>', unsafe_allow_html=True)
                
                st.plotly_chart(fig_matrix, use_container_width=True)
                
                # Display intensity stats for debugging when in replicate mode
                if replicate_mode:
                    intensity_min = np.min(intensity_matrix_final)
                    intensity_max = np.max(intensity_matrix_final)
                    intensity_mean = np.mean(intensity_matrix_final)
                    st.info(f"Mean Intensity range: {intensity_min:.4f} to {intensity_max:.4f} (mean: {intensity_mean:.4f})")
                
                # ============================================================
                # Display Standard Deviation (Replicate Mode)
                # ============================================================
                
                if replicate_mode and intensity_matrix_std is not None:
                    st.markdown("#### 📊 Standard Deviation Heatmap")
                    st.markdown("""
                    <div class="info-card">
                        <p>Standard deviation across replicates at each CCS-voltage point.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display std dev stats for debugging
                    std_min = np.min(intensity_matrix_std)
                    std_max = np.max(intensity_matrix_std)
                    std_mean = np.mean(intensity_matrix_std)
                    
                    st.info(f"Std Dev range: {std_min:.4f} to {std_max:.4f} (mean: {std_mean:.4f})")
                    
                    fig_std = go.Figure(data=go.Heatmap(
                        z=intensity_matrix_std,
                        x=trap_cv_values,
                        y=ccs_values,
                        colorscale='Reds',
                        colorbar=dict(
                            title=dict(
                                text='Std Dev',
                                font=dict(size=font_size, family=font_family)
                            ),
                            tickfont=dict(size=font_size, family=font_family)
                        ),
                        zmin=0,
                        zauto=True
                    ))
                    
                    fig_std.update_layout(
                        title=dict(
                            text=f'Standard Deviation Across Replicates (Z={selected_charge})',
                            font=dict(size=font_size, family=font_family),
                            x=0.5
                        ),
                        xaxis=dict(
                            title=dict(
                                text='Trap CV (V)',
                                font=dict(size=font_size, family=font_family)
                            ),
                            tickfont=dict(size=font_size, family=font_family),
                            showline=True,
                            linewidth=2,
                            linecolor='black',
                            mirror=True
                        ),
                        yaxis=dict(
                            title=dict(
                                text='Collision Cross Section (Å²)',
                                font=dict(size=font_size, family=font_family)
                            ),
                            tickfont=dict(size=font_size, family=font_family),
                            showline=True,
                            linewidth=2,
                            linecolor='black',
                            mirror=True
                        ),
                        width=figure_width,
                        height=figure_height,
                        font=dict(size=font_size, family=font_family),
                        plot_bgcolor='white',
                        margin=dict(l=80, r=150, t=100, b=80)
                    )
                    
                    if not auto_x_limits:
                        fig_std.update_xaxes(range=[x_min, x_max])
                    if not auto_y_limits:
                        fig_std.update_yaxes(range=[y_min, y_max])
                    
                    st.plotly_chart(fig_std, use_container_width=True)
                    
                    # Display RMSD metrics
                    if overall_rmsd is not None and rmsd_cv is not None:
                        st.markdown("#### 📈 RMSD Analysis (CIUSuite/ORIGAMI Style)")
                        st.markdown("""
                        <div class="info-card">
                            <p><strong>RMSD (Root Mean Square Deviation)</strong>: Quantifies differences between replicates.</p>
                            <ul>
                                <li><strong>Overall RMSD</strong>: Global difference indicator across entire fingerprint</li>
                                <li><strong>RMSD<sub>CV</sub></strong>: Local differences at each collision voltage (more sensitive to voltage-specific variations)</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Overall RMSD", f"{overall_rmsd:.4f}")
                            st.metric("Mean RMSD_CV", f"{np.mean(rmsd_cv):.4f}")
                            st.metric("Max RMSD_CV", f"{np.max(rmsd_cv):.4f}")
                        
                        with col2:
                            # Create RMSD_CV line plot
                            fig_rmsd_cv = go.Figure()
                            
                            fig_rmsd_cv.add_trace(go.Scatter(
                                x=trap_cv_values_for_rmsd,
                                y=rmsd_cv,
                                mode='lines+markers',
                                line=dict(color='#d62728', width=2),
                                marker=dict(size=6),
                                name='RMSD_CV'
                            ))
                            
                            fig_rmsd_cv.update_layout(
                                title=dict(
                                    text=f'RMSD<sub>CV</sub> vs Collision Voltage (Z={selected_charge})',
                                    font=dict(size=font_size, family=font_family),
                                    x=0.5
                                ),
                                xaxis=dict(
                                    title=dict(
                                        text='Trap CV (V)',
                                        font=dict(size=font_size, family=font_family)
                                    ),
                                    tickfont=dict(size=font_size, family=font_family),
                                    showline=True,
                                    linewidth=2,
                                    linecolor='black',
                                    mirror=True,
                                    showgrid=True,
                                    gridcolor='lightgray'
                                ),
                                yaxis=dict(
                                    title=dict(
                                        text='RMSD<sub>CV</sub>',
                                        font=dict(size=font_size, family=font_family)
                                    ),
                                    tickfont=dict(size=font_size, family=font_family),
                                    showline=True,
                                    linewidth=2,
                                    linecolor='black',
                                    mirror=True,
                                    showgrid=True,
                                    gridcolor='lightgray'
                                ),
                                width=figure_width,
                                height=int(figure_height * 0.6),
                                font=dict(size=font_size, family=font_family),
                                plot_bgcolor='white',
                                margin=dict(l=80, r=50, t=80, b=80),
                                showlegend=False
                            )
                            
                            if not auto_x_limits:
                                fig_rmsd_cv.update_xaxes(range=[x_min, x_max])
                            
                            st.plotly_chart(fig_rmsd_cv, use_container_width=True)
                        
                        st.info("💡 **Interpretation**: Higher RMSD values indicate greater variability between replicates at that voltage. Use RMSD_CV to identify voltage regions with consistent vs variable behavior.")
                
                # Display processing info
                st.markdown("""
                <div class="info-card">
                    <strong>✅ Figure generated successfully!</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"Matrix dimensions: {len(ccs_values)} CCS values × {len(trap_cv_values)} TrapCV values")
                
                processing_steps = []
                if normalize_cv:
                    processing_steps.append("CV slice normalisation")
                if interp_multiplier > 1:
                    processing_steps.append(f"{interp_multiplier}x {interp_method} interpolation")
                if apply_smoothing:
                    processing_steps.append(f"{smoothing_method} smoothing")
                
                if processing_steps:
                    st.info(f"Applied: {', '.join(processing_steps)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ============================================================
                # Download Options
                # ============================================================
                
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-header">📥 Download Options</h3>', unsafe_allow_html=True)
                
                # Create fingerprint data for download
                fingerprint_data = []
                for i, ccs in enumerate(ccs_values):
                    for j, trap_cv in enumerate(trap_cv_values):
                        intensity = intensity_matrix_final[i, j]
                        fingerprint_data.append({
                            'TrapCV': trap_cv,
                            'CCS': ccs,
                            'Intensity': intensity
                        })
                
                fingerprint_df = pd.DataFrame(fingerprint_data)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download CSV data
                    download_df = fingerprint_df[['TrapCV', 'CCS', 'Intensity']].copy()
                    download_df = download_df.sort_values(['TrapCV', 'CCS'])
                    
                    csv_buffer = io.StringIO()
                    download_df.to_csv(csv_buffer, index=False)
                    
                    # Create filename
                    filename_parts = [f"ccs_fingerprint_z{selected_charge}"]
                    if interp_multiplier > 1:
                        filename_parts.append(f"{interp_method}_{interp_multiplier}x")
                    if normalize_cv:
                        filename_parts.append("normalised")
                    if apply_smoothing:
                        filename_parts.append(f"{smoothing_method.lower()}_smooth")
                    
                    filename = "_".join(filename_parts) + ".csv"
                    
                    st.download_button(
                        label="📊 Download Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Download interactive HTML
                    html_buffer = io.StringIO()
                    fig_matrix.write_html(html_buffer)
                    
                    html_filename = filename.replace('.csv', '_interactive.html')
                    
                    st.download_button(
                        label="🌐 Download Interactive (HTML)",
                        data=html_buffer.getvalue(),
                        file_name=html_filename,
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col3:
                    # Download static PNG using matplotlib
                    # Set font before creating figure
                    plt.rcParams['font.family'] = font_family.lower()
                    plt.rcParams['font.size'] = font_size
                    
                    fig_static, ax = plt.subplots(
                        figsize=(figure_width_inches, figure_height_inches), 
                        dpi=figure_dpi
                    )
                    
                    # Convert colorscale for matplotlib
                    if use_custom_color:
                        hex_clean = hex_color.lstrip('#')
                        rgb_norm = tuple(int(hex_clean[i:i+2], 16)/255.0 for i in (0, 2, 4))
                        
                        if reverse_colors:
                            colors = [rgb_norm, (1.0, 1.0, 1.0)]
                        else:
                            colors = [(1.0, 1.0, 1.0), rgb_norm]
                        
                        cmap = LinearSegmentedColormap.from_list("custom", colors)
                    else:
                        colormap_name = color_scheme.lower()
                        cmap_dict = {
                            'viridis': plt.cm.viridis,
                            'plasma': plt.cm.plasma,
                            'inferno': plt.cm.inferno,
                            'magma': plt.cm.magma,
                            'cividis': plt.cm.cividis,
                            'blues': plt.cm.Blues,
                            'reds': plt.cm.Reds,
                            'greens': plt.cm.Greens,
                            'ylord': plt.cm.YlOrRd,
                            'ylgnbu': plt.cm.YlGnBu,
                            'rdylbu': plt.cm.RdYlBu,
                            'spectral': plt.cm.Spectral,
                            'coolwarm': plt.cm.coolwarm,
                            'jet': plt.cm.jet,
                            'hot': plt.cm.hot,
                            'cool': plt.cm.cool,
                        }
                        cmap = cmap_dict.get(colormap_name, plt.cm.viridis)
                        
                        if reverse_colors:
                            cmap = cmap.reversed()
                    
                    # Create coordinate grids for pcolormesh
                    if len(trap_cv_values) > 1:
                        trap_cv_spacing = (trap_cv_values[-1] - trap_cv_values[0]) / (len(trap_cv_values) - 1)
                        trap_cv_edges = np.linspace(
                            trap_cv_values[0] - trap_cv_spacing/2,
                            trap_cv_values[-1] + trap_cv_spacing/2,
                            len(trap_cv_values) + 1
                        )
                    else:
                        trap_cv_edges = np.array([trap_cv_values[0] - 0.5, trap_cv_values[0] + 0.5])
                    
                    if len(ccs_values) > 1:
                        ccs_spacing = (ccs_values[-1] - ccs_values[0]) / (len(ccs_values) - 1)
                        ccs_edges = np.linspace(
                            ccs_values[0] - ccs_spacing/2,
                            ccs_values[-1] + ccs_spacing/2,
                            len(ccs_values) + 1
                        )
                    else:
                        ccs_edges = np.array([ccs_values[0] - 0.5, ccs_values[0] + 0.5])
                    
                    # Use shading='nearest' to match plotly's cell-centered behavior
                    im = ax.pcolormesh(
                        trap_cv_values,
                        ccs_values,
                        intensity_matrix_final,
                        cmap=cmap,
                        shading='nearest'
                    )
                    
                    # Set axis limits if specified
                    if not auto_x_limits:
                        ax.set_xlim(x_min, x_max)
                    if not auto_y_limits:
                        ax.set_ylim(y_min, y_max)
                    
                    # Set labels and title
                    ax.set_xlabel('Trap CV (V)', fontsize=font_size, fontfamily=font_family.lower(), fontweight='normal')
                    ax.set_ylabel('Collision Cross Section (Å²)', fontsize=font_size, fontfamily=font_family.lower(), fontweight='normal')
                    ax.set_title(title, fontsize=font_size, fontfamily=font_family.lower(), fontweight='normal', pad=20)
                    
                    # Add black border
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(2)
                    
                    # Add colorbar if requested
                    if show_colorbar:
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label(colorbar_title, fontsize=font_size, fontfamily=font_family.lower(), fontweight='normal')
                        cbar.ax.tick_params(labelsize=font_size)
                        # Set colorbar tick label fonts
                        for label in cbar.ax.get_yticklabels():
                            label.set_fontfamily(font_family.lower())
                            label.set_fontsize(font_size)
                        cbar.outline.set_edgecolor('black')
                        cbar.outline.set_linewidth(2)
                    
                    # Set tick label fonts explicitly
                    ax.tick_params(axis='both', which='major', labelsize=font_size, colors='black')
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontfamily(font_family.lower())
                        label.set_fontsize(font_size)
                    
                    plt.tight_layout()
                    
                    # Save to buffer
                    png_buffer = io.BytesIO()
                    plt.savefig(png_buffer, format='png', dpi=figure_dpi, bbox_inches='tight')
                    plt.close(fig_static)
                    
                    png_filename = filename.replace('.csv', '_static.png')
                    
                    st.download_button(
                        label="🖼️ Download Static (PNG)",
                        data=png_buffer.getvalue(),
                        file_name=png_filename,
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Download RMSD data if available (replicate mode)
                if replicate_mode and rmsd_cv is not None and overall_rmsd is not None:
                    st.markdown("---")
                    st.markdown("**📊 RMSD Data (Replicate Analysis)**")
                    
                    col_rmsd1, col_rmsd2 = st.columns(2)
                    
                    with col_rmsd1:
                        # Create RMSD_CV dataframe
                        rmsd_cv_df = pd.DataFrame({
                            'TrapCV': trap_cv_values_for_rmsd,
                            'RMSD_CV': rmsd_cv
                        })
                        
                        rmsd_csv_buffer = io.StringIO()
                        rmsd_cv_df.to_csv(rmsd_csv_buffer, index=False)
                        
                        st.download_button(
                            label="📉 Download RMSD_CV Data (CSV)",
                            data=rmsd_csv_buffer.getvalue(),
                            file_name=f"rmsd_cv_z{selected_charge}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_rmsd2:
                        # Create summary file with overall RMSD
                        rmsd_summary = f"RMSD Analysis Summary (Z={selected_charge})\n\n"
                        rmsd_summary += f"Overall RMSD: {overall_rmsd:.6f}\n"
                        rmsd_summary += f"Mean RMSD_CV: {np.mean(rmsd_cv):.6f}\n"
                        rmsd_summary += f"Max RMSD_CV: {np.max(rmsd_cv):.6f}\n"
                        rmsd_summary += f"Min RMSD_CV: {np.min(rmsd_cv):.6f}\n"
                        rmsd_summary += f"Std RMSD_CV: {np.std(rmsd_cv):.6f}\n\n"
                        
                        # Add note about data range
                        if len(trap_cv_values_for_rmsd) < len(trap_cv_values) or not auto_y_limits:
                            rmsd_summary += f"Data Range (Cropped):\n"
                            rmsd_summary += f"  CV: [{trap_cv_values_for_rmsd[0]:.1f}, {trap_cv_values_for_rmsd[-1]:.1f}] V\n"
                            if not auto_y_limits:
                                rmsd_summary += f"  CCS: [{y_min:.1f}, {y_max:.1f}] Å²\n"
                            rmsd_summary += "\n"
                        
                        rmsd_summary += "Note: RMSD calculated from normalized replicates\n" if normalize_cv else "Note: RMSD calculated from raw intensities\n"
                        
                        st.download_button(
                            label="📋 Download RMSD Summary (TXT)",
                            data=rmsd_summary,
                            file_name=f"rmsd_summary_z{selected_charge}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store processed data in session state for CIU50 analysis
                st.session_state['processed_data'] = {
                    'ccs_values': ccs_values,
                    'trap_cv_values': trap_cv_values,
                    'intensity_matrix_final': intensity_matrix_final,
                    'selected_charge': selected_charge,
                    'font_size': font_size,
                    'font_family': font_family,
                    'figure_dpi': figure_dpi,
                    'figure_width_inches': figure_width_inches,
                    'figure_height_inches': figure_height_inches,
                    'replicate_mode': replicate_mode,
                    'stacked_matrices': stacked_matrices if replicate_mode else None
                }
                
    except Exception as e:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">❌ Error</h3>', unsafe_allow_html=True)
        st.error(f"Error processing files: {str(e)}")
        import traceback
        with st.expander("📋 Full error details"):
            st.code(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# CIU50 Analysis Section (Outside Button Block)
# ============================================================================

if ciu50_analysis and 'processed_data' in st.session_state:
    data = st.session_state['processed_data']
    ccs_values = data['ccs_values']
    trap_cv_values = data['trap_cv_values']
    intensity_matrix_final = data['intensity_matrix_final']
    selected_charge = data['selected_charge']
    font_size = data['font_size']
    font_family = data['font_family']
    figure_dpi = data['figure_dpi']
    figure_width_inches = data['figure_width_inches']
    figure_height_inches = data['figure_height_inches']
    replicate_mode = data.get('replicate_mode', False)
    stacked_matrices = data.get('stacked_matrices', None)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📊 CIU50 Analysis</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p>Extract the modal (peak) CCS at each collision voltage and identify conformational transitions.</p>
        <p><strong>How it works:</strong></p>
        <ul>
            <li>Finds the maximum intensity CCS at each voltage</li>
            <li>Plots a single curve showing how modal CCS changes with voltage</li>
            <li>Identifies transitions between conformers (e.g., Conformer 1 → 2 → 3)</li>
            <li>Fits sigmoid curves to each transition and calculates CIU₅₀ values</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # User inputs for conformer ranges
    st.markdown("#### Define Conformer CCS Ranges")
    st.markdown("Specify the CCS range for each conformer. The modal CCS curve will transition through these conformers as voltage increases:")
    
    # Initialize session state for conformer ranges if not exists
    if 'conformer_ranges_state' not in st.session_state:
        st.session_state.conformer_ranges_state = {}
    
    num_conformers = st.number_input(
        "Number of conformers to track",
        min_value=1,
        max_value=5,
        value=2,
        help="How many distinct conformers do you want to analyze?",
        key="num_conformers_ciu50"
    )
    
    conformer_ranges = []
    conformer_labels = []
    
    for i in range(num_conformers):
        st.markdown(f"**Conformer {i+1}:**")
        col1, col2, col3 = st.columns(3)
        
        # Initialize default values
        default_label = st.session_state.conformer_ranges_state.get(f'label_{i}', f"Conformer {i+1}")
        default_min = st.session_state.conformer_ranges_state.get(f'min_{i}', float(np.min(ccs_values)))
        default_max = st.session_state.conformer_ranges_state.get(f'max_{i}', float(np.max(ccs_values)))
        
        with col1:
            label = st.text_input(
                f"Label",
                value=default_label,
                key=f"conformer_label_{i}_input"
            )
            st.session_state.conformer_ranges_state[f'label_{i}'] = label
            conformer_labels.append(label)
        
        with col2:
            ccs_min = st.number_input(
                f"CCS Min (Ų)",
                value=default_min,
                key=f"conformer_min_{i}_input",
                step=1.0
            )
            st.session_state.conformer_ranges_state[f'min_{i}'] = ccs_min
        
        with col3:
            ccs_max = st.number_input(
                f"CCS Max (Ų)",
                value=default_max,
                key=f"conformer_max_{i}_input",
                step=1.0
            )
            st.session_state.conformer_ranges_state[f'max_{i}'] = ccs_max
        
        conformer_ranges.append((ccs_min, ccs_max))
    
    if st.button("🎯 Generate CIU50 Plot", key="ciu50_button"):
        with st.spinner("Performing CIU50 analysis..."):
            try:
                # Define function to analyze one matrix
                def analyze_single_matrix(intensity_matrix, ccs_vals, trap_cv_vals, matrix_name=""):
                    """Perform CIU50 analysis on a single intensity matrix"""
                    # Extract modal CCS at each voltage
                    modal_ccs_values = []
                    modal_ccs_indices = []
                    
                    st.write(f"🔍 {matrix_name} - Matrix shape: {intensity_matrix.shape}")
                    st.write(f"🔍 {matrix_name} - CCS vals shape: {ccs_vals.shape}, range: [{ccs_vals.min():.1f}, {ccs_vals.max():.1f}]")
                    st.write(f"🔍 {matrix_name} - Trap CV vals shape: {trap_cv_vals.shape}, range: [{trap_cv_vals.min():.1f}, {trap_cv_vals.max():.1f}]")
                    
                    for j in range(len(trap_cv_vals)):
                        intensity_slice = intensity_matrix[:, j]
                        max_idx = np.argmax(intensity_slice)
                        modal_ccs_values.append(ccs_vals[max_idx])
                        modal_ccs_indices.append(max_idx)
                        
                        if j < 3 or j >= len(trap_cv_vals) - 3:
                            st.write(f"🔍 {matrix_name} - Voltage {j} ({trap_cv_vals[j]:.2f}V): max_idx={max_idx}, modal_CCS={ccs_vals[max_idx]:.1f}")
                    
                    modal_ccs_values = np.array(modal_ccs_values)
                    st.write(f"🔍 {matrix_name} - Modal CCS range: [{modal_ccs_values.min():.1f}, {modal_ccs_values.max():.1f}]")
                    
                    # Calculate modal CCS values for each conformer
                    conformer_ccs_results = {}
                    for i, ((ccs_min, ccs_max), label) in enumerate(zip(sorted_ranges, sorted_labels)):
                        in_conformer = (modal_ccs_values >= ccs_min) & (modal_ccs_values <= ccs_max)
                        conformer_modal_ccs = modal_ccs_values[in_conformer]
                        
                        if len(conformer_modal_ccs) > 0:
                            conformer_ccs_results[label] = {
                                'mean': np.mean(conformer_modal_ccs),
                                'min': np.min(conformer_modal_ccs),
                                'max': np.max(conformer_modal_ccs)
                            }
                    
                    # Calculate CIU50 for each transition
                    transition_results = {}
                    for i in range(len(sorted_ranges) - 1):
                        conformer1_label = sorted_labels[i]
                        conformer2_label = sorted_labels[i + 1]
                        ccs_min1, ccs_max1 = sorted_ranges[i]
                        ccs_min2, ccs_max2 = sorted_ranges[i + 1]
                        
                        in_conformer1 = (modal_ccs_values >= ccs_min1) & (modal_ccs_values <= ccs_max1)
                        in_conformer2 = (modal_ccs_values >= ccs_min2) & (modal_ccs_values <= ccs_max2)
                        transition_mask = in_conformer1 | in_conformer2
                        
                        transition_voltages = trap_cv_vals[transition_mask]
                        transition_ccs = modal_ccs_values[transition_mask]
                        
                        if len(transition_voltages) >= 4:
                            ccs_change = np.max(transition_ccs) - np.min(transition_ccs)
                            
                            if ccs_change > 10:
                                try:
                                    from scipy.optimize import curve_fit
                                    
                                    initial_ccs = transition_ccs[0]
                                    final_ccs = transition_ccs[-1]
                                    L_guess = abs(final_ccs - initial_ccs)
                                    x0_guess = np.mean(transition_voltages)
                                    k_guess = 0.1
                                    b_guess = min(initial_ccs, final_ccs)
                                    
                                    popt, _ = curve_fit(
                                        sigmoid,
                                        transition_voltages,
                                        transition_ccs,
                                        p0=[L_guess, x0_guess, k_guess, b_guess],
                                        maxfev=10000
                                    )
                                    
                                    ciu50 = popt[1]
                                    
                                    residuals = transition_ccs - sigmoid(transition_voltages, *popt)
                                    ss_res = np.sum(residuals**2)
                                    ss_tot = np.sum((transition_ccs - np.mean(transition_ccs))**2)
                                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                                    
                                    transition_key = f'{conformer1_label}→{conformer2_label}'
                                    transition_results[transition_key] = {
                                        'ciu50': ciu50,
                                        'r_squared': r_squared,
                                        'points': len(transition_voltages),
                                        'fit_params': popt
                                    }
                                    
                                except Exception:
                                    pass
                    
                    return modal_ccs_values, conformer_ccs_results, transition_results
                
                # Sort conformer ranges by CCS (ascending order)
                sorted_indices = np.argsort([r[0] for r in conformer_ranges])
                sorted_ranges = [conformer_ranges[i] for i in sorted_indices]
                sorted_labels = [conformer_labels[i] for i in sorted_indices]
                
                # Perform analysis
                if replicate_mode and stacked_matrices is not None:
                    # Analyze each replicate
                    n_replicates = stacked_matrices.shape[0]
                    st.info(f"📊 Analyzing {n_replicates} replicates individually...")
                    st.info(f"📐 Data range: CV [{trap_cv_values.min():.1f}, {trap_cv_values.max():.1f}] V, CCS [{ccs_values.min():.1f}, {ccs_values.max():.1f}] Ų")
                    
                    all_conformer_results = []
                    all_transition_results = []
                    replicate_modal_ccs = []
                    
                    for rep_idx in range(n_replicates):
                        modal_ccs, conformer_ccs, transition_data = analyze_single_matrix(
                            stacked_matrices[rep_idx], ccs_values, trap_cv_values, f"Replicate {rep_idx + 1}"
                        )
                        replicate_modal_ccs.append(modal_ccs)
                        all_conformer_results.append(conformer_ccs)
                        all_transition_results.append(transition_data)
                    
                    # Verify dimensions
                    st.write(f"🔍 Debug: len(trap_cv_values)={len(trap_cv_values)}, len(replicate_modal_ccs[0])={len(replicate_modal_ccs[0])}, len(ccs_values)={len(ccs_values)}, stacked_matrices.shape={stacked_matrices.shape}")
                    st.write(f"🔍 Debug: trap_cv_values range: [{trap_cv_values[0]:.2f}, {trap_cv_values[-1]:.2f}]")
                    st.write(f"🔍 Debug: ccs_values range: [{ccs_values[0]:.2f}, {ccs_values[-1]:.2f}]")
                    st.write(f"🔍 Debug: Rep1 modal CCS first 5: {replicate_modal_ccs[0][:5]}")
                    st.write(f"🔍 Debug: Rep1 modal CCS last 5: {replicate_modal_ccs[0][-5:]}")
                    st.write(f"🔍 Debug: Corresponding voltages first 5: {trap_cv_values[:5]}")
                    st.write(f"🔍 Debug: Corresponding voltages last 5: {trap_cv_values[-5:]}")
                    
                    # Calculate statistics across replicates
                    # Conformer CCS statistics with individual values
                    conformer_ccs_stats = []
                    for label in sorted_labels:
                        means = [r[label]['mean'] for r in all_conformer_results if label in r]
                        if len(means) > 0:
                            row_data = {'Conformer': label}
                            for rep_idx, result in enumerate(all_conformer_results):
                                if label in result:
                                    row_data[f'Rep{rep_idx+1} CCS (Ų)'] = result[label]['mean']
                                else:
                                    row_data[f'Rep{rep_idx+1} CCS (Ų)'] = np.nan
                            row_data['Mean CCS (Ų)'] = np.mean(means)
                            row_data['Std CCS (Ų)'] = np.std(means, ddof=1) if len(means) > 1 else 0
                            conformer_ccs_stats.append(row_data)
                    
                    # Transition CIU50 statistics with individual values
                    ciu50_stats = []
                    all_transitions = set()
                    for tr in all_transition_results:
                        all_transitions.update(tr.keys())
                    
                    for transition_key in sorted(all_transitions):
                        ciu50_values = []
                        r2_values = []
                        row_data = {'Transition': transition_key.replace('→', ' → ')}
                        
                        for rep_idx, result in enumerate(all_transition_results):
                            if transition_key in result:
                                ciu50_val = result[transition_key]['ciu50']
                                r2_val = result[transition_key]['r_squared']
                                ciu50_values.append(ciu50_val)
                                r2_values.append(r2_val)
                                row_data[f'Rep{rep_idx+1} CIU50 (V)'] = ciu50_val
                                row_data[f'Rep{rep_idx+1} R²'] = r2_val
                            else:
                                row_data[f'Rep{rep_idx+1} CIU50 (V)'] = np.nan
                                row_data[f'Rep{rep_idx+1} R²'] = np.nan
                        
                        if len(ciu50_values) > 0:
                            row_data['Mean CIU50 (V)'] = np.mean(ciu50_values)
                            row_data['Std CIU50 (V)'] = np.std(ciu50_values, ddof=1) if len(ciu50_values) > 1 else 0
                            row_data['Mean R²'] = np.mean(r2_values)
                            row_data['N'] = len(ciu50_values)
                            ciu50_stats.append(row_data)
                    
                    # Use mean modal CCS for plotting
                    modal_ccs_values = np.mean(replicate_modal_ccs, axis=0)
                    
                else:
                    # Single analysis on averaged matrix
                    modal_ccs_values, conformer_ccs_results, transition_results = analyze_single_matrix(
                        intensity_matrix_final, ccs_values, trap_cv_values, "Average"
                    )
                    
                    # Format as single-replicate stats
                    conformer_ccs_stats = []
                    for label in sorted_labels:
                        if label in conformer_ccs_results:
                            conformer_ccs_stats.append({
                                'Conformer': label,
                                'Modal CCS (Ų)': conformer_ccs_results[label]['mean']
                            })
                    
                    ciu50_stats = []
                    for transition_key, data in transition_results.items():
                        ciu50_stats.append({
                            'Transition': transition_key.replace('→', ' → '),
                            'CIU50 (V)': data['ciu50'],
                            'R²': data['r_squared'],
                            'Points': data['points']
                        })
                
                # Set font before creating figures
                plt.rcParams['font.family'] = font_family
                plt.rcParams['font.size'] = font_size
                
                # Create plots for each replicate or single plot
                if replicate_mode and stacked_matrices is not None:
                    st.markdown("#### Individual Replicate CIU50 Plots")
                    
                    # Create a figure for each replicate
                    for rep_idx in range(n_replicates):
                        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches), dpi=figure_dpi)
                        
                        # Get modal CCS for this replicate
                        modal_ccs_rep = replicate_modal_ccs[rep_idx]
                        
                        # Plot modal CCS vs voltage
                        ax.plot(trap_cv_values, modal_ccs_rep, 'o-', color='black', 
                               markersize=6, linewidth=2, label=f'Modal CCS (Rep {rep_idx+1})', zorder=3)
                        
                        # Add colored backgrounds for conformer regions
                        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_ranges)))
                        for i, ((ccs_min, ccs_max), label, color) in enumerate(zip(sorted_ranges, sorted_labels, colors)):
                            ax.axhspan(ccs_min, ccs_max, alpha=0.2, color=color, 
                                      label=f'{label} region', zorder=1)
                        
                        # Plot fitted transitions for this replicate
                        ref_transitions = all_transition_results[rep_idx]
                        for transition_key, data in ref_transitions.items():
                            conformer1_label, conformer2_label = transition_key.split('→')
                            ccs_min1, ccs_max1 = sorted_ranges[sorted_labels.index(conformer1_label)]
                            ccs_min2, ccs_max2 = sorted_ranges[sorted_labels.index(conformer2_label)]
                            
                            in_conformer1 = (modal_ccs_rep >= ccs_min1) & (modal_ccs_rep <= ccs_max1)
                            in_conformer2 = (modal_ccs_rep >= ccs_min2) & (modal_ccs_rep <= ccs_max2)
                            transition_mask = in_conformer1 | in_conformer2
                            
                            transition_voltages = trap_cv_values[transition_mask]
                            
                            if len(transition_voltages) >= 4:
                                popt = data['fit_params']
                                ciu50 = data['ciu50']
                                r_squared = data['r_squared']
                                
                                v_fine = np.linspace(transition_voltages.min(), transition_voltages.max(), 200)
                                ccs_fitted = sigmoid(v_fine, *popt)
                                
                                ax.plot(v_fine, ccs_fitted, '--', color='red', linewidth=2, alpha=0.7, zorder=2)
                                
                                ccs_at_ciu50 = sigmoid(ciu50, *popt)
                                ax.axvline(ciu50, color='red', linestyle=':', alpha=0.5, linewidth=2, zorder=2)
                                ax.annotate(
                                    f'{conformer1_label}→{conformer2_label}\nCIU₅₀ = {ciu50:.1f}V\nR² = {r_squared:.3f}',
                                    xy=(ciu50, ccs_at_ciu50),
                                    xytext=(15, 0),
                                    textcoords='offset points',
                                    fontsize=font_size-1,
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
                                )
                        
                        # Formatting
                        ax.set_xlabel('Collision Voltage (V)', fontsize=font_size+2, 
                                     fontfamily=font_family, fontweight='bold')
                        ax.set_ylabel('Modal CCS (Ų)', fontsize=font_size+2, 
                                     fontfamily=font_family, fontweight='bold')
                        ax.set_title(f'CIU50 Analysis: Replicate {rep_idx+1} (Z={selected_charge})', 
                                    fontsize=font_size+3, fontfamily=font_family, 
                                    fontweight='bold', pad=15)
                        
                        # Let matplotlib auto-scale axes based on actual data
                        # Don't force axis limits to full data range
                        
                        # Set tick label fonts explicitly
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontfamily(font_family)
                            label.set_fontsize(font_size)
                        
                        ax.legend(loc='best', fontsize=font_size-1, framealpha=0.9, prop={'family': font_family, 'size': font_size-1})
                        ax.grid(True, alpha=0.3, linestyle='--')
                        
                        for spine in ax.spines.values():
                            spine.set_linewidth(1.5)
                            spine.set_edgecolor('black')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                else:
                    # Single plot for non-replicate mode
                    fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches), dpi=figure_dpi)
                    
                    # Plot modal CCS vs voltage  
                    ax.plot(trap_cv_values, modal_ccs_values, 'o-', color='black', 
                           markersize=6, linewidth=2, label='Modal CCS', zorder=3)
                    
                    # Add colored backgrounds for conformer regions
                    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_ranges)))
                    for i, ((ccs_min, ccs_max), label, color) in enumerate(zip(sorted_ranges, sorted_labels, colors)):
                        ax.axhspan(ccs_min, ccs_max, alpha=0.2, color=color, 
                                  label=f'{label} region', zorder=1)
                    
                    # Plot fitted transitions using stored parameters
                    ref_transitions = transition_results
                    for transition_key, data in ref_transitions.items():
                        conformer1_label, conformer2_label = transition_key.split('→')
                        ccs_min1, ccs_max1 = sorted_ranges[sorted_labels.index(conformer1_label)]
                        ccs_min2, ccs_max2 = sorted_ranges[sorted_labels.index(conformer2_label)]
                        
                        in_conformer1 = (modal_ccs_values >= ccs_min1) & (modal_ccs_values <= ccs_max1)
                        in_conformer2 = (modal_ccs_values >= ccs_min2) & (modal_ccs_values <= ccs_max2)
                        transition_mask = in_conformer1 | in_conformer2
                        
                        transition_voltages = trap_cv_values[transition_mask]
                        
                        if len(transition_voltages) >= 4:
                            popt = data['fit_params']
                            ciu50 = data['ciu50']
                            r_squared = data['r_squared']
                            
                            v_fine = np.linspace(transition_voltages.min(), transition_voltages.max(), 200)
                            ccs_fitted = sigmoid(v_fine, *popt)
                            
                            ax.plot(v_fine, ccs_fitted, '--', color='red', linewidth=2, alpha=0.7, zorder=2)
                            
                            ccs_at_ciu50 = sigmoid(ciu50, *popt)
                            ax.axvline(ciu50, color='red', linestyle=':', alpha=0.5, linewidth=2, zorder=2)
                            ax.annotate(
                                f'{conformer1_label}→{conformer2_label}\nCIU₅₀ = {ciu50:.1f}V\nR² = {r_squared:.3f}',
                                xy=(ciu50, ccs_at_ciu50),
                                xytext=(15, 0),
                                textcoords='offset points',
                                fontsize=font_size-1,
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
                            )
                    
                    # Formatting
                    ax.set_xlabel('Collision Voltage (V)', fontsize=font_size+2, 
                                 fontfamily=font_family, fontweight='bold')
                    ax.set_ylabel('Modal CCS (Ų)', fontsize=font_size+2, 
                                 fontfamily=font_family, fontweight='bold')
                    ax.set_title(f'CIU50 Analysis: Conformational Transitions (Z={selected_charge})', 
                                fontsize=font_size+3, fontfamily=font_family, 
                                fontweight='bold', pad=15)
                    
                    # Set tick label fonts explicitly
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontfamily(font_family)
                        label.set_fontsize(font_size)
                    
                    ax.legend(loc='best', fontsize=font_size-1, framealpha=0.9, prop={'family': font_family, 'size': font_size-1})
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_edgecolor('black')
                    
                    plt.tight_layout()
                    st.markdown("#### CIU50 Analysis Results")
                    st.pyplot(fig)
                
                # Display combined summary table
                st.markdown("#### CIU50 Analysis Summary")
                
                # Conformer CCS table with statistics
                if conformer_ccs_stats:
                    st.markdown("**Conformer Modal CCS Values:**")
                    conformer_df = pd.DataFrame(conformer_ccs_stats)
                    
                    # Format numeric columns
                    for col in conformer_df.columns:
                        if 'CCS' in col and col != 'Conformer':
                            conformer_df[col] = conformer_df[col].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A")
                    
                    st.table(conformer_df)
                
                # CIU50 transition results with statistics
                if ciu50_stats:
                    st.markdown("**CIU50 Transition Values:**")
                    results_df = pd.DataFrame(ciu50_stats)
                    
                    # Format numeric columns
                    for col in results_df.columns:
                        if 'CIU50' in col:
                            results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                        elif 'R²' in col:
                            results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                    
                    st.table(results_df)
                else:
                    st.warning("No transitions detected. Try adjusting conformer CCS ranges or check if data shows conformational changes.")
                
                # Download options for CIU50 data
                st.markdown("#### Download CIU50 Data:")
                
                if replicate_mode and stacked_matrices is not None:
                    # Download modal CCS data for all replicates
                    modal_ccs_dict = {'Voltage': trap_cv_values}
                    for rep_idx in range(n_replicates):
                        modal_ccs_dict[f'Rep{rep_idx+1}_Modal_CCS'] = replicate_modal_ccs[rep_idx]
                    modal_ccs_dict['Mean_Modal_CCS'] = modal_ccs_values
                    
                    modal_ccs_df = pd.DataFrame(modal_ccs_dict)
                    
                    csv_buffer_ciu50 = io.StringIO()
                    modal_ccs_df.to_csv(csv_buffer_ciu50, index=False)
                    
                    st.download_button(
                        label="📊 Download All Modal CCS Data (CSV)",
                        data=csv_buffer_ciu50.getvalue(),
                        file_name=f"modal_ccs_all_replicates_z{selected_charge}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download modal CCS data
                        modal_ccs_df = pd.DataFrame({
                            'Voltage': trap_cv_values,
                            'Modal_CCS': modal_ccs_values
                        })
                        
                        csv_buffer_ciu50 = io.StringIO()
                        modal_ccs_df.to_csv(csv_buffer_ciu50, index=False)
                        
                        st.download_button(
                            label="📊 Download Modal CCS (CSV)",
                            data=csv_buffer_ciu50.getvalue(),
                            file_name=f"modal_ccs_z{selected_charge}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download CIU50 plot
                        png_buffer_ciu50 = io.BytesIO()
                        fig.savefig(png_buffer_ciu50, format='png', dpi=figure_dpi, bbox_inches='tight')
                        
                        st.download_button(
                            label="🖼️ Download CIU50 Plot (PNG)",
                            data=png_buffer_ciu50.getvalue(),
                            file_name=f"ciu50_analysis_z{selected_charge}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                st.success("✅ CIU50 analysis complete!")
                
            except Exception as e:
                st.error(f"CIU50 analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-card">
        <strong>ℹ️ Getting Started:</strong> Please upload both calibration and TWIMExtract files to proceed with analysis.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Help Section
# ============================================================================

with st.expander("❓Help"):
    st.markdown("""
    ### Settings Management:
    
    **Save Settings:**
    - Click "Save Current Settings" to prepare your settings for download
    - Download the JSON file to save all your current parameter choices
    - Settings include: data processing options, smoothing parameters, color schemes, figure dimensions, axis limits, etc.
    
    **Load Settings:**
    - Upload a previously saved settings JSON file
    - All parameters will be restored to their saved values
    - The page will refresh automatically with your loaded settings
    - Use "Clear Settings File" button to remove the uploaded file and stop refreshing
    
    ### New Features:
    
    **Replicate Averaging Mode:**
    - Upload multiple replicate files to compute mean ± standard deviation
    - **Improved Interface:** Specify number of replicates, then upload calibration + TWIMExtract file for each replicate in dedicated sections
    - **File Pairing:** Each replicate has its own calibration file (accounts for daily calibration differences)
    - Files are matched by replicate number (Replicate 1 calibration → Replicate 1 TWIM data, etc.)
    - All replicates are CCS-converted, interpolated to a common grid
    - **IMPORTANT (ORIGAMI standard):** If CV normalization is enabled, each replicate is normalized **before** averaging
    - This means std represents variability in normalized intensities (0-1 scale), not raw intensities
    - **RMSD Analysis (CIUSuite/ORIGAMI style):**
      - **Overall RMSD**: Single value quantifying global difference across entire fingerprint
      - **RMSD_CV**: RMSD at each collision voltage, showing local variability (more sensitive to voltage-specific differences)
      - RMSD_CV plot helps identify voltage regions with consistent vs variable unfolding behavior
      - Useful for day-to-day comparisons, batch-to-batch analysis, and detecting subtle conformational changes
    - **For ORIGAMI format:** Each file's voltages are extracted independently from its range filenames
    - **Common voltage grid:** All unique voltages across replicates are combined into a single grid
    - **Interpolation:** Each replicate is interpolated to match the common voltage and CCS grids
    - Displays mean intensity heatmap, standard deviation heatmap, and RMSD_CV plot
    - Useful for assessing measurement reproducibility across different experiments
    - Requires at least 2 replicate files
    - **Important:** For ORIGAMI replicates with different voltage ranges, all data is interpolated to cover the full range
    
    **CIU50 Analysis:**
    - Extracts the modal (peak) CCS at each collision voltage to create a single transition curve
    - Shows how the dominant CCS shifts as voltage increases (e.g., Conformer 1 → 2 → 3)
    - Define CCS ranges to identify which conformer is present at different stages
    - Automatically fits sigmoidal curves to transitions between conformers
    - Calculates CIU₅₀ values for each transition (voltage where 50% change occurs)
    - Example: For 3 conformers, calculates CIU₅₀ for 1→2 and 2→3 transitions
    - Displays R² values for fit quality assessment
    - Export modal CCS data and transition plots for publication
    
    ### File Formats Expected:
    
    **Calibration File:**
    - CSV format with columns: Z, Drift, CCS, CCS Std.Dev.
    - Drift times should be in **seconds**
    - Multiple charge states supported
    - Rows with error ≥ CCS value are automatically removed
    - **In replicate mode:** Each replicate gets its own calibration file uploaded in dedicated sections
      - This accounts for different calibrations on different days
      - Files are automatically paired by replicate number (no manual ordering needed)
    
    **TWIMExtract File:**
    
    **Standard TWIMExtract Format:**
    - First two rows: Range file names and Raw file names (ignored)
    - Third row: $TrapCV: followed by TrapCV values
    - Subsequent rows: drift_time,intensity1,intensity2,intensity3...
    - Drift times are in **milliseconds**
    - Should have ~200 drift time points per TrapCV value
    
    **ORIGAMI Format:**
    - First row: Range file names (e.g., 20V.txt, 22V.txt, 25V.txt) - voltages extracted from these filenames
    - Second row onwards: drift_time,intensity1,intensity2,intensity3...
    - Drift times are in **milliseconds**
    - Each column corresponds to a collision voltage extracted from the range file name
    
    ### Processing Workflow:
    1. **File Upload**: 
       - **Single file mode:** Upload one calibration + one TWIMExtract file
       - **Replicate mode:** Specify number of replicates, then upload calibration + TWIMExtract for each replicate in dedicated sections
       - Files are automatically paired by replicate number
    2. **Replicate Processing**: (If enabled) Load and interpolate all replicates to common grid
    3. **Charge State Selection**: Choose charge state for analysis
    4. **CCS Conversion**: Convert drift times to CCS values using calibration (each replicate uses its own calibration)
    5. **Data Sorting**: Sort CCS values in ascending order
    6. **Duplicate Removal**: Remove duplicate CCS or TrapCV values
    7. **CV Slice Normalisation**: (If enabled in replicate mode) Normalise each replicate's CV slices individually BEFORE averaging
    8. **Replicate Averaging**: (If enabled) Compute mean and std across (normalized) replicates
    9. **CV Slice Normalisation**: (If enabled in single-file mode) Normalise each CV slice to maximum intensity of 1
    10. **2D Interpolation**: (If enabled) Add interpolated datapoints for smoother visualization
    11. **Smoothing**: (If enabled) Apply Gaussian or Savitzky-Golay smoothing
    12. **CIU50 Analysis**: (If enabled) Extract modal CCS, fit sigmoids, and calculate transition voltages
    
    ### CIU50 Analysis Tips:
    - Define CCS ranges that represent each conformer state
    - Order conformers by CCS: lowest to highest (e.g., Compact → Intermediate → Extended)
    - The algorithm finds the global maximum CCS at each voltage to create the transition curve
    - CIU₅₀ is calculated for transitions between adjacent conformers (e.g., 1→2, 2→3)
    - Sigmoid fits require at least 4 data points in the transition region and significant CCS change (>10 Ų)
    - Check R² values to assess fit quality (closer to 1.0 is better)
    - If no transitions are detected, check that the modal CCS curve actually crosses through your defined conformer ranges
    """)
