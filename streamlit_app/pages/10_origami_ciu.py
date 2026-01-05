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
                <li><strong>2D Interpolation:</strong> Increase data point density with linear or cubic interpolation</li>
                <li><strong>Smoothing:</strong> Gaussian or Savitzky-Golay smoothing</li>
                <li><strong>Normalisation:</strong> Normalise each collision voltage slice</li>
                <li><strong>Figure Generation:</strong> Both static <code>matplotlib</code> graphs and interactive <code>plotly</code> graphs.</li>
                <li><strong>Future Versions:</strong> Advanced ORIGAMI functionality</li>
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
    def show_file_upload() -> Tuple[Optional[Any], Optional[Any], str]:
        """Show file upload widgets for calibration and TWIMExtract files.
        
        Returns:
            Tuple of (calibration_file, twim_file, data_mode)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📁 Upload Files</h3>', unsafe_allow_html=True)
        
        # Data mode selection
        data_mode = st.radio(
            "Data Acquisition Mode:",
            ["Standard TWIMExtract", "ORIGAMI"],
            help="Standard: Voltages in $TrapCV: row. ORIGAMI: Voltages extracted from range file names in first row."
        )
        
        col1, col2 = st.columns(2)

        with col1:
            calibration_file = st.file_uploader(
                "Upload Calibration File",
                type=['csv', 'txt'],
                help="CSV file with columns: Z, Drift, CCS, CCS Std.Dev."
            )

        with col2:
            twim_file = st.file_uploader(
                "Upload TWIMExtract File",
                type=['csv', 'txt'],
                help="CSV file from TWIMExtract with drift times and intensities"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return calibration_file, twim_file, data_mode


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
    calibration_file, twim_file, data_mode = ORIGAMIInterface.show_file_upload()
    
    # Return uploaded files
    return calibration_file, twim_file, data_mode


# Run main application
calibration_file, twim_file, data_mode = main()


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
    range_files = first_row.split(',')[1:]  # Skip first column (which is empty or header)
    
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
    
    if not trap_cv_values:
        st.error("Could not extract any voltages from range file row")
        return None, None, None
    
    # Data starts from second row
    data_start_idx = 1
    
    # Remove duplicates
    original_trap_cv_count = len(trap_cv_values)
    trap_cv_values_clean, removed_trapcv_indices = remove_duplicate_values(trap_cv_values)
    
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


if calibration_file and twim_file:
    try:
        # Read calibration file
        cal_df = pd.read_csv(calibration_file)
        
        # Remove rows where error is larger than CCS value
        initial_rows = len(cal_df)
        cal_df = cal_df[cal_df['CCS Std.Dev.'] <= cal_df['CCS']]
        removed_rows = initial_rows - len(cal_df)
        
        st.success(f"Calibration file loaded: {len(cal_df)} calibration points")
        if removed_rows > 0:
            st.warning(f"Removed {removed_rows} points where error ≥ CCS value")
        
        # Read TWIMExtract file
        twim_content = twim_file.read().decode('utf-8').split('\n')
        
        # Parse data based on mode
        if data_mode == "ORIGAMI":
            # ORIGAMI format: voltages in first row (range file names)
            trap_cv_values, data_start_idx, removed_trapcv_indices = parse_origami_format(twim_content)
        else:
            # Standard format: voltages in $TrapCV: row
            trap_cv_values, data_start_idx, removed_trapcv_indices = parse_standard_format(twim_content)
        
        if trap_cv_values is None or data_start_idx is None:
            st.stop()
        
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
        
        # Create DataFrame
        columns = ['Drift_Time'] + [f'TrapCV_{cv}' for cv in trap_cv_values]
        twim_df = pd.DataFrame(data_rows, columns=columns)
        
        # Ensure all columns are numeric
        for col in twim_df.columns:
            if col != 'Drift_Time':
                twim_df[col] = twim_df[col].apply(safe_float_conversion)
        
        st.success(f"TWIMExtract file loaded: {len(twim_df)} drift time points, {len(trap_cv_values)} TrapCV values")
        if invalid_rows > 0:
            st.info(f"Skipped {invalid_rows} invalid rows during parsing")
        
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
                
                # Store dimensions for reporting
                original_ccs_points = len(ccs_values)
                original_trapcv_points = len(trap_cv_values)
                
                # Apply CV normalization if requested
                if normalize_cv:
                    for j in range(intensity_matrix_original.shape[1]):
                        col_max = np.max(intensity_matrix_original[:, j])
                        if col_max > 0:
                            intensity_matrix_original[:, j] = intensity_matrix_original[:, j] / col_max
                    st.info("Applied CV slice normalisation")
                
                # Apply interpolation if requested
                if interp_multiplier > 1:
                    try:
                        ccs_values, trap_cv_values, intensity_matrix_final = interpolate_matrix(
                            ccs_values,
                            trap_cv_values,
                            intensity_matrix_original,
                            method=interp_method,
                            multiplier=interp_multiplier,
                        )
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
                        st.info(f"Applied Gaussian smoothing (σ={gaussian_sigma}, truncate={gaussian_truncate})")
                    elif smoothing_method == "Savitzky-Golay":
                        intensity_matrix_final = smooth_matrix_savgol(
                            intensity_matrix_final,
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
                    showscale=show_colorbar
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
                # Display Results
                # ============================================================
                
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-header">📈 Result</h3>', unsafe_allow_html=True)
                
                st.plotly_chart(fig_matrix, use_container_width=True)
                
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
                    
                    X_edges, Y_edges = np.meshgrid(trap_cv_edges, ccs_edges)
                    
                    im = ax.pcolormesh(
                        X_edges,
                        Y_edges,
                        intensity_matrix_final,
                        cmap=cmap,
                        shading='flat'
                    )
                    
                    # Set axis limits if specified
                    if not auto_x_limits:
                        ax.set_xlim(x_min, x_max)
                    if not auto_y_limits:
                        ax.set_ylim(y_min, y_max)
                    
                    # Set labels and title
                    ax.set_xlabel('Trap CV (V)', fontsize=font_size, fontfamily=font_family.lower())
                    ax.set_ylabel('Collision Cross Section (Å²)', fontsize=font_size, fontfamily=font_family.lower())
                    ax.set_title(title, fontsize=font_size, fontfamily=font_family.lower(), pad=20)
                    
                    # Add black border
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(2)
                    
                    # Add colorbar if requested
                    if show_colorbar:
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label(colorbar_title, fontsize=font_size, fontfamily=font_family.lower())
                        cbar.ax.tick_params(labelsize=font_size)
                        cbar.outline.set_edgecolor('black')
                        cbar.outline.set_linewidth(2)
                    
                    ax.tick_params(axis='both', which='major', labelsize=font_size, colors='black')
                    
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">❌ Error</h3>', unsafe_allow_html=True)
        st.error(f"Error processing files: {str(e)}")
        import traceback
        with st.expander("📋 Full error details"):
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
    
    ### File Formats Expected:
    
    **Calibration File:**
    - CSV format with columns: Z, Drift, CCS, CCS Std.Dev.
    - Drift times should be in **seconds**
    - Multiple charge states supported
    - Rows with error ≥ CCS value are automatically removed
    
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
    1. **Charge State Selection**: Choose charge state
    2. **CCS Conversion**: Calibration file is used to convert drift times to CCSs
    3. **Data Sorting**: CCS values sorted in ascending order for interpolation
    4. **Duplicate Removal**: Remove duplicate CCS or TrapCV values
    5. **CV Slice Normalisation**: (If enabled) normalises each CV slice to maximum intensity of 1
    6. **2D Interpolation**: (If enabled) adds additional interpolated datapoints
    7. **Smoothing**: Applied to final CCS matrix (if enabled)
    """)

