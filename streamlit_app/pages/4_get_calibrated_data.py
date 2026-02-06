"""
Streamlit page for calibrating and scaling ATDs.

This page allows users to:
1. Upload raw ATDs and mass spectra
2. Upload calibration CSV files from previous step
3. Provide protein masses and charge ranges
4. Set integration ranges for mass spectrum scaling
5. Process data to generate calibrated and scaled CCSDs

Uses the IMSocio.processing module for all core scientific processing.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app import styling
from imsocio.processing import (
    DriftCalibrationProcessor,
    get_automatic_range,
    calculate_theoretical_mz,
    plot_spectrum_with_integration,
    plot_full_spectrum_with_charge_states,
    PROTON_MASS
)


@st.cache_data
def cached_calculate_scale_factor(
    ms_data_hash: str,
    protein: str,
    charge: int,
    mass: float,
    range_min: float,
    range_max: float,
    use_max_intensity: bool,
    smoothing_window: int,
    _ms_df: pd.DataFrame  # Leading underscore tells Streamlit not to hash this
):
    """Cached version of scale factor calculation."""
    scale_factor, mz = DriftCalibrationProcessor.calculate_scale_factor(
        _ms_df,
        protein,
        charge,
        mass,
        {(protein, charge): (range_min, range_max)},
        use_max_intensity=use_max_intensity,
        smoothing_window=smoothing_window
    )
    return scale_factor, mz


@st.cache_data
def cached_plot_spectrum(
    ms_data_hash: str,
    mz_theoretical: float,
    range_min: float,
    range_max: float,
    smoothing_window: int,
    show_zoomed: bool,
    _ms_df: pd.DataFrame
):
    """Cached version of spectrum plotting - only loads relevant m/z range."""
    # Determine the m/z window to plot
    if show_zoomed:
        mz_window_min = mz_theoretical * 0.90
        mz_window_max = mz_theoretical * 1.10
    else:
        mz_window_min = _ms_df["m/z"].min()
        mz_window_max = _ms_df["m/z"].max()
    
    # Only load data in the visible range - HUGE performance improvement
    ms_df_filtered = _ms_df[
        (_ms_df["m/z"] >= mz_window_min) & 
        (_ms_df["m/z"] <= mz_window_max)
    ].copy()
    
    area, range_outside, fig = plot_spectrum_with_integration(
        ms_df_filtered,
        mz_theoretical,
        (range_min, range_max),
        smoothing_window=smoothing_window,
        show_zoomed=show_zoomed
    )
    return area, range_outside, fig


class UI:
    """Streamlit UI components for drift calibration and scaling."""
    
    @staticmethod
    def show_main_header():
        """Display the main page header."""
        st.markdown(
            """
            <div class="main-header">
                <h1>Calibrate and Scale Data</h1>
                <p>Uses calibration files to convert drift times to CCSs, and normalizes and scales charge states according
                 to MS intensity</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_info_card():
        """Display information about the page functionality."""
        st.markdown(
            """
            <div class="info-card">
                <p>Use this page to generate calibrated and scaled CCSDs using the calibration files generated in the 
                previous step and your raw data. This step completes the calibration process by converting all drift
                times to CCS, normalising charge state ATD intensities to a maximum of 1, then scaling according to 
                mass spectrum intensity.</p>
                <p><strong>What you'll need:</strong></p>
                <ul>
                    <li><strong>ZIP file containing raw ATDs:</strong> Each protein folder should contain 
                    X.txt or X.csv files (where X is the charge state) and a mass_spectrum.txt file</li>
                    <li><strong>CSV files from the 'Process Output Files' step:</strong> These contain the 
                    calibration data generated in the previous step</li>
                    <li><strong>Protein masses:</strong> Molecular mass (Da) for each protein - this needs to be accurate
                    as it will inform the integration ranges for determining scale factors.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_upload_section():
        """Display upload section for drift files."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📁 Step 1: Upload Raw Data</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.file_uploader(
            "Upload zipped folder containing sample ATDs",
            type="zip",
            help="ZIP file should contain sample folders, within which there should be ATDs saved as .txt or .csv files and the mass spectrum saved as mass_spectrum.txt",
            key="drift_zip"
        )
    
    @staticmethod
    def show_calibration_upload():
        """Display upload section for calibration CSV files."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📊 Step 2: Upload Calibration Data</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.file_uploader(
            "Upload the calibration CSV files from the 'Process Output Files' page",
            type="csv", 
            accept_multiple_files=True,
            help="Select all CSV files generated in the previous step",
            key="cal_csvs"
        )
    
    @staticmethod
    def get_instrument_settings():
        """Get instrument configuration from user."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">⚙️ Step 3: Instrument and Injection Time</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            instrument_type = st.selectbox(
                "Instrument Type",
                ["Synapt", "Cyclic"],
                help="Synapt: Drift times copied directly to input files. Cyclic: requires injection time correction to drift times."
            )
        
        with col2:
            inject_time = 0.0
            if instrument_type == "Cyclic":
                inject_time = st.number_input(
                    "Injection Time (ms)",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                    help="Time subtracted from drift times for cyclic IMS"
                )
        
        return instrument_type, inject_time
    
    @staticmethod
    def get_protein_masses(protein_names: List[str]) -> Dict[str, float]:
        """Get molecular masses for each protein."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">⚖️ Step 4: Enter Protein Masses (nominal mass guided by experiment for larger proteins where adducts blah)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        masses = {}
        cols = st.columns(min(3, len(protein_names)))
        for i, name in enumerate(protein_names):
            with cols[i % len(cols)]:
                mass = st.number_input(
                    f"Mass (Da) for {name}",
                    min_value=0.0,
                    value=0.0,
                    key=f"mass_{name}"
                )
                masses[name] = mass
        return masses
    
    @staticmethod
    def get_charge_ranges(protein_names: List[str]) -> Dict[str, Tuple[int, int]]:
        """Get charge state ranges for each protein."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">🔋 Step 5: Select Charge State Ranges</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        charge_ranges = {}
        for protein in protein_names:
            col1, col2 = st.columns(2)
            with col1:
                min_charge = st.number_input(
                    f"{protein} - Min charge",
                    min_value=1,
                    value=2,
                    key=f"min_charge_{protein}"
                )
            with col2:
                max_charge = st.number_input(
                    f"{protein} - Max charge",
                    min_value=min_charge,
                    value=min_charge + 5,
                    key=f"max_charge_{protein}"
                )
            charge_ranges[protein] = (min_charge, max_charge)
        
        return charge_ranges
    
    @staticmethod
    def configure_integration_ranges(
        protein_names: List[str],
        protein_masses: Dict[str, float],
        charge_ranges: Dict[str, Tuple[int, int]],
        ms_data: Dict[str, pd.DataFrame],
        use_max_intensity: bool = False
    ) -> Tuple[Dict[Tuple[str, int], Tuple[float, float]], pd.DataFrame]:
        """
        Configure integration ranges for mass spectrum scaling.
        
        Provides interactive UI for each protein/charge combination with:
        - Automatic range suggestion
        - Manual adjustment
        - Live preview with baseline fitting
        - Real-time scale factor calculation and display
        
        Returns:
            Tuple of (scale_ranges, scale_factors_df)
        """
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📐 Step 7: Set Integration Ranges</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        method_text = "maximum intensity" if use_max_intensity else "baseline-corrected integration"
        st.info(
            f"For each protein/charge combination, set the m/z integration range for mass spectrum scaling. "
            f"Scale factors will be calculated using {method_text}."
        )
        
        scale_ranges = {}
        scale_factor_data = []  # For live table
        
        # Pre-compute hashes for all proteins (only once per protein's mass spectrum)
        if 'ms_hashes' not in st.session_state:
            st.session_state.ms_hashes = {}
        
        for protein in protein_names:
            if protein not in ms_data or protein_masses.get(protein, 0) == 0:
                continue
            
            # Compute hash only once per protein
            if protein not in st.session_state.ms_hashes:
                st.session_state.ms_hashes[protein] = hashlib.md5(
                    ms_data[protein].to_json().encode()
                ).hexdigest()
            
            st.markdown(f"#### {protein}")
            ms_df = ms_data[protein]
            ms_hash = st.session_state.ms_hashes[protein]
            mass = protein_masses[protein]
            min_charge, max_charge = charge_ranges.get(protein, (2, 4))
            
            # Protein-level defaults (collapsible section)
            with st.expander(f"⚙️ {protein} - Default Settings", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    # Default integration range percentage
                    default_percent_key = f"default_percent_{protein}"
                    if default_percent_key not in st.session_state:
                        st.session_state[default_percent_key] = 5.0
                    
                    default_percent = st.number_input(
                        "Default integration range (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=st.session_state[default_percent_key],
                        step=0.5,
                        key=default_percent_key,
                        help="This percentage will be used as default for all charge states of this protein"
                    )
                
                with col_b:
                    # Default smoothing window
                    default_smooth_key = f"default_smooth_{protein}"
                    if default_smooth_key not in st.session_state:
                        st.session_state[default_smooth_key] = 51
                    
                    # Ensure the value is odd before creating the widget
                    current_value = st.session_state[default_smooth_key]
                    if current_value % 2 == 0:
                        current_value += 1
                        st.session_state[default_smooth_key] = current_value
                    
                    default_smoothing = st.number_input(
                        "Default smoothing window",
                        min_value=3,
                        max_value=201,
                        value=current_value,
                        step=2,
                        key=default_smooth_key,
                        help="Window size for baseline smoothing (must be odd). Applies to all charge states."
                    )
                
                # Button to apply defaults to all charge states
                if st.button(f"Apply defaults to all {protein} charge states", key=f"apply_defaults_{protein}"):
                    charges = list(range(min_charge, max_charge + 1))
                    for charge in charges:
                        mz = calculate_theoretical_mz(mass, charge)
                        auto_min, auto_max = get_automatic_range(mz, default_percent)
                        st.session_state[f"range_min_{protein}_{charge}"] = auto_min
                        st.session_state[f"range_max_{protein}_{charge}"] = auto_max
                        st.session_state[f"smoothing_{protein}_{charge}"] = default_smoothing
                    st.success(f"✅ Applied defaults to all {protein} charge states")
            
            # Create tabs for each charge state
            charges = list(range(min_charge, max_charge + 1))
            tabs = st.tabs([f"Charge {c}+" for c in charges])
            
            for idx, charge in enumerate(charges):
                with tabs[idx]:
                    mz = calculate_theoretical_mz(mass, charge)
                    
                    # Initialize session state if not present (using protein defaults)
                    if f"range_min_{protein}_{charge}" not in st.session_state:
                        default_auto_min, default_auto_max = get_automatic_range(
                            mz, 
                            st.session_state.get(f"default_percent_{protein}", 5.0)
                        )
                        st.session_state[f"range_min_{protein}_{charge}"] = default_auto_min
                        st.session_state[f"range_max_{protein}_{charge}"] = default_auto_max
                    
                    # Initialize smoothing if not present
                    if f"smoothing_{protein}_{charge}" not in st.session_state:
                        st.session_state[f"smoothing_{protein}_{charge}"] = st.session_state.get(
                            f"default_smooth_{protein}", 51
                        )
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        with st.form(key=f"form_{protein}_{charge}"):
                            # Auto-range suggestion using protein default
                            auto_percent = st.slider(
                                "Auto-range %",
                                min_value=1.0,
                                max_value=20.0,
                                value=float(st.session_state.get(f"default_percent_{protein}", 5.0)),
                                step=0.5,
                                help="Adjust this to change the integration range for this specific charge state"
                            )
                            
                            # Ensure the smoothing value is odd
                            current_smoothing = st.session_state[f"smoothing_{protein}_{charge}"]
                            if current_smoothing % 2 == 0:
                                current_smoothing += 1
                                st.session_state[f"smoothing_{protein}_{charge}"] = current_smoothing
                            
                            # Manual range input
                            range_min = st.number_input(
                                "Min m/z",
                                value=st.session_state[f"range_min_{protein}_{charge}"],
                                format="%.3f"
                            )
                            range_max = st.number_input(
                                "Max m/z",
                                value=st.session_state[f"range_max_{protein}_{charge}"],
                                format="%.3f"
                            )
                            
                            # Smoothing window for this charge state
                            smoothing_window = st.number_input(
                                "Smoothing window",
                                min_value=3,
                                max_value=201,
                                value=current_smoothing,
                                step=2,
                                help="Window size for Savitzky-Golay smoothing (must be odd)"
                            )
                            
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                update_button = st.form_submit_button("🔄 Update", type="primary")
                            with col_btn2:
                                auto_button = st.form_submit_button("Use auto-range")
                            
                            if auto_button:
                                auto_min, auto_max = get_automatic_range(mz, auto_percent)
                                st.session_state[f"range_min_{protein}_{charge}"] = auto_min
                                st.session_state[f"range_max_{protein}_{charge}"] = auto_max
                            
                            if update_button:
                                st.session_state[f"range_min_{protein}_{charge}"] = range_min
                                st.session_state[f"range_max_{protein}_{charge}"] = range_max
                                st.session_state[f"smoothing_{protein}_{charge}"] = smoothing_window
                        
                        # Use values from session state for calculations
                        range_min = st.session_state[f"range_min_{protein}_{charge}"]
                        range_max = st.session_state[f"range_max_{protein}_{charge}"]
                        smoothing_window = st.session_state[f"smoothing_{protein}_{charge}"]
                        
                        scale_ranges[(protein, charge)] = (range_min, range_max)
                        
                        # Show theoretical m/z
                        st.metric("Theoretical m/z", f"{mz:.3f}")
                        
                        # Calculate scale factor using cached function (ms_hash already computed above)
                        scale_factor, _ = cached_calculate_scale_factor(
                            ms_hash,
                            protein,
                            charge,
                            mass,
                            range_min,
                            range_max,
                            use_max_intensity,
                            smoothing_window,
                            ms_df
                        )
                        
                        if scale_factor is not None:
                            st.metric("Scale Factor", f"{scale_factor:.2e}")
                            # Store for table
                            scale_factor_data.append({
                                "Protein": protein,
                                "Charge": charge,
                                "m/z": f"{mz:.3f}",
                                "Range": f"{range_min:.3f} - {range_max:.3f}",
                                "Scale Factor": f"{scale_factor:.2e}"
                            })
                        else:
                            st.warning("⚠ Could not calculate scale factor")
                    
                    with col2:
                        # Plot with integration preview
                        show_zoomed = st.checkbox(
                            "Zoom to ±10%",
                            value=True,
                            key=f"zoom_{protein}_{charge}"
                        )
                        
                        # Use cached plotting function (ms_hash already computed above)
                        area, range_outside, fig = cached_plot_spectrum(
                            ms_hash,
                            mz,
                            range_min,
                            range_max,
                            smoothing_window,
                            show_zoomed,
                            ms_df
                        )
                        
                        if fig:
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                        
                        # Display appropriate message based on method
                        if area is not None:
                            if use_max_intensity:
                                st.info(f"Scale factor based on max intensity in range")
                            else:
                                st.info(f"Integrated area: {area:.2e}")
                        
                        if range_outside:
                            st.warning("⚠ Integration range extends beyond visible window")
        
        # Create DataFrame of scale factors
        scale_factors_df = pd.DataFrame(scale_factor_data) if scale_factor_data else None
        
        return scale_ranges, scale_factors_df
    
    @staticmethod
    def show_processing_results(result, skipped_files: List[str]):
        """Display processing results."""
        st.markdown(
            f"""
            <div class="success-card">
                <strong>✅ Processing Complete!</strong><br>
                • Processed: <span class="metric-badge">{result.processed_files} files</span><br>
                • Matched: <span class="metric-badge">{result.matched_points} data points</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if skipped_files:
            with st.expander(f"⚠ Skipped {len(skipped_files)} files", expanded=False):
                for msg in skipped_files[:20]:
                    st.write(f"• {msg}")
                if len(skipped_files) > 20:
                    st.write(f"...and {len(skipped_files) - 20} more")


def extract_protein_names_from_csvs(cal_csvs: List) -> List[str]:
    """Extract protein names from calibration CSV filenames."""
    return [f.name.replace(".csv", "") for f in cal_csvs]


def load_mass_spectra(drift_zip) -> Dict[str, pd.DataFrame]:
    """Load mass spectrum files from ZIP with detailed troubleshooting."""
    import zipfile
    import tempfile
    import os
    
    ms_data = {}
    search_log = []  # Track where we looked
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract ZIP
        zip_path = Path(tmpdir) / "drift.zip"
        with open(zip_path, "wb") as f:
            f.write(drift_zip.getvalue())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find mass spectrum files
        for root, dirs, files in os.walk(tmpdir):
            protein_name = os.path.basename(root)
            
            # Skip root directory (same as module version)
            if root == tmpdir or protein_name == os.path.basename(tmpdir):
                search_log.append(f"⏭️ Skipped root directory: {protein_name}")
                continue
            
            # Skip __MACOSX and hidden folders
            if protein_name.startswith("__") or protein_name.startswith("."):
                search_log.append(f"⏭️ Skipped hidden folder: {protein_name}")
                continue
            
            # Log what we're checking
            relative_path = os.path.relpath(root, tmpdir)
            search_log.append(f"📂 Checking folder: {relative_path}")
            
            # Try exact match first
            if "mass_spectrum.txt" in files:
                ms_path = os.path.join(root, "mass_spectrum.txt")
                ms_df = DriftCalibrationProcessor.load_mass_spectrum(ms_path)
                if ms_df is not None:
                    ms_data[protein_name] = ms_df
                    search_log.append(f"✅ Found mass_spectrum.txt in {relative_path}")
                else:
                    search_log.append(f"⚠️ Found mass_spectrum.txt in {relative_path} but failed to load (check file format)")
            else:
                # Try case-insensitive match
                mass_spec_files = [f for f in files if f.lower() == "mass_spectrum.txt"]
                if mass_spec_files:
                    actual_name = mass_spec_files[0]
                    search_log.append(f"⚠️ Found '{actual_name}' (case mismatch) in {relative_path}")
                    ms_path = os.path.join(root, actual_name)
                    ms_df = DriftCalibrationProcessor.load_mass_spectrum(ms_path)
                    if ms_df is not None:
                        ms_data[protein_name] = ms_df
                        search_log.append(f"✅ Loaded '{actual_name}' successfully")
                    else:
                        search_log.append(f"❌ Failed to load '{actual_name}' (check file format)")
                else:
                    # Log what files ARE there
                    if files:
                        file_list = ", ".join(files[:5])
                        if len(files) > 5:
                            file_list += f", ... ({len(files)} total)"
                        search_log.append(f"❌ No mass_spectrum.txt in {relative_path}. Found: {file_list}")
                    else:
                        search_log.append(f"❌ No files in {relative_path}")
    
    # Display troubleshooting information if no files found
    if not ms_data:
        with st.expander("🔍 Troubleshooting: Where did we look?", expanded=True):
            st.markdown("**Search Log:**")
            for log_entry in search_log:
                st.text(log_entry)
            
            st.markdown("---")
            st.markdown("**Common Issues:**")
            st.markdown("""
            1. **Extra wrapper folder**: Your ZIP might have an extra folder layer
               - ❌ `data.zip/data/protein1/mass_spectrum.txt`
               - ✅ `data.zip/protein1/mass_spectrum.txt`
            
            2. **Case sensitivity**: File must be named exactly `mass_spectrum.txt`
               - ❌ `Mass_spectrum.txt`, `MASS_SPECTRUM.TXT`
               - ✅ `mass_spectrum.txt`
            
            3. **Hidden file extension**: Windows may hide `.txt` extension
               - Your file might actually be `mass_spectrum.txt.txt`
               - In File Explorer: View → Show → File name extensions
            
            4. **Mac hidden folders**: ZIP created on Mac may contain `__MACOSX` folders
               - These are automatically skipped
            
            5. **File format**: Must be tab-separated with two columns (m/z, Intensity)
            """)
    else:
        # Show success summary
        st.success(f"✅ Found mass spectra for: {', '.join(ms_data.keys())}")
    
    return ms_data


def main():
    """Main application logic."""
    # Load custom CSS
    styling.load_custom_css()
    
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    # Show header and info
    UI.show_main_header()
    UI.show_info_card()
    
    # Step 1: Upload drift files
    drift_zip = UI.show_upload_section()
    if not drift_zip:
        return
    
    # Step 2: Upload calibration CSVs
    cal_csvs = UI.show_calibration_upload()
    if not cal_csvs:
        st.warning("Please upload calibration CSV files to continue.")
        return
    
    # Extract protein names
    protein_names = extract_protein_names_from_csvs(cal_csvs)
    st.success(f"Found {len(protein_names)} protein(s): {', '.join(protein_names)}")
    
    # Step 3: Instrument settings
    instrument_type, inject_time = UI.get_instrument_settings()
    
    # Step 4: Protein masses
    protein_masses = UI.get_protein_masses(protein_names)
    if any(m == 0.0 for m in protein_masses.values()):
        st.warning("⚠ Please enter masses for all proteins.")
        return
    
    # Step 5: Charge ranges
    charge_ranges = UI.get_charge_ranges(protein_names)
    
    # Load mass spectra for integration range configuration
    with st.spinner("Loading mass spectra..."):
        ms_data = load_mass_spectra(drift_zip)
    
    if not ms_data:
        st.error("""
        ❌ **No mass spectrum files found in ZIP**
        
        Each protein folder must contain a file named exactly `mass_spectrum.txt`.
        Please check the troubleshooting information above for details about where we looked.
        """)
        return
    
    # Step 5.5: Scaling method selection
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">📏 Step 6: Select Scaling Method</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    use_max_intensity = st.radio(
        "How should I calculate my scale factors?",
        options=[False, True],
        format_func=lambda x: "Integration (baseline-corrected area)" if not x else "Maximum intensity in range",
        help="Integration provides more accurate quantification for salty proteins because you are including all signal intensity for the lower charge states where salt is retained."
             "Maximum intensity scaling is suitable in many cases - if the sample is not salty, or if you wish to omit salt adducts from the scaling."
    )
    
    # Step 7: Configure integration ranges
    scale_ranges, scale_factors_df = UI.configure_integration_ranges(
        protein_names,
        protein_masses,
        charge_ranges,
        ms_data,
        use_max_intensity
    )
    
    if not scale_ranges:
        st.warning("⚠ Please configure at least one integration range.")
        return
    
    # Collect smoothing parameters for each protein/charge
    smoothing_params = {}
    for protein in protein_names:
        min_charge, max_charge = charge_ranges.get(protein, (2, 4))
        for charge in range(min_charge, max_charge + 1):
            key = (protein, charge)
            if key in scale_ranges:  # Only include if there's a scale range
                smoothing_params[key] = st.session_state.get(
                    f"smoothing_{protein}_{charge}", 51
                )
    
    # Display scale factors table
    if scale_factors_df is not None and len(scale_factors_df) > 0:
        st.markdown("---")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📊 Scale Factors Summary</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(
            scale_factors_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Processing section
    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">🚀 Process Data</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Process All Data", type="primary"):
        with st.spinner("Processing..."):
            # Process using imsocio library
            result, skipped_files = DriftCalibrationProcessor.match_and_calibrate(
                drift_zip=drift_zip,
                cal_csvs=cal_csvs,
                instrument_type=instrument_type,
                inject_time=inject_time,
                charge_ranges=charge_ranges,
                scale_ranges=scale_ranges,
                protein_masses=protein_masses,
                ms_data=ms_data,
                use_max_intensity=use_max_intensity,
                smoothing_params=smoothing_params
            )
            
            # Show results
            UI.show_processing_results(result, skipped_files)
            
            # Prepare download
            if result.output_buffers:
                zip_buffer = DriftCalibrationProcessor.prepare_zip(result.output_buffers)
                
                st.markdown('<h3 class="section-header">📥 Download Results</h3>', unsafe_allow_html=True)
                st.markdown('change name as required ... ')
                
                zip_filename = st.text_input(
                    "ZIP filename", 
                    value="calibrated_scaled_data.zip", 
                    key="zip_filename"
                )
                
                st.download_button(
                    label="📦 Download Calibrated Data (ZIP)",
                    data=zip_buffer,
                    file_name=zip_filename if zip_filename else "calibrated_scaled_data.zip",
                    mime="application/zip"
                )



if __name__ == "__main__":
    main()

