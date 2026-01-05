"""
Alternative Calibration Method - ln-ln Plot

This page provides an alternative calibration visualization using the ln-ln method.
Input: CSV file from the standard calibration page (1_calibrate.py)
Output: Plot of ln(modified drift time) vs ln(literature CCS)

Modified drift time = dt - (EDC * sqrt(m/z) / 1000)
where EDC = Enhanced Duty Cycle
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Streamlit specific imports
from streamlit_app import styling

# imsocio imports
from imsocio.calibration import (
    prepare_alternative_calibration_data,
    calculate_modified_drift_time,
    calculate_modified_modified_drift_time
)
from imsocio.io.writers import dataframe_to_csv_buffer


class AlternativeCalibrationInterface:
    """Interface for alternative (ln-ln) calibration visualization."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown(
            '<div class="main-header">'
            '<h1>📈 Alternative Calibration Method</h1>'
            '<p>Calibration using method proposed by Richardson <em>et al.</em><sup>2</sup></p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <h3>About this Method:</h3>
            <p>Details of this method can be found in the following publication: https://doi.org/10.1021/acs.analchem.0c04948<sup>2</sup></p>
            <p>This has been included in IMSocio to allow for comparison with results obtained using IMSCal<sup>1</sup>.</p>

        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> Bush, M. F., Hall, Z., Giles, K., Hoyes, J., Robinson, C. V., & Ruotolo, B. T. (2010). Collision cross sections of proteins and their complexes: A calibration framework and database for gas-phase structural biology. <em>Analytical Chemistry</em>, 82(22), 9557–9565. https://doi.org/10.1021/ac1022953</p>
            <p><sup>2</sup> Ujma, J., Giles, K., Morris, M., & Barran, P. E. (2016). New high resolution ion mobility mass spectrometer capable of measurements of collision cross sections from 150 to 520 K. <em>Analytical Chemistry</em>, 88(19), 9469–9478. https://doi.org/10.1021/acs.analchem.6b01812</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def upload_calibration_file() -> Optional[pd.DataFrame]:
        """Handle calibration CSV file upload."""
        st.markdown(
            '<div class="section-header">📁 Step 1: Upload Calibration Data</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Upload the CSV file generated from the <strong>Calibrate</strong> page.</p>
            <p>This file should contain columns: protein, mass, charge state, drift time, r2, calibrant_value</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose calibration CSV file",
            type=['csv'],
            key="calibration_csv_upload"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['mass', 'charge state', 'drift time', 'calibrant_value']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    return None
                
                st.success(f"✅ Loaded {len(df)} calibration points")
                
                # Display preview
                with st.expander("Preview Data"):
                    st.dataframe(df)
                
                return df
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
        
        return None
    
    @staticmethod
    def get_parameters() -> tuple[float, str, str, float]:
        """Get Enhanced Duty Cycle, drift gas, instrument type, and inject time from user."""
        st.markdown(
            '<div class="section-header">⚙️ Step 2: Enter Parameters</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Enter the instrument parameters used in your experiment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            edc = st.number_input(
                "Enhanced Duty Cycle (EDC)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                format="%.3f",
                help="Enter the EDC value for your instrument"
            )
        
        with col2:
            drift_gas = st.selectbox(
                "Drift Gas",
                options=["Nitrogen", "Helium"],
                help="Select the drift gas used in your experiment"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            instrument_type = st.selectbox(
                "Instrument Type",
                options=["Synapt", "Cyclic"],
                help="Select the instrument type"
            )
        
        with col4:
            if instrument_type == "Cyclic":
                inject_time = st.number_input(
                    "Inject Time (ms)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.01,
                    format="%.3f",
                    help="Inject time to subtract from drift times"
                )
            else:
                inject_time = 0.0
                st.info("ℹ️ No inject time correction needed")
        
        return edc, drift_gas.lower(), instrument_type, inject_time
    
    @staticmethod
    def create_lnln_plot(df: pd.DataFrame, show_regression: bool = True) -> tuple[plt.Figure, float, float]:
        """
        Create ln-ln calibration plot.
        
        Args:
            df: DataFrame with ln_modified_drift_time and ln_modified_ccs columns
            show_regression: Whether to show linear regression line
            
        Returns:
            Tuple of (matplotlib Figure object, slope, intercept)
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Scatter plot - swapped axes: ln(tD') on x-axis, ln(CCS') on y-axis
        ax.scatter(
            df['ln_modified_drift_time'],
            df['ln_modified_ccs'],
            c='blue',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label='Calibration points'
        )
        
        # Add linear regression if requested
        if show_regression:
            # Perform linear regression - ln(CCS') vs ln(tD')
            z = np.polyfit(df['ln_modified_drift_time'], df['ln_modified_ccs'], 1)
            p = np.poly1d(z)
            
            # Calculate R²
            y_pred = p(df['ln_modified_drift_time'])
            ss_res = np.sum((df['ln_modified_ccs'] - y_pred) ** 2)
            ss_tot = np.sum((df['ln_modified_ccs'] - np.mean(df['ln_modified_ccs'])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Plot regression line
            x_line = np.linspace(df['ln_modified_drift_time'].min(), df['ln_modified_drift_time'].max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                'r-',
                linewidth=2,
                label=f'Linear fit (R² = {r_squared:.4f})'
            )
            
            # Add equation to plot
            equation_text = f'y = {z[0]:.4f}x + {z[1]:.4f}'
            ax.text(
                0.05, 0.95,
                equation_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # Formatting - swapped axis labels
        ax.set_xlabel('ln(tD\') [ln(ms)]', fontsize=12, fontweight='bold')
        ax.set_ylabel('ln(CCS\') [ln(nm²·√μ/z)]', fontsize=12, fontweight='bold')
        ax.set_title('Alternative Calibration: ln-ln Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Return slope and intercept if regression was shown
        if show_regression:
            z = np.polyfit(df['ln_modified_drift_time'], df['ln_modified_ccs'], 1)
            return fig, z[0], z[1]
        else:
            return fig, None, None
    
    @staticmethod
    def create_td_double_prime_plot(df: pd.DataFrame, show_regression: bool = True) -> tuple[plt.Figure, float, float]:
        """
        Create tD'' vs CCS literature plot.
        
        Args:
            df: DataFrame with modified_modified_drift_time and calibrant_value columns
            show_regression: Whether to show linear regression line
            
        Returns:
            Tuple of (matplotlib Figure object, slope, intercept)
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Scatter plot
        ax.scatter(
            df['calibrant_value'],
            df['modified_modified_drift_time'],
            c='green',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label='Calibration points'
        )
        
        slope, intercept = None, None
        
        # Add linear regression if requested
        if show_regression:
            # Perform linear regression - tD'' vs CCS
            z = np.polyfit(df['calibrant_value'], df['modified_modified_drift_time'], 1)
            p = np.poly1d(z)
            slope, intercept = z[0], z[1]
            
            # Calculate R²
            y_pred = p(df['calibrant_value'])
            ss_res = np.sum((df['modified_modified_drift_time'] - y_pred) ** 2)
            ss_tot = np.sum((df['modified_modified_drift_time'] - np.mean(df['modified_modified_drift_time'])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Plot regression line
            x_line = np.linspace(df['calibrant_value'].min(), df['calibrant_value'].max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                'r-',
                linewidth=2,
                label=f'Linear fit (R² = {r_squared:.4f})'
            )
            
            # Add equation to plot
            equation_text = f'y = {slope:.6f}x + {intercept:.4f}'
            ax.text(
                0.05, 0.95,
                equation_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
            )
        
        # Formatting
        ax.set_xlabel('CCS Literature (nm²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('tD\'\' [ms]', fontsize=12, fontweight='bold')
        ax.set_title('Calibration: tD\'\' vs CCS Literature', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig, slope, intercept
    
    @staticmethod
    def display_statistics(df: pd.DataFrame):
        """Display statistical summary of the calibration data."""
        st.markdown(
            '<div class="section-header">📊 Statistics</div>',
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Points", len(df))
        
        with col2:
            st.metric(
                "CCS Range (literature)",
                f"{df['calibrant_value'].min():.0f} - {df['calibrant_value'].max():.0f} nm²"
            )
        
        with col3:
            st.metric(
                "CCS' Range",
                f"{df['modified_ccs'].min():.2f} - {df['modified_ccs'].max():.2f}"
            )
        
        # Detailed statistics in expander
        with st.expander("View Detailed Statistics"):
            stats_df = pd.DataFrame({
                'Modified Drift Time (tD\')': df['modified_drift_time'].describe(),
                'ln(tD\')': df['ln_modified_drift_time'].describe(),
                'Modified CCS (CCS\')': df['modified_ccs'].describe(),
                'ln(CCS\')': df['ln_modified_ccs'].describe(),
                'Modified Modified Drift Time (tD\'\')': df['modified_modified_drift_time'].describe()
            })
            st.dataframe(stats_df)
    
    @staticmethod
    def export_results(df: pd.DataFrame):
        """Provide download options for processed data."""
        st.markdown(
            '<div class="section-header">💾 Export Results</div>',
            unsafe_allow_html=True
        )
        
        # CSV download
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Processed Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="alternative_calibration_data.csv",
            mime="text/csv",
            help="Download the full dataset including calculated ln values"
        )
    
    @staticmethod
    def upload_raw_data_for_calibration():
        """Upload raw data ZIP file for generating calibration files."""
        st.markdown(
            '<div class="section-header">📁 Generate Calibration Files from Raw Data</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Upload a zipped folder containing your raw data organized by protein.</p>
            <p><strong>Expected structure:</strong> Protein folders containing .txt or .csv files (one per charge state)</p>
            <p>This will convert drift times to CCS values using the calibration parameters above.</p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader(
            "Upload zipped folder with raw data",
            type=['zip'],
            key="raw_data_zip_upload",
            help="ZIP file should contain protein folders with raw ATD files"
        )
    
    @staticmethod
    def parse_raw_atd_file(file_path: Path) -> pd.DataFrame:
        """
        Parse a raw ATD file (txt or csv) to extract drift time and intensity data.
        
        Args:
            file_path: Path to the ATD file
            
        Returns:
            DataFrame with columns 'Drift' and 'Intensity'
        """
        # Try reading as CSV first
        try:
            df = pd.read_csv(file_path, comment='#')
            # Assume first column is drift time, second is intensity
            df.columns = ['Drift', 'Intensity']
            return df[['Drift', 'Intensity']]
        except:
            pass
        
        # Try reading as space/tab separated
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)
            df.columns = ['Drift', 'Intensity']
            return df[['Drift', 'Intensity']]
        except Exception as e:
            raise ValueError(f"Could not parse file {file_path}: {str(e)}")
    
    @staticmethod
    def extract_charge_from_filename(filename: str) -> Optional[int]:
        """
        Extract charge state from filename.
        
        Supports patterns like:
        - #range_15.txt (ORIGAMI format)
        - charge15.csv, charge_15.txt
        - 15.csv, 15.txt
        - name_charge15.csv
        
        Args:
            filename: Name of the file
            
        Returns:
            Charge state as integer, or None if not found
        """
        import re
        
        # Remove extension
        name = Path(filename).stem
        
        # Patterns to find charge state (matching readers.py)
        patterns = [
            r'#range_(\d+)',         # Matches "#range_24" (ORIGAMI format)
            r'range_(\d+)\.txt',     # Matches "range_24.txt"
            r'range_(\d+)_',         # Matches "range_24_"
            r'_(\d+)\.txt_raw',      # Matches "_24.txt_raw"
            r'_(\d+)_raw$',          # Matches "_24_raw" at end
            r'charge(\d+)',          # Matches "charge15" or "charge_15"
            r'_(\d+)$',              # Matches "_24" at end
            r'^(\d+)$'               # Matches just a number
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                charge = int(match.group(1))
                # Validate charge state is reasonable
                if 1 <= charge <= 200:
                    return charge
        
        return None
    
    @staticmethod
    def generate_calibration_files(
        zip_buffer,
        slope_td_ccs: float,
        intercept_td_ccs: float,
        slope_lnln: float,
        edc: float,
        mass_dict: Dict[str, float],
        drift_gas: str = 'nitrogen',
        instrument_type: str = 'Synapt',
        inject_time: float = 0.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate calibration files from raw data using alternative calibration method.
        
        Args:
            zip_buffer: Uploaded ZIP file buffer
            slope_td_ccs: Slope from tD'' vs CCS calibration (for final CCS calculation)
            intercept_td_ccs: Intercept from tD'' vs CCS calibration
            slope_lnln: Slope from ln-ln calibration (for calculating tD'')
            edc: Enhanced duty cycle value
            mass_dict: Dictionary mapping protein names to their masses
            drift_gas: Drift gas type
            instrument_type: Instrument type ('Synapt' or 'Cyclic')
            inject_time: Inject time to subtract for Cyclic instruments (ms)
            
        Returns:
            Dictionary mapping protein names to calibration DataFrames
        """
        # Constants
        PROTON_MASS = 1.00727647
        DRIFT_GAS_MASSES = {
            'nitrogen': 28.0134,
            'helium': 4.0026
        }
        mass_drift_gas = DRIFT_GAS_MASSES[drift_gas.lower()]
        
        protein_calibrations = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP file
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Find all protein folders
            tmpdir_path = Path(tmpdir)
            
            for protein_folder in tmpdir_path.iterdir():
                if not protein_folder.is_dir():
                    continue
                
                protein_name = protein_folder.name
                
                # Skip if no mass provided for this protein
                if protein_name not in mass_dict:
                    st.warning(f"No mass provided for {protein_name}, skipping...")
                    continue
                
                protein_mass = mass_dict[protein_name]
                reduced_mass = (protein_mass * mass_drift_gas) / (protein_mass + mass_drift_gas)
                
                # Find all ATD files in the folder
                atd_files = list(protein_folder.glob('*.txt')) + list(protein_folder.glob('*.csv'))
                
                # Filter out mass spectrum files
                atd_files = [f for f in atd_files if 'mass_spectrum' not in f.name.lower()]
                
                if not atd_files:
                    continue
                
                # Process each charge state file
                all_charge_data = []
                
                for atd_file in atd_files:
                    # Extract charge state from filename
                    charge = AlternativeCalibrationInterface.extract_charge_from_filename(atd_file.name)
                    
                    if charge is None:
                        st.warning(f"Could not extract charge state from {atd_file.name}, skipping...")
                        continue
                    
                    try:
                        # Parse the ATD file
                        atd_df = AlternativeCalibrationInterface.parse_raw_atd_file(atd_file)
                        
                        # Calculate m/z
                        mz = (protein_mass + PROTON_MASS * charge) / charge
                        
                        # For each drift time, calculate CCS
                        drift_times = atd_df['Drift'].values
                        intensities = atd_df['Intensity'].values
                        
                        ccs_values = []
                        corrected_drift_times = []
                        
                        for dt in drift_times:
                            # Apply inject time correction for Cyclic instruments
                            if instrument_type == 'Cyclic':
                                dt_corrected = dt - inject_time
                            else:
                                dt_corrected = dt
                            
                            corrected_drift_times.append(dt_corrected)
                            
                            # Calculate modified drift time (tD')
                            td_prime = calculate_modified_drift_time(dt_corrected, edc, mz)
                            
                            # Calculate modified modified drift time (tD'')
                            # Using the slope from ln-ln calibration
                            # tD'' = (tD')^slope_lnln * |z| * sqrt(1/reduced_mass)
                            td_double_prime = calculate_modified_modified_drift_time(
                                td_prime,
                                slope_lnln,  # Use ln-ln slope here
                                charge,
                                reduced_mass
                            )
                            
                            # Convert tD'' to CCS using inverted linear relationship from tD'' vs CCS plot
                            # The plot fits: tD'' = slope_td_ccs * CCS + intercept_td_ccs
                            # Inverting to solve for CCS: CCS = (tD'' - intercept_td_ccs) / slope_td_ccs
                            ccs = (td_double_prime - intercept_td_ccs) / slope_td_ccs
                            ccs_values.append(ccs)
                        
                        # Create DataFrame for this charge state with corrected drift times
                        charge_df = pd.DataFrame({
                            'Z': charge,
                            'Drift': corrected_drift_times,
                            'CCS': ccs_values,
                            'CCS Std.Dev.': 0.0,  # Not calculated in this method
                            'Intensity': intensities
                        })
                        
                        all_charge_data.append(charge_df)
                        
                    except Exception as e:
                        st.warning(f"Error processing {atd_file.name}: {str(e)}")
                        continue
                
                # Combine all charge states for this protein
                if all_charge_data:
                    combined_df = pd.concat(all_charge_data, ignore_index=True)
                    protein_calibrations[protein_name] = combined_df
        
        return protein_calibrations


def main():
    """Main application logic."""
    # Clear any cached data to ensure fresh imports
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    
    # Apply custom styling
    styling.load_custom_css()
    
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    # Initialize interface
    interface = AlternativeCalibrationInterface()
    
    # Show header
    interface.show_header()
    
    # Step 1: Upload calibration file
    calibration_df = interface.upload_calibration_file()
    
    if calibration_df is None:
        st.info("👆 Please upload a calibration CSV file to continue.")
        # References section at bottom when no file uploaded
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> Bush, M.F., Hall, Z., Giles, K., Hoyes, J., Robinson, C.V., Ruotolo, B.T., 2010. Collision Cross Sections of Proteins and Their Complexes: A Calibration Framework and Database for Gas-Phase Structural Biology. Anal. Chem. 82, 9557–9565. https://doi.org/10.1021/ac1022953
</p>
            <p><sup>2</sup> Richardson, K., Langridge, D., Dixit, S.M., Ruotolo, B.T., 2021. An Improved Calibration Approach for Traveling Wave Ion Mobility Spectrometry: Robust, High-Precision Collision Cross Sections. Anal. Chem. 93, 3542–3550. https://doi.org/10.1021/acs.analchem.0c04948
</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Step 2: Get parameters
    edc_value, drift_gas, instrument_type, inject_time = interface.get_parameters()
    
    # Process data button
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🔄 Calculate and Plot", type="primary"):
        try:
            # Apply inject time correction to calibration data if Cyclic
            calibration_df_corrected = calibration_df.copy()
            if instrument_type == 'Cyclic':
                calibration_df_corrected['drift time'] = calibration_df_corrected['drift time'] - inject_time
                st.info(f"ℹ️ Applied inject time correction: {inject_time:.3f} ms subtracted from drift times")
            
            # Process the data
            with st.spinner("Processing calibration data..."):
                processed_df = prepare_alternative_calibration_data(
                    calibration_df_corrected,
                    edc_value,
                    drift_gas
                )
            
            st.success("✅ Data processed successfully!")
            
            # Display statistics
            interface.display_statistics(processed_df)
            
            # Create and display ln-ln plot
            st.markdown(
                '<div class="section-header">📈 ln-ln Calibration Plot</div>',
                unsafe_allow_html=True
            )
            
            # Plot options
            col1, col2 = st.columns([3, 1])
            with col2:
                show_regression = st.checkbox("Show linear fit", value=True, key="lnln_regression")
            
            # Generate and display ln-ln plot
            fig, slope_lnln, intercept_lnln = interface.create_lnln_plot(processed_df, show_regression)
            st.pyplot(fig)
            
            # Store ln-ln slope for later use in calibration file generation
            if slope_lnln is not None:
                st.session_state['lnln_slope'] = slope_lnln
                st.session_state['lnln_intercept'] = intercept_lnln
            
            # Create and display tD'' vs CCS plot
            st.markdown(
                '<div class="section-header">📊 tD\'\' vs CCS Literature Plot</div>',
                unsafe_allow_html=True
            )
            
            # Plot options for second plot
            col1, col2 = st.columns([3, 1])
            with col2:
                show_regression_td = st.checkbox("Show linear fit", value=True, key="td_regression")
            
            # Generate and display tD'' plot
            fig_td, slope_td, intercept_td = interface.create_td_double_prime_plot(processed_df, show_regression_td)
            st.pyplot(fig_td)
            
            # Display calibration parameters
            if slope_td is not None:
                st.markdown(
                    '<div class="section-header">🔢 Calibration Parameters</div>',
                    unsafe_allow_html=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Slope (tD\'\' vs CCS)", f"{slope_td:.8f}")
                with col2:
                    st.metric("Intercept", f"{intercept_td:.4f}")
                
                # Store calibration parameters in session state for later use
                st.session_state['calibration_slope'] = slope_td
                st.session_state['calibration_intercept'] = intercept_td
                st.session_state['calibration_edc'] = edc_value
                st.session_state['calibration_drift_gas'] = drift_gas
                st.session_state['calibration_processed_df'] = processed_df
                st.session_state['calibration_instrument_type'] = instrument_type
                st.session_state['calibration_inject_time'] = inject_time
                st.session_state['calibration_instrument_type'] = instrument_type
                st.session_state['calibration_inject_time'] = inject_time
            
            # Show processed data table
            with st.expander("View Processed Data"):
                st.dataframe(processed_df)
            
            # Export options
            interface.export_results(processed_df)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Add section for generating calibration files from raw data
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Check if calibration parameters are available
    if 'calibration_slope' in st.session_state and 'calibration_intercept' in st.session_state:
        # Upload raw data section
        raw_data_zip = interface.upload_raw_data_for_calibration()
        
        if raw_data_zip:
            # Get protein masses
            st.markdown(
                '<div class="section-header">📋 Enter Protein Information</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("""
            <div class="info-card">
                <p>Enter the mass (Da) for each protein in your raw data.</p>
                <p>Use the format: <code>ProteinName: Mass</code> (one per line)</p>
            </div>
            """, unsafe_allow_html=True)
            
            mass_input = st.text_area(
                "Protein Masses (Da)",
                placeholder="myoglobin: 17000\nBSA: 66400",
                help="Enter protein name and mass separated by colon, one per line"
            )
            
            # Parse mass input
            mass_dict = {}
            if mass_input:
                for line in mass_input.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':')
                        protein_name = parts[0].strip()
                        try:
                            mass = float(parts[1].strip())
                            mass_dict[protein_name] = mass
                        except ValueError:
                            st.warning(f"Could not parse mass for {protein_name}")
            
            if mass_dict:
                st.success(f"✅ Loaded masses for {len(mass_dict)} protein(s)")
                
                # Get the slope value from ln-ln calibration for tD'' calculation
                slope_lnln = st.session_state.get('lnln_slope', None)
                if slope_lnln is None:
                    st.error("⚠️ ln-ln slope not found. Please process calibration data with 'Show linear fit' enabled.")
                else:
                    # Get instrument parameters from session state
                    instrument_type = st.session_state.get('calibration_instrument_type', 'Synapt')
                    inject_time = st.session_state.get('calibration_inject_time', 0.0)
                    
                    # Display calibration parameters being used
                    st.info(f"📊 Using calibration parameters: ln-ln slope = {slope_lnln:.4f}, tD''-CCS slope = {st.session_state['calibration_slope']:.6f}, intercept = {st.session_state['calibration_intercept']:.4f}")
                    st.info(f"🔧 Instrument: {instrument_type}, Inject time: {inject_time:.3f} ms")
                    
                    # Generate calibration files button
                    if st.button("🔄 Generate Calibration Files", type="primary", key="generate_cal"):
                        try:
                            with st.spinner("Generating calibration files..."):
                                calibration_files = interface.generate_calibration_files(
                                    raw_data_zip,
                                    st.session_state['calibration_slope'],  # tD'' vs CCS slope
                                    st.session_state['calibration_intercept'],  # tD'' vs CCS intercept
                                    slope_lnln,  # ln-ln slope
                                    st.session_state['calibration_edc'],
                                    mass_dict,
                                    st.session_state['calibration_drift_gas'],
                                    instrument_type,
                                    inject_time
                                )
                        
                            if calibration_files:
                                st.success(f"✅ Generated calibration files for {len(calibration_files)} protein(s)")
                            
                                # Display and download each calibration file
                                st.markdown(
                                    '<div class="section-header">📥 Download Calibration Files</div>',
                                    unsafe_allow_html=True
                                )
                                
                                for protein_name, cal_df in calibration_files.items():
                                    # Show protein card
                                    st.markdown(
                                        f"""
                                        <div class="protein-card">
                                            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">🧪 {protein_name}</h4>
                                            <p style="margin: 0; color: #64748b;">
                                                <span class="metric-badge">{len(cal_df)} data points</span>
                                                <span class="metric-badge">{len(cal_df['Z'].unique())} charge state(s)</span>
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Preview data
                                    with st.expander(f"Preview {protein_name} data"):
                                        st.dataframe(cal_df.head(20))
                                    
                                    # Create download buffer
                                    buffer = dataframe_to_csv_buffer(cal_df)
                                    
                                    # Download button
                                    st.download_button(
                                        label=f"📊 Download {protein_name}.csv",
                                        data=buffer,
                                        file_name=f"{protein_name}.csv",
                                        mime="text/csv",
                                        key=f"download_cal_{protein_name}"
                                    )
                                
                                # Next steps info
                                st.markdown("""
                                <div class="info-card">
                                    <h4 style="color: #667eea; margin-top: 0;">📋 Next Steps</h4>
                                    <p>Your calibration files are ready! Each CSV file contains:</p>
                                    <ul>
                                        <li><strong>Z:</strong> Charge state</li>
                                        <li><strong>Drift:</strong> Drift time (ms)</li>
                                        <li><strong>CCS:</strong> Collision cross-section value (nm²)</li>
                                        <li><strong>CCS Std.Dev.:</strong> Standard deviation</li>
                                        <li><strong>Intensity:</strong> Signal intensity</li>
                                    </ul>
                                    <p>Go to <strong>'Get Calibrated Data'</strong> to use these calibration files.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ No valid data found in the uploaded ZIP file.")
                                
                        except Exception as e:
                            st.error(f"Error generating calibration files: {str(e)}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
            else:
                st.info("👆 Please enter protein masses to continue.")
    else:
        st.info("👆 Process calibration data above first to get slope and intercept values.")
    
    # References section always at bottom
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> Bush, M.F., Hall, Z., Giles, K., Hoyes, J., Robinson, C.V., Ruotolo, B.T., 2010. Collision Cross Sections of Proteins and Their Complexes: A Calibration Framework and Database for Gas-Phase Structural Biology. Anal. Chem. 82, 9557–9565. https://doi.org/10.1021/ac1022953
</p>
        <p><sup>2</sup> Richardson, K., Langridge, D., Dixit, S.M., Ruotolo, B.T., 2021. An Improved Calibration Approach for Traveling Wave Ion Mobility Spectrometry: Robust, High-Precision Collision Cross Sections. Anal. Chem. 93, 3542–3550. https://doi.org/10.1021/acs.analchem.0c04948
</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
