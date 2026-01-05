"""
Generate TWIMExtract Range Files
Streamlit page for generating range files for TWIMExtract.
"""

import sys
from pathlib import Path

# Add parent directory to path to import myutils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import tempfile
from typing import Tuple

# Import from imsocio package
from imsocio.io import (
    RangeParameters,
    CIURangeParameters,
    RangeFileGenerator,
    CIURangeFileGenerator,
    RangeFilePackager
)

# Import Streamlit UI helpers
from streamlit_app import styling, import_tools


class RangeFileInterface:
    """Streamlit interface for range file generation."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown(
            '<div class="main-header">'
            '<h1>Generate TWIMExtract Range Files</h1>'
            '<p>Generate range files for standard extraction or ORIGAMI experiments.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Use this page to generate range files for TWIMExtract<sup>1</sup>. Range files define the m/z, retention time, and drift time windows for data extraction.</p>
            <p><strong>Two modes available:</strong></p>
            <ul>
                <li><strong>Standard Mode:</strong> Generate range files for multiple charge states from a single dataset</li>
                <li><strong>ORIGAMI Mode:</strong> Generate range files for multiple collision voltages from data acquired using ORIGAMI<sup><em>MS</em>,</sup><sup>2</sup></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_mode_selection() -> str:
        """Get the mode selection from user.
        
        Returns:
            Selected mode: "Standard" or "ORIGAMI"
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📋 Mode Selection</h3>', unsafe_allow_html=True)
        
        mode = st.radio(
            "Select Mode:",
            options=["Standard", "ORIGAMI"],
            help="Standard: Generate files for multiple charge states. ORIGAMI: Generate files for multiple collision voltages."
        )
        
        if mode == "Standard":
            st.info("**Standard Mode:** Generate range files for different charge states of your protein.")
        else:
            st.info("**ORIGAMI Mode:** Generate range files for each collision voltage in your ORIGAMI experiment.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mode
    
    @staticmethod
    def get_protein_parameters() -> Tuple[float, float, Tuple[int, int], str]:
        """Get protein mass, m/z range size, charge range, and folder name from user.
        
        Returns:
            Tuple of (mass, mz_range_size, charge_range, folder_name)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Sample Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.number_input(
                "Mass (Da)",
                min_value=100.0, 
                value=15000.0, 
                step=100.0,
                help="Enter mass in Daltons"
            )
            
            mz_range_size = st.number_input(
                "m/z Range Size", 
                min_value=1.0, 
                value=50.0, 
                step=1.0,
                help="Total m/z window size (will be split equally around calculated m/z)"
            )
            
            folder_name = st.text_input(
                "Output Folder Name", 
                value="protein_ranges",
                help="Name for the folder containing the range files"
            )
        
        with col2:
            min_charge = st.number_input(
                "Minimum Charge State", 
                min_value=1, 
                value=10, 
                step=1,
                help="Lowest charge state to generate"
            )
            
            max_charge = st.number_input(
                "Maximum Charge State", 
                min_value=1, 
                value=20, 
                step=1,
                help="Highest charge state to generate"
            )
        
        # Validate charge range
        if min_charge > max_charge:
            st.error("Minimum charge state must be less than or equal to maximum charge state")
            st.markdown('</div>', unsafe_allow_html=True)
            return mass, mz_range_size, (min_charge, min_charge), folder_name
        
        # Validate folder name
        if not folder_name.strip():
            st.error("Please enter a folder name")
            folder_name = "ranges"
        
        charge_range = (min_charge, max_charge)
        
        # Show preview using the generator
        st.markdown("**Charge states and m/z values:**")
        temp_params = RangeParameters(
            mass=mass,
            mz_range_size=mz_range_size,
            charge_range=charge_range,
            rt_start=0.0,
            rt_end=100.0,
            dt_start=1,
            dt_end=200,
            folder_name=folder_name.strip()
        )
        temp_generator = RangeFileGenerator(temp_params)
        preview_data = temp_generator.generate_preview_data()
        st.table(preview_data)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mass, mz_range_size, charge_range, folder_name.strip()
    
    @staticmethod
    def get_ciu_protein_parameters() -> Tuple[float, float, int, str]:
        """Get protein mass, m/z range size, single charge state, and folder name for CIU mode.
        
        Returns:
            Tuple of (mass, mz_range_size, charge, folder_name)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Sample Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.number_input(
                "Mass (Da)",
                min_value=100.0, 
                value=15000.0, 
                step=100.0,
                help="Enter mass in Daltons"
            )
            
            mz_range_size = st.number_input(
                "m/z Range Size", 
                min_value=1.0, 
                value=50.0, 
                step=1.0,
                help="Total m/z window size (will be split equally around calculated m/z)"
            )
        
        with col2:
            charge = st.number_input(
                "Charge State", 
                min_value=1, 
                value=15, 
                step=1,
                help="Charge state for the protein"
            )
            
            folder_name = st.text_input(
                "Output Folder Name", 
                value="ciu_ranges",
                help="Name for the folder containing the range files"
            )
        
        # Validate folder name
        if not folder_name.strip():
            st.error("Please enter a folder name")
            folder_name = "ranges"
        
        # Show m/z preview
        mz = (mass + charge) / charge
        st.markdown(f"**m/z for charge {charge}+:** {mz:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mass, mz_range_size, charge, folder_name.strip()
    
    @staticmethod
    def get_ciu_parameters() -> Tuple[int, float, int, float, float, int]:
        """Get CIU-specific parameters.
        
        Returns:
            Tuple of (first_scan, first_voltage, scans_per_voltage, voltage_increment, seconds_per_scan, num_voltages)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">ORIGAMI Parameters</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <p>Specify how your collision voltages are organized in the Origami data file:</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_scan = st.number_input(
                "First Scan Number", 
                min_value=0, 
                value=1, 
                step=1,
                help="The scan number where your data starts"
            )
            
            first_voltage = st.number_input(
                "First Voltage (V)", 
                min_value=0.0, 
                value=10.0, 
                step=1.0,
                help="The first collision voltage value"
            )
            
            scans_per_voltage = st.number_input(
                "Scans per Voltage", 
                min_value=1, 
                value=100, 
                step=1,
                help="Number of scans collected at each collision voltage"
            )
        
        with col2:
            voltage_increment = st.number_input(
                "Voltage Increment (V)", 
                min_value=0.1, 
                value=5.0, 
                step=0.5,
                help="Increment between successive collision voltages"
            )
            
            seconds_per_scan = st.number_input(
                "Seconds per Scan", 
                min_value=0.01, 
                value=1.0, 
                step=0.1,
                format="%.2f",
                help="Duration of each scan in seconds"
            )
            
            num_voltages = st.number_input(
                "Number of Voltages", 
                min_value=1, 
                value=10, 
                step=1,
                help="Total number of different collision voltages"
            )
        
        # Show preview of voltages
        st.markdown("**Collision Voltages:**")
        voltages = [first_voltage + (i * voltage_increment) for i in range(min(num_voltages, 10))]
        voltage_str = ", ".join([f"{v:.0f}V" for v in voltages])
        if num_voltages > 10:
            voltage_str += f", ... (+{num_voltages - 10} more)"
        st.markdown(f"*{voltage_str}*")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return first_scan, first_voltage, scans_per_voltage, voltage_increment, seconds_per_scan, num_voltages
    
    @staticmethod
    def get_experimental_parameters() -> Tuple[float, float, int, int]:
        """Get retention time and drift time.
        
        Returns:
            Tuple of (rt_start, rt_end, dt_start, dt_end)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚙️ Experimental Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Retention Time Range**")
            rt_start = st.number_input(
                "RT Start (minutes)", 
                min_value=0.0, 
                value=0.0, 
                step=0.1,
                help="Start of retention time window"
            )
            
            rt_end = st.number_input(
                "RT End (minutes)", 
                min_value=0.1, 
                value=100.0, 
                step=0.1,
                help="End of retention time window"
            )
        
        with col2:
            st.write("**Drift Time Range**")
            dt_start = st.number_input(
                "DT Start (bins)", 
                min_value=1, 
                value=1, 
                step=1,
                help="Start of drift time window in bins"
            )
            
            dt_end = st.number_input(
                "DT End (bins)", 
                min_value=2, 
                value=200, 
                step=1,
                help="End of drift time window in bins"
            )
        
        # Validate ranges
        if rt_start >= rt_end:
            st.error("RT Start must be less than RT End")
        
        if dt_start >= dt_end:
            st.error("DT Start must be less than DT End")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return rt_start, rt_end, dt_start, dt_end
    
    @staticmethod
    def show_generation_results(result, params: RangeParameters):
        """Display the results of range file generation.
        
        Args:
            result: RangeFileResult object
            params: RangeParameters object
        """
        st.markdown(
            f"""
            <div class="success-card">
                ✅ <strong>Range File Generated!</strong><br>
                • Generated <strong>{len(result.generated_files)}</strong> range files<br>
                • Charge states: <strong>{params.charge_range[0]}+ to {params.charge_range[1]}+</strong><br>
                • m/z range: <strong>{params.mz_range_size} Da</strong> window per charge state<br>
                • Output folder: <strong>{params.folder_name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show detailed results
        with st.expander("📊 Detailed Information"):
            for charge in result.charge_states:
                mz = result.mz_values[charge]
                half_range = params.mz_range_size / 2.0
                st.write(f"**{params.folder_name}/range_{charge}.txt:**")
                st.write(f"  • Charge: {charge}+")
                st.write(f"  • Calculated m/z: {mz:.3f}")
                st.write(f"  • m/z range: {mz - half_range:.1f} - {mz + half_range:.1f}")
    
    @staticmethod
    def show_ciu_generation_results(result, params: CIURangeParameters):
        """Display the results of CIU range file generation.
        
        Args:
            result: RangeFileResult object
            params: CIURangeParameters object
        """
        st.markdown(
            f"""
            <div class="success-card">
                ✅ <strong>CIU Range Files Generated!</strong><br>
                • Generated <strong>{len(result.generated_files)}</strong> range files<br>
                • Collision voltages: <strong>{params.first_voltage:.0f}V to {params.first_voltage + (params.num_voltages - 1) * params.voltage_increment:.0f}V</strong><br>
                • Charge state: <strong>{params.charge}+</strong><br>
                • Output folder: <strong>{params.folder_name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show detailed results
        with st.expander("📊 Detailed Information"):
            mz = (params.mass + params.charge) / params.charge
            half_range = params.mz_range_size / 2.0
            
            for i, voltage in enumerate(result.charge_states):  # charge_states contains voltages for CIU
                scan_start = params.first_scan + (i * params.scans_per_voltage)
                scan_end = scan_start + params.scans_per_voltage - 1
                rt_start = (scan_start - params.first_scan) * params.seconds_per_scan / 60.0
                rt_end = (scan_end - params.first_scan + 1) * params.seconds_per_scan / 60.0
                
                st.write(f"**{params.folder_name}/{voltage:.0f}V.txt:**")
                st.write(f"  • Collision Voltage: {voltage:.0f} V")
                st.write(f"  • Scans: {scan_start}-{scan_end}")
                st.write(f"  • RT range: {rt_start:.2f}-{rt_end:.2f} minutes")
                st.write(f"  • m/z range: {mz - half_range:.1f} - {mz + half_range:.1f}")
    
    @staticmethod
    def show_download_section(zip_buffer, params: RangeParameters):
        """Show download button for the generated range files.
        
        Args:
            zip_buffer: BytesIO buffer containing ZIP file
            params: RangeParameters object
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📥 Download Range Files</h3>', unsafe_allow_html=True)
        
        filename = RangeFilePackager.get_zip_filename(params.folder_name)
        
        st.download_button(
            label="📦 Download Range Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=filename,
            mime="application/zip",
            help="Download ZIP file containing all generated range files"
        )
    
    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> Haynes, S.E., Polasky, D.A., Dixit, S.M., Majmudar, J.D., Neeson, K., Ruotolo, B.T., Martin, B.R., 2017. Variable-Velocity Traveling-Wave Ion Mobility Separation Enhancing Peak Capacity for Data-Independent Acquisition Proteomics. Anal. Chem. 89, 5669–5672. https://doi.org/10.1021/acs.analchem.7b00112
            </p>
            <p><sup>2</sup> Migas, L.G., France, A.P., Bellina, B., Barran, P.E., 2018. ORIGAMI: A software suite for activated ion mobility mass spectrometry (aIM-MS) applied to multimeric protein assemblies. International Journal of Mass Spectrometry, Richard Smith Honor Issue 427, 20–28. https://doi.org/10.1016/j.ijms.2017.08.014
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Load custom styling
    styling.load_custom_css()
    
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    # Show header
    RangeFileInterface.show_header()
    
    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    # Step 1: Get mode selection
    mode = RangeFileInterface.get_mode_selection()
    
    if mode == "Standard":
        # Standard mode workflow
        # Step 2: Get protein parameters
        mass, mz_range_size, charge_range, folder_name = RangeFileInterface.get_protein_parameters()
        
        if charge_range[0] > charge_range[1]:
            st.stop()
        
        # Step 3: Get experimental parameters
        rt_start, rt_end, dt_start, dt_end = RangeFileInterface.get_experimental_parameters()
        
        if rt_start >= rt_end or dt_start >= dt_end:
            st.warning("Please check ranges.")
            st.stop()
        
        # Create parameters object
        params = RangeParameters(
            mass=mass,
            mz_range_size=mz_range_size,
            charge_range=charge_range,
            rt_start=rt_start,
            rt_end=rt_end,
            dt_start=dt_start,
            dt_end=dt_end,
            folder_name=folder_name
        )
        
        # Step 4: Generate button
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🚀 Generate Files</h3>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Range Files", type="primary"):
            with st.spinner("Generating range files..."):
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Generate range files
                    generator = RangeFileGenerator(params)
                    result = generator.generate_all_files(temp_dir)
                    
                    # Show results
                    RangeFileInterface.show_generation_results(result, params)
                    
                    # Generate ZIP
                    zip_buffer = RangeFilePackager.create_zip(temp_dir, result, folder_name)
                    
                    # Show download button
                    RangeFileInterface.show_download_section(zip_buffer, params)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # ORIGAMI mode
        # CIU mode workflow
        # Step 2: Get CIU protein parameters
        mass, mz_range_size, charge, folder_name = RangeFileInterface.get_ciu_protein_parameters()
        
        # Step 3: Get CIU parameters
        first_scan, first_voltage, scans_per_voltage, voltage_increment, seconds_per_scan, num_voltages = RangeFileInterface.get_ciu_parameters()
        
        # Step 4: Get drift time parameters
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚙️ Drift Time Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            dt_start = st.number_input(
                "DT Start (bins)", 
                min_value=1, 
                value=1, 
                step=1,
                help="Start of drift time window in bins"
            )
        
        with col2:
            dt_end = st.number_input(
                "DT End (bins)", 
                min_value=2, 
                value=200, 
                step=1,
                help="End of drift time window in bins"
            )
        
        if dt_start >= dt_end:
            st.error("DT Start must be less than DT End")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if dt_start >= dt_end:
            st.warning("Please check drift time range.")
            st.stop()
        
        # Create CIU parameters object
        ciu_params = CIURangeParameters(
            mass=mass,
            charge=charge,
            mz_range_size=mz_range_size,
            dt_start=dt_start,
            dt_end=dt_end,
            first_scan=first_scan,
            first_voltage=first_voltage,
            scans_per_voltage=scans_per_voltage,
            voltage_increment=voltage_increment,
            seconds_per_scan=seconds_per_scan,
            num_voltages=num_voltages,
            folder_name=folder_name
        )
        
        # Show preview
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Preview</h3>', unsafe_allow_html=True)
        
        ciu_generator = CIURangeFileGenerator(ciu_params)
        preview_data = ciu_generator.generate_preview_data()
        st.table(preview_data)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Generate button
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🚀 Generate Files</h3>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate ORIGAMI Range Files", type="primary"):
            with st.spinner("Generating CIU range files..."):
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Generate range files
                    result = ciu_generator.generate_all_files(temp_dir)
                    
                    # Show results
                    RangeFileInterface.show_ciu_generation_results(result, ciu_params)
                    
                    # Generate ZIP
                    zip_buffer = RangeFilePackager.create_zip(temp_dir, result, folder_name)
                    
                    # Show download button
                    RangeFileInterface.show_download_section(zip_buffer, ciu_params)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add references section
    RangeFileInterface.show_references()


if __name__ == "__main__":
    main()

