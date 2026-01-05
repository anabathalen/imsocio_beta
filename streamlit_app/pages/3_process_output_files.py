"""
Streamlit page for processing IMSCal output files.

This page allows users to:
1. Upload a ZIP file containing IMSCal output files organized by sample
2. Extract calibrated CCS data from [CALIBRATED DATA] sections
3. Combine all charge states for each protein into a single dataframe
4. Download processed data as CSV files

Uses the IMSocio.extraction module for all core processing logic.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app import styling, import_tools
from imsocio.extraction import OutputFileProcessor
from imsocio.io.writers import dataframe_to_csv_buffer


class UI:
    """Streamlit UI components for output file processing."""
    
    @staticmethod
    def show_main_header():
        """Display the main page header."""
        st.markdown(
            """
            <div class="main-header">
                <h1>Process Output Files</h1>
                <p>Generate calibration files with a CCS value for every drift time for every charge state of every protein.</p>
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
                <p>Use this page to process the output files from IMSCal<sup>1</sup> to generate calibration files: CSV 
                files containing the CCS values corresponding to every timepoint in the ATD for all charge states of all
                samples. These files can then be used to calibrate your data in the next step.</p>
                <p>Separating out this step is particularly useful when you have performed multiple experiments on the 
                same protein (e.g., activated ion mobility experiments at different collision voltages). In such 
                cases, you only need to calibrate once using IMSCal, then use this tool to generate a calibration file
                which will allow you to process all your data under all experimental conditions.</p>
                <p>To use this tool:</p>
                <ul>
                    <li>Upload a ZIP file containing folders (called your sample names) with your IMSCal output files</li>
                    <li>Output files should be named <code>output_X.dat</code> where X is the charge state</li>
                </ul>
                <p><strong>Note:</strong> This step does not perform any fitting - it just extracts the calibration 
                data from IMSCal.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_upload_section():
        """Display upload section and return uploaded file."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">Upload Your Data</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.file_uploader(
            "Upload a zipped folder containing your output files",
            type="zip",
            help="Select a ZIP file containing folders with output_X.dat files"
        )
    
    @staticmethod
    def show_processing_status(files_processed: int, protein_count: int):
        """Display processing status."""
        st.markdown(
            f"""
            <div class="success-card">
                <strong>✅ Processing Complete!</strong><br>
                Found data for <span class="metric-badge">{protein_count} proteins</span>
                from <span class="metric-badge">{files_processed} files</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_download_section():
        """Display download section header."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📥 Download Calibration Files</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_protein_card(protein_name: str, combined_df: pd.DataFrame, num_files: int):
        """Display protein information card."""
        st.markdown(
            f"""
            <div class="protein-card">
                <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">🧪 {protein_name}</h4>
                <p style="margin: 0; color: #64748b;">
                    <span class="metric-badge">{len(combined_df)} data points</span>
                    <span class="metric-badge">{num_files} files combined</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_next_steps():
        """Display next steps information."""
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #667eea; margin-top: 0;">📋 Next Steps</h4>
                <p>Your calibration data is now ready for download. Each CSV file contains:</p>
                <ul>
                    <li><strong>Z:</strong> Charge state</li>
                    <li><strong>Drift:</strong> Drift time</li>
                    <li><strong>CCS:</strong> Collision cross-section value</li>
                    <li><strong>CCS Std.Dev.:</strong> Standard deviation</li>
                </ul>
                <p>Go to 'Get Calibrated Data' to finish your analysis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_warning_card():
        """Display warning when no valid data found."""
        st.markdown(
            """
            <div class="warning-card">
                <strong>⚠️ No Valid Data Found</strong><br>
                No valid output_X.dat files were found in the uploaded ZIP file.
                Please ensure your ZIP contains folders with properly formatted output files.
            </div>
            """,
            unsafe_allow_html=True
        )

        @staticmethod
        def show_help_section():
            """Display help information about expected file structure."""
            st.markdown(
                """
                <div class="info-card">
                    <h4>📖 Expected File Structure</h4>
                    <p>Your ZIP file should contain:</p>
                    <pre style="background: #f1f5f9; padding: 1rem; border-radius: 6px; font-size: 0.9rem; overflow-x: auto;"><code>your_data.zip/
    ├── Sample1/
    │   ├── output_1.dat
    │   ├── output_2.dat
    │   └── ...
    ├── Sample2/
    │   ├── output_1.dat
    │   └── ...
    └── ...</code></pre>
                </div>
                """,
                unsafe_allow_html=True
            )

    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown(
            """
            <div class="info-card">
                <h3>📚 References</h3>
                <p><sup>2</sup> Richardson, K., Langridge, D., Dixit, S.M., Ruotolo, B.T., 2021. An Improved Calibration Approach for Traveling Wave Ion Mobility Spectrometry: Robust, High-Precision Collision Cross Sections. Anal. Chem. 93, 3542–3550. https://doi.org/10.1021/acs.analchem.0c04948
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    @classmethod
    def show_help_section(cls):
        pass


def main():
    """Main application logic."""
    # Load custom CSS
    styling.load_custom_css()
    
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    # Show header and info
    UI.show_main_header()
    UI.show_info_card()
    
    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    # Get uploaded file
    uploaded_zip = UI.show_upload_section()
    
    # If no file uploaded, show references and exit
    if not uploaded_zip:
        UI.show_references()
        return
    
    # Process the uploaded ZIP file using imsocio library
    result = OutputFileProcessor.extract_protein_data(uploaded_zip)
    
    # Check if any data was found
    if result.protein_outputs:
        # Show processing status
        UI.show_processing_status(result.files_processed, len(result.protein_outputs))
        
        # Show download section
        UI.show_download_section()
        
        # For each protein, show card and download button
        for protein_name, protein_output in result.protein_outputs.items():
            # Combine all charge state data
            combined_df = OutputFileProcessor.combine_protein_data(protein_output)
            
            # Show protein card
            UI.show_protein_card(protein_name, combined_df, len(protein_output.dataframes))
            
            # Create download buffer
            buffer = dataframe_to_csv_buffer(combined_df)
            
            # Download button
            st.download_button(
                label=f"📊 Download {protein_name}.csv",
                data=buffer,
                file_name=f"{protein_name}.csv",
                mime="text/csv",
                key=f"download_{protein_name}"
            )
        
        # Show next steps
        UI.show_next_steps()
    else:
        # No valid data found
        UI.show_warning_card()
        UI.show_help_section()
    
    # Show references at the end
    UI.show_references()


if __name__ == "__main__":
    main()

