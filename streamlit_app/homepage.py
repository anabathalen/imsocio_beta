"""
IMSocio - Ion Mobility Mass Spectrometry Data Analysis Toolkit

All-in-one workflow for calibrating, processing and analyzing ion mobility-mass spectrometry data.
"""

import streamlit as st
from streamlit_app import styling

# Page configuration
st.set_page_config(
    page_title="IMSocio",
    page_icon="️🧰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

def main():
    """Main home page function."""
    styling.load_custom_css()

    # Main header
    st.markdown(
        '<div class="main-header">'
        '<h1>🧰 IMSocio v0.0</h1>'
        '<p>Ion Mobility Mass Spectrometry Data Processing Toolkit</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # About section
    st.markdown("""
    <div class="info-card">
        <h3>❓About IMSocio</h3>
        <p>IMSocio is a Python toolkit for processing ion mobility mass spectrometry data. 
        It interfaces with established tools like TWIMExtract<sup>1</sup> and IMSCal<sup>2</sup> to provide start-to-finish 
        analysis workflows from raw data to publication-ready figures.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>TWIMS CCS calibration</li>
            <li>Automated CCSD fitting</li>
            <li>aIMS processing (ORIGAMI)</li>
            <li>Visualization tools</li>
            <li>Streamlit-based web interface for accessibility (⬅ this is what you are using now!)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Navigation info
    st.markdown("""
    <div class="info-card">
        <h3>🔧 Available Tools</h3>
        <p>Use the sidebar to navigate between processing tools:</p>
        <ul>
            <li><strong>Calibrate:</strong> Process calibrant ATDs and generate IMSCal calibration files</li>
            <li><strong>Generate Input Files:</strong> Create input files for IMSCal from your data</li>
            <li><strong>Process Output Files:</strong> Interpret IMSCal output files to generate arrival time ➡ CCS conversions for your samples</li>
            <li><strong>Get Calibrated Data:</strong> Generate calibrated and scaled datasets</li>
            <li><strong>Plot CCSDs:</strong> Visualize collision cross section distributions</li>
            <li><strong>Fit Data:</strong> Peak detection and fitting (fit CCSDs)</li>
            <li><strong>Mass Spectrum Plotting:</strong> Plot mass spectra with <code>matplotlib</code></li>
            <li><strong>Range File Generator:</strong> Create TWIMExtract range files for automated extraction</li>
            <li><strong>ESIProt:</strong> Quick native ESI deconvolution using ESIProt<sup>3</sup></li>
            <li><strong>ORIGAMI:</strong> Generate aIMS fingerprints and stacked plots</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Getting Started
    st.markdown("""
    <div class="info-card">
        <h3>🏁 Getting Started</h3>
        <p><strong>You will need:</strong></p>
        <ul>
            <li>Some TWIMS data for a set of calibrants and samples</li>
            <li>Either TWIMExtract or MassLynx for data extraction</li>
            <li>IMSCal for CCS calibration (steps 1-3)</li>
        </ul>
        <p><strong>Workflow:</strong></p>
        <ol>
            <li>Extract ATDs for calibrants using TWIMExtract or MassLynx</li>
            <li>Generate IMSCal calibration file using <em>calibrate</em></li>
            <li>Generate IMSCal input files using <em>generate input files</em></li>
            <li>Run IMSCal</li>
            <li>Process IMSCal outputs in <em>process output files</em></li>
            <li>Generate final calibrated dataset <em>get calibrated data</em></li>
            <li>Process and visualize your data: <em>plot ccsds</em>, <em>fit data</em>, <em>plot mass spectra</em>, <em>origami ciu</em></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # References section
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> Haynes, S.E., Polasky, D.A., Dixit, S.M., Majmudar, J.D., Neeson, K., Ruotolo, B.T., Martin, B.R., 2017. Variable-Velocity Traveling-Wave Ion Mobility Separation Enhancing Peak Capacity for Data-Independent Acquisition Proteomics. Anal. Chem. 89, 5669–5672. https://doi.org/10.1021/acs.analchem.7b00112
</p>
        <p><sup>2</sup> Richardson, K., Langridge, D., Dixit, S.M., Ruotolo, B.T., 2021. An Improved Calibration Approach for Traveling Wave Ion Mobility Spectrometry: Robust, High-Precision Collision Cross Sections. Anal. Chem. 93, 3542–3550. https://doi.org/10.1021/acs.analchem.0c04948
</p>
        <p><sup>3</sup> Winkler, R., 2010. ESIprot: a universal tool for charge state determination and molecular weight calculation of proteins from electrospray ionization mass spectrometry data. Rapid Commun Mass Spectrom 24, 285–294. https://doi.org/10.1002/rcm.4384
</p>
    </div>
    """, unsafe_allow_html=True)

    # Citation
    st.markdown("""
    <div class="info-card">
        <h3>📖 Citation</h3>
        <p>If you use IMSocio in your research, please cite:</p>
        <p><em>[ADD HERE]</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
