# imsocio

**Ion Mobility Mass Spectrometry Data Processing Toolkit**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

imsocio is a Python toolkit for processing ion mobility mass spectrometry data. It interfaces with  established tools like TWIMExtract^1^ and IMSCal^2^ to provide a start-to-finish analysis workflow from raw data to publication-ready figures.

### Features

- TWIMS CCS Calibration
- Automated CCS Distribution Fitting
- aIMS Processing (ORIGAMI)
- Visualisation tools
- Streamlit-based web interface

## Statement of Need

Native ion mobility mass spectrometry (nIM-MS) is one of a variety of tools available for characterisation of protein structure and function. Analysis of nIM-MS data is complex, often involving several disconnected software packages. Several tools exist to automate this workflow, including UniDec^3^ (mass spectrum deconvolution), ORIGAMI^4^ (nIM-MS processing toolkit) and CIUSuite^5^ (all-in-one calibration and processing of gas-phase unfolding datasets), but at present there is no single application that incorporates calibration, processing and visualisation with customisation for a variety of use-cases. To expand the adoption of nIM-MS in structural biology, a streamlined analysis workflow is required.

Here we present `IMSocio`: a Python package for processing and visualising IM-MS data, designed to serve as a toolkit for getting from raw IM-MS data to publication quality figures. The accompanying `streamlit` web application provides a graphical user interface. 

## Installation

### Core Library Only

Install just the imsocio library without the web interface:

```bash
# Clone the repository
git clone https://github.com/anabathalen/imsocius
cd imsocio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core library only
pip install -e .
```

### With Streamlit Web App

To install with the optional Streamlit web interface:

```bash
# Install with app dependencies
pip install -e ".[app]"
```

### For Development

To install with development tools (testing, linting, etc.):

```bash
# Install with all dependencies
pip install -e ".[all]"
```

### Core Dependencies

The core library requires:
- Python >=3.8
- pandas >=2.0.0
- numpy >=1.24.0
- scipy >=1.10.0
- matplotlib >=3.7.0
- plotly >=5.14.0
- scikit-learn >=1.3.0

**Optional:**
- streamlit >=1.28.0 (for web interface, install with `[app]`)

## Usage

### Python API

Use imsocio as a Python library in your scripts:

```python
from imsocio import CalibrantProcessor, CalibrantDatabase, load_bush_database

# Load calibrant database
db = load_bush_database()

# Process calibrant data
processor = CalibrantProcessor(db, min_r2=0.9)
# ... your analysis code
```

### Web Interface

Launch the interactive web application (requires `[app]` installation):

```bash
streamlit run homepage.py
# or
streamlit run streamlit_app/homepage.py
```

This opens a browser interface with 10 tools:

1. **Calibrate**: Process calibrant ATDs and generate IMSCal calibration files
2. **Generate Input Files**: Create input files for IMSCal from your data
3. **Process Output Files**:  Interpret IMSCal output files to generate arrival time ➡ CCS conversions for your samples
4. **Get Calibrated Data**: Generate calibrated and scaled datasets
5. **Plot CCSDs**: Visualise collision cross section distributions
6. **Fit Data**: Peak detection and fitting (fit CCS distributions)
7. **Plot Mass Spectra**: Plot mass spectra with `matplotlib`
8. **Generate Range Files**: Create TWIMExtract range files for automated extraction
9. **ESIProt**: Quick native ESI deconvolution using ESIProt^6^
10. **ORIGAMI**: Generate aIMS fingerprints and stacked plots

## Package Structure

```
imsocio/
├── imsocio/              # Core library (can be used independently)
│   ├── calibration/          # Calibrant processing and CCS calibration
│   ├── extraction/           # Input/output file processing
│   ├── fitting/              # Peak detection and fitting algorithms
│   ├── io/                   # File readers and writers
│   ├── processing/           # Data processing utilities (ORIGAMI, ESIProt, etc.)
│   ├── visualization/        # Plotting and visualization tools
│   └── utils/                # General utility functions
├── streamlit_app/            # Optional web interface (requires streamlit)
│   ├── pages/                # Individual Streamlit page modules
│   ├── static/               # CSS and static assets
│   ├── homepage.py           # Main Streamlit app
│   ├── styling.py            # UI styling utilities
│   └── import_tools.py       # File upload and import helpers
└── data/                     # Calibrant databases and reference data
```

### Python API Examples

imsocio can be used programmatically in your own scripts:

```python
from imsocio import CalibrantProcessor, CalibrantDatabase, load_bush_database
from imsocio.fitting import PeakDetector, FittingEngine
from imsocio.visualization import CCSDPlotter

# Example 1: Calibrant processing
db = load_bush_database()
processor = CalibrantProcessor(db, min_r2=0.9)
# ... process your calibrant data

# Example 2: Peak fitting
detector = PeakDetector()
peaks = detector.detect_peaks(ccs_values, intensities)

fitter = FittingEngine()
results = fitter.fit_peaks(peaks, function="gaussian")

# Example 3: Visualization
plotter = CCSDPlotter()
plotter.plot_ccsd(ccs_values, intensities, fitted_data=results)
plotter.save("output.png", dpi=300)
```

## Documentation

Documentation is available in the following files:

- **README.md**: This file - overview, installation, and quick start guide
- **INSTALL.md**: Detailed installation instructions
- **CHANGELOG.md**: Version history and release notes
- **LICENSE**: MIT license terms

For API documentation, all modules include comprehensive docstrings that can be accessed via Python's `help()` function or by reading the source code.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

To be added on publication.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/anabathalen/imsocio/issues)
- **Email**: ana.bathalen@manchester.ac.uk
- **Website**: https://github.com/anabathalen/imsocio

## References

1. Haynes, S.E., Polasky, D.A., Dixit, S.M., Majmudar, J.D., Neeson, K., Ruotolo, B.T., Martin, B.R., 2017. Variable-Velocity Traveling-Wave Ion Mobility Separation Enhancing Peak Capacity for Data-Independent Acquisition Proteomics. Anal. Chem. 89, 5669–5672. https://doi.org/10.1021/acs.analchem.7b00112

2. Richardson, K., Langridge, D., Dixit, S.M., Ruotolo, B.T., 2021. An Improved Calibration Approach for Traveling Wave Ion Mobility Spectrometry: Robust, High-Precision Collision Cross Sections. Anal. Chem. 93, 3542–3550. https://doi.org/10.1021/acs.analchem.0c04948

3. Marty, M.T., Baldwin, A.J., Marklund, E.G., Hochberg, G.K.A., Benesch, J.L.P., Robinson, C.V., 2015. Bayesian Deconvolution of Mass and Ion Mobility Spectra: From Binary Interactions to Polydisperse Ensembles. Anal. Chem. 87, 4370–4376. https://doi.org/10.1021/acs.analchem.5b00140

4. Migas, L.G., France, A.P., Bellina, B., Barran, P.E., 2018. ORIGAMI: A software suite for activated ion mobility mass spectrometry (aIM-MS) applied to multimeric protein assemblies. International Journal of Mass Spectrometry 427, 20–28. https://doi.org/10.1016/j.ijms.2017.08.014

5. Eschweiler, J.D., Rabuck-Gibbons, J.N., Tian, Y., Ruotolo, B.T., 2015. CIUSuite: A Quantitative Analysis Package for Collision Induced Unfolding Measurements of Gas-Phase Protein Ions. Anal. Chem. 87, 11516–11522. https://doi.org/10.1021/acs.analchem.5b03292
