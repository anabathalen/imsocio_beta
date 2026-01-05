# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.0] - 2026-01-05

### Added
- Initial release of imsocio toolkit
- Core calibration module with drift time to CCS conversion
- Peak detection and fitting engine with multiple peak functions (Gaussian, pseudo-Voigt, asymmetric Gaussian)
- Charge state deconvolution using ESIProt algorithm
- ORIGAMI-style CIU (Collision-Induced Unfolding) analysis
- Interactive Streamlit web interface for all analysis tools
- Comprehensive I/O utilities for reading/writing calibration and mass spectrum data
- Visualization tools for publication-ready figures
- Bush calibrant database integration
- Support for multiple data formats (CSV, text files)
- Parameter estimation and optimization for peak fitting
- CCSD (Charge State Distribution) analysis tools
- Baseline correction functionality
- Range-based data extraction utilities

### Features

#### Calibration Tools
- Automated drift time to CCS calibration
- Multi-charge state calibration support
- Configurable R² thresholds and filtering
- Power law and linear calibration methods

#### Peak Analysis
- Multiple peak function support
- Automated peak detection with customizable thresholds
- Parameter bounds and constraints
- Goodness-of-fit reporting (R², RMSE, reduced χ²)

#### Visualization
- Interactive Plotly-based plots
- Static matplotlib figures for publications
- Customizable plot styling and formatting
- CIU fingerprint heatmaps
- Charge state distribution plots

#### Web Interface
- 10 integrated analysis tools accessible via web browser
- Real-time parameter adjustment
- File upload/download capabilities
- Session state management
- Responsive design with custom CSS styling

### Dependencies
- Python >=3.8
- pandas >=2.0.0
- numpy >=1.24.0
- scipy >=1.10.0
- matplotlib >=3.7.0
- plotly >=5.14.0
- scikit-learn >=1.3.0
- streamlit >=1.28.0 (optional, for web interface)

### Documentation
- Comprehensive README with installation and usage instructions
- Installation guide (INSTALL.md)
- MIT License

---

## [Unreleased]

### Planned
- Comprehensive test suite
- Extended documentation with API reference
- Additional calibrant databases
- Enhanced error handling and validation
- Performance optimizations for large datasets

---

**Note**: For upgrade instructions between versions, see the migration guide (when available).
