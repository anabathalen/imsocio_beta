# IMSocio Package Structure & Content Outline

## Overview
**IMSocio** (Ion Mobility Spectrometry Toolkit) is a Python package for processing and analyzing ion mobility mass spectrometry (IM-MS) data. It provides both a programmatic API and a Streamlit-based web interface for end-to-end workflows from raw data to publication-ready figures.

**Version:** 1.0.0  
**License:** MIT  
**Python Requirements:** >=3.8

---

## Core Package Structure (`imsocio/`)

### 1. `calibration/`
**Purpose:** Handle collision cross-section (CCS) calibration for TWIM-MS data

- **`database.py`** - Calibrant database management (Bush database)
- **`processor.py`** - CalibrantProcessor for processing calibration data
- **`utils.py`** - Utility functions for calibration workflows
- **Functions:** Load calibrants, fit arrival time distributions (ATDs), generate IMSCal input files

### 2. `extraction/`
**Purpose:** Generate input files for external tools and process their outputs

- **`input_generator.py`** - Create input files for TWIMExtract and other tools
- **`output_processor.py`** - Parse and process output from external analysis tools
- **Typical workflow:** Generate extraction parameters → Run TWIMExtract → Process results

### 3. `fitting/`
**Purpose:** Automated fitting of collision cross-section distributions (CCSDs)

- **`baseline_functions.py`** - Baseline correction functions (linear, polynomial, etc.)
- **`ccsd_processor.py`** - CCSD-specific data processing
- **`data_processor.py`** - General data preprocessing for fitting
- **`fitting_engine.py`** - Core curve-fitting engine using scipy
- **`parameter_estimation.py`** - Initial parameter guess algorithms
- **`parameter_manager.py`** - Manage fitting parameters and constraints
- **`peak_detection.py`** - Automatic peak detection in distributions
- **`peak_functions.py`** - Peak shape functions (Gaussian, Lorentzian, Voigt, etc.)
- **`result_analyzer.py`** - Analyze and report fitting results

**Key Features:**
- Multiple peak shapes (Gaussian, Lorentzian, Pseudo-Voigt)
- Baseline correction
- Multi-peak deconvolution
- Automated initial parameter estimation

### 4. `io/`
**Purpose:** Input/output operations for various file formats

- **`readers.py`** - Read CSV, TXT, and other data formats from MS instruments
- **`writers.py`** - Write calibration files, output results
- **`range_generator.py`** - Generate mass/CCS range files for targeted extraction

### 5. `processing/`
**Purpose:** Advanced data processing workflows

- **`drift_calibration.py`** - Drift time to CCS calibration
- **`esiprot.py`** - ESIProt implementation for charge state determination and MW calculation
- **`origami.py`** - Interface with ORIGAMI for aIMS processing
- **`visualization.py`** - Data visualization utilities

**Key Capabilities:**
- Charge state deconvolution
- Molecular weight calculation from low-resolution ESI data
- ORIGAMI data processing (CIU experiments)

### 6. `utils/`
**Purpose:** General utility functions

- **`data_tools.py`** - Data manipulation helpers (smoothing, normalization, etc.)
- **`origami.py`** - ORIGAMI-specific utilities

### 7. `visualization/`
**Purpose:** Plotting and figure generation

- **`ccsd.py`** - CCSD plotting functions (interactive Plotly charts)
- **`mass_spectrum.py`** - Mass spectrum visualization

---

## Streamlit Web Application (`streamlit_app/`)

### Main Application Files
- **`app.py`** - Main entry point and home page
- **`import_tools.py`** - File upload and import utilities
- **`styling.py`** - CSS styling and UI components

### Application Pages (`pages/`)

#### **Home (app.py)**
- Welcome page with overview of IMSocio features
- Navigation guide
- Quick start instructions
- **TODO:** Rename to "Home" (not "app")

#### **Page 1: Calibrate (`1_calibrate.py`)**
- Upload calibrant ATD data (zip file)
- Fit Gaussian curves to each calibrant
- Extract centroid arrival times
- Generate IMSCal calibration file (.dat)
- **Workflow:** Raw calibrants → Fitted ATDs → Calibration file

#### **Page 2: Generate Input Files (`2_generate_input_files.py`)**
- Create input files for TWIMExtract
- Define extraction parameters (m/z ranges, drift time windows)
- Batch processing capabilities

#### **Page 3: Process Output Files (`3_process_output_files.py`)**
- Parse TWIMExtract output files
- Format data for downstream analysis
- Quality control checks

#### **Page 4: Get Calibrated Data (`4_get_calibrated_data.py`)**
- Apply calibration to drift time data
- Convert drift times to CCS values
- Export calibrated datasets

#### **Page 5: Plot CCSDs (`5_plot_ccsds.py`)**
- Visualize collision cross-section distributions
- Interactive Plotly charts
- Compare multiple charge states or conditions
- Export publication-ready figures

#### **Page 6: Fit Data (`6_fit_data.py`)**
- Interactive CCSD fitting interface
- Choose peak shapes and baseline functions
- Adjust fitting parameters
- View residuals and goodness-of-fit metrics
- **TODO:** Ensure fully functional (Perdi's feedback)

#### **Page 7: Plot Mass Spectra (`7_plot_mass_spectra.py`)**
- Visualize mass spectra
- Annotate peaks
- Export figures

#### **Page 8: Generate Range Files (`8_generate_range_files.py`)**
- Create mass range files for targeted extraction
- Define m/z windows for specific proteins/complexes

#### **Page 9: ESIProt (`9_esiprot.py`)**
- Charge state determination from ESI mass spectra
- Molecular weight calculation
- Based on ESIProt algorithm (Winkler, 2009-2017)
- Handles low-resolution data

#### **Page 10: ORIGAMI CIU (`10_origami_ciu.py`)**
- Process collision-induced unfolding (CIU) data
- Interface with ORIGAMI software
- Visualize unfolding pathways

#### **Page 11: Alternative Calibration (`11_alternative_calibration.py`)**
- Alternative calibration methods
- Custom calibrant handling
- Manual calibration curve fitting

---

## Data Structure (`data/`)

### Sample Data (`data/sample_data/`)

#### **`calibration/`**
- `BSA/` - Bovine Serum Albumin calibrant files (charges 14-16)
- `myoglobin/` - Myoglobin calibrant files (charges 15-24)
- `polyalanine10/`, `polyalanine12/` - Polyalanine calibrants
- `test_calibration_results.csv` - Example calibration output

#### **`IM1/`**
- Example protein TWIM-MS data
- `all_proteins_scaled_calibrated.csv` - Processed data example
- `scale_factors.csv` - Scaling parameters
- `run_imscal.bat` - Example IMSCal execution script
- `mAb4/` - Monoclonal antibody dataset

### Reference Data
- **`bush.csv`** - Bush calibrant database (reference CCS values)

---

## Key Dependencies

### Core
- pandas >=2.0.0 - Data manipulation
- numpy >=1.24.0 - Numerical computing
- scipy >=1.10.0 - Scientific computing and optimization
- matplotlib >=3.7.0 - Static plotting
- plotly >=5.14.0 - Interactive plotting
- scikit-learn >=1.3.0 - Machine learning utilities

### Optional
- streamlit >=1.28.0 - Web interface (install with `pip install -e ".[app]"`)

---

## Typical Workflows

### 1. **Calibration Workflow**
1. Upload calibrant ATD files (Page 1)
2. Fit peaks and extract arrival times
3. Generate IMSCal calibration file
4. Run IMSCal externally
5. Import calibrated data (Page 4)

### 2. **Data Extraction & Processing**
1. Generate TWIMExtract input files (Page 2)
2. Run TWIMExtract externally
3. Process output files (Page 3)
4. Apply calibration (Page 4)

### 3. **Analysis & Visualization**
1. Plot CCSDs (Page 5)
2. Fit distributions (Page 6)
3. Extract populations and report metrics
4. Export figures for publication

### 4. **ESI Data Analysis**
1. Import mass spectrum
2. Run ESIProt (Page 9)
3. Determine charge states
4. Calculate molecular weight

### 5. **CIU Analysis**
1. Process ORIGAMI data (Page 10)
2. Visualize unfolding transitions
3. Compare different conditions

---

## External Tool Integration

IMSocio interfaces with:
- **TWIMExtract** - Extract arrival time distributions from Waters raw data
- **IMSCal** - Calibrate drift times to CCS values
- **ORIGAMI** - Process and visualize aIMS data

---

## Documentation Files

- **`README.md`** - Project overview and installation
- **`INSTALL.md`** - Detailed installation guide
- **`CHANGELOG.md`** - Version history
- **`LICENSE`** - MIT license
- **`manual_testing_guide.ipynb`** - Interactive testing notebook
- **`pyproject.toml`** - Python package configuration
- **`requirements.txt`** - Python dependencies
- **`paper.md`** / **`paper.bib`** - Academic paper and references

---

## Development Status

**Current Version:** 1.0.0 (Beta)  
**Status:** Near completion, pending final testing and documentation

**Remaining Tasks:**
- Complete page testing
- Finalize documentation
- Implement Perdi's suggestions
- Upload to GitHub
