"""
IMSocio - Core library for processing native ion mobility mass spectrometry data.

This library provides tools for:
- Calibrant data processing and Gaussian fitting
- CCS (collision cross section) calculations
- Data visualization and analysis
- File I/O for various IMS data formats
- Input file generation for IMSCal software
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key classes/functions available at package level
from imsocio.calibration.processor import CalibrantProcessor
from imsocio.calibration.database import CalibrantDatabase, load_bush_database
from imsocio.io.readers import load_atd_data, load_mass_spectrum, is_valid_calibrant_file
from imsocio.io.writers import generate_zip_archive
from imsocio.extraction.input_generator import InputProcessor, InputParams, InputProcessingResult
from imsocio.extraction.output_processor import OutputFileProcessor, ProteinOutput, OutputProcessingResult
from imsocio.visualization import CCSDData, GaussianFitData, PlotSettings, CCSDPlotter

__all__ = [
    "CalibrantProcessor",
    "CalibrantDatabase", 
    "load_bush_database",
    "load_atd_data",
    "load_mass_spectrum",
    "is_valid_calibrant_file",
    "generate_zip_archive",
    "InputProcessor",
    "InputParams",
    "InputProcessingResult",
    "OutputFileProcessor",
    "ProteinOutput",
    "OutputProcessingResult",
    "CCSDData",
    "GaussianFitData",
    "PlotSettings",
    "CCSDPlotter",
]
