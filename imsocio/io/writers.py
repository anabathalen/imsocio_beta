"""
File writing utilities for IMS data.

This module handles writing data to various output formats used in
ion mobility mass spectrometry workflows.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

# Set up logger for this module
logger = logging.getLogger(__name__)


def write_imscal_dat(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    velocity: Optional[float] = None,
    voltage: Optional[float] = None,
    pressure: Optional[float] = None,
    length: Optional[float] = None
) -> str:
    """
    Write data in IMSCal .dat format.
    
    IMSCal is a calibration tool that can accept two different formats:
    1. Calibration format: Header with instrument parameters + calibrant data
    2. Input format: Simple data table without header
    
    For calibration (with header), provide all instrument parameters and a DataFrame with:
    - 'protein': protein name
    - 'charge state': integer charge state
    - 'mass': protein mass in Da
    - 'calibrant_value': CCS value in nm² (will be converted to Å^2)
    - 'drift time': measured drift time in ms
    
    For input files (no header), just provide a DataFrame with:
    - 'index': row index
    - 'mass': protein mass in Da
    - 'charge': integer charge state
    - 'intensity': signal intensity
    - 'drift_time': measured drift time in ms
    
    Args:
        data: DataFrame with data to write
        output_path: Where to save the .dat file (optional, if None will only return string)
        velocity: Wave velocity in m/s (optional, for calibration format)
        voltage: Wave height in V (optional, for calibration format)
        pressure: IMS pressure in mbar (optional, for calibration format)
        length: Drift cell length in m (optional, for calibration format)
        
    Returns:
        The .dat file content as a string
        
    Raises:
        ValueError: If DataFrame is empty, missing required columns, or contains invalid data
        IOError: If file cannot be written
        
    Example (calibration format):
        >>> df = pd.DataFrame({
        ...     'protein': ['myoglobin', 'myoglobin'],
        ...     'charge state': [24, 25],
        ...     'mass': [16952.3, 16952.3],
        ...     'calibrant_value': [31.2, 29.8],
        ...     'drift time': [5.23, 4.87]
        ... })
        >>> content = write_imscal_dat(df, Path("out.dat"), 281.0, 20.0, 1.63, 0.98)
        
    Example (input format):
        >>> df = pd.DataFrame({
        ...     'index': [0, 1, 2],
        ...     'mass': [16952.3, 16952.3, 16952.3],
        ...     'charge': [24, 24, 24],
        ...     'intensity': [100, 200, 150],
        ...     'drift_time': [5.1, 5.2, 5.3]
        ... })
        >>> content = write_imscal_dat(df, Path("input.dat"))
    """
    # Validate input DataFrame
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    if len(data) == 0:
        raise ValueError("DataFrame is empty - no data to write")
    
    # Detect format based on columns and parameters
    has_instrument_params = all(p is not None for p in [velocity, voltage, pressure, length])
    is_calibration_format = 'protein' in data.columns and 'calibrant_value' in data.columns
    
    if has_instrument_params and is_calibration_format:
        # Validate required columns for calibration format
        required_cols = ['protein', 'charge state', 'mass', 'calibrant_value', 'drift time']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for calibration format: {missing_cols}")
        
        # Validate instrument parameters
        if velocity <= 0:
            raise ValueError(f"Wave velocity must be positive, got {velocity}")
        if voltage <= 0:
            raise ValueError(f"Wave height must be positive, got {voltage}")
        if pressure <= 0:
            raise ValueError(f"Pressure must be positive, got {pressure}")
        if length <= 0:
            raise ValueError(f"Drift cell length must be positive, got {length}")
        
        # Check for NaN or Inf in critical columns
        for col in required_cols:
            if data[col].isna().any():
                nan_count = data[col].isna().sum()
                raise ValueError(f"Column '{col}' contains {nan_count} NaN values")
            
            # Check numeric columns for Inf
            if col in ['charge state', 'mass', 'calibrant_value', 'drift time']:
                if np.isinf(data[col]).any():
                    raise ValueError(f"Column '{col}' contains Inf values")
        
        # Validate charge states
        if (data['charge state'] <= 0).any():
            raise ValueError("Charge states must be positive integers")
        if (data['charge state'] > 200).any():
            logger.warning("Charge states > 200 detected - this may be unusual")
        
        # Validate masses
        if (data['mass'] <= 0).any():
            raise ValueError("Mass values must be positive")
        
        # Validate CCS values
        if (data['calibrant_value'] <= 0).any():
            raise ValueError("CCS calibrant values must be positive")
        
        # Validate drift times
        if (data['drift time'] < 0).any():
            raise ValueError("Drift times must be non-negative")
        
        # Calibration format with header
        header = (
            f"# length {length}\n"
            f"# velocity {velocity}\n"
            f"# voltage {voltage}\n"
            f"# pressure {pressure}\n"
        )
        
        content_lines = []
        for _, row in data.iterrows():
            protein = row['protein']
            charge_state = int(row['charge state'])
            mass = row['mass']
            calibrant_value = row['calibrant_value'] * 100  # nm² to Å^2
            drift_time = row['drift time']
            
            line = f"{protein}_{charge_state} {mass} {charge_state} {calibrant_value} {drift_time}"
            content_lines.append(line)
        
        full_content = header + "\n".join(content_lines)
    else:
        # Validate required columns for input format
        required_cols = ['index', 'mass', 'charge', 'intensity', 'drift_time']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for input format: {missing_cols}")
        
        # Check for NaN or Inf in critical columns
        for col in required_cols:
            if data[col].isna().any():
                nan_count = data[col].isna().sum()
                raise ValueError(f"Column '{col}' contains {nan_count} NaN values")
            
            if col != 'index':  # Don't check index for Inf
                if np.isinf(data[col]).any():
                    raise ValueError(f"Column '{col}' contains Inf values")
        
        # Validate charge states
        if (data['charge'] <= 0).any():
            raise ValueError("Charge states must be positive integers")
        
        # Validate masses
        if (data['mass'] <= 0).any():
            raise ValueError("Mass values must be positive")
        
        # Validate drift times
        if (data['drift_time'] < 0).any():
            raise ValueError("Drift times must be non-negative")
        
        # Validate intensities (can be zero but not negative)
        if (data['intensity'] < 0).any():
            raise ValueError("Intensity values cannot be negative")
        
        # Input format without header - simple space-delimited table
        content_lines = []
        for _, row in data.iterrows():
            # Format: index mass charge intensity drift_time
            line = f"{int(row['index'])} {row['mass']} {int(row['charge'])} {row['intensity']} {row['drift_time']}"
            content_lines.append(line)
        
        full_content = "\n".join(content_lines)
    
    # Write to file only if output_path is provided
    if output_path is not None:
        try:
            # Ensure parent directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(full_content)
            
            logger.info(f"Successfully wrote {len(data)} rows to {output_path}")
            
        except (IOError, OSError) as e:
            raise IOError(f"Failed to write file {output_path}: {e}")
    
    return full_content


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a CSV-formatted string.
    
    This is useful for creating downloadable CSV content in web apps
    without actually writing to disk.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV-formatted string
        
    Raises:
        ValueError: If DataFrame is None or empty
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> csv_str = dataframe_to_csv_string(df)
        >>> print(csv_str)
        A,B
        1,3
        2,4
    """
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if len(df) == 0:
        logger.warning("Converting empty DataFrame to CSV")
    
    # Use StringIO to write CSV to a string instead of a file
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def write_calibration_results_csv(
    calibrant_data: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Write calibration results to a CSV file.
    
    This creates a human-readable CSV with all the fitting results.
    
    Args:
        calibrant_data: DataFrame with calibration results
        output_path: Where to save the CSV file
        
    Raises:
        ValueError: If DataFrame is None or empty
        IOError: If file cannot be written
        
    Example:
        >>> write_calibration_results_csv(results_df, Path("results.csv"))
    """
    # Validate input
    if calibrant_data is None or not isinstance(calibrant_data, pd.DataFrame):
        raise ValueError("calibrant_data must be a pandas DataFrame")
    
    if len(calibrant_data) == 0:
        raise ValueError("DataFrame is empty - no data to write")
    
    try:
        # Ensure parent directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        calibrant_data.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote calibration results to {output_path}")
        
    except (IOError, OSError) as e:
        raise IOError(f"Failed to write CSV file {output_path}: {e}")


def generate_zip_archive(sample_paths: Dict[str, Path]) -> io.BytesIO:
    """
    Create a ZIP archive containing .dat files from multiple sample folders.
    
    This function is used to bundle processed data files for download. It searches
    each sample folder for .dat files and packages them into a single ZIP file
    with a folder structure that preserves the sample organization.
    
    Args:
        sample_paths: Dictionary mapping sample names to their folder paths
        
    Returns:
        BytesIO buffer containing the ZIP file data (ready for download)
        
    Raises:
        ValueError: If sample_paths is empty or contains invalid paths
        IOError: If files cannot be read or ZIP cannot be created
        
    Example:
        >>> sample_paths = {
        ...     'myoglobin': Path('temp/myoglobin'),
        ...     'ubiquitin': Path('temp/ubiquitin')
        ... }
        >>> zip_buffer = generate_zip_archive(sample_paths)
        >>> # Can now use zip_buffer.getvalue() for downloads
    """
    # Validate input
    if not sample_paths or len(sample_paths) == 0:
        raise ValueError("sample_paths dictionary is empty")
    
    # Validate all paths exist
    invalid_paths = []
    for sample_name, sample_path in sample_paths.items():
        sample_path = Path(sample_path)
        if not sample_path.exists():
            invalid_paths.append(f"{sample_name}: {sample_path} (does not exist)")
        elif not sample_path.is_dir():
            invalid_paths.append(f"{sample_name}: {sample_path} (not a directory)")
    
    if invalid_paths:
        raise ValueError(f"Invalid sample paths:\n" + "\n".join(invalid_paths))
    
    zip_buffer = io.BytesIO()
    files_added = 0
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for sample_name, sample_path in sample_paths.items():
                # Find all .dat files in this sample folder
                dat_files = list(Path(sample_path).glob('*.dat'))
                
                if not dat_files:
                    logger.warning(f"No .dat files found in {sample_name} folder: {sample_path}")
                
                for dat_file in dat_files:
                    try:
                        # Archive path: sample_name/filename.dat
                        archive_path = f"{sample_name}/{dat_file.name}"
                        zipf.write(dat_file, arcname=archive_path)
                        files_added += 1
                    except (IOError, OSError) as e:
                        logger.error(f"Failed to add {dat_file} to ZIP: {e}")
                        # Continue with other files
        
        if files_added == 0:
            logger.warning("No files were added to the ZIP archive")
        else:
            logger.info(f"Created ZIP archive with {files_added} files from {len(sample_paths)} folders")
        
    except Exception as e:
        raise IOError(f"Failed to create ZIP archive: {e}")
    
    # Reset buffer position to beginning for reading
    zip_buffer.seek(0)
    return zip_buffer


def dataframe_to_csv_buffer(df: pd.DataFrame) -> io.BytesIO:
    """
    Convert a DataFrame to a BytesIO buffer containing CSV data.
    
    This is useful for creating downloadable CSV files in web apps
    without writing to disk.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        BytesIO buffer containing the CSV data (ready for download)
        
    Raises:
        ValueError: If DataFrame is None or empty
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> buffer = dataframe_to_csv_buffer(df)
        >>> # Can use buffer for download buttons or file operations
    """
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if len(df) == 0:
        logger.warning("Converting empty DataFrame to CSV buffer")
    
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def generate_imscal_batch_file(
    output_path: Path,
    calibration_file: str,
    input_dir: str,
    output_dir: str,
    velocity: float,
    voltage: float,
    pressure: float,
    length: float,
    imscal_bin_path: str = r"C:\Program Files\IMSCal19\bin",
    lambda_value: float = 0.012,
    temperature: float = 300.0,
    accuracy: float = 2.0,
    instrument_type: str = "synapt"
) -> Path:
    """
    Generate a Windows batch file to run IMSCal on input files.
    
    This creates a .bat script that processes all input_*.dat files through
    IMSCal's TWaveCalibrate.exe with the specified instrument parameters.
    
    The batch file uses relative paths from the batch file location and
    automatically navigates to the correct directory.
    
    Args:
        output_path: Where to save the .bat file
        calibration_file: Relative path to calibration .dat file (from notebook root)
        input_dir: Relative path to directory containing input_*.dat files
        output_dir: Relative path where output_*.dat files will be saved
        velocity: Wave velocity in m/s
        voltage: Wave height in V
        pressure: IMS pressure in mbar
        length: Drift cell length in m
        imscal_bin_path: Path to IMSCal installation bin folder
        lambda_value: Reduced mass parameter for drift gas (default: 0.012 for nitrogen)
        temperature: Temperature in Kelvin (default: 300)
        accuracy: IMSCal accuracy parameter (default: 2.0)
        instrument_type: "synapt" or "cyclic"
        
    Returns:
        Path to the created batch file
        
    Raises:
        ValueError: If parameters are invalid
        IOError: If batch file cannot be written
        
    Example:
        >>> batch_path = generate_imscal_batch_file(
        ...     output_path=Path("data/run_imscal.bat"),
        ...     calibration_file="data/calibration/calib.dat",
        ...     input_dir="data/samples/mAb4",
        ...     output_dir="data/samples/mAb4",
        ...     velocity=800,
        ...     voltage=15.0,
        ...     pressure=3,
        ...     length=0.25
        ... )
    """
    # Validate parameters
    if velocity <= 0:
        raise ValueError(f"Velocity must be positive, got {velocity}")
    if voltage <= 0:
        raise ValueError(f"Voltage must be positive, got {voltage}")
    if pressure <= 0:
        raise ValueError(f"Pressure must be positive, got {pressure}")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")
    if lambda_value <= 0:
        raise ValueError(f"Lambda must be positive, got {lambda_value}")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Convert paths to Windows format (backslashes)
    calib_win = str(Path(calibration_file)).replace('/', '\\')
    input_win = str(Path(input_dir)).replace('/', '\\')
    output_win = str(Path(output_dir)).replace('/', '\\')
    
    # Calculate how many levels up to go from batch file location to notebook root
    output_path = Path(output_path)
    # Count path components to determine navigation depth
    # e.g., "data/sample_data/IM1/run.bat" -> 3 levels up
    depth = len(output_path.parent.parts)
    nav_path = "\\".join([".." for _ in range(depth)])
    
    # Generate batch file content
    batch_content = f"""@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM IMSCal Batch Processing Script
REM Auto-generated by imsocio
REM ============================================================================

REM Change to notebook directory to use relative paths
REM Batch file is in: {output_path.parent}
REM Need to go up {depth} levels to reach notebook root
cd /d "%~dp0{nav_path}"

REM === Paths ===
set "BIN={imscal_bin_path}"
set "INPUT_DIR={input_win}"
set "OUTPUT_DIR={output_win}"

REM === Calibration reference file ===
set "CALIB={calib_win}"

REM === Instrument Parameters ===
REM Wave velocity: {velocity} m/s
REM Wave height: {voltage} V
REM Pressure: {pressure} mbar
REM Length: {length} m
REM Instrument type: {instrument_type}

echo ============================================================================
echo IMSCal Batch Processing
echo ============================================================================
echo.
echo Calibration file: %CALIB%
echo Input directory:  %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%
echo.
echo Instrument Parameters:
echo   Length:   {length} m
echo   Lambda:   {lambda_value}
echo   Velocity: {velocity} m/s
echo   Voltage:  {voltage} V
echo   Pressure: {pressure} mbar
echo   Temp:     {temperature} K
echo   Accuracy: {accuracy}
echo.
echo ============================================================================
echo.

REM Check if calibration file exists
if not exist "%CALIB%" (
    echo ERROR: Calibration file not found: %CALIB%
    pause
    exit /b 1
)

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found: %INPUT_DIR%
    pause
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM === Loop through all input_*.dat files ===
set COUNT=0
for %%F in ("%INPUT_DIR%\\input_*.dat") do (
    echo Processing %%~nxF ...

    set "BASENAME=%%~nF"
    set "NUM=!BASENAME:input_=!"

    "%BIN%\\TWaveCalibrate.exe" ^
        -ref "%CALIB%" ^
        -input "%%F" ^
        -output "%OUTPUT_DIR%\\output_!NUM!.dat" ^
        -length {length} ^
        -lambda {lambda_value} ^
        -velocity {velocity} ^
        -voltage {voltage} ^
        -pressure {pressure} ^
        -temp {temperature} ^
        -accuracy {accuracy}
    
    if !errorlevel! equ 0 (
        echo   SUCCESS: output_!NUM!.dat created
    ) else (
        echo   ERROR: Failed to process %%~nxF
    )
    echo.
    
    set /a COUNT+=1
)

echo ============================================================================
echo Processing complete! Processed !COUNT! files.
echo ============================================================================
pause
"""
    
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(batch_content)
        
        logger.info(f"Successfully created IMSCal batch file: {output_path}")
        
    except (IOError, OSError) as e:
        raise IOError(f"Failed to write batch file {output_path}: {e}")
    
    return output_path
