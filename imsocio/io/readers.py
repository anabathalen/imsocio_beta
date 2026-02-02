"""
File reading utilities for IMS data.

This module handles loading data from various file formats used in ion mobility
mass spectrometry, including:
- Text files (.txt) from MassLynx
- CSV files from TWIMExtract
"""

import re
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def is_valid_calibrant_file(file_path: Path) -> bool:
    """
    Check if a file is a valid calibrant data file.
    
    This function determines if a file should be processed as calibrant data
    by checking:
    1. File extension (.txt, .csv, or _raw without extension)
    2. Filename pattern indicating charge state
    
    For .txt files: The filename should start with a number (the charge state)
    Example: "24.txt" for charge state 24
    
    For .csv files: The filename should contain a charge state pattern like:
    - "range_24.txt" 
    - "range_24_raw"
    - "DT_sample_range_24.txt_raw"
    
    For TWIM extract files (ending with _raw): The filename should contain:
    - "#range_24" (e.g., "DT_20260120_sample_#range_24.txt_raw")
    - Other charge state patterns
    
    Args:
        file_path: Path object pointing to the file to check
        
    Returns:
        True if the file is valid for processing, False otherwise
        
    Examples:
        >>> is_valid_calibrant_file(Path("24.txt"))
        True
        >>> is_valid_calibrant_file(Path("range_24.csv"))
        True
        >>> is_valid_calibrant_file(Path("notes.txt"))
        False
    """
    # Validate input
    if not isinstance(file_path, Path):
        logger.warning(f"Invalid input type: expected Path, got {type(file_path)}")
        return False
    
    # Check if file exists
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    # Check if it's actually a file (not a directory)
    if not file_path.is_file():
        logger.warning(f"Path is not a file: {file_path}")
        return False
    
    # Skip hidden/system files (like .DS_Store on Mac)
    if file_path.name.startswith('.'):
        return False
    
    # Get filename for pattern matching
    filename = file_path.name
    
    # Check if it's a CSV file
    if file_path.suffix.lower() == '.csv':
        # Get filename without the .csv extension
        filename_without_ext = file_path.stem
        
        # First check if the filename is just a number (e.g., "14.csv")
        try:
            charge = int(filename_without_ext)
            if 1 <= charge <= 200:
                return True
        except ValueError:
            pass  # Not just a number, try patterns
        
        # These patterns match different ways charge states appear in filenames
        patterns = [
            r'range_(\d+)\.txt',   # Matches "range_24.txt"
            r'range_(\d+)_',        # Matches "range_24_"
            r'_(\d+)\.txt_raw',     # Matches "_24.txt_raw"
            r'_(\d+)_raw$',         # Matches "_24_raw" at the end
            r'charge(\d+)',         # Matches "charge15" or "charge_15"
            r'_(\d+)$'              # Matches "_24" at the end
        ]
        
        # Try each pattern to see if we can find a charge state
        for pattern in patterns:
            if re.search(pattern, filename_without_ext):
                return True
        
        logger.debug(f"CSV file does not match charge state pattern: {file_path.name}")
        return False
    
    # Check if it's a .txt file
    elif file_path.suffix == '.txt':
        # For .txt files, the filename should start with a digit
        # Example: "24.txt" for charge state 24
        is_valid = file_path.name[0].isdigit()
        if not is_valid:
            logger.debug(f"TXT file does not start with digit: {file_path.name}")
        return is_valid
    
    # Check if it's a TWIM extract file (ends with _raw but has no extension)
    elif filename.endswith('_raw'):
        # These files are comma-separated like CSV files
        # Pattern matching for charge state in the filename
        patterns = [
            r'#range_(\d+)',        # Matches "#range_24" (ORIGAMI/TWIM extract format)
            r'range_(\d+)\.txt',    # Matches "range_24.txt"
            r'range_(\d+)_',        # Matches "range_24_"
            r'_(\d+)\.txt_raw',     # Matches "_24.txt_raw"
            r'charge(\d+)',         # Matches "charge15" or "charge_15"
        ]
        
        # Try each pattern to see if we can find a charge state
        for pattern in patterns:
            if re.search(pattern, filename):
                return True
        
        logger.debug(f"TWIM extract file does not match charge state pattern: {file_path.name}")
        return False
    
    # If it's neither .txt nor .csv nor _raw file, it's not valid
    else:
        logger.debug(f"Unsupported file extension: {file_path.suffix}")
        return False


def extract_charge_state_from_filename(filename: str) -> Optional[int]:
    """
    Extract the charge state from a calibrant filename.
    
    This function looks for numeric patterns in filenames that indicate
    the charge state of the ion.
    
    For .txt files: "24.txt" -> 24
    For .csv files: "DT_sample_range_24.txt_raw.csv" -> 24
    
    Args:
        filename: The filename (with or without extension) to parse
        
    Returns:
        The charge state as an integer, or None if not found
        
    Examples:
        >>> extract_charge_state_from_filename("24.txt")
        24
        >>> extract_charge_state_from_filename("range_18_raw.csv")
        18
        >>> extract_charge_state_from_filename("unknown.txt")
        None
    """
    # Remove file extension to work with just the name
    file_path = Path(filename)
    filename_without_ext = file_path.stem
    
    # First, try the simple case: filename is just a number
    # Example: "24.txt" -> stem is "24"
    try:
        charge = int(filename_without_ext)
        # Validate charge state is reasonable (1-200)
        if 1 <= charge <= 200:
            return charge
        else:
            logger.warning(f"Charge state {charge} outside valid range (1-200): {filename}")
            return None
    except ValueError:
        pass  # Not just a number, try regex patterns
    
    # Patterns to find charge state in more complex filenames
    patterns = [
        r'#range_(\d+)',         # Matches "#range_24" (ORIGAMI format)
        r'range_(\d+)\.txt',     # Matches "range_24.txt"
        r'range_(\d+)_',         # Matches "range_24_"
        r'_(\d+)\.txt_raw',      # Matches "_24.txt_raw"
        r'_(\d+)_raw$',          # Matches "_24_raw" at end
        r'charge(\d+)',          # Matches "charge15" or "charge_15"
        r'_(\d+)$'               # Matches "_24" at end
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, filename_without_ext)
        if match:
            # match.group(1) gets the number inside the parentheses in the pattern
            charge = int(match.group(1))
            # Validate charge state is reasonable
            if 1 <= charge <= 200:
                return charge
            else:
                logger.warning(f"Charge state {charge} outside valid range (1-200): {filename}")
                return None
    
    # If no pattern matched, log and return None
    logger.debug(f"Could not extract charge state from filename: {filename}")
    return None


def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load drift time and intensity data from an ATD (arrival time distribution) file.
    
    This function supports three formats:
    1. .txt files: Two-column space-separated data from MassLynx
    2. .csv files: Comma-separated data from TWIMExtract (can include # comments)
    3. _raw files: Comma-separated TWIM extract files (can include # comments)
    
    Args:
        file_path: Path to the ATD data file
        
    Returns:
        A tuple of two numpy arrays: (drift_time, intensity)
        - drift_time: Array of drift time values (usually in ms)
        - intensity: Array of intensity values corresponding to each drift time
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If no valid data is found in the file or file format is invalid
        IOError: If the file cannot be read
        
    Examples:
        >>> drift_time, intensity = load_atd_data(Path("24.txt"))
        >>> print(f"Loaded {len(drift_time)} data points")
        Loaded 500 data points
    """
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a file
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    # Check file size (warn if empty or suspiciously small/large)
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path}")
    if file_size > 100_000_000:  # 100 MB
        logger.warning(f"Large file detected ({file_size / 1_000_000:.1f} MB): {file_path}")
    
    # Handle CSV files and TWIM extract files (from TWIMExtract - ends with _raw)
    if file_path.suffix.lower() == '.csv' or file_path.name.endswith('_raw'):
        data_rows = []
        invalid_lines = 0
        
        # Read file line by line
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip comment lines (start with # or $)
                    if line.startswith('#') or line.startswith('$'):
                        continue
                    
                    # Try to parse the data
                    try:
                        # Split by comma
                        values = line.split(',')
                        
                        # We need at least 2 values (drift time and intensity)
                        if len(values) >= 2:
                            drift_time = float(values[0])
                            intensity = float(values[1])
                            
                            # Check for NaN or Inf values
                            if np.isnan(drift_time) or np.isnan(intensity):
                                logger.debug(f"NaN value at line {line_num}, skipping")
                                invalid_lines += 1
                                continue
                            if np.isinf(drift_time) or np.isinf(intensity):
                                logger.debug(f"Inf value at line {line_num}, skipping")
                                invalid_lines += 1
                                continue
                            
                            # Validate data is reasonable
                            if drift_time < 0:
                                logger.warning(f"Negative drift time at line {line_num}: {drift_time}")
                                invalid_lines += 1
                                continue
                            if intensity < 0:
                                logger.warning(f"Negative intensity at line {line_num}: {intensity}")
                                invalid_lines += 1
                                continue
                            
                            data_rows.append([drift_time, intensity])
                        else:
                            invalid_lines += 1
                            
                    except (ValueError, IndexError) as e:
                        # If line can't be parsed, count it
                        invalid_lines += 1
                        if invalid_lines <= 5:  # Only log first few errors
                            logger.debug(f"Could not parse line {line_num} in {file_path.name}: {e}")
                        continue
        
        except (IOError, OSError) as e:
            raise IOError(f"Error reading file {file_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error (not valid UTF-8): {file_path}: {e}")
        
        # Warn if many invalid lines
        if invalid_lines > 10:
            logger.warning(f"Skipped {invalid_lines} invalid lines in {file_path.name}")
        
        # Check if we got any data
        if not data_rows:
            raise ValueError(f"No valid data found in CSV file: {file_path}")
        
        # Check if we have enough data points
        if len(data_rows) < 10:
            logger.warning(f"Very few data points ({len(data_rows)}) in {file_path.name}")
        
        # Convert list to numpy array
        data = np.array(data_rows)
        
        # Return as two separate arrays
        return data[:, 0], data[:, 1]
    
    # Handle .txt files (from MassLynx)
    else:
        try:
            # Try to load the data
            # First try space/tab separated (default)
            try:
                data = np.loadtxt(file_path)
            except ValueError:
                # If that fails, try comma-separated (some .txt files use commas)
                data = np.loadtxt(file_path, delimiter=',')
            
            # Handle single-row files (np.loadtxt returns 1D array)
            if data.ndim == 1:
                if len(data) >= 2:
                    data = data.reshape(1, -1)
                    logger.warning(f"File contains only one data point: {file_path.name}")
                else:
                    raise ValueError(f"Insufficient columns in single-row file: {file_path}")
            
            # Validate shape
            if data.ndim != 2:
                raise ValueError(f"Expected 2D data, got {data.ndim}D array")
            
            if data.shape[1] < 2:
                raise ValueError(f"Expected at least 2 columns, got {data.shape[1]}")
            
            # Check if we have any rows
            if len(data) == 0:
                raise ValueError(f"No data rows found in file: {file_path}")
            
            # Handle NaN values
            if np.any(np.isnan(data)):
                nan_count = np.isnan(data).any(axis=1).sum()
                logger.warning(f"Found {nan_count} rows with NaN values in {file_path.name} - removing")
                data = data[~np.isnan(data).any(axis=1)]
                if len(data) == 0:
                    raise ValueError(f"All data rows contain NaN values: {file_path}")
            
            # Handle Inf values
            if np.any(np.isinf(data)):
                inf_count = np.isinf(data).any(axis=1).sum()
                logger.warning(f"Found {inf_count} rows with Inf values in {file_path.name} - removing")
                data = data[~np.isinf(data).any(axis=1)]
                if len(data) == 0:
                    raise ValueError(f"All data rows contain Inf values: {file_path}")
            
            # Check for reasonable number of data points
            if len(data) < 10:
                logger.warning(f"Very few data points ({len(data)}) in {file_path.name}")
            
            # Check for negative values (warn but don't remove)
            if np.any(data[:, 0] < 0):
                neg_count = (data[:, 0] < 0).sum()
                logger.warning(f"Found {neg_count} negative drift times in {file_path.name}")
            if np.any(data[:, 1] < 0):
                neg_count = (data[:, 1] < 0).sum()
                logger.warning(f"Found {neg_count} negative intensities in {file_path.name}")
            
            return data[:, 0], data[:, 1]
            
        except ValueError as e:
            # Check if it's a np.loadtxt error (common: mixed types, wrong delimiters)
            if "could not convert" in str(e).lower():
                raise ValueError(f"File contains non-numeric data or wrong delimiter: {file_path}")
            raise ValueError(f"Error parsing TXT file {file_path}: {e}")
        except (IOError, OSError) as e:
            raise IOError(f"Error reading file {file_path}: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            raise ValueError(f"Unexpected error loading TXT file {file_path}: {type(e).__name__}: {e}")


def load_multiple_atd_files(folder_path: Path) -> dict:
    """
    Load all valid ATD files from a folder.
    
    This is a convenience function that processes an entire folder
    of calibrant files at once.
    
    Args:
        folder_path: Path to folder containing ATD files
        
    Returns:
        Dictionary mapping charge states to (drift_time, intensity) tuples
        Example: {24: (drift_array, intensity_array), 25: (...), ...}
        
    Raises:
        FileNotFoundError: If folder_path does not exist
        ValueError: If folder_path is not a directory
        
    Examples:
        >>> data = load_multiple_atd_files(Path("myoglobin"))
        >>> print(f"Found charge states: {list(data.keys())}")
        Found charge states: [24, 25, 26, 27]
    """
    # Validate input
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    results = {}
    skipped_files = []
    failed_files = []
    
    # Iterate through all files in the folder
    for file_path in folder_path.iterdir():
        # Skip directories
        if file_path.is_dir():
            continue
        
        # Skip if not a valid calibrant file
        if not is_valid_calibrant_file(file_path):
            skipped_files.append(file_path.name)
            continue
        
        # Extract charge state from filename
        charge_state = extract_charge_state_from_filename(file_path.name)
        if charge_state is None:
            skipped_files.append(file_path.name)
            continue
        
        # Try to load the data
        try:
            drift_time, intensity = load_atd_data(file_path)
            
            # Check for duplicate charge states
            if charge_state in results:
                logger.warning(f"Duplicate charge state {charge_state}: {file_path.name} (keeping first)")
                continue
            
            results[charge_state] = (drift_time, intensity)
            logger.info(f"Loaded {file_path.name} (charge state {charge_state}, {len(drift_time)} points)")
            
        except (FileNotFoundError, ValueError, IOError) as e:
            # Log specific error but continue processing other files
            logger.error(f"Failed to load {file_path.name}: {e}")
            failed_files.append(file_path.name)
            continue
    
    # Log summary
    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} files (invalid format or no charge state)")
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {', '.join(failed_files)}")
    
    if not results:
        logger.warning(f"No valid data files found in {folder_path}")
    else:
        logger.info(f"Successfully loaded {len(results)} charge states from {folder_path}")
    
    return results


def load_mass_spectrum(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mass spectrum file (m/z vs intensity).
    
    This function supports:
    1. Tab-separated files (most common from MassLynx)
    2. Comma-separated files
    3. Space-separated files
    4. Files with comment lines (# or $)
    
    Args:
        file_path: Path to the mass spectrum file
        
    Returns:
        A tuple of two numpy arrays: (mz, intensity)
        - mz: Array of m/z values
        - intensity: Array of intensity values corresponding to each m/z
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If no valid data is found or file format is invalid
        
    Examples:
        >>> mz, intensity = load_mass_spectrum(Path("mass_spectrum.txt"))
        >>> print(f"m/z range: {mz.min():.1f} - {mz.max():.1f}")
        m/z range: 500.0 - 5000.0
    """
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    data_rows = []
    invalid_lines = 0
    
    # Try reading as text file line by line for maximum compatibility
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip comment lines (start with # or $)
                if line.startswith('#') or line.startswith('$'):
                    continue
                
                # Try to parse the data
                try:
                    # Try different delimiters in order of likelihood
                    values = None
                    if '\t' in line:
                        values = line.split('\t')
                    elif ',' in line:
                        values = line.split(',')
                    else:
                        values = line.split()
                    
                    # We need at least 2 values (m/z and intensity)
                    if len(values) >= 2:
                        mz = float(values[0])
                        intensity = float(values[1])
                        
                        # Check for NaN or Inf values
                        if np.isnan(mz) or np.isnan(intensity):
                            invalid_lines += 1
                            continue
                        if np.isinf(mz) or np.isinf(intensity):
                            invalid_lines += 1
                            continue
                        
                        # Validate data is reasonable for mass spectrum
                        if mz < 0:
                            logger.warning(f"Negative m/z at line {line_num}: {mz}")
                            invalid_lines += 1
                            continue
                        if intensity < 0:
                            logger.warning(f"Negative intensity at line {line_num}: {intensity}")
                            invalid_lines += 1
                            continue
                        
                        data_rows.append([mz, intensity])
                    else:
                        invalid_lines += 1
                        
                except (ValueError, IndexError) as e:
                    invalid_lines += 1
                    if invalid_lines <= 5:  # Only log first few errors
                        logger.debug(f"Could not parse line {line_num} in {file_path.name}: {e}")
                    continue
    
    except (IOError, OSError) as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error (not valid UTF-8): {file_path}: {e}")
    
    # Warn if many invalid lines
    if invalid_lines > 10:
        logger.warning(f"Skipped {invalid_lines} invalid lines in {file_path.name}")
    
    # Check if we got any data
    if not data_rows:
        raise ValueError(f"No valid data found in mass spectrum file: {file_path}")
    
    # Check if we have enough data points
    if len(data_rows) < 10:
        logger.warning(f"Very few data points ({len(data_rows)}) in mass spectrum {file_path.name}")
    
    # Convert list to numpy array
    data = np.array(data_rows)
    
    # Return as two separate arrays
    return data[:, 0], data[:, 1]
