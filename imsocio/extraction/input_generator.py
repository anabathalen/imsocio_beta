"""
Input file generation for IMSCal processing.

This module handles the creation of .dat input files from sample data folders.
It processes multiple samples, applies drift time corrections, and generates 
properly formatted files for IMSCal calibration software.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from ..io.readers import is_valid_calibrant_file, extract_charge_state_from_filename, load_atd_data
from ..io.writers import write_imscal_dat


@dataclass
class InputParams:
    """
    Parameters for input file processing.
    
    Attributes:
        drift_mode: Either 'synapt' or 'cyclic' - determines drift time adjustment
        inject_time: Injection time in milliseconds (only used for cyclic mode)
        sample_mass_map: Dictionary mapping sample names to their molecular masses
    """
    drift_mode: str
    inject_time: float
    sample_mass_map: Dict[str, float]


@dataclass
class InputProcessingResult:
    """
    Results from processing input files.
    
    Attributes:
        processed_files: Dictionary mapping sample names to lists of successfully processed files
        failed_files: Dictionary mapping sample names to lists of files that failed processing
        sample_paths: Dictionary mapping sample names to their output directory paths
    """
    processed_files: Dict[str, List[str]] = field(default_factory=dict)
    failed_files: Dict[str, List[str]] = field(default_factory=dict)
    sample_paths: Dict[str, Path] = field(default_factory=dict)


class InputProcessor:
    """
    Processes sample data folders to generate IMSCal input files.
    
    This class handles:
    - Loading ATD data from multiple charge state files
    - Applying drift time corrections (for cyclic IMS instruments)
    - Generating .dat files with proper formatting for IMSCal software
    - Tracking processing success/failure
    
    Example:
        >>> params = InputParams(
        ...     drift_mode='cyclic',
        ...     inject_time=0.5,
        ...     sample_mass_map={'protein_A': 15000.0, 'protein_B': 20000.0}
        ... )
        >>> processor = InputProcessor(base_path=Path('data'), params=params)
        >>> result = processor.process_all(['protein_A', 'protein_B'])
    """
    
    def __init__(self, base_path: Path, params: InputParams):
        """
        Initialize the input processor.
        
        Args:
            base_path: Root directory containing sample folders
            params: Processing parameters (drift mode, inject time, sample masses)
        """
        self.base_path = Path(base_path)
        self.params = params
    
    def process_sample_folder(self, sample_name: str, sample_folder: Path) -> tuple[List[str], List[str]]:
        """
        Process a single sample folder to generate .dat files.
        
        This method:
        1. Finds all valid data files in the folder
        2. Loads ATD data from each file
        3. Extracts charge state from filename
        4. Applies drift time corrections if needed
        5. Creates input_{charge}.dat files
        
        Args:
            sample_name: Name of the sample being processed
            sample_folder: Path to the folder containing data files
            
        Returns:
            Tuple of (processed_files, failed_files) - lists of filenames
        """
        processed_files = []
        failed_files = []
        
        # Get the mass for this sample
        mass = self.params.sample_mass_map.get(sample_name, 0.0)
        if mass == 0.0:
            return processed_files, failed_files
        
        # Find all valid data files
        all_files = list(sample_folder.glob('*'))
        data_files = [f for f in all_files if is_valid_calibrant_file(f)]
        
        if not data_files:
            return processed_files, failed_files
        
        # Process each data file
        for file_path in data_files:
            try:
                # Extract charge state from filename
                charge = extract_charge_state_from_filename(file_path.name)
                if charge is None:
                    failed_files.append(file_path.name)
                    continue
                
                # Load the ATD data
                drift_times, intensities = load_atd_data(file_path)
                
                # Apply drift time correction for cyclic IMS
                if self.params.drift_mode.lower() == "cyclic":
                    drift_times = drift_times - self.params.inject_time
                    # Ensure no negative drift times
                    drift_times = np.maximum(drift_times, 0)
                
                # Create DataFrame for output
                df = pd.DataFrame({
                    'index': range(len(drift_times)),
                    'mass': mass,
                    'charge': charge,
                    'intensity': intensities,
                    'drift_time': drift_times
                })
                
                # Write the .dat file
                output_filename = f"input_{charge}.dat"
                output_path = sample_folder / output_filename
                write_imscal_dat(df, output_path)
                
                processed_files.append(file_path.name)
                
            except Exception as e:
                failed_files.append(f"{file_path.name} - {str(e)}")
        
        return processed_files, failed_files
    
    def process_all(self, sample_folders: List[str]) -> InputProcessingResult:
        """
        Process all sample folders.
        
        Args:
            sample_folders: List of sample folder names to process
            
        Returns:
            InputProcessingResult containing all processing outcomes
        """
        result = InputProcessingResult()
        
        for sample_name in sample_folders:
            sample_path = self.base_path / sample_name
            
            if not sample_path.exists() or not sample_path.is_dir():
                result.failed_files[sample_name] = [f"Folder not found: {sample_name}"]
                continue
            
            # Process this sample folder
            processed, failed = self.process_sample_folder(sample_name, sample_path)
            
            result.processed_files[sample_name] = processed
            result.failed_files[sample_name] = failed
            result.sample_paths[sample_name] = sample_path
        
        return result
