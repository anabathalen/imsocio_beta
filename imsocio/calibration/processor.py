"""
Gaussian fitting for calibrant arrival time distributions.

This module handles the processing of calibrant ATD files, including
Gaussian fitting to extract peak positions (drift times).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from imsocio.io.readers import (
    load_atd_data,
    is_valid_calibrant_file,
    extract_charge_state_from_filename
)
from imsocio.calibration.database import CalibrantDatabase


@dataclass
class GaussianFitResult:
    """
    Results from fitting a Gaussian to an ATD.
    
    Attributes:
        amplitude: Height of the Gaussian peak
        apex: Center position (drift time in ms where peak maximum occurs)
        std_dev: Standard deviation (width) of the Gaussian
        r_squared: R² goodness of fit (0-1, higher is better)
        fitted_values: The fitted Gaussian curve (same length as input data)
        drift_time: Original drift time data
        intensity: Original intensity data
    """
    amplitude: float
    apex: float
    std_dev: float
    r_squared: float
    fitted_values: np.ndarray
    drift_time: np.ndarray
    intensity: np.ndarray


@dataclass
class CalibrantMeasurement:
    """
    Complete measurement for a single calibrant charge state.
    
    This combines the experimental measurement (drift time from Gaussian fit)
    with literature values from the database (CCS).
    
    Attributes:
        protein: Protein name (e.g., 'myoglobin')
        mass: Protein mass in Da
        charge_state: Integer charge state
        drift_time: Measured drift time in ms (from Gaussian fit apex)
        r_squared: Quality of Gaussian fit
        ccs_literature: Literature CCS value in nm²
        fit_result: Complete Gaussian fitting results
        filename: Original filename that was processed
    """
    protein: str
    mass: float
    charge_state: int
    drift_time: float
    r_squared: float
    ccs_literature: float
    fit_result: Optional[GaussianFitResult] = None
    filename: Optional[str] = None


class CalibrantProcessor:
    """
    Process calibrant ATD files to extract drift times via Gaussian fitting.
    
    This class handles:
    1. Loading ATD data files
    2. Fitting Gaussians to extract drift times
    3. Looking up literature CCS values
    4. Quality control (R² thresholds)
    
    Example:
        >>> from imsocio.calibration import CalibrantDatabase, load_bush_database
        >>> bush_df = load_bush_database()
        >>> db = CalibrantDatabase(bush_df)
        >>> processor = CalibrantProcessor(db, min_r2=0.9)
        >>> result = processor.process_file(Path("24.txt"), "myoglobin")
        >>> if result:
        ...     print(f"Drift time: {result.drift_time:.2f} ms")
        Drift time: 5.23 ms
    """
    
    def __init__(
        self,
        calibrant_db: CalibrantDatabase,
        min_r2: float = 0.9,
        fitting_function = None
    ):
        """
        Initialize the calibrant processor.
        
        Args:
            calibrant_db: CalibrantDatabase instance for looking up CCS values
            min_r2: Minimum R² value to accept a fit (default 0.9)
            fitting_function: Optional custom Gaussian fitting function.
                            If None, will use imsocio.utils.fit_gaussian_with_retries
        """
        self.calibrant_db = calibrant_db
        self.min_r2 = min_r2
        
        # Try to import fitting function if not provided
        if fitting_function is None:
            try:
                from imsocio.utils import fit_gaussian_with_retries
                self.fitting_function = fit_gaussian_with_retries
            except ImportError:
                raise ImportError(
                    "Could not import fit_gaussian_with_retries. "
                    "Please provide a fitting_function."
                )
        else:
            self.fitting_function = fitting_function
    
    def _fit_gaussian(
        self,
        drift_time: np.ndarray,
        intensity: np.ndarray
    ) -> Optional[GaussianFitResult]:
        """
        Fit a Gaussian to ATD data.
        
        Args:
            drift_time: Array of drift time values
            intensity: Array of intensity values
            
        Returns:
            GaussianFitResult object, or None if fitting failed
        """
        # Call the fitting function (from imsocio.utils.data_tools)
        # It returns: (params, r2, fitted_values) or (None, None, None) on failure
        params, r2, fitted_values = self.fitting_function(drift_time, intensity)
        
        if params is None:
            return None
        
        # Unpack parameters: amplitude, center, std_dev
        amplitude, apex, std_dev = params
        
        # Create and return result object
        return GaussianFitResult(
            amplitude=amplitude,
            apex=apex,
            std_dev=std_dev,
            r_squared=r2,
            fitted_values=fitted_values,
            drift_time=drift_time,
            intensity=intensity
        )
    
    def process_file(
        self,
        file_path: Path,
        protein_name: str,
        gas_type: str = 'helium'
    ) -> Optional[CalibrantMeasurement]:
        """
        Process a single calibrant ATD file.
        
        This function:
        1. Extracts charge state from filename
        2. Loads the ATD data
        3. Fits a Gaussian to find the drift time
        4. Looks up literature CCS value
        5. Checks quality (R² threshold)
        
        Args:
            file_path: Path to the ATD file
            protein_name: Name of the calibrant (e.g., 'myoglobin')
            gas_type: 'helium' or 'nitrogen'
            
        Returns:
            CalibrantMeasurement object if successful, None otherwise
            
        Example:
            >>> result = processor.process_file(
            ...     Path("myoglobin/24.txt"),
            ...     "myoglobin",
            ...     "helium"
            ... )
            >>> if result:
            ...     print(f"Success! R² = {result.r_squared:.3f}")
            ... else:
            ...     print("Processing failed")
        """
        # Step 1: Extract charge state from filename
        charge_state = extract_charge_state_from_filename(file_path.name)
        if charge_state is None:
            return None
        
        # Step 2: Load the data
        try:
            drift_time, intensity = load_atd_data(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        # Step 3: Fit Gaussian
        fit_result = self._fit_gaussian(drift_time, intensity)
        if fit_result is None:
            return None
        
        # Step 4: Check R² threshold
        if fit_result.r_squared < self.min_r2:
            # Still return the result, but caller can check r_squared
            # This allows visualization of poor fits
            pass
        
        # Step 5: Look up calibrant in database
        calibrant_info = self.calibrant_db.lookup_calibrant(
            protein_name,
            charge_state,
            gas_type
        )
        
        if calibrant_info is None:
            return None
        
        # Step 6: Create measurement object
        measurement = CalibrantMeasurement(
            protein=protein_name,
            mass=calibrant_info['mass'],
            charge_state=charge_state,
            drift_time=fit_result.apex,
            r_squared=fit_result.r_squared,
            ccs_literature=calibrant_info['ccs'],
            fit_result=fit_result,
            filename=file_path.name
        )
        
        return measurement
    
    def process_folder(
        self,
        folder_path: Path,
        protein_name: str,
        gas_type: str = 'helium'
    ) -> Tuple[List[CalibrantMeasurement], List[Any]]:
        """
        Process all ATD files in a folder.
        
        Args:
            folder_path: Path to folder containing ATD files
            protein_name: Name of the calibrant protein
            gas_type: 'helium' or 'nitrogen'
            
        Returns:
            Tuple of (successful_measurements, skipped_items)
            - successful_measurements: List of CalibrantMeasurement objects that passed R² threshold
            - skipped_items: List containing either:
                - CalibrantMeasurement objects with low R² (still have fit_result for plotting)
                - String error messages for files that failed processing
            
        Example:
            >>> measurements, skipped = processor.process_folder(
            ...     Path("data/myoglobin"),
            ...     "myoglobin"
            ... )
            >>> print(f"Processed {len(measurements)} files")
            >>> print(f"Skipped {len(skipped)} files")
        """
        measurements = []
        skipped = []
        
        # Process each file in the folder
        for file_path in folder_path.iterdir():
            # Skip non-data files
            if not is_valid_calibrant_file(file_path):
                continue
            
            # Try to process the file
            measurement = self.process_file(file_path, protein_name, gas_type)
            
            if measurement is None:
                # Complete failure - add error string
                skipped.append(f"{file_path.name}: Processing failed")
            elif measurement.r_squared < self.min_r2:
                # Low R² - add the measurement itself so we can still plot it
                skipped.append(measurement)
            else:
                # Success - add to results
                measurements.append(measurement)
        
        return measurements, skipped
    
    def process_calibrant_set(
        self,
        base_path: Path,
        gas_type: str = 'helium'
    ) -> pd.DataFrame:
        """
        Process a complete calibrant dataset with multiple proteins.
        
        Expects folder structure:
        base_path/
            myoglobin/
                24.txt
                25.txt
            cytochromec/
                18.txt
                19.txt
        
        Args:
            base_path: Path to folder containing protein subfolders
            gas_type: 'helium' or 'nitrogen'
            
        Returns:
            DataFrame with all measurements
            
        Example:
            >>> df = processor.process_calibrant_set(Path("calibrants"))
            >>> print(df.columns)
            ['protein', 'mass', 'charge_state', 'drift_time', 'r2', 'ccs_literature']
        """
        all_measurements = []
        
        # Process each subfolder
        for folder_path in base_path.iterdir():
            if not folder_path.is_dir():
                continue
            
            # Use folder name as protein name
            protein_name = folder_path.name
            
            # Process this protein's files
            measurements, _ = self.process_folder(
                folder_path,
                protein_name,
                gas_type
            )
            
            all_measurements.extend(measurements)
        
        # Convert to DataFrame
        if not all_measurements:
            return pd.DataFrame()
        
        data = {
            'protein': [m.protein for m in all_measurements],
            'mass': [m.mass for m in all_measurements],
            'charge state': [m.charge_state for m in all_measurements],
            'drift time': [m.drift_time for m in all_measurements],
            'r2': [m.r_squared for m in all_measurements],
            'calibrant_value': [m.ccs_literature for m in all_measurements]
        }
        
        return pd.DataFrame(data)


def measurements_to_dataframe(measurements: List[CalibrantMeasurement]) -> pd.DataFrame:
    """
    Convert a list of CalibrantMeasurement objects to a DataFrame.
    
    Args:
        measurements: List of CalibrantMeasurement objects
        
    Returns:
        DataFrame with columns matching the original format
        
    Example:
        >>> df = measurements_to_dataframe(measurements)
        >>> df.to_csv("results.csv", index=False)
    """
    if not measurements:
        return pd.DataFrame()
    
    data = {
        'protein': [m.protein for m in measurements],
        'mass': [m.mass for m in measurements],
        'charge state': [m.charge_state for m in measurements],
        'drift time': [m.drift_time for m in measurements],
        'r2': [m.r_squared for m in measurements],
        'calibrant_value': [m.ccs_literature for m in measurements]
    }
    
    return pd.DataFrame(data)
