"""Range file generation for TWIMExtract.

This module provides functionality to generate range files for TWIMExtract
analysis based on protein mass and charge states.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import io
import logging
import zipfile
from pathlib import Path

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class RangeParameters:
    """Parameters for range file generation.
    
    Attributes:
        mass: Protein molecular mass in Daltons
        mz_range_size: Total m/z window size
        charge_range: Tuple of (min_charge, max_charge)
        rt_start: Retention time start in minutes
        rt_end: Retention time end in minutes
        dt_start: Drift time start in bins
        dt_end: Drift time end in bins
        folder_name: Name for output folder
    """
    mass: float
    mz_range_size: float
    charge_range: Tuple[int, int]
    rt_start: float
    rt_end: float
    dt_start: int
    dt_end: int
    folder_name: str


@dataclass
class CIURangeParameters:
    """Parameters for CIU range file generation from Origami data.
    
    Attributes:
        mass: Protein molecular mass in Daltons
        charge: Single charge state
        mz_range_size: Total m/z window size
        dt_start: Drift time start in bins
        dt_end: Drift time end in bins
        first_scan: First scan number in the data
        first_voltage: First collision voltage value
        scans_per_voltage: Number of scans per collision voltage
        voltage_increment: Increment between collision voltages
        seconds_per_scan: Seconds per scan (for RT conversion)
        num_voltages: Number of different voltages
        folder_name: Name for output folder
    """
    mass: float
    charge: int
    mz_range_size: float
    dt_start: int
    dt_end: int
    first_scan: int
    first_voltage: float
    scans_per_voltage: int
    voltage_increment: float
    seconds_per_scan: float
    num_voltages: int
    folder_name: str


@dataclass
class RangeFileResult:
    """Results from range file generation.
    
    Attributes:
        generated_files: List of generated filenames
        charge_states: List of charge states
        mz_values: Dictionary mapping charge state to m/z value
    """
    generated_files: List[str]
    charge_states: List[int]
    mz_values: Dict[int, float]


class RangeFileGenerator:
    """Generate TWIMExtract range files from protein parameters."""
    
    def __init__(self, params: RangeParameters):
        """Initialize generator with parameters.
        
        Args:
            params: RangeParameters object
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not isinstance(params, RangeParameters):
            raise ValueError("params must be a RangeParameters object")
        
        # Validate mass
        if params.mass <= 0:
            raise ValueError(f"Mass must be positive, got {params.mass}")
        if params.mass > 1e6:  # 1 MDa limit seems reasonable
            logger.warning(f"Very large mass detected: {params.mass} Da")
        
        # Validate m/z range size
        if params.mz_range_size <= 0:
            raise ValueError(f"m/z range size must be positive, got {params.mz_range_size}")
        if params.mz_range_size > 1000:
            logger.warning(f"Very large m/z range: {params.mz_range_size}")
        
        # Validate charge range
        min_charge, max_charge = params.charge_range
        if min_charge <= 0:
            raise ValueError(f"Minimum charge must be positive, got {min_charge}")
        if max_charge <= 0:
            raise ValueError(f"Maximum charge must be positive, got {max_charge}")
        if min_charge > max_charge:
            raise ValueError(f"Minimum charge ({min_charge}) cannot be greater than maximum charge ({max_charge})")
        if max_charge > 200:
            logger.warning(f"Very high charge state: {max_charge}")
        if max_charge - min_charge > 100:
            logger.warning(f"Large charge range: {min_charge} to {max_charge} ({max_charge - min_charge + 1} states)")
        
        # Validate retention times
        if params.rt_start < 0:
            raise ValueError(f"RT start cannot be negative, got {params.rt_start}")
        if params.rt_end < 0:
            raise ValueError(f"RT end cannot be negative, got {params.rt_end}")
        if params.rt_start >= params.rt_end:
            raise ValueError(f"RT start ({params.rt_start}) must be less than RT end ({params.rt_end})")
        
        # Validate drift times
        if params.dt_start < 0:
            raise ValueError(f"DT start cannot be negative, got {params.dt_start}")
        if params.dt_end < 0:
            raise ValueError(f"DT end cannot be negative, got {params.dt_end}")
        if params.dt_start >= params.dt_end:
            raise ValueError(f"DT start ({params.dt_start}) must be less than DT end ({params.dt_end})")
        
        # Validate folder name
        if not params.folder_name or not params.folder_name.strip():
            raise ValueError("Folder name cannot be empty")
        
        # Check for invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in params.folder_name for char in invalid_chars):
            raise ValueError(f"Folder name contains invalid characters: {invalid_chars}")
        
        self.params = params
    
    def calculate_mz(self, charge: int) -> float:
        """Calculate m/z for given charge state.
        
        Uses the formula: m/z = (mass + charge * PROTON_MASS) / charge
        
        Args:
            charge: Charge state
            
        Returns:
            Calculated m/z value
            
        Raises:
            ValueError: If charge is invalid
        """
        if charge <= 0:
            raise ValueError(f"Charge must be positive, got {charge}")
        
        PROTON_MASS = 1.007276
        return (self.params.mass + charge * PROTON_MASS) / charge
    
    def generate_range_content(self, charge: int) -> str:
        """Generate content for a single range file.
        
        Args:
            charge: Charge state for this range file
            
        Returns:
            String content for the range file
            
        Raises:
            ValueError: If charge is invalid
        """
        if charge <= 0:
            raise ValueError(f"Charge must be positive, got {charge}")
        
        mz = self.calculate_mz(charge)
        half_range = self.params.mz_range_size / 2.0
        
        mz_start = mz - half_range
        mz_end = mz + half_range
        
        # Validate m/z range is positive
        if mz_start <= 0:
            logger.warning(f"m/z start ({mz_start:.1f}) is <= 0 for charge {charge} - this may be invalid")
        
        content = f"""MZ_start: {mz_start:.1f}
MZ_end: {mz_end:.1f}
RT_start_(minutes): {self.params.rt_start:.1f}
RT_end_(minutes): {self.params.rt_end:.1f}
DT_start_(bins): {self.params.dt_start}
DT_end_(bins): {self.params.dt_end}"""
        
        return content
    
    def generate_all_files(self, output_dir: str) -> RangeFileResult:
        """Generate all range files for the specified charge range.
        
        Args:
            output_dir: Directory to write range files to
            
        Returns:
            RangeFileResult containing generated file information
            
        Raises:
            ValueError: If output_dir is invalid
            IOError: If files cannot be written
        """
        # Validate output directory
        if not output_dir or not output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        
        output_path = Path(output_dir)
        
        # Create directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, IOError) as e:
            raise IOError(f"Failed to create output directory {output_dir}: {e}")
        
        # Verify it's a directory
        if not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_dir}")
        
        generated_files = []
        charge_states = []
        mz_values = {}
        failed_charges = []
        
        min_charge, max_charge = self.params.charge_range
        
        for charge in range(min_charge, max_charge + 1):
            try:
                filename = f"range_{charge}.txt"
                filepath = output_path / filename
                
                content = self.generate_range_content(charge)
                mz = self.calculate_mz(charge)
                
                # Write file
                try:
                    with open(filepath, 'w') as f:
                        f.write(content)
                except (IOError, OSError) as e:
                    logger.error(f"Failed to write file {filename}: {e}")
                    failed_charges.append(charge)
                    continue
                
                generated_files.append(filename)
                charge_states.append(charge)
                mz_values[charge] = mz
                
            except Exception as e:
                logger.error(f"Failed to generate range file for charge {charge}: {e}")
                failed_charges.append(charge)
                continue
        
        if not generated_files:
            raise IOError(f"Failed to generate any range files for charges {min_charge}-{max_charge}")
        
        if failed_charges:
            logger.warning(f"Failed to generate files for {len(failed_charges)} charge states: {failed_charges}")
        
        logger.info(f"Successfully generated {len(generated_files)} range files in {output_dir}")
        
        return RangeFileResult(generated_files, charge_states, mz_values)
    
    def generate_preview_data(self, max_preview: int = 5) -> List[Dict[str, str]]:
        """Generate preview data for charge states and m/z values.
        
        Args:
            max_preview: Maximum number of charge states to preview (must be >= 1)
            
        Returns:
            List of dictionaries with preview information
            
        Raises:
            ValueError: If max_preview is invalid
        """
        if max_preview < 1:
            raise ValueError(f"max_preview must be >= 1, got {max_preview}")
        
        preview_data = []
        min_charge, max_charge = self.params.charge_range
        
        for charge in range(min_charge, min(max_charge + 1, min_charge + max_preview)):
            try:
                mz = self.calculate_mz(charge)
                half_range = self.params.mz_range_size / 2.0
                preview_data.append({
                    "Charge": f"{charge}+",
                    "m/z": f"{mz:.2f}",
                    "Range": f"{mz - half_range:.1f} - {mz + half_range:.1f}"
                })
            except Exception as e:
                logger.warning(f"Failed to generate preview for charge {charge}: {e}")
                continue
        
        if max_charge - min_charge >= max_preview:
            preview_data.append({
                "Charge": "...",
                "m/z": "...",
                "Range": f"(+{max_charge - min_charge + 1 - max_preview} more)"
            })
        
        return preview_data


class RangeFilePackager:
    """Package range files into downloadable formats."""
    
    @staticmethod
    def create_zip(output_dir: str, result: RangeFileResult, folder_name: str) -> io.BytesIO:
        """Create ZIP file containing all range files.
        
        Args:
            output_dir: Directory containing the range files
            result: RangeFileResult with file information
            folder_name: Name for folder inside ZIP
            
        Returns:
            BytesIO buffer containing ZIP file
            
        Raises:
            ValueError: If inputs are invalid
            IOError: If ZIP cannot be created or files cannot be read
        """
        # Validate inputs
        if not output_dir or not output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        
        if not isinstance(result, RangeFileResult):
            raise ValueError("result must be a RangeFileResult object")
        
        if not folder_name or not folder_name.strip():
            raise ValueError("folder_name cannot be empty")
        
        # Validate output directory exists
        output_path = Path(output_dir)
        if not output_path.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
        if not output_path.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        
        # Check if there are files to add
        if not result.generated_files:
            raise ValueError("No files to add to ZIP - result.generated_files is empty")
        
        zip_buffer = io.BytesIO()
        files_added = 0
        missing_files = []
        
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename in result.generated_files:
                    filepath = output_path / filename
                    
                    # Check if file exists
                    if not filepath.exists():
                        logger.warning(f"File not found, skipping: {filepath}")
                        missing_files.append(filename)
                        continue
                    
                    try:
                        # Add files to a folder inside the ZIP
                        zipf.write(filepath, os.path.join(folder_name, filename))
                        files_added += 1
                    except (IOError, OSError) as e:
                        logger.error(f"Failed to add {filename} to ZIP: {e}")
                        missing_files.append(filename)
                        continue
            
            if files_added == 0:
                raise IOError(f"No files were added to ZIP archive from {output_dir}")
            
            if missing_files:
                logger.warning(f"Failed to add {len(missing_files)} files to ZIP: {missing_files}")
            
            logger.info(f"Created ZIP archive with {files_added} files in folder '{folder_name}'")
            
        except Exception as e:
            raise IOError(f"Failed to create ZIP archive: {e}")
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def get_zip_filename(folder_name: str) -> str:
        """Generate appropriate ZIP filename.
        
        Args:
            folder_name: Base folder name
            
        Returns:
            ZIP filename
            
        Raises:
            ValueError: If folder_name is invalid
        """
        if not folder_name or not folder_name.strip():
            raise ValueError("folder_name cannot be empty")
        
        # Sanitize folder name for use in filename
        # Remove invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        sanitized = folder_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        return f"{sanitized}_range_files.zip"


class CIURangeFileGenerator:
    """Generate TWIMExtract range files for Origami CIU data."""
    
    def __init__(self, params: CIURangeParameters):
        """Initialize CIU generator with parameters.
        
        Args:
            params: CIURangeParameters object
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not isinstance(params, CIURangeParameters):
            raise ValueError("params must be a CIURangeParameters object")
        
        # Validate mass
        if params.mass <= 0:
            raise ValueError(f"Mass must be positive, got {params.mass}")
        
        # Validate charge
        if params.charge <= 0:
            raise ValueError(f"Charge must be positive, got {params.charge}")
        
        # Validate m/z range size
        if params.mz_range_size <= 0:
            raise ValueError(f"m/z range size must be positive, got {params.mz_range_size}")
        
        # Validate drift times
        if params.dt_start < 0:
            raise ValueError(f"DT start cannot be negative, got {params.dt_start}")
        if params.dt_end < 0:
            raise ValueError(f"DT end cannot be negative, got {params.dt_end}")
        if params.dt_start >= params.dt_end:
            raise ValueError(f"DT start ({params.dt_start}) must be less than DT end ({params.dt_end})")
        
        # Validate CIU-specific parameters
        if params.first_scan < 0:
            raise ValueError(f"First scan cannot be negative, got {params.first_scan}")
        
        if params.first_voltage <= 0:
            raise ValueError(f"First voltage must be positive, got {params.first_voltage}")
        
        if params.scans_per_voltage <= 0:
            raise ValueError(f"Scans per voltage must be positive, got {params.scans_per_voltage}")
        
        if params.voltage_increment <= 0:
            raise ValueError(f"Voltage increment must be positive, got {params.voltage_increment}")
        
        if params.seconds_per_scan <= 0:
            raise ValueError(f"Seconds per scan must be positive, got {params.seconds_per_scan}")
        
        if params.num_voltages <= 0:
            raise ValueError(f"Number of voltages must be positive, got {params.num_voltages}")
        
        # Validate folder name
        if not params.folder_name or not params.folder_name.strip():
            raise ValueError("Folder name cannot be empty")
        
        # Check for invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in params.folder_name for char in invalid_chars):
            raise ValueError(f"Folder name contains invalid characters: {invalid_chars}")
        
        self.params = params
    
    def calculate_mz(self) -> float:
        """Calculate m/z for the specified charge state.
        
        Returns:
            Calculated m/z value
        """
        PROTON_MASS = 1.007276
        return (self.params.mass + self.params.charge * PROTON_MASS) / self.params.charge
    
    def calculate_voltage(self, voltage_index: int) -> float:
        """Calculate collision voltage for a given index.
        
        Args:
            voltage_index: Index of the voltage (0-based)
            
        Returns:
            Collision voltage value
        """
        return self.params.first_voltage + (voltage_index * self.params.voltage_increment)
    
    def calculate_scan_range(self, voltage_index: int) -> Tuple[int, int]:
        """Calculate scan range for a given voltage index.
        
        Args:
            voltage_index: Index of the voltage (0-based)
            
        Returns:
            Tuple of (scan_start, scan_end)
        """
        scan_start = self.params.first_scan + (voltage_index * self.params.scans_per_voltage)
        scan_end = scan_start + self.params.scans_per_voltage - 1
        return scan_start, scan_end
    
    def calculate_rt_range(self, voltage_index: int) -> Tuple[float, float]:
        """Calculate retention time range for a given voltage index.
        
        Converts scan numbers to retention time in minutes.
        
        Args:
            voltage_index: Index of the voltage (0-based)
            
        Returns:
            Tuple of (rt_start, rt_end) in minutes
        """
        scan_start, scan_end = self.calculate_scan_range(voltage_index)
        
        # Convert scans to time in minutes
        # RT = (scan - first_scan) * seconds_per_scan / 60
        rt_start = (scan_start - self.params.first_scan) * self.params.seconds_per_scan / 60.0
        rt_end = (scan_end - self.params.first_scan + 1) * self.params.seconds_per_scan / 60.0
        
        return rt_start, rt_end
    
    def generate_range_content(self, voltage_index: int) -> str:
        """Generate content for a single CIU range file.
        
        Args:
            voltage_index: Index of the voltage (0-based)
            
        Returns:
            String content for the range file
        """
        mz = self.calculate_mz()
        half_range = self.params.mz_range_size / 2.0
        
        # Debug logging
        logger.info(f"CIU Range Generation Debug:")
        logger.info(f"  mass={self.params.mass}, charge={self.params.charge}, mz_range_size={self.params.mz_range_size}")
        logger.info(f"  Calculated mz={mz}, half_range={half_range}")
        
        mz_start = mz - half_range
        mz_end = mz + half_range
        
        rt_start, rt_end = self.calculate_rt_range(voltage_index)
        
        # Validate m/z range is positive
        if mz_start <= 0:
            logger.warning(f"m/z start ({mz_start:.1f}) is <= 0 - this may be invalid")
        
        content = f"""MZ_start: {mz_start:.1f}
MZ_end: {mz_end:.1f}
RT_start_(minutes): {rt_start:.2f}
RT_end_(minutes): {rt_end:.2f}
DT_start_(bins): {self.params.dt_start}
DT_end_(bins): {self.params.dt_end}"""
        
        return content
    
    def generate_all_files(self, output_dir: str) -> RangeFileResult:
        """Generate all range files for all voltages.
        
        Args:
            output_dir: Directory to write range files to
            
        Returns:
            RangeFileResult containing generated file information
            
        Raises:
            ValueError: If output_dir is invalid
            IOError: If files cannot be written
        """
        # Validate output directory
        if not output_dir or not output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        
        output_path = Path(output_dir)
        
        # Create directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, IOError) as e:
            raise IOError(f"Failed to create output directory {output_dir}: {e}")
        
        # Verify it's a directory
        if not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_dir}")
        
        generated_files = []
        voltages = []
        mz_values = {}
        failed_voltages = []
        
        mz = self.calculate_mz()
        
        for voltage_index in range(self.params.num_voltages):
            try:
                voltage = self.calculate_voltage(voltage_index)
                filename = f"{voltage:.0f}V.txt"
                filepath = output_path / filename
                
                content = self.generate_range_content(voltage_index)
                
                # Write file
                try:
                    with open(filepath, 'w') as f:
                        f.write(content)
                except (IOError, OSError) as e:
                    logger.error(f"Failed to write file {filename}: {e}")
                    failed_voltages.append(voltage)
                    continue
                
                generated_files.append(filename)
                voltages.append(voltage)
                mz_values[voltage] = mz
                
            except Exception as e:
                logger.error(f"Failed to generate range file for voltage index {voltage_index}: {e}")
                failed_voltages.append(voltage_index)
                continue
        
        if not generated_files:
            raise IOError(f"Failed to generate any range files for {self.params.num_voltages} voltages")
        
        if failed_voltages:
            logger.warning(f"Failed to generate files for {len(failed_voltages)} voltages: {failed_voltages}")
        
        logger.info(f"Successfully generated {len(generated_files)} CIU range files in {output_dir}")
        
        # Return result with voltages instead of charge states
        # Note: We're reusing RangeFileResult but charge_states will contain voltage values
        return RangeFileResult(generated_files, voltages, mz_values)
    
    def generate_preview_data(self, max_preview: int = 5) -> List[Dict[str, str]]:
        """Generate preview data for voltages and scan ranges.
        
        Args:
            max_preview: Maximum number of voltages to preview
            
        Returns:
            List of dictionaries with preview information
        """
        if max_preview < 1:
            raise ValueError(f"max_preview must be >= 1, got {max_preview}")
        
        preview_data = []
        
        for voltage_index in range(min(self.params.num_voltages, max_preview)):
            try:
                voltage = self.calculate_voltage(voltage_index)
                scan_start, scan_end = self.calculate_scan_range(voltage_index)
                rt_start, rt_end = self.calculate_rt_range(voltage_index)
                
                preview_data.append({
                    "Voltage": f"{voltage:.0f} V",
                    "Scans": f"{scan_start}-{scan_end}",
                    "RT (min)": f"{rt_start:.2f}-{rt_end:.2f}"
                })
            except Exception as e:
                logger.warning(f"Failed to generate preview for voltage index {voltage_index}: {e}")
                continue
        
        if self.params.num_voltages > max_preview:
            preview_data.append({
                "Voltage": "...",
                "Scans": "...",
                "RT (min)": f"(+{self.params.num_voltages - max_preview} more)"
            })
        
        return preview_data
