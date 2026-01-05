"""
Output file processing for IMSCal results.

This module handles extraction and organization of calibrated CCS data from
IMSCal output files. It parses the [CALIBRATED DATA] sections and organizes
the results by protein/sample.
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import pandas as pd
import zipfile
import tempfile
import os
from io import StringIO


@dataclass
class ProteinOutput:
    """
    Container for processed output data for a single protein.
    
    Attributes:
        protein_name: Name of the protein/sample
        dataframes: List of DataFrames, one per charge state
    """
    protein_name: str
    dataframes: List[pd.DataFrame]


@dataclass
class OutputProcessingResult:
    """
    Results from processing IMSCal output files.
    
    Attributes:
        protein_outputs: Dictionary mapping protein names to their processed data
        files_processed: Total number of output files successfully processed
    """
    protein_outputs: Dict[str, ProteinOutput]
    files_processed: int


class OutputFileProcessor:
    """
    Processor for IMSCal output files.
    
    This class extracts calibrated CCS data from IMSCal output files.
    Each output file contains a [CALIBRATED DATA] section with columns:
    - Z: Charge state
    - Drift: Drift time
    - CCS: Collision cross-section value
    - CCS Std.Dev.: Standard deviation
    - Intensity: Signal intensity (optional)
    
    Example:
        >>> from io import BytesIO
        >>> processor = OutputFileProcessor()
        >>> result = processor.extract_protein_data(zip_buffer)
        >>> print(f"Processed {result.files_processed} files")
        >>> for protein_name, output in result.protein_outputs.items():
        ...     print(f"{protein_name}: {len(output.dataframes)} charge states")
    """
    
    @staticmethod
    def parse_output_file(file_path: Path) -> pd.DataFrame:
        """
        Parse a single IMSCal output file to extract calibrated data.
        
        IMSCal output files contain multiple sections. This function extracts
        only the [CALIBRATED DATA] section which contains the final results.
        
        Args:
            file_path: Path to the output_X.dat file
            
        Returns:
            DataFrame with columns: Z, Drift, CCS, CCS Std.Dev., and optionally Intensity
            
        Raises:
            ValueError: If [CALIBRATED DATA] section not found or data cannot be parsed
            
        Example:
            >>> df = OutputFileProcessor.parse_output_file(Path("output_24.dat"))
            >>> print(df.columns)
            Index(['Z', 'Drift', 'CCS', 'CCS Std.Dev.', 'Intensity'], dtype='object')
        """
        # Read all lines from file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the [CALIBRATED DATA] section
        try:
            cal_index = next(
                i for i, line in enumerate(lines) 
                if line.strip() == "[CALIBRATED DATA]"
            )
        except StopIteration:
            raise ValueError("No [CALIBRATED DATA] section found in file")
        
        # Extract data lines after the section header
        data_lines = lines[cal_index + 1:]
        
        # Parse CSV data
        df = pd.read_csv(StringIO(''.join(data_lines)))
        
        # Select required columns
        required_cols = ['Z', 'Drift', 'CCS', 'CCS Std.Dev.']
        
        # Include Intensity if present
        if 'Intensity' in df.columns:
            required_cols.append('Intensity')
        
        return df[required_cols]
    
    @staticmethod
    def extract_protein_data(zip_buffer) -> OutputProcessingResult:
        """
        Extract calibrated data from a ZIP file containing IMSCal output files.
        
        The expected structure is:
        ```
        your_data.zip/
        ├── Protein1/
        │   ├── output_1.dat
        │   ├── output_2.dat
        │   └── ...
        ├── Protein2/
        │   ├── output_1.dat
        │   └── ...
        ```
        
        Args:
            zip_buffer: BytesIO buffer containing the ZIP file data
            
        Returns:
            OutputProcessingResult with organized data by protein
            
        Example:
            >>> result = OutputFileProcessor.extract_protein_data(zip_buffer)
            >>> for protein_name, output in result.protein_outputs.items():
            ...     combined = pd.concat(output.dataframes, ignore_index=True)
            ...     print(f"{protein_name}: {len(combined)} total data points")
        """
        protein_outputs = {}
        files_processed = 0
        
        # Use temporary directory to extract ZIP
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write uploaded ZIP to temporary file
            zip_path = Path(tmpdir) / "uploaded.zip"
            with open(zip_path, "wb") as f:
                f.write(zip_buffer.getvalue())
            
            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Walk through extracted files
            for root, dirs, files in os.walk(tmpdir):
                for file_name in files:
                    # Look for output_X.dat files
                    if not (file_name.startswith("output_") and file_name.endswith(".dat")):
                        continue
                    
                    # Full path to the file
                    file_path = Path(root) / file_name
                    
                    # Get protein name from folder structure
                    try:
                        rel_path = os.path.relpath(file_path, tmpdir)
                        parts = rel_path.split(os.sep)
                        
                        if len(parts) < 2:
                            continue
                        
                        # Use the immediate parent folder of the output file
                        # This handles cases where ZIP has intermediate folders
                        protein_name = parts[-2]
                    except (ValueError, IndexError):
                        continue
                    
                    # Parse the output file
                    try:
                        df = OutputFileProcessor.parse_output_file(file_path)
                        
                        # Create protein output entry if needed
                        if protein_name not in protein_outputs:
                            protein_outputs[protein_name] = ProteinOutput(
                                protein_name=protein_name,
                                dataframes=[]
                            )
                        
                        # Add DataFrame to this protein's data
                        protein_outputs[protein_name].dataframes.append(df)
                        files_processed += 1
                        
                    except (ValueError, Exception):
                        # Skip files that can't be parsed
                        continue
        
        return OutputProcessingResult(
            protein_outputs=protein_outputs,
            files_processed=files_processed
        )
    
    @staticmethod
    def combine_protein_data(protein_output: ProteinOutput) -> pd.DataFrame:
        """
        Combine all charge state data for a protein into a single DataFrame.
        
        Args:
            protein_output: ProteinOutput object with multiple DataFrames
            
        Returns:
            Single DataFrame combining all charge states
            
        Example:
            >>> combined = OutputFileProcessor.combine_protein_data(protein_output)
            >>> print(f"Combined {len(combined)} data points from {len(protein_output.dataframes)} files")
        """
        return pd.concat(protein_output.dataframes, ignore_index=True)
