"""
Bush calibrant database handling.

This module provides an interface to the Bush et al. database of protein
collision cross sections, which is commonly used for IMS calibration.

Reference:
Bush, M. F., et al. "Collision Cross Sections of Protein Ions." 
Journal of the American Society for Mass Spectrometry, 2010, 21, 1003-1010.
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd


# Standard names for calibrant proteins and their folder names
CALIBRANT_FOLDER_MAPPING = {
    'Denatured Myoglobin': 'myoglobin',
    'Denatured Cytochrome C': 'cytochromec',
    'Polyalanine Peptide of Length X': 'polyalanineX',
    'Denatured Ubiquitin': 'ubiquitin',
    'Native BSA': 'BSA',
    'GRGDS Peptide': 'GRGDS',
    'SDGRG Peptide': 'SDGRG'
}


def load_bush_database(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Bush calibrant database from a CSV file.
    
    The Bush database contains literature CCS values for common calibrant
    proteins in different drift gases (helium and nitrogen).
    
    Args:
        file_path: Path to the bush.csv file. If None, looks in default location
                   (data/bush.csv relative to the current directory)
        
    Returns:
        DataFrame with columns like:
        - 'protein': protein name (e.g., 'myoglobin')
        - 'charge': charge state
        - 'mass': protein mass in Da
        - 'CCS_he': CCS in helium (nm²)
        - 'CCS_n2': CCS in nitrogen (nm²)
        
    Raises:
        FileNotFoundError: If the database file cannot be found
        
    Example:
        >>> bush_df = load_bush_database()
        >>> print(bush_df.head())
           protein  charge      mass  CCS_he  CCS_n2
        0  myoglobin    18  16952.3   28.5    30.2
        1  myoglobin    19  16952.3   29.1    30.8
    """
    # If no path provided, try the default location
    if file_path is None:
        # Try common locations
        possible_paths = [
            Path('data/bush.csv'),
            Path('../data/bush.csv'),
            Path('../../data/bush.csv'),
        ]
        
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(
                "Could not find bush.csv. Please provide the path explicitly."
            )
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Bush database not found at: {file_path}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    return df


class CalibrantDatabase:
    """
    Interface to the Bush calibrant database.
    
    This class provides convenient methods to look up calibrant CCS values
    and manage the database.
    
    Attributes:
        df: The Bush database DataFrame
        
    Example:
        >>> bush_df = load_bush_database()
        >>> db = CalibrantDatabase(bush_df)
        >>> result = db.lookup_calibrant('myoglobin', 24, 'helium')
        >>> print(f"CCS: {result['ccs']:.1f} nm², Mass: {result['mass']:.1f} Da")
        CCS: 31.2 nm², Mass: 16952.3 Da
    """
    
    def __init__(self, bush_df: pd.DataFrame):
        """
        Initialize the database interface.
        
        Args:
            bush_df: DataFrame containing the Bush calibrant data
        """
        self.df = bush_df
    
    def get_calibrant_column(self, gas_type: str) -> str:
        """
        Get the column name for CCS values based on drift gas.
        
        Args:
            gas_type: Either 'helium' or 'nitrogen' (case-insensitive)
            
        Returns:
            Column name: 'CCS_he' for helium, 'CCS_n2' for nitrogen
            
        Example:
            >>> db = CalibrantDatabase(bush_df)
            >>> column = db.get_calibrant_column('helium')
            >>> print(column)
            CCS_he
        """
        if gas_type.lower() == 'helium':
            return 'CCS_he'
        elif gas_type.lower() == 'nitrogen':
            return 'CCS_n2'
        else:
            raise ValueError(f"Unknown gas type: {gas_type}. Use 'helium' or 'nitrogen'")
    
    def lookup_calibrant(
        self,
        protein: str,
        charge_state: int,
        gas_type: str = 'helium'
    ) -> Optional[Dict[str, float]]:
        """
        Look up CCS and mass values for a specific protein and charge state.
        
        Args:
            protein: Protein name (e.g., 'myoglobin', 'cytochromec')
            charge_state: Integer charge state (e.g., 24)
            gas_type: Drift gas type ('helium' or 'nitrogen')
            
        Returns:
            Dictionary with 'ccs' and 'mass' keys, or None if not found
            Example: {'ccs': 31.2, 'mass': 16952.3}
            
        Example:
            >>> db = CalibrantDatabase(bush_df)
            >>> result = db.lookup_calibrant('myoglobin', 24, 'helium')
            >>> if result:
            ...     print(f"Found: CCS = {result['ccs']:.2f} nm²")
            Found: CCS = 31.20 nm²
        """
        # Get the appropriate column name for the gas type
        ccs_column = self.get_calibrant_column(gas_type)
        
        # Find matching row in database
        # We need both protein name AND charge state to match
        matching_rows = self.df[
            (self.df['protein'] == protein) & 
            (self.df['charge'] == charge_state)
        ]
        
        # Check if we found a match
        if matching_rows.empty:
            return None
        
        # Get the first matching row (should only be one)
        row = matching_rows.iloc[0]
        
        # Get CCS value
        ccs_value = row[ccs_column]
        
        # Check if CCS value is missing (NaN)
        if pd.isna(ccs_value):
            return None
        
        # Return both CCS and mass
        return {
            'ccs': float(ccs_value),
            'mass': float(row['mass'])
        }
    
    def get_available_charge_states(self, protein: str) -> list:
        """
        Get all available charge states for a given protein.
        
        Args:
            protein: Protein name
            
        Returns:
            List of charge states available in the database
            
        Example:
            >>> db = CalibrantDatabase(bush_df)
            >>> charges = db.get_available_charge_states('myoglobin')
            >>> print(f"Available charges: {charges}")
            Available charges: [18, 19, 20, 21, 22, 23, 24, 25]
        """
        protein_data = self.df[self.df['protein'] == protein]
        return sorted(protein_data['charge'].unique().tolist())
    
    def get_available_proteins(self) -> list:
        """
        Get all proteins in the database.
        
        Returns:
            List of protein names
            
        Example:
            >>> db = CalibrantDatabase(bush_df)
            >>> proteins = db.get_available_proteins()
            >>> print(proteins)
            ['myoglobin', 'cytochromec', 'ubiquitin', 'BSA']
        """
        return sorted(self.df['protein'].unique().tolist())
