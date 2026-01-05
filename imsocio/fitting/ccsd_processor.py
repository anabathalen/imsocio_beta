"""
CCSD Data Processor Module
===========================

This module provides tools for processing collision cross-section distribution (CCSD)
data, including summing intensity across charge states.

Classes
-------
CCSDDataProcessor
    Static methods for CCSD data manipulation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class CCSDDataProcessor:
    """
    CCSD data processing tools.
    
    Provides static methods for processing collision cross-section distribution
    data, particularly for charge state deconvolution and summing.
    
    Methods
    -------
    create_summed_data(df)
        Create summed CCSD data across all charge states
        
    Examples
    --------
    Sum intensity across charge states:
    
    >>> from imsocio.fitting import CCSDDataProcessor
    >>> # df has columns: CCS, Scaled_Intensity, Charge
    >>> summed_df = CCSDDataProcessor.create_summed_data(df)
    >>> # summed_df has columns: CCS, Scaled_Intensity
    """
    
    @staticmethod
    def create_summed_data(df):
        """
        Create summed data across charge states.
        
        This method interpolates each charge state's CCSD onto a common CCS grid
        and sums the intensities. Useful for charge state deconvolution analysis.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with columns:
            - CCS : float - Collision cross-section values
            - Scaled_Intensity : float - Intensity values
            - Charge : int - Charge state
            
        Returns
        -------
        pandas.DataFrame
            Summed dataframe with columns:
            - CCS : float - Common CCS grid (1000 points)
            - Scaled_Intensity : float - Summed intensity across all charge states
            
        Notes
        -----
        - Automatically cleans data (removes NaN values)
        - Uses linear interpolation for each charge state
        - Creates uniform CCS grid from min to max CCS in dataset
        - Handles duplicate CCS values by dropping them
        
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'CCS': [800, 810, 820, 800, 810, 820],
        ...     'Scaled_Intensity': [100, 150, 80, 120, 180, 90],
        ...     'Charge': [5, 5, 5, 6, 6, 6]
        ... })
        >>> summed = CCSDDataProcessor.create_summed_data(df)
        >>> print(len(summed))  # 1000 points
        >>> print(summed.columns.tolist())  # ['CCS', 'Scaled_Intensity']
        """
        # Clean inputs
        df = df.copy()
        df['CCS'] = pd.to_numeric(df['CCS'], errors='coerce')
        df['Scaled_Intensity'] = pd.to_numeric(df['Scaled_Intensity'], errors='coerce')
        df = df.dropna(subset=['CCS', 'Scaled_Intensity'])

        ccs_min, ccs_max = df['CCS'].min(), df['CCS'].max()
        ccs_range = np.linspace(ccs_min, ccs_max, 1000)
        summed_intensity = np.zeros_like(ccs_range)
        
        for charge in df['Charge'].unique():
            df_charge = df[df['Charge'] == charge]
            if len(df_charge) >= 2:
                df_charge = df_charge.drop_duplicates(subset=['CCS']).sort_values('CCS')
                interp_func = interp1d(
                    df_charge['CCS'],
                    df_charge['Scaled_Intensity'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0
                )
                summed_intensity += interp_func(ccs_range)
        
        return pd.DataFrame({'CCS': ccs_range, 'Scaled_Intensity': summed_intensity})
