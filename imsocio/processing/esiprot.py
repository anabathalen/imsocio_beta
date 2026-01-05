"""ESIProt - Protein charge state deconvolution and m/z calculation.

This module implements the ESIProt algorithm by Robert Winkler for deconvoluting
protein charge states from electrospray ionization mass spectrometry data.

Reference: ESIProt 1.1 - Robert Winkler, 2009-2017, GPLv3 License
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from math import sqrt, pow


@dataclass
class DeconvolutionResult:
    """Results from ESIProt deconvolution.
    
    Attributes:
        mz_values: Input m/z values (filtered for non-zero)
        charge_states: Assigned charge states for each m/z
        molecular_weights: Calculated molecular weights for each charge state
        errors: Deviation from average for each MW
        average_mw: Final deconvoluted molecular weight
        stdev: Standard deviation of molecular weights
        original_indices: Indices of non-zero input values
    """
    mz_values: List[float]
    charge_states: List[int]
    molecular_weights: List[float]
    errors: List[float]
    average_mw: float
    stdev: float
    original_indices: List[int]


@dataclass
class MZCalculation:
    """Calculated m/z value for a given charge state.
    
    Attributes:
        charge: Charge state
        mz: Calculated m/z value
    """
    charge: int
    mz: float


class ESIProtCalculator:
    """ESIProt algorithm implementation for protein charge state analysis."""
    
    # EXACT hydrogen mass from original ESIProt (IUPAC 2005)
    HYDROGEN_MASS = 1.00794 - 0.0005485799
    
    @classmethod
    def deconvolute(
        cls,
        mz_values: List[float],
        charge_min: int = 1,
        charge_max: int = 100
    ) -> Tuple[Optional[DeconvolutionResult], Optional[str]]:
        """Deconvolute protein charge states from m/z values using ESIProt algorithm.
        
        This implements the exact ESIProt algorithm by Robert Winkler, which:
        1. Tests consecutive decreasing charge states
        2. Calculates MW for each assignment
        3. Finds assignment with minimum standard deviation
        
        Args:
            mz_values: List of m/z values (up to 9 peaks, use 0 to skip)
            charge_min: Minimum charge state to test (default: 1)
            charge_max: Maximum charge state to test (default: 100)
            
        Returns:
            Tuple of (DeconvolutionResult or None, error_message or None)
        """
        # Filter out zero values but track original positions
        mz = []
        original_indices = []
        for i, mz_val in enumerate(mz_values):
            if mz_val > 0:
                mz.append(mz_val)
                original_indices.append(i)
        
        n_peaks = len(mz)
        
        if n_peaks < 2:
            return None, "Please enter at least 2 m/z values for calculation."
        
        # Initialize best result trackers
        stdev_champion = 1000
        average_champion = 1000000
        charge_champion = [0] * 9
        mw_champion = [0] * 9
        error_champion = [0] * 9
        
        # Test each possible starting charge state
        for chargecount_1 in range(charge_min, charge_max + 1):
            # Initialize arrays for this iteration
            mw = [0] * 9
            error = [0] * 9
            charge = [0] * 9
            
            # Set initial charge for first peak
            charge[0] = chargecount_1
            
            # Set consecutive decreasing charges for subsequent peaks
            for i in range(1, 9):
                charge[i] = charge[i-1] - 1
            
            # Calculate molecular weights
            sum_mw = 0
            nulls = 0
            
            for i in range(9):
                if i < len(mz):
                    # Calculate MW using exact ESIProt formula
                    mw[i] = (mz[i] * charge[i]) - (charge[i] * cls.HYDROGEN_MASS)
                    sum_mw += mw[i]
                else:
                    mw[i] = 0
                    nulls += 1
            
            notnulls = 9 - nulls
            if notnulls == 0:
                continue
            
            average = sum_mw / notnulls
            
            # Calculate errors and standard deviation
            errorsquaresum = 0
            for i in range(notnulls):
                error[i] = mw[i] - average
                errorsquare = pow(error[i], 2)
                errorsquaresum += errorsquare
            
            # Calculate standard deviation
            if notnulls > 1:
                stdev = sqrt(errorsquaresum * pow((notnulls - 1), -1))
            else:
                stdev = 0
            
            # Check if this is the best result (lowest standard deviation)
            if stdev < stdev_champion:
                stdev_champion = stdev
                average_champion = average
                
                # Store best approximation
                for z in range(9):
                    charge_champion[z] = charge[z]
                    mw_champion[z] = mw[z]
                    if mw_champion[z] == 0:
                        charge_champion[z] = 0
                    error_champion[z] = error[z]
        
        # Create result object
        result = DeconvolutionResult(
            mz_values=mz[:n_peaks],
            charge_states=charge_champion[:n_peaks],
            molecular_weights=mw_champion[:n_peaks],
            errors=error_champion[:n_peaks],
            average_mw=average_champion,
            stdev=stdev_champion,
            original_indices=original_indices
        )
        
        return result, None
    
    @classmethod
    def calculate_mz_from_mass(
        cls,
        molecular_weight: float,
        charge_min: int,
        charge_max: int
    ) -> List[MZCalculation]:
        """Calculate m/z values for a given molecular weight and charge range.
        
        Uses the reverse ESIProt formula: m/z = (MW + (charge × H)) / charge
        
        Args:
            molecular_weight: Known molecular weight in Daltons
            charge_min: Minimum charge state
            charge_max: Maximum charge state
            
        Returns:
            List of MZCalculation objects
        """
        calculations = []
        
        for charge in range(charge_min, charge_max + 1):
            # Use exact reverse formula
            mz = (molecular_weight + (charge * cls.HYDROGEN_MASS)) / charge
            calculations.append(MZCalculation(charge=charge, mz=mz))
        
        return calculations


class ESIProtDataExporter:
    """Export ESIProt results to various formats."""
    
    @staticmethod
    def to_dict_list(result: DeconvolutionResult, mz_inputs: List[float]) -> List[Dict[str, str]]:
        """Convert deconvolution result to list of dictionaries for CSV/DataFrame.
        
        Args:
            result: DeconvolutionResult object
            mz_inputs: Original input m/z values
            
        Returns:
            List of dictionaries with formatted results
        """
        data = []
        
        # Add individual charge state results
        for i, orig_idx in enumerate(result.original_indices):
            if i < len(result.mz_values):
                data.append({
                    'm/z': f"{result.mz_values[i]:.4f}",
                    'charge (z)': result.charge_states[i],
                    'mass (da)': f"{result.molecular_weights[i]:.2f}",
                    'error (da)': f"{result.errors[i]:.4f}"
                })
        
        # Add final deconvoluted mass row
        data.append({
            'm/z': "Final Result",
            'charge (z)': "N/A",
            'mass (da)': f"{result.average_mw:.2f}",
            'error (da)': f"{result.stdev:.4f}"
        })
        
        return data
    
    @staticmethod
    def to_esiprot_format(result: DeconvolutionResult) -> str:
        """Convert result to original ESIProt text format.
        
        Args:
            result: DeconvolutionResult object
            
        Returns:
            Formatted text string matching ESIProt output
        """
        lines = []
        lines.append("FINAL RESULTS")
        lines.append("*" * 79)
        
        for mz_val, charge, mw, error in zip(
            result.mz_values,
            result.charge_states,
            result.molecular_weights,
            result.errors
        ):
            lines.append(
                f"m/z: {mz_val:.1f} charge: {charge}+ "
                f"MW [Da]: {mw:.2f} Error [Da]: {error:.2f}"
            )
        
        lines.append(
            f"Deconvoluted MW [Da]: {result.average_mw:.2f} "
            f"Standard deviation [Da]: {result.stdev:.2f}"
        )
        
        return '\n'.join(lines)
    
    @staticmethod
    def mz_calculations_to_dict_list(calculations: List[MZCalculation]) -> List[Dict[str, str]]:
        """Convert m/z calculations to list of dictionaries.
        
        Args:
            calculations: List of MZCalculation objects
            
        Returns:
            List of dictionaries with formatted calculations
        """
        data = []
        
        for calc in calculations:
            data.append({
                'charge (z)': calc.charge,
                'm/z': f"{calc.mz:.4f}"
            })
        
        return data
