"""
Calibration utilities including drift time adjustments and file generation.

This module handles instrument-specific corrections and file format generation
for calibration workflows.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class InstrumentParams:
    """
    IMS instrument parameters needed for calibration.
    
    Attributes:
        wave_velocity: Traveling wave velocity in m/s
        wave_height: Wave height (voltage) in V
        pressure: IMS pressure in mbar
        drift_length: Drift cell length in m
        instrument_type: 'cyclic' or 'synapt'
        inject_time: Injection time in ms (only for Cyclic IMS)
    """
    wave_velocity: float
    wave_height: float
    pressure: float
    drift_length: float
    instrument_type: str
    inject_time: float = 0.0


def adjust_drift_time_for_injection(
    drift_time: float,
    inject_time: float,
    instrument_type: str
) -> float:
    """
    Adjust drift time for injection time (Cyclic IMS only).
    
    In Cyclic IMS, the measured arrival time includes both the drift time
    AND the time the ion spent in the injection region. We need to subtract
    the injection time to get the true drift time.
    
    For Synapt instruments, no adjustment is needed.
    
    Args:
        drift_time: Measured arrival time in ms
        inject_time: Injection time in ms
        instrument_type: 'cyclic' or 'synapt' (case-insensitive)
        
    Returns:
        Adjusted drift time in ms
        
    Example:
        >>> # Cyclic IMS measurement
        >>> measured_time = 5.5  # ms
        >>> inject_time = 0.3    # ms
        >>> true_drift = adjust_drift_time_for_injection(
        ...     measured_time, inject_time, 'cyclic'
        ... )
        >>> print(true_drift)
        5.2
        
        >>> # Synapt - no adjustment
        >>> drift = adjust_drift_time_for_injection(5.5, 0.3, 'synapt')
        >>> print(drift)
        5.5
    """
    if instrument_type.lower() == 'cyclic':
        # Subtract injection time for Cyclic IMS
        return drift_time - inject_time
    else:
        # No adjustment needed for Synapt
        return drift_time


def adjust_dataframe_drift_times(
    df: pd.DataFrame,
    instrument_params: InstrumentParams
) -> pd.DataFrame:
    """
    Adjust all drift times in a DataFrame based on instrument type.
    
    Args:
        df: DataFrame with 'drift time' column
        instrument_params: Instrument parameters including inject_time
        
    Returns:
        New DataFrame with adjusted drift times
        
    Example:
        >>> params = InstrumentParams(
        ...     wave_velocity=281.0,
        ...     wave_height=20.0,
        ...     pressure=1.63,
        ...     drift_length=0.98,
        ...     instrument_type='cyclic',
        ...     inject_time=0.3
        ... )
        >>> adjusted_df = adjust_dataframe_drift_times(results_df, params)
    """
    # Make a copy to avoid modifying the original
    adjusted_df = df.copy()
    
    # Apply adjustment to each drift time
    adjusted_df['drift time'] = adjusted_df['drift time'].apply(
        lambda dt: adjust_drift_time_for_injection(
            dt,
            instrument_params.inject_time,
            instrument_params.instrument_type
        )
    )
    
    return adjusted_df


def calculate_modified_drift_time(
    drift_time: float,
    enhanced_duty_cycle: float,
    mz: float
) -> float:
    """
    Calculate modified drift time for alternative calibration method.
    
    The modified drift time is calculated as:
        dt' = dt - (EDC * sqrt(m/z) / 1000)
    
    where:
        dt = measured drift time (ms)
        EDC = enhanced duty cycle
        m/z = mass-to-charge ratio
    
    Args:
        drift_time: Measured drift time in ms
        enhanced_duty_cycle: Enhanced duty cycle value
        mz: Mass-to-charge ratio
        
    Returns:
        Modified drift time
        
    Example:
        >>> dt_modified = calculate_modified_drift_time(5.5, 0.3, 1000.0)
        >>> print(f"{dt_modified:.4f}")
    """
    return drift_time - (enhanced_duty_cycle * np.sqrt(mz) / 1000)


def calculate_modified_ccs(
    ccs: float,
    charge: int,
    mass_calibrant: float,
    mass_drift_gas: float
) -> float:
    """
    Calculate modified CCS for alternative calibration method.
    
    The modified CCS is calculated as:
        CCS' = CCS / (|charge| * sqrt(1 / reduced_mass))
    
    which simplifies to:
        CCS' = CCS * sqrt(reduced_mass) / |charge|
    
    where:
        CCS = literature collision cross section (nm²)
        charge = charge state
        reduced_mass = (M_cal * M_gas) / (M_cal + M_gas)
    
    Args:
        ccs: Literature CCS value in nm²
        charge: Charge state (will use absolute value)
        mass_calibrant: Mass of calibrant in Da
        mass_drift_gas: Mass of drift gas in Da (e.g., 28.0134 for N2, 4.0026 for He)
        
    Returns:
        Modified CCS value
        
    Example:
        >>> ccs_mod = calculate_modified_ccs(2500.0, 24, 16952.3, 28.0134)
        >>> print(f"{ccs_mod:.4f}")
    """
    reduced_mass = (mass_calibrant * mass_drift_gas) / (mass_calibrant + mass_drift_gas)
    return ccs * np.sqrt(reduced_mass) / abs(charge)


def calculate_modified_modified_drift_time(
    modified_drift_time: float,
    slope: float,
    charge: int,
    reduced_mass: float
) -> float:
    """
    Calculate modified modified drift time (tD'') for alternative calibration.
    
    The modified modified drift time is calculated as:
        tD'' = (tD')^slope * |charge| * sqrt(1 / reduced_mass)
    
    where:
        tD' = modified drift time (dt - EDC*sqrt(m/z)/1000)
        slope = slope of ln(CCS') vs ln(tD') linear fit
        charge = charge state
        reduced_mass = (M_cal * M_gas) / (M_cal + M_gas)
    
    Args:
        modified_drift_time: Modified drift time (tD') in ms
        slope: Slope from ln-ln calibration plot
        charge: Charge state (will use absolute value)
        reduced_mass: Reduced mass in Da
        
    Returns:
        Modified modified drift time (tD'')
        
    Example:
        >>> td_double_prime = calculate_modified_modified_drift_time(5.49, 0.395, 24, 27.98)
        >>> print(f"{td_double_prime:.4f}")
    """
    return (modified_drift_time ** slope) * abs(charge) * np.sqrt(1 / reduced_mass)


def prepare_alternative_calibration_data(
    calibration_df: pd.DataFrame,
    enhanced_duty_cycle: float,
    drift_gas: str = 'nitrogen'
) -> pd.DataFrame:
    """
    Prepare calibration data for alternative (ln-ln) calibration method.
    
    This function:
    1. Calculates m/z from mass and charge state
    2. Calculates reduced mass for each calibrant
    3. Calculates modified drift time (tD'): dt' = dt - (EDC * sqrt(m/z) / 1000)
    4. Calculates modified CCS (CCS'): CCS' = CCS / (|z| * sqrt(1/reduced_mass))
    5. Calculates ln(tD') and ln(CCS')
    6. Calculates slope from ln-ln plot
    7. Calculates modified modified drift time (tD''): tD'' = (tD')^slope * |z| * sqrt(1/reduced_mass)
    
    The resulting DataFrame can be used to create a ln-ln calibration plot.
    
    Args:
        calibration_df: DataFrame with columns 'mass', 'charge state', 
                       'drift time', 'calibrant_value' (CCS literature)
        enhanced_duty_cycle: Enhanced duty cycle value
        drift_gas: Drift gas type ('nitrogen' or 'helium')
        
    Returns:
        DataFrame with additional columns:
            - 'mz': calculated m/z
            - 'reduced_mass': reduced mass in Da
            - 'modified_drift_time': tD'
            - 'modified_ccs': CCS'
            - 'ln_modified_drift_time': ln(tD')
            - 'ln_modified_ccs': ln(CCS')
            - 'modified_modified_drift_time': tD''
            
    Raises:
        ValueError: If required columns are missing from input DataFrame
        
    Example:
        >>> # After loading calibration.csv from the calibrate page
        >>> alt_data = prepare_alternative_calibration_data(cal_df, edc=0.3, drift_gas='nitrogen')
        >>> # Now plot ln_modified_drift_time vs ln_modified_ccs
    """
    # Validate required columns
    required_cols = ['mass', 'charge state', 'drift time', 'calibrant_value']
    missing_cols = [col for col in required_cols if col not in calibration_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying original
    result_df = calibration_df.copy()
    
    # Drift gas masses in Da
    DRIFT_GAS_MASSES = {
        'nitrogen': 28.0134,
        'helium': 4.0026
    }
    
    if drift_gas.lower() not in DRIFT_GAS_MASSES:
        raise ValueError(f"Unknown drift gas: {drift_gas}. Must be 'nitrogen' or 'helium'")
    
    mass_drift_gas = DRIFT_GAS_MASSES[drift_gas.lower()]
    
    # Calculate m/z using same formula as in processing module
    # m/z = (mass + charge * proton_mass) / charge
    # Using proton mass = 1.00727647 Da
    PROTON_MASS = 1.00727647
    result_df['mz'] = (result_df['mass'] + PROTON_MASS * result_df['charge state']) / result_df['charge state']
    
    # Calculate reduced mass for each row
    result_df['reduced_mass'] = (result_df['mass'] * mass_drift_gas) / (result_df['mass'] + mass_drift_gas)
    
    # Calculate modified drift time
    result_df['modified_drift_time'] = result_df.apply(
        lambda row: calculate_modified_drift_time(
            row['drift time'],
            enhanced_duty_cycle,
            row['mz']
        ),
        axis=1
    )
    
    # Calculate modified CCS
    result_df['modified_ccs'] = result_df.apply(
        lambda row: calculate_modified_ccs(
            row['calibrant_value'],
            row['charge state'],
            row['mass'],
            mass_drift_gas
        ),
        axis=1
    )
    
    # Calculate natural logarithms
    result_df['ln_modified_drift_time'] = np.log(result_df['modified_drift_time'])
    result_df['ln_modified_ccs'] = np.log(result_df['modified_ccs'])
    
    # Calculate slope of ln(CCS') vs ln(tD') for the entire dataset
    slope = np.polyfit(result_df['ln_modified_drift_time'], result_df['ln_modified_ccs'], 1)[0]
    
    # Calculate modified modified drift time (tD'')
    result_df['modified_modified_drift_time'] = result_df.apply(
        lambda row: calculate_modified_modified_drift_time(
            row['modified_drift_time'],
            slope,
            row['charge state'],
            row['reduced_mass']
        ),
        axis=1
    )
    
    return result_df
