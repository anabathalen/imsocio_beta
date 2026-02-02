"""
Drift time calibration and scaling processor.

This module handles the matching of calibration data with ATD data
and applies scaling factors from mass spectra.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import pandas as pd
import numpy as np
import zipfile
import tempfile
from io import BytesIO

from . import fit_baseline_and_integrate, calculate_theoretical_mz, PROTON_MASS


@dataclass
class CalibratedDriftResult:
    """
    Results from drift time calibration and scaling.
    
    Attributes:
        output_buffers: Dictionary mapping filenames to lists of DataFrames
        processed_files: Number of ATD files successfully processed
        matched_points: Total number of calibration points matched
    """
    output_buffers: Dict[str, List[pd.DataFrame]]
    processed_files: int
    matched_points: int


class DriftCalibrationProcessor:
    """
    Processor for calibrated and scaled drift time data.
    
    This class:
    1. Loads and normalizes ATD data
    2. Calculates scaling factors from mass spectra
    3. Matches calibration data to ATD points
    4. Outputs scaled intensity vs CCS
    """
    
    @staticmethod
    def load_and_normalize_atd(
        file_path: str,
        instrument_type: str,
        inject_time: float
    ) -> Optional[pd.DataFrame]:
        """
        Load ATD file and normalize intensities to max value of 1.
        
        Args:
            file_path: Path to ATD file (tab or comma-separated: drift time, intensity)
            instrument_type: 'Synapt' or 'Cyclic'
            inject_time: Injection time in ms (for Cyclic IMS)
            
        Returns:
            DataFrame with columns: Drift, Intensity (normalized to 0-1)
            
        Example:
            >>> df = DriftCalibrationProcessor.load_and_normalize_atd(
            ...     "24.txt", "Cyclic", 0.5
            ... )
            >>> print(df['Intensity'].max())
            1.0
        """
        try:
            # Use the readers module to load the data
            from ..io.readers import load_atd_data
            
            drift_time, intensity = load_atd_data(Path(file_path))
            
            # Create DataFrame
            raw_df = pd.DataFrame({
                "Drift": drift_time,
                "Intensity": intensity
            })
            
            # Apply instrument-specific drift time correction
            if instrument_type.lower() == "cyclic" and inject_time is not None:
                raw_df["Drift"] = raw_df["Drift"] - inject_time
            
            # ATD files should already be in ms
            # (both text files and TWIM extract files are in ms)
            
            # Normalize ATD intensities so maximum is 1
            max_intensity = raw_df["Intensity"].max()
            if max_intensity > 0:
                raw_df["Intensity"] = raw_df["Intensity"] / max_intensity
            
            return raw_df
            
        except Exception:
            return None
    
    @staticmethod
    def load_mass_spectrum(ms_path: str, verbose: bool = False) -> Optional[pd.DataFrame]:
        """
        Load mass spectrum file with error handling.
        
        Uses the unified load_mass_spectrum function from imsocio.io.readers.
        
        Args:
            ms_path: Path to mass spectrum file
            verbose: If True, print detailed error information
            
        Returns:
            DataFrame with columns: m/z, Intensity, or None if loading fails
            
        Example:
            >>> ms_df = DriftCalibrationProcessor.load_mass_spectrum("mass_spectrum.txt")
        """
        # Check if file exists
        if not os.path.exists(ms_path):
            if verbose:
                print(f"❌ File not found: {ms_path}")
                # Check for case-insensitive alternatives
                parent_dir = os.path.dirname(ms_path)
                if os.path.exists(parent_dir):
                    files = os.listdir(parent_dir)
                    similar = [f for f in files if f.lower() == "mass_spectrum.txt"]
                    if similar:
                        print(f"   Found similar file(s): {similar}")
            return None
        
        try:
            # Use the unified loader from io.readers
            from ..io.readers import load_mass_spectrum
            
            mz, intensity = load_mass_spectrum(Path(ms_path))
            
            # Convert to DataFrame for compatibility with existing code
            ms_df = pd.DataFrame({
                "m/z": mz,
                "Intensity": intensity
            })
            
            return ms_df
            
        except Exception as e:
            if verbose:
                print(f"❌ Error loading {ms_path}: {type(e).__name__}: {e}")
                print(f"   Check that file has two columns (m/z, Intensity)")
            return None
    
    @staticmethod
    def calculate_scale_factor(
        ms_df: pd.DataFrame,
        protein_name: str,
        charge_state: int,
        protein_mass: float,
        scale_ranges: Dict[Tuple[str, int], Tuple[float, float]],
        use_max_intensity: bool = False,
        smoothing_window: int = 51
    ) -> Tuple[Optional[float], float]:
        """
        Calculate scaling factor from mass spectrum integration or max intensity.
        
        Args:
            ms_df: Mass spectrum DataFrame
            protein_name: Name of the protein
            charge_state: Charge state
            protein_mass: Protein mass in Da
            scale_ranges: Dictionary mapping (protein, charge) to (min_mz, max_mz)
            use_max_intensity: If True, use max instead of integration
            smoothing_window: Window size for smoothing
            
        Returns:
            Tuple of (scale_factor, theoretical_mz)
            
        Example:
            >>> scale_factor, mz = DriftCalibrationProcessor.calculate_scale_factor(
            ...     ms_df, "myoglobin", 24, 16952.3, scale_ranges
            ... )
        """
        if ms_df is None or protein_mass is None:
            return None, None
            
        mz = calculate_theoretical_mz(protein_mass, charge_state)
        range_key = (protein_name, charge_state)
        
        if range_key not in scale_ranges:
            return None, mz
            
        mz_min, mz_max = scale_ranges[range_key]
        
        try:
            # Get data in range
            mask = (ms_df["m/z"] >= mz_min) & (ms_df["m/z"] <= mz_max)
            if np.sum(mask) < 3:
                return None, mz
            
            # Apply smoothing
            smoothed_intensity = ms_df["Intensity"].rolling(
                window=smoothing_window,
                center=True,
                min_periods=1
            ).mean()
            
            if use_max_intensity:
                # Use maximum intensity in the range
                scale_factor = smoothed_intensity[mask].max()
            else:
                # Use baseline fitting for more accurate integration
                scale_factor, _ = fit_baseline_and_integrate(
                    ms_df["m/z"].values,
                    smoothed_intensity.values,
                    (mz_min, mz_max)
                )
            
            return scale_factor if scale_factor > 0 else None, mz
            
        except Exception:
            return None, mz
    
    @staticmethod
    def match_and_calibrate(
        drift_zip: BytesIO,
        cal_csvs: List,
        instrument_type: str,
        inject_time: float,
        charge_ranges: Dict[str, Tuple[int, int]],
        scale_ranges: Dict[Tuple[str, int], Tuple[float, float]],
        protein_masses: Dict[str, float],
        ms_data: Dict[str, pd.DataFrame],
        use_max_intensity: bool = False,
        smoothing_params: Optional[Dict[Tuple[str, int], int]] = None
    ) -> Tuple[CalibratedDriftResult, List[str]]:
        """
        Match calibration data with ATD data and apply scaling.
        
        This is the main processing function that:
        1. Extracts ZIP with ATD files
        2. Loads calibration CSVs
        3. For each protein/charge:
           - Loads and normalizes ATD
           - Calculates scale factor from mass spectrum
           - Matches calibration points to ATD
           - Outputs scaled intensity vs CCS
        
        Args:
            drift_zip: ZIP file containing protein folders with ATD files
            cal_csvs: List of calibration CSV files
            instrument_type: 'Synapt' or 'Cyclic'
            inject_time: Injection time for Cyclic IMS
            charge_ranges: Dict mapping protein names to (min_charge, max_charge)
            scale_ranges: Dict mapping (protein, charge) to (min_mz, max_mz)
            protein_masses: Dict mapping protein names to masses in Da
            ms_data: Dict mapping protein names to mass spectrum DataFrames
            use_max_intensity: Use max intensity instead of integration
            smoothing_params: Optional dict mapping (protein, charge) to smoothing window size
            
        Returns:
            Tuple of (CalibratedDriftResult, list of skipped file messages)
        """
        output_buffers = {}
        processed_files = 0
        matched_points = 0
        skipped_files = []
        
        # Use default smoothing if not provided
        if smoothing_params is None:
            smoothing_params = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP file
            drift_zip_path = os.path.join(tmpdir, "drift.zip")
            with open(drift_zip_path, "wb") as f:
                f.write(drift_zip.getvalue())
            with zipfile.ZipFile(drift_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Load calibration data once
            calibration_lookup = {}
            for file in cal_csvs:
                protein_name = file.name.replace(".csv", "")
                df = pd.read_csv(file)
                
                # Calibration CSVs are generated in seconds (for IMSCal)
                # but ATD files are in ms, so convert calibration to ms
                df["Drift"] = df["Drift"] * 1000.0
                
                for _, row in df.iterrows():
                    key = (protein_name, int(row["Z"]))
                    calibration_lookup.setdefault(key, []).append({
                        "Drift": row["Drift"],
                        "CCS": row["CCS"],
                        "CCS Std.Dev.": row["CCS Std.Dev."]
                    })

            # Process each protein directory
            for root, _, files in os.walk(tmpdir):
                protein_name = os.path.basename(root)
                if protein_name == os.path.basename(tmpdir):  # Skip root
                    continue

                # Get charge range for this protein
                charge_range = charge_ranges.get(protein_name, (2, 4))

                # Get mass spectrum from provided ms_data
                ms_df = ms_data.get(protein_name, None)
                protein_mass = protein_masses.get(protein_name, None)

                # Process ATD files for each charge state
                # Import file validation functions
                from ..io.readers import is_valid_calibrant_file, extract_charge_state_from_filename
                
                atd_files = [
                    f for f in files
                    if is_valid_calibrant_file(Path(os.path.join(root, f)))
                ]

                for file in atd_files:
                    charge_state = extract_charge_state_from_filename(file)
                    if charge_state is None:
                        continue

                    # Skip if outside charge range
                    if charge_state < charge_range[0] or charge_state > charge_range[1]:
                        continue
                    
                    # Skip if no calibration data available
                    key = (protein_name, charge_state)
                    cal_data = calibration_lookup.get(key)
                    if not cal_data:
                        skipped_files.append(
                            f"{protein_name} charge {charge_state}: No calibration data"
                        )
                        continue
                    
                    # Skip if no scale factor defined
                    if key not in scale_ranges:
                        skipped_files.append(
                            f"{protein_name} charge {charge_state}: No scale factor defined"
                        )
                        continue
                    
                    # Load and normalize ATD
                    file_path = os.path.join(root, file)
                    normalized_df = DriftCalibrationProcessor.load_and_normalize_atd(
                        file_path, instrument_type, inject_time
                    )
                    
                    if normalized_df is None:
                        skipped_files.append(
                            f"{protein_name} charge {charge_state}: Could not load ATD"
                        )
                        continue
                    
                    processed_files += 1
                    
                    # Get smoothing window for this protein/charge (default to 51)
                    smoothing_window = smoothing_params.get((protein_name, charge_state), 51)
                    
                    # Calculate scaling factor from mass spectrum
                    scale_factor, mz = DriftCalibrationProcessor.calculate_scale_factor(
                        ms_df, protein_name, charge_state, protein_mass,
                        scale_ranges, use_max_intensity, smoothing_window=smoothing_window
                    )
                    
                    # Skip if scale factor calculation failed
                    if scale_factor is None:
                        skipped_files.append(
                            f"{protein_name} charge {charge_state}: "
                            "Scale factor calculation failed"
                        )
                        continue
                    
                    # Match calibration points to normalized ATD data
                    out_rows = []
                    for entry in cal_data:
                        drift_val = entry["Drift"]
                        
                        # Find closest drift time in normalized data
                        closest_idx = (
                            normalized_df["Drift"] - drift_val
                        ).abs().idxmin()
                        normalized_intensity = normalized_df.loc[
                            closest_idx, "Intensity"
                        ]
                        
                        # Apply scaling to normalized intensity
                        scaled_intensity = normalized_intensity * scale_factor
                        
                        out_rows.append({
                            "Charge": charge_state,
                            "Drift": drift_val,
                            "CCS": entry["CCS"],
                            "CCS Std.Dev.": entry["CCS Std.Dev."],
                            "Normalized_Intensity": normalized_intensity,
                            "Scaled_Intensity": scaled_intensity,
                            "m/z": mz
                        })
                        matched_points += 1
                    
                    if out_rows:
                        out_df = pd.DataFrame(out_rows)
                        out_key = f"{protein_name}.csv"
                        output_buffers.setdefault(out_key, []).append(out_df)

        result = CalibratedDriftResult(
            output_buffers, processed_files, matched_points
        )
        return result, skipped_files
    
    @staticmethod
    def prepare_zip(output_buffers: Dict[str, List[pd.DataFrame]]) -> BytesIO:
        """
        Prepare ZIP file with all results.
        
        Args:
            output_buffers: Dictionary mapping filenames to DataFrames
            
        Returns:
            BytesIO buffer containing ZIP file
        """
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
            for filename, dfs in output_buffers.items():
                combined = pd.concat(dfs, ignore_index=True)
                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                zip_out.writestr(filename, csv_bytes)
        zip_buffer.seek(0)
        return zip_buffer
