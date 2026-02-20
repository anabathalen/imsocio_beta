"""
Streamlit Page for Fitting CCS Distributions

This page aims to replicate fitting functionality present in ORIGIN but tailor it to fitting CCS Distributions.
"""

import sys
from pathlib import Path

# Add parent directory to path to import myutils
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import trapezoid
import warnings
import copy
warnings.filterwarnings('ignore')

# Import from imsocio package
from imsocio.fitting import (
    # Peak functions
    multi_peak_function,
    get_params_per_peak,
    get_parameter_names,
    # Classes
    PeakDetector,
    ParameterEstimator,
    ParameterManager,
    FittingEngine,
    DataProcessor,
    ResultAnalyzer,
    CCSDDataProcessor
)

from streamlit_app import styling


# --- UI Components ---
class FitDataUI:
    """UI components for the fit data page."""
    
    @staticmethod
    def show_main_header():
        """Display main page header."""
        styling.load_custom_css()
        
        st.markdown("""
        <div class="main-header">
            <h1>📊 CCS Distribution Fitting</h1>
            <p>Perform peak fitting on calibrated CCS Distributions</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_peak_detection_controls():
        """Show peak detection controls."""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🔍 Peak Detection</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Method selection
        detection_method = st.radio(
            "Peak Detection Method",
            ["Manual Centroids (Recommended)", "Auto-detect peaks"],
            help="Manual centroids give you precise control over peak positions"
        )
        
        if detection_method == "Manual Centroids (Recommended)":
            st.info("💡 **Enter the CCS/m/z values where you see peaks in your data**")
            
            # Text input for centroids
            centroid_input = st.text_area(
                "Peak Centroids (one per line or comma-separated)",
                height=100,
                placeholder="Example:\n1250.5\n1275.2\n1300.8\n\nOr: 1250.5, 1275.2, 1300.8",
                help="Enter the approximate center positions of peaks you want to fit"
            )
            
            # Parse centroids
            manual_centroids = []
            if centroid_input.strip():
                try:
                    # Try comma-separated first
                    if ',' in centroid_input:
                        manual_centroids = [float(x.strip()) for x in centroid_input.split(',') if x.strip()]
                    else:
                        # Try newline-separated
                        manual_centroids = [float(x.strip()) for x in centroid_input.split('\n') if x.strip()]
                    
                    if manual_centroids:
                        st.success(f"✓ {len(manual_centroids)} centroid(s) specified: {', '.join([f'{c:.2f}' for c in sorted(manual_centroids)])}")
                except ValueError as e:
                    st.error(f"⚠️ Invalid centroid format. Please enter numbers only. Error: {str(e)}")
                    manual_centroids = []
            
            detection_params = None
            return manual_centroids, detection_params
            
        else:  # Auto-detect
            st.info("📊 **Automatic peak detection** - adjust parameters to find peaks")
            col1, col2 = st.columns(2)
            with col1:
                min_height = st.slider("Minimum Height (%)", 1, 50, 5)
                min_prominence = st.slider("Minimum Prominence (%)", 1, 20, 2)
            with col2:
                min_distance = st.slider("Minimum Distance (%)", 1, 20, 5)
                smoothing = st.slider("Smoothing Points", 0, 20, 5)
            
            detection_params = {
                'min_height_percent': min_height,
                'min_prominence_percent': min_prominence,
                'min_distance_percent': min_distance,
                'smoothing_points': smoothing
            }
            manual_centroids = []
            return manual_centroids, detection_params
    
    @staticmethod
    def show_fitting_options():
        """Show fitting options controls."""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">⚙️ Fitting Options</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            peak_type = st.selectbox(
                "Peak Type (Gaussian Recommended)",
                ["Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG"]
            )
            
            baseline_type = st.selectbox(
                "Baseline Correction",
                ["None", "Linear", "Polynomial", "Exponential"]
            )
            
            poly_degree = 2
            if baseline_type == "Polynomial":
                poly_degree = st.slider("Polynomial Degree", 2, 5, 2)
        
        with col2:
            fit_method = st.selectbox(
                "Fitting Method",
                ["Levenberg-Marquardt", "Global"]
            )
            
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            tolerance = st.select_slider(
                "Tolerance",
                options=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
                value=1e-8,
                format_func=lambda x: f"{x:.0e}"
            )
            
            use_weights = st.checkbox("Use weighted fitting", value=False)
            
        return {
            'peak_type': peak_type,
            'baseline_type': baseline_type,
            'poly_degree': poly_degree,
            'fit_method': fit_method,
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'use_weights': use_weights
        }
    
    @staticmethod
    def show_preprocessing_options():
        """Show data preprocessing options."""
        with st.expander("🔧 Preprocessing Options", expanded=False):
            smooth_data = st.checkbox("Smooth data", value=False)
            
            if smooth_data:
                col1, col2 = st.columns(2)
                with col1:
                    smooth_method = st.selectbox(
                        "Smoothing Method",
                        ["Savitzky-Golay", "Moving Average"]
                    )
                    window_size = st.slider("Window Size", 3, 21, 5, step=2)
                with col2:
                    poly_order = 2
                    if smooth_method == "Savitzky-Golay":
                        poly_order = st.slider("Polynomial Order", 1, 5, 2)
            else:
                smooth_method = "Savitzky-Golay"
                window_size = 5
                poly_order = 2
                
        return {
            'smooth_data': smooth_data,
            'smooth_method': smooth_method,
            'window_size': window_size,
            'poly_order': poly_order
        }
    
    @staticmethod
    def display_peak_table(peak_info):
        """Display detected peaks in a table."""
        st.markdown("### 🎯 Detected Peaks")
        
        peak_data = []
        for i, peak in enumerate(peak_info):
            peak_data.append({
                'Peak': i + 1,
                'CCS': f"{peak['x']:.2f}",
                'Intensity': f"{peak['y']:.2f}",
                'Width (FWHM)': f"{peak.get('width_half', 0):.2f}",
                'Prominence': f"{peak.get('prominence', 0):.2f}",
                'Area (Est.)': f"{peak.get('area_estimate', 0):.2f}"
            })
        
        st.dataframe(pd.DataFrame(peak_data), use_container_width=True)
    
    @staticmethod
    def display_fit_statistics(result, peak_stats):
        """Display fitting statistics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R²", f"{result['r_squared']:.4f}")
        with col2:
            st.metric("Adjusted R²", f"{result['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{result['rmse']:.4f}")
        with col4:
            st.metric("Reduced χ²", f"{result['reduced_chi_squared']:.4f}")
        
        # Additional statistics in expander
        with st.expander("📊 Additional Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**AIC:** {result['aic']:.2f}")
                st.write(f"**BIC:** {result['bic']:.2f}")
            with col2:
                st.write(f"**Reduced χ²:** {result['reduced_chi_squared']:.2f}")
                st.write(f"**Fit Success:** {'✅ Yes' if result['success'] else '❌ No'}")
        
        # Peak statistics table
        if peak_stats:
            st.markdown("""
            <div class="section-card">
                <div class="section-header">📈 Peak Statistics</div>
            </div>
            """, unsafe_allow_html=True)
            peak_df = pd.DataFrame(peak_stats)
            st.dataframe(peak_df, use_container_width=True)
    
    @staticmethod
    def create_fit_plot(x_data, y_data, result, baseline=None, show_components=True):
        """Create interactive plot of fit results."""
        fig = go.Figure()
        
        # Determine if we need to add baseline back
        y_corrected = result.get('y_corrected', y_data)
        has_baseline = baseline is not None and not np.allclose(baseline, 0)
        
        # Original data
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Data',
            marker=dict(size=4, color='rgba(0,0,0,0.5)')
        ))
        
        # Baseline
        if has_baseline:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=baseline,
                mode='lines',
                name='Baseline',
                line=dict(color='gray', dash='dash', width=1)
            ))
        
        # Fitted curve (add baseline back if it exists)
        fitted_curve_display = result['fitted_curve'].copy()
        if has_baseline:
            fitted_curve_display = fitted_curve_display + baseline
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=fitted_curve_display,
            mode='lines',
            name='Fitted Curve',
            line=dict(color='red', width=2)
        ))
        
        # Individual peak components (add baseline back if it exists)
        if show_components and 'peak_components' in result and result['peak_components']:
            colors = px.colors.qualitative.Set2
            for i, component in enumerate(result['peak_components']):
                component_display = component.copy()
                if has_baseline:
                    component_display = component_display + baseline
                    
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=component_display,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=colors[i % len(colors)], dash='dot', width=1.5),
                    opacity=0.7
                ))
        
        # Residuals (always relative to baseline-subtracted data)
        fig.add_trace(go.Scatter(
            x=x_data,
            y=result['residuals'],
            mode='lines',
            name='Residuals',
            line=dict(color='green', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Peak Fitting Results',
            xaxis_title='CCS (Å^2)',
            yaxis_title='Intensity',
            yaxis2=dict(
                title='Residuals',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.15
            )
        )
        
        return fig
    
    @staticmethod
    def show_parameter_editor(param_manager, x_data, y_data):
        """Show interactive parameter editor."""
        st.markdown("### 🎚️ Parameter Editor")
        
        param_names = get_parameter_names(param_manager.peak_type)
        
        # Create tabs for each peak
        n_peaks = param_manager.n_peaks
        if n_peaks == 0:
            st.info("No peaks to edit")
            return
        
        tabs = st.tabs([f"Peak {i+1}" for i in range(n_peaks)])
        
        for peak_idx, tab in enumerate(tabs):
            with tab:
                cols = st.columns(len(param_names) + 2)  # Extra column for constrain button
                
                for param_idx, param_name in enumerate(param_names):
                    with cols[param_idx]:
                        global_idx = peak_idx * param_manager.params_per_peak + param_idx
                        current_value = param_manager.parameters[global_idx]
                        
                        # Check if parameter has custom bounds
                        has_bounds = global_idx in param_manager.param_bounds
                        bounds_info = ""
                        if has_bounds:
                            lower, upper = param_manager.param_bounds[global_idx]
                            bounds_info = f" 🔒 [{lower:.4f}, {upper:.4f}]"
                        
                        # Parameter value input with bounds indicator
                        new_value = st.number_input(
                            param_name + bounds_info,
                            value=float(current_value),
                            format="%.4f",
                            key=f"param_{peak_idx}_{param_idx}"
                        )
                        
                        # Update if changed
                        if new_value != current_value:
                            param_manager.update_parameter(peak_idx, param_idx, new_value)
                        
                        # Fix/unfix checkbox
                        is_fixed = param_manager.is_parameter_fixed(peak_idx, param_idx)
                        fixed = st.checkbox(
                            "Fix",
                            value=is_fixed,
                            key=f"fix_{peak_idx}_{param_idx}",
                            help="Fix this parameter (prevent it from changing during fit)"
                        )
                        
                        if fixed != is_fixed:
                            param_manager.fix_parameter(peak_idx, param_idx, fixed)
                        
                        # Constrain to ±1% button
                        if st.button("±1%", key=f"constrain_{peak_idx}_{param_idx}",
                                   help="Constrain parameter to ±1% of current value"):
                            param_manager.set_tight_bounds(peak_idx, param_idx, tolerance_percent=1.0)
                            st.rerun()  # Rerun to show bounds indicator
                
                # Delete peak button
                with cols[-2]:
                    st.write("")  # Spacing
                    st.write("")
                    if st.button("🗑️ Delete", key=f"delete_{peak_idx}"):
                        param_manager.delete_peak(peak_idx)
                        st.rerun()
        
        # Add new peak section
        st.markdown("---")
        st.markdown("#### ➕ Add New Peak")
        with st.expander("Add Peak Manually", expanded=False):
            cols = st.columns(len(param_names) + 1)
            new_peak_params = []
            
            for param_idx, param_name in enumerate(param_names):
                with cols[param_idx]:
                    # Suggest sensible defaults
                    if param_idx == 0:  # Amplitude
                        default = float(np.max(y_data) * 0.5)
                    elif param_idx == 1:  # Center
                        default = float((x_data.min() + x_data.max()) / 2)
                    else:  # Width parameters
                        default = float((x_data.max() - x_data.min()) * 0.05)
                    
                    value = st.number_input(
                        param_name,
                        value=default,
                        format="%.4f",
                        key=f"new_peak_{param_idx}"
                    )
                    new_peak_params.append(value)
            
            with cols[-1]:
                st.write("")  # Spacing
                st.write("")
                if st.button("➕ Add Peak", type="primary"):
                    param_manager.add_peak(new_peak_params)
                    st.success(f"Added new peak!")
                    st.rerun()
    
    @staticmethod
    def export_fitted_data(all_charge_results, export_points=1000):
        """Export fitted data for all charge states into a single DataFrame.
        
        Parameters
        ----------
        all_charge_results : dict
            Dictionary of charge state results
        export_points : int
            Number of points per charge state
            
        Returns
        -------
        pd.DataFrame or None
            Combined fitted data for all charges
        """
        if not all_charge_results:
            return None
        
        all_data = []
        
        for charge in sorted(all_charge_results.keys()):
            result_data = all_charge_results[charge]
            fit_result = result_data['fit_result']
            fitting_options = result_data['fitting_options']
            data_info = result_data['data_info']
            
            # Generate high-resolution x values
            ccs_min, ccs_max = data_info['ccs_range']
            x_hr = np.linspace(ccs_min, ccs_max, export_points)
            
            # Calculate fitted curve
            y_hr = multi_peak_function(
                x_hr, 
                fitting_options['peak_type'], 
                *fit_result['parameters']
            )
            
            # Create dataframe for this charge
            charge_df = pd.DataFrame({
                'Charge': charge,
                'CCS': x_hr,
                'Fitted_Intensity': y_hr
            })
            
            all_data.append(charge_df)
        
        # Combine all charges
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df


def perform_fitting(x_data, y_data, fitting_options, detection_params=None,
                   preprocessing_options=None, manual_peaks=None):
    """
    Perform complete fitting workflow.
    
    Parameters
    ----------
    x_data : ndarray
        CCS values
    y_data : ndarray
        Intensity values
    fitting_options : dict
        Fitting configuration options
    detection_params : dict, optional
        Peak detection parameters
    preprocessing_options : dict, optional
        Data preprocessing options
    manual_peaks : list, optional
        Manually specified peak positions
        
    Returns
    -------
    dict
        Complete fitting results including parameters, statistics, and components
    """
    # Initialize processor
    processor = DataProcessor()
    
    # Preprocessing
    if preprocessing_options and preprocessing_options['smooth_data']:
        y_processed = processor.smooth_data(
            x_data, y_data,
            method=preprocessing_options['smooth_method'],
            window_size=preprocessing_options['window_size'],
            poly_order=preprocessing_options['poly_order']
        )
    else:
        y_processed = y_data.copy()
    
    # Baseline subtraction
    y_corrected, baseline = processor.subtract_baseline(
        x_data, y_processed,
        method=fitting_options['baseline_type'],
        poly_degree=fitting_options.get('poly_degree', 2)
    )
    
    # Peak detection
    if detection_params:
        detector = PeakDetector()
        peak_info = detector.find_peaks_origin_style(
            x_data, y_corrected,
            min_height_percent=detection_params['min_height_percent'],
            min_prominence_percent=detection_params['min_prominence_percent'],
            min_distance_percent=detection_params['min_distance_percent'],
            smoothing_points=detection_params['smoothing_points']
        )
    elif manual_peaks:
        # Create peak info from manual positions with better estimates
        peak_info = []
        for center in manual_peaks:
            # Find closest data point
            center_idx = np.argmin(np.abs(x_data - center))
            actual_center = x_data[center_idx]
            
            # Get intensity at that position
            intensity_at_center = y_corrected[center_idx]
            
            # Estimate width by finding nearby local minima or using local data
            # Look for FWHM by examining data around the peak
            search_radius = max(5, len(x_data) // 20)  # Search ±5 points or 5% of data
            start_idx = max(0, center_idx - search_radius)
            end_idx = min(len(x_data), center_idx + search_radius)
            
            local_x = x_data[start_idx:end_idx]
            local_y = y_corrected[start_idx:end_idx]
            
            if len(local_y) > 2:
                # Find half-maximum
                half_max = intensity_at_center / 2
                # Find points closest to half max on each side
                left_side = local_y[:search_radius]
                right_side = local_y[search_radius:]
                
                # Estimate width from local data spread
                if len(left_side) > 0 and len(right_side) > 0:
                    left_idx = np.argmin(np.abs(left_side - half_max))
                    right_idx = np.argmin(np.abs(right_side - half_max))
                    width_estimate = abs(local_x[search_radius + right_idx] - local_x[left_idx])
                    width_estimate = max(width_estimate, (x_data.max() - x_data.min()) / 50)
                else:
                    width_estimate = (x_data.max() - x_data.min()) / 20
            else:
                width_estimate = (x_data.max() - x_data.min()) / 20
            
            peak_info.append({
                'index': center_idx,
                'x': actual_center,
                'y': intensity_at_center,
                'prominence': intensity_at_center,
                'width_half': width_estimate,
                'width_base': width_estimate * 2,
                'area_estimate': intensity_at_center * width_estimate
            })
    else:
        st.error("Either enable auto-detection or specify manual peak positions")
        return None
    
    if not peak_info:
        st.warning("No peaks detected. Try adjusting detection parameters.")
        return None
    
    # Warn if only one peak detected - may need more sensitive detection
    if len(peak_info) == 1 and detection_params:
        data_range = y_corrected.max() - y_corrected.min()
        if data_range > 0:
            # Check if there might be more peaks by looking at data variation
            # Calculate coefficient of variation (std/mean) as complexity indicator
            relative_variation = np.std(y_corrected) / np.mean(y_corrected) if np.mean(y_corrected) > 0 else 0
            
            # If data has high variation, there might be multiple peaks
            if relative_variation > 0.3:  # Threshold for "complex" data
                st.info(f"""
                ℹ️ **Only 1 peak detected, but data shows complexity**
                
                Your data may contain multiple overlapping peaks that weren't detected.
                
                **Try adjusting peak detection:**
                - **Lower min height** (currently {detection_params['min_height_percent']}% → try 3-5%)
                - **Lower min prominence** (currently {detection_params['min_prominence_percent']}% → try 1-2%)
                - **Reduce min distance** (currently {detection_params['min_distance_percent']}% → try 2-3%)
                - **Increase smoothing** (currently {detection_params['smoothing_points']} → try 7-10 points)
                """)
    
    # Parameter estimation
    try:
        estimator = ParameterEstimator()
        initial_params = estimator.estimate_parameters(
            x_data, y_corrected, peak_info, fitting_options['peak_type']
        )
    except Exception as e:
        st.error(f"**Parameter estimation failed:** {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.write(f"Peak Type: {fitting_options['peak_type']}")
            st.write(f"Number of detected peaks: {len(peak_info)}")
            st.write(f"Data range: {x_data.min():.2f} - {x_data.max():.2f}")
        st.info("💡 Try adjusting peak detection settings or using manual peak specification")
        return None
    
    # Create parameter manager
    try:
        x_range = (x_data.min(), x_data.max())
        param_manager = ParameterManager(
            fitting_options['peak_type'],
            initial_params,
            x_range
        )
    except Exception as e:
        st.error(f"**Parameter manager initialization failed:** {str(e)}")
        st.info(f"💡 This usually means the initial parameters are invalid for {fitting_options['peak_type']} peaks")
        return None
    
    # Setup fitting engine
    try:
        engine = FittingEngine()
        engine.set_fitting_options(
            peak_type=fitting_options['peak_type'],
            baseline_type="None",  # Already subtracted
            fit_method=fitting_options['fit_method'],
            max_iterations=fitting_options['max_iterations'],
            tolerance=fitting_options['tolerance'],
            use_weights=fitting_options['use_weights']
        )
        engine.set_parameter_manager(param_manager)
    except Exception as e:
        st.error(f"**Fitting engine setup failed:** {str(e)}")
        st.info("💡 Check that all fitting options are valid")
        return None
    
    # Perform fitting
    try:
        weights = None
        if fitting_options['use_weights']:
            weights = processor.calculate_weights(y_corrected)
        
        result = engine.fit_peaks(x_data, y_corrected, initial_params, weights=weights)
    except Exception as e:
        st.error(f"**Fitting execution failed:** {str(e)}")
        with st.expander("🔍 Debug Information", expanded=True):
            st.write(f"Peak type: {fitting_options['peak_type']}")
            st.write(f"Number of peaks: {len(peak_info)}")
            st.write(f"Fitting method: {fitting_options['fit_method']}")
            st.write(f"Max iterations: {fitting_options['max_iterations']}")
            st.write(f"Using weights: {fitting_options['use_weights']}")
        st.info("💡 This error occurred during the optimization process. Try simpler settings.")
        return None
    
    if not result['success']:
        error_msg = result.get('error', result.get('message', 'Unknown error'))
        st.error(f"**Fitting failed:** {error_msg}")
        
        # Show detailed debugging information in expander
        with st.expander("🔍 Debug Information", expanded=True):
            st.write(f"**Peak Type:** {fitting_options['peak_type']}")
            st.write(f"**Number of Peaks:** {len(peak_info)}")
            st.write(f"**Fitting Method:** {fitting_options['fit_method']}")
            st.write(f"**Data Points:** {len(x_data)}")
            st.write(f"**CCS Range:** {x_data.min():.2f} - {x_data.max():.2f}")
            st.write(f"**Intensity Range:** {y_corrected.min():.2e} - {y_corrected.max():.2e}")
            
            if peak_info:
                st.write("**Detected Peak Positions:**")
                peak_positions = [f"{p['x']:.2f}" for p in peak_info]
                st.write(", ".join(peak_positions))
        
        st.warning("**💡 Suggestions:**")
        st.markdown("""
        - **Try Gaussian peak type** (most robust for CCSD data)
        - **Reduce number of peaks** - Remove minor peaks or increase detection thresholds
        - **Check data quality** - Ensure sufficient signal-to-noise ratio
        - **Adjust detection parameters** - Use higher minimum height/prominence
        - **Simplify baseline** - Try 'None' or 'Linear' baseline correction
        - **Check CCS range** - Ensure you're fitting a reasonable region
        """)
        return None
    
    # ========== VALIDATE FIT QUALITY ==========
    # Check for obviously poor fits that should be rejected
    r_squared = result.get('r_squared', 0)
    
    # Warn if fit quality is poor (R² < 0.7 is concerning for peak data)
    if r_squared < 0.7:
        st.warning(f"⚠️ **Poor fit quality detected (R² = {r_squared:.3f})**")
        st.info("""
        **The fit has very low R² which suggests:**
        - Peak detection may have missed important peaks
        - Wrong peak type selected
        - Data has high noise or artifacts
        
        **Try:**
        - Adjust peak detection to find more peaks (lower min height/prominence)
        - Increase smoothing in preprocessing
        - Try different peak types (Voigt for complex shapes)
        - Check if baseline correction is appropriate
        """)
    
    # Check if fitted peaks are within reasonable bounds
    params_per_peak = engine.get_params_per_peak()
    n_peaks = len(result['parameters']) // params_per_peak
    x_range = x_data.max() - x_data.min()
    
    peaks_outside_data = 0
    for i in range(n_peaks):
        center_idx = i * params_per_peak + 1
        peak_center = result['parameters'][center_idx]
        # Check if peak center is far outside the data range
        if peak_center < x_data.min() - x_range * 0.01 or peak_center > x_data.max() + x_range * 0.01:
            peaks_outside_data += 1
    
    if peaks_outside_data > 0:
        st.warning(f"⚠️ **{peaks_outside_data} peak(s) fitted outside the data range**")
        st.info("This suggests the fit may have converged to an unrealistic solution. Try adjusting initial peak positions.")
    
    # Analyze results
    try:
        analyzer = ResultAnalyzer()
        peak_stats = analyzer.calculate_peak_statistics(
            x_data,
            y_corrected,
            result['fitted_curve'],
            result['parameters'],
            fitting_options['peak_type']
        )
    except Exception as e:
        st.error(f"**Result analysis failed:** {str(e)}")
        st.warning("⚠️ Fit may have succeeded but statistics calculation failed. You can still view the fitted curve.")
        # Return partial result if possible
        if result.get('success'):
            st.info("💡 Continuing with limited statistics...")
            peak_stats = []  # Empty stats
        else:
            return None
    
    # Add peak components to result
    try:
        n_peaks = len(peak_info)
        params_per_peak = get_params_per_peak(fitting_options['peak_type'])
        peak_components = []
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            end_idx = (i + 1) * params_per_peak
            peak_params = result['parameters'][start_idx:end_idx]
            
            component = multi_peak_function(
                x_data,
                fitting_options['peak_type'],
                *peak_params
            )
            peak_components.append(component)
        
        result['peak_components'] = peak_components
        result['baseline'] = baseline
        result['peak_info'] = peak_info
        result['y_corrected'] = y_corrected
    except Exception as e:
        st.error(f"Peak component calculation failed: {str(e)}")
        st.warning("Fit succeeded but component visualization may not work")
        # Still return result without components
        result['peak_components'] = []
        result['baseline'] = baseline
        result['peak_info'] = peak_info
        result['y_corrected'] = y_corrected
    
    return result, param_manager, peak_stats


def main():
    """Main Streamlit application with intuitive step-by-step workflow."""
    # App banner
    st.markdown('<div class="app-banner">🧰 IMSocio</div>', unsafe_allow_html=True)
    
    FitDataUI.show_main_header()
    
    # Initialize session state with charge-specific tracking
    if 'all_charge_results' not in st.session_state:
        st.session_state['all_charge_results'] = {}
    if 'current_analysis_key' not in st.session_state:
        st.session_state['current_analysis_key'] = None
    
    # ========== STEP 1: DATA UPLOAD ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📁 Step 1: Upload Data</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your calibrated CCSD data (CSV format)",
        type=['csv'],
        help="File must contain: Charge, CCS, and Scaled_Intensity columns"
    )
    
    if uploaded_file is None:
        st.info("👆 **Start here!** Upload your calibrated CCSD CSV file to begin peak fitting.")
        with st.expander("ℹ️ Required CSV Format", expanded=False):
            st.markdown("""
            Your CSV file must include these columns:
            - **Charge** - Charge state values (integer)
            - **CCS** - Collision Cross Section values (Ų)
            - **Scaled_Intensity** - Intensity values (arbitrary units)
            
            Optional columns (needed for export):
            - **Drift** - Drift time values (ms)
            - **m/z** - Mass-to-charge ratio
            """)
        return
    
    # Load and validate data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean and validate data
        if 'CCS' in df.columns:
            df['CCS'] = pd.to_numeric(df['CCS'].astype(str).str.replace(',', ''), errors='coerce')
        if 'Scaled_Intensity' in df.columns:
            df['Scaled_Intensity'] = pd.to_numeric(df['Scaled_Intensity'], errors='coerce')
        if 'Charge' in df.columns:
            df['Charge'] = pd.to_numeric(df['Charge'], errors='coerce', downcast='integer')
        
        df = df.dropna(subset=['Charge', 'CCS', 'Scaled_Intensity'])
        
        required_cols = ['Charge', 'CCS', 'Scaled_Intensity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            return
        
        if df.empty:
            st.error("❌ The uploaded file is empty or has no valid data rows")
            return
            
        st.success(f"✅ Data loaded: {len(df)} data points from {len(df['Charge'].unique())} charge states")
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return
    
    # ========== STEP 2: SELECT DATA TO ANALYZE ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">🎯 Step 2: Select Data to Analyze</div>
    </div>
    """, unsafe_allow_html=True)
    
    charges = sorted(df['Charge'].unique())
    ccs_min_raw = float(df['CCS'].min())
    ccs_max_raw = float(df['CCS'].max())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analysis Mode**")
        mode = st.radio(
            "Choose what to analyze:",
            ["Individual Charge State", "Summed Data"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if mode == "Individual Charge State":
            selected_charge = st.selectbox("Select Charge State:", charges)
            plot_data = df[df['Charge'] == selected_charge].copy().sort_values('CCS')
            data_label = f"Charge {selected_charge}"
            analysis_key = f"charge_{selected_charge}"
        else:
            # Create summed data ensuring proper column structure
            plot_data = CCSDDataProcessor.create_summed_data(df)
            # Verify summed data has required columns
            if 'CCS' not in plot_data.columns or 'Scaled_Intensity' not in plot_data.columns:
                st.error("❌ Summed data creation failed - missing required columns")
                st.write("Available columns:", plot_data.columns.tolist())
                return
            # Ensure data types are correct
            plot_data['CCS'] = pd.to_numeric(plot_data['CCS'], errors='coerce')
            plot_data['Scaled_Intensity'] = pd.to_numeric(plot_data['Scaled_Intensity'], errors='coerce')
            plot_data = plot_data.dropna(subset=['CCS', 'Scaled_Intensity'])
            
            data_label = "Summed Data"
            selected_charge = None
            analysis_key = "summed"
    
    # Clear old fit results if analysis selection changed
    if st.session_state['current_analysis_key'] != analysis_key:
        # Clear all fit-related session state
        for key in ['fit_result', 'fit_result_analysis_key', 'parameter_manager', 'peak_stats', 'fitting_options']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['current_analysis_key'] = analysis_key
        st.info(f"📝 **Switched to {data_label}** - Previous fit cleared. Configure new fitting below.")
    
    with col2:
        st.markdown("**CCS Range**")
        if 'fit_ccs_min' not in st.session_state:
            st.session_state['fit_ccs_min'] = ccs_min_raw
        if 'fit_ccs_max' not in st.session_state:
            st.session_state['fit_ccs_max'] = ccs_max_raw
        
        # Validate and reset session state if out of bounds
        if st.session_state['fit_ccs_min'] < ccs_min_raw or st.session_state['fit_ccs_min'] > ccs_max_raw:
            st.session_state['fit_ccs_min'] = ccs_min_raw
        if st.session_state['fit_ccs_max'] > ccs_max_raw or st.session_state['fit_ccs_max'] < ccs_min_raw:
            st.session_state['fit_ccs_max'] = ccs_max_raw
            
        ccs_min = st.number_input(
            "Minimum CCS:",
            min_value=ccs_min_raw,
            max_value=ccs_max_raw,
            value=float(st.session_state['fit_ccs_min']),
            step=max((ccs_max_raw - ccs_min_raw) / 100, 0.1),
            format="%.2f"
        )
        ccs_max = st.number_input(
            "Maximum CCS:",
            min_value=ccs_min,
            max_value=ccs_max_raw,
            value=float(st.session_state['fit_ccs_max']),
            step=max((ccs_max_raw - ccs_min_raw) / 100, 0.1),
            format="%.2f"
        )
        
        st.session_state['fit_ccs_min'] = ccs_min
        st.session_state['fit_ccs_max'] = ccs_max
    
    # Apply filters
    plot_data = plot_data[(plot_data['CCS'] >= ccs_min) & (plot_data['CCS'] <= ccs_max)].copy()
    plot_data = plot_data[plot_data['Scaled_Intensity'] > 0].copy()
    
    if len(plot_data) == 0:
        st.error("❌ No data points in selected range. Adjust your CCS range.")
        return
    
    # Ensure plot_data is properly sorted and clean
    plot_data = plot_data.sort_values('CCS').reset_index(drop=True)
    
    st.info(f"📊 Analyzing **{data_label}** with **{len(plot_data)} data points** in range {ccs_min:.1f} - {ccs_max:.1f} Ų")
    
    # Show debug info for summed data to help troubleshoot
    if mode == "Summed Data":
        with st.expander("ℹ️ Summed Data Info", expanded=False):
            st.write(f"**Total points in summed data:** {len(plot_data)}")
            st.write(f"**CCS range:** {plot_data['CCS'].min():.2f} - {plot_data['CCS'].max():.2f}")
            st.write(f"**Intensity range:** {plot_data['Scaled_Intensity'].min():.2e} - {plot_data['Scaled_Intensity'].max():.2e}")
            st.write(f"**Original charge states summed:** {len(df['Charge'].unique())}")
    
    # Show saved status for current charge
    if mode == "Individual Charge State":
        if selected_charge in st.session_state['all_charge_results']:
            st.success(f"✅ Charge {selected_charge} has saved results. You can re-fit to update or change to another charge state.")
        else:
            if st.session_state['all_charge_results']:
                saved_charges = [c for c in st.session_state['all_charge_results'].keys() if c != 'summed']
                st.info(f"💾 Saved charge states: {', '.join(map(str, sorted(saved_charges)))} | Current: {selected_charge} (not saved yet)")
    else:  # Summed Data mode
        if 'summed' in st.session_state['all_charge_results']:
            st.success(f"✅ Summed Data has saved results. You can re-fit to update or change to charge states.")
        else:
            if st.session_state['all_charge_results']:
                saved_charges = [c for c in st.session_state['all_charge_results'].keys() if c != 'summed']
                if saved_charges:
                    st.info(f"💾 Saved charge states: {', '.join(map(str, sorted(saved_charges)))} | Current: Summed (not saved yet)")
    
    # Show data preview FIRST (always expanded) so users can see peaks before specifying centroids
    st.markdown("### 👁️ Data Preview")
    preview_fig = go.Figure()
    preview_fig.add_trace(go.Scatter(
        x=plot_data['CCS'],
        y=plot_data['Scaled_Intensity'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=1.5),
        marker=dict(size=2),
        name='Data'
    ))
    preview_fig.update_layout(
        title=f"Current Data: {data_label}",
        xaxis_title="CCS (Ų)",
        yaxis_title="Scaled Intensity",
        height=400,
        showlegend=False,
        hovermode='closest'
    )
    st.plotly_chart(preview_fig, use_container_width=True)
    st.markdown("💡 **Look at the preview above to identify peak positions before configuring peak detection below**")
    
    # ========== STEP 3: CONFIGURE FITTING ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">⚙️ Step 3: Configure Fitting Parameters</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract data arrays with validation
    try:
        x_data = plot_data['CCS'].values
        y_data = plot_data['Scaled_Intensity'].values
        
        if len(x_data) == 0 or len(y_data) == 0:
            st.error("❌ No valid data points available for fitting")
            return
            
        if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
            st.warning("⚠️ Data contains NaN values - cleaning automatically")
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
    except KeyError as e:
        st.error(f"❌ Required column missing from data: {e}")
        st.info("Expected columns: 'CCS' and 'Scaled_Intensity'")
        return
    
    # Get UI options in organized tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Peak Detection", "⚙️ Fitting Options", "🔧 Advanced"])
    
    with tab1:
        manual_centroids, detection_params = FitDataUI.show_peak_detection_controls()
    
    with tab2:
        fitting_options = FitDataUI.show_fitting_options()
    
    with tab3:
        preprocessing_options = FitDataUI.show_preprocessing_options()
    
    # Store options in session state
    st.session_state['fitting_options'] = fitting_options
    st.session_state['data_label'] = data_label
    
    # ========== STEP 4: PERFORM FITTING ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">🚀 Step 4: Run Peak Fitting</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ **Perform Fitting**", type="primary", use_container_width=True):
        # Clear any previous fit results for clean state
        for key in ['fit_result', 'fit_result_analysis_key', 'parameter_manager', 'peak_stats']:
            if key in st.session_state:
                del st.session_state[key]
            
        with st.spinner("🔄 Fitting peaks... This may take a moment."):
            result = perform_fitting(
                x_data, y_data,
                fitting_options,
                detection_params=detection_params,
                preprocessing_options=preprocessing_options,
                manual_peaks=manual_centroids if manual_centroids else None
            )
            
            if result:
                fit_result, param_manager, peak_stats = result
                
                # Store results with the current analysis_key to prevent cross-contamination
                st.session_state['fit_result'] = fit_result
                st.session_state['fit_result_analysis_key'] = analysis_key  # Track which data this fit belongs to
                st.session_state['parameter_manager'] = param_manager
                st.session_state['peak_stats'] = peak_stats if peak_stats is not None else []
                st.success("✅ **Fitting completed successfully!** Scroll down to view results.")
    
    # ========== STEP 5: VIEW RESULTS ==========
    # Check if fit results exist AND match the current analysis
    if 'fit_result' not in st.session_state:
        st.info("👆 Click **Perform Fitting** above to start the analysis")
        return
    
    # CRITICAL: Verify fit results belong to current analysis (not from a previous charge state)
    stored_analysis_key = st.session_state.get('fit_result_analysis_key', None)
    if stored_analysis_key != analysis_key:
        st.info(f"👆 Click **Perform Fitting** above to analyze {data_label}")
        st.warning(f"⚠️ Previous fit results are from a different analysis and have been hidden.")
        return
    
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📊 Step 5: Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    result = st.session_state['fit_result']
    peak_stats = st.session_state.get('peak_stats', [])
    
    # Display statistics
    st.markdown("#### Fit Quality Metrics")
    FitDataUI.display_fit_statistics(result, peak_stats)
    
    # Display plot
    st.markdown("#### Visualization")
    show_components = st.checkbox("Show individual peak components", value=True)
    fig = FitDataUI.create_fit_plot(
        x_data, y_data, result,
        baseline=result.get('baseline'),
        show_components=show_components
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak table
    if peak_stats:
        FitDataUI.display_peak_table(peak_stats)
    
    # ========== STEP 6: REFINE (OPTIONAL) ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">🎚️ Step 6: Refine Fit (Optional)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['parameter_manager']:
        # Sync parameter_manager with current fitted parameters
        if 'fit_result' in st.session_state and 'parameters' in st.session_state['fit_result']:
            current_fit_params = st.session_state['fit_result']['parameters']
            pm = st.session_state['parameter_manager']
            
            # Update parameter manager if out of sync
            if not np.array_equal(pm.parameters, current_fit_params):
                # Clear parameter widget states
                keys_to_delete = [key for key in st.session_state.keys() 
                                 if key.startswith(('param_', 'fix_', 'constrain_'))]
                for key in keys_to_delete:
                    del st.session_state[key]
                
                pm.parameters = np.array(current_fit_params, copy=True)
                st.session_state['parameter_manager'] = pm
                st.rerun()
        
        with st.expander("✏️ Edit Parameters and Re-fit", expanded=False):
            st.info("💡 Adjust individual peak parameters below and click **Re-fit** to improve the fit.")
            FitDataUI.show_parameter_editor(
                st.session_state['parameter_manager'],
                x_data,
                y_data
            )
            
            if st.button("🔄 Re-fit with Edited Parameters", type="secondary"):
                with st.spinner("🔄 Re-fitting with updated parameters..."):
                    # Get updated parameters from parameter manager
                    param_manager = st.session_state['parameter_manager']
                    
                    # Debug info
                    st.info(f"Re-fitting with {param_manager.n_peaks} peaks. Fixed parameters: {len(param_manager.fixed_params)}")
                    
                    # Setup engine with parameter manager
                    engine = FittingEngine()
                    engine.set_fitting_options(
                        peak_type=fitting_options['peak_type'],
                        baseline_type="None",
                        fit_method=fitting_options['fit_method'],
                        max_iterations=fitting_options['max_iterations'],
                        tolerance=fitting_options['tolerance'],
                        use_weights=fitting_options['use_weights']
                    )
                    engine.set_parameter_manager(param_manager)
                    
                    # Re-fit using the parameter manager
                    # Note: fit_peaks will use param_manager internally for fixed params and bounds
                    weights = None
                    if fitting_options['use_weights']:
                        processor = DataProcessor()
                        weights = processor.calculate_weights(result['y_corrected'])
                    
                    # Pass the current parameters from the manager
                    new_result = engine.fit_peaks(
                        x_data,
                        result['y_corrected'],
                        param_manager.parameters,
                        weights=weights
                    )
                    
                    if new_result['success']:
                        # Update result
                        analyzer = ResultAnalyzer()
                        peak_stats = analyzer.calculate_peak_statistics(
                            x_data,
                            result['y_corrected'],
                            new_result['fitted_curve'],
                            new_result['parameters'],
                            fitting_options['peak_type']
                        )
                        
                        # Add components
                        n_peaks = param_manager.n_peaks
                        params_per_peak = get_params_per_peak(fitting_options['peak_type'])
                        peak_components = []
                        
                        for i in range(n_peaks):
                            start_idx = i * params_per_peak
                            end_idx = (i + 1) * params_per_peak
                            peak_params = new_result['parameters'][start_idx:end_idx]
                            
                            component = multi_peak_function(
                                x_data,
                                fitting_options['peak_type'],
                                *peak_params
                            )
                            peak_components.append(component)
                        
                        new_result['peak_components'] = peak_components
                        new_result['baseline'] = result['baseline']
                        new_result['peak_info'] = result['peak_info']
                        new_result['y_corrected'] = result['y_corrected']
                        
                        # Update parameter manager with new fitted parameters
                        param_manager.parameters = new_result['parameters'].copy()
                        
                        st.session_state['fit_result'] = new_result
                        st.session_state['fit_result_analysis_key'] = analysis_key  # Maintain analysis tracking
                        st.session_state['parameter_manager'] = param_manager
                        # Ensure peak_stats is never None
                        st.session_state['peak_stats'] = peak_stats if peak_stats is not None else []
                        
                        # Show what changed
                        with st.expander("📊 Parameter Changes"):
                            param_names = get_parameter_names(fitting_options['peak_type'])
                            for i in range(n_peaks):
                                st.write(f"**Peak {i+1}:**")
                                for j, pname in enumerate(param_names):
                                    idx = i * params_per_peak + j
                                    st.write(f"  - {pname}: {new_result['parameters'][idx]:.4f}")
                        
                        st.success("✅ Re-fitting completed! Updated results displayed below.")
                        st.rerun()
                    else:
                        error_msg = new_result.get('error', 'Unknown error')
                        st.error(f"❌ Re-fitting failed: {error_msg}")
                        with st.expander("🔍 Debug Information"):
                            st.write("**Error Details:**")
                            st.write(new_result)
    else:
        st.info("💡 No parameter manager available for editing.")
    
    # ========== STEP 7: SAVE RESULTS ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">💾 Step 7: Save Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if mode == "Individual Charge State":
            st.write(f"Save the fit results for **Charge {selected_charge}** to include in the final export.")
        else:
            st.write(f"Save the fit results for **Summed Data** to include in the final export.")
    with col2:
        if mode == "Individual Charge State":
            if st.button(f"💾 **Save Charge {selected_charge}**", type="primary", use_container_width=True):
                # Safety check: Verify fit results match current analysis
                stored_analysis_key = st.session_state.get('fit_result_analysis_key', None)
                if stored_analysis_key != analysis_key:
                    st.error("❌ Cannot save - fit results don't match current analysis. Please re-fit.")
                    return
                
                # Deep copy all fit data to preserve it independently
                fit_result_to_save = {}
                for key, value in st.session_state['fit_result'].items():
                    if isinstance(value, np.ndarray):
                        fit_result_to_save[key] = value.copy()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        fit_result_to_save[key] = [comp.copy() for comp in value]
                    else:
                        fit_result_to_save[key] = value
                
                st.session_state['all_charge_results'][selected_charge] = {
                    'fit_result': fit_result_to_save,
                    'peak_stats': copy.deepcopy(st.session_state.get('peak_stats', [])),
                    'fitting_options': st.session_state['fitting_options'].copy(),
                    'parameter_manager': copy.deepcopy(st.session_state['parameter_manager']),
                    'data_info': {
                        'charge': selected_charge,
                        'n_points': len(plot_data),
                        'ccs_range': (plot_data['CCS'].min(), plot_data['CCS'].max()),
                        'analysis_key': analysis_key  # Store for verification
                    }
                }
                
                st.success(f"✅ Saved results for Charge {selected_charge}!")
                st.rerun()
        else:
            if st.button(f"💾 **Save Summed Data**", type="primary", use_container_width=True):
                # Safety check: Verify fit results match current analysis
                stored_analysis_key = st.session_state.get('fit_result_analysis_key', None)
                if stored_analysis_key != 'summed':
                    st.error("❌ Cannot save - fit results don't match current analysis. Please re-fit.")
                    return
                
                # Deep copy all fit data
                fit_result_to_save = {}
                for key, value in st.session_state['fit_result'].items():
                    if isinstance(value, np.ndarray):
                        fit_result_to_save[key] = value.copy()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        fit_result_to_save[key] = [comp.copy() for comp in value]
                    else:
                        fit_result_to_save[key] = value
                
                st.session_state['all_charge_results']['summed'] = {
                    'fit_result': fit_result_to_save,
                    'peak_stats': copy.deepcopy(st.session_state.get('peak_stats', [])),
                    'fitting_options': st.session_state['fitting_options'].copy(),
                    'parameter_manager': copy.deepcopy(st.session_state['parameter_manager']),
                    'data_info': {
                        'charge': 'summed',
                        'n_points': len(plot_data),
                        'ccs_range': (float(plot_data['CCS'].min()), float(plot_data['CCS'].max())),
                        'analysis_key': 'summed'  # Store for verification
                    }
                }
                
                # Show confirmation with peak count
                n_peaks = len(st.session_state.get('peak_stats', []))
                st.success(f"✅ Saved Summed Data with {n_peaks} peak(s)!")
                st.rerun()
    
    # Show saved results summary
    if st.session_state['all_charge_results']:
        st.markdown("#### 📋 Saved Results Summary")
        
        # Sort keys: numeric charges first, then 'summed' at the end
        sorted_keys = sorted([k for k in st.session_state['all_charge_results'].keys() if k != 'summed'])
        if 'summed' in st.session_state['all_charge_results']:
            sorted_keys.append('summed')
        
        for charge in sorted_keys:
            result_data = st.session_state['all_charge_results'][charge]
            peak_stats = result_data.get('peak_stats', [])
            fit_result = result_data.get('fit_result', {})
            fitting_options = result_data.get('fitting_options', {})
            
            # Get peak count
            if peak_stats and len(peak_stats) > 0:
                n_peaks = len(peak_stats)
            elif 'parameter_manager' in result_data and result_data['parameter_manager']:
                n_peaks = result_data['parameter_manager'].n_peaks
            else:
                if 'parameters' in fit_result and 'peak_type' in fitting_options:
                    params_per_peak = get_params_per_peak(fitting_options['peak_type'])
                    n_peaks = len(fit_result['parameters']) // params_per_peak
                else:
                    n_peaks = 0
            
            r_squared = fit_result.get('r_squared', 0.0)
            display_charge = "Summed" if charge == 'summed' else charge
            
            # Create expandable section for each charge state
            with st.expander(f"**Charge {display_charge}** - {n_peaks} peak(s), R² = {r_squared:.4f}"):
                if n_peaks > 0 and 'parameters' in fit_result:
                    # Extract peak parameters
                    params_per_peak = get_params_per_peak(fitting_options.get('peak_type', 'Gaussian'))
                    param_names = get_parameter_names(fitting_options.get('peak_type', 'Gaussian'))
                    
                    peak_details = []
                    for i in range(n_peaks):
                        peak_dict = {'Peak': i + 1}
                        
                        # Extract amplitude, center, and width(s)
                        base_idx = i * params_per_peak
                        amplitude = fit_result['parameters'][base_idx]
                        center = fit_result['parameters'][base_idx + 1]
                        
                        peak_dict['Amplitude'] = f"{amplitude:.2f}"
                        peak_dict['Centroid (CCS)'] = f"{center:.2f}"
                        
                        # Handle different peak types
                        if params_per_peak == 3:  # Gaussian or Lorentzian
                            width = fit_result['parameters'][base_idx + 2]
                            peak_dict['Width (σ)'] = f"{width:.3f}"
                        elif params_per_peak == 4:  # Voigt, BiGaussian, EMG
                            width1 = fit_result['parameters'][base_idx + 2]
                            width2 = fit_result['parameters'][base_idx + 3]
                            peak_dict['Width 1'] = f"{width1:.3f}"
                            peak_dict['Width 2'] = f"{width2:.3f}"
                        
                        # Add FWHM from peak_stats if available
                        if peak_stats and i < len(peak_stats):
                            fwhm = peak_stats[i].get('fwhm', None)
                            if fwhm:
                                peak_dict['FWHM'] = f"{fwhm:.3f}"
                        
                        peak_details.append(peak_dict)
                    
                    st.dataframe(pd.DataFrame(peak_details), use_container_width=True, hide_index=True)
                else:
                    st.info("No peak data available")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Saved Results"):
            st.session_state['all_charge_results'] = {}
            st.rerun()
    
    # ========== STEP 8: EXPORT ALL RESULTS ==========
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📤 Step 8: Export All Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state['all_charge_results']:
        st.info("💡 Save results for individual charge states first (Step 7), then export all saved results here.")
    else:
        st.write(f"**{len(st.session_state['all_charge_results'])} charge state(s)** ready for export")
        
        # Export points configuration
        export_points = st.number_input(
            "Points per charge in export:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of evenly spaced CCS points for each charge state",
            key="export_points_final"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export combined fitted data
            combined_data = FitDataUI.export_fitted_data(
                st.session_state['all_charge_results'],
                export_points
            )
            if combined_data is not None:
                csv_combined = combined_data.to_csv(index=False)
                st.download_button(
                    "📊 **Export All Charges**",
                    csv_combined,
                    "fitted_data_all_charges.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col2:
            # Export parameters summary
            all_params_data = []
            
            for charge in sorted(st.session_state['all_charge_results'].keys()):
                result_data = st.session_state['all_charge_results'][charge]
                result = result_data['fit_result']
                peak_stats = result_data.get('peak_stats', [])
                # Use the saved fitting options for THIS charge state
                fitting_options_saved = result_data.get('fitting_options', fitting_options)
                params_per_peak = get_params_per_peak(fitting_options_saved['peak_type'])
                param_names = get_parameter_names(fitting_options_saved['peak_type'])
                
                # Calculate number of peaks from parameters array
                n_peaks = len(result['parameters']) // params_per_peak if 'parameters' in result else 0
                
                for i in range(n_peaks):
                    peak_params = {'Charge': charge, 'Peak': i + 1}
                    for j, param_name in enumerate(param_names):
                        peak_params[param_name] = result['parameters'][i * params_per_peak + j]
                    all_params_data.append(peak_params)
            
            if all_params_data:
                param_df = pd.DataFrame(all_params_data)
                csv_params = param_df.to_csv(index=False)
                st.download_button(
                    "📋 **Export Parameters**",
                    csv_params,
                    "fit_parameters_summary.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Export comprehensive metrics with Gaussian parameters
            metrics_data = []
            for charge in sorted(st.session_state['all_charge_results'].keys()):
                result_data = st.session_state['all_charge_results'][charge]
                result = result_data['fit_result']
                fitting_options_saved = result_data.get('fitting_options', {})
                peak_stats = result_data.get('peak_stats', [])
                
                # Get number of peaks
                if peak_stats and len(peak_stats) > 0:
                    n_peaks = len(peak_stats)
                elif 'parameter_manager' in result_data and result_data['parameter_manager']:
                    n_peaks = result_data['parameter_manager'].n_peaks
                else:
                    params_per_peak_calc = get_params_per_peak(fitting_options_saved.get('peak_type', 'Gaussian'))
                    n_peaks = len(result['parameters']) // params_per_peak_calc
                
                # Create one row per peak with all its parameters
                params_per_peak_calc = get_params_per_peak(fitting_options_saved.get('peak_type', 'Gaussian'))
                param_names = get_parameter_names(fitting_options_saved.get('peak_type', 'Gaussian'))
                
                for i in range(n_peaks):
                    peak_row = {
                        'Charge': charge,
                        'Peak': i + 1,
                        'R_Squared': result.get('r_squared', 0.0),
                        'RMSE': result.get('rmse', 0.0)
                    }
                    
                    # Add all Gaussian/peak parameters
                    for j, param_name in enumerate(param_names):
                        param_idx = i * params_per_peak_calc + j
                        if param_idx < len(result['parameters']):
                            peak_row[param_name] = result['parameters'][param_idx]
                    
                    # Add peak statistics if available
                    if peak_stats and i < len(peak_stats):
                        peak_row['Area'] = peak_stats[i].get('area', 0.0)
                        peak_row['Height'] = peak_stats[i].get('height', 0.0)
                        peak_row['FWHM'] = peak_stats[i].get('fwhm', 0.0)
                    
                    metrics_data.append(peak_row)
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                csv_metrics = metrics_df.to_csv(index=False)
                st.download_button(
                    "📈 **Export Gaussian Fits**",
                    csv_metrics,
                    "fit_metrics_summary.csv",
                    "text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()

