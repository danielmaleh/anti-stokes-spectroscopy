"""
Raman Spectroscopy Data Processing Tool

This module provides comprehensive analysis of Raman spectroscopy data including:
- Silicon-based wavelength calibration
- Cosmic ray removal
- Background subtraction
- Stokes/Anti-Stokes separation
- FFT analysis and peak detection
- Polarization analysis (co/cross-polarized)
- Automated PDF report generation

Author: Daniel Abraham Elmaleh
Institution: EPFL - NAM Lab
Project: Anti-Stokes Spectroscopy of Crystalline Plasmonic Metals
Year: 2024
"""

import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, fftfreq
from tkinter import Tk, filedialog, simpledialog
from fpdf import FPDF


# =============================================================================
# Constants
# =============================================================================

# Standard Raman shift for silicon reference peak (cm^-1)
SILICON_PEAK_CM = 520

# Laser wavelengths (nm)
LASER_GREEN = 532.0
LASER_RED = 633.0

# Default filter window around laser line (nm)
DEFAULT_LASER_WINDOW = 5.0

# Anti-Stokes range below laser line (nm)
ANTI_STOKES_RANGE = 45.0


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_spectrum_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Raman spectrum data from CSV file.

    Args:
        file_path: Path to CSV file with wavelength and count columns.

    Returns:
        Tuple of (wavelength_array, count_array).
    """
    data = pd.read_csv(
        file_path,
        delimiter=",",
        header=None,
        names=["wavelength", "count"]
    )
    data["wavelength"] = pd.to_numeric(data["wavelength"], errors="coerce")
    data["count"] = pd.to_numeric(data["count"], errors="coerce")
    data.dropna(inplace=True)

    return data["wavelength"].values, data["count"].values


def remove_cosmic_rays(
    wavelength: np.ndarray,
    counts: np.ndarray,
    window_size: int = 5,
    sigma: float = 5.0,
    slope_factor: float = 5.0
) -> np.ndarray:
    """
    Remove cosmic ray artifacts from spectrum data.

    Uses a combination of local outlier detection and slope-based checks
    to identify and replace sharp spikes caused by cosmic rays.

    Args:
        wavelength: Wavelength array (unused, kept for API consistency).
        counts: Count/intensity array.
        window_size: Size of local window for outlier detection.
        sigma: Number of standard deviations for outlier threshold.
        slope_factor: Factor for slope-based spike detection.

    Returns:
        Cleaned count array with cosmic rays removed.
    """
    cleaned_counts = counts.copy()
    n = len(counts)

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        local_data = counts[start:end]

        local_mean = np.mean(local_data)
        local_std = np.std(local_data)

        if local_std == 0:
            continue

        # Sigma-based outlier detection
        if abs(counts[i] - local_mean) > sigma * local_std:
            cleaned_counts[i] = local_mean
            continue

        # Slope-based spike detection
        if 0 < i < n - 1:
            left_slope = abs(counts[i] - counts[i - 1])
            right_slope = abs(counts[i] - counts[i + 1])
            local_slope_mean = np.mean(np.abs(np.diff(local_data)))

            if (left_slope > slope_factor * local_slope_mean or
                    right_slope > slope_factor * local_slope_mean):
                cleaned_counts[i] = local_mean

    return cleaned_counts


def smooth_signal(
    counts: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter to spectrum.

    Args:
        counts: Count/intensity array.
        window_length: Length of the filter window (must be odd).
        polyorder: Order of polynomial for fitting.

    Returns:
        Smoothed count array.
    """
    return savgol_filter(counts, window_length, polyorder)


def subtract_background(
    wavelength: np.ndarray,
    counts: np.ndarray,
    bg_wavelength: np.ndarray,
    bg_counts: np.ndarray
) -> np.ndarray:
    """
    Subtract interpolated background from spectrum.

    Args:
        wavelength: Sample wavelength array.
        counts: Sample count array.
        bg_wavelength: Background wavelength array.
        bg_counts: Background count array.

    Returns:
        Background-corrected count array (shifted so minimum is 0).
    """
    bg_interpolated = np.interp(wavelength, bg_wavelength, bg_counts)
    corrected = counts - bg_interpolated

    # Shift negative values to zero
    min_val = np.min(corrected)
    if min_val < 0:
        corrected -= min_val

    return corrected


def cut_laser_region(
    wavelength: np.ndarray,
    counts: np.ndarray,
    laser_wavelength: float,
    window: float = DEFAULT_LASER_WINDOW
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the region around the laser line from spectrum.

    Args:
        wavelength: Wavelength array.
        counts: Count array.
        laser_wavelength: Center wavelength of laser line.
        window: Half-width of region to remove.

    Returns:
        Tuple of (wavelength, counts) with laser region removed.
    """
    mask = (wavelength < laser_wavelength - window) | (wavelength > laser_wavelength + window)
    return wavelength[mask], counts[mask]


# =============================================================================
# Peak Detection and FFT Analysis
# =============================================================================

def detect_peaks(
    wavelength: np.ndarray,
    counts: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks in spectrum data.

    Args:
        wavelength: Wavelength array.
        counts: Count array.
        height: Minimum peak height.
        distance: Minimum distance between peaks.

    Returns:
        Tuple of (peak_wavelengths, peak_counts).
    """
    peaks, _ = find_peaks(counts, height=height, distance=distance)
    return wavelength[peaks], counts[peaks]


def extract_highest_peaks(
    wavelength: np.ndarray,
    counts: np.ndarray,
    num_peaks: int = 5,
    height: Optional[float] = None,
    distance: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Extract the N highest peaks from spectrum.

    Args:
        wavelength: Wavelength array.
        counts: Count array.
        num_peaks: Number of peaks to return.
        height: Minimum peak height.
        distance: Minimum distance between peaks.

    Returns:
        List of (wavelength, count) tuples for highest peaks.
    """
    peaks, _ = find_peaks(counts, height=height, distance=distance)
    peak_wl = wavelength[peaks]
    peak_cnt = counts[peaks]

    # Sort by intensity (descending)
    combined = sorted(zip(peak_cnt, peak_wl), reverse=True)
    top_peaks = combined[:num_peaks]

    return [(wl, cnt) for cnt, wl in top_peaks]


def compute_fft(
    wavelength: np.ndarray,
    counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of spectrum data.

    Args:
        wavelength: Wavelength array.
        counts: Count array.

    Returns:
        Tuple of (frequencies, amplitudes).
    """
    n = len(counts)
    if n < 2:
        return np.array([]), np.array([])

    timestep = abs(wavelength[1] - wavelength[0])
    centered = counts - np.mean(counts)

    frequencies = fftfreq(n, d=timestep)[:n // 2]
    amplitudes = np.abs(fft(centered))[:n // 2]

    return frequencies, amplitudes


# =============================================================================
# Calibration
# =============================================================================

def calibrate_with_silicon(
    silicon_file: str,
    laser_wavelength: float,
    si_peak_cm: float = SILICON_PEAK_CM,
    tolerance: float = 10.0
) -> Tuple[float, float]:
    """
    Calibrate wavelength using silicon reference peak.

    Args:
        silicon_file: Path to silicon reference spectrum.
        laser_wavelength: Nominal laser wavelength (nm).
        si_peak_cm: Expected silicon peak position (cm^-1).
        tolerance: Search window around expected peak (nm).

    Returns:
        Tuple of (wavelength_shift, expected_peak_position_nm).

    Raises:
        ValueError: If silicon peak cannot be found.
    """
    # Calculate expected silicon peak position in nm
    expected_si_nm = 1 / (1 / laser_wavelength - si_peak_cm / 1e7)

    # Load and preprocess silicon spectrum
    wavelength, counts = load_spectrum_data(silicon_file)
    counts = remove_cosmic_rays(wavelength, counts)

    # Extract region around expected peak
    mask = (wavelength > expected_si_nm - tolerance) & (wavelength < expected_si_nm + tolerance)
    wl_region = wavelength[mask]
    cnt_region = counts[mask]

    # Smooth and find peaks
    smoothed = smooth_signal(cnt_region)
    peak_wl, _ = detect_peaks(wl_region, smoothed, height=np.max(smoothed) * 0.5)

    # Find closest peak to expected position
    valid_peaks = [p for p in peak_wl if abs(p - expected_si_nm) <= tolerance]
    if not valid_peaks:
        raise ValueError(
            f"No silicon peak found near {expected_si_nm:.2f} nm. "
            f"Detected peaks: {peak_wl}"
        )

    si_peak_position = min(valid_peaks, key=lambda p: abs(p - expected_si_nm))
    shift = expected_si_nm - si_peak_position

    print(f"[Calibration] Detected Si peak at {si_peak_position:.2f} nm; shift = {shift:.2f} nm")

    return shift, expected_si_nm


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_analysis(
    wavelength: np.ndarray,
    original_counts: np.ndarray,
    analysis_data: List[np.ndarray],
    labels: List[str],
    title: str,
    export_path: str
):
    """
    Create analysis plot with original and processed data.

    Args:
        wavelength: Wavelength array.
        original_counts: Original count array.
        analysis_data: List of processed count arrays.
        labels: Labels for each processed array.
        title: Plot title.
        export_path: Path to save figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, original_counts, label="Original", color="blue")

    for data, label in zip(analysis_data, labels):
        plt.plot(wavelength, data, label=label, color="orange")

    if len(original_counts) > 0:
        min_val, max_val = np.min(original_counts), np.max(original_counts)
        plt.axhline(min_val, color="red", linestyle="--", label=f"Min = {min_val:.2f}")
        plt.axhline(max_val, color="green", linestyle="--", label=f"Max = {max_val:.2f}")

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(export_path)
    plt.close()


def plot_raw_vs_background(
    wl_raw: np.ndarray,
    cnt_raw: np.ndarray,
    wl_bg: np.ndarray,
    cnt_bg: np.ndarray,
    title: str,
    export_path: str
):
    """
    Plot raw data overlaid with background spectrum.

    Args:
        wl_raw: Raw wavelength array.
        cnt_raw: Raw count array.
        wl_bg: Background wavelength array.
        cnt_bg: Background count array.
        title: Plot title.
        export_path: Path to save figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wl_raw, cnt_raw, label="Raw Data", color="blue")
    plt.plot(wl_bg, cnt_bg, label="Ag Mirror Background", color="red")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.savefig(export_path)
    plt.close()


def plot_spectrum_comparison(
    wavelengths: List[np.ndarray],
    counts: List[np.ndarray],
    laser_wavelength: float,
    title: str,
    export_path: str,
    labels: Optional[List[str]] = None
):
    """
    Plot multiple spectra for comparison.

    Args:
        wavelengths: List of wavelength arrays.
        counts: List of count arrays.
        laser_wavelength: Laser wavelength for reference line.
        title: Plot title.
        export_path: Path to save figure.
        labels: Labels for each spectrum.
    """
    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [f"Data {i}" for i in range(len(wavelengths))]

    for wl, cnt, label in zip(wavelengths, counts, labels):
        plt.plot(wl, cnt, label=label)

    plt.axvline(laser_wavelength, color="black", linestyle="--", label="Laser WL")
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(export_path)
    plt.close()


# =============================================================================
# PDF Report Generation
# =============================================================================

def generate_pdf_report(
    pdf_path: str,
    laser_wavelength: float,
    power_levels: List[float],
    thickness: float,
    window: float,
    shift: float,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List[Tuple[float, float]]],
    peaks_info_fft: Dict[str, List[Tuple[float, float]]],
    sample_image: Optional[str] = None,
    sample_name: str = "",
    integrals_info: Optional[Dict[str, float]] = None
):
    """
    Generate comprehensive PDF analysis report.

    Args:
        pdf_path: Output path for PDF file.
        laser_wavelength: Calibrated laser wavelength.
        power_levels: List of laser power levels used.
        thickness: Sample thickness.
        window: Laser filter window size.
        shift: Calibration wavelength shift.
        plot_paths: Dictionary mapping plot titles to file paths.
        peaks_info: Dictionary of peak information for each spectrum.
        peaks_info_fft: Dictionary of FFT peak information.
        sample_image: Optional path to sample image.
        sample_name: Name of the sample.
        integrals_info: Dictionary of integral values.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    if integrals_info is None:
        integrals_info = {}

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Raman Spectroscopy Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Sample information
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Sample: {sample_name}", ln=True, align="C")
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Laser Wavelength (calibrated): {laser_wavelength:.2f} nm", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Power Levels: {', '.join(map(str, power_levels))} mW", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Thickness: {thickness} nm", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Laser Filter Window: +/-{window} nm", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Calibration Shift: {shift:.2f} nm", ln=True, align="L")
    pdf.ln(10)

    # Insert calibration and overview plots
    overview_keys = ["Calibration Region", "Global", "Raw Data", "Ag Mirror Background"]
    for title, path in plot_paths.items():
        if any(key in title for key in overview_keys):
            pdf.add_page()
            pdf.cell(200, 10, txt=title, ln=True, align="C")
            pdf.image(path, x=10, y=30, w=190)

    # Stokes/Anti-Stokes analysis per power level
    for power in power_levels:
        _add_power_level_pages(
            pdf, power, plot_paths, peaks_info, peaks_info_fft, integrals_info
        )

    # Polarization analysis pages
    _add_polarization_pages(pdf, plot_paths, peaks_info, peaks_info_fft)

    # Sample image (if provided)
    if sample_image and os.path.exists(sample_image):
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Sample Image", ln=True, align="C")
        pdf.image(sample_image, x=10, y=30, w=140)

    pdf.output(pdf_path)


def _add_power_level_pages(
    pdf: FPDF,
    power: float,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List],
    integrals_info: Dict[str, float]
):
    """Add PDF pages for a specific power level analysis."""
    stokes_title = f"Stokes for {power} mW"
    fft_stokes_title = f"FFT Stokes for {power} mW"
    anti_title = f"Anti-Stokes for {power} mW"
    fft_anti_title = f"FFT Anti-Stokes for {power} mW"

    # Stokes pages
    if stokes_title in plot_paths:
        pdf.add_page()
        pdf.cell(200, 10, txt=stokes_title, ln=True, align="C")
        pdf.image(plot_paths[stokes_title], x=10, y=30, w=190)
        pdf.ln(120)

        if stokes_title in integrals_info:
            pdf.cell(200, 10, txt=f"Integral (Area) = {integrals_info[stokes_title]:.2f}", ln=True, align="L")

        pdf.cell(200, 10, txt="Peak Info (Stokes):", ln=True, align="L")
        if stokes_title in peaks_info:
            for peak in peaks_info[stokes_title]:
                pdf.cell(200, 10, txt=f"  WL = {peak[0]:.2f}, Count = {peak[1]:.2f}", ln=True, align="L")

    if fft_stokes_title in plot_paths:
        pdf.add_page()
        pdf.cell(200, 10, txt=fft_stokes_title, ln=True, align="C")
        pdf.image(plot_paths[fft_stokes_title], x=10, y=30, w=190)
        pdf.ln(120)

        pdf.cell(200, 10, txt="Peak Info (FFT Stokes):", ln=True, align="L")
        if fft_stokes_title in peaks_info_fft:
            for peak in peaks_info_fft[fft_stokes_title]:
                pdf.cell(200, 10, txt=f"  Freq = {peak[0]:.2f}, Amp = {peak[1]:.2f}", ln=True, align="L")

    # Anti-Stokes pages
    if anti_title in plot_paths:
        pdf.add_page()
        pdf.cell(200, 10, txt=anti_title, ln=True, align="C")
        pdf.image(plot_paths[anti_title], x=10, y=30, w=190)
        pdf.ln(120)

        if anti_title in integrals_info:
            pdf.cell(200, 10, txt=f"Integral (Area) = {integrals_info[anti_title]:.2f}", ln=True, align="L")

        pdf.cell(200, 10, txt="Peak Info (Anti-Stokes):", ln=True, align="L")
        if anti_title in peaks_info:
            for peak in peaks_info[anti_title]:
                pdf.cell(200, 10, txt=f"  WL = {peak[0]:.2f}, Count = {peak[1]:.2f}", ln=True, align="L")

    if fft_anti_title in plot_paths:
        pdf.add_page()
        pdf.cell(200, 10, txt=fft_anti_title, ln=True, align="C")
        pdf.image(plot_paths[fft_anti_title], x=10, y=30, w=190)
        pdf.ln(120)

        pdf.cell(200, 10, txt="Peak Info (FFT Anti-Stokes):", ln=True, align="L")
        if fft_anti_title in peaks_info_fft:
            for peak in peaks_info_fft[fft_anti_title]:
                pdf.cell(200, 10, txt=f"  Freq = {peak[0]:.2f}, Amp = {peak[1]:.2f}", ln=True, align="L")


def _add_polarization_pages(
    pdf: FPDF,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List]
):
    """Add PDF pages for polarization analysis."""
    polarization_keys = [
        "Co Raw vs BG (14 mW)",
        "Cross Raw vs BG (14 mW)",
        "Co Stokes (14 mW)",
        "FFT Co Stokes (14 mW)",
        "Co Anti-Stokes (14 mW)",
        "FFT Co Anti-Stokes (14 mW)",
        "Cross Stokes (14 mW)",
        "FFT Cross Stokes (14 mW)",
        "Cross Anti-Stokes (14 mW)",
        "FFT Cross Anti-Stokes (14 mW)",
        "Polarization 14 mW"
    ]

    for key in polarization_keys:
        if key in plot_paths:
            pdf.add_page()
            pdf.cell(200, 10, txt=key, ln=True, align="C")
            pdf.image(plot_paths[key], x=10, y=30, w=190)
            pdf.ln(120)

            if key in peaks_info:
                pdf.cell(200, 10, txt="Peak Info:", ln=True, align="L")
                for peak in peaks_info[key]:
                    pdf.cell(200, 10, txt=f"  WL = {peak[0]:.2f}, Count = {peak[1]:.2f}", ln=True, align="L")

            if key in peaks_info_fft:
                pdf.ln(10)
                pdf.cell(200, 10, txt="FFT Peak Info:", ln=True, align="L")
                for peak in peaks_info_fft[key]:
                    pdf.cell(200, 10, txt=f"  Freq = {peak[0]:.2f}, Amp = {peak[1]:.2f}", ln=True, align="L")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def main():
    """Main entry point for Raman spectroscopy analysis."""
    Tk().withdraw()

    # Storage for analysis results
    integrals_info = {}
    plot_paths = {}
    peaks_info = {}
    peaks_info_fft = {}

    # Get sample information
    sample_name = simpledialog.askstring("Sample Name", "Enter the sample name:")

    # Laser selection and calibration
    color_choice = simpledialog.askstring("Laser", "Green or Red?")
    if color_choice and color_choice.lower().startswith("g"):
        laser_wavelength = LASER_GREEN
    else:
        laser_wavelength = LASER_RED

    silicon_file = filedialog.askopenfilename(
        title="Select Silicon Spectrum for Calibration"
    )
    if not silicon_file:
        print("No silicon file selected. Exiting.")
        return

    try:
        shift, expected_si_nm = calibrate_with_silicon(silicon_file, laser_wavelength)
    except ValueError as e:
        print(f"Calibration error: {e}")
        return

    laser_wavelength += shift

    # Get sample parameters
    thickness = simpledialog.askfloat("Thickness", "Enter sample thickness (nm):")

    power_str = simpledialog.askstring("Power Levels", "Enter powers, e.g. '10,14,20':")
    power_levels = [float(x.strip()) for x in power_str.split(",")]

    # Select measurement files
    meas_files = filedialog.askopenfilenames(
        title="Select measured data files for ALL powers in correct order"
    )
    if len(meas_files) != len(power_levels):
        print(f"Number of files ({len(meas_files)}) != number of powers ({len(power_levels)})")
        return

    bg_files = filedialog.askopenfilenames(
        title="Select background files for ALL powers in correct order"
    )
    if len(bg_files) != len(power_levels):
        print(f"Number of background files ({len(bg_files)}) != number of powers ({len(power_levels)})")
        return

    # Plot calibration region
    _create_calibration_plot(silicon_file, expected_si_nm, shift, plot_paths)

    # Analyze each power level
    all_stokes = []
    all_antistokes = []
    all_labels = []

    for power, meas_file, bg_file in zip(power_levels, meas_files, bg_files):
        _analyze_power_level(
            power, meas_file, bg_file, shift, laser_wavelength,
            plot_paths, peaks_info, peaks_info_fft, integrals_info,
            all_stokes, all_antistokes, all_labels
        )

    # Create global overview plots
    _create_global_plots(
        all_stokes, all_antistokes, all_labels, laser_wavelength,
        plot_paths, bg_files[-1], meas_files, shift
    )

    # Optional polarization analysis
    _handle_polarization_analysis(
        power_levels, shift, laser_wavelength,
        plot_paths, peaks_info, peaks_info_fft
    )

    # Optional sample image
    sample_image = filedialog.askopenfilename(
        title="Select a sample image (optional)"
    )

    # Generate PDF report
    pdf_path = filedialog.asksaveasfilename(
        title="Save PDF As",
        defaultextension=".pdf"
    )
    if not pdf_path:
        print("No PDF path selected. Exiting.")
        return

    generate_pdf_report(
        pdf_path=pdf_path,
        laser_wavelength=laser_wavelength,
        power_levels=power_levels,
        thickness=thickness,
        window=DEFAULT_LASER_WINDOW,
        shift=shift,
        plot_paths=plot_paths,
        peaks_info=peaks_info,
        peaks_info_fft=peaks_info_fft,
        sample_image=sample_image,
        sample_name=sample_name,
        integrals_info=integrals_info
    )

    print(f"Analysis complete. PDF saved at: {pdf_path}")


def _create_calibration_plot(
    silicon_file: str,
    expected_si_nm: float,
    shift: float,
    plot_paths: Dict[str, str]
):
    """Create and save calibration region plot."""
    si_wl, si_cnt = load_spectrum_data(silicon_file)
    si_cnt = remove_cosmic_rays(si_wl, si_cnt)

    mask = (si_wl > expected_si_nm - 10) & (si_wl < expected_si_nm + 10)
    cal_wl = si_wl[mask]
    cal_cnt = si_cnt[mask]
    cal_smooth = smooth_signal(cal_cnt)

    plt.figure(figsize=(10, 6))
    plt.plot(cal_wl, cal_cnt, label="Calibration Region", color="orange")
    plt.plot(cal_wl, cal_smooth, label="Smoothed", color="purple")
    plt.axvline(expected_si_nm, color="red", linestyle="--", label="Expected Si Peak")
    plt.axvline(expected_si_nm - shift, color="green", linestyle="--", label="Detected Si Peak")
    plt.title("Silicon Calibration Region")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.legend()

    cal_plot = "calibration_region.png"
    plt.savefig(cal_plot)
    plt.close()
    plot_paths["Calibration Region"] = cal_plot


def _analyze_power_level(
    power: float,
    meas_file: str,
    bg_file: str,
    shift: float,
    laser_wavelength: float,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List],
    integrals_info: Dict[str, float],
    all_stokes: List,
    all_antistokes: List,
    all_labels: List
):
    """Analyze spectrum for a single power level."""
    # Load and preprocess measurement data
    wl, cnt = load_spectrum_data(meas_file)
    wl += shift
    cnt = remove_cosmic_rays(wl, cnt)

    # Load and preprocess background
    bg_wl, bg_cnt = load_spectrum_data(bg_file)
    bg_wl += shift
    bg_cnt = remove_cosmic_rays(bg_wl, bg_cnt)

    # Background subtraction and laser region removal
    cnt_corrected = subtract_background(wl, cnt, bg_wl, bg_cnt)
    wl_cut, cnt_cut = cut_laser_region(wl, cnt_corrected, laser_wavelength)

    # Separate Stokes and Anti-Stokes regions
    stokes_mask = wl_cut > laser_wavelength
    anti_mask = (wl_cut > (laser_wavelength - ANTI_STOKES_RANGE)) & (wl_cut < laser_wavelength)

    stokes_wl = wl_cut[stokes_mask]
    stokes_cnt = cnt_cut[stokes_mask]
    anti_wl = wl_cut[anti_mask]
    anti_cnt = cnt_cut[anti_mask]

    # Process Stokes region
    _process_spectral_region(
        stokes_wl, stokes_cnt, f"Stokes for {power} mW", f"stokes_{power}.png",
        f"FFT Stokes for {power} mW", f"fft_stokes_{power}.png",
        plot_paths, peaks_info, peaks_info_fft, integrals_info
    )

    # Process Anti-Stokes region
    _process_spectral_region(
        anti_wl, anti_cnt, f"Anti-Stokes for {power} mW", f"antistokes_{power}.png",
        f"FFT Anti-Stokes for {power} mW", f"fft_antistokes_{power}.png",
        plot_paths, peaks_info, peaks_info_fft, integrals_info
    )

    # Store for global plots
    all_stokes.append((stokes_wl, stokes_cnt))
    all_antistokes.append((anti_wl, anti_cnt))
    all_labels.append(f"{power} mW")


def _process_spectral_region(
    wavelength: np.ndarray,
    counts: np.ndarray,
    title: str,
    plot_file: str,
    fft_title: str,
    fft_plot_file: str,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List],
    integrals_info: Dict[str, float]
):
    """Process a spectral region (Stokes or Anti-Stokes)."""
    # Smooth signal
    smoothed = smooth_signal(counts)

    # Main spectrum plot
    plot_analysis(wavelength, counts, [smoothed], ["Smoothed"], title, plot_file)
    plot_paths[title] = plot_file

    # FFT analysis
    freq, fft_vals = compute_fft(wavelength, counts)

    plt.figure(figsize=(10, 6))
    plt.plot(freq, fft_vals, label="FFT", color="purple")
    plt.title(fft_title)
    plt.xlabel("Frequency (1/nm)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(fft_plot_file)
    plt.close()
    plot_paths[fft_title] = fft_plot_file

    # Peak detection
    peaks_info[title] = extract_highest_peaks(wavelength, counts)
    peaks_info_fft[fft_title] = extract_highest_peaks(freq, fft_vals)

    # Integral calculation
    integrals_info[title] = np.trapz(counts, x=wavelength)


def _create_global_plots(
    all_stokes: List,
    all_antistokes: List,
    all_labels: List,
    laser_wavelength: float,
    plot_paths: Dict[str, str],
    last_bg_file: str,
    meas_files: tuple,
    shift: float
):
    """Create global overview plots."""
    # Global Stokes
    if all_stokes:
        plot_spectrum_comparison(
            [w for w, _ in all_stokes],
            [c for _, c in all_stokes],
            laser_wavelength,
            "Global Stokes Spectrum",
            "global_stokes.png",
            labels=all_labels
        )
        plot_paths["Global Stokes Spectrum"] = "global_stokes.png"

    # Global Anti-Stokes
    if all_antistokes:
        plot_spectrum_comparison(
            [w for w, _ in all_antistokes],
            [c for _, c in all_antistokes],
            laser_wavelength,
            "Global Anti-Stokes Spectrum",
            "global_antistokes.png",
            labels=all_labels
        )
        plot_paths["Global Anti-Stokes Spectrum"] = "global_antistokes.png"

    # Load last background for comparison plot
    bg_wl, bg_cnt = load_spectrum_data(last_bg_file)
    bg_wl += shift
    bg_cnt = remove_cosmic_rays(bg_wl, bg_cnt)

    # Raw data plot
    plt.figure(figsize=(10, 6))
    for meas_file, label in zip(meas_files, all_labels):
        rwl, rcnt = load_spectrum_data(meas_file)
        rwl += shift
        rcnt = subtract_background(rwl, rcnt, bg_wl, bg_cnt)
        rcnt = remove_cosmic_rays(rwl, rcnt)
        plt.plot(rwl, rcnt, label=label)

    plt.axvline(laser_wavelength, color="black", linestyle="--", label="Laser WL")
    plt.title("Raw Data (background removed)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig("raw_data.png")
    plt.close()
    plot_paths["Raw Data"] = "raw_data.png"


def _handle_polarization_analysis(
    power_levels: List[float],
    shift: float,
    laser_wavelength: float,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List]
):
    """Handle optional polarization analysis at 14 mW."""
    pol_power = 14.0
    if pol_power not in power_levels:
        return

    do_pol = simpledialog.askstring(
        "Polarization?",
        f"Do you want co/cross analysis at {pol_power} mW? (yes/no)"
    )

    if not do_pol or not do_pol.lower().startswith("y"):
        return

    # Get polarization files
    co_file = filedialog.askopenfilename(title=f"Select Co-Polarized Data at {pol_power} mW")
    co_bg_file = filedialog.askopenfilename(title=f"Select Co-Polarized Background at {pol_power} mW")
    cross_file = filedialog.askopenfilename(title=f"Select Cross-Polarized Data at {pol_power} mW")
    cross_bg_file = filedialog.askopenfilename(title=f"Select Cross-Polarized Background at {pol_power} mW")

    if not all([co_file, co_bg_file, cross_file, cross_bg_file]):
        return

    # Process co-polarized data
    _process_polarization_data(
        co_file, co_bg_file, shift, laser_wavelength,
        "Co", plot_paths, peaks_info, peaks_info_fft
    )

    # Process cross-polarized data
    _process_polarization_data(
        cross_file, cross_bg_file, shift, laser_wavelength,
        "Cross", plot_paths, peaks_info, peaks_info_fft
    )

    # Combined polarization plot
    _create_combined_polarization_plot(
        co_file, co_bg_file, cross_file, cross_bg_file,
        shift, laser_wavelength, plot_paths
    )


def _process_polarization_data(
    data_file: str,
    bg_file: str,
    shift: float,
    laser_wavelength: float,
    prefix: str,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List]
):
    """Process co or cross polarized data."""
    # Load data
    wl_raw, cnt_raw = load_spectrum_data(data_file)
    wl_raw += shift
    cnt_raw = remove_cosmic_rays(wl_raw, cnt_raw)

    bg_wl_raw, bg_cnt_raw = load_spectrum_data(bg_file)
    bg_wl_raw += shift
    bg_cnt_raw = remove_cosmic_rays(bg_wl_raw, bg_cnt_raw)

    # Raw vs background plot
    plot_raw_vs_background(
        wl_raw, cnt_raw, bg_wl_raw, bg_cnt_raw,
        f"{prefix} Raw vs BG (14 mW)",
        f"{prefix.lower()}_raw_vs_bg_14mW.png"
    )
    plot_paths[f"{prefix} Raw vs BG (14 mW)"] = f"{prefix.lower()}_raw_vs_bg_14mW.png"

    # Background subtraction
    cnt_corr = subtract_background(wl_raw, cnt_raw, bg_wl_raw, bg_cnt_raw)
    wl_cut, cnt_cut = cut_laser_region(wl_raw, cnt_corr, laser_wavelength)

    # Stokes
    st_mask = wl_cut > laser_wavelength
    wl_st = wl_cut[st_mask]
    cnt_st = cnt_cut[st_mask]

    _process_spectral_region_simple(
        wl_st, cnt_st,
        f"{prefix} Stokes (14 mW)", f"{prefix.lower()}_stokes_14mW.png",
        f"FFT {prefix} Stokes (14 mW)", f"fft_{prefix.lower()}_stokes_14mW.png",
        plot_paths, peaks_info, peaks_info_fft
    )

    # Anti-Stokes
    as_mask = (wl_cut > (laser_wavelength - ANTI_STOKES_RANGE)) & (wl_cut < laser_wavelength)
    wl_as = wl_cut[as_mask]
    cnt_as = cnt_cut[as_mask]

    _process_spectral_region_simple(
        wl_as, cnt_as,
        f"{prefix} Anti-Stokes (14 mW)", f"{prefix.lower()}_antistokes_14mW.png",
        f"FFT {prefix} Anti-Stokes (14 mW)", f"fft_{prefix.lower()}_antistokes_14mW.png",
        plot_paths, peaks_info, peaks_info_fft
    )


def _process_spectral_region_simple(
    wavelength: np.ndarray,
    counts: np.ndarray,
    title: str,
    plot_file: str,
    fft_title: str,
    fft_plot_file: str,
    plot_paths: Dict[str, str],
    peaks_info: Dict[str, List],
    peaks_info_fft: Dict[str, List]
):
    """Process a spectral region without integral calculation."""
    smoothed = smooth_signal(counts)

    plot_analysis(wavelength, counts, [smoothed], ["Smoothed"], title, plot_file)
    plot_paths[title] = plot_file

    freq, fft_vals = compute_fft(wavelength, counts)

    plt.figure(figsize=(10, 6))
    plt.plot(freq, fft_vals, label="FFT", color="purple")
    plt.title(fft_title)
    plt.xlabel("Frequency (1/nm)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(fft_plot_file)
    plt.close()
    plot_paths[fft_title] = fft_plot_file

    peaks_info[title] = extract_highest_peaks(wavelength, counts)
    peaks_info_fft[fft_title] = extract_highest_peaks(freq, fft_vals)


def _create_combined_polarization_plot(
    co_file: str,
    co_bg_file: str,
    cross_file: str,
    cross_bg_file: str,
    shift: float,
    laser_wavelength: float,
    plot_paths: Dict[str, str]
):
    """Create combined co/cross polarization comparison plot."""
    # Load and process co-polarized
    co_wl, co_cnt = load_spectrum_data(co_file)
    co_wl += shift
    co_cnt = remove_cosmic_rays(co_wl, co_cnt)

    co_bg_wl, co_bg_cnt = load_spectrum_data(co_bg_file)
    co_bg_wl += shift
    co_bg_cnt = remove_cosmic_rays(co_bg_wl, co_bg_cnt)

    co_cnt_corr = subtract_background(co_wl, co_cnt, co_bg_wl, co_bg_cnt)
    co_wl_cut, co_cnt_cut = cut_laser_region(co_wl, co_cnt_corr, laser_wavelength)

    # Load and process cross-polarized
    cross_wl, cross_cnt = load_spectrum_data(cross_file)
    cross_wl += shift
    cross_cnt = remove_cosmic_rays(cross_wl, cross_cnt)

    cross_bg_wl, cross_bg_cnt = load_spectrum_data(cross_bg_file)
    cross_bg_wl += shift
    cross_bg_cnt = remove_cosmic_rays(cross_bg_wl, cross_bg_cnt)

    cross_cnt_corr = subtract_background(cross_wl, cross_cnt, cross_bg_wl, cross_bg_cnt)
    cross_wl_cut, cross_cnt_cut = cut_laser_region(cross_wl, cross_cnt_corr, laser_wavelength)

    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(co_wl_cut, co_cnt_cut, label="Co-Polarized", color="blue")
    plt.plot(cross_wl_cut, cross_cnt_cut, label="Cross-Polarized", color="red")
    plt.axvline(laser_wavelength, color="gray", linestyle="--", label="Laser WL")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Counts")
    plt.title("Polarization 14 mW (Corrected)")
    plt.legend()
    plt.savefig("polarization_14mW_corrected.png")
    plt.close()
    plot_paths["Polarization 14 mW"] = "polarization_14mW_corrected.png"


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
