"""
Raman Spectroscopy Processing Module

Comprehensive analysis pipeline for Raman spectroscopy data including
calibration, background subtraction, Stokes/Anti-Stokes separation,
FFT analysis, and automated PDF report generation.
"""

from .process_raman import (
    load_spectrum_data,
    remove_cosmic_rays,
    smooth_signal,
    subtract_background,
    calibrate_with_silicon,
    main
)

__all__ = [
    "load_spectrum_data",
    "remove_cosmic_rays",
    "smooth_signal",
    "subtract_background",
    "calibrate_with_silicon",
    "main"
]
