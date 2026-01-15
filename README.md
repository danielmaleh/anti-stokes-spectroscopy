# Anti-Stokes Spectroscopy of Crystalline Plasmonic Metals

A Python toolkit for analyzing Anti-Stokes Raman spectroscopy data from crystalline gold flakes, with additional tools for color-to-thickness mapping of thin metal films.

## Overview

This project provides two main analysis tools developed during a research internship at the EPFL NAM (Nanophotonics and Metrology) Laboratory:

1. **Raman Spectroscopy Processor**: Comprehensive analysis pipeline for Raman spectroscopy data including calibration, background subtraction, Stokes/Anti-Stokes separation, FFT analysis, and automated PDF report generation.

2. **Color-to-Thickness Mapper**: GUI application for correlating optical microscope color observations of gold flakes to their measured thicknesses, enabling non-destructive thickness estimation.

## Repository Structure

```
anti-stokes-spectroscopy/
├── src/
│   ├── color_mapping/
│   │   └── color_to_thickness.py    # Color-thickness correlation tool
│   └── raman_processing/
│       └── process_raman.py         # Raman spectroscopy analysis
├── data/
│   ├── lookup_table_Float.csv       # Pre-built lookup table (Float glass)
│   └── lookup_table_D263.csv        # Pre-built lookup table (D263 glass)
├── docs/
│   └── NAM_Semester_Project_Report.pdf
├── exports/                          # Generated analysis reports
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/[username]/anti-stokes-spectroscopy.git
cd anti-stokes-spectroscopy
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Raman Spectroscopy Processor

Analyze Raman spectroscopy data with automatic calibration and comprehensive reporting:

```bash
python src/raman_processing/process_raman.py
```

The interactive workflow guides you through:

1. **Sample Information**: Enter sample name
2. **Laser Selection**: Choose green (532 nm) or red (633 nm) laser
3. **Calibration**: Select silicon reference spectrum for wavelength calibration
4. **Parameters**: Enter sample thickness and power levels used
5. **Data Files**: Select measurement and background files for each power level
6. **Polarization Analysis** (optional): Analyze co/cross-polarized spectra
7. **Report Generation**: Specify output PDF path

#### Features

- **Silicon-based Calibration**: Uses the well-known Si peak at 520 cm⁻¹ for wavelength correction
- **Cosmic Ray Removal**: Automated detection and removal of cosmic ray artifacts
- **Background Subtraction**: Interpolated background subtraction with baseline correction
- **Stokes/Anti-Stokes Separation**: Automatic splitting around laser line
- **FFT Analysis**: Frequency domain analysis for pattern detection
- **Peak Detection**: Automated identification of significant peaks
- **Integral Calculation**: Area under curve for quantitative analysis
- **Polarization Analysis**: Co/cross-polarized comparison at specified power

### Color-to-Thickness Mapping

Launch the GUI application for color-thickness correlation:

```bash
python src/color_mapping/color_to_thickness.py
```

#### Functionalities

**1. Creating Lookup Tables**
- Select substrate type (Float, Borofloat, Si, D263)
- Choose entry method (Manual RGB values or Image-based ROI selection)
- Enter known thickness value
- Build lookup table iteratively

**2. Mapping Unknown Samples**
- Load sample image and select region of interest
- Compare normalized colors against lookup table
- Get thickness estimates in both RGB and LAB color spaces

#### Supported Substrates

| Substrate | Description |
|-----------|-------------|
| Float | Standard float glass |
| Borofloat | Borosilicate glass |
| Si | Silicon wafer |
| D263 | Schott D263 thin glass |

## Data Format

### Raman Spectrum Files

CSV format with two columns:
```
wavelength,count
530.5,1234
530.6,1256
...
```

### Lookup Tables

CSV format with normalized color values:
```
R,G,B,L,a,b,Thickness [nm]
1.32143,3.10345,4.78125,4.07143,1.16406,1.3125,43
...
```

## Theory

### Anti-Stokes Raman Spectroscopy

Anti-Stokes scattering occurs when incident photons gain energy from thermally excited molecular vibrations. For crystalline gold flakes, this phenomenon provides information about:

- Phonon populations and thermal properties
- Crystal quality and defect density
- Surface-enhanced Raman effects (SERS)

The ratio of Anti-Stokes to Stokes intensities is temperature-dependent:
```
I_AS/I_S = exp(-ℏω/kT)
```

### Color-Thickness Correlation

Thin gold films exhibit thickness-dependent colors due to interference effects. By building empirical lookup tables correlating observed colors (in both RGB and LAB color spaces) with measured thicknesses, non-destructive thickness estimation becomes possible.

## Documentation

Detailed theoretical background and experimental procedures are available in:
- `docs/NAM_Semester_Project_Report.pdf`

## Example Output

The Raman processor generates comprehensive PDF reports including:
- Calibration verification plots
- Raw data and background comparison
- Stokes and Anti-Stokes spectra for each power level
- FFT analysis results
- Peak detection tables
- Polarization analysis (if performed)
- Sample images

## License

MIT License

Copyright (c) 2024 Daniel Abraham Elmaleh

See [LICENSE](LICENSE) for details.

## Author

**Daniel Abraham Elmaleh**

Research conducted at:
- EPFL - NAM Laboratory (Nanophotonics and Metrology)
- Lausanne, Switzerland

## Acknowledgments

This project was developed as part of a semester research project at EPFL's NAM Laboratory, focusing on the optical characterization of crystalline plasmonic metals.

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{elmaleh2024antistokes,
  author = {Elmaleh, Daniel Abraham},
  title = {Anti-Stokes Spectroscopy Analysis Toolkit},
  year = {2024},
  institution = {EPFL NAM Laboratory},
  url = {https://github.com/[username]/anti-stokes-spectroscopy}
}
```
