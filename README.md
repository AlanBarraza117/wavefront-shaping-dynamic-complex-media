# Wavefront Shaping Through Dynamic Complex Media

Summer internship project focused on adaptive optics and wavefront shaping for imaging through scattering media. The project demonstrates feedback-loop optimization for focusing light through both static and periodically alternating dynamic complex media using Digital Micromirror Devices (DMD) and Phase Light Modulators (PLM).

## Project Overview

This work implements the sequential phase-stepping algorithm described in [Vellekoop & Mosk, Opt. Lett. 2007](https://doi.org/10.1364/OL.32.002309) for focusing coherent light through opaque scattering media. The key innovation is extending this approach to dynamic complex media that periodically switch between multiple random states.

### Key Features

- **Sequential Phase Optimization**: Iteratively adjusts DMD wavefront segments to maximize camera intensity at a target ROI
- **Dynamic Media Support**: Optimizes focusing through media oscillating between 2 or 4 random states at up to 700 Hz
- **Real-time GUI**: Interactive control with live camera preview, ROI selection, and adjustable parameters
- **Hardware Integration**: Synchronized DMD-camera triggering via ALP-4.3 API for high-speed optimization
- **Auto-exposure**: Dynamic exposure adjustment to prevent saturation during optimization

## Hardware Setup

- **DMD**: ViALUX ALP-4.3 SuperSpeed V-module (2560×1600 micromirrors)
- **PLM**: Texas Instruments DLP6750Q1EVM (1358×800 phase-only modulator)
- **Camera**: Basler (via pypylon), hardware-triggered via DMD Line3 sync
- **Laser**: 632.8 nm HeNe

## Repository Structure

```
wavefront-shaping-portfolio/
├── code/
│   ├── light_focusing_algorithm.py   # Core optimization algorithm
│   ├── dmd_controller_ui.py          # GUI application
│   └── tests.ipynb                   # PLM control guide
├── docs/
│   ├── main.pdf                      # Full report
│   ├── main.tex                      # LaTeX source
│   ├── references.bib                # Bibliography
│   └── Wavefront Shaping through dynamic complex media.pptx
├── results/                          # Example results
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the GUI

```bash
python code/dmd_controller_ui.py
```

The GUI provides:
- Live camera preview with zoom/pan
- Interactive ROI selection
- Mask loading and projection
- Sequential phase optimization
- Results saving (holograms, fields)

### Running the Algorithm Directly

```bash
python code/light_focusing_algorithm.py
```

### PLM Control (Jupyter Notebook)

Open `code/tests.ipynb` for the complete PLM control workflow including:
- Controller initialization
- Configuration and calibration
- Hologram generation and display
- Speckle pattern generation

## Theory

### Static Media Focusing

The algorithm divides the DMD wavefront into N=144 macro-pixel segments (12×12 grid). For each segment, 8 equally spaced phase values (0 to 2π) are tested. The phase maximizing ROI intensity is retained, progressively phase-locking all segments for constructive interference at the target.

### Dynamic Media Extension

For media oscillating between n states, the optimal input field is the eigenvector of the Hermitian matrix H_k constructed from transmission-matrix rows of all n constituent states:

$$I_k = (\overline{E}^{(\text{in})})^\dagger \, H_k \, \overline{E}^{(\text{in})}, \qquad H_k \equiv \sum_{i=1}^{n} \overline{t}^{(i)*}_k \otimes \overline{t}^{(i)}_k$$

## Results

### 2-Media Configuration (700 Hz switching)
The dynamic complex media oscillates between two independent random states. The optimization achieves focus with 67% correlation between the experimentally obtained solution and the theoretical eigenvector.

### 4-Media Configuration (350 Hz per state)
With four alternating states, the correlation improves to 81%, as the optimization better approximates the Hermitian matrix eigenvector solution.

## Technical Highlights

- **Batch bitmap pre-computation** via numpy broadcasting for vectorized fields
- **Multi-frame DMD sequence upload** for high-speed phase testing
- **Hardware-triggered camera capture** synchronized with DMD frame transitions
- **Row-by-row hologram computation** to minimize memory usage
- **Auto-exposure** with saturation detection to maintain dynamic range

## References

1. Vellekoop, I. M., & Mosk, A. P. (2007). Focusing coherent light through opaque strongly scattering media. *Optics Letters*, 32(16), 2309-2311.
2. Popoff, S., et al. (2010). Measuring the transmission matrix of an optical medium. *Physical Review Letters*.

## Author

Jose Alan Barraza Villaverde

M1 Internship at Université Paris-Saclay, Laboratoire Lumière, Matière et Interfaces (LuMIn), Orsay, France

Master in Intelligent Photonics for Security, Reliability, Sustainability and Safety (iPSRS)

## License

This project is available for academic and portfolio purposes.
