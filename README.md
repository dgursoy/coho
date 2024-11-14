# COHO: Coded Holography Package

**COHO** is a Python package designed for simulating coherent holography experiments. It provides a comprehensive toolkit for researchers and engineers to model wave propagation, material interactions, and detector responses in holographic imaging systems.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Installing from PyPI](#installing-from-pypi)
  - [Installing from Source](#installing-from-source)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Wave Propagation Models**: Supports Fresnel and Fraunhofer diffraction, with the flexibility to add custom propagation methods.
- **Material Properties Calculation**: Utilizes `xraylib` to compute complex refractive indices and attenuation coefficients.
- **Pattern Generation**: Includes functions for creating binary coded apertures and object patterns.
- **Detector Simulation**: Models realistic detector responses, including noise and efficiency variations.
- **Visualization Tools**: Provides functions to visualize wavefronts, intensities, and propagation animations.
- **Configurable Simulation Pipeline**: Allows users to define simulation components and parameters via YAML configuration files.
- **Modular Design**: Easy to extend and customize for specific research needs.

---

## Installation

### Requirements

- **Python**: 3.7 or higher
- **Dependencies**:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `xraylib`
  - `scikit-image`
  - `pyyaml`
  - `fire`

### Installing from PyPI

COHO can be installed from the Python Package Index (PyPI) using `pip`:

```bash
pip install coho
```

This command will install COHO along with all required dependencies.

### Installing from Source

To install COHO from the source code, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/dgursoy/coho.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd coho
   ```

3. **Install the Package**:

   ```bash
   pip install .
   ```

   This command installs COHO and its dependencies using `pip`.

---

## Quick Start

Here's a simple example to get you started with COHO:

1. **Create a Simulation Configuration File**:

   Save the following YAML content as `config.yml`:

   ```yaml
   simulation:
     name: "Example Simulation"
     elements:
       - type: "wavefront"
         name: "source"
         parameters:
           energy: 8.0  # keV
           pixels: 512
           pixel_size: 1e-6  # cm
           type: "plane"
       - type: "component"
         name: "coded_aperture"
         position: 10.0  # cm
         material: "Gold"
         thickness: 2.0  # μm
         parameters:
           type: "coded_aperture"
           pattern:
             bit_size: 64
             rotation_angle: 15
             seed: 42
       - type: "detector"
         name: "detector"
         position: 50.0  # cm
         parameters:
           normalize: True
           incident_intensity: 1000
   ```

2. **Create a Simulation Script**:

   Save the following Python code as `run_simulation.py`:

   ```python
   import sys
   import coho

   def main(config_file):
       # Load the configuration
       config = coho.load_configuration(config_file)

       # Validate the configuration
       coho.validate_config(config)

       # Initialize the simulation
       simulation = coho.Simulation(config)

       # Run the simulation
       simulation.run()

       # Access results (e.g., detected intensity)
       detected_intensity = simulation.results.get('detected_intensity')

       # Visualize the detected intensity
       if detected_intensity is not None:
           coho.plot_intensity(detected_intensity, title='Detected Intensity')

   if __name__ == "__main__":
       if len(sys.argv) != 2:
           print("Usage: python run_simulation.py <config_file.yml>")
           sys.exit(1)
       config_file = sys.argv[1]
       main(config_file)
   ```

3. **Run the Simulation**:

   ```bash
   python run_simulation.py config.yml
   ```

   This will execute the simulation based on the configuration file and display the detected intensity.

---

## Examples

Additional examples can be found in the `examples/` directory of the source code repository. These examples demonstrate various features of COHO, including:

- Using different propagation methods
- Modeling partially coherent sources
- Simulating complex objects and apertures
- Visualizing wavefront propagation

To run an example:

1. **Navigate to the Examples Directory**:

   ```bash
   cd examples
   ```

2. **Run an Example Script**:

   ```bash
   python main_example.py config_example.yml
   ```

---

## Documentation

Comprehensive documentation for COHO is available at:

[**COHO Documentation**](https://coho.readthedocs.org)

The documentation includes:

- Detailed API references
- Tutorials and usage guides
- Information on extending and contributing to the package

---

## Contributing

Contributions to COHO are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button on the GitHub repository page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/dgursoy/coho.git
   ```

3. **Create a Feature Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your feature or bug fix.

5. **Install Development Dependencies**:

   Install development dependencies using Poetry:

   ```bash
   poetry install
   ```

   Activate the virtual environment:

   ```bash
   poetry shell
   ```

6. **Run Tests**:

   ```bash
   pytest
   ```

7. **Commit and Push Changes**:

   ```bash
   git add .
   git commit -m "Add your commit message here"
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**: Submit a pull request to the `main` branch of the original repository.

---

## License

COHO is licensed under the **BSD3 License**. See the [LICENSE](https://github.com/dgursoy/coho/blob/main/LICENSE) file for more information.

---

## Contact

For questions, suggestions, or feedback, please contact:

- **Doga Gursoy**
- **Email**: sparsedata@gmail.com
- **GitHub**: [dgursoy](https://github.com/dgursoy)

---

## Acknowledgments

Special thanks to all contributors and the open-source community for their invaluable support.

---

## Frequently Asked Questions (FAQ)

### Do I need to install Poetry to use COHO?

**No**, end-users do not need to install Poetry. Poetry is used by the developers for managing dependencies and packaging. You can install COHO using `pip` as shown in the [Installation](#installation) section.

### Where can I find more examples and documentation?

Visit the [Documentation](#documentation) section for links to detailed guides, API references, and examples.

### How can I report bugs or request new features?

Please use the [GitHub Issues](https://github.com/dgursoy/coho/issues) page to report bugs or suggest new features.

---

## Version History

- **1.0.0**: Initial release with core functionalities.

For a full changelog, see the [CHANGELOG](https://github.com/dgursoy/coho/blob/main/CHANGELOG.md).

---

## Support

If you need assistance, feel free to open an issue on GitHub or contact the maintainer directly.

---

Thank you for using COHO! We hope it serves as a valuable tool in your holography research and simulations.
