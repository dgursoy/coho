# Quick Start

This guide will help you get started with Coho quickly.

## Prerequisites

- Coho package installed (see [installation guide](installation.md#installation))
- Familiarity with [Python](https://www.python.org/) programming and [YAML](https://yaml.org/) configuration files 
- Understanding of [basic concepts](usage/basics.md) in optical physics and wave propagation

## Basic Usage

Coho enables users to design custom imaging setups through configuration files. Users can define wavefronts, optical elements like coded apertures, samples in the beam path, and detectors. Each component can be customized with specific properties, including wave characteristics, material details, and geometry, allowing for tailored imaging simulations.

### Create a Configuration File

Here's an example configuration file that demonstrates a typical Coho simulation. It models a wavefront as it propagates through a system containing a coded aperture and a thin object, before being captured by a detector. The configuration includes descriptions of all necessary components, their positions, and physical properties. For detailed information about configuration options and parameters, see the [Configuration Guide](usage/configuration.md).

Create a file named `myconfig.yaml` with the following content:

```yaml
# myconfig.yaml - Example configuration for wavefront simulation

experiment:
  - { component_id: "my_wavefront", geometry: { position: 0.0 } }
  - { component_id: "my_aperture", geometry: { position: 1.0 } }
  - { component_id: "my_sample", geometry: { position: 2.0 } }
  - { component_id: "my_detector", geometry: { position: 100.0 } }

operator:
  propagator: { type: "fresnel" }
  interactor: { type: "thin_object" }

simulation:
  wavefront:
    id: "my_wavefront"
    type: "constant"
    properties: { amplitude: 1.0, phase: 0.0, energy: 10.0, shape: 512, spacing: 0.0001 }

  optic:
    id: "my_aperture"
    type: "coded_aperture"
    properties: { material: "Au", density: 19.3, thickness: 0.0001, bit_size: 64 }

  sample:
    id: "my_sample"
    type: "custom_profile"
    properties: 
      material: "C5H8O2"
      density: 1.18
      thickness: 0.001
      custom_profile: "coho/resources/samples/lena.npy"

  detector: { id: "my_detector", type: "integrating" }

optimization:
  objective: { id: "my_objective", type: "least_squares" }
  solver:
    id: "my_solver"
    type: "gradient_descent"
    properties: { step_size: 0.01, iterations: 100 }

```

You can download the example configuration file [here](resources/files/myconfig.yaml).

### Create a Python Script

Create a file named `myscript.py` with the following code to run the simulation and visualize the intensity of the wavefront at the detector after it passes through the system defined in `myconfig.yaml`:

```python
import coho
import matplotlib.pyplot as plt

# Build and initialize the Simulation instance from configuration
simulation = coho.build_simulation_from_config("myconfig.yaml")

# Run and get results
simulation.run()
results = simulation.get_results()

# Plot results (first and only image)
plt.figure(figsize=(8, 6))
plt.imshow(results[0], cmap='gray')
plt.title("Simulated Wavefront Intensity at Detector")
plt.colorbar(label="Intensity")
plt.show()
```

You can download the example script [here](resources/files/mymain.py).

### Run the Simulation

Run the simulation using Python by executing the following command in your terminal:
```bash
python myscript.py myconfig.yaml
```

Voila! You've just run your first Coho simulation.

For more detailed examples and advanced usage, check out the [Examples](examples/index.md) section.
