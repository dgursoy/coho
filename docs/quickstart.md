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
experiment:
  id: "my_experiment"
  model: "holography"
  properties:
    components:
      - "my_wavefront"
      - "my_aperture"
      - "my_sample"
      - "my_detector"


simulation:
  wavefront:
    id: my_wavefront
    model: constant
    properties:
      physical:
        amplitude: 1.0
        phase: 1.0
        energy: 10.0
      grid:
        size: 512
        spacing: 0.0001
      geometry:
        position:
          z: 0.0

  optic:
    id: "my_aperture"
    model: "coded_aperture"
    properties:
      physical:
        formula: "Au"
        density: 19.3
        thickness: 0.0001
      profile:
        model: "binary_random"
        bit_size: 64
        seed: 0
      geometry:
        position:
          z: 1.0

  sample:
    id: "my_sample"
    model: "custom_profile"
    properties:
      physical:
        formula: "C5H8O2"
        density: 1.18
        thickness: 0.001
      profile:
        model: "custom_profile"
        file_path: "coho/resources/samples/lena.npy"
      geometry:
        position:
          z: 2.0

  detector:
    id: "my_detector"
    model: "integrating"
    properties: 
      geometry:
        position:
          z: 100.0

operator:
  propagator:
    id: "my_propagator"
    model: "fresnel"

  interactor:
    id: "my_interactor"
    model: "thin_object"
```

You can download the example configuration file [here](resources/files/myconfig.yaml).

### Create a Python Script

Create a file named `myscript.py` with the following code to run the simulation and visualize the intensity of the wavefront at the detector after it passes through the system defined in `myconfig.yaml`:

```python
import coho
import matplotlib.pyplot as plt

# Load configuration from file
config = coho.load_config("myconfig.yaml")

# Create and run simulation
forward = coho.Holography(config)
image = forward.run()

# Plot detector image
plt.imshow(image[0], cmap='gray')
plt.title("Image captured by detector")
plt.colorbar()
plt.show()
```

You can download the example script [here](resources/files/mymain.py).

### Run the Simulation

Run the simulation using Python by executing the following command in your terminal:
```bash
python myscript.py
```

Voila! You've just run your first Coho simulation.

For more detailed examples and advanced usage, check out the [Examples](examples/index.md) section.
