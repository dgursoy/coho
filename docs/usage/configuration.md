# Configuration Guide

We use configuration files to define the components in the simulation and the experiment states. The configuration files are written in [YAML](https://yaml.org/), [JSON](https://www.json.org/), or [TOML](https://toml.io/) format. If your configuration file doesn't match the expected schema, Coho will provide error messages to help you identify and fix any issues


### Structure and Requirements

The experiment configuration includes the following sections:

- `simulation`: Defines the components in the simulation, including wavefront, optic, sample, and detector.
- `operator`: Defines the operators, including propagator and interactor.
- `experiment`: Defines the experiment, including components and experiment states.

### Configuration Example

Here's an example component configuration in YAML:

```yaml
simulation:
  wavefront:
    id: "my_source"
    model: "gaussian"
    properties: 
      physical:
        amplitude: 1.0
        phase: 0.0
        energy: 10.0
      profile:
        sigma: 256
        width: 512
        height: 512
      grid:
        size: 512
        spacing: 0.0001
      geometry:
        position:
          x: 0.0
          y: 0.0
          z: 0.0
        rotation: 0.0
```

Each component needs an `id` as a unique identifier, a `model` to specify its behavior, and `properties` that include both component-specific properties (like `physical` and `profile`) as well as common properties (like `grid` and `geometry`).

We can access the configuration in Python as follows:

```python
from coho import load_config

# Build data classes from configuration file
config = load_config("snippets/test_config.yaml")

# Access data classes
simulation = config.simulation
wavefront = simulation.wavefront

# Properties are strongly typed
print(f"Wavefront amplitude: {wavefront.properties.physical.amplitude}")
```
