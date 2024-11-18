# Basic Concepts

## Configuration Management

### Overview

The `Config` component in Coho handles user input configurations by processing files (e.g., YAML, JSON), validating their content against predefined schemas, mapping the data to structured Python objects, and making these objects accessible to core packages. It supports reliable validation and efficient integration of configuration data.

The configuration management process follows these steps:

1. **Data Reading:** Configuration file content is read.
2. **Validation:** Data is validated against a predefined schema.
3. **Building Models:** Validated data is converted into a Pydantic model.
4. **Deployment:** The model is accessible to other components in the codebase.

### Configuration File Formats

Coho currently supports [YAML](https://yaml.org/), [JSON](https://www.json.org/), and [TOML](https://toml.io/) configuration file formats. If your configuration file doesn't match the expected schema, Coho will provide error messages to help you identify and fix any issues.

### Structure and Requirements

The experiment configuration includes the following sections:

- `simulation`: Defines the components in the simulation, including wavefront, optic, sample, and detector.
- `operator`: Defines the operators, including propagator and interactor.
- `experiment`: Defines the experiment, including components and experiment states.

### Component Definitions

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
from coho.config import load_config

# Load and validate configuration
config = load_config("simulation_config.yaml")

# Access typed configuration objects
simulation = config["simulation"]
wavefront = simulation.wavefront

# Properties are strongly typed
print(f"Wavefront amplitude: {wavefront.properties.physical.amplitude}")
```

For more details on component definitions, see the [Configuration Guide](./configuration.md) section.

