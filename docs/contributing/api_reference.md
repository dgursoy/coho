# API Reference

Not yet available online. Check out the [source code](https://github.com/dgursoy/coho/tree/main/coho) for now.

<!-- ::: coho -->

## Configuration Management

The `Config` component in Coho manages user input configurations by processing files (e.g., YAML, JSON), validating their content against predefined schemas, mapping the data to structured Python objects implemented as data classes. These objects are then made accessible to core packages. It supports reliable validation and efficient integration of configuration data.

The configuration management process follows these steps:

1. **Data Reading:** Configuration file content is read.
2. **Validation:** Data is validated against a predefined schema.
3. **Building Models:** Validated data is converted into a [Pydantic](https://docs.pydantic.dev/) model.
4. **Deployment:** The model is accessible to other components in the codebase.
