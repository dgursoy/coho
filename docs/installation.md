# Installation

## Requirements

- [**Python**](https://www.python.org/downloads/): 3.12 or higher
- **Dependencies**: Managed via [Poetry](https://python-poetry.org/)

## Installing from PyPI

> **Note:** Coho is not available on PyPI yet.

You can install Coho directly from PyPI using `pip`:

```bash
pip install coho
```

For an isolated environment (recommended), use:

```bash
# Create a new virtual environment named 'mycoho'
python -m venv mycoho

# Activate the virtual environment
source venv/bin/activate

# Install Coho in the virtual environment
pip install coho
```

## Installing from Source

To install Coho from the source code:

```bash
# Clone the repository
git clone https://github.com/dgursoy/coho.git
cd coho

# Install Poetry if you haven't already
pip install poetry

# Or using the official installer script
curl -sSL https://install.python-poetry.org | python3 -

# Or using Homebrew
brew install poetry

# Install Coho and its dependencies 
poetry install

# Activate the shell
poetry shell
```

## Verifying Installation

To verify that Coho was installed correctly:

```bash
coho --version
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you have the correct Python version installed
2. Update pip to the latest version: `pip install --upgrade pip`
3. Check our [GitHub Issues](https://github.com/dgursoy/coho/issues) for known problems
4. For Poetry-related issues, refer to the [Poetry documentation](https://python-poetry.org/docs/)
