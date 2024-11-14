# Developer Setup Guide

This guide provides a step-by-step setup process for developing Coho.

## Prerequisites

- [Git](https://git-scm.com/) for cloning the repository and managing commits
- [Poetry](https://python-poetry.org/) for managing Python dependencies
- [MkDocs](https://www.mkdocs.org/) for building and previewing the documentation locally
- [Pre-commit](https://pre-commit.com/) for running pre-commit hooks (optional)

## Environment Setup

This section guides you through setting up your local development environment.

### Clone the Repository

Coho is hosted on [GitHub](https://github.com/dgursoy/coho). Use the following commands to clone the repository and install the project dependencies:

   ```bash
   # Clone the repository
   git clone https://github.com/dgursoy/coho.git
   cd coho

   # Install project dependencies using Poetry
   poetry install

   # Create and activate the virtual environment
   poetry shell

   # Verify the installation
   poetry run python -c "import coho; print(coho.__version__)"
   ```

### Managing Dependencies

Coho uses [Poetry](https://python-poetry.org/) for dependency management. Here's how to manage project dependencies:

```bash
# Add a production dependency
poetry add numpy

# Add a development-only dependency
poetry add --group dev black

# Add a documentation-related dependency
poetry add --group docs mkdocs-material

# Install pre-commit as a dev dependency
poetry add --group dev pre-commit
```

> **Note:** You can also manually edit dependencies in `pyproject.toml` in the project root. 

Coho uses a lock file to manage dependencies. To update the lock file after any changes to dependencies:

```bash
poetry lock  # Update the lock file
poetry install  # Apply the changes
```

> **Note:** If you want to update all dependencies to their latest compatible versions:
> 
> ```bash
> poetry update
> ```

See, much easier than others!

### Testing Environment

To verify your environment setup is correct:

```bash
poetry run pytest tests/
```

> **Note:** Always use `poetry run` to ensure tests run within the project's virtual environment. Running `pytest tests` directly will either use your global Python environment (which may have different package versions) or fail if `pytest` isn't installed globally.

### Documentation Setup

The documentation is located in the `docs` folder in the project root. You can build and preview the documentation locally using [MkDocs](https://www.mkdocs.org):

```bash
# Start the documentation server with live preview
poetry run mkdocs serve

# Access the documentation in your browser
# View at http://localhost:8000
```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run automated checks before each commit, configured via `.pre-commit-config.yaml` in the project root.

```bash
# Install pre-commit as a dev dependency
poetry add --group dev pre-commit

# Install pre-commit hooks into your local repository
poetry run pre-commit install

# Run all pre-commit checks on staged files (those about to be committed)
poetry run pre-commit run
```

> **Note**: When you make a git commit, `pre-commit` automatically runs on only the staged files (those being committed). The `--all-files` flag is useful for checking your entire codebase, including files that aren't tracked by git yet.
> 
> **Common Commands**:
> ```bash
> # Update hooks to latest versions
> poetry run pre-commit autoupdate
> 
> # Check all files (including untracked, respects .gitignore)
> poetry run pre-commit run --all-files
> 
> # Check specific files
> poetry run pre-commit run --files path/to/file.py
> 
> # Check specific file types
> poetry run pre-commit run --files "*.py" "*.yaml"
>
> # Clean up hooks to remove unused hooks
> poetry run pre-commit clean
> ```

> **Note:** It's recommended to set up pre-commit hooks right after cloning the repository to ensure code quality from the start.

Now that you've set up your development environment, check out the [code style guide](code_style.md) before writing any code.
