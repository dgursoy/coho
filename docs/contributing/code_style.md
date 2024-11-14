# Code Style and Conventions

This guide outlines the coding standards and conventions for Coho. Take it with a grain of salt.

## General Guidelines
- **Code Style:** Follow [PEP 8](https://peps.python.org/pep-0008/) as closely as possible.
- **Indentation:** 4 spaces (no tabs)
- **Line Length:** Maximum 88 characters 
- **Docstrings:** Use Google-style docstrings.
- **Type Hints:** Include type hints for all public APIs.

## Example Style

Here's an example of how to format your code:

```python
from typing import List, Optional

class WavePropagator:
    """Class for wave propagation calculations.
    
    Attributes:
        wavelength: The wavelength in meters
        grid_size: Size of the computation grid
    """
    
    def __init__(self, wavelength: float, grid_size: int = 512):
        self.wavelength = wavelength
        self.grid_size = grid_size
        
    def propagate(self, field: np.ndarray, distance: float) -> np.ndarray:
        """Propagates the wave field.
        
        Args:
            field: Input complex field
            distance: Propagation distance in meters
            
        Returns:
            Propagated complex field
        """
        # Implementation
        pass
```

Now that you've developed your code, check out how to [submit a pull request](pull_request.md).
