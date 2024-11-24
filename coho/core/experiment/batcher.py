# coho/core/experiment/batcher.py

"""Component state management for experiments.

This module provides base functionality for managing multiple states
of simulation components during parameter sweeps.
"""

from typing import Type, Any, Dict
import copy
import numpy as np

__all__ = ['Batch']

class Batch:
    """Container for managing multiple states of a component."""
    
    def __init__(self, component_class: Type, base_properties: Any, 
                 parameter_arrays: Dict[str, np.ndarray]):
        """Initialize batch of component states."""
        self.base_properties = base_properties
        self.component_class = component_class
        self.parameter_arrays = parameter_arrays
        self.num_states = len(next(iter(parameter_arrays.values())))
        self._validate_arrays()
        self.states = self._create_states()
    
    def _validate_arrays(self):
        """Ensure all parameter arrays have the same length."""
        lengths = [len(arr) for arr in self.parameter_arrays.values()]
        if not all(length == self.num_states for length in lengths):
            raise ValueError("All parameter arrays must have the same length")
    
    def _create_states(self):
        """Create all component instances."""
        return [self.component_class(self._configure(idx)) 
                for idx in range(self.num_states)]
    
    def _configure(self, idx):
        """Configure properties for a specific state."""
        properties = copy.deepcopy(self.base_properties)
        for path, values in self.parameter_arrays.items():
            target = properties
            *parts, attr = path.split('.')
            for part in parts:
                target = getattr(target, part)
            setattr(target, attr, values[idx])
        return properties

    def __getitem__(self, idx):
        return self.states[idx]

    def __len__(self):
        return self.num_states 