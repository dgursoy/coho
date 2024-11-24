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
    """Container for managing parameter states of a component."""
    
    def __init__(self, component_class: Type, base_properties: Any, 
                 parameter_arrays: Dict[str, np.ndarray]):
        """Initialize batch parameters."""
        self.base_properties = base_properties
        self.component_class = component_class
        self.parameter_arrays = parameter_arrays
        self.num_states = len(next(iter(parameter_arrays.values())))
        self._validate_arrays()
        
        # Create single reusable component instance
        self._instance = self.component_class(base_properties)
    
    def _validate_arrays(self):
        """Ensure all parameter arrays have the same length."""
        lengths = [len(arr) for arr in self.parameter_arrays.values()]
        if not all(length == self.num_states for length in lengths):
            raise ValueError("All parameter arrays must have the same length")
    
    def get_state(self, idx):
        """Get component state for specific index."""
        # Update instance properties for this state
        for path, values in self.parameter_arrays.items():
            target = self._instance.properties
            *parts, attr = path.split('.')
            for part in parts:
                target = getattr(target, part)
            setattr(target, attr, values[idx])
            
        # Clear component's cached computations
        if hasattr(self._instance, 'clear_cache'):
            self._instance.clear_cache()
        
        return self._instance

    def __getitem__(self, idx):
        return self.get_state(idx)

    def __len__(self):
        return self.num_states 