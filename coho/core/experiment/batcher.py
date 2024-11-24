# coho/core/experiment/batcher.py

"""Component state management for experiments.

This module provides base functionality for managing multiple states
of simulation components during parameter sweeps.
"""

from typing import Type, Any, Dict
import numpy as np

__all__ = [
    'Batch',
    'BatchWavefront',
    'BatchOptic',
    'BatchSample',
    'BatchDetector'
]

class Batch:
    """Base container for managing parameter states of a component."""
    
    def __init__(self, component_class: Type, base_properties: Any, 
                 parameter_arrays: Dict[str, np.ndarray] = None):
        """Initialize batch parameters.
        
        Args:
            component_class: Class of the component to batch
            base_properties: Base properties for the component
            parameter_arrays: Optional dictionary of parameter arrays for sweeping
        """
        self.base_properties = base_properties
        self.component_class = component_class
        self.parameter_arrays = parameter_arrays or {}
        
        # Create single reusable component instance
        self._instance = self.component_class(base_properties)
        
        # Set num_states based on parameter arrays if they exist
        self.num_states = (len(next(iter(self.parameter_arrays.values())))
                          if self.parameter_arrays else 1)
        
        if self.parameter_arrays:
            self._validate_arrays()
    
    def _validate_arrays(self):
        """Ensure all parameter arrays have the same length."""
        lengths = [len(arr) for arr in self.parameter_arrays.values()]
        if not all(length == self.num_states for length in lengths):
            raise ValueError("All parameter arrays must have the same length")
    
    def get_state(self, idx):
        """Get component state for specific index."""
        # Only update properties if we have parameter arrays
        if self.parameter_arrays:
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

class BatchWavefront(Batch):
    """Container for wavefront parameter states."""
    
    @property
    def complex_wavefront(self):
        """Generate array of complex wavefronts."""
        return np.array([
            self.get_state(idx).complex_wavefront 
            for idx in range(self.num_states)
        ])

class BatchOptic(Batch):
    """Container for optics with varying parameters."""
    
    @property
    def profile(self):
        """Get array of all profiles."""
        for idx in range(self.num_states):
            print (idx, self.get_state(idx))
        return np.array([
            self.get_state(idx).profile 
            for idx in range(self.num_states)
        ])

class BatchSample(Batch):
    """Container for samples with varying parameters."""
    
    @property
    def profile(self):
        """Get array of all profiles."""
        return np.array([
            self.get_state(idx).profile 
            for idx in range(self.num_states)
        ])

class BatchDetector(Batch):
    """Detector for vectorized batch measurements."""
    
    def detect(self, batch_wavefront: BatchWavefront) -> np.ndarray:
        """Record and return vectorized measurements for batch of wavefronts."""
        intensities = np.abs(batch_wavefront.complex_wavefront) ** 2
        return intensities