# coho/core/experiment/sweeper.py

"""Parameter sweep generation for experiments.

This module handles the generation of parameter combinations for
experimental sweeps based on configuration settings.
"""

from typing import Dict, Any
import numpy as np
from coho.config.models.experiment import ExperimentConfig, ComponentSweep

__all__ = ['prepare']

def get_sweep_parameters(scan_config):
    """Extract sweep parameters in correct order from scan configuration.
    
    Args:
        scan_config: Configuration containing sweeps and their order
    Returns:
        components: List of component IDs in order
        paths: List of parameter paths in order
        ranges: List of parameter ranges in order
    """
    if scan_config is None:
        return [], [], []
    
    # Create ordered sweeps first
    ordered_sweeps = []
    for idx in scan_config.order:
        ordered_sweeps.append(scan_config.sweeps[idx-1])  # -1 because order is 1-based
        
    # Collect parameters maintaining component associations
    components = []
    paths = []
    ranges = []
    for sweep in ordered_sweeps:
        for param in sweep.parameters:
            components.append(sweep.component)
            paths.append(param.path)
            ranges.append(tuple(param.range))
    
    return components, paths, ranges

def prepare(config):
    """Generate parameter arrays for sweeping.
    
    Args:
        config: Configuration containing experiment and simulation settings
    Returns:
        Dictionary mapping component IDs to their parameter arrays
    """
    # Initialize parameter arrays for all components
    param_dict = {comp: {} for comp in config.experiment.properties.components}
    
    # If no scan config, return empty parameter dictionaries for all components
    scan_config = config.experiment.properties.scan
    if not scan_config:
        return param_dict
        
    # Get ordered parameters
    components, paths, ranges = get_sweep_parameters(scan_config)
    if not paths:
        return param_dict
        
    # Generate sweep combinations
    result = _generate_sweep(ranges)
    
    # Add sweep parameters to the corresponding components
    for comp, path, values in zip(components, paths, result):
        param_dict[comp][path] = values
        
    return param_dict

def _generate_sweep(ranges: list) -> np.ndarray:
    """Generate parameter sweep combinations.
    
    Args:
        ranges: List of (start, end, step) tuples
        
    Returns:
        Array of parameter combinations
    """
    # Generate arrays for each parameter
    arrays = [np.arange(start, end + step/2, step) 
             for start, end, step in ranges]
    
    # Create mesh grid of all combinations
    mesh = np.meshgrid(*arrays, indexing='ij')
    
    # Return flattened arrays
    return np.array([grid.flatten() for grid in mesh])