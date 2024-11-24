# coho/core/experiment/sweeper.py

"""Parameter sweep generation for experiments.

This module handles the generation of parameter combinations for
experimental sweeps based on configuration settings.
"""

import numpy as np

__all__ = ['prepare']

def prepare(config):
    """Convert config sweep definitions into parameter arrays."""
    # Check if scan properties exist
    print (config.experiment.properties.scan)
    if config.experiment.properties.scan is None:
        return {}  # Return empty dict if no scan defined
    
    # Collect paths and ranges
    paths = []
    ranges = []
    for sweep in config.experiment.properties.scan.sweeps:
        for param in sweep.parameters:
            paths.append(param.path)
            ranges.append(tuple(param.range))
    
    # Sort paths and ranges according to order
    order = config.experiment.properties.scan.order
    ranges = [r for _, r in sorted(zip(order, ranges))]
    paths = [p for _, p in sorted(zip(order, paths))]
    
    # Generate parameter sweep combinations
    result = _generate_sweep(ranges)
    return dict(zip(paths, result))

def _generate_sweep(ranges):
    """Generate parameter sweep combinations."""
    arrays = [np.arange(start, end + step/2, step) for start, end, step in ranges]
    mesh = np.meshgrid(*arrays, indexing='ij')
    return np.array([grid.flatten() for grid in mesh])