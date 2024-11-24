# coho/core/experiment/sweeper.py

"""Parameter sweep generation for experiments.

This module handles the generation of parameter combinations for
experimental sweeps based on configuration settings.
"""

import numpy as np

__all__ = ['prepare']

def prepare(config):
    """Convert config sweep definitions into parameter arrays."""
    paths = []
    ranges = []
    for sweep in config.experiment.properties.scan.sweeps:
        for param in sweep.parameters:
            paths.append(param.path)
            ranges.append(tuple(param.range))
    
    result = _generate_sweep(ranges, config.experiment.properties.scan.order)
    return dict(zip(paths, result))

def _generate_sweep(ranges, order):
    """Generate parameter sweep combinations."""
    arrays = [np.arange(start, end + step/2, step) for start, end, step in ranges]
    mesh = np.meshgrid(*arrays, indexing='ij')
    flattened = np.array([grid.flatten() for grid in mesh])
    return flattened[np.array(order)-1]