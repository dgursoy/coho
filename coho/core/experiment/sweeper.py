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
    
    paths = [p for _, p in sorted(zip(config.experiment.properties.scan.order, paths))]
    ranges = [r for _, r in sorted(zip(config.experiment.properties.scan.order, ranges))]
    result = _generate_sweep(ranges)
    return dict(zip(paths, result))

def _generate_sweep(ranges):
    """Generate parameter sweep combinations."""
    arrays = [np.arange(start, end + step/2, step) for start, end, step in ranges]
    mesh = np.meshgrid(*arrays, indexing='ij')
    flattened = np.array([grid.flatten() for grid in mesh])
    return flattened