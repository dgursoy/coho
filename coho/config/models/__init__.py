# config/models/__init__.py

"""Configuration models package.

This package contains Pydantic model definitions for all configuration
components used in optical simulations.

Model Categories:
    Common:
        Basic shared data structures for geometry and grids
    
    Simulation:
        Models for optical simulation components (wavefront, optics, etc.)
    
    Operator:
        Models for wave propagation and interaction operators
    
    Experiment:
        Models for experiment workflow and component management
    
    Optimization:
        Models for optimization objectives and solvers
"""

from .common import Grid, Geometry, Position
from .simulation import *
from .operator import *
from .experiment import *
from .optimization import *

__all__ = [
    # Common models
    'Grid', 'Geometry', 'Position',
    
    # Simulation models
    'Wavefront', 'WavefrontProperties', 'WavefrontPhysical', 'WavefrontProfile',
    'Optic', 'OpticProperties', 'OpticPhysical', 'OpticProfile',
    'Sample', 'SampleProperties', 'SamplePhysical', 'SampleProfile',
    'Detector', 'DetectorProperties',
    'SimulationConfig',
    
    # Operator models
    'Interactor', 
    'Propagator', 
    'OperatorConfig',
    
    # Experiment models
    'ExperimentProperties', 
    'ExperimentConfig',
    
    # Optimization models
    'Solver', 'SolverProperties', 
    'Objective', 
    'OptimizationConfig'
]

