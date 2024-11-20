# config/models/base.py

"""Base configuration model.

This module defines the base configuration models used throughout the system.
"""

from pydantic import BaseModel
from .simulation import SimulationConfig
from .operator import OperatorConfig
from .experiment import ExperimentConfig
from .optimization import OptimizationConfig

__all__ = [
    'Config'
]

class Config(BaseModel):
    simulation: SimulationConfig = None
    operator: OperatorConfig = None
    experiment: ExperimentConfig = None
    optimization: OptimizationConfig = None 