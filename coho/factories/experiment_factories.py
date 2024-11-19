# factories/experiment_factories.py

"""Factories for experiment templates.

This module provides factories for creating high-level experiment
configurations that combine simulation and optimization components.

Classes:
    ExperimentFactory: Creates experiment template instances

Types:
    phase_retrieval: Phase retrieval experiments
    holography: Holographic imaging experiments
    tomography: Tomographic reconstruction experiments
"""

from .base_factory import ComponentFactory
from ..core.experiment import *
from ..config.models import *

__all__ = ['ExperimentFactory']

# Component type mappings
EXPERIMENT_TYPES = {
    'holography': Holography,
    'tomography': Tomography
}

class ExperimentFactory(ComponentFactory[ExperimentProperties, Experiment]):
    """Factory for experiment template creation."""
    def __init__(self):
        super().__init__(EXPERIMENT_TYPES)
