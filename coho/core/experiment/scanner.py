"""Experiment scanning orchestration.

This module coordinates parameter sweeps and component batching
for experimental workflows.
"""

from typing import Dict, Type
import numpy as np
from .sweeper import prepare
from .batcher import Batch

__all__ = ['Scanner']

class Scanner:
    """Orchestrates parameter scanning experiments."""
    
    def __init__(self, config):
        """Initialize scanner with experiment config."""
        self.config = config
        self.parameter_arrays = prepare(config)
        self.batches = {}
    
    def create_batch(self, component_class: Type, base_properties: Dict) -> Batch:
        """Create a new component batch for scanning."""
        return Batch(component_class, base_properties, self.parameter_arrays)
    
    def run(self):
        """Execute scanning workflow."""
        # To be implemented based on experiment requirements
        pass 