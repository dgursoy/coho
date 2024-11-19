# engine/simulation.py

"""Core engine for simulation orchestration.

This module handles the orchestration of simulations, including
wavefront propagation, element interactions and measurement

Classes:
    Simulation: Orchestrates simulation workflow

Workflow Stages:
    1. Wavefront initialization
    2. Sequential propagation through elements
    3. Detection and measurement
"""

from typing import Union
from ..config.models import *
from ..factories import *

__all__ = ['Simulation']

class Simulation:
    """Orchestrates experiment workflow."""

    def __init__(
        self, 
        simulation_config: SimulationConfig,
        operator_config: OperatorConfig,
        experiment_config: ExperimentConfig
    ):
        """Initialize simulation components and their states.
        """
        
        # Initialize components with IDs
        self.wavefront = WavefrontFactory().create(
            model=simulation_config.wavefront.model,
            properties=simulation_config.wavefront.properties
        )
        self.optic = OpticFactory().create(
            model=simulation_config.optic.model,
            properties=simulation_config.optic.properties
        )
        self.sample = SampleFactory().create(
            model=simulation_config.sample.model,
            properties=simulation_config.sample.properties
        )
        self.detector = DetectorFactory().create(
            model=simulation_config.detector.model,
            properties=simulation_config.detector.properties
        )

        # Initialize operators
        self.propagator = PropagatorFactory().create(
            model=operator_config.propagator.model,
            properties=operator_config.propagator.properties
        )
        self.interactor = InteractorFactory().create(
            model=operator_config.interactor.model,
            properties=operator_config.interactor.properties
        )

        # Initialize current position
        self.current_position = self.wavefront.properties.geometry.position.z

        # Initialize experiment config
        self.experiment_config = experiment_config

        # Component map with type information
        self.component_map = {
            simulation_config.wavefront.id: {
                'object': self.wavefront,
                'model': 'wavefront'
            },
            simulation_config.optic.id: {
                'object': self.optic,
                'model': 'optic'
            },
            simulation_config.sample.id: {
                'object': self.sample,
                'model': 'sample'
            },
            simulation_config.detector.id: {
                'object': self.detector,
                'model': 'detector'
            }
        }

    def run(self) -> None:
        """Execute complete simulation workflow."""
        current_position = self.wavefront.properties.geometry.position.z
        
        for component_id in self.experiment_config.properties.components:
            component_info = self.component_map[component_id]
            component = component_info['object']
            target_position = component.properties.geometry.position.z
            
            # Propagate to element position if needed
            self.wavefront = self.propagator.propagate(
                self.wavefront, 
                distance=target_position - current_position
            )
            current_position = target_position

            # Interact with element if needed
            if component_info['model'] in ('optic', 'sample'):
                self.wavefront = self.interactor.interact(self.wavefront, component)
            
            # Detect if needed
            if component_info['model'] == 'detector':
                component.detect(self.wavefront)

        # Return detector image
        return self.detector.acquire()
