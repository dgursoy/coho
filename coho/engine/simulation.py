# simulation.py

"""Core simulation engine for optical wavefront propagation.

This module handles sequential propagation, element interactions,
and detection in configurable workflows.

Classes:
    Simulation: Orchestrates wavefront propagation sequence

Workflow Stages:
    1. Wavefront initialization
    2. Sequential propagation through elements
    3. Detection and measurement
"""

from typing import List, Dict
from ..core.wavefront import Wavefront
from ..core.element import Element
from ..core.detector import Detector
from ..core.propagator import Propagator
from ..core.interactor import Interactor


class Simulation:
    """Orchestrates wavefront propagation through optical system."""

    def __init__(
        self, 
        wavefront: Wavefront, 
        propagator: Propagator,
        elements: List[Element], 
        detector: Detector,
        interactor: Interactor, 
        workflow: List[Dict]
    ) -> None:
        """Initialize simulation components and workflow.
        
        Args:
            wavefront: Initial field
            propagator: Propagation method
            elements: List of optical elements
            detector: Measurement device
            interactor: Interaction handler
            workflow: Component sequence:
                [{"component_id": str, 
                  "geometry": {"position": float}}, 
                 ...]
        """
        # Core components
        self.wavefront = wavefront
        self.propagator = propagator
        self.detector = detector
        self.interactor = interactor
        
        # Create element lookup by ID
        self.elements = {element.id: element for element in elements}
        
        # Workflow and position tracking
        self.workflow = workflow
        self.current_position = 0.0

    def propagate(self, target_position: float) -> None:
        """Propagate wavefront to specified position.
        
        Args:
            target_position: Absolute position to propagate to
        """
        distance = target_position - self.current_position
        self.wavefront = self.propagator.propagate(self.wavefront, distance=distance)
        self.current_position = target_position

    def process_stage(self, stage: Dict) -> None:
        """Process a single workflow stage.
        
        Args:
            stage: Workflow stage configuration
        """
        component_id = stage["component_id"]
        position = stage["geometry"]["position"]
        
        # Move to component position
        self.propagate(position)
        
        # Handle component interaction
        if component_id in self.elements:
            element = self.elements[component_id]
            self.wavefront = self.interactor.apply_interaction(self.wavefront, element)
        elif component_id == self.detector.id:
            self.detector.record_intensity(self.wavefront.amplitude)

    def run(self) -> None:
        """Execute complete simulation workflow."""
        for stage in self.workflow:
            self.process_stage(stage)

    def get_results(self) -> List[Dict]:
        """Retrieve simulation results.
        
        Returns:
            List of detector measurements
        """
        return self.detector.acquire_images()
