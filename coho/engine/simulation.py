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

from typing import List, Dict
from ..core.simulation.wavefront import Wavefront
from ..core.simulation.optic import Optic
from ..core.simulation.sample import Sample
from ..core.simulation.detector import Detector
from ..core.operator.propagator import Propagator
from ..core.operator.interactor import Interactor


class Simulation:
    """Orchestrates experiment workflow."""

    def __init__(
        self, 
        wavefront: Wavefront, 
        propagator: Propagator,
        optic: Optic, 
        sample: Sample,
        detector: Detector,
        interactor: Interactor,
        workflow: List[Dict]
    ) -> None:
        """Initialize engine components and workflow.
        
        Args:
            wavefront: Initial field
            propagator: Propagation method
            optic: Optical element
            sample: Sample
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
        self.optic = optic
        self.sample = sample
        
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
        
        # Handle component interaction based on component ID matching
        if component_id == self.optic.id:
            self.wavefront = self.interactor.apply_interaction(self.wavefront, self.optic)
        elif component_id == self.sample.id:
            self.wavefront = self.interactor.apply_interaction(self.wavefront, self.sample)
        elif component_id == self.detector.id:
            self.detector.record_intensity(self.wavefront.amplitude)
        # Skip wavefront component as it's the initial state

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