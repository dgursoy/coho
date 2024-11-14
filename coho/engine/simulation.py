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

from typing import List, Dict, Optional
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
        """Initialize simulation.

        Args:
            wavefront: Initial field
            propagator: Propagation method
            elements: Optical elements
            detector: Measurement device
            interactor: Interaction handler
            workflow: Component sequence:
                [{"component_id": str, 
                  "geometry": {"position": float}}, 
                 ...]
        """
        self.wavefront = wavefront
        self.propagator = propagator
        self.elements = {element.id: element for element in elements}
        self.detector = detector
        self.interactor = interactor
        self.workflow = workflow
        self.previous_position = 0.0

    def _propagate_to_position(self, position: float) -> None:
        """Propagate to absolute position.

        Args:
            position: Target position
        """
        distance = position - self.previous_position
        self.wavefront = self.propagator.propagate(
            self.wavefront, 
            distance=distance
        )
        self.previous_position = position

    def _process_wavefront(self) -> None:
        """Initialize wavefront position."""
        for stage in self.workflow:
            if stage["component_id"] == self.wavefront.id:
                self._propagate_to_position(
                    stage["geometry"]["position"]
                )
                return

    def _process_elements(self) -> None:
        """Process element interactions."""
        for stage in self.workflow:
            if stage["component_id"] in self.elements:
                element = self.elements[stage["component_id"]]
                self._propagate_to_position(
                    stage["geometry"]["position"]
                )
                self.interactor.apply_interaction(element)

    def _process_detector(self) -> None:
        """Record detector measurements."""
        for stage in self.workflow:
            if stage["component_id"] == self.detector.id:
                self._propagate_to_position(
                    stage["geometry"]["position"]
                )
                self.detector.record_intensity(
                    self.wavefront.amplitude
                )
                return

    def run(self) -> None:
        """Execute simulation workflow.

        Raises:
            KeyError: Unknown component in workflow
        """
        self._process_wavefront()
        self._process_elements()
        self._process_detector()

    def get_results(self) -> List[Dict]:
        """Get simulation results.

        Returns:
            Detector measurements
        """
        return self.detector.acquire_images()
