# engine/workflow.py

"""Core engine for experiment orchestration.

This module handles the orchestration of experiments, including
wavefront propagation, element interactions, measurement and
reconstruction.

Classes:
    Engine: Orchestrates experiment workflow

Workflow Stages:
    1. Wavefront initialization
    2. Sequential propagation through elements
    3. Detection and measurement
"""

from typing import List, Dict
from ..core.simulation.wavefront import Wavefront
from ..core.simulation.sample import Element
from ..core.simulation.detector import Detector
from ..core.operator.propagator import Propagator
from ..core.operator.interactor import Interactor
from ..core.optimization.objectives import Objective
from ..core.optimization.solvers import Solver


class Engine:
    """Orchestrates experiment workflow."""

    def __init__(
        self, 
        wavefront: Wavefront, 
        propagator: Propagator,
        elements: List[Element], 
        detector: Detector,
        interactor: Interactor, 
        objective: Objective,
        solver: Solver,
        workflow: List[Dict]
    ) -> None:
        """Initialize engine components and workflow.
        
        Args:
            wavefront: Initial field
            propagator: Propagation method
            elements: List of optical elements
            detector: Measurement device
            interactor: Interaction handler
            objective: Optimization objective
            solver: Optimization solver
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
        self.objective = objective
        self.solver = solver
        
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

    def simulate(self) -> None:
        """Execute complete simulation workflow."""
        for stage in self.workflow:
            self.process_stage(stage)

    def optimize(self, target_data: Dict) -> Dict:
        """Run optimization loop to match simulation with target data.
        
        Args:
            target_data: Reference measurements to optimize against
            
        Returns:
            Dict containing optimization results and recovered parameters
        """
        # Initialize detector with target data
        self.detector.set_target(target_data)
        
        # Run optimization loop
        result = self.solver.solve(self.objective)
        
        # Run final simulation with optimized parameters
        self.run()
        
        return {
            'parameters': result,
            'reconstruction': self.get_optimization_results()
        }

    def get_simulation_results(self) -> List[Dict]:
        """Retrieve simulation results.
        
        Returns:
            List of detector measurements
        """
        return self.detector.acquire_images()
    
    def get_optimization_results(self) -> Dict:
        """Retrieve optimization results.
        
        Returns:
            Dict containing optimization results and recovered parameters
        """
        return self.optimize()