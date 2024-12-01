from typing import Dict
import numpy as np
from ..operator import Pipeline, FresnelPropagate, Interact, Detect
from ..component import Component

class HolographyScan:
    """Holographic scan experiment."""
    
    def __init__(self):
        """Initialize holography operators."""
        self.fresnel = FresnelPropagate()
        self.interact = Interact()
        self.detect = Detect()
        self.distances = [1.0, 1.0, 50.0]  # Default distances
    
    def _create_pipeline(self, components: Dict[str, Component]) -> Pipeline:
        """Create pipeline based on available components."""
        pipeline_steps = []
        distance_idx = 0
        
        # Always start with wavefront
        if 'wavefront' not in components:
            raise ValueError("Wavefront component is required")
            
        # Add optic if present
        if 'optic' in components:
            pipeline_steps.extend([
                (self.fresnel, {'distance': self.distances[distance_idx]}),
                (self.interact, {'component': components['optic']})
            ])
            distance_idx += 1
            
        # Add sample if present
        if 'sample' in components:
            pipeline_steps.extend([
                (self.fresnel, {'distance': self.distances[distance_idx]}),
                (self.interact, {'component': components['sample']})
            ])
            distance_idx += 1
            
        # Always end with detector
        if 'detector' not in components:
            raise ValueError("Detector component is required")
            
        pipeline_steps.extend([
            (self.fresnel, {'distance': self.distances[distance_idx]}),
            (self.detect, {'component': components['detector']})
        ])
        
        return Pipeline(pipeline_steps)
    
    def run(self, components: Dict[str, Component]) -> np.ndarray:
        """Run holography experiment."""
        # Create dynamic pipeline
        pipeline = self._create_pipeline(components)
        
        # Run forward propagation
        return pipeline.apply(components['wavefront'])
