# factories/detector_factory.py

"""Factory for creating detector instances.

This module implements factory pattern for detector creation
in optical simulations.

Classes:
    DetectorFactory: Creates configured detector instances

Types:
    photon_counting: Discrete photon detection
    integrating: Continuous intensity measurement

DETECTOR_TYPES:
    DETECTOR_TYPES: Mapping of type names to classes
"""

from typing import Dict, Any
from ..core.detector import (
    Detector,
    PhotonCountingDetector, 
    IntegratingDetector
)


DETECTOR_TYPES = {
    'photon_counting': PhotonCountingDetector,
    'integrating': IntegratingDetector
}


class DetectorFactory:
    """Factory for detector instance creation."""
    
    @staticmethod
    def create_detector(
        id: Any, 
        type: str, 
        parameters: Dict[str, Any]
    ) -> Detector:
        """Create configured detector instance.

        Args:
            id: Unique identifier
            type: Detector type
                'photon_counting': Discrete detection
                'integrating': Continuous measurement
            parameters: Configuration dictionary

        Returns:
            Configured detector instance

        Raises:
            ValueError: Unknown detector type
        """
        detector_type = type.lower()
        detector_class = DETECTOR_TYPES.get(detector_type)
        
        if detector_class is None:
            raise ValueError(
                f"Unknown detector type: {type}. "
                f"Supported types: {list(DETECTOR_TYPES.keys())}"
            )
            
        return detector_class(id, parameters)
