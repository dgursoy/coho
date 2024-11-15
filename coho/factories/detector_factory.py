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

from ..core.detector import (
    PhotonCountingDetector, 
    IntegratingDetector
)
from .base_factory import ComponentFactory


DETECTOR_TYPES = {
    'photon_counting': PhotonCountingDetector,
    'integrating': IntegratingDetector
}


class DetectorFactory(ComponentFactory):
    """Factory for detector creation."""
    
    def __init__(self):
        super().__init__(DETECTOR_TYPES)
