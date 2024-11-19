# factories/simulation_factories.py

"""Factories for physical simulation components.

This module provides factories for creating physical components
used in optical simulations.

Classes:
    DetectorFactory: Creates detector instances
    OpticFactory: Creates optic instances
    SampleFactory: Creates sample instances
    WavefrontFactory: Creates wavefront instances
"""

from .base_factory import ComponentFactory
from ..core.simulation.detector import *
from ..core.simulation.optic import *
from ..core.simulation.sample import *
from ..core.simulation.wavefront import *
from ..config.models import *

__all__ = ['DetectorFactory', 'OpticFactory', 'SampleFactory', 'WavefrontFactory']

WAVEFRONT_TYPES = {
    'gaussian': GaussianWavefront,
    'constant': ConstantWavefront,
    'rectangular': RectangularWavefront
}

DETECTOR_TYPES = {
    'photon_counting': PhotonCountingDetector,
    'integrating': IntegratingDetector
}

SAMPLE_TYPES = {
    'custom_profile': CustomProfileSample
}

OPTIC_TYPES = {
    'coded_aperture': CodedApertureOptic,
    'slit_aperture': SlitApertureOptic,
    'circle_aperture': CircleApertureOptic,
    'custom_profile': CustomProfileOptic
}

class DetectorFactory(ComponentFactory[DetectorProperties, Detector]):
    """Factory for detector creation."""
    def __init__(self):
        super().__init__(DETECTOR_TYPES)


class OpticFactory(ComponentFactory[OpticProperties, Optic]):
    """Factory for optical component creation."""
    def __init__(self):
        super().__init__(OPTIC_TYPES)


class SampleFactory(ComponentFactory[SampleProperties, Sample]):
    """Factory for sample creation."""
    def __init__(self):
        super().__init__(SAMPLE_TYPES)


class WavefrontFactory(ComponentFactory[WavefrontProperties, Wavefront]):
    """Factory for wavefront creation."""
    def __init__(self):
        super().__init__(WAVEFRONT_TYPES)

