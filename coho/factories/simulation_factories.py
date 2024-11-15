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

from ..core.simulation.detector import (
    PhotonCountingDetector,
    IntegratingDetector
)
from ..core.simulation.optic import (
    CodedApertureOptic,
    SlitApertureOptic,
    CircleApertureOptic,
    CustomProfileOptic
)
from ..core.simulation.sample import (
    CustomProfileSample
)
from ..core.simulation.wavefront import (
    ConstantWavefront,
    GaussianWavefront,
    RectangularWavefront
)
from .base_factory import ComponentFactory


WAVEFRONT_TYPES = {
    'constant': ConstantWavefront,
    'gaussian': GaussianWavefront,
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

class DetectorFactory(ComponentFactory):
    """Factory for detector creation."""
    def __init__(self):
        super().__init__(DETECTOR_TYPES)


class OpticFactory(ComponentFactory):
    """Factory for optical component creation."""
    def __init__(self):
        super().__init__(OPTIC_TYPES)


class SampleFactory(ComponentFactory):
    """Factory for sample creation."""
    def __init__(self):
        super().__init__(SAMPLE_TYPES)


class WavefrontFactory(ComponentFactory):
    """Factory for wavefront creation."""
    def __init__(self):
        super().__init__(WAVEFRONT_TYPES)

