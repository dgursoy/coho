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


class DetectorFactory(ComponentFactory[DetectorProperties, Detector]):
    """Factory for detector creation."""
    def __init__(self):
        super().__init__()
        self.register('photon_counting', PhotonCountingDetector)
        self.register('integrating', IntegratingDetector)


class OpticFactory(ComponentFactory[OpticProperties, Optic]):
    """Factory for optical component creation."""
    def __init__(self):
        super().__init__()
        self.register('coded_aperture', CodedApertureOptic)
        self.register('slit_aperture', SlitApertureOptic)
        self.register('circle_aperture', CircleApertureOptic)
        self.register('custom_profile', CustomProfileOptic)


class SampleFactory(ComponentFactory[SampleProperties, Sample]):
    """Factory for sample creation."""
    def __init__(self):
        super().__init__()
        self.register('custom_profile', CustomProfileSample)


class WavefrontFactory(ComponentFactory[WavefrontProperties, Wavefront]):
    """Factory for wavefront creation."""
    def __init__(self):
        super().__init__()
        self.register('gaussian', GaussianWavefront)
        self.register('constant', ConstantWavefront)
        self.register('rectangular', RectangularWavefront)
