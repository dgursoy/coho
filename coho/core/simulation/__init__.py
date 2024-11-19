# core/simulation/__init__.py

"""Core simulation components for optical systems.

This module provides base classes for physical components used in
optical simulations.

Components:
    Detector: Measurement devices
        integrating: Continuous intensity measurement
        photon_counting: Discrete photon detection

    Element: Optical elements
        coded_aperture: Patterned transmission masks
        slit_aperture: Single slit openings
        circle_aperture: Circular openings
        custom_profile: User-defined patterns

    Wavefront: Light field representations
        constant: Uniform amplitude profile
        gaussian: Gaussian amplitude distribution
        rectangular: Rectangular amplitude profile
"""

from .detector import *
from .sample import *
from .optic import *
from .wavefront import *

__all__ = [
    'IntegratingDetector',
    'PhotonCountingDetector',
    'CodedApertureOptic', 
    'SlitApertureOptic',
    'CircleApertureOptic',
    'CustomProfileOptic',
    'CustomProfileSample',
    'ConstantWavefront',
    'GaussianWavefront',
    'RectangularWavefront'
]
