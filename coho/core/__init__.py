# core/__init__.py

"""Core components for optical simulation.

This package provides classes for light propagation,
interaction, and detection.

Modules:
    wavefront: Light field representations
        constant: Uniform profile
        gaussian: Gaussian profile
        rectangular: Rectangular profile

    element: Optical elements
        coded_aperture: Pattern masks
        slit_aperture: Slit openings
        circle_aperture: Circular openings
        custom_profile: User patterns

    propagator: Field propagation
        fresnel: Near-field
        fraunhofer: Far-field

    interactor: Wave-element interaction
        thin_object: Surface effects
        thick_object: Volume effects

    detector: Measurement devices
        integrating: Intensity
        photon_counting: Discrete photons
"""

from .detector import (
    IntegratingDetector,
    PhotonCountingDetector
)
from .element import (
    CodedApertureElement,
    SlitApertureElement,
    CircleApertureElement,
    CustomProfileElement
)
from .interactor import (
    ThinObjectInteractor,
    ThickObjectInteractor
)
from .propagator import (
    FresnelPropagator,
    FraunhoferPropagator
)
from .wavefront import (
    ConstantWavefront,
    GaussianWavefront,
    RectangularWavefront
)

__all__ = (
    "IntegratingDetector",
    "PhotonCountingDetector",
    "CodedApertureElement",
    "SlitApertureElement",
    "CircleApertureElement",
    "CustomProfileElement",
    "ThinObjectInteractor",
    "ThickObjectInteractor",
    "FresnelPropagator",
    "FraunhoferPropagator",
    "ConstantWavefront",
    "GaussianWavefront",
    "RectangularWavefront"
)
