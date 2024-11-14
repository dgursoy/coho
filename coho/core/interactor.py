# core/interactor.py

"""Classes for simulating wavefront interactions with optical elements.

This module handles wavefront modifications through different interaction models
for both thin and thick optical elements.

Classes:
    Interactor: Base class for all interactions
    ThinObjectInteractor: Simple transmission/reflection effects
    ThickObjectInteractor: Multi-slice propagation effects

Methods:
    apply_interaction: Apply element effects to wavefront
    compute_amplitude_attenuation: Calculate amplitude changes
    compute_phase_shift: Calculate phase modifications

Attributes:
    wavefront: Wavefront being modified
    element: Optical element causing modification
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import xraylib
from coho.core.wavefront import Wavefront
from coho.core.element import Element


class Interactor(ABC):
    """Base class for wavefront-element interactions."""

    def __init__(self, id: Any, wavefront: Wavefront) -> None:
        """Initialize interactor.

        Args:
            id: Unique identifier
            wavefront: Wavefront to modify
        """
        self.id = id
        self.wavefront = wavefront

    @abstractmethod
    def apply_interaction(self, element: Element) -> None:
        """Apply element effects to wavefront.

        Args:
            element: Optical element
        """
        pass


class ThinObjectInteractor(Interactor):
    """Thin optical element interaction handler."""

    def compute_amplitude_attenuation(self, energy: float, element: Element) -> np.ndarray:
        """Calculate amplitude attenuation.

        Args:
            energy: Photon energy (keV)
            element: Optical element

        Returns:
            Attenuation factors
        """
        beta = xraylib.Refractive_Index_Im(
            element.material, 
            energy, 
            element.density
        )
        return np.exp(
            -self.wavefront.wavenumber * 
            beta * 
            element.thickness * 
            element.pattern
        )

    def compute_phase_shift(self, energy: float, element: Element) -> np.ndarray:
        """Calculate phase shift.

        Args:
            energy: Photon energy (keV)
            element: Optical element

        Returns:
            Phase shifts
        """
        delta = 1 - xraylib.Refractive_Index_Re(
            element.material, 
            energy, 
            element.density
        )
        return -self.wavefront.wavenumber * delta * element.thickness * element.pattern

    def apply_interaction(self, element: Element) -> None:
        """Apply amplitude and phase modifications.

        Args:
            element: Optical element
        """
        attenuation = self.compute_amplitude_attenuation(
            self.wavefront.energy, 
            element
        )
        phase_shift = self.compute_phase_shift(
            self.wavefront.energy, 
            element
        )

        self.wavefront.amplitude *= attenuation
        self.wavefront.phase += phase_shift


class ThickObjectInteractor(Interactor):
    """Thick optical element interaction handler.
    
    Future implementation will include:
    - Multi-slice propagation
    - Internal structure effects
    - Volume interactions
    """

    def compute_amplitude(self, energy: float, element: Element) -> np.ndarray:
        """Calculate multi-slice amplitude (not implemented).

        Args:
            energy: Photon energy (keV)
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Multi-slice propagation not implemented")

    def compute_phase_shift(self, energy: float, element: Element) -> np.ndarray:
        """Calculate thick object phase shift (not implemented).

        Args:
            energy: Photon energy (keV)
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Phase calculation not implemented")

    def apply_interaction(self, element: Element) -> None:
        """Apply thick object effects (not implemented).

        Args:
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Thick object interaction not implemented")
