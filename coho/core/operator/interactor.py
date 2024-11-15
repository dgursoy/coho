# core/operator/interactor.py

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
from typing import Any, Optional, Dict
import numpy as np
import xraylib
from coho.core.simulation.wavefront import Wavefront
from coho.core.simulation.element import Element


class Interactor(ABC):
    """Base class for wavefront-element interactions."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize interactor.

        Args:
            id: Unique identifier
            parameters: Configuration dictionary
        """
        self.id = id
        self.parameters = parameters or {}

    @abstractmethod
    def apply_interaction(self, wavefront: Wavefront, element: Element) -> Wavefront:
        """Apply element effects to wavefront.

        Args:
            wavefront: Wavefront to modify
            element: Optical element
        """
        pass


class ThinObjectInteractor(Interactor):
    """Thin optical element interaction handler."""

    def compute_amplitude_attenuation(self, wavefront: Wavefront, element: Element) -> np.ndarray:
        """Calculate amplitude attenuation.

        Args:
            wavefront: Wavefront to modify
            element: Optical element

        Returns:
            Attenuation factors
        """
        beta = xraylib.Refractive_Index_Im(
            element.material, 
            wavefront.energy, 
            element.density
        )
        return np.exp(
            -wavefront.wavenumber * 
            beta * 
            element.thickness * 
            element.pattern
        )

    def compute_phase_shift(self, wavefront: Wavefront, element: Element) -> np.ndarray:
        """Calculate phase shift.

        Args:
            wavefront: Wavefront to modify
            element: Optical element

        Returns:
            Phase shifts
        """
        delta = 1 - xraylib.Refractive_Index_Re(
            element.material, 
            wavefront.energy, 
            element.density
        )
        return -wavefront.wavenumber * delta * element.thickness * element.pattern
        
    def apply_interaction(self, wavefront: Wavefront, element: Element) -> Wavefront:
        """Apply amplitude and phase modifications.

        Args:
            element: Optical element
        """
        attenuation = self.compute_amplitude_attenuation(wavefront, element)
        phase_shift = self.compute_phase_shift(wavefront, element)

        wavefront.amplitude *= attenuation
        wavefront.phase += phase_shift
        return wavefront


class ThickObjectInteractor(Interactor):
    """Thick optical element interaction handler.
    
    Future implementation will include:
    - Multi-slice propagation
    - Internal structure effects
    - Volume interactions
    """

    def compute_amplitude(self, wavefront: Wavefront, element: Element) -> np.ndarray:
        """Calculate multi-slice amplitude (not implemented).

        Args:
            wavefront: Wavefront to modify
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Multi-slice propagation not implemented")

    def compute_phase_shift(self, wavefront: Wavefront, element: Element) -> np.ndarray:
        """Calculate thick object phase shift (not implemented).

        Args:
            wavefront: Wavefront to modify
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Phase calculation not implemented")

    def apply_interaction(self, wavefront: Wavefront, element: Element) -> Wavefront:
        """Apply thick object effects (not implemented).

        Args:
            element: Optical element

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError("Thick object interaction not implemented")
