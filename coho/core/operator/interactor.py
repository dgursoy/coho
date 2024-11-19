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
from typing import Optional, Union
import numpy as np
import xraylib
from coho.core.simulation.wavefront import Wavefront
from coho.core.simulation.optic import Optic
from coho.core.simulation.sample import Sample
from coho.config.models import InteractorProperties

__all__ = [
    'ThinObjectInteractor', 'ThickObjectInteractor'
]

class Interactor(ABC):
    """Base class for wavefront-element interactions."""

    def __init__(self, properties: Optional[InteractorProperties]):
        """Initialize interactor."""
        self.properties = properties

    @abstractmethod
    def interact(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Interact element with wavefront."""
        pass

    def forward(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Forward model (alias for interact)."""
        return self.interact(wavefront, element)
    
    def adjoint(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Adjoint model."""
        pass

class ThinObjectInteractor(Interactor):
    """Thin optical element interaction handler."""

    def _compute_refractive_index(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> float:
        """
        Compute the complex refractive index of the optical element.
        """
        return xraylib.Refractive_Index(
            element.properties.physical.formula,
            wavefront.properties.physical.energy,
            element.properties.physical.density
        )
    
    def _compute_modulation(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> float:
        """Compute phase shift."""
        refractive_index = self._compute_refractive_index(wavefront, element)
        return wavefront.wavenumber * (refractive_index - 1) * element.properties.physical.thickness * element.profile
        
    def interact(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Apply amplitude and phase modifications."""
        
        # Compute refractive index
        modulation = self._compute_modulation(wavefront, element)
        
        # Compute complex wavefront
        wavefront.complex_wavefront *= np.exp(1j * modulation)
        return wavefront
    
    def adjoint(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Reverse amplitude and phase modifications."""
        modulation = self._compute_modulation(wavefront, element)

        # The conjugate of the exponential term is applied to reverse the interaction
        wavefront.complex_wavefront *= np.exp(-1j * modulation)
        return wavefront


class ThickObjectInteractor(Interactor):
    """Thick optical element interaction handler."""

    def interact(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Apply thick object effects (not implemented)."""
        raise NotImplementedError("Thick object interaction not implemented")
    
    def adjoint(self, wavefront: Wavefront, element: Union[Optic, Sample]) -> Wavefront:
        """Reverse thick object effects (not implemented)."""
        raise NotImplementedError("Thick object interaction not implemented")

