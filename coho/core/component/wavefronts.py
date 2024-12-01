"""Wavefronts."""

import numpy as np
from typing import Any
from abc import ABC, abstractmethod

# Local imports
from . import Component

__all__ = [
    'UniformWavefront',
    'GaussianWavefront',
    'CustomWavefront',
]

class Wavefront(Component):
    """Base wavefront class."""
    
    def __init__(self, properties: Any):
        """Initialize the wavefront with specified properties."""
        super().__init__(properties)

    @property
    def complexform(self) -> np.ndarray:
        """Complex form with batch dimension."""
        if self._complexform is None:
            self._complexform = np.expand_dims(self.generate_form(), axis=0)
        return self._complexform
    
    @complexform.setter
    def complexform(self, value: np.ndarray):
        """Set the complex form."""
        self._complexform = value

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength derived from energy in keV."""
        return np.divide(1.23984193e-7, self.physical.energy)
    
    @property
    def wavenumber(self) -> np.ndarray:
        """Wavenumber (2π divided by wavelength)."""
        return np.divide(2 * np.pi, self.wavelength)

    @abstractmethod
    def generate_form(self) -> np.ndarray:
        """Generate a wavefront form."""
        pass

class CustomWavefront(Wavefront):
    def generate_form(self) -> np.ndarray:
        """Generate a custom wavefront form."""
        path = self.profile.path
        form = np.load(path)
        return form / np.max(form)
    
class GaussianWavefront(Wavefront):
    def generate_form(self) -> np.ndarray:
        """Generate a Gaussian wavefront form."""
        size = self.profile.size
        sigma = self.profile.sigma
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
    
class UniformWavefront(Wavefront):
    def generate_form(self) -> np.ndarray:
        """Generate a uniform wavefront form."""
        size = self.profile.size
        return np.ones((size, size))
    

