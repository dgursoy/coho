# core/simulation/optic.py

"""
Optic classes for wavefront manipulation.

This module provides optics that modify wavefronts through transmission
and phase effects based on material properties and geometric profiles.

Classes:
    Optic: Abstract base class for optics
    CodedApertureOptic: Binary coded aperture with random profiles
    SlitApertureOptic: Rectangular slit aperture
    CircleApertureOptic: Circular aperture
    CustomProfileOptic: Optic with arbitrary transmission profile
"""

import numpy as np
from .element import Element

__all__ = [
    'CodedApertureOptic',
    'SlitApertureOptic',
    'CircleApertureOptic',
    'CustomProfileOptic'
]

class Optic(Element):
    """Base class for optical elements."""
    pass

class CodedApertureOptic(Optic):
    """Binary coded aperture profile."""

    def generate_profile(self) -> np.ndarray:
        """Generate random binary profile.
        """
        # Get parameters
        bit_size = self.properties.profile.bit_size
        seed = self.properties.profile.seed

        # Generate profile
        if seed is not None:
            np.random.seed(seed)
        num_bits = self.size // bit_size
        bits = np.random.choice([0, 1], size=(num_bits, num_bits))
        profile = np.kron(bits, np.ones((bit_size, bit_size)))
        profile = profile[:self.size, :self.size]

        return profile


class SlitApertureOptic(Optic):
    """Rectangular slit aperture."""

    def generate_profile(self) -> np.ndarray:
        """Generate rectangular slit.
        """
        # Get parameters
        width = self.properties.profile.width
        height = self.properties.profile.height

        # Generate profile
        profile = np.zeros((self.size, self.size))
        center = self.size // 2
        profile[center - height//2:center + height//2,
                center - width//2:center + width//2] = 1

        return profile


class CircleApertureOptic(Optic):
    """Circular aperture."""

    def generate_profile(self) -> np.ndarray:
        """Generate circular aperture.
        """
        # Get parameters
        radius = self.properties.profile.radius

        # Generate profile
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        profile = np.zeros((self.size, self.size))
        profile[mask] = 1

        return profile


class CustomProfileOptic(Optic):
    """Custom transmission profile."""

    def generate_profile(self) -> np.ndarray:
        """Load custom profile.
        """
        # Get parameters
        file_path = self.properties.profile.file_path

        # Load profile
        profile = np.load(file_path)

        # Normalize profile
        profile = profile / np.max(profile)
        
        return profile
