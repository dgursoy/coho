# core/simulation/optic.py

"""
Optic classes for wavefront manipulation.

This module provides optics that modify wavefronts through transmission
and phase effects based on material properties and geometric patterns.

Classes:
    Optic: Abstract base class for optics
    CodedApertureOptic: Binary coded aperture with random patterns
    SlitApertureOptic: Rectangular slit aperture
    CircleApertureOptic: Circular aperture
    CustomProfileOptic: Optic with arbitrary transmission profile
"""

from typing import Dict, Any
import numpy as np
from .element import Element, PATTERN_PARAMS

# Optic-specific parameters
CODED_PARAMS: Dict[str, Any] = {"BIT_SIZE": 8}
SLIT_PARAMS: Dict[str, Any] = {"WIDTH": 256, "HEIGHT": 256}
CIRCLE_PARAMS: Dict[str, Any] = {"RADIUS": 128}


class Optic(Element):
    """Base class for optical elements."""
    pass


class CodedApertureOptic(Optic):
    """Binary coded aperture pattern."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate random binary pattern.

        Args:
            parameters: Pattern settings
                bit_size: Pattern bit size
                resolution: Grid size
                rotation: Rotation angle
                seed: Random seed

        Returns:
            Binary pattern array
        """
        # Get parameters
        bit_size = parameters.get("bit_size", CODED_PARAMS["BIT_SIZE"])
        resolution = parameters.get("resolution", PATTERN_PARAMS["RESOLUTION"])
        rotation = parameters.get("rotation", PATTERN_PARAMS["ROTATION"])
        seed = parameters.get("seed")

        # Generate pattern
        if seed is not None:
            np.random.seed(seed)
        num_bits = resolution // bit_size
        bits = np.random.choice([0, 1], size=(num_bits, num_bits))
        pattern = np.kron(bits, np.ones((bit_size, bit_size)))
        pattern = pattern[:resolution, :resolution]

        return self.apply_rotation(pattern.astype(float), rotation)


class SlitApertureOptic(Optic):
    """Rectangular slit aperture."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate rectangular slit.

        Args:
            parameters: Pattern settings
                width: Slit width
                height: Slit height
                resolution: Grid size
                rotation: Rotation angle

        Returns:
            Slit pattern array
        """
        # Get parameters
        width = parameters.get("width", SLIT_PARAMS["WIDTH"])
        height = parameters.get("height", SLIT_PARAMS["HEIGHT"])
        resolution = parameters.get("resolution", PATTERN_PARAMS["RESOLUTION"])
        rotation = parameters.get("rotation", PATTERN_PARAMS["ROTATION"])

        # Generate pattern
        pattern = np.zeros((resolution, resolution))
        center = resolution // 2
        pattern[center - height//2:center + height//2,
                center - width//2:center + width//2] = 1

        return self.apply_rotation(pattern, rotation)


class CircleApertureOptic(Optic):
    """Circular aperture."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate circular aperture.

        Args:
            parameters: Pattern settings
                radius: Circle radius
                resolution: Grid size
                rotation: Rotation angle

        Returns:
            Circle pattern array
        """
        # Get parameters
        radius = parameters.get("radius", CIRCLE_PARAMS["RADIUS"])
        resolution = parameters.get("resolution", PATTERN_PARAMS["RESOLUTION"])
        rotation = parameters.get("rotation", PATTERN_PARAMS["ROTATION"])

        # Generate pattern
        y, x = np.ogrid[:resolution, :resolution]
        center = resolution // 2
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        pattern = np.zeros((resolution, resolution))
        pattern[mask] = 1

        return self.apply_rotation(pattern, rotation)


class CustomProfileOptic(Optic):
    """Custom transmission profile."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Load custom pattern.

        Args:
            parameters: Pattern settings
                custom_profile: Array or file path
                rotation: Rotation angle

        Returns:
            Custom pattern array

        Raises:
            KeyError: Missing profile
            FileNotFoundError: File not found
            ValueError: Invalid file
        """
        profile = parameters.get("custom_profile")
        rotation = parameters.get("rotation", PATTERN_PARAMS["ROTATION"])

        if isinstance(profile, np.ndarray):
            pattern = profile
        elif isinstance(profile, str):
            try:
                pattern = np.load(profile)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile not found: {profile}")
            except ValueError:
                raise ValueError(f"Invalid profile file: {profile}")
        else:
            raise KeyError("custom_profile required (array or path)")

        pattern = pattern / np.max(pattern)
        return self.apply_rotation(pattern, rotation)
