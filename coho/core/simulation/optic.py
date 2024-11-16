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
        bit_size = parameters.get("profile", {}).get("bit_size")
        resolution = parameters.get("grid", {}).get("size")
        rotation = parameters.get("geometry", {}).get("rotation")
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
        width = parameters.get("profile", {}).get("width")
        height = parameters.get("profile", {}).get("height")
        resolution = parameters.get("grid", {}).get("size")
        rotation = parameters.get("geometry", {}).get("rotation")

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
        radius = parameters.get("profile", {}).get("radius")
        resolution = parameters.get("grid", {}).get("size")
        rotation = parameters.get("geometry", {}).get("rotation")

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
        file_path = parameters.get("profile", {}).get("file_path")
        rotation = parameters.get("geometry", {}).get("rotation")

        if isinstance(file_path, str):
            try:
                pattern = np.load(file_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile not found: {file_path}")
            except ValueError:
                raise ValueError(f"Invalid profile file: {file_path}")
        else:
            raise KeyError("custom_profile required (path)")

        pattern = pattern / np.max(pattern)
        return self.apply_rotation(pattern, rotation)
