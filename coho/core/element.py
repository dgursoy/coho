# core/element.py

"""
Optical element classes for wavefront manipulation.

This module provides elements that modify wavefronts through transmission
and phase effects based on material properties and geometric patterns.

Classes:
    Element: Abstract base class for optical elements
    CodedApertureElement: Binary coded aperture with random patterns
    SlitApertureElement: Rectangular slit aperture
    CircleApertureElement: Circular aperture
    CustomProfileElement: Element with arbitrary transmission profile

Methods:
    generate_pattern: Create element-specific patterns
    apply_rotation: Rotate pattern by specified angle

Constants:
    MATERIAL_PARAMS: Default material properties
    PATTERN_PARAMS: Default pattern generation settings
    CODED_PARAMS: Coded aperture defaults
    SLIT_PARAMS: Slit aperture defaults
    CIRCLE_PARAMS: Circle aperture defaults
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import rotate
import numpy as np


# Default parameters
MATERIAL_PARAMS: Dict[str, Any] = {
    "MATERIAL": "Au",      # Element material
    "DENSITY": 19.32,      # g/cm³
    "THICKNESS": 0.01,     # cm
}

PATTERN_PARAMS: Dict[str, Any] = {
    "RESOLUTION": 512,     # pixels
    "ROTATION": 0,         # degrees
}

CODED_PARAMS: Dict[str, Any] = {"BIT_SIZE": 8}
SLIT_PARAMS: Dict[str, Any] = {"WIDTH": 256, "HEIGHT": 256}
CIRCLE_PARAMS: Dict[str, Any] = {"RADIUS": 128}


class Element(ABC):
    """Base class for optical elements."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize element.

        Args:
            id: Unique identifier
            parameters: Configuration dict
        """
        self.id = id
        params = parameters or {}
        
        # Physical properties
        self.material = params.get("material", MATERIAL_PARAMS["MATERIAL"])
        self.thickness = params.get("thickness", MATERIAL_PARAMS["THICKNESS"])
        self.density = params.get("density", MATERIAL_PARAMS["DENSITY"])
        
        # Generate transmission pattern
        self.pattern = self.generate_pattern(params)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get pattern dimensions."""
        return self.pattern.shape

    @abstractmethod
    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate element pattern.

        Args:
            parameters: Pattern settings

        Returns:
            Pattern array
        """
        pass

    def apply_rotation(self, pattern: np.ndarray, angle: float) -> np.ndarray:
        """Rotate pattern.

        Args:
            pattern: Input pattern
            angle: Rotation degrees

        Returns:
            Rotated pattern
        """
        return rotate(pattern, angle, reshape=False, order=1, mode='nearest')


class CodedApertureElement(Element):
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


class SlitApertureElement(Element):
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


class CircleApertureElement(Element):
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


class CustomProfileElement(Element):
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
