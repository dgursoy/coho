# core/simulation/sample.py

"""
Sample classes for wavefront manipulation.

This module provides samples that modify wavefronts through transmission
and phase effects based on material properties and geometric patterns.

Classes:
    Sample: Base class for samples
    CustomProfileSample: Sample with arbitrary transmission profile
"""

from typing import Dict, Any
import numpy as np
from .element import Element, PATTERN_PARAMS


class Sample(Element):
    """Base class for samples."""
    pass


class CustomProfileSample(Element):
    """Custom transmission profile sample."""

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
