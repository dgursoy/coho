# core/simulation/sample.py

"""
Sample classes for wavefront manipulation.

This module provides samples that modify wavefronts through transmission
and phase effects based on material properties and geometric profiles.

Classes:
    Sample: Base class for samples
    CustomProfileSample: Sample with arbitrary transmission profile
    BatchSample: Container for multiple samples with varying parameters
"""

import numpy as np
from .element import Element

__all__ = [
    'CustomProfileSample',
]

class Sample(Element):
    """Base class for samples."""
    pass

class CustomProfileSample(Element):
    """Custom transmission profile sample."""

    def _generate_profile(self) -> np.ndarray:
        """Load custom profile.
        """
        # Get parameters
        file_path = self.properties.profile.file_path

        # Load profile
        profile = np.load(file_path)

        # Normalize profile
        profile = profile / np.max(profile)
        return profile
