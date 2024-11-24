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
from ..experiment.batcher import Batch

__all__ = [
    'CustomProfileSample',
    'BatchSample'
]

class Sample(Element):
    """Base class for samples."""
    pass

class CustomProfileSample(Element):
    """Custom transmission profile sample."""

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


class BatchSample(Batch):
    """Container for multiple samples with varying parameters."""
    
    @property
    def profiles(self):
        """Get array of all profiles."""
        return np.array([s.profile for s in self.components])
