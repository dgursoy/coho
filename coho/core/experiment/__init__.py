# core/experiment/__init__.py

"""Core experiment templates for optical systems.

This module provides base classes for high-level experiment configurations
that combine simulation and optimization components.

Components:
    Experiments:
        holography: Holographic imaging experiments
        tomography: Tomographic reconstruction experiments
"""

from .holography import Holography, Tomography

__all__ = [
    'Holography',
    'Tomography'
]
