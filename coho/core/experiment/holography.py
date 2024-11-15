# core/experiment/holography.py

from abc import ABC


class Experiment(ABC):
    """Base class for all experiments."""
    pass


class Holography(Experiment):
    """Holographic imaging experiment."""
    pass

class Tomography(Experiment):
    """Tomographic reconstruction experiment."""
    pass
