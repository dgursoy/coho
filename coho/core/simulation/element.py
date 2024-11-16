# core/simulation/element.py

"""Base element class for optical simulation."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import rotate
import numpy as np

PATTERN_PARAMS: Dict[str, Any] = {
    "RESOLUTION": 512,     
    "ROTATION": 0,         
}

class Element(ABC):
    """Base class for all optical elements."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.id = id
        params = parameters or {}
        
        # Physical properties
        self.material = params.get("physical", {}).get("formula")
        self.thickness = params.get("physical", {}).get("thickness")
        self.density = params.get("physical", {}).get("density")
        
        # Generate transmission pattern
        self.pattern = self.generate_pattern(params)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.pattern.shape

    @abstractmethod
    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        pass

    def apply_rotation(self, pattern: np.ndarray, angle: float) -> np.ndarray:
        return rotate(pattern, angle, reshape=False, order=1, mode='nearest')
