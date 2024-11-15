# core/simulation/element.py

"""Base element class for optical simulation."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import rotate
import numpy as np

# Shared default parameters
MATERIAL_PARAMS: Dict[str, Any] = {
    "MATERIAL": "Au",      
    "DENSITY": 19.32,      
    "THICKNESS": 0.01,     
}

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
        self.material = params.get("material", MATERIAL_PARAMS["MATERIAL"])
        self.thickness = params.get("thickness", MATERIAL_PARAMS["THICKNESS"])
        self.density = params.get("density", MATERIAL_PARAMS["DENSITY"])
        
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
