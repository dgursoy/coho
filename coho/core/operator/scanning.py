"""Classes for geometric transformations."""

from scipy.ndimage import rotate, shift
import numpy as np
from typing import Tuple

# Local imports
from . import Operator
from ..component import Component

__all__ = [
    'Rotate', 
    'Translate',
    'Scan',
    'ScanParameter',
]

from dataclasses import dataclass
from typing import List

@dataclass
class ScanParameter:
    rotation: List[float] = None
    translation: Tuple[List[float], List[float]] = None

class Scan(Operator):
    """Scan a component."""
    
    def apply(self, component: Component, parameter: ScanParameter) -> Component:
        """Scan component."""
        pass

    def adjoint(self, component: Component, parameter: ScanParameter) -> Component:
        """Adjoint scan."""
        pass

class Rotate(Operator):
    """Rotate a component."""
    
    def apply(self, component: Component, parameter: ScanParameter) -> Component:
        """Rotate component."""
        # Determine which form to use
        use_complex = hasattr(component, 'complexform') and component.complexform is not None
        input_form = component.complexform if use_complex else component.form
        angles = np.atleast_1d(parameter.rotation)
        output = np.zeros((len(angles), *input_form.shape[1:]), dtype=input_form.dtype)
        
        for i, angle in enumerate(angles):
            output[i] = rotate(input_form[0], angle=angle, axes=(0, 1), reshape=False, order=1) 
            
        if use_complex:
            component.complexform = output
        else:
            component.form = output
        return component

    def adjoint(self, component: Component, parameter: ScanParameter) -> Component:
        """Adjoint rotation."""
        # Determine which form to use
        use_complex = hasattr(component, 'complexform') and component.complexform is not None
        input_form = component.complexform if use_complex else component.form
        
        # Rotate each slice by negative angle
        angles = np.atleast_1d(parameter.rotation)
        output = np.zeros_like(input_form)
        
        for i, angle in enumerate(angles):
            output[i] = rotate(input_form[i], angle=-angle, axes=(0, 1), reshape=False, order=1)
            
        # Average all rotated slices
        if use_complex:
            component.complexform = np.mean(output, axis=0, keepdims=True)
        else:
            component.form = np.mean(output, axis=0, keepdims=True)
        return component

class Translate(Operator):
    """Translate a component."""

    def __init__(self, order: int = 1):
        self.order = order
    
    def apply(self, component: Component, parameter: ScanParameter) -> Component:
        """Translate component."""
        # Determine which form to use
        use_complex = hasattr(component, 'complexform') and component.complexform is not None
        input_form = component.complexform if use_complex else component.form
        
        # Apply translations
        shifts = np.atleast_2d(parameter.translation)
        output = np.zeros((len(shifts), *input_form.shape[1:]), dtype=input_form.dtype)
        
        for i, (dx, dy) in enumerate(shifts):
            output[i] = shift(input_form[0], shift=[dx, dy], order=self.order)
            
        # Update the appropriate form
        if use_complex:
            component.complexform = output
        else:
            component.form = output
        return component

    def adjoint(self, component: Component, parameter: ScanParameter) -> Component:
        """Adjoint translation."""
        use_complex = hasattr(component, 'complexform') and component.complexform is not None
        input_form = component.complexform if use_complex else component.form
        
        shifts = np.atleast_2d(parameter.translation)
        output = np.mean(
            [shift(input_form[i], shift=[-dx, -dy], order=self.order)
             for i, (dx, dy) in enumerate(shifts)],
            axis=0, keepdims=True
        )
        
        if use_complex:
            component.complexform = output
        else:
            component.form = output
        return component
    