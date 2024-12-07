"""Common decorators for the coho package."""

from functools import wraps
from typing import Callable, Any, TYPE_CHECKING, Union
import torch

if TYPE_CHECKING:
    from ..component import Wave

def as_tensor(*parameter_names: str, dtype: torch.dtype = torch.float64) -> Callable:
    """Convert specified parameters to torch tensors if they aren't already."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Get device from form attribute or fallback to cpu
            device = 'cpu'
            if hasattr(self, 'form'):
                device = self.form.device
                
            # Convert each named parameter in kwargs
            for param_name in parameter_names:
                if param_name in kwargs and not isinstance(kwargs[param_name], torch.Tensor):
                    kwargs[param_name] = torch.as_tensor(kwargs[param_name], dtype=dtype, device=device)
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def requires_attrs(*attributes: str, wave_param: str = 'wave') -> Callable:
    """Validate wave has required attributes defined."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Get wave from kwargs or first positional arg
            wave = kwargs.get(wave_param, args[0])
            
            # Check attributes
            for attr in attributes:
                if getattr(wave, attr) is None:
                    raise ValueError(f"Wave must have {attr} defined")
                    
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def requires_matching(*attributes: str) -> Callable:
    """Validate that wave attributes match between two waves."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, reference: 'Wave', modulator: 'Wave', *args, **kwargs) -> Any:
            for attr in attributes:
                attr1 = getattr(reference, attr)
                attr2 = getattr(modulator, attr)
                
                if any(a is None for a in [attr1, attr2]):
                    raise ValueError(f"Both waves must have {attr} defined")
                
                if isinstance(attr1, torch.Tensor) and isinstance(attr2, torch.Tensor):
                    if not (torch.allclose(attr1, attr2) or 
                           torch.allclose(attr1, attr2[0]) or 
                           torch.allclose(attr1[0], attr2)):
                        raise ValueError(f"Wave {attr}s do not match: {attr1} and {attr2}")
                elif attr == 'form':
                    if attr1.ndim != attr2.ndim:
                        raise ValueError(f"Wave dimensions do not match: {attr1.ndim} != {attr2.ndim}")
                elif attr1 != attr2:
                    raise ValueError(f"Wave {attr}s do not match: {attr1} and {attr2}")
                    
            return func(self, reference, modulator, *args, **kwargs)
        return wrapper
    return decorator