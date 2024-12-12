"""Common decorators for the coho package."""

# Standard imports
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING
import torch

# Local imports
if TYPE_CHECKING:
    from ..wave import Wave

def requires_cached_tensors(func):
    """Ensure cached tensors are available before executing function."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Graph cache check
        if hasattr(self, 'cache'):
            if not self.cache:
                raise RuntimeError(f"Empty cache in {self.__class__.__name__}")
            return func(self, *args, **kwargs)
            
        # Operator args check
        if not args and not kwargs:
            raise RuntimeError(f"No tensors provided to {self.__class__.__name__}.{func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

def as_tensor(*parameter_names: str, dtype: torch.dtype = torch.float64) -> Callable:
    """Convert specified parameters to torch tensors if they aren't already."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Get device from first arg (wave) or fallback to cpu
            device = getattr(args[0], 'device', 'cpu') if args else 'cpu'
            
            # Convert positional args after the first one (wave)
            new_args = list(args)
            for i, (param_name, arg) in enumerate(zip(parameter_names, args[1:]), 1):
                if not isinstance(arg, torch.Tensor):
                    new_args[i] = torch.as_tensor(arg, dtype=dtype, device=device)
            
            # Convert kwargs
            for param_name in parameter_names:
                if param_name in kwargs and not isinstance(kwargs[param_name], torch.Tensor):
                    kwargs[param_name] = torch.as_tensor(kwargs[param_name], dtype=dtype, device=device)
            
            return func(self, *new_args, **kwargs)
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

def requires_unstacked(func):
    """Ensure wave has stack dimension of 1."""
    @wraps(func)
    def wrapper(self, wave: 'Wave', *args, **kwargs):
        if wave.form.shape[0] != 1:
            raise ValueError(f"Wave must be unstacked (got stack size {wave.form.shape[0]})")
        return func(self, wave, *args, **kwargs)
    return wrapper

def requires_stacked(func):
    """Ensure wave has stack dimension larger than 1."""
    @wraps(func)
    def wrapper(self, wave: 'Wave', *args, **kwargs):
        if wave.form.shape[0] <= 1:
            raise ValueError(f"Wave must be stacked (got stack size {wave.form.shape[0]})")
        return func(self, wave, *args, **kwargs)
    return wrapper