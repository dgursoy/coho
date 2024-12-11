"""Decorators for operator validation."""

# Standard imports
from functools import wraps
import numpy as np
from typing import Callable, Any

# Local imports
from ..component import Wave

def validate_form(func: Callable) -> Callable:
    """Validate wave has form defined."""
    @wraps(func)
    def wrapper(self, wave: Wave, *args, **kwargs) -> Any:
        if wave.form is None:
            raise ValueError("Wave must have form defined")
        return func(self, wave, *args, **kwargs)
    return wrapper

def validate_spacing(func: Callable) -> Callable:
    """Validate wave has spacing defined."""
    @wraps(func)
    def wrapper(self, wave: Wave, *args, **kwargs) -> Any:
        if wave.spacing is None:
            raise ValueError("Wave must have spacing defined")
        return func(self, wave, *args, **kwargs)
    return wrapper

def validate_energy(func: Callable) -> Callable:
    """Validate wave has energy defined."""
    @wraps(func)
    def wrapper(self, wave: Wave, *args, **kwargs) -> Any:
        if wave.energy is None:
            raise ValueError("Wave must have energy defined")
        return func(self, wave, *args, **kwargs)
    return wrapper

def validate_position(func: Callable) -> Callable:
    """Validate wave has position defined."""
    @wraps(func)
    def wrapper(self, wave: Wave, *args, **kwargs) -> Any:
        if wave.position is None:
            raise ValueError("Wave must have position defined")
        return func(self, wave, *args, **kwargs)
    return wrapper

def validate_matching_position(func: Callable) -> Callable:
    """Validate that wave positions match."""
    @wraps(func)
    def wrapper(self, reference: Wave, modulator: Wave, *args, **kwargs) -> Any:
        print (reference.position)
        print (modulator.position)
        if any(w.position is None for w in [reference, modulator]):
            raise ValueError("Both waves must have positions defined")
        pos1, pos2 = np.atleast_1d(reference.position), np.atleast_1d(modulator.position)
        if not (np.allclose(pos1, pos2) or np.allclose(pos1, pos2[0]) or np.allclose(pos1[0], pos2)):
            raise ValueError(f"Wave positions do not match: {reference.position} and {modulator.position}")
        return func(self, reference, modulator, *args, **kwargs)
    return wrapper

def validate_matching_energy(func: Callable) -> Callable:
    """Validate that waves have matching energy."""
    @wraps(func)
    def wrapper(self, reference: Wave, modulator: Wave, *args, **kwargs) -> Any:
        if reference.energy != modulator.energy:
            raise ValueError(f"Wave energies do not match: {reference.energy} and {modulator.energy}")
        return func(self, reference, modulator, *args, **kwargs)
    return wrapper

def validate_matching_spacing(func: Callable) -> Callable:
    """Validate that waves have matching spacing."""
    @wraps(func)
    def wrapper(self, reference: Wave, modulator: Wave, *args, **kwargs) -> Any:
        if any(w.spacing is None for w in [reference, modulator]):
            raise ValueError("Both waves must have spacing defined")
        if reference.spacing != modulator.spacing:
            raise ValueError(f"Wave spacings do not match: {reference.spacing} and {modulator.spacing}")
        return func(self, reference, modulator, *args, **kwargs)
    return wrapper

def validate_matching_dimensions(func: Callable) -> Callable:
    """Validate waves have compatible dimensions."""
    @wraps(func)
    def wrapper(self, reference: Wave, modulator: Wave, *args, **kwargs) -> Any:
        if reference.form.ndim != modulator.form.ndim:
            raise ValueError(f"Wave dimensions do not match: {reference.form.ndim} != {modulator.form.ndim}")
        return func(self, reference, modulator, *args, **kwargs)
    return wrapper