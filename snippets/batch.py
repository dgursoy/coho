from functools import wraps
from typing import List, Any, Callable
from dataclasses import dataclass
import numpy as np

def batch(func: Callable) -> Callable:
    """Modified to accept parameters at runtime instead of decoration time"""
    def set_nested(d: dict, path: str, value: Any):
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    @wraps(func)
    def wrapper(**kwargs) -> List[Any]:
        param_names = list(kwargs.keys())
        value_combinations = zip(*kwargs.values())
        
        results = []
        for values in value_combinations:
            nested_kwargs = {}
            for path, value in zip(param_names, values):
                set_nested(nested_kwargs, path, value)
            results.append(func(**nested_kwargs))
        return results
    return wrapper

@dataclass(frozen=True)
class WavefrontProperties:
    energy: float
    phasor: np.ndarray

@dataclass(frozen=True)
class Wavefront:
    properties: WavefrontProperties

    @batch
    def create_wavefront(**kwargs) -> 'Wavefront':
        return Wavefront(WavefrontProperties(**kwargs['properties']))

# Example usage
if __name__ == "__main__":

    params = {
        'properties.energy': [1.0, 2.0],
        'properties.phasor': [np.ones(3), 2 * np.ones(3)]
    }

    wavefronts = Wavefront.create_wavefront(**params)
    print("Generated wavefronts:")
    for i, wf in enumerate(wavefronts, 1):
        print(f"\nWavefront {i}:")
        print(f"  Energy: {wf.properties.energy}")
        print(f"  Phasor: {wf.properties.phasor}")
