"""Configuration models for components."""

from dataclasses import dataclass
from typing import Union, Optional, TypeAlias
import numpy as np
from numbers import Number

# Type aliases for fields that can be batched
BatchableFloat: TypeAlias = Union[float, np.ndarray]
BatchableInt: TypeAlias = Union[int, np.ndarray]
BatchableStr: TypeAlias = Union[str, np.ndarray]
Batchable: TypeAlias = Union[Number, str, np.ndarray]

# Physical models
@dataclass
class Physical:
    energy: Optional[BatchableFloat] = 10.0
    spacing: Optional[BatchableFloat] = 0.0001
    formula: Optional[BatchableStr] = "Au"
    density: Optional[BatchableFloat] = 19.3
    thickness: Optional[BatchableFloat] = 0.0001
    position: Optional[BatchableFloat] = 1.0

# Generic Profile model
@dataclass
class Profile:
    model: str
    size: BatchableInt = 512
    sigma: Optional[BatchableFloat] = 32
    bit_size: Optional[BatchableInt] = 64
    seed: Optional[BatchableInt] = 0
    path: Optional[BatchableStr] = None

# Base component
@dataclass
class ComponentBase:
    name: str
    physical: Optional[Physical] = None
    profile: Optional[Profile] = None

@dataclass
class Components:
    wavefront: Optional[ComponentBase] = None
    optic: Optional[ComponentBase] = None
    sample: Optional[ComponentBase] = None
    detector: Optional[ComponentBase] = None 

@dataclass
class SolverProperties:
    method: str
    iterations: BatchableInt = 100
    step_size: BatchableFloat = 0.1
    initial_guess: Optional[BatchableFloat] = 0.0

@dataclass
class ObjectiveProperties:
    weight: BatchableFloat = 1.0
    regularization: BatchableFloat = 1e-4

@dataclass
class ComponentsConfig:
    components: Components
    solver: Optional[SolverProperties] = None
    objective: Optional[ObjectiveProperties] = None