# config/models/operator.py

"""Operator configuration models.

This module defines models for wave propagation and interaction operators
used in optical simulations.

Classes:
    Interactor: Wave-matter interaction operator configuration
    Propagator: Wave propagation operator configuration
    OperatorConfig: Complete operator configuration combining both components
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel

__all__ = [
    'Interactor', 'InteractorProperties', 
    'Propagator', 'PropagatorProperties',
    'OperatorConfig'
]

class InteractorProperties(BaseModel):
    pass

class Interactor(BaseModel):
    id: Optional[str] = None
    model: str
    properties: InteractorProperties = InteractorProperties()

class PropagatorProperties(BaseModel):
    pass

class Propagator(BaseModel):
    id: Optional[str] = None
    model: str
    properties: PropagatorProperties = PropagatorProperties()

class OperatorConfig(BaseModel):
    interactor: Interactor
    propagator: Propagator
