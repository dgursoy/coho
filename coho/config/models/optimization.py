# config/models/optimization.py

"""Optimization configuration models.

This module defines models for optimization components used in
parameter fitting and system optimization.

Classes:
    SolverProperties: Optimization solver parameters
    Solver: Optimization solver configuration
    Objective: Optimization objective configuration
    OptimizationConfig: Complete optimization configuration
"""

from typing import Dict, Any
from pydantic import BaseModel

class SolverProperties(BaseModel):
    step_size: float = 0.01
    iterations: int = 1000

class Solver(BaseModel):
    id: str
    model: str
    properties: SolverProperties = SolverProperties()

class Objective(BaseModel):
    id: str
    model: str
    properties: Dict[str, Any] = {}

class OptimizationConfig(BaseModel):
    objective: Objective
    solver: Solver
