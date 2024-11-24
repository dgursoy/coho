# config/models/experiment.py

"""Experiment configuration models.

This module defines models for experiment workflow and component management,
including step sequencing and component relationships.

Classes:
    WorkflowStep: Single step in experiment workflow
    ExperimentProperties: Experiment component and workflow definitions
    ExperimentConfig: Complete experiment configuration
"""

from typing import List, Optional
from pydantic import BaseModel, Field

__all__ = [
    'ExperimentProperties', 'Experiment', 
    'ExperimentConfig'
]

class SweepParameter(BaseModel):
    path: str
    range: List[float]

class ComponentSweep(BaseModel):
    component: str
    parameters: List[SweepParameter]

class ScanConfig(BaseModel):
    sweeps: List[ComponentSweep] = Field(default_factory=list)
    order: List[int] = Field(default_factory=list)

class ExperimentProperties(BaseModel):
    components: List[str] = Field(default_factory=list)
    scan: Optional[ScanConfig] = None

class Experiment(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = Field(default_factory=ExperimentProperties)

class ExperimentConfig(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = ExperimentProperties()
