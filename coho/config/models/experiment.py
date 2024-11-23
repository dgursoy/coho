# config/models/experiment.py

"""Experiment configuration models.

This module defines models for experiment workflow and component management,
including step sequencing and component relationships.

Classes:
    WorkflowStep: Single step in experiment workflow
    ExperimentProperties: Experiment component and workflow definitions
    ExperimentConfig: Complete experiment configuration
"""

from typing import List, Dict
from pydantic import BaseModel, Field

__all__ = [
    'ExperimentProperties', 'Experiment', 
    'ExperimentConfig'
]

class NestedRange(BaseModel):
    start: float
    end: float
    step: float

class ExperimentScans(BaseModel):
    targets: List[str] = Field(default_factory=list)
    sweeps: Dict[str, NestedRange] = Field(default_factory=dict)
    order: List[str] = Field(default_factory=list)

class ExperimentProperties(BaseModel):
    components: List[str] = []
    scans: ExperimentScans = ExperimentScans()

class Experiment(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = ExperimentProperties()

class ExperimentConfig(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = ExperimentProperties()
