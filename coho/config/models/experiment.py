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
from pydantic import BaseModel

class ExperimentProperties(BaseModel):
    components: List[str]

class ExperimentConfig(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = ExperimentProperties(components=[])
