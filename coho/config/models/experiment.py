from typing import List, Dict
from pydantic import BaseModel

class WorkflowStep(BaseModel):
    action: str

class ExperimentProperties(BaseModel):
    components: List[str]
    workflow: Dict[str, List[WorkflowStep]]

class ExperimentConfig(BaseModel):
    id: str
    model: str
    properties: ExperimentProperties = ExperimentProperties(components=[], workflow={})
