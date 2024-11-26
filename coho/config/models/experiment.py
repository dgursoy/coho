from typing import Dict, List, Union, Any
from pydantic import BaseModel, Field

class ScanSweeps(BaseModel):
    sweeps: Dict[str, List[Union[int, float]]] = Field(default_factory=dict)
    order: List[int] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    scan: ScanSweeps = Field(default_factory=ScanSweeps) 