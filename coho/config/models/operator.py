from typing import Optional, Dict, Any
from pydantic import BaseModel

class Interactor(BaseModel):
    id: Optional[str] = None
    model: str
    properties: Dict[str, Any] = {}

class Propagator(BaseModel):
    id: Optional[str] = None
    model: str
    properties: Dict[str, Any] = {}

class OperatorConfig(BaseModel):
    interactor: Interactor
    propagator: Propagator
