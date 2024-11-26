"""Configuration models package."""

from .component import ComponentsConfig
from .experiment import ExperimentConfig
from pydantic import BaseModel, Field

__all__ = [
    'RootConfig', 
    'ComponentsConfig', 
    'ExperimentConfig'
]

# Root config
class RootConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)

