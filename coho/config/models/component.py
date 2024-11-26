"""Configuration models for components."""

from typing import Optional, Dict
from pydantic import BaseModel, Field, RootModel

class Position(BaseModel):
    x: float = Field(0.0)
    y: float = Field(0.0)

class Geometry(BaseModel):
    distance: float = Field(0.0)
    rotation: float = Field(0.0)
    position: Position = Field(default_factory=Position)

class Grid(BaseModel):
    size: int = Field(512)
    spacing: float = Field(0.0001)

# Physical models
class WavefrontPhysical(BaseModel):
    amplitude: float = Field(1.0)
    phase: float = Field(0.0)
    energy: float = Field(10.0)

class OpticPhysical(BaseModel):
    formula: str = Field("Au")
    density: float = Field(19.3)
    thickness: float = Field(0.0001)

class SamplePhysical(BaseModel):
    formula: str = Field("C5H8O2")
    density: float = Field(1.18)
    thickness: float = Field(0.001)

class DetectorPhysical(BaseModel):
    formula: str = Field("Si")
    density: float = Field(2.33)
    thickness: float = Field(0.001)

# Profile models
class WavefrontProfile(BaseModel):
    sigma: Optional[float] = Field(32.0)
    file_path: Optional[str] = None

class OpticProfile(BaseModel):
    sigma: Optional[float] = None
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    bit_size: Optional[int] = None
    seed: Optional[int] = None
    file_path: Optional[str] = None

class SampleProfile(BaseModel):
    file_path: Optional[str] = None

class DetectorProfile(BaseModel):
    pass

# Base component
class ComponentBase(BaseModel):
    model: str
    physical: WavefrontPhysical | OpticPhysical | SamplePhysical | DetectorPhysical = None
    profile: WavefrontProfile | OpticProfile | SampleProfile | DetectorProfile = None
    grid: Grid = Field(default_factory=Grid)
    geometry: Geometry = Field(default_factory=Geometry)

class ComponentsConfig(RootModel[Dict[str, ComponentBase]]):
    pass