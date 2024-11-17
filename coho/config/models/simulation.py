from typing import Optional, Dict, Any
from pydantic import BaseModel
from .common import Grid, Geometry

class WavefrontPhysical(BaseModel):
    amplitude: float = 1.0
    phase: float = 0.0
    energy: float = 10.0

class WavefrontProfile(BaseModel):
    sigma: float = 256
    width: float = 256
    height: float = 256

class WavefrontProperties(BaseModel):
    physical: WavefrontPhysical = WavefrontPhysical()
    profile: WavefrontProfile = WavefrontProfile()
    grid: Grid = Grid()
    geometry: Geometry = Geometry()

class Wavefront(BaseModel):
    id: str
    model: str
    properties: WavefrontProperties = WavefrontProperties()

class OpticPhysical(BaseModel):
    formula: str = "Au"
    density: float = 19.3
    thickness: float = 0.0001

class OpticProfile(BaseModel):
    model: str = "binary_random"
    bit_size: int = 64
    seed: int = 0
    width: float = 256
    height: float = 256

class OpticProperties(BaseModel):
    physical: OpticPhysical = OpticPhysical()
    profile: OpticProfile = OpticProfile()
    grid: Grid = Grid()
    geometry: Geometry = Geometry()

class Optic(BaseModel):
    id: str
    model: str
    properties: OpticProperties = OpticProperties()

class SamplePhysical(BaseModel):
    formula: str = "C5H8O2"
    density: float = 1.18
    thickness: float = 0.001

class SampleProfile(BaseModel):
    model: str = "custom_profile"
    file_path: str = "coho/resources/samples/lena.npy"

class SampleProperties(BaseModel):
    physical: SamplePhysical = SamplePhysical()
    profile: SampleProfile = SampleProfile()
    grid: Grid = Grid()
    geometry: Geometry = Geometry()

class Sample(BaseModel):
    id: str
    model: str
    properties: SampleProperties = SampleProperties()

class DetectorProperties(BaseModel):
    grid: Grid = Grid()
    geometry: Geometry = Geometry()

class Detector(BaseModel):
    id: str
    model: str
    properties: DetectorProperties = DetectorProperties()

class SimulationConfig(BaseModel):
    wavefront: Wavefront
    optic: Optional[Optic] = None
    sample: Optional[Sample] = None
    detector: Detector
