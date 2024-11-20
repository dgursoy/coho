# config/models/simulation.py

"""Simulation configuration models.

This module defines models for optical simulation components including
wavefronts, optical elements, samples, and detectors.

Classes:
    WavefrontPhysical: Physical properties of light source
    WavefrontProfile: Spatial profile of light source
    WavefrontProperties: Complete wavefront properties
    Wavefront: Light source configuration
    
    OpticPhysical: Physical properties of optical element
    OpticProfile: Spatial profile of optical element
    OpticProperties: Complete optic properties
    Optic: Optical element configuration
    
    SamplePhysical: Physical properties of sample
    SampleProfile: Spatial profile of sample
    SampleProperties: Complete sample properties
    Sample: Sample configuration
    
    DetectorProperties: Detector properties
    Detector: Detector configuration
    
    SimulationConfig: Complete simulation configuration
"""

from typing import Optional
from pydantic import BaseModel
from .common import Grid, Geometry

__all__ = [
    'WavefrontPhysical', 'WavefrontProfile', 'WavefrontProperties', 'Wavefront',
    'OpticPhysical', 'OpticProfile', 'OpticProperties', 'Optic',
    'SamplePhysical', 'SampleProfile', 'SampleProperties', 'Sample',
    'DetectorProperties', 'Detector', 
    'SimulationConfig'
]

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
