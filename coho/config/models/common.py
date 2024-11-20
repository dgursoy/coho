# config/models/common.py

"""Common configuration models.

This module provides basic shared data structures used across
different components of the optical simulation system.

Classes:
    Position: 3D spatial position (x, y, z)
    Geometry: Position and rotation in space
    Grid: Spatial discretization parameters
"""

from pydantic import BaseModel

__all__ = [
    'Position', 'Geometry', 'Grid'
]

class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class Geometry(BaseModel):
    position: Position = Position()
    rotation: float = 0.0

class Grid(BaseModel):
    size: int = 512
    spacing: float = 0.001
