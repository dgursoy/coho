from pydantic import BaseModel

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
