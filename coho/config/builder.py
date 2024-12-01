"""Configuration object builder."""

from typing import Dict, Any
from .models import ComponentsConfig, Components, ComponentBase, Physical, Profile

def build_config(config: Dict[str, Any]) -> ComponentsConfig:
    """Build configuration object."""
    components_data = config.get('components', {})
    
    components = Components(
        wavefront=_build_component(components_data.get('wavefront')) if 'wavefront' in components_data else None,
        optic=_build_component(components_data.get('optic')) if 'optic' in components_data else None,
        sample=_build_component(components_data.get('sample')) if 'sample' in components_data else None,
        detector=_build_component(components_data.get('detector')) if 'detector' in components_data else None
    )
    
    return ComponentsConfig(components=components)

def _build_component(data: Dict[str, Any]) -> ComponentBase:
    """Build a component from dictionary data."""
    if not data:
        return None
        
    return ComponentBase(
        name=data.get('name', ''),
        physical=Physical(**data.get('physical', {})) if 'physical' in data else None,
        profile=Profile(**data.get('profile', {})) if 'profile' in data else None
    )
