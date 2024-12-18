"""Resource utilities for loading data."""

import jax.numpy as jnp
from pathlib import Path
from typing import Union

def load_image(name: str) -> jnp.ndarray:
    """Load an image from the resources/images directory.
    
    Args:
        name: Name of the image file (without extension)
        
    Returns:
        JAX array containing the image data
    
    Example:
        >>> checkerboard = load_image('checkerboard')
    """
    image_path = Path('coho/resources/images') / f"{name}.npz"
    if not image_path.exists():
        raise FileNotFoundError(f"Image {name} not found at {image_path}")
    
    # Load npz and convert to jax array
    with jnp.load(image_path) as data:
        return jnp.array(data['data'])
    

def load_form(name1: str, name2: str) -> jnp.ndarray:
    """Load two images from the resources/images directory and
    combine them into a complex-valued wave form."""
    form1_path = Path('coho/resources/images') / f"{name1}.npz"
    form2_path = Path('coho/resources/images') / f"{name2}.npz"
    if not form1_path.exists() or not form2_path.exists():
        raise FileNotFoundError(f"Images {name1} and {name2} not found at {form1_path} and {form2_path}")
    with jnp.load(form1_path) as data1, jnp.load(form2_path) as data2:
        return jnp.array(data1['data']) * jnp.exp(1j * jnp.array(data2['data']))
