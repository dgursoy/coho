# Standard imports
from abc import ABC, abstractmethod
from typing import Any, Union, List, TypeAlias, Dict
import torch

# Type aliases
TensorLike: TypeAlias = Union[float, List[float], torch.Tensor]
TensorDict: TypeAlias = Dict[str, TensorLike]

__all__ = ['Operator', 'TensorLike', 'TensorDict']

class Operator(ABC):
    """Base class for operators."""
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Forward operator."""
        pass

    @abstractmethod
    def adjoint(self, *args, **kwargs) -> Any:
        """Adjoint operator."""
        pass

