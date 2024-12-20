
from abc import ABC, abstractmethod
from typing import Any

__all__ = ['Operator']

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
