# Standard imports
from typing import Any

# Local imports
from coho.core.operator.base import Operator

class Pipeline(Operator):
    """Pipeline of operators."""
    
    def __init__(self, operators: list[tuple[Operator, dict[str, Any]]]):
        """Initialize pipeline with operators and their arguments."""
        self.operators = operators
    
    def _process(self, wave: Any, forward: bool = True) -> Any:
        """Process wave through pipeline.
        
        Args:
            wave: Input wave
            forward: If True, apply forward pipeline, else adjoint
        """
        result = wave
        ops = self.operators if forward else reversed(self.operators)
        
        for op, kwargs in ops:
            operation = op.apply if forward else op.adjoint
            result = operation(result, **kwargs)
            
        return result

    def apply(self, wave: Any) -> Any:
        """Forward pipeline."""
        return self._process(wave, forward=True)
    
    def adjoint(self, wave: Any) -> Any:
        """Adjoint pipeline."""
        return self._process(wave, forward=False)