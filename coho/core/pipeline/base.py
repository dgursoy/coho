# Standard imports
from typing import Union, TypeAlias, Tuple, Any, List, Dict
import torch

# Local imports
from coho.core.operator.base import Operator

# Type aliases
TensorLike: TypeAlias = Union[float, List[float], torch.Tensor]
TensorDict: TypeAlias = Dict[str, TensorLike]
OperatorSequence: TypeAlias = List[Tuple[Operator, Dict[str, Any]]]

class Pipeline(Operator):
    """Pipeline of operators."""
    
    def __init__(self, operators: OperatorSequence):
        """Initialize pipeline with operators and their arguments."""
        self.operators = operators
    
    def apply(self, wave: Any) -> Any:
        """Apply pipeline in forward direction."""
        result = wave
        for op, kwargs in self.operators:
            result = op.apply(result, **kwargs)
        return result
    
    def adjoint(self, wave: Any) -> Any:
        """Apply pipeline in adjoint direction."""
        result = wave
        for op, kwargs in reversed(self.operators):
            result = op.adjoint(result, **kwargs)
        return result
    
class CompositeOperator(Operator):
    """Composite operator for nested transformations."""
    def __init__(self, operator, *sub_operators):
        self.operator = operator
        self.sub_operators = sub_operators
    
    def apply(self, *inputs, **kwargs):
        """Apply nested operators in forward direction."""
        output = inputs
        for sub_op in self.sub_operators:
            output = (sub_op.apply(*output, **kwargs),)
        return self.operator.apply(*output, **kwargs)
    
    def adjoint(self, *inputs, **kwargs):
        """Apply nested operators in adjoint direction."""
        output = inputs
        output = self.operator.adjoint(*output, **kwargs)
        for sub_op in reversed(self.sub_operators):
            output = sub_op.adjoint(*output, **kwargs)
        return output

class CachedOperator(Operator):
    """Wraps an operator to cache specific inputs."""
    def __init__(self, operator):
        self.operator = operator
        self.cache = {}

    def apply(self, *inputs):
        """Compute or retrieve cached result."""
        key = tuple(inputs)  # Use inputs as cache key
        if key not in self.cache:
            self.cache[key] = self.operator.apply(*inputs)
        return self.cache[key]

    def adjoint(self, *inputs):
        """Clear cache if adjoint computation is needed."""
        # Adjoint assumes inputs from the cache have been used
        return self.operator.adjoint(*inputs)

    def clear_cache(self):
        """Reset the cache for new computations."""
        self.cache.clear()

class FrozenOperator(Operator):
    """Freezes certain inputs of an operator by precomputing them."""
    def __init__(self, operator, *precomputed_inputs):
        self.operator = operator
        self.precomputed_inputs = precomputed_inputs
        self.frozen_output = operator.apply(*precomputed_inputs)

    def apply(self, *remaining_inputs):
        """Use precomputed results with remaining inputs."""
        return self.operator.apply(self.frozen_output, *remaining_inputs)

    def adjoint(self, *remaining_inputs):
        """Compute adjoint, assuming frozen inputs."""
        adj_result = self.operator.adjoint(*remaining_inputs)
        if isinstance(adj_result, tuple):
            # Exclude adjoint contributions of frozen inputs
            return adj_result[len(self.precomputed_inputs):]
        return adj_result

