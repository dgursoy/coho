"""Graph module."""

# Standard imports
from typing import List, Set, Dict, Union, Type, Any
from dataclasses import dataclass
from torch import nn, Tensor

# Local imports
from .wave import Wave

__all__ = [
    'Node',
    'Pipeline'
]


@dataclass
class Node:
    """Configuration for a pipeline node.
    
    Args:
        operator: Wave operator class (Propagate, Modulate, etc)
        inputs: List of input tensor/wave names for routing
        outputs: Output tensor/wave names for routing
        params: Optional parameters for operator initialization
    
    Example:
        NodeConfig(
            operator=Propagate,
            inputs=['wave', 'distance'],
            outputs='propagated',
            params={'use_custom_grads': True}
        )
    """
    operator: Type[nn.Module]
    inputs: List[str]
    outputs: Union[str, List[str]]
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Convert single output to list."""
        if isinstance(self.outputs, str):
            self.outputs = [self.outputs]


class Pipeline(nn.Module):
    """DAG-structured pipeline of wave operators.
    
    Example:
        config = [
            NodeConfig(Propagate, ['wave', 'distance'], 'prop1'),
            NodeConfig(Modulate, ['prop1', 'reference'], 'mod1'),
            NodeConfig(Detect, ['mod1'], 'amplitude')
        ]
        pipeline = Pipeline(config)
        result = pipeline({'wave': wave, 'distance': d, 'reference': ref})
    """
    
    def __init__(self, config: List[Node]):
        super().__init__()
        self.ops = nn.ModuleList()
        self.op_inputs: List[List[str]] = []
        self.op_outputs: List[List[str]] = []
        self.final_outputs: Set[str] = set()
        
        # Build the graph
        for node in config:
            params = node.params or {}
            op = node.operator(**params)
            self.ops.append(op)
            self.op_inputs.append(node.inputs)
            self.op_outputs.append(node.outputs)
            self.final_outputs.update(node.outputs)

    def forward(self, inputs: Dict[str, Union[Tensor, Wave]]) -> Dict[str, Union[Tensor, Wave]]:
        """Forward pass through the pipeline."""
        # Store intermediate values
        self.cache = dict(inputs)
        
        # Forward pass
        for i, op in enumerate(self.ops):
            args = [self.cache[name] for name in self.op_inputs[i]]
            outputs = op(*args)
            
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
                
            for name, out in zip(self.op_outputs[i], outputs):
                self.cache[name] = out
        
        return {name: self.cache[name] for name in self.final_outputs}
