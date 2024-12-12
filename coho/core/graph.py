"""Graph module."""

# Standard imports
from typing import Tuple, List, Set, Any
from torch import nn
from dataclasses import dataclass
import copy
import torch

# Local imports
from .wave import Wave
from .operator import Operator
from .utils.decorators import requires_cached_tensors

__all__ = [
    'Node',
    'Graph'
]

@dataclass
class Node:
    """Node in a graph.
    
    Args:
        operator: The operator to apply
        inputs: Input names
        outputs: Output names
        save_for_backward: Names of inputs to save for backward pass
                         (like PyTorch's ctx.save_for_backward)
                         If None, all inputs are saved (default).
    """
    operator: nn.Module
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    save_for_backward: Tuple[str, ...] = None

    def __post_init__(self):
        """Ensure inputs and outputs are tuples."""
        if isinstance(self.inputs, str):
            self.inputs = (self.inputs,)
        if isinstance(self.outputs, str):
            self.outputs = (self.outputs,)
        if self.save_for_backward is None:
            self.save_for_backward = self.inputs
        elif isinstance(self.save_for_backward, str):
            self.save_for_backward = (self.save_for_backward,)

class Graph(Operator):
    """Graph of operators that handles DAG-like flows."""
    def __init__(self, debug: bool = False):
        """Initialize Graph."""
        super().__init__()
        self.nodes = []
        self.cache = {}
        self.debug = debug

    def add_node(self, node: Node):
        """Add node to graph."""
        self.nodes.append(node)

    @property
    def cache_keys(self) -> List[str]:
        """Get available keys in the graph cache."""
        return list(self.cache.keys())

    def clear_cache(self):
        """Clear the entire cache."""
        self.cache = {}

    def get_output_nodes(self) -> List[Node]:
        """Get nodes whose outputs aren't used by any other node."""
        all_inputs = set()
        for node in self.nodes:
            all_inputs.update(node.inputs)
        
        return [
            node for node in self.nodes 
            if not any(out in all_inputs for out in node.outputs)
        ]

    @property
    def _node_sets(self) -> Tuple[Set[str], Set[str]]:
        """Get all input and output keys in the graph."""
        outputs = {out for node in self.nodes for out in node.outputs}
        inputs = {inp for node in self.nodes for inp in node.inputs}
        return inputs, outputs

    @property
    def leaf_keys(self) -> Set[str]:
        """Keys that are graph inputs only (sources)."""
        inputs, outputs = self._node_sets
        return inputs - outputs

    @property
    def root_keys(self) -> Set[str]:
        """Keys that are graph outputs only (sinks)."""
        inputs, outputs = self._node_sets
        return outputs - inputs

    @property
    def intermediate_keys(self) -> Set[str]:
        """Keys that are both inputs and outputs."""
        inputs, outputs = self._node_sets
        return inputs & outputs

    def forward(self, **inputs):
        """Forward pass through the graph.
        
        Args:
            **inputs: Named inputs to the graph
            
        Returns:
            Single output or tuple of outputs from final node
        """
        def safe_copy(val):
            """Create a copy if debug mode is on."""
            if not self.debug:
                return val
            return val.clone() if isinstance(val, torch.Tensor) else copy.deepcopy(val)
        
        # Initialize cache with inputs
        self.cache = {k: safe_copy(v) for k, v in inputs.items()}
        
        # Process each node
        for node in self.nodes:
            # Get inputs from cache and copy them
            args = [safe_copy(self.cache[name]) for name in node.inputs]
            
            # Apply operator
            outputs = node.operator.forward(*args)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            
            # Store outputs in cache
            self.cache.update({
                name: safe_copy(out)
                for name, out in zip(node.outputs, outputs)
            })

        return outputs[0] if len(outputs) == 1 else outputs

    @requires_cached_tensors
    def gradient(self, grad_outputs: dict, num_nodes: int = None) -> Any:
        """Compute gradients through the graph (non-in-place).
        
        Args:
            grad_outputs: Dictionary mapping output names to their gradients
            num_nodes: Number of nodes to traverse in reverse (None for all)
        """
        def safe_copy(val):
            if isinstance(val, Wave):
                return val.copy()
            return val.clone() if isinstance(val, torch.Tensor) else copy.deepcopy(val)

        # Create a copy for non-in-place operation
        graph_copy = copy.deepcopy(self)

        # Copy the grad_outputs before passing to backward_
        grads_copy = {name: safe_copy(grad) for name, grad in grad_outputs.items()}
        return graph_copy.backward_(grads_copy, num_nodes)

    def backward_(self, grad_outputs: dict, num_nodes: int = None) -> Any:
        """Compute gradients through the graph in-place.
        
        Args:
            grad_outputs: Dictionary mapping output names to their gradients
            num_nodes: Number of nodes to traverse in reverse (None for all)
        """
        # Extend cache with gradients (no copying)
        self.cache.update(grad_outputs)

        # Get nodes to process (all or limited number from the end)
        nodes_to_process = self.nodes[-num_nodes:] if num_nodes else self.nodes

        # Traverse nodes in reverse order
        for node in reversed(nodes_to_process):
            # Get cached inputs for backward
            cached_inputs = [self.cache[name] for name in node.save_for_backward]

            # Get outputs from cache and copy them
            args = [self.cache[name] for name in node.outputs]
            
            # Compute gradient through this node
            grad_output = node.operator.gradient(*args, *cached_inputs)

            # Update cache with gradients (handles both tuple and single cases)
            grads = (grad_output,) if not isinstance(grad_output, tuple) else grad_output
            for name, grad in zip(node.inputs, grads):
                self.cache[name] = grad

        # Return gradients for leaf inputs
        leaf_grads = {
            key: self.cache[key]
            for key in self.leaf_keys
        }

        return leaf_grads if len(leaf_grads) > 1 else next(iter(leaf_grads.values()))