# Graphs

Graphs represent a sequence of operations performed on a wave. They are implemented using the `Graph` class, which inherits from the `Operator` base class. As such, graphs can be used in the same way as any other operator.

Graphs are expressed as [Directed Acyclic Graphs (DAGs)](https://en.wikipedia.org/wiki/Directed_acyclic_graph), where:

- Nodes correspond to operators.
- Edges represent the data flow between these operators.

## Core Functionality

- **Node Representation**: Each node in the graph is an operator.
- **Data Flow**: Edges define how data flows between nodes, determining the sequence of operations.
- **Execution**
    - **`forward` Method**: Executes the graph by processing nodes in order.
        - Inputs are provided as a dictionary of key-value pairs.
        - Outputs are returned as a dictionary of results, allowing flexible retrieval.
    - **Intermediate Outputs**: The output of each node is passed as input to subsequent nodes.

**Example: A Simple Graph for `(a * b + c)^2`**

```python
# Define individual nodes
node1 = Node(
    operator=Multiply(), 
    inputs=('a', 'b'), 
    outputs=('out1',),
    save_for_backward=('a', 'b')
)
node2 = Node(
    operator=Add(), 
    inputs=('out1', 'c'), 
    outputs=('out2',),
    save_for_backward=('out2')
)
node3 = Node(
    operator=Power(), 
    inputs=('out2'), 
    outputs=('result',),
    save_for_backward=('out2')
)

# Define graph
graph = Graph(debug=True)
graph.add_node(node1)
graph.add_node(node2)
graph.add_node(node3)

inputs = {
    'a': 1,
    'b': 2,
    'c': 3
}

result = graph.forward(**inputs)
print(result['result'])  # Output: 36  ((1 * 2 + 3)^2 = 5^2 = 25)
```

## Gradients and Optimization

- **`gradient` Method**: Computes gradients of the loss with respect to the input wave, facilitating backpropagation. It uses the chain rule to propagate gradients through the graph, similar to PyTorch or other automatic differentiation frameworks. The benefit of having a custom gradient is that it can be more efficient and flexible than the default PyTorch gradient, but one can also use the default PyTorch `autograd` if desired.

- **Dynamic Inputs**: Support for dynamic inputs with the `requires_grad` property, similar to PyTorch, enabling integration into optimization loops. This is a Graph-specific feature, as Operators don't know if they are required to compute gradients until the graph is executed.

## Caching and Debugging

- **Intermediate Caching**: Selected intermediate results can be cached for reuse in gradient computations, improving efficiency. This is handled by the `save_for_gradient` key in the Graph definition, and can be set for each node.

- **Debugging**: Access intermediate results by using the debug method and querying specific keys, simplifying troubleshooting. This is handled by the `debug` key in the Graph initialization. 

## Metadata and Reproducibility

- **Blueprint for Execution**: Graphs act as a reproducible blueprint, describing a sequence of operations.

- **Reusability**: Graphs can be reused across contexts, maintaining a consistent execution pipeline.

## Hardware Flexibility

- **Device Control**: Nodes can be executed on either the CPU or GPU by setting the device property. Each node can be set to a different device, and the graph will be executed on the devices specified in the graph. This can be useful for optimizing performance, or for running the same graph on different devices or in the future when we have parallel execution of graph nodes.

