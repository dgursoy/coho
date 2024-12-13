# Operators

An operator transforms waves using two key methods:

- `forward`: Performs the wave transformation
- `backward`: Computes gradients for backpropagation

This follows [PyTorch](https://pytorch.org/)'s convention for implementing differentiable operations.

> **Note:** The `Operator` class inherits from [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), following PyTorch's convention for implementing differentiable operations. While we currently use only basic functionality, this inheritance provides future flexibility for integrating with PyTorch's ecosystem.

## Key Features

- **Base Class Inheritance:** All operators inherit from the `Operator` base class, which inherits from [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
- **Core Methods:** All implement `forward` and `gradient` methods for forward transformation and gradient computations.
- **Reusability:** Operators are stateless, meaning their behavior depends only on the inputs to their forward or backward methods. This ensures they can be reused across different contexts without introducing side effects.
- **Caching:** Operators can optionally maintain a temporary cache for intermediate results via the `_cache` attribute. This cache is accessed through keys, and is external to the operator's core logic. It is used to store computationally expensive results that may be reused in optimization loops. The based `Operator` class provides methods for managing this cache.

## Example

Below is an example showing how to use the `Propagate` operator.

```python
from coho.core.wave import Wave
from coho.core.operator import Propagate

# Create a wave
wave = Wave(torch.rand(128, 128) + 1j * torch.rand(128, 128), energy=10, position=0, spacing=1e-4)

# Create an instance of the Propagate operator
propagate = Propagate() 

# Propagate the wave by 100cm (propagation kernel is computed and cached)
wave = propagate.forward(wave, distance=100) 
print(wave.position) # Output: 100.0

# Propagate the wave by 100cm again (propagation kernel is used from the cache)
wave = propagate.forward(wave, distance=100) 
print(wave.position) # Output: 200.0

# Backpropagate the wave by 200cm (gradient kernel is computed and cached)
wave = propagate.gradient(wave, distance=200) 
print(wave.position) # Output: 0.0
```
