# Operators

Operators enable modular wave transformations, interactions, and optimizations. They are categorized into **Simple**, **Composite**, and **Optimization-enhanced** operators.

- [Simple Operators](#simple-operators)
- [Composite Operators](#composite-operators)
- [Operators for Optimization](#operators-for-optimization)

## Simple Operators

Simple operators perform individual wave transformations:

```python
# Propagate wave by distance
wave = Propagate().apply(wave, distance) 

# Modulate wave by another wave
wave = Modulate().apply(wave1, wave2)

# Detect wave intensity
wave = Detect().apply(wave)

# Shift wave by a shift value
wave = Shift().apply(wave, shift)

# Crop wave to match another wave
wave = Crop().apply(wave1, wave2)

# Broadcast wave to multiple values
wave = Broadcast().apply(wave, values)
```

## Composite Operators

Composite operators allow composing multiple operators together to form a new operator. 

**Example:** Compose `detect(propagate(modulate(propagate(wave1, distance1), wave2), distance2))` from `propagate`, `modulate`, and `detect` operators.

```python
from operators import CompositeOperator, Modulate, Propagate, Detect

# Create the composite operator
composite_operator = CompositeOperator(
    Detect(), 
    CompositeOperator(
        Propagate(), 
        CompositeOperator(
            Modulate(), 
            Propagate()
        )
    )
)

# Inputs to the composite operator
wave1 = ...
distance1 = ...
wave2 = ...
distance2 = ...

# Apply the composite operator
result = composite_operator.forward(wave1, distance1, wave2, distance2)
```

## Operators for Optimization

Efficient optimization workflows often involve repeated operator calls with a mix of fixed and dynamic inputs. Leveraging caching and parameter freezing can enhance performance, clarity, and robustness.


### Caching Operator Outputs

Caching eliminates redundant operations by reusing results from previous computations with identical inputs, making it particularly useful in optimization loops and nested workflows.

**Example:** Cache the propagated wave with `distance` using the `propagate` operator.

```python
from operators import CachedOperator, Propagate

# Create a cached propagate operator
cached_propagate = CachedOperator(Propagate())

# Compute and cache the result
result1 = cached_propagate.forward(wave1, distance1)  # Computes and caches
result2 = cached_propagate.forward(wave1, distance1)  # Reuses cache

# Clear cache when inputs change
cached_propagate.clear_cache()

# Apply the cached operator again with cleared cache
result3 = cached_propagate.forward(wave1, distance1)  # Recomputes and caches new result
```

### Parameter Freezing

Freezing marks specific parameters as immutable, simplifying input handling and improving consistency. Combined with caching, it eliminates the need for runtime input checks for cached operators.

**Example:** Freeze `distance` in `propagate(wave, distance)`.

```python
from operators import FrozenParameter, Propagate

# Create a frozen parameter
distance = FrozenParameter()

# Freeze input parameters
wave = ...
distance.freeze(...)

# Apply the propagation
propagate = Propagate()
wave_prop = propagate.forward(wave, distance) 

# Unfreeze distance
distance.unfreeze()

# New dynamic value for distance
distance = ... 
```

### Freezing with Caching

When combined with `CachedOperator`, freezing optimizes caching further by ensuring consistent inputs:

```python
from operators import CachedOperator, Propagate, FrozenParameter

# Create a cached propagate operator
cached_propagate = CachedOperator(Propagate())

# Create a frozen distance parameter
distance = FrozenParameter()

# Freeze distance
frozen_distance = distance.freeze(...)

# Apply the cached operator with frozen inputs
result1 = cached_propagate.forward(wave, frozen_distance)  # Cache created
result2 = cached_propagate.forward(wave, frozen_distance)  # Cache reused
```


