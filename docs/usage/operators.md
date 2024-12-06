# Operators

## Contents

- [Propagation](#propagation)
- [Interactions](#interactions)
- [Manipulations](#manipulations)
- [Broadcasting](#broadcasting)

## Propagation

```python
# Propagate wave by distance
propagate = Propagate()
wave = propagate.apply(wave, distance)
```

## Interactions

```python
# Modulate wave by another wave
modulate = Modulate()
wave = modulate.apply(wave1, wave2)

# Detect wave
detect = Detect()
wave = detect.apply(wave)
```

## Manipulations

```python
# Shift wave
shift = Shift()
wave = shift.apply(wave, shift)

# Crop wave
crop = Crop()
wave = crop.apply(wave, crop)
```

## Broadcasting

```python
# Broadcast wave
broadcast = Broadcast()
wave = broadcast.apply(wave, values)
```