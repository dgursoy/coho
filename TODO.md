1. Evaluate a full dict based approach for the configuration.

```yaml
experiment:
  model: holography
  properties:
    components:
      my_wavefront:
        sweep:
          physical.amplitude: [50, 500, 50]
          physical.energy: [10, 100, 10]
      my_aperture:
        sweep:
          geometry.rotation: [0, 360, 30]
          profile.bit_size: [16, 64, 16]
      my_sample: {}  # no sweeps
      my_detector: {} # no sweeps
  scan:
    order: [1, 2, 3, 4]
```
