# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Local imports
from coho.core.component import Wave
from coho.core.operator import Crop, Broadcast 

# Load test images
lena = np.load('./coho/resources/images/lena.npy') + 256.
cameraman = np.load('./coho/resources/images/cameraman.npy') + 256.
lena_zoom = zoom(lena, 0.25)

# Initialize waves
modulator = Wave(lena[0:512, 0:256], energy=10.0, spacing=1e-4, position=0.0).normalize().multiply(6)
reference = Wave(cameraman[0:128, 0:256], energy=10.0, spacing=1e-4, position=0.0).normalize().multiply(6)


crop_op = Crop()

# Forward: modify wave to match reference
modified = crop_op.apply(reference, modulator)     # modulator -> reference size
print(f"Forward operation:")
print(f"modulator shape: {modulator.shape}")
print(f"Reference shape: {reference.shape}")
print(f"Modified shape: {modified.shape}")

# Adjoint: restore modulator shape
restored = crop_op.adjoint(modified, modulator)    # back to modulator size
print(f"\nAdjoint operation:")
print(f"Modified shape: {modified.shape}")
print(f"modulator shape: {modulator.shape}")
print(f"Restored shape: {restored.shape}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(modulator.amplitude[2])
axes[0, 0].set_title('modulator Wave')
axes[0, 1].imshow(reference.amplitude[2])
axes[0, 1].set_title('Reference Wave')
axes[1, 0].imshow(modified.amplitude[2])
axes[1, 0].set_title('Modified to Reference Size')
axes[1, 1].imshow(restored.amplitude[2])
axes[1, 1].set_title('Restored to modulator Size')
plt.tight_layout()
plt.show()