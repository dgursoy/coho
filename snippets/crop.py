# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Local imports
from coho.core.component import Wave
from coho.core.operator import Crop, Broadcast, Shift

# Load test images
lena = np.load('./coho/resources/images/lena.npy') + 256.
cameraman = np.load('./coho/resources/images/cameraman.npy') + 256.
lena_zoom = zoom(lena, 0.25)

# Initialize waves
modulator = Wave(lena[0:512, 0:512], energy=10.0, spacing=1e-4, position=0.0).normalize().multiply(3)
reference = Wave(cameraman[0:256, 0:256], energy=10.0, spacing=1e-4, position=0.0).normalize().multiply(6)

def test_shift_padding():
    wave = modulator
    
    # Define shifts including large shifts
    y_shifts = np.array([200, 100, 50])
    x_shifts = np.array([50, 10, -50])
    
    # Create operators
    shift_op = Shift()
    
    # Apply operations (in-place)
    shifted = shift_op.apply(wave, y_shifts, x_shifts)
    shifted_copy = shifted.copy()
    restored = shift_op.adjoint(shifted, y_shifts, x_shifts)
    
    # Visualize
    for m in range(shifted.shape[0]):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(wave.amplitude[0])
        plt.title('Original')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(shifted_copy.amplitude[m])
        plt.title('Shifted')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(restored.amplitude[m])
        plt.colorbar()
        plt.title('Restored')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_shift_padding()