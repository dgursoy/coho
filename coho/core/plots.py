"""Plot utilities for wave fields and optimization results."""

import matplotlib.pyplot as plt
import jax.numpy as jnp
from .wave import Wave
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union
from .metrics import WaveMetrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass
class WavePlotter:
    """Wave field plotting utilities."""
    
    @staticmethod
    def plot_waves(waves: Union[Wave, List[Wave]]) -> None:
        """Plot wave field amplitudes and phases.
        
        Displays amplitude and phase for one or more wave fields. For multiple waves,
        each wave is shown in a column with amplitude and phase stacked vertically.
        
        Args:
            waves: Single Wave or list of Wave fields, each with shape (ny, nx)
                  If a single Wave is provided, it will be treated as a list of length 1.
        
        Example:
            # Plot single wave
            WavePlotter.plot_waves(wave)
            
            # Plot multiple waves
            WavePlotter.plot_waves([wave1, wave2])
        """
        # Convert input to list if single wave
        if isinstance(waves, Wave):
            waves = [waves]
            
        n_waves = len(waves)
        
        # Create figure with subplots arranged in columns
        fig, axes = plt.subplots(2, n_waves, figsize=(4*n_waves, 8))
        if n_waves == 1:
            axes = axes.reshape(-1, 1)  # Make 2D for consistent indexing
            
        for i, wave in enumerate(waves):
            # Convert JAX arrays to NumPy for plotting
            form_np = jnp.asarray(wave.form).copy()
            
            # Amplitude
            amplitude = jnp.abs(form_np)
            # amplitude = jnp.real(form_np)
            im_amp = axes[0,i].imshow(amplitude, cmap='gray')
            divider = make_axes_locatable(axes[0,i])
            cax1 = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im_amp, cax=cax1)
            axes[0,i].set_title(f'Amplitude {i+1}')
            
            # Phase
            phase = jnp.angle(form_np)
            # phase = jnp.imag(form_np)
            im_phase = axes[1,i].imshow(phase, cmap='twilight_shifted', vmin=-jnp.pi, vmax=jnp.pi)
            divider = make_axes_locatable(axes[1,i])
            cax2 = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im_phase, cax=cax2)
            axes[1,i].set_title(f'Phase {i+1}')
        
        fig.tight_layout()
        plt.show()


@dataclass
class DataPlotter:
    """Data plotting utilities."""
    
    @staticmethod
    def plot_data(data: jnp.ndarray,
                  title: str = "Data",
                  figsize: Optional[Tuple[int, int]] = None,
                  max_plots: int = 4) -> None:
        """Plot intensity patterns.
        
        Args:
            data: Array of shape (n_distances, ny, nx)
            title: Base title for the plots
            figsize: Optional figure size
            max_plots: Maximum number of plots to show (default: 4)
        """
        n_distances = min(data.shape[0], max_plots)
        
        if figsize is None:
            figsize = (4*n_distances, 4)
            
        # Create figure and axes array
        fig, axes = plt.subplots(1, n_distances, figsize=figsize)
        if n_distances == 1:
            axes = [axes]  # Make iterable for single distance
            
        # Convert to numpy and plot
        data_np = jnp.asarray(data[:n_distances]).copy()
        
        for i in range(n_distances):
            im = axes[i].imshow(data_np[i], cmap='gray')
            axes[i].set_title(f'{title} at distance {i}')
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
        
        
        fig.tight_layout()
        plt.show()


class OptimizationPlotter:
    """Plotting utilities for optimization results."""
    
    @staticmethod
    def plot_convergence(metrics_history: Dict[str, List[float]], 
                         metric_groups: Optional[Dict[str, List[str]]] = None,
                         figsize: Tuple[int, int] = (14, 4)) -> None:
        """Plot optimization convergence history.
        
        Args:
            metrics_history: Dictionary of metric names to their history
            figsize: Figure size (width, height)
        """
        # Default grouping
        metric_groups = {
            'Cost': ['cost'],
            'Relative Error': [k for k in metrics_history.keys() if k.startswith('Rel(')],
            'Absolute Error': [k for k in metrics_history.keys() if k.startswith('Abs(')]
        }
        # Remove empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        
        # Create figure with subplots in a row
        n_groups = len(metric_groups)
        if n_groups == 0:
            return
            
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        
        # Plot each metric group
        for ax, (group_name, metric_names) in zip(axes, metric_groups.items()):
            for name in metric_names:
                if name in metrics_history:
                    label = name.split('(')[-1].rstrip(')') if '(' in name else name
                    ax.semilogy(metrics_history[name], label=label)
            ax.set_title(group_name)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(group_name)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    

