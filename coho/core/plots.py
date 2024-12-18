"""Plot utilities for wave fields and optimization results."""

import matplotlib.pyplot as plt
import jax.numpy as jnp
from .wave import Wave
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from .metrics import WaveMetrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass
class WavePlotter:
    """Wave field plotting utilities."""
    
    @staticmethod
    def plot_wave(wave: Wave) -> None:
        """Plot wave field amplitude and phase.
        
        Args:
            wave: Wave field with shape (ny, nx)
        """
        # Convert JAX arrays to NumPy for plotting
        form_np = jnp.asarray(wave.form).copy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        # Amplitude
        amplitude = jnp.abs(form_np)
        im_amp = ax1.imshow(amplitude, cmap='gray')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im_amp, cax=cax1)
        ax1.set_title('Amplitude')
        
        # Phase
        phase = jnp.angle(form_np)
        im_phase = ax2.imshow(phase, cmap='RdGy', vmin=-jnp.pi, vmax=jnp.pi)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im_phase, cax=cax2)
        ax2.set_title('Phase')
        
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
    

