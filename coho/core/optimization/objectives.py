# core/optimization/objectives.py

"""Objective functions for optimization problems.

This module provides implementations of various objective functions
used in optimization algorithms.

Classes:
    Objective: Abstract base class for objective functions
    LeastSquares: L2 norm fitting objective
    MagnitudeFitting: Amplitude-only matching objective

Methods:
    evaluate: Compute objective value
"""

from typing import Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod


class Objective(ABC):
    """Base class for objective functions."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize objective.

        Args:
            id: Unique identifier
            parameters: Configuration dict
        """
        self.id = id
        self.parameters = parameters or {}

    @abstractmethod
    def evaluate(self, simulation: np.ndarray, measurement: np.ndarray) -> float:
        """Compute objective value.

        Args:
            simulation: Simulated values
            measurement: Measurement values

        Returns:
            Objective value
        """
        pass


class LeastSquares(Objective):
    """L2 norm fitting objective."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize least squares objective.
        
        Args:
            id: Unique identifier
            parameters: Configuration dict (optional)
        """
        super().__init__(id, parameters)

    def evaluate(self, simulation: np.ndarray, measurement: np.ndarray) -> float:
        """Compute L2 norm difference.

        Args:
            simulation: Simulated values
            measurement: Measurement values

        Returns:
            Sum of squared differences
        """
        return np.sum((simulation - measurement) ** 2)
    
    def gradient(self, simulation: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        """Compute gradient of L2 norm difference.

        Args:
            simulation: Simulated values
            measurement: measurement values
        """
        return 2 * (simulation - measurement)


class MagnitudeFitting(Objective):
    """Amplitude-only matching objective."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize magnitude fitting objective.
        
        Args:
            id: Unique identifier
            parameters: Configuration dict (optional)
        """
        super().__init__(id, parameters)

    def evaluate(self, simulation: np.ndarray, measurement: np.ndarray) -> float:
        """Compute magnitude difference.

        Args:
            simulation: Complex Simulated values
            measurement: Complex measurement values

        Returns:
            Sum of squared magnitude differences
        """
        return np.sum((np.abs(simulation) - np.abs(measurement)) ** 2)
    
    def gradient(self, simulation: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        """Compute gradient of magnitude difference.

        Args:
            simulation: Complex Simulated values
            measurement: Complex measurement values
        """
        return 2 * (np.abs(simulation) - np.abs(measurement)) * np.conj(simulation) / np.abs(simulation)
