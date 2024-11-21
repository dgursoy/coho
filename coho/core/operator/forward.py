# core/operator/forward.py

"""Forward operator."""

from abc import ABC, abstractmethod
from coho.config.models import Config

__all__ = ['Forward', 'Holography']


class Forward(ABC):
    """Forward operator."""

    def __init__(self, config: Config):
        """Initialize forward operator."""
        self._init_components(config.simulation)
        self._init_operators(config.operator)
        self._init_experiment(config.experiment)

    def _init_components(self, config):
        """Initialize optical components."""
        from coho.factories import (
            WavefrontFactory,
            OpticFactory,
            SampleFactory,
            DetectorFactory
        )
        
        self.wavefront = WavefrontFactory().create(
            model=config.wavefront.model,
            properties=config.wavefront.properties
        )
        self.optic = OpticFactory().create(
            model=config.optic.model,
            properties=config.optic.properties
        )
        self.sample = SampleFactory().create(
            model=config.sample.model,
            properties=config.sample.properties
        )
        self.detector = DetectorFactory().create(
            model=config.detector.model,
            properties=config.detector.properties
        )

        # Map components to their types
        self.component_map = {
            config.wavefront.id: ('wavefront', self.wavefront),
            config.optic.id: ('optic', self.optic),
            config.sample.id: ('sample', self.sample),
            config.detector.id: ('detector', self.detector)
        }

    def _init_operators(self, config):
        """Initialize propagation and interaction operators."""
        from coho.factories import PropagatorFactory, InteractorFactory
        self.propagator = PropagatorFactory().create(model='fresnel')
        self.interactor = InteractorFactory().create(model='thin_object')

    def _init_experiment(self, config):
        """Initialize experiment parameters."""
        self.component_sequence = config.properties.components
        self.current_position = self.wavefront.properties.geometry.position.z

    @abstractmethod
    def run(self):
        """Execute simulation workflow."""
        pass

    def forward(self):
        """Forward model alias for run."""
        return self.run()

    def adjoint(self):
        """Adjoint model."""
        pass

class Holography(Forward):
    """Holography operator."""

    def run(self):
        """Execute holography workflow."""
        for component_id in self.component_sequence:
            component_type, component = self.component_map[component_id]
            target_position = component.properties.geometry.position.z
            
            # Propagate wavefront
            self._propagate(target_position)
            
            # Handle component interaction
            if component_type in ('optic', 'sample'):
                self.wavefront = self.interactor.interact(self.wavefront, component)
            elif component_type == 'detector':
                component.detect(self.wavefront)

        return self.detector.acquire()

    def _propagate(self, target_position: float):
        """Propagate wavefront to target position."""
        distance = target_position - self.current_position
        self.wavefront = self.propagator.propagate(self.wavefront, distance)
        self.current_position = target_position

    def forward(self):
        """Forward model alias for run."""
        return self.run()

    def adjoint(self):
        """Adjoint model."""
        pass