"""Experiment scanning orchestration."""

from typing import Dict, Type
from .sweeper import prepare
from .batcher import (
    Batch, WavefrontBatch, OpticBatch, SampleBatch, DetectorBatch
)

__all__ = ['Scanner']

class Scanner:
    """Orchestrates parameter scanning experiments."""
    
    def __init__(self, config):
        """Initialize scanner with experiment config."""
        self.config = config
        self.parameter_arrays = prepare(config)
        self.batches = {}
        self._factory_map = self._get_factory_defaults()
        self._batch_map = self._get_batch_defaults()
    
    def _get_factory_defaults(self):
        """Get default factory mapping."""
        from coho.factories.simulation_factories import (
            WavefrontFactory, OpticFactory, SampleFactory, DetectorFactory
        )
        return {
            'wavefront': WavefrontFactory,
            'optic': OpticFactory,
            'sample': SampleFactory,
            'detector': DetectorFactory
        }
    
    def _get_batch_defaults(self):
        """Get default batch class mapping."""
        return {
            'wavefront': WavefrontBatch,
            'optic': OpticBatch,
            'sample': SampleBatch,
            'detector': DetectorBatch
        }
    
    def components(self):
        """Get the components from the config."""
        return self.config.experiment.properties.components
    
    def _get_config_item(self, component_id):
        """Get configuration for a component by its ID."""
        for item_type, item in self.config.simulation:
            if item.id == component_id:
                return item_type, item
        return None, None
    
    def _initialize_batches(self):
        """Execute scanning workflow."""
        for component_id in self.components():
            item_type, item = self._get_config_item(component_id)
            if not item:
                continue
                
            factory_class = self._factory_map.get(item_type)
            if factory_class:
                factory = factory_class()
                component_class = factory.get_class(item.model)
                batch_class = self._batch_map.get(item_type, Batch)  # Use default Batch if type not found
                self.batches[component_id] = batch_class(
                    component_class, 
                    item.properties, 
                    self.parameter_arrays[component_id]
                )
        
        return self.batches
    
    def run(self):
        """Execute scanning workflow."""
        self._initialize_batches()

        # TODO: Implement scanning workflow
        print (self.batches['my_wavefront'])


        return self.batches
    

        