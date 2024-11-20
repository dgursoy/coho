# common/registry.py

"""Central component registry."""

from ..core.simulation import *
from ..core.operator import *
from ..core.optimization import *

__all__ = ['MODEL_REGISTRY']

# Schema of valid component models
MODEL_REGISTRY = {
    'simulation': {
        'wavefront': {
            'gaussian': GaussianWavefront,
            'constant': ConstantWavefront,
            'rectangular': RectangularWavefront
        },
        'detector': {
            'photon_counting': PhotonCountingDetector,
            'integrating': IntegratingDetector
        },
        'sample': {
            'custom_profile': CustomProfileSample
        },
        'optic': {
            'coded_aperture': CodedApertureOptic,
            'slit_aperture': SlitApertureOptic,
            'circle_aperture': CircleApertureOptic,
            'custom_profile': CustomProfileOptic
        }
    },
    'operator': {
        'propagator': {
            'fresnel': FresnelPropagator,
            'fraunhofer': FraunhoferPropagator
        },
        'interactor': {
            'thin_object': ThinObjectInteractor,
            'thick_object': ThickObjectInteractor
        }
    },
    'optimization': {
        'solver': {
            'gradient_descent': GradientDescent
        },
        'objective': {
            'least_squares': LeastSquares
        }
    }
}
