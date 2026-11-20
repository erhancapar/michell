"""michell: generative design (topology optimisation) with a Rust engine.

Pre-alpha. Currently available: the 2D analysis, that is a grid of square elements, plane-stress
elasticity (solve, adjoint, sensitivity, element strain energy and stress) and the engine's error type.
"""

from importlib.metadata import version as _version

from michell._engine import (
    Elasticity,
    EngineError,
    Grid,
    element_stiffness,
    engine_version,
)

__version__ = _version("michell")

__all__ = [
    "Elasticity",
    "EngineError",
    "Grid",
    "__version__",
    "element_stiffness",
    "engine_version",
]
