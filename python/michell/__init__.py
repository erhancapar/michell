"""michell: generative design (topology optimisation) with a Rust engine.

Pre-alpha: the package exposes the engine version and one function that tests the Rust boundary.
"""

from importlib.metadata import version as _version

from michell._engine import engine_version, sum_f64

__version__ = _version("michell")

__all__ = ["__version__", "engine_version", "sum_f64"]
