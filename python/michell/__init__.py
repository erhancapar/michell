"""michell: generative design (topology optimisation) with a Rust engine.

The package holds:

- the 2D analysis: ``Grid`` (square elements), ``Elasticity`` (plane stress: solve, adjoint, sensitivity,
  element strain energy and stress) and ``element_stiffness``;
- the problem: ``Material``, ``LoadCase``, ``Problem``;
- the module interface (``docs/design/modules.md``): ``Response``, ``Filter``, ``Partials``, ``Row`` and
  the read-only ``Evaluation``;
- the responses ``Compliance`` and ``Mass``, and the ``DensityFilter``;
- the optimisation loop ``optimise`` with its ``Limit``, the ``OptimalityCriteria`` optimiser (the engine's
  ``Oc`` update), the ``MovingAsymptotes`` optimiser (the engine's ``Mma`` update), and the
  finite-difference gradient check ``check_gradient``;
- ``EngineError``, the error every engine call raises.

Masks (active elements, held degrees of freedom) may be given as boolean arrays.
"""

from importlib.metadata import version as _version

from michell._engine import (
    EngineError,
    Mma,
    Oc,
    element_stiffness,
    engine_version,
)
from michell.analysis import Elasticity, Grid
from michell.evaluation import Evaluation
from michell.filters import DensityFilter
from michell.gradient_check import GradientCheck, check_gradient, check_on_small_problem
from michell.mma import MovingAsymptotes
from michell.modules import Filter, Partials, Response, Row
from michell.oc import OptimalityCriteria
from michell.optimise import Limit, Pass, Result, optimise
from michell.problem import LoadCase, Material, Problem
from michell.responses import Compliance, Mass

__version__ = _version("michell")

__all__ = [
    "Compliance",
    "DensityFilter",
    "Elasticity",
    "EngineError",
    "Evaluation",
    "Filter",
    "GradientCheck",
    "Grid",
    "Limit",
    "LoadCase",
    "Mass",
    "Material",
    "Mma",
    "MovingAsymptotes",
    "Oc",
    "OptimalityCriteria",
    "Partials",
    "Pass",
    "Problem",
    "Response",
    "Result",
    "Row",
    "__version__",
    "check_gradient",
    "check_on_small_problem",
    "element_stiffness",
    "engine_version",
    "optimise",
]
