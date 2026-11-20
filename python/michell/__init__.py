"""michell: generative design (topology optimisation) with a Rust engine.

The package holds:

- the 2D analysis: ``Grid`` (square elements), ``Elasticity`` (plane stress: solve, adjoint, sensitivity,
  element strain energy and stress) and ``element_stiffness``;
- the 3D analysis: ``Grid3d`` (cube elements), ``Elasticity3d`` (the same calls, solved by conjugate
  gradients on a stiffness matrix that is never assembled) and ``element_stiffness_3d``;
- the problem: ``Material``, ``LoadCase``, ``Problem``;
- the module interface (``docs/design/modules.md``): ``Response``, ``Filter``, ``Partials`` and the
  read-only ``Evaluation``, with ``Row`` (``michell.row``), the result of one check, which a module's
  self-test and the verification table both use;
- the responses ``Compliance`` and ``Mass``, and the ``DensityFilter``;
- the optimisation loop ``optimise`` with its ``Limit``, the ``OptimalityCriteria`` optimiser (the engine's
  ``Oc`` update), the ``MovingAsymptotes`` optimiser (the engine's ``Mma`` update), and the
  finite-difference gradient check ``check_gradient``;
- ``dot``, the dot product of two vectors in the engine, for responses (with the OpenBLAS build that NumPy
  ships, ``a @ b`` runs long products on every core);
- ``EngineError``, the error every engine call raises.

The verification rows of ``docs/verification.md`` are in the subpackage ``michell.verification``
(``python -m michell.verification`` writes the table); it is not imported with ``michell``.

Masks (active elements, held degrees of freedom) may be given as boolean arrays.
"""

from importlib.metadata import version as _version

from michell._engine import (
    EngineError,
    Mma,
    Oc,
    dot,
    element_stiffness,
    element_stiffness_3d,
    engine_version,
)
from michell.analysis import Elasticity, Grid
from michell.analysis3d import Elasticity3d, Grid3d
from michell.evaluation import Evaluation
from michell.filters import DensityFilter
from michell.gradient_check import GradientCheck, check_gradient, check_on_small_problem
from michell.mma import MovingAsymptotes
from michell.modules import Filter, Partials, Response
from michell.oc import OptimalityCriteria
from michell.optimise import Limit, Pass, Result, optimise
from michell.problem import LoadCase, Material, Problem
from michell.responses import Compliance, Mass
from michell.row import Row

__version__ = _version("michell")

__all__ = [
    "Compliance",
    "DensityFilter",
    "Elasticity",
    "Elasticity3d",
    "EngineError",
    "Evaluation",
    "Filter",
    "GradientCheck",
    "Grid",
    "Grid3d",
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
    "dot",
    "element_stiffness",
    "element_stiffness_3d",
    "engine_version",
    "optimise",
]
