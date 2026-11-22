"""michell: generative design (topology optimisation) with a Rust engine.

The package holds:

- the 2D analysis: ``Grid`` (square elements), ``Elasticity`` (plane stress: solve, adjoint, sensitivity,
  element strain energy and stress) and ``element_stiffness``;
- the 3D analysis: ``Grid3d`` (cube elements), ``Elasticity3d`` (the same calls, solved by conjugate
  gradients on a stiffness matrix that is never assembled) and ``element_stiffness_3d``;
- the problem: ``Material``, ``LoadCase``, ``Problem`` (on a 2D or a 3D grid);
- the module interface (``docs/design/modules.md``): ``Response``, ``Filter``, ``Partials`` and the
  read-only ``Evaluation``, with ``Row`` (``michell.row``), the result of one check, which a module's
  self-test and the verification table both use;
- the responses ``Compliance`` and ``Mass``, the ``DensityFilter`` (2D and 3D), the ``HelmholtzFilter``
  (3D) and the ``Projection`` (both);
- the minimum member size as a limit (``michell.member_size``): ``MemberSize``, a response on the filtered
  field, and ``member_threshold``, the threshold that a wanted member size gives;
- several designs from one set of design variables (``michell.designs``): ``on``, which names the design a
  goal or a limit reads, and ``robust_designs``, the eroded, nominal and dilated designs of the robust
  formulation;
- the optimisation loop ``optimise`` with its ``Limit``, its ``Result`` and the ``LimitNotMet`` warning,
  the ``OptimalityCriteria`` optimiser (the engine's ``Oc`` update), the ``MovingAsymptotes`` optimiser
  (the engine's ``Mma`` update), and the finite-difference gradient check ``check_gradient``;
- ``michell.recipes``: whole procedures from one call; ``stiff_part``, the stiffest part of a given mass
  with a minimum member size (the robust formulation, then the member-size limit), and
  ``member_size_bound``, the bound of that limit for a filter radius in elements, and ``GridTooCoarse``, the
  warning ``stiff_part`` gives when the member size is fewer than five elements;
- ``michell.select``: nodes, elements and degrees of freedom picked by position in mm and by face, the
  ``held`` mask of a problem and the load vector of a force spread over nodes, without index arithmetic;
- ``michell.export``: ``write_vtk``, per-element fields written as a legacy VTK file for ParaView, and
  ``write_result`` and ``read_result``, a ``Result`` or a ``StiffPart`` written to an ``.npz`` file (the
  arrays) and a ``.json`` file (the rest: values, limits met, stop reason, the record of every pass) and read
  back as a plain dictionary;
- ``michell.mesh``: ``Mesh``, a closed triangle mesh in mm with its volume and its check (``closed()``),
  ``read_stl`` and ``write_stl``, and the bodies ``box``, ``sphere``, ``cylinder`` and ``torus`` with the
  exact volumes of their facets;
- ``michell.regions``: ``occupancy``, the share of every voxel of a grid inside a union of meshes minus
  another (the engine's voxeliser), and ``voxel_model``, the ``VoxelModel`` of a part from its design, solid
  and void regions (the grid, the active and the held-solid voxels, a volume table, the bodies, an optional
  mirror), with the warning ``VoxelVolumeDiffers``;
- ``michell.attach``: supports and loads on the voxel model (``Support``, ``Load`` and ``Case``, a load case
  given by what its loads act on) and ``set_up``, which turns them into a ``Problem``, the voxels held
  solid (the solid regions and a skin under every contact patch on design voxels) and the density filter's
  faces, with a report, as a ``SetUp``; the warning ``LoadOnSupport``;
- ``michell.problem_file``: the problem file (TOML, ``docs/design/problem-file.md``): ``read`` (the file
  checked into a ``ProblemFile``), ``prepare`` (the geometry, the voxel model and the set-up, solving
  nothing: what ``michell check`` shows) and ``run`` (``prepare``, ``stiff_part`` and the files
  ``result.json``, ``result.npz`` and optionally ``result.vtk``: what ``michell run`` does), also as
  ``michell.prepare`` and ``michell.run``; its refusals are the ``ProblemFileError`` family of
  ``michell.errors``, one stable code per kind of fault;
- ``dot``, the dot product of two vectors in the engine, for responses (with the OpenBLAS build that NumPy
  ships, ``a @ b`` runs long products on every core);
- ``EngineError``, the error every engine call raises; the checks written in Python raise it too, with
  code ``invalid_input`` (also a ``ValueError``, ``TypeError`` or ``KeyError``, see ``michell.errors``).

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
from michell.designs import On, on, robust_designs
from michell.evaluation import Evaluation
from michell.filters import DensityFilter, HelmholtzFilter, KeepSolid, Projection
from michell.gradient_check import GradientCheck, check_gradient, check_on_small_problem
from michell.member_size import MemberSize, member_threshold
from michell.mma import MovingAsymptotes
from michell.modules import Filter, Partials, Response
from michell.oc import OptimalityCriteria
from michell.optimise import Limit, LimitNotMet, Pass, Result, optimise
from michell.problem import LoadCase, Material, Problem
from michell import attach, export, mesh, problem_file, recipes, regions, select
from michell.attach import Case, Load, LoadOnSupport, SetUp, Support, set_up
from michell.export import read_result, write_result, write_vtk
from michell.mesh import Mesh, read_stl, write_stl
from michell.problem_file import Prepared, ProblemFile, RunOutcome, prepare, run
from michell.recipes import GridTooCoarse, RecipePass, StiffPart, member_size_bound, stiff_part
from michell.regions import VolumeRow, VoxelModel, VoxelVolumeDiffers, voxel_model
from michell.responses import Compliance, Mass
from michell.row import Row

__version__ = _version("michell")

__all__ = [
    "Case",
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
    "GridTooCoarse",
    "HelmholtzFilter",
    "KeepSolid",
    "Limit",
    "LimitNotMet",
    "Load",
    "LoadCase",
    "LoadOnSupport",
    "Mass",
    "Material",
    "MemberSize",
    "Mesh",
    "Mma",
    "MovingAsymptotes",
    "Oc",
    "On",
    "OptimalityCriteria",
    "Partials",
    "Pass",
    "Prepared",
    "Problem",
    "ProblemFile",
    "Projection",
    "RecipePass",
    "Response",
    "Result",
    "Row",
    "RunOutcome",
    "SetUp",
    "StiffPart",
    "Support",
    "VolumeRow",
    "VoxelModel",
    "VoxelVolumeDiffers",
    "__version__",
    "attach",
    "check_gradient",
    "check_on_small_problem",
    "dot",
    "element_stiffness",
    "element_stiffness_3d",
    "engine_version",
    "export",
    "member_size_bound",
    "member_threshold",
    "mesh",
    "on",
    "optimise",
    "prepare",
    "problem_file",
    "read_result",
    "read_stl",
    "recipes",
    "regions",
    "robust_designs",
    "run",
    "select",
    "set_up",
    "stiff_part",
    "voxel_model",
    "write_result",
    "write_stl",
    "write_vtk",
]
