"""The finite-difference check of a response's gradient, for any response.

The gradient the loop uses (partials, adjoint, filters backwards) is compared with central differences of
the response's value, element by element:

    dg/dx_e  ~  (g(x + s e_e) - g(x - s e_e)) / (2 s).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from michell.adjoint import total_gradient
from michell.analysis import Grid
from michell.analysis3d import Grid3d
from michell.evaluation import evaluate
from michell.filters import DensityFilter, HelmholtzFilter
from michell.modules import Filter, Response
from michell.problem import LoadCase, Material, Problem


@dataclass(frozen=True)
class GradientCheck:
    """The two gradients on the checked elements."""

    response: str
    elements: np.ndarray
    adjoint: np.ndarray
    finite_difference: np.ndarray

    @property
    def largest_difference(self) -> float:
        """max_e |fd_e - adjoint_e| / max_e |adjoint_e|: the difference relative to the gradient's size."""
        return float(np.abs(self.finite_difference - self.adjoint).max() / np.abs(self.adjoint).max())

    @property
    def largest_element_difference(self) -> float:
        """max_e |fd_e - adjoint_e| / |adjoint_e|: the worst element on its own scale."""
        return float((np.abs(self.finite_difference - self.adjoint) / np.abs(self.adjoint)).max())


def check_gradient(
    problem: Problem,
    response: Response,
    x: np.ndarray,
    filters: Sequence[Filter] = (),
    step: float = 1e-6,
    elements: Sequence[int] | None = None,
) -> GradientCheck:
    """Compares the gradient of ``response`` at ``x`` with central differences of step ``step``, on
    ``elements`` (default: every active element). x must stay in [0, 1] after the step."""
    x = np.asarray(x, dtype=np.float64)
    ev, fields = evaluate(problem, x, filters, response.reads)
    gradient = total_gradient(problem, response, ev, fields, filters)
    if elements is None:
        elements = np.flatnonzero(problem.grid.active)
    elements = np.asarray(elements)
    differences = np.empty(len(elements))
    for k, element in enumerate(elements):
        up, down = x.copy(), x.copy()
        up[element] += step
        down[element] -= step
        value_up = response.value(evaluate(problem, up, filters, response.reads)[0])
        value_down = response.value(evaluate(problem, down, filters, response.reads)[0])
        differences[k] = (value_up - value_down) / (2 * step)
    return GradientCheck(response.name, elements, gradient[elements], differences)


def small_problem(seed: int = 0) -> tuple[Problem, np.ndarray, list[Filter]]:
    """The small problem every module's gradient is checked on.

    A plate of 12 by 6 elements (side 1.5 mm, thickness 2 mm, steel), clamped on its left edge, with a
    hole of 3 by 2 inactive elements, two weighted load cases (a downward tip load, and a pull at the top
    right corner), a density filter of radius 3.75 mm, and design variables drawn at random from [0.1, 0.9].
    """
    nx, ny = 12, 6
    active = np.ones(nx * ny, dtype=bool)
    for j in (2, 3):
        for i in (5, 6, 7):
            active[i + nx * j] = False
    grid = Grid(nx, ny, 1.5, 2.0, active=active)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for j in range(ny + 1):
        held[2 * grid.node(0, j)] = held[2 * grid.node(0, j) + 1] = True
    tip = np.zeros(grid.n_dofs)
    tip[2 * grid.node(nx, ny // 2) + 1] = -1000.0
    pull = np.zeros(grid.n_dofs)
    pull[2 * grid.node(nx, ny)] = 500.0
    material = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_mm3=7.85e-3)
    problem = Problem(grid, held, material, [LoadCase("tip", tip, 1.0), LoadCase("pull", pull, 0.5)])
    x = np.random.default_rng(seed).uniform(0.1, 0.9, grid.n_elements)
    return problem, x, [DensityFilter(grid, 3.75)]


def small_problem_3d(
    seed: int = 0, solver_tolerance: float = 1e-12, filter_tolerance: float = 1e-12
) -> tuple[Problem, np.ndarray, list[Filter]]:
    """A small 3D problem for gradient checks.

    A block of 8 by 4 by 4 cube elements (side 1.5 mm, steel) clamped on its face x = 0, with a pocket of
    2 by 2 by 2 inactive elements, two weighted load cases (a downward load at the middle of the lower far
    edge, and a sideways pull at a top far corner), a Helmholtz filter of radius 3.75 mm (2.5 element sides)
    and design variables drawn at random from [0.1, 0.9]. Its 675 degrees of freedom keep the multigrid
    to one level, so every elasticity solve is direct; ``solver_tolerance`` and ``filter_tolerance`` are the
    relative residuals of the elasticity and the filter solves.
    """
    nx, ny, nz = 8, 4, 4
    active = np.ones(nx * ny * nz, dtype=bool)
    for k in (1, 2):
        for j in (1, 2):
            for i in (4, 5):
                active[i + nx * (j + ny * k)] = False
    grid = Grid3d(nx, ny, nz, 1.5, active=active)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(nz + 1):
        for j in range(ny + 1):
            node = grid.node(0, j, k)
            held[3 * node : 3 * node + 3] = True
    down = np.zeros(grid.n_dofs)
    down[3 * grid.node(nx, ny // 2, 0) + 2] = -1000.0
    pull = np.zeros(grid.n_dofs)
    pull[3 * grid.node(nx, ny, nz) + 1] = 500.0
    material = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_mm3=7.85e-3)
    problem = Problem(
        grid, held, material, [LoadCase("down", down, 1.0), LoadCase("pull", pull, 0.5)],
        solver_tolerance=solver_tolerance,
    )  # fmt: skip
    x = np.random.default_rng(seed).uniform(0.1, 0.9, grid.n_elements)
    return problem, x, [HelmholtzFilter(grid, 3.75, tolerance=filter_tolerance)]


def check_on_small_problem(response: Response, seed: int = 0, step: float = 1e-6) -> GradientCheck:
    """``check_gradient`` of ``response`` on ``small_problem(seed)``: the check no module skips."""
    problem, x, filters = small_problem(seed)
    return check_gradient(problem, response, x, filters, step)
