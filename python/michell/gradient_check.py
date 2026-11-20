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
from michell.evaluation import evaluate
from michell.filters import DensityFilter
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


def check_on_small_problem(response: Response, seed: int = 0, step: float = 1e-6) -> GradientCheck:
    """``check_gradient`` of ``response`` on ``small_problem(seed)``: the check no module skips."""
    problem, x, filters = small_problem(seed)
    return check_gradient(problem, response, x, filters, step)
