"""The optimisation loop.

Each pass: filter x into the density, solve, measure the goal and the limits, complete their gradients with
the adjoint method and push them back through the filters, and let the optimiser move x. Every heavy step is
one call into the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from michell.adjoint import total_gradient
from michell.evaluation import apply_filters, evaluate
from michell.modules import Filter, Response
from michell.problem import Problem


@dataclass(frozen=True)
class Limit:
    """A limit on a response: ``response`` must not exceed ``at_most`` (in the response's unit)."""

    response: Response
    at_most: float


class Optimiser(Protocol):
    """What the loop needs from an optimiser."""

    def start(self, problem: Problem, x: np.ndarray, limits: Sequence[Limit], filters: Sequence[Filter]) -> None:
        """Called once before the first pass, with everything that stays fixed during the run."""
        ...

    def update(
        self,
        x: np.ndarray,
        goal: float,
        d_goal: np.ndarray,
        limits: Sequence[float],
        d_limits: Sequence[np.ndarray],
    ) -> np.ndarray:
        """The next design variables, from the values and gradients (with respect to x) at x."""
        ...


@dataclass(frozen=True)
class Pass:
    """The record of one pass: the goal and limit values at the start of the pass, and the largest change
    of a design variable made by its update."""

    number: int
    goal: float
    limits: tuple[float, ...]
    change: float


@dataclass(frozen=True)
class Result:
    """The design variables and density after the last pass, and the record of every pass."""

    x: np.ndarray
    density: np.ndarray
    passes: tuple[Pass, ...]


def optimise(
    problem: Problem,
    goal: Response,
    limits: Sequence[Limit],
    filters: Sequence[Filter],
    optimiser: Optimiser,
    x: np.ndarray,
    passes: int,
    stop_change: float | None = None,
) -> Result:
    """Minimises ``goal`` subject to ``limits``, starting from ``x`` (one value per element, in [0, 1]).

    Runs ``passes`` passes, or fewer when ``stop_change`` is given and the largest change of a design
    variable in one pass falls to it or below. Elements outside the part keep x = 0.
    """
    x = np.where(problem.grid.active == 1.0, np.asarray(x, dtype=np.float64), 0.0)
    reads = set(goal.reads).union(*(limit.response.reads for limit in limits))
    optimiser.start(problem, x, limits, filters)
    history = []
    for number in range(1, passes + 1):
        ev, fields = evaluate(problem, x, filters, reads)
        goal_value = goal.value(ev)
        d_goal = total_gradient(problem, goal, ev, fields, filters)
        limit_values = [limit.response.value(ev) for limit in limits]
        d_limits = [total_gradient(problem, limit.response, ev, fields, filters) for limit in limits]

        x_new = optimiser.update(x, goal_value, d_goal, limit_values, d_limits)
        change = float(np.abs(x_new - x).max())
        history.append(Pass(number, goal_value, tuple(limit_values), change))
        x = x_new
        if stop_change is not None and change <= stop_change:
            break
    return Result(x=x, density=apply_filters(x, filters)[-1], passes=tuple(history))
