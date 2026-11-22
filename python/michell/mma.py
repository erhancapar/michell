"""The method of moving asymptotes (MMA) for the optimisation loop.

MMA (Svanberg 1987, in the form of Svanberg's 2007 note) handles one goal and any number of limits, each a
smooth function of the design variables. The update runs in the engine (``michell.Mma``); this class turns
the loop's goal and limits into the numbers MMA works with.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue
from michell.modules import Filter
from michell.optimise import Limit
from michell.problem import Problem


def limit_scale(limit: Limit) -> float:
    """The scale a limit reaches MMA with: its own ``scale``, or ``|at_most|`` (1 when that is 0)."""
    if limit.scale is not None:
        return float(limit.scale)
    bound = float(limit.at_most)
    return abs(bound) if bound != 0.0 else 1.0


class MovingAsymptotes:
    """MMA over the active elements, each design variable between 0 and 1.

    ``move_limit`` is the largest change of a design variable in one update (MMA's move limit, as a share
    of the variable's range 0 to 1): a number above 0 and at most 1, by default 0.5, the value of
    Svanberg's note. MMA's asymptotes already keep a step within bounds that adapt from pass to pass; a
    smaller move limit adds a fixed bound on top, which damps the large steps a sharp projection can
    provoke, at the price of more passes for a large change.

    Scaling. MMA's fixed constants (the penalty c_i = 1000 on a limit that cannot be met, the term
    1e-5 that makes each approximation strictly convex, the tolerance of the subproblem solver) are plain
    numbers, so they only mean the same thing on every problem when the goal and the limits reach MMA as
    numbers of order one, whatever their units (a compliance may be 1000 N mm, a mass 0.02 g). So:

    - the goal is divided by the size of its value at the first pass, so it starts at 1 or -1;
    - a limit ``value <= at_most`` reaches MMA as ``(value - at_most) / scale <= 0``, with the limit's
      ``scale`` (``michell.Limit``); by default ``scale = |at_most|``, so the limit reaches MMA as its
      relative excess over the bound, 0 where it is just met (a bound of 0 is used without division).

    Both scales are fixed for the whole run: MMA's asymptotes and approximations assume that the functions
    do not change from one pass to the next.
    """

    name = "MMA"

    def __init__(self, move_limit: float = 0.5):
        if isinstance(move_limit, bool) or not isinstance(move_limit, numbers.Real):
            raise InvalidType(f"the move limit must be a number; got {move_limit!r}")
        move_limit = float(move_limit)
        if not 0.0 < move_limit <= 1.0:
            raise InvalidValue(f"the move limit must lie above 0 and at most 1; got {move_limit!r}")
        self.move_limit = move_limit

    def start(self, problem: Problem, x: np.ndarray, limits: Sequence[Limit], filters: Sequence[Filter]) -> None:
        # Only the active elements are design variables; the others keep x = 0 and never reach MMA.
        self._active = problem.grid.active == 1.0
        n = int(self._active.sum())
        self._bounds = np.array([float(limit.at_most) for limit in limits])
        self._limit_scales = np.array([limit_scale(limit) for limit in limits])
        self._goal_scale: float | None = None  # the goal is first known at the first update
        # The update runs on the problem's threads; its designs do not depend on how many there are.
        self._engine = _engine.Mma(np.zeros(n), np.ones(n), len(limits), problem.threads, self.move_limit)

    def update(
        self,
        x: np.ndarray,
        goal: float,
        d_goal: np.ndarray,
        limits: Sequence[float],
        d_limits: Sequence[np.ndarray],
    ) -> np.ndarray:
        if self._goal_scale is None:
            self._goal_scale = abs(goal) if goal != 0.0 else 1.0
        active = self._active
        f0 = goal / self._goal_scale
        d_f0 = np.ascontiguousarray(d_goal[active] / self._goal_scale)
        f = (np.asarray(limits, dtype=np.float64) - self._bounds) / self._limit_scales
        d_f = np.zeros((len(self._bounds), int(active.sum())))
        for i, gradient in enumerate(d_limits):
            d_f[i] = gradient[active] / self._limit_scales[i]
        x_active = np.ascontiguousarray(x[active], dtype=np.float64)
        x_new = np.zeros_like(x, dtype=np.float64)
        x_new[active] = self._engine.update(x_active, f0, d_f0, np.ascontiguousarray(f), d_f)
        return x_new
