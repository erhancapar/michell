"""The method of moving asymptotes (MMA) for the optimisation loop.

MMA (Svanberg 1987, in the form of Svanberg's 2007 note) handles one goal and any number of limits, each a
smooth function of the design variables. The update runs in the engine (``michell.Mma``); this class turns
the loop's goal and limits into the numbers MMA works with.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from michell import _engine
from michell.modules import Filter
from michell.optimise import Limit
from michell.problem import Problem


class MovingAsymptotes:
    """MMA over the active elements, each design variable between 0 and 1.

    Scaling. MMA's fixed constants (the penalty c_i = 1000 on a limit that cannot be met, the term
    1e-5 that makes each approximation strictly convex, the tolerance of the subproblem solver) are plain
    numbers, so they only mean the same thing on every problem when the goal and the limits reach MMA as
    numbers of order one, whatever their units (a compliance may be 1000 N mm, a mass 0.02 g). So:

    - the goal is divided by the size of its value at the first pass, so it starts at 1 or -1;
    - a limit ``value <= at_most`` reaches MMA as ``(value - at_most) / |at_most| <= 0``: its relative
      excess over the bound, 0 where the limit is just met (a bound of 0 is used without division).

    Both scales are fixed for the whole run: MMA's asymptotes and approximations assume that the functions
    do not change from one pass to the next.
    """

    name = "MMA"

    def start(self, problem: Problem, x: np.ndarray, limits: Sequence[Limit], filters: Sequence[Filter]) -> None:
        # Only the active elements are design variables; the others keep x = 0 and never reach MMA.
        self._active = problem.grid.active == 1.0
        n = int(self._active.sum())
        self._bounds = np.array([float(limit.at_most) for limit in limits])
        self._limit_scales = np.array([abs(b) if b != 0.0 else 1.0 for b in self._bounds])
        self._goal_scale: float | None = None  # the goal is first known at the first update
        self._engine = _engine.Mma(np.zeros(n), np.ones(n), len(limits))

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
