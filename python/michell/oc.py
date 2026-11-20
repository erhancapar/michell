"""The optimality-criteria (OC) optimiser of the 88-line code, for the optimisation loop.

OC handles one goal that falls as material is added (such as compliance) and exactly one limit that is
linear in the density (such as mass). It measures that limit inside its bisection on the filtered field,
so the filter chain must be empty or the engine's density filter alone. The update runs in the engine
(``michell.Oc``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from michell import _engine
from michell._engine import dot
from michell.evaluation import Evaluation
from michell.modules import Filter
from michell.optimise import Limit
from michell.problem import Problem

# The limit counts as linear in rho when its value equals weights . rho to this fraction of the scale.
LINEARITY_TOLERANCE = 1e-10


class OptimalityCriteria:
    """OC with a move limit (0.2 in the 88-line code)."""

    name = "OC"

    def __init__(self, move_limit: float = 0.2):
        self.move_limit = move_limit

    def start(self, problem: Problem, x: np.ndarray, limits: Sequence[Limit], filters: Sequence[Filter]) -> None:
        if len(limits) != 1:
            raise ValueError(f"OC handles exactly one limit; got {len(limits)}")
        limit = limits[0]
        if limit.response.reads:
            raise ValueError(
                f"OC needs a limit that reads no physics; {limit.response.name} reads {limit.response.reads}"
            )
        if len(filters) > 1 or (filters and not isinstance(filters[0], _engine.DensityFilter)):
            raise ValueError("OC measures its limit on the filtered field and supports only the density filter")
        self._filter = filters[0] if filters else None
        self._weights = self._linear_weights(problem, limit)
        self._bound = float(limit.at_most)
        self._engine = _engine.Oc(problem.grid, self.move_limit)

    def update(
        self,
        x: np.ndarray,
        goal: float,
        d_goal: np.ndarray,
        limits: Sequence[float],
        d_limits: Sequence[np.ndarray],
    ) -> np.ndarray:
        return self._engine.update(x, d_goal, d_limits[0], self._weights, self._bound, self._filter)

    @staticmethod
    def _linear_weights(problem: Problem, limit: Limit) -> np.ndarray:
        """The weights a with value = a . rho, taken from the limit's d_density and checked on three
        density fields; a limit that is not linear in rho is refused."""
        n_elements = problem.grid.n_elements
        fields = [np.zeros(n_elements), np.ones(n_elements), np.linspace(0.1, 0.9, n_elements)]
        weights = None
        for rho in fields:
            ev = Evaluation(problem, rho, states={})
            partials = limit.response.partials(ev)
            if partials.d_density is None or partials.d_state:
                raise ValueError(f"OC needs a limit with d_density only; {limit.response.name} has other partials")
            if weights is None:
                weights = np.array(partials.d_density, dtype=np.float64)
            scale = np.abs(weights).sum()
            if abs(limit.response.value(ev) - dot(weights, rho)) > LINEARITY_TOLERANCE * scale:
                raise ValueError(f"OC needs a limit linear in the density; {limit.response.name} is not")
        return weights
