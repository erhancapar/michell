"""One evaluation of a design: the density, the solved states, and the read-only view responses get."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from michell.modules import Filter
from michell.problem import Problem

# The physics a response can read. Elasticity is solved per load case.
KNOWN_PHYSICS = ("elasticity",)


def read_only(values) -> np.ndarray:
    """A float64 copy of ``values`` that cannot be written to."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


class Evaluation:
    """What a response may look at, read-only (``docs/design/modules.md``).

    ``grid``, ``density`` (rho, one value per element), ``material``, ``cases``, ``state(physics, case)``
    (the solved state, one value per degree of freedom), ``loads(physics, case)``, and ``engine`` (the
    elasticity model, for heavy helpers such as element strain energy and stress).
    """

    def __init__(self, problem: Problem, density, states: dict[tuple[str, str], np.ndarray]):
        self._problem = problem
        self._density = read_only(density)
        self._states = {key: read_only(value) for key, value in states.items()}

    @property
    def grid(self):
        return self._problem.grid

    @property
    def density(self) -> np.ndarray:
        return self._density

    @property
    def material(self):
        return self._problem.material

    @property
    def cases(self):
        return self._problem.cases

    @property
    def engine(self):
        return self._problem.elasticity

    def state(self, physics: str, case: str) -> np.ndarray:
        if (physics, case) not in self._states:
            raise KeyError(
                f"no solved state for ({physics!r}, {case!r}): a response must list the physics it "
                f"reads in `reads`"
            )
        return self._states[(physics, case)]

    def loads(self, physics: str, case: str) -> np.ndarray:
        if physics != "elasticity":
            raise KeyError(f"unknown physics {physics!r}; known: {KNOWN_PHYSICS}")
        return self._problem.case(case).loads


def apply_filters(x: np.ndarray, filters: Sequence[Filter]) -> list[np.ndarray]:
    """The chain x -> filter 1 -> filter 2 -> ... -> rho. Returns every field: [x, ..., rho]."""
    fields = [x]
    for each in filters:
        fields.append(each.forward(fields[-1]))
    return fields


def evaluate(
    problem: Problem, x: np.ndarray, filters: Sequence[Filter], reads: Iterable[str]
) -> tuple[Evaluation, list[np.ndarray]]:
    """Filters x into the density, solves every physics in ``reads`` for every load case, and returns the
    evaluation with the list of fields from ``apply_filters`` (needed to go back through the filters)."""
    fields = apply_filters(x, filters)
    density = fields[-1]
    states = {}
    for physics in sorted(set(reads)):
        if physics != "elasticity":
            raise ValueError(f"unknown physics {physics!r}; known: {KNOWN_PHYSICS}")
        for case in problem.cases:
            states[(physics, case.name)] = problem.elasticity.solve(density, case.loads)
    return Evaluation(problem, density, states), fields
