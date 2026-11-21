"""One evaluation of a design: the density, the solved states, and the read-only view responses get."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from michell.errors import InvalidValue, UnknownName
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

    For the loop, not for responses: ``states`` (every solved state, keyed by (physics, case): the warm
    start of the next pass) and ``solver_steps`` (the conjugate-gradient steps the solves took, summed; 0
    for the direct 2D solver).
    """

    def __init__(
        self, problem: Problem, density, states: dict[tuple[str, str], np.ndarray], solver_steps: int = 0
    ):
        self._problem = problem
        self._density = read_only(density)
        self._states = {key: read_only(value) for key, value in states.items()}
        self._solver_steps = solver_steps

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

    @property
    def states(self) -> dict[tuple[str, str], np.ndarray]:
        """Every solved state, keyed by (physics, case); the arrays are read-only."""
        return dict(self._states)

    @property
    def solver_steps(self) -> int:
        return self._solver_steps

    def state(self, physics: str, case: str) -> np.ndarray:
        if (physics, case) not in self._states:
            raise UnknownName(
                f"no solved state for ({physics!r}, {case!r}): a response must list the physics it "
                f"reads in `reads`"
            )
        return self._states[(physics, case)]

    def loads(self, physics: str, case: str) -> np.ndarray:
        if physics != "elasticity":
            raise UnknownName(f"unknown physics {physics!r}; known: {KNOWN_PHYSICS}")
        return self._problem.case(case).loads


def apply_filters(x: np.ndarray, filters: Sequence[Filter]) -> list[np.ndarray]:
    """The chain x -> filter 1 -> filter 2 -> ... -> rho. Returns every field: [x, ..., rho]."""
    fields = [x]
    for each in filters:
        fields.append(each.forward(fields[-1]))
    return fields


def evaluate(
    problem: Problem,
    x: np.ndarray,
    filters: Sequence[Filter],
    reads: Iterable[str],
    start: Mapping[tuple[str, str], np.ndarray] | None = None,
) -> tuple[Evaluation, list[np.ndarray]]:
    """Filters x into the density, solves every physics in ``reads`` for every load case, and returns the
    evaluation with the list of fields from ``apply_filters`` (needed to go back through the filters).

    ``start``, if given, maps (physics, case) to a first guess of that state, usually ``states`` of the
    previous pass's evaluation: the iterative 3D solver starts from it (a warm start), the direct 2D solver
    ignores it. A state missing from ``start`` is solved from zero. Nothing is kept between calls."""
    fields = apply_filters(x, filters)
    density = fields[-1]
    states = {}
    steps = 0
    for physics in sorted(set(reads)):
        if physics != "elasticity":
            raise InvalidValue(f"unknown physics {physics!r}; known: {KNOWN_PHYSICS}")
        for case in problem.cases:
            first_guess = None if start is None else start.get((physics, case.name))
            states[(physics, case.name)], case_steps = problem.solve(density, case.loads, start=first_guess)
            steps += case_steps
    return Evaluation(problem, density, states, solver_steps=steps), fields
