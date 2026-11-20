"""Several designs from one set of design variables, and the robust set of three.

The design variables x pass through the problem's ``filters`` once; each named design then applies its own
filters to that field and is a density field of its own, with its own solves. A goal or a limit says which
design it reads with ``on(name, response)``. The gradient of a response goes back through its design's
filters and then through ``filters``, so every design pulls on the same x.

The robust formulation of Wang, Lazarov and Sigmund ("On projection methods, convergence and robust
formulations in topology optimization", Struct. Multidisc. Optim. 43, 2011) is the use this is made for:
three projections of one filtered field at the thresholds 0.5 + spread, 0.5 and 0.5 - spread give an
eroded, a nominal and a dilated design, standing for a part made too thin, as drawn and too thick
(``robust_designs``). Its goal is the worst of the three; for compliance that is the eroded design, since
stiffness only grows with density, so the goal reads ``eroded`` alone. The mass limit sits on ``dilated``,
with its bound rescaled so that ``nominal`` has the asked mass; see ``docs/design/modules.md``.

Without named designs (``designs=None``) the loop runs one design whose density is the output of
``filters``: the same code, with the design's own filters empty.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from michell.adjoint import backward_through_filters, total_gradient
from michell.evaluation import Evaluation, apply_filters, evaluate
from michell.filters import Projection
from michell.modules import Filter, Partials, Response
from michell.problem import Problem
from michell.row import Row

# The one design of a run without named designs: its density is the output of the problem's filters.
UNNAMED = None


@dataclass(frozen=True)
class On:
    """A response read on the named design ``design``. It is a response itself: name, unit, the physics it
    reads, value, partials and self-test are those of ``response``, which sees the ``Evaluation`` of that
    design and nothing else."""

    design: str
    response: Response

    @property
    def name(self) -> str:
        return self.response.name

    @property
    def unit(self) -> str:
        return self.response.unit

    @property
    def reads(self) -> Sequence[str]:
        return self.response.reads

    def value(self, ev: Evaluation) -> float:
        return self.response.value(ev)

    def partials(self, ev: Evaluation) -> Partials:
        return self.response.partials(ev)

    def self_test(self) -> Sequence[Row]:
        return self.response.self_test()


def on(design: str, response: Response) -> On:
    """``response`` read on the design called ``design``: ``on("eroded", Compliance())``."""
    if not isinstance(design, str):
        raise TypeError(f"a design is named by a string; got {design!r}")
    if isinstance(response, On):
        raise ValueError(f"{response.name} already reads the design {response.design!r}")
    return On(design, response)


def robust_designs(beta: float, spread: float) -> dict[str, list[Filter]]:
    """The three designs of the robust formulation (Wang, Lazarov and Sigmund 2011): ``eroded``,
    ``nominal`` and ``dilated``, each one ``Projection`` of sharpness ``beta`` at the threshold
    0.5 + spread, 0.5 and 0.5 - spread. ``0 < spread < 0.5``."""
    if not 0.0 < spread < 0.5:
        raise ValueError(f"the spread of the thresholds must lie between 0 and 0.5; got {spread}")
    return {
        "eroded": [Projection(beta, 0.5 + spread)],
        "nominal": [Projection(beta, 0.5)],
        "dilated": [Projection(beta, 0.5 - spread)],
    }


def design_chains(designs: Mapping[str, Sequence[Filter]] | None) -> dict[str | None, tuple[Filter, ...]]:
    """Each design's own filters, applied after the problem's filters; ``{UNNAMED: ()}`` without designs."""
    if designs is None:
        return {UNNAMED: ()}
    if not isinstance(designs, Mapping) or not designs:
        raise ValueError("designs must be a mapping from a name to a list of filters, with at least one design")
    chains = {}
    for name, chain in designs.items():
        if not isinstance(name, str):
            raise TypeError(f"a design is named by a string; got {name!r}")
        if not isinstance(chain, (list, tuple)):
            raise TypeError(f"design {name!r}: the filters of a design must be a list; got {type(chain).__name__}")
        chains[name] = tuple(chain)
    return chains


def design_of(response: Response, chains: Mapping[str | None, Sequence[Filter]]) -> tuple[str | None, Response]:
    """The design a goal or a limit reads, and the response itself (without its ``on``).

    With named designs every response must name one of them; without, none may name a design."""
    named = isinstance(response, On)
    if UNNAMED in chains:
        if named:
            raise ValueError(
                f"{response.name} reads the design {response.design!r}, but no designs were given; pass "
                f"designs= to name them"
            )
        return UNNAMED, response
    if not named:
        raise ValueError(
            f"when designs are given, every goal and limit must name the design it reads, as "
            f"michell.on(name, response); {response.name} does not (the designs are {list(chains)})"
        )
    if response.design not in chains:
        raise ValueError(f"{response.name} reads the design {response.design!r}; the designs are {list(chains)}")
    return response.design, response.response


def reads_by_design(
    parts: Iterable[tuple[str | None, Response]], chains: Mapping[str | None, Sequence[Filter]]
) -> dict[str | None, set[str]]:
    """The physics to solve on each design that some response reads, in the order of ``chains``. A design
    no response reads is left out, so it is never solved."""
    reads: dict[str | None, set[str]] = {}
    for design, response in parts:
        reads.setdefault(design, set()).update(response.reads)
    return {design: reads[design] for design in chains if design in reads}


def evaluate_designs(
    problem: Problem,
    x: np.ndarray,
    filters: Sequence[Filter],
    chains: Mapping[str | None, Sequence[Filter]],
    reads: Mapping[str | None, Iterable[str]],
    start: Mapping[str | None, Mapping[tuple[str, str], np.ndarray]] | None = None,
) -> tuple[list[np.ndarray], dict[str | None, tuple[Evaluation, list[np.ndarray]]]]:
    """Filters x once through ``filters``, then evaluates every design in ``reads`` from that field: its own
    filters, and a solve of each physics it reads (``michell.evaluation.evaluate``).

    ``start`` maps a design to the states it hands to its solves as their first guess (the ``states`` of
    that design's evaluation in the previous pass). Returns the fields of ``filters`` (from
    ``apply_filters``) and, per design, its evaluation and its own fields."""
    base_fields = apply_filters(x, filters)
    evaluated = {}
    for design, physics in reads.items():
        first_guess = None if start is None else start.get(design)
        evaluated[design] = evaluate(problem, base_fields[-1], chains[design], physics, start=first_guess)
    return base_fields, evaluated


def design_gradient(
    problem: Problem,
    response: Response,
    design: str | None,
    filters: Sequence[Filter],
    chains: Mapping[str | None, Sequence[Filter]],
    base_fields: Sequence[np.ndarray],
    evaluated: Mapping[str | None, tuple[Evaluation, list[np.ndarray]]],
    solver_steps: list[int] | None = None,
) -> np.ndarray:
    """dg/dx of ``response`` read on ``design``: the adjoint gradient at that design's evaluation, back
    through the design's own filters, then back through ``filters``. The steps of any adjoint solve are
    appended to ``solver_steps`` (as in ``michell.adjoint.total_gradient``)."""
    ev, fields = evaluated[design]
    d_filtered = total_gradient(problem, response, ev, fields, chains[design], solver_steps)
    return backward_through_filters(filters, base_fields, d_filtered)
