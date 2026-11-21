"""The optimisation loop.

Each pass: filter x into the density, solve, measure the goal and the limits, complete their gradients with
the adjoint method and push them back through the filters, and let the optimiser move x. Every heavy step is
one call into the engine.

Several designs can be made from the one x (``designs=``, see ``michell.designs``): the output of
``filters`` then passes through each design's own filters, each design read by a goal or a limit is solved
on its own, and each gradient goes back through its design's filters before it goes back through
``filters``. Without ``designs`` the output of ``filters`` is the one design.

The states of one pass are handed to the solves of the next as their first guess (a warm start), design by
design: the iterative 3D solver then needs fewer steps once the design changes little. They are passed in
the open, from one evaluation to the next; no object keeps a solution on its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from michell.designs import (
    design_chains,
    design_gradient,
    design_of,
    evaluate_designs,
    reads_by_design,
)
from michell.evaluation import apply_filters
from michell.modules import Filter, Response
from michell.problem import Problem


@dataclass(frozen=True)
class Limit:
    """A limit on a response: ``response`` must not exceed ``at_most`` (in the response's unit). With
    several designs, the response names the design it reads: ``Limit(on("dilated", Mass()), at_most=...)``.

    ``scale`` (in the response's unit, positive) is the size by which an optimiser measures how far the
    limit is from being met: ``MovingAsymptotes`` hands MMA ``(value - at_most) / scale``. The default,
    ``None``, is ``|at_most|`` (1 when ``at_most`` is 0): the limit's relative excess over its bound. A
    bound far below the values the response takes before the limit is met (the member-size limit, whose
    bound is of the order of 1e-7) needs a larger scale; see ``michell.member_size``."""

    response: Response
    at_most: float
    scale: float | None = None

    def __post_init__(self):
        if self.scale is not None and not (np.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"the scale of a limit must be a positive number; got {self.scale!r}")


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
    """The record of one pass: the goal and limit values at the start of the pass (each on the design it
    reads), the largest change of a design variable made by its update, and ``solver_steps``, the
    conjugate-gradient steps of the pass's solves (the states of every load case and any adjoint solve, on
    every design solved), summed; 0 for the direct 2D solver. With named designs, ``design_solver_steps``
    holds the same count per design solved (empty without named designs)."""

    number: int
    goal: float
    limits: tuple[float, ...]
    change: float
    solver_steps: int = 0
    design_solver_steps: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    """The design variables and density after the last pass, the record of every pass, and the number of
    threads the solves ran on. ``density`` is the output of ``filters``; with named designs, ``designs``
    maps each name to that design's field after the last pass (empty without named designs).

    Each record holds the values at the start of its pass, so the last recorded goal belongs to the design
    before the last update: one update behind ``x``, ``density`` and ``designs``."""

    x: np.ndarray
    density: np.ndarray
    passes: tuple[Pass, ...]
    threads: int = 1
    designs: Mapping[str, np.ndarray] = field(default_factory=dict)


def optimise(
    problem: Problem,
    goal: Response,
    limits: Sequence[Limit],
    filters: Sequence[Filter],
    optimiser: Optimiser,
    x: np.ndarray,
    passes: int,
    stop_change: float | None = None,
    designs: Mapping[str, Sequence[Filter]] | None = None,
) -> Result:
    """Minimises ``goal`` subject to ``limits``, starting from ``x`` (one value per element, in [0, 1]).

    Runs ``passes`` passes, or fewer when ``stop_change`` is given and the largest change of a design
    variable in one pass falls to it or below. Elements outside the part keep x = 0.

    ``designs``, if given, maps a name to the filters that make that design from the output of
    ``filters``; the goal and every limit must then name the design they read, with
    ``michell.on(name, response)``. Only the designs a goal or a limit reads are evaluated, and a design
    is solved only for the physics its responses read.
    """
    x = np.where(problem.grid.active == 1.0, np.asarray(x, dtype=np.float64), 0.0)
    chains = design_chains(designs)
    goal_design, goal_response = design_of(goal, chains)
    limit_parts = [design_of(limit.response, chains) for limit in limits]
    reads = reads_by_design([(goal_design, goal_response), *limit_parts], chains)
    optimiser.start(problem, x, limits, filters)
    history = []
    states = None  # per design, the warm start of the next pass's solves
    for number in range(1, passes + 1):
        base_fields, evaluated = evaluate_designs(problem, x, filters, chains, reads, start=states)
        steps = {design: [ev.solver_steps] for design, (ev, _) in evaluated.items()}

        def gradient(design, response):
            return design_gradient(problem, response, design, filters, chains, base_fields, evaluated, steps[design])

        goal_value = goal_response.value(evaluated[goal_design][0])
        d_goal = gradient(goal_design, goal_response)
        limit_values = [response.value(evaluated[design][0]) for design, response in limit_parts]
        d_limits = [gradient(design, response) for design, response in limit_parts]

        x_new = optimiser.update(x, goal_value, d_goal, limit_values, d_limits)
        change = float(np.abs(x_new - x).max())
        solved = {design: sum(counts) for design, counts in steps.items() if reads[design]}
        by_design = {} if designs is None else solved
        total_steps = sum(sum(counts) for counts in steps.values())
        history.append(Pass(number, goal_value, tuple(limit_values), change, total_steps, by_design))
        x = x_new
        states = {design: ev.states for design, (ev, _) in evaluated.items()}
        if stop_change is not None and change <= stop_change:
            break
    density = apply_filters(x, filters)[-1]
    final = {} if designs is None else {name: apply_filters(density, chain)[-1] for name, chain in chains.items()}
    return Result(x=x, density=density, passes=tuple(history), threads=problem.threads, designs=final)
