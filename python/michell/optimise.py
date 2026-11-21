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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from michell._engine import EngineError
from michell.designs import (
    design_chains,
    design_gradient,
    design_of,
    evaluate_designs,
    reads_by_design,
)
from michell.errors import InvalidValue
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
    bound is of the order of 1e-7) needs a larger scale; see ``michell.member_size``.

    ``at_most`` must be a finite number. A response may also refuse a bound that no design can meet: if it
    has a method ``impossible_bound(problem, at_most, filters)`` (``filters``: every filter between x and
    the design it reads), returning the reason as a string or ``None``, ``optimise`` calls it before the
    first pass and refuses the limit with that reason. ``Mass`` and ``Compliance`` have one."""

    response: Response
    at_most: float
    scale: float | None = None

    def __post_init__(self):
        try:
            finite = bool(np.isfinite(self.at_most))
        except TypeError:
            finite = False
        if not finite:
            raise InvalidValue(f"the bound of a limit must be a finite number; got at_most={self.at_most!r}")
        if self.scale is not None and not (np.isfinite(self.scale) and self.scale > 0):
            raise InvalidValue(f"the scale of a limit must be a positive number; got {self.scale!r}")


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


class LimitNotMet(UserWarning):
    """Issued by ``optimise`` when the design it returns does not meet one of its limits; the message names
    each such limit with its value and its bound. The result is returned all the same (see
    ``Result.limits_met``)."""


@dataclass(frozen=True)
class Result:
    """The design variables and density after the last pass, the record of every pass, and the number of
    threads the solves ran on. ``density`` is the output of ``filters``; with named designs, ``designs``
    maps each name to that design's field after the last pass (empty without named designs).

    Each record holds the values at the start of its pass, so the last recorded goal belongs to the design
    before the last update: one update behind ``x``, ``density`` and ``designs``. ``goal`` and ``limits``
    belong to the returned design itself: ``optimise`` evaluates it once more after the last update, each
    value on the design it reads, the limits in the order of the ``limits`` argument. ``limits_met`` says
    for each limit whether ``(value - at_most) / size`` is at most the run's ``limit_tolerance``, where
    ``size`` is ``|at_most|`` (1 when ``at_most`` is 0); ``feasible`` is true when every limit is met.

    ``stop_reason`` is ``"passes"`` when the run made the passes it was given, ``"change"`` when it stopped
    on ``stop_change``, and ``"error"`` in the ``partial_result`` of an ``EngineError``; there ``goal``,
    ``limits`` and ``limits_met`` are ``None`` when the returned design could not be evaluated."""

    x: np.ndarray
    density: np.ndarray
    passes: tuple[Pass, ...]
    threads: int = 1
    designs: Mapping[str, np.ndarray] = field(default_factory=dict)
    goal: float | None = None
    limits: tuple[float, ...] | None = None
    limits_met: tuple[bool, ...] | None = None
    stop_reason: str = "passes"

    @property
    def feasible(self) -> bool:
        """Every limit met: true without limits, false when the limits could not be evaluated."""
        return self.limits_met is not None and all(self.limits_met)


def limit_met(value: float, at_most: float, tolerance: float) -> bool:
    """Whether ``value`` meets the bound ``at_most``: ``(value - at_most) / size <= tolerance``, with
    ``size`` = ``|at_most|``, or 1 when ``at_most`` is 0. The limit's ``scale`` plays no part, so a limit
    with a large scale is judged by its bound alone."""
    size = abs(at_most) if at_most != 0 else 1.0
    return bool((value - at_most) / size <= tolerance)


def _unit_text(limit: Limit) -> str:
    unit = limit.response.unit
    return "" if unit in ("", "1") else f" {unit}"


def _design_text(limit: Limit) -> str:
    design = getattr(limit.response, "design", None)
    return "" if design is None else f" on the design {design!r}"


def _refuse_impossible_bounds(problem, limits, limit_parts, filters, chains) -> None:
    # Each response that can tell a bound no design meets is asked before the first pass.
    for index, (limit, (design, response)) in enumerate(zip(limits, limit_parts)):
        check = getattr(response, "impossible_bound", None)
        if check is None:
            continue
        reason = check(problem, float(limit.at_most), (*filters, *chains[design]))
        if reason is not None:
            raise InvalidValue(
                f"limit {index} ({limit.response.name}{_design_text(limit)}, at most {limit.at_most:.6g}"
                f"{_unit_text(limit)}) can never be met: {reason}",
                {"limit": index, "at_most": float(limit.at_most)},
            )


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
    limit_tolerance: float = 1e-3,
) -> Result:
    """Minimises ``goal`` subject to ``limits``, starting from ``x`` (one value per element, in [0, 1]).

    Runs ``passes`` passes, or fewer when ``stop_change`` is given and the largest change of a design
    variable in one pass falls to it or below. Elements outside the part keep x = 0. After the last update
    the returned design is evaluated once more (values only, the solves started from the states of the
    last pass), so that ``Result.goal`` and ``Result.limits`` are its own values.

    ``designs``, if given, maps a name to the filters that make that design from the output of
    ``filters``; the goal and every limit must then name the design they read, with
    ``michell.on(name, response)``. Only the designs a goal or a limit reads are evaluated, and a design
    is solved only for the physics its responses read.

    ``limit_tolerance`` (at least 0) is the relative excess over its bound that a limit may have in the
    returned design and still count as met: ``(value - at_most) / size <= limit_tolerance``, with ``size``
    = ``|at_most|`` (1 when ``at_most`` is 0), not the limit's ``scale``. When a limit is not met the result
    is returned all the same, and a ``michell.LimitNotMet`` warning names each limit missed.

    A limit whose response refuses its bound (``impossible_bound``, see ``Limit``) and any other argument
    refused before the first pass raise ``EngineError`` with code ``invalid_input`` and ``partial_result``
    ``None``. An ``EngineError`` raised on the way (by a solve, a filter or the optimiser) leaves
    ``optimise`` with the attribute ``partial_result``: the ``Result`` of the passes completed before the
    error, that is ``x`` after the last completed update, its ``density`` and ``designs``, the records of
    those passes (the starting ``x`` and no records if no pass was completed) and ``stop_reason``
    ``"error"``, with the goal and limit values of that ``x`` when they could be measured (``None``
    otherwise; an ``x`` whose evaluation raised is not evaluated again). Past the refused arguments it is
    ``None`` only when the filters cannot be applied to that ``x`` either. The same holds when the
    evaluation after the last update raises.
    """
    try:
        try:
            tolerance_ok = bool(np.isfinite(limit_tolerance) and limit_tolerance >= 0)
        except TypeError:
            tolerance_ok = False
        if not tolerance_ok:
            raise InvalidValue(f"limit_tolerance must be a finite number at least 0; got {limit_tolerance!r}")
        chains = design_chains(designs)
        goal_design, goal_response = design_of(goal, chains)
        limit_parts = [design_of(limit.response, chains) for limit in limits]
        _refuse_impossible_bounds(problem, limits, limit_parts, filters, chains)
    except EngineError as error:
        error.partial_result = None
        raise
    x = np.where(problem.grid.active == 1.0, np.asarray(x, dtype=np.float64), 0.0)
    reads = reads_by_design([(goal_design, goal_response), *limit_parts], chains)
    history = []
    states = None  # per design, the warm start of the next solves
    measured = None  # the goal and limit values of the current x, once measured
    tried = False  # whether an evaluation of the current x was begun
    stop_reason = "passes"

    def values_of(evaluated) -> tuple[float, list[float]]:
        goal_value = goal_response.value(evaluated[goal_design][0])
        limit_values = [response.value(evaluated[design][0]) for design, response in limit_parts]
        return goal_value, limit_values

    def result(reason: str) -> Result:
        # The result of the passes completed so far: `x` after the last completed update, its fields and,
        # once measured, its goal and limit values.
        density = apply_filters(x, filters)[-1]
        final = {} if designs is None else {name: apply_filters(density, chain)[-1] for name, chain in chains.items()}
        goal_value = limit_values = met = None
        if measured is not None:
            goal_value = float(measured[0])
            limit_values = tuple(float(value) for value in measured[1])
            met = tuple(
                limit_met(value, float(limit.at_most), limit_tolerance) for value, limit in zip(limit_values, limits)
            )
        return Result(
            x=x,
            density=density,
            passes=tuple(history),
            threads=problem.threads,
            designs=final,
            goal=goal_value,
            limits=limit_values,
            limits_met=met,
            stop_reason=reason,
        )

    try:
        optimiser.start(problem, x, limits, filters)
        for number in range(1, passes + 1):
            tried = True
            base_fields, evaluated = evaluate_designs(problem, x, filters, chains, reads, start=states)
            steps = {design: [ev.solver_steps] for design, (ev, _) in evaluated.items()}

            def gradient(design, response):
                return design_gradient(
                    problem, response, design, filters, chains, base_fields, evaluated, steps[design]
                )

            goal_value, limit_values = values_of(evaluated)
            measured = (goal_value, limit_values)
            d_goal = gradient(goal_design, goal_response)
            d_limits = [gradient(design, response) for design, response in limit_parts]

            x_new = optimiser.update(x, goal_value, d_goal, limit_values, d_limits)
            change = float(np.abs(x_new - x).max())
            solved = {design: sum(counts) for design, counts in steps.items() if reads[design]}
            by_design = {} if designs is None else solved
            total_steps = sum(sum(counts) for counts in steps.values())
            history.append(Pass(number, goal_value, tuple(limit_values), change, total_steps, by_design))
            x, measured, tried = x_new, None, False
            states = {design: ev.states for design, (ev, _) in evaluated.items()}
            if stop_change is not None and change <= stop_change:
                stop_reason = "change"
                break
        # The returned design, evaluated once more: values only, no gradients.
        tried = True
        _, evaluated = evaluate_designs(problem, x, filters, chains, reads, start=states)
        measured = values_of(evaluated)
    except EngineError as error:
        if measured is None and not tried:
            # The error came before this x was evaluated (in the optimiser's start): try it once. An x whose
            # evaluation was begun is not evaluated again, since the same call would raise the same error.
            try:
                measured = values_of(evaluate_designs(problem, x, filters, chains, reads, start=states)[1])
            except EngineError:
                measured = None
        try:
            error.partial_result = result("error")
        except EngineError:
            error.partial_result = None
        raise
    outcome = result(stop_reason)
    missed = [
        f"{limit.response.name}{_design_text(limit)} = {value:.6g}{_unit_text(limit)}, at most "
        f"{limit.at_most:.6g}{_unit_text(limit)}"
        for limit, value, met in zip(limits, outcome.limits, outcome.limits_met)
        if not met
    ]
    if missed:
        warnings.warn(
            LimitNotMet(
                f"the returned design does not meet {len(missed)} of its {len(limits)} limits "
                f"(limit_tolerance {limit_tolerance:g}): " + "; ".join(missed)
            ),
            stacklevel=2,
        )
    return outcome
