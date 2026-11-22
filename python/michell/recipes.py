"""Procedures written out in full: a whole optimisation from one call.

``stiff_part`` gives the stiffest part of a given mass with a minimum member size, in two steps: the robust
formulation of Wang, Lazarov and Sigmund ("On projection methods, convergence and robust formulations in
topology optimization", Struct. Multidisc. Optim. 43, 2011) forms the layout, and the member-size limit of
Zhou, Lazarov, Wang and Sigmund ("Minimum length scale in topology optimization by geometric constraints",
Comput. Methods Appl. Mech. Engrg. 293 (2015) 266-282) then reshapes its members. Every piece it uses is a
public part of the package (``DensityFilter``, ``KeepSolid``, ``robust_designs``, ``Projection``,
``MemberSize``, ``member_threshold``, ``MovingAsymptotes``, ``optimise``); the function only fixes the order
and the settings, so the same run can be written out by hand from its docstring.

``member_size_bound`` is the bound eps of the member-size limit read off figure 8 of Zhou et al. (2015).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from michell._engine import EngineError, dot
from michell.designs import on, robust_designs
from michell.errors import InvalidType, InvalidValue
from michell.filters import DensityFilter, KeepSolid, Projection
from michell.member_size import MemberSize, member_threshold
from michell.mma import MovingAsymptotes
from michell.optimise import Limit, LimitNotMet, Result, limit_met, optimise
from michell.problem import Problem
from michell.responses import Compliance, Mass

# eps of the member-size limit read from figure 8 of Zhou et al. (2015) for the filter radius r in elements
# (with c = r^4): the figure's markers on its logarithmic axis, read by interpolating between the decades
# in the drawing's own coordinates. Above r = 10, the last marker, `member_size_bound` keeps the value at 10.
EPS_OF_FIGURE_8 = {3: 1e-6, 4: 3e-7, 5: 1e-7, 6: 7.5e-8, 7: 6e-8, 8: 5.5e-8, 9: 5e-8, 10: 5e-8}

# The default settings of `stiff_part`: (beta, passes) of the first step, 100 passes in all.
SCHEDULE = ((1.0, 21), (2.0, 14), (4.0, 14), (8.0, 14), (16.0, 18), (32.0, 19))
SPREAD = 0.25
PASSES_PER_BOUND = 10
SECOND_STEP_PASSES = 40
SECOND_STEP_BETA = 32.0
MOVE_LIMIT = 0.5
SECOND_STEP_MOVE_LIMIT = 0.1
SECOND_STEP_RETURNS = "best"
RETURNS = ("last", "best")
# The tolerance of `optimise`'s test of a limit (its default `limit_tolerance`, which `stiff_part` keeps).
LIMIT_TOLERANCE = 1e-3

# Below this member size in element sides, `stiff_part` warns (`GridTooCoarse`).
COARSE_BELOW_ELEMENTS = 5

ROBUST_STEP = "robust"
MEMBER_SIZE_STEP = "member size"


def member_size_bound(radius_in_elements: float) -> float:
    """The bound eps of the member-size limit for a filter radius of ``radius_in_elements`` element sides,
    read off figure 8 of Zhou, Lazarov, Wang and Sigmund (2015) (``EPS_OF_FIGURE_8``, with c = r^4): at
    r = 3, 4, 5, 6, 7, 8, 9, 10 the values 1e-6, 3e-7, 1e-7, 7.5e-8, 6e-8, 5.5e-8, 5e-8 and 5e-8, exactly;
    between them linear in log10(eps); below 3 and above 10 the value at 3 or at 10."""
    r = _positive_number(radius_in_elements, "the radius in elements")
    nodes = sorted(EPS_OF_FIGURE_8)
    if r <= nodes[0]:
        return EPS_OF_FIGURE_8[nodes[0]]
    if r >= nodes[-1]:
        return EPS_OF_FIGURE_8[nodes[-1]]
    for low, high in zip(nodes, nodes[1:]):
        if r == low:
            return EPS_OF_FIGURE_8[low]
        if low < r < high:
            share = (r - low) / (high - low)
            log_eps = (1.0 - share) * math.log10(EPS_OF_FIGURE_8[low]) + share * math.log10(EPS_OF_FIGURE_8[high])
            return 10.0**log_eps
    return EPS_OF_FIGURE_8[nodes[-1]]  # not reached


class GridTooCoarse(UserWarning):
    """Issued by ``stiff_part`` when the member size is fewer than five element sides (``size_mm / h < 5``):
    on such a grid the member-size limit makes members thicker than asked. The message gives the member size
    in elements and the element side that removes the effect. The run proceeds. The ratio is compared with a
    relative tolerance of 1e-9, so that five elements given in decimal millimetres count as five."""


@dataclass(frozen=True)
class RecipePass:
    """The record of one pass of ``stiff_part``: the step (``"robust"`` or ``"member size"``), the number of
    the pass counted through that step from 1, the sharpness ``beta`` of the projections, the goal and the
    limit values at the start of the pass (each on the design it reads, as in ``michell.Pass``), the limits'
    bounds in that pass, and the largest change of a design variable made by its update.

    In the robust step the goal is the compliance of the eroded design (N mm) and the one limit is the mass
    of the dilated design (g); in the member-size step the goal is the compliance of the nominal design and
    the limits are its mass and the member-size value."""

    step: str
    number: int
    beta: float
    goal: float
    limits: tuple[float, ...]
    bounds: tuple[float, ...]
    change: float


@dataclass(frozen=True)
class StiffPart:
    """What ``stiff_part`` returns.

    - ``x``: the design variables of the returned design (by default those after the last pass, see
      ``returned_pass``); ``filtered``: its filtered field (the output of the filters, ``KeepSolid`` first and
      the density filter last); ``nominal``: the design, the projection of ``filtered`` at beta 32 and the
      threshold 0.5 with the held elements solid.
    - ``result``: the ``Result`` of the member-size step, with ``goal`` (the compliance of ``nominal``),
      ``limits`` (its mass and member-size value), ``limits_met``, ``stop_reason``, and ``x``, ``density`` and
      ``designs`` of the returned design; its ``passes`` are the records of every pass of the step, whichever
      design is returned.
    - ``passes``: the record of every pass of both steps, in order (``RecipePass``).
    - ``returned_pass``: which design of the member-size step is returned: ``k`` is the design that pass ``k``
      of the step started from (1: the design at the end of the robust step), its values those of the record
      of pass ``k``; ``second_step_passes + 1`` is the design after the last update (always so with
      ``second_step_returns="last"``).
    - ``first_step_nominal``: the nominal design at the end of the robust step, at beta 32 like ``nominal``,
      so that what the member-size step changed can be measured.
    - ``member_size``: the ``MemberSize`` response of the second step, with its ``threshold`` (the value a
      member of the asked size reaches, ``member_threshold(filter, size)``), its per-element thresholds
      ``thresholds`` and its constant ``c``; ``threshold`` repeats the first.
    - ``eps``: the bound of the member-size limit; ``value_at_switch``: the limit's value when it was
      switched on; ``scale``: the limit's scale, ``max(value_at_switch, eps)``.
    """

    x: np.ndarray
    filtered: np.ndarray
    nominal: np.ndarray
    result: Result
    passes: tuple[RecipePass, ...]
    first_step_nominal: np.ndarray
    member_size: MemberSize
    threshold: float
    eps: float
    value_at_switch: float
    scale: float
    returned_pass: int


class _Watched:
    """Hands every call to a fresh ``MovingAsymptotes`` and reports each completed update: the goal and the
    limit values it was given and the largest change it made, the numbers ``optimise`` records in its
    ``Pass``. ``look``, if given, is shown every design before its update, with its goal and limit values."""

    name = "MMA"

    def __init__(
        self,
        report: Callable[[float, tuple[float, ...], float], None],
        move_limit: float,
        look: Callable[[np.ndarray, float, tuple[float, ...]], None] | None = None,
    ):
        self._inner = MovingAsymptotes(move_limit)
        self._report = report
        self._look = look

    def start(self, problem, x, limits, filters):
        self._inner.start(problem, x, limits, filters)

    def update(self, x, goal, d_goal, limits, d_limits):
        if self._look is not None:
            self._look(x, float(goal), tuple(float(value) for value in limits))
        x_new = self._inner.update(x, goal, d_goal, limits, d_limits)
        self._report(float(goal), tuple(float(value) for value in limits), float(np.abs(x_new - x).max()))
        return x_new


def _whole_number(value, name: str, at_least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidType(f"{name} must be a whole number; got {value!r}")
    if value < at_least:
        raise InvalidValue(f"{name} must be at least {at_least}; got {value}")
    return int(value)


def _positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidType(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise InvalidValue(f"{name} must be a finite number above 0; got {value!r}")
    return number


def _schedule(schedule) -> tuple[tuple[float, int], ...]:
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise InvalidValue(f"the schedule must be a non-empty list of (beta, passes) pairs; got {schedule!r}")
    out = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise InvalidValue(f"each entry of the schedule is a pair (beta, passes); got {entry!r}")
        out.append((_positive_number(entry[0], "beta"), _whole_number(entry[1], "the passes of a beta", 1)))
    return tuple(out)


def stiff_part(
    problem: Problem,
    mass_g: float,
    size_mm: float,
    keep_solid: KeepSolid | None = None,
    filters: Sequence = (),
    *,
    outside="closed",
    inactive="closed",
    schedule: Sequence[tuple[float, int]] = SCHEDULE,
    spread: float = SPREAD,
    passes_per_bound: int = PASSES_PER_BOUND,
    second_step_passes: int = SECOND_STEP_PASSES,
    move_limit: float = MOVE_LIMIT,
    second_step_move_limit: float = SECOND_STEP_MOVE_LIMIT,
    second_step_returns: str = SECOND_STEP_RETURNS,
    eps: float | None = None,
    callback: Callable[[str, int, RecipePass], None] | None = None,
) -> StiffPart:
    """The stiffest part of mass ``mass_g`` (g) with members at least ``size_mm`` (mm) thick: least
    compliance (the weighted sum over the problem's load cases), by MMA, in two steps.

    Arguments: ``problem`` (a ``Problem`` on a 2D or a 3D grid); ``mass_g``, the asked mass, above the mass
    of the held elements and at most that of the whole part; ``size_mm``, the minimum member size, which is
    also the radius of the density filter; ``keep_solid``, the elements held solid (a ``KeepSolid``, or
    ``None``); ``filters``, filters of the caller's own (a symmetry, say) applied after ``KeepSolid`` and
    before the density filter; ``outside`` and ``inactive``, what the density filter sees beyond each face
    of the grid and in the inactive elements (``DensityFilter``: a free face of the part should be
    ``"empty"``, a symmetry plane or a face that continues into a support ``"closed"``).

    The filters from x to the filtered field are ``keep_solid`` (if given), then ``filters``, then
    ``DensityFilter(grid, size_mm, problem.threads, outside=outside, inactive=inactive)``: the cone filter
    with its radius R equal to the member size, the set-up of Zhou et al. (2015). The procedure:

    1. Start from a uniform x whose mass, taken as a density, is the asked mass, with the held elements at 1.
    2. The robust step (Wang, Lazarov and Sigmund 2011; ``michell.designs``). For each ``(beta, passes)``
       of ``schedule``, in calls of ``optimise`` of at most ``passes_per_bound`` passes each: the designs
       of ``robust_designs(beta, spread)``, each projection followed by ``keep_solid``; the nominal and the
       dilated mass of the current x; the bound on the mass of the dilated design set to
       ``mass_g * dilated / nominal``, so that the nominal design has the asked mass; then the compliance
       of the eroded design minimised under that bound. Between calls only the bound changes.
    3. The member-size step (Zhou et al. 2015, section 3: the limit is switched on once a layout has
       formed, at a high, fixed sharpness). The nominal design is the projection at beta 32 and 0.5,
       followed by ``keep_solid``; ``MemberSize(density_filter, projection, member_threshold(density_filter,
       size_mm), keep_solid=keep_solid)`` reads the filtered field (a design with no filters of its own);
       its scale is its value at the switch, or ``eps`` if that is larger (``michell.member_size``). One
       call of ``optimise`` for ``second_step_passes`` passes minimises the compliance of the nominal design
       under its mass (at most ``mass_g``) and the member-size limit (at most ``eps``).
    4. The returned design (``second_step_returns``): ``"last"``, the design after the last update; or
       ``"best"``, the design of lowest compliance among those of the member-size step that meet both
       limits by ``optimise``'s own test (``limit_met`` with its default ``limit_tolerance``, 1e-3), each
       judged on the values of its own pass record: the design each pass started from (the first is the
       design at the end of the robust step) and the design after the last update. When that is not the
       last design it is evaluated once more (``optimise`` with no pass), so that ``result`` describes it;
       when no design meets both limits the last one is returned, with ``LimitNotMet`` as with ``"last"``.

    Settings: ``schedule``, the sharpness of the projections and the passes at each, by default beta
    doubling from 1 to 32 with 21, 14, 14, 14, 18 and 19 passes (100 in all); ``spread``, the thresholds of
    the eroded and dilated designs are 0.5 + spread and 0.5 - spread (0.25, as in
    ``examples/cantilever_3d.py``); ``passes_per_bound``, how often the bound on the dilated mass is set
    again (every 10 passes); ``second_step_passes``, 40; ``move_limit`` and ``second_step_move_limit``,
    MMA's move limit in the robust step and in the member-size step (``MovingAsymptotes``: the largest change
    of a design variable in one pass), 0.5 and 0.1; ``eps``, by default ``member_size_bound(size_mm / h)`` for
    the element side h, figure 8 of Zhou et al. (2015) for the filter radius in elements;
    ``second_step_returns``, ``"best"``.

    Why the member-size step moves by at most 0.1 and returns its best design: with MMA's usual move limit of
    0.5 the step at a fine grid (746 496 elements, members of 10.8 element sides) diverged every few passes, its
    compliance rising from 25 to millions, and ended over its member-size bound; with 0.1 it is stable there and
    meets the bound. The step then moves about the bound, so the design it ends on may lie just over it while
    one a few passes earlier meets it; returning the best design that meets both limits makes the result
    feasible whenever any design of the step is. The sources of pyMOTO and of TopOpt in PETSc state move limits
    of 0.1 and 0.2.

    ``callback``, if given, is called after every pass as ``callback(step, number, record)``, with the
    step (``"robust"`` or ``"member size"``), the number of the pass counted through the step from 1, and
    its ``RecipePass``. Nothing is printed and nothing is kept between calls.

    Limits of the procedure, to know before relying on it:

    - Every call of ``optimise`` starts MMA afresh (its asymptotes and its goal scale), so the robust step
      restarts the optimiser every ``passes_per_bound`` passes and at every change of beta.
    - The robust step's mass limit sits on the dilated design, with its bound set between calls only; those
      calls may end a little off it without warning. Only the member-size step's limits are judged: when
      the returned design misses one, ``optimise`` warns (``LimitNotMet``) and ``result.limits_met`` says so.
    - The member-size limit reshapes members of the layout the robust step formed; where the asked size
      cannot be reached without changing the layout it does not get there (``michell.member_size``). On a
      grid where the member size is fewer than about five elements it makes members thicker than asked:
      its threshold is that of the continuous filter, which a member exactly the asked size thick reaches
      only on a fine grid (``member_threshold``). ``stiff_part`` then warns (``GridTooCoarse``, below five
      elements) and runs all the same.

    Refused arguments raise ``EngineError`` with code ``invalid_input`` (``michell.errors``) before any
    pass. An ``EngineError`` raised during a step leaves the function with the ``partial_result`` that
    ``optimise`` gives it (the ``Result`` of that call's completed passes, or ``None`` when the call was
    refused before its first pass; an error between two calls, while a bound or the limit's value at the
    switch is computed, carries the ``Result`` of the call before) and two attributes: ``step``, the step,
    and ``pass_number``, the number in that step of the pass at which it happened (one past the last
    completed pass).
    """
    if not isinstance(problem, Problem):
        raise InvalidType(f"stiff_part takes a michell.Problem; got {type(problem).__name__}")
    grid = problem.grid
    mass_g = _positive_number(mass_g, "the asked mass")
    size_mm = _positive_number(size_mm, "the member size")
    if keep_solid is not None and not isinstance(keep_solid, KeepSolid):
        raise InvalidType(f"keep_solid takes a michell.KeepSolid or None; got {type(keep_solid).__name__}")
    if not isinstance(filters, (list, tuple)):
        raise InvalidType(f"filters must be a list of filters; got {type(filters).__name__}")
    for extra in filters:
        if not (callable(getattr(extra, "forward", None)) and callable(getattr(extra, "backward", None))):
            raise InvalidType(f"a filter needs forward and backward; got {type(extra).__name__}")
    schedule = _schedule(schedule)
    if isinstance(spread, bool) or not isinstance(spread, numbers.Real):
        raise InvalidType(f"the spread must be a number; got {spread!r}")
    spread = float(spread)
    if not 0.0 < spread < 0.5:
        raise InvalidValue(f"the spread of the thresholds must lie between 0 and 0.5; got {spread}")
    passes_per_bound = _whole_number(passes_per_bound, "passes_per_bound", 1)
    second_step_passes = _whole_number(second_step_passes, "second_step_passes", 1)
    move_limit = MovingAsymptotes(move_limit).move_limit
    second_step_move_limit = MovingAsymptotes(second_step_move_limit).move_limit
    if not isinstance(second_step_returns, str) or second_step_returns not in RETURNS:
        raise InvalidValue(f"second_step_returns must be 'last' or 'best'; got {second_step_returns!r}")
    if callback is not None and not callable(callback):
        raise InvalidType(f"the callback must be callable; got {type(callback).__name__}")

    active = np.asarray(grid.active) == 1.0
    held = np.zeros(grid.n_elements, dtype=bool)
    if keep_solid is not None:
        if keep_solid.mask.size != grid.n_elements:
            raise InvalidValue(
                f"keep_solid needs {grid.n_elements} entries, one per element (got {keep_solid.mask.size})"
            )
        if (keep_solid.mask & ~active).any():
            raise InvalidValue("keep_solid marks inactive elements; only elements of the part can be held solid")
        held = np.asarray(keep_solid.mask, dtype=bool)
    design = active & ~held
    element_mass = float(problem.material.mass_g(grid.voxel_volume_mm3))
    if not design.any():
        raise InvalidValue("the part has no design elements: every active element is held solid")
    fraction = (mass_g / element_mass - int(held.sum())) / int(design.sum())
    if not 0.0 < fraction <= 1.0:
        held_mass, part_mass = element_mass * int(held.sum()), element_mass * int(active.sum())
        raise InvalidValue(
            f"the asked mass {mass_g:.6g} g must lie above the mass of the held elements ({held_mass:.6g} g) "
            f"and at most at the mass of the whole part ({part_mass:.6g} g)"
        )

    density_filter = DensityFilter(grid, size_mm, problem.threads, outside=outside, inactive=inactive)
    if eps is None:
        eps = member_size_bound(size_mm / float(density_filter.element_size))
    eps = _positive_number(eps, "eps")
    h = float(grid.h)
    if size_mm / h < COARSE_BELOW_ELEMENTS * (1.0 - 1e-9):  # 0.35 mm / 0.07 mm gives 4.999999999999999
        warnings.warn(
            f"the member size {size_mm:g} mm is {size_mm / h:.3g} element sides of {h:g} mm, fewer than "
            f"{COARSE_BELOW_ELEMENTS}: on such a grid the member-size limit makes members thicker than asked; a "
            f"finer grid, with elements of at most {size_mm / COARSE_BELOW_ELEMENTS:g} mm, removes the effect",
            GridTooCoarse,
            stacklevel=2,
        )
    chain = [*([keep_solid] if keep_solid is not None else []), *filters, density_filter]
    tail = [keep_solid] if keep_solid is not None else []
    mass = Mass()
    records: list[RecipePass] = []

    def filtered_of(x: np.ndarray) -> np.ndarray:
        y = x
        for f in chain:
            y = f.forward(y)
        return y

    def mass_of(field: np.ndarray, design_chain) -> float:
        for f in design_chain:
            field = f.forward(field)
        return element_mass * dot(np.asarray(grid.active, dtype=np.float64), np.ascontiguousarray(field))

    def run(step: str, beta: float, bounds: tuple[float, ...], move: float, look=None, **kwargs) -> Result:
        def report(goal: float, limits: tuple[float, ...], change: float) -> None:
            number = sum(1 for record in records if record.step == step) + 1
            record = RecipePass(step, number, beta, goal, limits, bounds, change)
            records.append(record)
            if callback is not None:
                callback(step, number, record)

        try:
            return optimise(problem, filters=chain, optimiser=_Watched(report, move, look), **kwargs)
        except EngineError as error:
            if not hasattr(error, "partial_result"):
                error.partial_result = None
            error.step = step
            error.pass_number = sum(1 for record in records if record.step == step) + 1
            raise

    # Step 1: the robust set, the bound on the dilated mass set again before every call.
    x = np.where(design, fraction, np.where(held, 1.0, 0.0))
    result = None
    for beta, passes in schedule:
        left = passes
        while left > 0:
            n = min(passes_per_bound, left)
            designs = {name: [projection, *tail] for name, (projection,) in robust_designs(beta, spread).items()}
            try:
                base = filtered_of(x)
                bound = mass_g * mass_of(base, designs["dilated"]) / mass_of(base, designs["nominal"])
            except EngineError as error:
                error.partial_result = result
                error.step = ROBUST_STEP
                error.pass_number = sum(1 for record in records if record.step == ROBUST_STEP) + 1
                raise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LimitNotMet)
                result = run(
                    ROBUST_STEP, beta, (bound,), move_limit, goal=on("eroded", Compliance()),
                    limits=[Limit(on("dilated", mass), at_most=bound)], x=x, passes=n, designs=designs,
                )  # fmt: skip
            x = result.x
            left -= n
    # Step 2: the member-size limit on the filtered field, at beta 32, beside the nominal design.
    projection = Projection(SECOND_STEP_BETA, 0.5)
    threshold = member_threshold(density_filter, size_mm)
    member_size = MemberSize(density_filter, projection, threshold, keep_solid=keep_solid)
    try:
        first_step_filtered = filtered_of(x)
        first_step_nominal = projection.forward(first_step_filtered)
        for f in tail:
            first_step_nominal = f.forward(first_step_nominal)
        value_at_switch = float(member_size.evaluate(grid, first_step_filtered)[0])
    except EngineError as error:
        error.partial_result = result
        error.step = MEMBER_SIZE_STEP
        error.pass_number = 1
        raise
    scale = max(value_at_switch, eps)
    second_limits = [
        Limit(on("nominal", mass), at_most=mass_g),
        Limit(on("filtered", member_size), at_most=eps, scale=scale),
    ]
    second = {"goal": on("nominal", Compliance()), "limits": second_limits,
              "designs": {"filtered": [], "nominal": [projection, *tail]}}  # fmt: skip
    returned_pass = second_step_passes + 1
    if second_step_returns == "last":
        result = run(
            MEMBER_SIZE_STEP, SECOND_STEP_BETA, (mass_g, eps), second_step_move_limit, x=x, passes=second_step_passes,
            **second,
        )  # fmt: skip
    else:
        best: list = []  # [compliance, pass number, design variables] of the best design that meets both limits

        def look(x_pass: np.ndarray, goal: float, values: tuple[float, ...]) -> None:
            number = sum(1 for record in records if record.step == MEMBER_SIZE_STEP) + 1
            if _meets(values, (mass_g, eps)) and (not best or goal < best[0]):
                best[:] = [goal, number, np.array(x_pass, copy=True)]

        # The step's own LimitNotMet is held back and given again below only if the last design is returned.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LimitNotMet)
            result = run(
                MEMBER_SIZE_STEP, SECOND_STEP_BETA, (mass_g, eps), second_step_move_limit, look, x=x,
                passes=second_step_passes, **second,
            )  # fmt: skip
        last_is_best = result.feasible and (not best or result.goal <= best[0])
        if best and not last_is_best:
            returned_pass = best[1]
            try:
                again = optimise(
                    problem, filters=chain, optimiser=MovingAsymptotes(second_step_move_limit), x=best[2], passes=0,
                    **second,
                )  # fmt: skip
            except EngineError as error:
                error.partial_result = result
                error.step = MEMBER_SIZE_STEP
                error.pass_number = second_step_passes + 1
                raise
            result = dataclasses.replace(again, passes=result.passes, stop_reason=result.stop_reason)
        for caught_warning in caught:
            if returned_pass == second_step_passes + 1 or not issubclass(caught_warning.category, LimitNotMet):
                warnings.warn_explicit(
                    caught_warning.message, caught_warning.category, caught_warning.filename, caught_warning.lineno
                )
    return StiffPart(
        x=result.x,
        filtered=result.designs["filtered"],
        nominal=result.designs["nominal"],
        result=result,
        passes=tuple(records),
        first_step_nominal=first_step_nominal,
        member_size=member_size,
        threshold=float(member_size.threshold),
        eps=eps,
        value_at_switch=value_at_switch,
        scale=scale,
        returned_pass=returned_pass,
    )


def _meets(values: Sequence[float], bounds: Sequence[float]) -> bool:
    """Whether every value meets its bound by ``optimise``'s own test, with its default tolerance."""
    return all(limit_met(value, bound, LIMIT_TOLERANCE) for value, bound in zip(values, bounds))


__all__ = [
    "COARSE_BELOW_ELEMENTS",
    "EPS_OF_FIGURE_8",
    "MEMBER_SIZE_STEP",
    "MOVE_LIMIT",
    "PASSES_PER_BOUND",
    "ROBUST_STEP",
    "SCHEDULE",
    "SECOND_STEP_BETA",
    "SECOND_STEP_MOVE_LIMIT",
    "SECOND_STEP_PASSES",
    "SECOND_STEP_RETURNS",
    "SPREAD",
    "GridTooCoarse",
    "RecipePass",
    "StiffPart",
    "member_size_bound",
    "stiff_part",
]
