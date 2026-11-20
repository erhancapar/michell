"""Verification rows 9 and 10: the method of moving asymptotes on two published problems and on the MBB beam.

Each row's limit, its measure and its settings were fixed before any result was computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from michell._engine import Mma
from michell.filters import DensityFilter
from michell.mma import MovingAsymptotes
from michell.oc import OptimalityCriteria
from michell.optimise import Limit, optimise
from michell.responses import Compliance, Mass
from michell.verification.mbb import REFERENCE_NOT_FOUND, mbb_problem, read_reference
from michell.verification.row import PUBLISHED_OPTIMUM, REFERENCE_CODE, Row, checked, g3, not_run

# ---------------------------------------------------------------------------------------------------------
# Row 9. Limit: on two small problems with published optima (Svanberg 1987: the five-variable cantilever
# beam and the two-bar truss), each also solved independently with SciPy, MMA reaches the optimum within
# 0.1 % of the objective (against the published value and against SciPy's) and satisfies every limit to 1e-6.
#
# Settings (fixed with the limit): MMA (`michell.Mma`, the engine object, on the problem as published) from
# the published start, until the largest change of a variable in one update is 1e-8 or less, at most 200
# updates. SciPy: SLSQP with ftol 1e-12 from the published start and from two other starts; the best
# successful run that meets the limits to 1e-9 is the reference. A limit is measured as f_i(x) - 1, in the
# published scaling (every limit of both problems reads "... <= 1").
#
# The published optima carry the digits the paper prints: the cantilever's 1.340 to four digits, the
# truss's 1.51 to three, so that comparison can only be as fine as the rounding (+-0.33 % for 1.51).


class Cantilever:
    """Svanberg (1987): a cantilever of five hollow square segments; x_j is the outer side of segment j.
    Weight 0.0624 sum x_j; one limit on the tip deflection, 61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3
    <= 1."""

    name = "cantilever"
    c1, coefficients = 0.0624, np.array([61.0, 37.0, 19.0, 7.0, 1.0])
    x_min, x_max, start = np.full(5, 1.0), np.full(5, 10.0), np.full(5, 5.0)
    other_starts = (np.full(5, 3.0), np.full(5, 8.0))
    published_x = np.array([6.016, 5.309, 4.494, 3.502, 2.153])
    published_goal, published_text = 1.340, "1.340"

    def goal(self, x):
        return self.c1 * x.sum(), np.full(5, self.c1)

    def limits(self, x):
        """The limits as f_i(x) - 1 (<= 0) and their gradients, one row per limit."""
        value = (self.coefficients / x**3).sum() - 1.0
        return np.array([value]), (-3.0 * self.coefficients / x**4).reshape(1, 5)


class TwoBarTruss:
    """Svanberg (1987): a symmetric two-bar truss under an oblique load; x1 is the bars' cross-section,
    x2 half the distance between the supports. Weight c1 x1 sqrt(1 + x2^2); the stresses of the two bars,
    c2 sqrt(1 + x2^2) (8/x1 +- 1/(x1 x2)), at most 1; c1 = 1, c2 = 0.124."""

    name = "two-bar truss"
    c1, c2 = 1.0, 0.124
    x_min, x_max, start = np.array([0.2, 0.1]), np.array([4.0, 1.6]), np.array([1.5, 0.5])
    other_starts = (np.array([1.0, 1.0]), np.array([3.0, 0.3]))
    published_x = np.array([1.41, 0.38])
    published_goal, published_text = 1.51, "1.51"

    def goal(self, x):
        root = np.sqrt(1.0 + x[1] ** 2)
        return self.c1 * x[0] * root, np.array([self.c1 * root, self.c1 * x[0] * x[1] / root])

    def limits(self, x):
        x1, x2 = x
        root = np.sqrt(1.0 + x2**2)
        values, gradients = [], []
        for sign in (1.0, -1.0):
            bracket = 8.0 / x1 + sign / (x1 * x2)
            values.append(self.c2 * root * bracket - 1.0)
            d_x1 = self.c2 * root * (-8.0 / x1**2 - sign / (x1**2 * x2))
            d_x2 = self.c2 * (x2 / root * bracket + root * (-sign / (x1 * x2**2)))
            gradients.append([d_x1, d_x2])
        return np.array(values), np.array(gradients)


def solve_with_mma(problem) -> tuple[np.ndarray, int]:
    """MMA from the published start; the design and the number of updates."""
    mma = Mma(problem.x_min, problem.x_max, len(problem.limits(problem.start)[0]))
    x = problem.start.copy()
    for _ in range(200):
        goal, d_goal = problem.goal(x)
        limits, d_limits = problem.limits(x)
        x_new = mma.update(x, goal, d_goal, limits, np.ascontiguousarray(d_limits))
        change = np.abs(x_new - x).max()
        x = x_new
        if change <= 1e-8:
            break
    return x, mma.iteration


def solve_with_scipy(problem, minimize) -> float | None:
    """The best objective SLSQP finds from the three starts, or None when no run is feasible."""
    best = None
    for start in (problem.start, *problem.other_starts):
        result = minimize(
            lambda x: problem.goal(x)[0],
            start,
            jac=lambda x: problem.goal(x)[1],
            method="SLSQP",
            bounds=list(zip(problem.x_min, problem.x_max)),
            constraints=[
                {"type": "ineq", "fun": lambda x: -problem.limits(x)[0], "jac": lambda x: -problem.limits(x)[1]}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success and problem.limits(result.x)[0].max() <= 1e-9:
            if best is None or result.fun < best:
                best = float(result.fun)
    return best


def row_9_published_optima() -> list[Row]:
    """Two lines per problem: the objective, and the worst limit."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    rows = []
    for problem in (Cantilever(), TwoBarTruss()):
        goal_title = f"MMA on Svanberg's {problem.name}: objective against the published optimum and SciPy"
        limit_title = f"MMA on Svanberg's {problem.name}: worst limit f_i - 1 at the optimum"
        if minimize is None:
            rows.append(not_run(9, goal_title, PUBLISHED_OPTIMUM, 1e-3, "SciPy not installed"))
            rows.append(not_run(9, limit_title, PUBLISHED_OPTIMUM, 1e-6, "SciPy not installed"))
            continue
        x, updates = solve_with_mma(problem)
        goal = float(problem.goal(x)[0])
        worst_limit = float(problem.limits(x)[0].max())
        scipy_goal = solve_with_scipy(problem, minimize)
        to_published = abs(goal - problem.published_goal) / problem.published_goal
        if scipy_goal is None:
            to_scipy, scipy_text = float("inf"), "SciPy found no feasible optimum"
        else:
            to_scipy = abs(goal - scipy_goal) / scipy_goal
            scipy_text = f"to SciPy's optimum ({g3(to_scipy)})"
        note = (
            f"the larger relative difference, to the published optimum {problem.published_text} "
            f"({g3(to_published)}) and {scipy_text}; {updates} updates"
        )
        rows.append(checked(9, goal_title, PUBLISHED_OPTIMUM, max(to_published, to_scipy), 1e-3, note))
        rows.append(checked(9, limit_title, PUBLISHED_OPTIMUM, worst_limit, 1e-6, "every limit written as f_i <= 1 in the published scaling"))
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 10. Limit: the MBB beam (60 x 20, density filter) with MMA: the volume limit satisfied to 1e-6, and
# the final compliance not above 1.02 times OC's on the same problem after the same number of passes.
#
# Settings (fixed with the limit): the set-up of row 8 (the 88-line code's MBB beam, filter radius 2.4,
# volume fraction 0.5 as a mass limit of 600 g, x starting at 0.5). Both optimisers run for as many passes
# as the reference file has rows (144); OC with move limit 0.2, MMA with the settings of
# `michell.MovingAsymptotes`. The final compliance is the goal recorded in the last pass of each run (as in
# row 8). The volume limit is measured as mass / 600 - 1, on the design of that last pass and on the design
# the run returns; both must be at most 1e-6.


def row_10_mbb_with_mma(reference: Path | str | None) -> list[Row]:
    """Two lines: the compliance ratio MMA / OC, and the volume limit."""
    ratio_title = "MBB beam 60 x 20 with MMA: final compliance over OC's after the same passes"
    volume_title = "MBB beam 60 x 20 with MMA: volume excess mass / bound - 1"
    reference_passes = read_reference(reference, "mbb_60x20_r2.4_density")
    if reference_passes is None:
        return [
            not_run(10, ratio_title, REFERENCE_CODE, 1.02, REFERENCE_NOT_FOUND),
            not_run(10, volume_title, REFERENCE_CODE, 1e-6, REFERENCE_NOT_FOUND),
        ]
    passes = len(reference_passes)
    nx, ny = 60, 20
    problem = mbb_problem(nx, ny)
    bound = 0.5 * nx * ny
    runs = {}
    for name, optimiser in [("OC", OptimalityCriteria(move_limit=0.2)), ("MMA", MovingAsymptotes())]:
        runs[name] = optimise(
            problem,
            goal=Compliance(),
            limits=[Limit(Mass(), at_most=bound)],
            filters=[DensityFilter(problem.grid, 2.4)],
            optimiser=optimiser,
            x=np.full(problem.grid.n_elements, 0.5),
            passes=passes,
        )
    oc, mma = runs["OC"], runs["MMA"]
    if not len(oc.passes) == len(mma.passes) == passes:
        raise RuntimeError(f"the runs made {len(oc.passes)} and {len(mma.passes)} passes, not {passes}")
    ratio = mma.passes[-1].goal / oc.passes[-1].goal
    excess_last_pass = mma.passes[-1].limits[0] / bound - 1.0
    excess_returned = float(mma.density.sum()) / bound - 1.0
    ratio_note = (
        f"{passes} passes each; compliance MMA {g3(mma.passes[-1].goal)} N mm, OC {g3(oc.passes[-1].goal)} N mm"
    )
    volume_note = (
        f"larger of the last pass ({g3(excess_last_pass)}) and the returned design ({g3(excess_returned)})"
    )
    return [
        checked(10, ratio_title, REFERENCE_CODE, ratio, 1.02, ratio_note),
        checked(10, volume_title, REFERENCE_CODE, max(excess_last_pass, excess_returned), 1e-6, volume_note),
    ]
