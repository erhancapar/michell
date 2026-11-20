"""Verification rows 21 to 23: the projection, the gradients of the robust formulation in 3D, and the robust
MBB beam.

Sources: Wang, Lazarov and Sigmund, "On projection methods, convergence and robust formulations in topology
optimization", Struct. Multidisc. Optim. 43 (2011), for the projection, the eroded, nominal and dilated
designs and the rescaling of the mass bound of the dilated design (the rule as stated by Kumar and
Langelaar, Mech. Mach. Theory, 2022, doi:10.1016/j.mechmachtheory.2022.104871); Sigmund,
"Morphology-based black and white filters for topology optimization", Struct. Multidisc. Optim. 33 (2007),
for the measure of non-discreteness.

Each row's measure, set-up and limit were fixed before any result was computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from michell.designs import on, robust_designs
from michell.evaluation import apply_filters
from michell.filters import DensityFilter, Projection
from michell.gradient_check import check_gradient, small_problem_3d
from michell.mma import MovingAsymptotes
from michell.optimise import Limit, Result, optimise
from michell.problem import Problem
from michell.responses import Compliance, Mass
from michell.row import CLOSED_FORM, DESIGN_PROPERTY, GRADIENT, Row, checked, g3
from michell.verification.mbb import mbb_problem

# ---------------------------------------------------------------------------------------------------------
# Row 21. Limits, each on its own line:
# (a) the derivative of the projection (its `backward` applied to a gradient of ones) agrees with central
#     differences of `forward` to a relative 1e-7, for beta in 1, 8, 32 and eta in 0.25, 0.5, 0.75;
# (b) P(0) = 0 and P(1) = 1 within 1e-15 for the same nine projections.
# Measure of (a), as rows 7 and 19: for each projection, max |fd_e - P'_e| / max |P'_e| over the field; the
# largest over the nine is judged. (Per entry the relative difference means nothing at a large beta: far from
# the threshold the numerator of P cancels to exactly 0 on both sides of the step, so the difference is 0
# while P' is a tiny positive number.) Measure of (b): the largest of |P(0)| and |P(1) - 1|.
# Set-up: 1000 values drawn from [1e-6, 1 - 1e-6] (seed 21), central differences with step 1e-6.

ROW_21_BETAS = (1.0, 8.0, 32.0)
ROW_21_ETAS = (0.25, 0.5, 0.75)


def row_21_projection() -> list[Row]:
    step = 1e-6
    x = np.random.default_rng(21).uniform(step, 1.0 - step, 1000)
    worst_derivative, worst_case = 0.0, ""
    worst_end = 0.0
    for beta in ROW_21_BETAS:
        for eta in ROW_21_ETAS:
            projection = Projection(beta, eta)
            difference = (projection.forward(x + step) - projection.forward(x - step)) / (2 * step)
            derivative = projection.backward(x, np.ones_like(x))
            measured = float(np.abs(difference - derivative).max() / np.abs(derivative).max())
            if measured >= worst_derivative:
                worst_derivative, worst_case = measured, f"beta {g3(beta)}, eta {g3(eta)}"
            ends = projection.forward(np.array([0.0, 1.0]))
            worst_end = max(worst_end, abs(float(ends[0])), abs(float(ends[1]) - 1.0))
    combinations = f"beta in {', '.join(g3(b) for b in ROW_21_BETAS)}; eta in {', '.join(g3(e) for e in ROW_21_ETAS)}"
    return [
        checked(
            21,
            "Projection: derivative against central differences",
            GRADIENT,
            worst_derivative,
            1e-7,
            f"largest difference relative to the derivative's size over {combinations} (worst: {worst_case}); "
            f"1000 random values, step 1e-6",
        ),
        checked(
            21,
            "Projection keeps 0 and 1: P(0) = 0, P(1) = 1",
            CLOSED_FORM,
            worst_end,
            1e-15,
            f"largest of |P(0)| and |P(1) - 1| over {combinations}",
        ),
    ]


# ---------------------------------------------------------------------------------------------------------
# Row 22. Limit: the gradients of compliance read on the eroded design and of mass read on the dilated design,
# through the Helmholtz filter and the projection (`robust_designs(beta=8, spread=0.25)`), agree with central
# finite differences to a relative 1e-5 on a 3D grid.
#
# Measure (as rows 7 and 19): over every active element e, max |fd_e - adjoint_e| / max |adjoint_e|; the
# worst element on its own scale is reported beside it and not judged. Set-up: the problem of row 19
# (`michell.gradient_check.small_problem_3d(seed=0)`, solves at 1e-12, elasticity solved directly), with the
# three designs of the robust set made from the output of its Helmholtz filter; the gradients are taken by
# `check_gradient(..., designs=)`, which runs the code of the optimisation loop; step 1e-6.


def row_22_robust_gradients() -> list[Row]:
    rows = []
    for design, response in (("eroded", Compliance()), ("dilated", Mass())):
        problem, x, filters = small_problem_3d(seed=0, solver_tolerance=1e-12, filter_tolerance=1e-12)
        designs = robust_designs(8.0, 0.25)
        check = check_gradient(problem, on(design, response), x, filters, step=1e-6, designs=designs)
        note = (
            f"{len(check.elements)} elements of a 3D block with an inactive pocket, two load cases, random "
            f"density; Helmholtz filter, then the projection of the {design} design (beta 8, threshold "
            f"{g3(designs[design][0].eta)}); solves at 1e-12; worst element on its own scale "
            f"{g3(check.largest_element_difference)} (reported, not judged)"
        )
        name = (
            f"Gradient of {response.name} on the {design} design (Helmholtz filter, projection) against central "
            f"differences"
        )
        rows.append(checked(22, name, GRADIENT, check.largest_difference, 1e-5, note))
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 23. Limits, each on its own line, on the robust MBB beam at 120 x 40:
# (a) the mass of the nominal design within 1 % of the asked mass: |M_nominal - M*| / M*;
# (b) the non-discreteness of the nominal design, sum over the elements of 4 rho (1 - rho) / n (Sigmund
#     2007), at most 10 %;
# (c) the compliances in the order C_eroded >= C_nominal >= C_dilated: measured as
#     max(0, (C_nominal - C_eroded) / C_nominal, (C_dilated - C_nominal) / C_nominal), at most 0.
# Set-up (that of `examples/mbb_beam_robust.py` at 120 x 40): the MBB beam of the 88-line code
# (`michell.verification.mbb`), asked mass M* = 0.5 nx ny g (half the design space), density filter of radius
# 0.04 nx = 4.8 element sides (the proportion of the 88-line code's 150 x 50 case), `robust_designs(beta,
# 0.25)`, the goal compliance on `eroded` and the mass limit on `dilated`, MMA, x starting at 0.5.
# Continuation: beta = 1, 2, 4, 8, 16, 32, each for 50 passes, each step one call of `optimise` from the x
# of the step before. Before each step the bound of the dilated mass is set to M* M_dilated / M_nominal,
# the two masses taken at the current x with that step's beta (after Wang et al. 2011), so that the
# nominal design is steered to the asked mass. The three compliances are solved on the designs after the
# last pass.

ROBUST_BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ROBUST_PASSES_PER_STEP = 50
ROBUST_SPREAD = 0.25
ROBUST_FRACTION = 0.5


@dataclass(frozen=True)
class RobustRun:
    """The outcome of a robust MBB run: the result of the last continuation step, the bound of the dilated
    mass set before each step, and the asked mass."""

    result: Result
    bounds: tuple[float, ...]
    asked_mass: float
    passes: int


def robust_mbb(
    nx: int,
    ny: int,
    betas: tuple[float, ...] = ROBUST_BETAS,
    passes_per_step: int = ROBUST_PASSES_PER_STEP,
    spread: float = ROBUST_SPREAD,
) -> tuple[Problem, RobustRun]:
    """The robust MBB beam of row 23 on nx by ny elements, with the continuation and the rescaling of the
    dilated mass bound described above it."""
    problem = mbb_problem(nx, ny)
    density_filter = DensityFilter(problem.grid, 0.04 * nx)
    asked_mass = ROBUST_FRACTION * nx * ny
    mass = Mass()
    x = np.full(problem.grid.n_elements, ROBUST_FRACTION)
    bounds = []
    passes = 0
    result = None
    for beta in betas:
        designs = robust_designs(beta, spread)
        filtered = density_filter.forward(x)
        nominal = apply_filters(filtered, designs["nominal"])[-1]
        dilated = apply_filters(filtered, designs["dilated"])[-1]
        bound = asked_mass * float(dilated.sum()) / float(nominal.sum())  # unit density and unit element volume
        bounds.append(bound)
        result = optimise(
            problem,
            goal=on("eroded", Compliance()),
            limits=[Limit(on("dilated", mass), at_most=bound)],
            filters=[density_filter],
            optimiser=MovingAsymptotes(),
            x=x,
            passes=passes_per_step,
            designs=designs,
        )
        x = result.x
        passes += len(result.passes)
    return problem, RobustRun(result, tuple(bounds), asked_mass, passes)


def compliance_of(problem: Problem, density: np.ndarray) -> float:
    """The compliance f . u of the problem's one load case at ``density``."""
    loads = problem.cases[0].loads
    u, _ = problem.solve(density, loads)
    return float(np.dot(loads, u))


def row_23_robust_mbb(nx: int = 120, ny: int = 40) -> list[Row]:
    problem, run = robust_mbb(nx, ny)
    designs = run.result.designs
    nominal = designs["nominal"]
    n = problem.grid.n_elements
    mass_nominal = float(nominal.sum())
    mass_error = abs(mass_nominal - run.asked_mass) / run.asked_mass
    non_discreteness = float((4.0 * nominal * (1.0 - nominal)).sum() / n)
    compliance = {name: compliance_of(problem, designs[name]) for name in ("eroded", "nominal", "dilated")}
    c_e, c_n, c_d = compliance["eroded"], compliance["nominal"], compliance["dilated"]
    out_of_order = max(0.0, (c_n - c_e) / c_n, (c_d - c_n) / c_n)
    set_up = (
        f"MBB beam {nx} x {ny}, density filter radius {g3(0.04 * nx)}, spread 0.25, beta 1 to 32 doubling every "
        f"{ROBUST_PASSES_PER_STEP} passes ({run.passes} passes), MMA"
    )
    masses = ", ".join(g3(m) for m in (float(designs[name].sum()) for name in ("eroded", "nominal", "dilated")))
    return [
        checked(
            23,
            f"Robust MBB beam {nx} x {ny}: mass of the nominal design against the asked mass",
            DESIGN_PROPERTY,
            mass_error,
            0.01,
            f"nominal {g3(mass_nominal)} g, asked {g3(run.asked_mass)} g; masses eroded, nominal, dilated: "
            f"{masses} g; last bound on the dilated mass {g3(run.bounds[-1])} g; {set_up}",
        ),
        checked(
            23,
            f"Robust MBB beam {nx} x {ny}: non-discreteness of the nominal design, sum 4 rho (1 - rho) / n",
            DESIGN_PROPERTY,
            non_discreteness,
            0.10,
            f"over all {n} elements (Sigmund 2007); {set_up}",
        ),
        checked(
            23,
            f"Robust MBB beam {nx} x {ny}: compliance eroded >= nominal >= dilated",
            DESIGN_PROPERTY,
            out_of_order,
            0.0,
            f"compliance eroded {g3(c_e)}, nominal {g3(c_n)}, dilated {g3(c_d)} N mm; measured: the largest "
            f"relative step out of that order, 0 when in order; {set_up}",
        ),
    ]
