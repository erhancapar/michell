"""Verification rows 7 and 8: the gradients through the filter, and the MBB beam against the 88-line code.

Each row's measure, set-up and limit were fixed before any result was computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from michell.filters import DensityFilter
from michell.gradient_check import check_on_small_problem
from michell.oc import OptimalityCriteria
from michell.optimise import Limit, optimise
from michell.responses import Compliance, Mass
from michell.verification.mbb import REFERENCE_NOT_FOUND, mbb_case, mbb_problem, read_reference
from michell.verification.row import GRADIENT, REFERENCE_CODE, Row, checked, g3, not_run

# ---------------------------------------------------------------------------------------------------------
# Row 7. Limit: the gradients of compliance and of mass through the density filter agree with central
# finite differences to a relative 1e-5 on a random field.
#
# Measure (fixed with the limit): over every active element e, max |fd_e - adjoint_e| / max |adjoint_e|,
# that is the difference relative to the size of the gradient. The worst element on its own scale,
# max |fd_e - adjoint_e| / |adjoint_e|, is reported beside it and not judged. The problem is
# `michell.gradient_check.small_problem(seed=0)`: 12 x 6 elements with a hole, two weighted load cases,
# density filter of radius 2.5 elements, x drawn from [0.1, 0.9]; central differences with step 1e-6.


def row_7_gradients() -> list[Row]:
    rows = []
    for response in (Compliance(), Mass()):
        check = check_on_small_problem(response, seed=0, step=1e-6)
        note = (
            f"{len(check.elements)} elements with a hole, two load cases, random density; largest difference "
            f"relative to the gradient's size; worst element on its own scale "
            f"{g3(check.largest_element_difference)} (reported, not judged)"
        )
        name = f"Gradient of {response.name} through the density filter against central differences"
        rows.append(checked(7, name, GRADIENT, check.largest_difference, 1e-5, note))
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 8. Limit: the MBB beam with the density filter and OC, run for exactly as many passes as the
# reference has rows: compliance equal to the reference within a relative 1e-6 on each of the first 10
# passes, and within 0.1 % on every later pass and at the end. Both reference files.
#
# Set-up of the 88-line code (`michell.verification.mbb`); x starting at 0.5, OC with move limit 0.2.


def row_8_mbb_with_oc(reference: Path | str | None, name: str) -> list[Row]:
    """Row 8 on the reference case ``name`` (one of ``MBB_CASES``): two lines, passes 1 to 10 and the
    later passes."""
    nx, ny, radius = mbb_case(name)
    first_title = f"MBB beam {nx} x {ny} with OC: compliance on passes 1 to 10 against the 88-line code"
    later_title = f"MBB beam {nx} x {ny} with OC: compliance on every later pass against the 88-line code"
    passes = read_reference(reference, name)
    if passes is None:
        return [
            not_run(8, first_title, REFERENCE_CODE, 1e-6, REFERENCE_NOT_FOUND),
            not_run(8, later_title, REFERENCE_CODE, 1e-3, REFERENCE_NOT_FOUND),
        ]
    problem = mbb_problem(nx, ny)
    grid = problem.grid
    result = optimise(
        problem,
        goal=Compliance(),
        limits=[Limit(Mass(), at_most=0.5 * nx * ny)],
        filters=[DensityFilter(grid, radius)],
        optimiser=OptimalityCriteria(move_limit=0.2),
        x=np.full(grid.n_elements, 0.5),
        passes=len(passes),
    )
    if len(result.passes) != len(passes):
        raise RuntimeError(f"the run made {len(result.passes)} passes, the reference {len(passes)}")
    errors = np.array(
        [abs(p.goal - float(row["compliance"])) / float(row["compliance"]) for p, row in zip(result.passes, passes)]
    )
    first_note = f"largest relative difference; filter radius {g3(radius)}, move limit 0.2"
    later_note = (
        f"largest relative difference over passes 11 to {len(passes)}; final compliance "
        f"{g3(result.passes[-1].goal)} N mm"
    )
    return [
        checked(8, first_title, REFERENCE_CODE, float(errors[:10].max()), 1e-6, first_note),
        checked(8, later_title, REFERENCE_CODE, float(errors[10:].max()), 1e-3, later_note),
    ]
