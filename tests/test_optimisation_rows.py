"""Verification rows 7 and 8: the gradients through the filter, and the MBB beam against the 88-line code.

Each row's limit was fixed before any result was computed. Each test prints what it measured, so
`pytest -s -k row` shows the numbers that go into the verification table.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

import michell

REFERENCE = Path(__file__).parent / "reference" / "top88"


# ---------------------------------------------------------------------------------------------------------
# Row 7. Limit: the gradients of compliance and of mass through the density filter agree with central
# finite differences to a relative 1e-5 on a random field.
#
# Measure (fixed with the limit): over every active element e, max |fd_e - adjoint_e| / max |adjoint_e|,
# that is the difference relative to the size of the gradient. The worst element on its own scale,
# max |fd_e - adjoint_e| / |adjoint_e|, is printed beside it and not judged. The problem is
# `michell.gradient_check.small_problem(seed=0)`: 12 x 6 elements with a hole, two weighted load cases,
# density filter of radius 2.5 elements, x drawn from [0.1, 0.9]; central differences with step 1e-6.


@pytest.mark.parametrize("response", [michell.Compliance(), michell.Mass()], ids=["compliance", "mass"])
def test_row_7_gradients_through_the_filter_match_finite_differences(response):
    check = michell.check_on_small_problem(response, seed=0, step=1e-6)
    print(f"ROW 7 ({response.name}): {len(check.elements)} elements, largest difference "
          f"{check.largest_difference:.2e} of the gradient's size (limit 1e-5); worst element on its own "
          f"scale {check.largest_element_difference:.2e} (reported)")
    assert check.largest_difference <= 1e-5


# ---------------------------------------------------------------------------------------------------------
# Row 8. Limit: the MBB beam with the density filter and OC, run for exactly as many passes as the
# reference has rows: compliance equal to the reference within a relative 1e-6 on each of the first 10
# passes, and within 0.1 % on every later pass and at the end. Both reference files.
#
# Set-up of the 88-line code (as row 6): unit modulus, E_min 1e-9, Poisson 0.3, penalty 3, unit load down
# on the top-left node, x held on the whole left edge, y held at the bottom-right node; volume fraction
# 0.5 (here: a mass limit of 0.5 nx ny g with unit element volume and material density), x starting at 0.5,
# OC with move limit 0.2.


def read_reference(name):
    with open(REFERENCE / f"{name}.csv", newline="") as file:
        return list(csv.DictReader(file))


def mbb_problem(nx, ny):
    grid = michell.Grid(nx, ny, 1.0, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for j in range(ny + 1):
        held[2 * grid.node(0, j)] = True
    held[2 * grid.node(nx, 0) + 1] = True
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(0, ny) + 1] = -1.0
    material = michell.Material(youngs_modulus=1.0, poisson_ratio=0.3, density_g_mm3=1.0)
    return michell.Problem(grid, held, material, [michell.LoadCase("load", loads)], penalty=3.0, e_min=1e-9)


@pytest.mark.parametrize(
    "name, nx, ny, radius",
    [("mbb_60x20_r2.4_density", 60, 20, 2.4), ("mbb_150x50_r6_density", 150, 50, 6.0)],
)
def test_row_8_mbb_beam_with_oc_follows_top88(name, nx, ny, radius):
    reference = read_reference(name)
    problem = mbb_problem(nx, ny)
    grid = problem.grid
    result = michell.optimise(
        problem,
        goal=michell.Compliance(),
        limits=[michell.Limit(michell.Mass(), at_most=0.5 * nx * ny)],
        filters=[michell.DensityFilter(grid, radius)],
        optimiser=michell.OptimalityCriteria(move_limit=0.2),
        x=np.full(grid.n_elements, 0.5),
        passes=len(reference),
    )
    assert len(result.passes) == len(reference)
    errors = np.array([abs(p.goal - float(row["compliance"])) / float(row["compliance"])
                       for p, row in zip(result.passes, reference)])
    first, later = errors[:10].max(), errors[10:].max()
    final = result.passes[-1].goal
    stop = next((p.number for p in result.passes if p.change <= 0.01), None)
    print(f"ROW 8 ({name}): {len(reference)} passes; largest relative difference on passes 1-10 {first:.2e} "
          f"(limit 1e-6), on later passes {later:.2e} (limit 1e-3); final compliance {final:.10g}, top88 "
          f"{float(reference[-1]['compliance']):.10g}; the change first falls to 0.01 at pass {stop}")
    assert first <= 1e-6
    assert later <= 1e-3
