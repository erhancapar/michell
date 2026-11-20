"""Verification rows 19 and 20: the gradients of the 3D loop through the Helmholtz filter, and the Helmholtz
filter itself (Lazarov and Sigmund, "Filters in topology optimization based on Helmholtz-type differential
equations", IJNME 86, 2011).

Each row's measure, set-up and limit were fixed before any result was computed.
"""

from __future__ import annotations

import numpy as np

from michell.analysis3d import Grid3d
from michell.filters import HelmholtzFilter
from michell.gradient_check import check_gradient, small_problem_3d
from michell.responses import Compliance, Mass
from michell.row import CLOSED_FORM, GRADIENT, Row, checked, g3

ROW_THREADS = 4

# ---------------------------------------------------------------------------------------------------------
# Row 19. Limit: the gradients of compliance and of mass through the Helmholtz filter on a 3D grid agree
# with central finite differences to a relative 1e-5 on a random field, every solve (elasticity and filter)
# at a relative residual of 1e-12.
#
# Measure (as row 7): over every active element e, max |fd_e - adjoint_e| / max |adjoint_e|; the worst
# element on its own scale is reported beside it and not judged. The problem is
# `michell.gradient_check.small_problem_3d(seed=0)`: 8 x 4 x 4 elements with a pocket of 2 x 2 x 2
# inactive ones, two weighted load cases, Helmholtz filter of radius 2.5 element sides, x drawn from
# [0.1, 0.9]; central differences with step 1e-6. Its 675 degrees of freedom keep the multigrid to one
# level, so the elasticity solves are direct and only the filter's iterative solve adds noise to the
# differences.


def row_19_gradients() -> list[Row]:
    rows = []
    for response in (Compliance(), Mass()):
        problem, x, filters = small_problem_3d(seed=0, solver_tolerance=1e-12, filter_tolerance=1e-12)
        check = check_gradient(problem, response, x, filters, step=1e-6)
        note = (
            f"{len(check.elements)} elements of a 3D block with an inactive pocket, two load cases, random "
            f"density, solves at 1e-12; largest difference relative to the gradient's size; worst element on "
            f"its own scale {g3(check.largest_element_difference)} (reported, not judged)"
        )
        name = f"Gradient of {response.name} through the Helmholtz filter in 3D against central differences"
        rows.append(checked(19, name, GRADIENT, check.largest_difference, 1e-5, note))
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 20. Limits, each on its own line:
# (a) the sum of the field is kept to a relative 1e-8: |sum rho - sum x| / |sum x| over the active elements;
# (b) the map is symmetric to a relative 1e-8: |<F x, y> - <x, F y>| / |<F x, y>|;
# (c) on a bar one element thick and more than 10 r long on each side of a source element set to 1, the
#     filtered value at the distances r, 2r and 3r on both sides equals the solution of the 1D equation
#     -r^2 psi'' + psi = (the source), h / (2 r) exp(-|x| / r), within 2 %: the largest relative
#     difference of the six values; r = 4 h. The filter is built with a lumped mass matrix (it keeps every
#     field in [0, 1] for any radius; the consistent one does not below a radius of about two elements),
#     so this compares the lumped discretisation with the continuous 1D solution.
# Set-up of (a) and (b): 24 x 12 x 10 elements of side 1 mm with the elements i >= 16, j >= 6 outside the
# part (an L-shaped section), radius 2.5 mm, x and y drawn from [0, 1], the filter at its own tolerance
# (1e-10). Set-up of (c): 91 x 1 x 1 elements of side 1 mm (45 elements, 11.25 r, on each side of the
# source), radius 2 sqrt(3) r. Every filter runs on `ROW_THREADS` threads.

BAR_ELEMENTS_PER_SIDE = 45


def row_20_filter() -> list[Row]:
    rows = []
    nx, ny, nz = 24, 12, 10
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))  # x fastest within each layer
    active = np.tile(~((i >= 16) & (j >= 6)).ravel(), nz)
    grid = Grid3d(nx, ny, nz, 1.0, active=active)
    pde_filter = HelmholtzFilter(grid, 2.5, threads=ROW_THREADS)
    random = np.random.default_rng(20)
    x = np.where(active, random.uniform(0.0, 1.0, grid.n_elements), 0.0)
    y = np.where(active, random.uniform(0.0, 1.0, grid.n_elements), 0.0)
    fx = pde_filter.forward(x)
    sum_x, sum_fx = float(x.sum()), float(fx.sum())
    set_up = f"{nx} x {ny} x {nz} elements, {int((~active).sum())} outside the part, radius 2.5 h, random fields"
    rows.append(
        checked(
            20,
            "Helmholtz filter keeps the sum of the field",
            CLOSED_FORM,
            abs(sum_fx - sum_x) / abs(sum_x),
            1e-8,
            f"relative difference of the sums; {set_up}",
        )
    )
    left = float(np.dot(fx, y))
    right = float(np.dot(x, pde_filter.forward(y)))
    rows.append(
        checked(
            20,
            "Helmholtz filter is symmetric: <F x, y> = <x, F y>",
            CLOSED_FORM,
            abs(left - right) / abs(left),
            1e-8,
            f"relative difference; {set_up}",
        )
    )

    h, r = 1.0, 4.0
    n = 2 * BAR_ELEMENTS_PER_SIDE + 1
    bar = HelmholtzFilter(Grid3d(n, 1, 1, h), 2.0 * np.sqrt(3.0) * r, threads=ROW_THREADS)
    source = np.zeros(n)
    source[BAR_ELEMENTS_PER_SIDE] = 1.0
    response = bar.forward(source)
    errors = []
    for distance in (r, 2 * r, 3 * r):
        exact = h / (2 * r) * np.exp(-distance / r)
        steps = int(round(distance / h))
        for side in (-1, 1):
            value = float(response[BAR_ELEMENTS_PER_SIDE + side * steps])
            errors.append(abs(value - exact) / exact)
    by_distance = ", ".join(g3(max(errors[2 * d], errors[2 * d + 1])) for d in range(3))
    rows.append(
        checked(
            20,
            "Helmholtz filter on a long bar against the 1D solution h / (2 r) exp(-|x| / r)",
            CLOSED_FORM,
            max(errors),
            0.02,
            f"largest relative difference at r, 2r and 3r on both sides ({by_distance}); r = 4 h, "
            f"{n} x 1 x 1 elements, {g3(BAR_ELEMENTS_PER_SIDE * h / r)} r on each side; lumped mass matrix, "
            f"which keeps every filtered field in [0, 1] for any radius",
        )
    )
    return rows
