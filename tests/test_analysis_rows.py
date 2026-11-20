"""Verification rows 1 to 6: the 2D analysis against closed forms, beam theory, Kirsch and top88.

Each row's limit was fixed before any result was computed. Each test prints what it measured,
so `pytest -s -k row` shows the numbers that go into the verification table.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

import michell

REFERENCE = Path(__file__).parent / "reference" / "top88"


def plane_stress_matrix(e, nu):
    """D with [sx, sy, txy] = D [ex, ey, gxy], gxy the engineering shear strain."""
    c = e / (1 - nu**2)
    return c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def boundary_nodes(grid):
    """Indices of the nodes on the outer edge of the grid."""
    return [
        grid.node(i, j)
        for j in range(grid.ny + 1)
        for i in range(grid.nx + 1)
        if i in (0, grid.nx) or j in (0, grid.ny)
    ]


def node_coordinates(grid):
    """x and y of every node, in node order."""
    i, j = np.meshgrid(np.arange(grid.nx + 1), np.arange(grid.ny + 1))  # j rows, i columns: x fastest
    return grid.h * i.ravel().astype(float), grid.h * j.ravel().astype(float)


def solve_with_boundary_field(grid, e, nu, field):
    """Hold every boundary node at the displacement `field(x, y) -> (u, v)`; no loads. Returns the model,
    the displacements, the exact displacements, and the boundary node indices."""
    x, y = node_coordinates(grid)
    exact_u, exact_v = field(x, y)
    exact = np.empty(grid.n_dofs)
    exact[0::2], exact[1::2] = exact_u, exact_v
    held = np.zeros(grid.n_dofs)
    boundary = boundary_nodes(grid)
    for node in boundary:
        held[2 * node] = held[2 * node + 1] = 1.0
    model = michell.Elasticity(grid, held, youngs_modulus=e, poisson_ratio=nu, e_min=1e-9 * e, penalty=3.0)
    u = model.solve(np.ones(grid.n_elements), np.zeros(grid.n_dofs), held_values=exact)
    return model, u, exact, boundary


# ---------------------------------------------------------------------------------------------------------
# Row 1. Limit: element matrix equal to the closed form to 1e-12 (largest entry difference, relative to the
# largest entry of the matrix).


def closed_form_element_matrix(e, nu, t):
    """The textbook closed form of the square bilinear plane-stress element, unit side, thickness t.

    Source: Bendsøe and Sigmund, "Topology Optimization" (2003), and Andreassen, Clausen, Schevenels,
    Lazarov and Sigmund, "Efficient topology optimization in MATLAB using 88 lines of code", Struct.
    Multidisc. Optim. 43 (2011), the matrix KE. Nodes counter-clockwise from the lower left.
    """
    k = [
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ]  # fmt: skip
    order = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]
    return t * e / (1 - nu**2) * np.array([[k[index] for index in row] for row in order])


@pytest.mark.parametrize("e, nu, h, t", [(1.0, 0.3, 1.0, 1.0), (210000.0, 0.3, 2.5, 3.0), (70000.0, 0.33, 0.1, 1.5)])
def test_row_1_element_matrix_equals_the_closed_form(e, nu, h, t):
    ours = michell.element_stiffness(e, nu, h, t)
    expected = closed_form_element_matrix(e, nu, t)
    measured = np.abs(ours - expected).max() / np.abs(expected).max()
    print(f"ROW 1 (E={e}, nu={nu}, h={h}, t={t}): largest relative difference {measured:.2e}, limit 1e-12")
    assert measured <= 1e-12


# ---------------------------------------------------------------------------------------------------------
# Row 2. Limit: a linear displacement field on the boundary is reproduced inside to 1e-10, with uniform
# stress (both relative to the largest value of the exact field).


def test_row_2_patch_test_reproduces_a_linear_field():
    e, nu = 210000.0, 0.3
    grid = michell.Grid(7, 5, 2.0, 1.5)
    a0, a1, a2, b0, b1, b2 = 0.01, 2e-3, -1e-3, -0.02, 5e-4, 1.5e-3

    def field(x, y):
        return a0 + a1 * x + a2 * y, b0 + b1 * x + b2 * y

    model, u, exact, boundary = solve_with_boundary_field(grid, e, nu, field)
    inside = np.ones(grid.n_nodes, dtype=bool)
    inside[boundary] = False
    inside_dofs = np.repeat(inside, 2)
    displacement_error = np.abs(u - exact)[inside_dofs].max() / np.abs(exact).max()

    stress = model.element_stress(np.ones(grid.n_elements), u)
    expected_stress = plane_stress_matrix(e, nu) @ np.array([a1, b2, a2 + b1])
    stress_error = np.abs(stress - expected_stress).max() / np.abs(expected_stress).max()
    print(f"ROW 2: inside displacement error {displacement_error:.2e}, stress error {stress_error:.2e}, limit 1e-10")
    assert displacement_error <= 1e-10
    assert stress_error <= 1e-10


# ---------------------------------------------------------------------------------------------------------
# Row 3. Limit: in pure shear, element shear stress equals G * gamma to 1e-10 (relative), normal stresses
# zero to 1e-10 of G * gamma.


def test_row_3_pure_shear_gives_g_times_gamma():
    e, nu, gamma = 210000.0, 0.3, 1e-3
    grid = michell.Grid(4, 4, 1.5, 2.0)
    model, u, _, _ = solve_with_boundary_field(grid, e, nu, lambda x, y: (gamma / 2 * y, gamma / 2 * x))
    stress = model.element_stress(np.ones(grid.n_elements), u)
    g_gamma = e / (2 * (1 + nu)) * gamma
    shear_error = np.abs(stress[:, 2] - g_gamma).max() / g_gamma
    normal_error = np.abs(stress[:, :2]).max() / g_gamma
    print(f"ROW 3: G*gamma = {g_gamma:.6g} MPa; shear error {shear_error:.2e}, normal {normal_error:.2e}, limit 1e-10")
    assert shear_error <= 1e-10
    assert normal_error <= 1e-10


# ---------------------------------------------------------------------------------------------------------
# Row 4. Limit: cantilever of length 8 x depth with an end load against Timoshenko beam theory
# PL^3/(3EI) + 6PL/(5GA): the error shrinks from 10 to 20 to 40 elements through the depth, under 2 % at 40.
#
# Set-up chosen with the limit: left edge fully clamped; the end load P is a parabolic shear traction on the
# right edge (the distribution beam theory assumes), as consistent nodal forces; the deflection is that of
# the node at mid-depth of the right edge.


def cantilever_tip_deflection(n):
    depth, length, t, e, nu, p = 10.0, 80.0, 2.0, 210000.0, 0.3, 1000.0
    h = depth / n
    grid = michell.Grid(8 * n, n, h, t)
    held = np.zeros(grid.n_dofs)
    for j in range(n + 1):
        node = grid.node(0, j)
        held[2 * node] = held[2 * node + 1] = 1.0

    def traction(y):  # downward shear stress on the end face, integrating to P over the area t * depth
        return 1.5 * p / (t * depth) * (1 - (2 * y / depth - 1) ** 2)

    loads = np.zeros(grid.n_dofs)
    for j in range(n):  # each edge piece [y0, y1]: Simpson's rule is exact for traction * linear shape
        y0, y1 = j * h, (j + 1) * h
        ym = (y0 + y1) / 2
        lower = t * h / 6 * (traction(y0) * 1 + 4 * traction(ym) * 0.5 + traction(y1) * 0)
        upper = t * h / 6 * (traction(y0) * 0 + 4 * traction(ym) * 0.5 + traction(y1) * 1)
        loads[2 * grid.node(8 * n, j) + 1] -= lower
        loads[2 * grid.node(8 * n, j + 1) + 1] -= upper
    assert -loads.sum() == pytest.approx(p, rel=1e-12)

    model = michell.Elasticity(grid, held, youngs_modulus=e, poisson_ratio=nu, e_min=1e-9 * e, penalty=3.0)
    u = model.solve(np.ones(grid.n_elements), loads)
    deflection = -u[2 * grid.node(8 * n, n // 2) + 1]

    inertia, area, shear_modulus = t * depth**3 / 12, t * depth, e / (2 * (1 + nu))
    timoshenko = p * length**3 / (3 * e * inertia) + 6 * p * length / (5 * shear_modulus * area)
    return deflection, timoshenko


def test_row_4_cantilever_converges_to_timoshenko_beam_theory():
    errors = {}
    for n in (10, 20, 40):
        deflection, timoshenko = cantilever_tip_deflection(n)
        errors[n] = abs(deflection - timoshenko) / timoshenko
        print(f"ROW 4: {n} elements through the depth: deflection {deflection:.6f} mm, "
              f"Timoshenko {timoshenko:.6f} mm, error {100 * errors[n]:.3f} %")
    print("ROW 4: limit: error shrinks 10 -> 20 -> 40 and is under 2 % at 40")
    assert errors[10] > errors[20] > errors[40]
    assert errors[40] < 0.02


# ---------------------------------------------------------------------------------------------------------
# Row 5. Limit: plate with a round hole under tension (hole radius a = 16 elements, void elements for the
# hole, plate half-width 10a): the stress across the narrowest section equals Kirsch's solution within 3 %
# at distances 2a and 3a from the centre. The peak at the stair-stepped edge is reported, not judged.
#
# Set-up chosen with the limit: a quarter of the plate by symmetry, 10a by 10a (so half-width and half-length
# 10a), the hole centred on the corner (0, 0); x held on the edge x = 0, y held on y = 0; a uniform tension
# on the edge x = 10a as consistent nodal forces. Elements whose centre lies inside r < a are outside the
# part (inactive). The narrowest section is x = 0; the elements next to it have their centres at x = h/2,
# so the measured value at y = 2a and 3a is interpolated linearly between the two element centres on either
# side, and compared with Kirsch's sx at the same point (h/2, y).


def kirsch_sx(x, y, a, s):
    """Kirsch (1898): sx at (x, y) around a hole of radius a in an infinite plate under tension s along x."""
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    q2, q4 = (a / r) ** 2, (a / r) ** 4
    s_rr = s / 2 * (1 - q2) + s / 2 * (1 - 4 * q2 + 3 * q4) * c2
    s_tt = s / 2 * (1 + q2) - s / 2 * (1 + 3 * q4) * c2
    s_rt = -s / 2 * (1 + 2 * q2 - 3 * q4) * s2
    c, sn = np.cos(theta), np.sin(theta)
    return s_rr * c**2 + s_tt * sn**2 - 2 * s_rt * sn * c


def test_row_5_plate_with_a_hole_matches_kirsch():
    e, nu, sigma, t, h = 210000.0, 0.3, 100.0, 1.0, 1.0
    a_elements = 16
    n = 10 * a_elements
    a = a_elements * h
    centres = (np.arange(n) + 0.5) * h
    cx, cy = np.meshgrid(centres, centres)  # x fastest
    active = (np.hypot(cx, cy).ravel() >= a).astype(float)
    grid = michell.Grid(n, n, h, t, active=active)

    held = np.zeros(grid.n_dofs)
    for k in range(n + 1):
        held[2 * grid.node(0, k)] = 1.0  # x held on x = 0
        held[2 * grid.node(k, 0) + 1] = 1.0  # y held on y = 0
    loads = np.zeros(grid.n_dofs)
    for j in range(n + 1):
        share = 0.5 if j in (0, n) else 1.0
        loads[2 * grid.node(n, j)] = sigma * t * h * share
    model = michell.Elasticity(grid, held, youngs_modulus=e, poisson_ratio=nu, e_min=1e-9 * e, penalty=3.0)
    rho = np.ones(grid.n_elements)
    stress = model.element_stress(rho, model.solve(rho, loads))
    first_column = np.array([stress[grid.element(0, j), 0] for j in range(n)])

    for distance in (2, 3):
        y = distance * a
        below = int(y / h - 0.5)  # the element whose centre is just below y
        weight = (y - centres[below]) / h
        measured = (1 - weight) * first_column[below] + weight * first_column[below + 1]
        expected = kirsch_sx(h / 2, y, a, sigma)
        error = abs(measured - expected) / expected
        print(f"ROW 5: sx at {distance}a: {measured:.3f} MPa, Kirsch {expected:.3f} MPa, "
              f"error {100 * error:.2f} %, limit 3 %")
        assert error < 0.03
    peak = first_column.max()
    print(f"ROW 5: peak sx on the narrowest section {peak:.1f} MPa = {peak / sigma:.3f} x the applied stress "
          f"(Kirsch at the hole edge: 3.000); reported, not judged")


# ---------------------------------------------------------------------------------------------------------
# Row 6. Limit: compliance of the uniform half-density MBB beam equal to the first row of the top88
# reference files to a relative 1e-8. Set-up of that code: unit modulus, E_min 1e-9, Poisson 0.3, penalty
# 3, unit load down on the top-left node, x held on the whole left edge, y held at the bottom-right node.


@pytest.mark.parametrize("name, nx, ny", [("mbb_60x20_r2.4_density", 60, 20), ("mbb_150x50_r6_density", 150, 50)])
def test_row_6_first_pass_compliance_equals_top88(name, nx, ny):
    with open(REFERENCE / f"{name}.csv", newline="") as file:
        first_row = next(csv.DictReader(file))
    assert first_row["pass"] == "1"
    reference = float(first_row["compliance"])

    grid = michell.Grid(nx, ny, 1.0, 1.0)
    held = np.zeros(grid.n_dofs)
    for j in range(ny + 1):
        held[2 * grid.node(0, j)] = 1.0
    held[2 * grid.node(nx, 0) + 1] = 1.0
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(0, ny) + 1] = -1.0
    model = michell.Elasticity(grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0)
    rho = np.full(grid.n_elements, 0.5)
    compliance = loads @ model.solve(rho, loads)
    error = abs(compliance - reference) / reference
    print(f"ROW 6 ({name}): compliance {compliance:.12g}, top88 {reference:.12g}, relative error {error:.2e}, "
          f"limit 1e-8")
    assert error <= 1e-8
