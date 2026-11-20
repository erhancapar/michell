"""Verification rows 16 to 18: the multigrid-preconditioned solver of the 3D analysis against a sparse direct
solution, its step counts, and a cantilever against beam theory.

Each row's measure, set-up and limit were fixed before any result was computed. Unit cube elements, Poisson
ratio 0.3, E = 210000 MPa, E_min = 1e-9 E; the designs of rows 16 and 17 are in ``designs3d``. Every solve
runs on ``ROW_THREADS`` threads (the numbers do not depend on the count; row 15 checks that).
"""

from __future__ import annotations

import numpy as np

from michell.analysis3d import Elasticity3d
from michell.row import CLOSED_FORM, CONVERGENCE, INDEPENDENT_COMPUTATION, Row, checked, g3, not_run
from michell.verification.analysis3d_rows import assembled_matrix
from michell.verification.designs3d import E, E_MIN_RATIO, NU, irregular, l_shaped, lattice, uniform

ROW_THREADS = 4


def model_for(setup, tolerance: float, max_steps: int) -> Elasticity3d:
    return Elasticity3d(
        setup.grid, setup.held, youngs_modulus=E, poisson_ratio=NU, e_min=E_MIN_RATIO * E, penalty=3.0,
        tolerance=tolerance, max_steps=max_steps, threads=ROW_THREADS,
    )  # fmt: skip


def build_case(case: str, shape: tuple[int, int, int]):
    """The set-up of ``case`` on ``shape``, and None or the reason it cannot be used."""
    if case == "uniform":
        return uniform(shape), None
    if case == "lattice":
        return lattice(shape), None
    if case == "irregular":
        setup, joined = irregular(shape)
        return setup, None if joined else "the solid does not join the held slab to the loaded slab"
    if case == "L-shaped":
        return l_shaped(shape), None
    raise ValueError(f"unknown case {case!r}")


# ---------------------------------------------------------------------------------------------------------
# Row 16. Limit: the solution of the 3D analysis (solved to a relative residual of 1e-10) equals a sparse
# direct solution computed with SciPy to a relative 1e-7 in the displacement, measured as
# |u - u_direct| / |u_direct| in the Euclidean norm over all degrees of freedom. Four cases of about 20 000
# elements (``designs3d``): uniform density, the lattice and the irregular 0-and-1 designs on a 50 x 20 x 20
# cantilever, and the L-shaped region of 35 x 25 x 27 elements. The direct solution factors the stiffness
# matrix of the free degrees of freedom, assembled independently (row 14), by SciPy's sparse LU with the
# unknowns in nested-dissection order (below), which keeps the factor small enough for a test.

ROW_16_CASES = {
    "uniform": (50, 20, 20),
    "lattice": (50, 20, 20),
    "irregular": (50, 20, 20),
    "L-shaped": (35, 25, 27),
}


def nested_dissection_nodes(nx: int, ny: int, nz: int) -> np.ndarray:
    """The nodes of an (nx + 1) x (ny + 1) x (nz + 1) grid in nested-dissection order: the box is cut by
    the middle plane of its longest side, the two halves are numbered first (each cut again the same way)
    and the plane last (George, "Nested dissection of a regular finite element mesh", SIAM J. Numer. Anal.
    10, 1973)."""
    parts = []

    def number(low, high):
        size = [h - lo + 1 for lo, h in zip(low, high)]
        if min(size) <= 0:
            return
        if np.prod(size) <= 64 or max(size) < 3:
            k, j, i = np.meshgrid(*[np.arange(low[a], high[a] + 1) for a in (2, 1, 0)], indexing="ij")
            parts.append((i + (nx + 1) * (j + (ny + 1) * k)).ravel())
            return
        axis = int(np.argmax(size))
        middle = (low[axis] + high[axis]) // 2
        below, above, plane_low, plane_high = list(high), list(low), list(low), list(high)
        below[axis], above[axis] = middle - 1, middle + 1
        plane_low[axis] = plane_high[axis] = middle
        number(low, below)
        number(above, high)
        number(plane_low, plane_high)

    number([0, 0, 0], [nx, ny, nz])
    return np.concatenate(parts)


def direct_solution(setup, sparse, linalg) -> np.ndarray:
    """u from the assembled matrix, factored by SciPy; 0 on held degrees of freedom and on those that
    touch no active element."""
    grid = setup.grid
    active = grid.active.astype(bool)
    matrix = assembled_matrix(grid, active, setup.rho, setup.held, sparse)
    free = np.zeros(grid.n_dofs, dtype=bool)
    steps = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for element in np.flatnonzero(active):
        k, rest = divmod(int(element), grid.nx * grid.ny)
        j, i = divmod(rest, grid.nx)
        for di, dj, dk in steps:
            node = grid.node(i + di, j + dj, k + dk)
            free[3 * node : 3 * node + 3] = True
    free &= ~setup.held
    nodes = nested_dissection_nodes(grid.nx, grid.ny, grid.nz)
    order = (3 * nodes[:, None] + np.arange(3)).ravel()
    order = order[free[order]]
    reduced = matrix[order][:, order].tocsc()
    factor = linalg.splu(reduced, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    u = np.zeros(grid.n_dofs)
    u[order] = factor.solve(setup.loads[order])
    return u


def row_16_direct_solve(case: str, shape: tuple[int, int, int] | None = None) -> list[Row]:
    """Row 16 on one case (a key of ``ROW_16_CASES``), at its own size unless ``shape`` is given."""
    shape = ROW_16_CASES[case] if shape is None else shape
    name = f"3D solver against a sparse direct solution: {case}, {shape[0]} x {shape[1]} x {shape[2]}"
    try:
        from scipy import sparse
        from scipy.sparse import linalg
    except ImportError:
        return [not_run(16, name, INDEPENDENT_COMPUTATION, 1e-7, "SciPy not installed")]
    setup, unusable = build_case(case, shape)
    if unusable is not None:
        return [Row(16, name, INDEPENDENT_COMPUTATION, None, 1e-7, False, f"set-up not usable: {unusable}")]
    model = model_for(setup, 1e-10, 5000)
    u = model.solve(setup.rho, setup.loads)
    report = model.last_solve_report()
    reference = direct_solution(setup, sparse, linalg)
    error = float(np.linalg.norm(u - reference) / np.linalg.norm(reference))
    largest = float(np.abs(u - reference).max() / np.abs(reference).max())
    return [
        checked(
            16,
            name,
            INDEPENDENT_COMPUTATION,
            error,
            1e-7,
            f"|u - u_direct| / |u_direct|; {setup.description}; {model.multigrid_levels} multigrid levels, "
            f"{model.multigrid_dropped} coarsest degrees of freedom left out, {report['steps']} steps to 1e-10; "
            f"largest entry difference over the largest entry {g3(largest)}",
        )
    ]


# ---------------------------------------------------------------------------------------------------------
# Row 17. Limits: conjugate-gradient steps to a relative residual of 1e-8 on the cantilever of
# ``designs3d``: with uniform density at 24 x 12 x 12, 48 x 24 x 24 and 96 x 48 x 48, at most 30 at each
# size and the largest count at most 1.5 times the smallest (these two test that the multigrid is right);
# on the lattice design at 48 x 24 x 24 at most 300, on the irregular design at 48 x 24 x 24 at most 600
# (these test only that it converges). Every count is reported with the number of products with the fine
# stiffness matrix per solve.

ROW_17_UNIFORM_SIZES = ((24, 12, 12), (48, 24, 24), (96, 48, 48))
ROW_17_DESIGN_LIMITS = {"lattice": 300, "irregular": 600}
ROW_17_DESIGN_SHAPE = (48, 24, 24)


def steps_to_1e_8(setup) -> dict:
    model = model_for(setup, 1e-8, 10000)
    model.solve(setup.rho, setup.loads)
    return model.last_solve_report()


def row_17_uniform(sizes=ROW_17_UNIFORM_SIZES) -> list[Row]:
    reports = {shape: steps_to_1e_8(uniform(shape)) for shape in sizes}
    steps = [report["steps"] for report in reports.values()]
    spread = max(steps) / min(steps)
    passed = bool(max(steps) <= 30 and spread <= 1.5)
    counts = "; ".join(
        f"{a} x {b} x {c}: {r['steps']} steps, {r['products']} products" for (a, b, c), r in reports.items()
    )
    note = (
        f"largest step count; {counts}; largest over smallest {g3(spread)} (limit 1.5): "
        f"{'within' if spread <= 1.5 else 'above'}"
    )
    return [Row(17, "3D solver steps on a uniform cantilever at growing sizes", CONVERGENCE, float(max(steps)), 30,
                passed, note)]  # fmt: skip


def row_17_design(case: str, shape: tuple[int, int, int] = ROW_17_DESIGN_SHAPE) -> list[Row]:
    """Row 17 on the ``lattice`` or the ``irregular`` design."""
    limit = ROW_17_DESIGN_LIMITS[case]
    name = f"3D solver steps on the {case} 0-and-1 design, {shape[0]} x {shape[1]} x {shape[2]}"
    setup, unusable = build_case(case, shape)
    if unusable is not None:
        return [Row(17, name, CONVERGENCE, None, limit, False, f"set-up not usable: {unusable}")]
    report = steps_to_1e_8(setup)
    note = f"{setup.description}; {report['products']} products with the fine stiffness matrix"
    return [checked(17, name, CONVERGENCE, report["steps"], limit, note)]


# ---------------------------------------------------------------------------------------------------------
# Row 18. Limit: a cantilever of square section, length 8 times the depth, fully clamped at x = 0 (every
# displacement of every node there), with an end load P spread over the face x = L as the shear stress of
# beam theory (parabolic over the depth, uniform across the width, as consistent nodal forces), deflection
# read at the centre of the loaded face: against Timoshenko beam theory, PL^3/(3EI) + 6PL/(5GA), the error
# shrinks from 4 to 8 to 16 elements through the depth and is under 2 % at 16. Solves to 1e-10.

ROW_18_DEPTHS = (4, 8, 16)


def cantilever_deflection_3d(n: int) -> tuple[float, float]:
    """The deflection with n elements through the depth (unit elements: depth n mm), and Timoshenko's."""
    from michell.analysis3d import Grid3d
    from michell.verification.designs3d import clamped_face

    depth, p = float(n), 1000.0
    length = 8 * depth
    grid = Grid3d(8 * n, n, n, 1.0)
    held = clamped_face(grid)

    def traction(z):  # shear stress over the end face, integrating to P over depth * depth
        return 1.5 * p / (depth * depth) * (1 - (2 * z / depth - 1) ** 2)

    loads = np.zeros(grid.n_dofs)
    for j in range(n):  # each square piece [j, j+1] x [k, k+1]: linear in y, Simpson's rule in z
        for k in range(n):
            z0, z1 = float(k), float(k + 1)
            zm = (z0 + z1) / 2
            lower = (traction(z0) + 2 * traction(zm)) / 6
            upper = (2 * traction(zm) + traction(z1)) / 6
            for dj in (0, 1):
                loads[3 * grid.node(8 * n, j + dj, k) + 2] -= 0.5 * lower
                loads[3 * grid.node(8 * n, j + dj, k + 1) + 2] -= 0.5 * upper
    if abs(-loads.sum() - p) > 1e-12 * p:
        raise ValueError("the nodal forces do not add up to the end load")

    model = Elasticity3d(
        grid, held, youngs_modulus=E, poisson_ratio=NU, e_min=E_MIN_RATIO * E, penalty=3.0,
        tolerance=1e-10, max_steps=5000, threads=ROW_THREADS,
    )  # fmt: skip
    u = model.solve(np.ones(grid.n_elements), loads)
    deflection = -u[3 * grid.node(8 * n, n // 2, n // 2) + 2]
    inertia, area, shear_modulus = depth**4 / 12, depth**2, E / (2 * (1 + NU))
    timoshenko = p * length**3 / (3 * E * inertia) + 6 * p * length / (5 * shear_modulus * area)
    return float(deflection), timoshenko


def row_18_cantilever(depths=ROW_18_DEPTHS) -> list[Row]:
    """Row 18; ``depths`` (elements through the depth, increasing) defaults to 4, 8, 16, and the limit
    applies at the last."""
    errors = {}
    for n in depths:
        deflection, timoshenko = cantilever_deflection_3d(n)
        errors[n] = abs(deflection - timoshenko) / timoshenko
    values = list(errors.values())
    shrinks = all(a > b for a, b in zip(values, values[1:]))
    finest = depths[-1]
    passed = bool(shrinks and errors[finest] < 0.02)
    listed = ", ".join(str(n) for n in depths)
    note = (
        f"relative error with {finest} elements through the depth; with {listed}: "
        f"{', '.join(g3(e) for e in values)}; the error must also shrink: {'it does' if shrinks else 'it does not'}"
    )
    name = "3D cantilever, length 8 x depth, square section: deflection against Timoshenko beam theory"
    return [Row(18, name, CLOSED_FORM, errors[finest], 0.02, passed, note, strict=True)]
