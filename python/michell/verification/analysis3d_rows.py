"""Verification rows 11 to 15: the 3D analysis against an exact integration, closed forms and an
independently assembled stiffness matrix, and the same numbers on any thread count (the Helmholtz filter
included).

Each row's measure, set-up and limit were fixed before any result was computed. Unless a row says
otherwise: unit cube elements, Poisson ratio 0.3, solves to a relative residual of 1e-10.
"""

from __future__ import annotations

import itertools

import numpy as np

from michell._engine import element_stiffness_3d
from michell.analysis3d import Elasticity3d, Grid3d
from michell.filters import HelmholtzFilter
from michell.row import CLOSED_FORM, INDEPENDENT_COMPUTATION, REPEATABILITY, Row, checked, g3, not_run

E, NU = 210000.0, 0.3
SOLVE_TOLERANCE = 1e-10

# The natural coordinates (-1 or 1) of the element's eight nodes: the face z = 0 counter-clockwise seen
# from above, then the face z = h in the same order (the order of ``element_stiffness_3d``).
NODES = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)


def material_matrix_3d(e: float, nu: float) -> np.ndarray:
    """D with stress = D strain, order xx, yy, zz, yz, zx, xy, engineering shear strains."""
    lam, g = e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[[0, 1, 2], [0, 1, 2]] = lam + 2 * g
    d[[3, 4, 5], [3, 4, 5]] = g
    return d


def node_coordinates_3d(grid) -> np.ndarray:
    """x, y and z of every node, in node order (x fastest), shape (n_nodes, 3)."""
    k, j, i = np.meshgrid(np.arange(grid.nz + 1), np.arange(grid.ny + 1), np.arange(grid.nx + 1), indexing="ij")
    return grid.h * np.column_stack([i.ravel(), j.ravel(), k.ravel()]).astype(float)


def boundary_node_mask(grid) -> np.ndarray:
    """True for the nodes on the outer faces of the grid."""
    k, j, i = np.meshgrid(np.arange(grid.nz + 1), np.arange(grid.ny + 1), np.arange(grid.nx + 1), indexing="ij")
    on_face = (i == 0) | (i == grid.nx) | (j == 0) | (j == grid.ny) | (k == 0) | (k == grid.nz)
    return on_face.ravel()


def solve_with_boundary_field_3d(grid, gradient: np.ndarray, offset: np.ndarray):
    """Hold every boundary node at the linear displacement u(x) = offset + gradient @ x; no loads. Returns
    the model, the displacements, the exact displacements and the boundary node mask."""
    exact = (node_coordinates_3d(grid) @ gradient.T + offset).ravel()  # node dofs together
    boundary = boundary_node_mask(grid)
    held = np.repeat(boundary, 3)
    model = Elasticity3d(
        grid, held, youngs_modulus=E, poisson_ratio=NU, e_min=1e-9 * E, penalty=3.0,
        tolerance=SOLVE_TOLERANCE, max_steps=2000,
    )  # fmt: skip
    u = model.solve(np.ones(grid.n_elements), np.zeros(grid.n_dofs), held_values=exact)
    return model, u, exact, boundary


def strain_of_gradient(gradient: np.ndarray) -> np.ndarray:
    """The strain of the field u = gradient @ x, order xx, yy, zz, yz, zx, xy, engineering shear."""
    g = gradient
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 2] + g[2, 1], g[2, 0] + g[0, 2], g[0, 1] + g[1, 0]])


# ---------------------------------------------------------------------------------------------------------
# Row 11. Limits: the element matrix equals an exact integration from the shape functions to a relative
# 1e-12 (largest entry difference over the largest entry, worst of three sets of modulus, Poisson ratio and
# side); it is symmetric to 1e-14 of its largest entry; it has exactly six zero eigenvalues (below 1e-10
# of the largest): the sixth smallest must be below that bound and the seventh above it.

ROW_11_CASES = [(1.0, 0.3, 1.0), (210000.0, 0.3, 2.5), (70000.0, 0.33, 0.1)]


def exact_element_matrix(e: float, nu: float, h: float) -> np.ndarray:
    """K of the 8-node cube of side h, integrated exactly, in index form:

        K[3a+i, 3b+j] = integral of dN_a/dx_p C_ipjq dN_b/dx_q dV,
        C_ipjq = lambda d_ip d_jq + G (d_ij d_pq + d_iq d_jp).

    With N_a = (1 + xi xi_a)(1 + eta eta_a)(1 + zeta zeta_a) / 8 and d/dx = (2/h) d/dxi, each integral is
    a product of three one-dimensional integrals over [-1, 1]:
        a derivative factor with a derivative factor: s_a s_b * 2,
        a derivative factor with a linear factor: s_a * integral of (1 + t s_b) dt = 2 s_a,
        two linear factors: integral of (1 + t s_a)(1 + t s_b) dt = 2 + 2 s_a s_b / 3.
    No quadrature is used, so this does not share the engine's method.
    """
    lam, g = e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))
    delta = np.eye(3)
    c = np.zeros((3, 3, 3, 3))
    for i, p, j, q in itertools.product(range(3), repeat=4):
        c[i, p, j, q] = lam * delta[i, p] * delta[j, q] + g * (delta[i, j] * delta[p, q] + delta[i, q] * delta[j, p])

    def one_dimensional(sa: float, sb: float, a_derived: bool, b_derived: bool) -> float:
        if a_derived and b_derived:
            return 2 * sa * sb
        if a_derived:
            return 2 * sa
        if b_derived:
            return 2 * sb
        return 2 + 2 * sa * sb / 3

    # dN/dx_p dN/dx_q dV = (2/h)^2 (1/8)^2 (h/2)^3 times the product of the three integrals.
    scale = (2 / h) ** 2 / 64 * (h / 2) ** 3
    gradient_products = np.zeros((8, 8, 3, 3))  # [a, b, p, q]
    for a, b, p, q in itertools.product(range(8), range(8), range(3), range(3)):
        product = 1.0
        for r in range(3):
            product *= one_dimensional(NODES[a, r], NODES[b, r], r == p, r == q)
        gradient_products[a, b, p, q] = scale * product

    k = np.zeros((24, 24))
    for a, b, i, j in itertools.product(range(8), range(8), range(3), range(3)):
        k[3 * a + i, 3 * b + j] = np.sum(gradient_products[a, b] * c[i, :, j, :])
    return k


def row_11_element_matrix() -> list[Row]:
    differences, asymmetries, sixth, seventh = [], [], [], []
    for e, nu, h in ROW_11_CASES:
        ours = element_stiffness_3d(e, nu, h)
        largest = np.abs(ours).max()
        differences.append(np.abs(ours - exact_element_matrix(e, nu, h)).max() / largest)
        asymmetries.append(np.abs(ours - ours.T).max() / largest)
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh((ours + ours.T) / 2)))
        sixth.append(eigenvalues[5] / eigenvalues[-1])
        seventh.append(eigenvalues[6] / eigenvalues[-1])
    cases = f"worst of {len(ROW_11_CASES)} sets of modulus, Poisson ratio and side"
    six_zero = bool(max(sixth) < 1e-10 and min(seventh) > 1e-10)
    return [
        checked(
            11,
            "3D element stiffness matrix against an exact integration of the shape functions",
            CLOSED_FORM,
            max(differences),
            1e-12,
            f"largest entry difference relative to the largest entry, {cases}",
        ),
        checked(
            11,
            "3D element stiffness matrix is symmetric",
            CLOSED_FORM,
            max(asymmetries),
            1e-14,
            f"largest difference between K and its transpose relative to the largest entry, {cases}",
        ),
        Row(
            11,
            "3D element stiffness matrix has exactly six zero eigenvalues (the rigid motions)",
            CLOSED_FORM,
            float(max(sixth)),
            1e-10,
            six_zero,
            f"the sixth smallest eigenvalue relative to the largest, {cases}; the seventh must be above the "
            f"limit: its smallest relative value is {g3(min(seventh))}",
            strict=True,
        ),
    ]


# ---------------------------------------------------------------------------------------------------------
# Row 12. Limit: a linear displacement field put on the boundary of a 4 x 3 x 5 block is reproduced inside
# to 1e-8, with uniform stress (displacements relative to the largest value of the exact field, stresses
# relative to the largest exact stress component).


def row_12_patch_test() -> list[Row]:
    grid = Grid3d(4, 3, 5, 1.0)
    gradient = np.array([[2e-3, -1e-3, 5e-4], [5e-4, 1.5e-3, -7e-4], [-3e-4, 8e-4, -1.2e-3]])
    offset = np.array([0.01, -0.02, 0.015])
    model, u, exact, boundary = solve_with_boundary_field_3d(grid, gradient, offset)
    inside = np.repeat(~boundary, 3)
    displacement_error = np.abs(u - exact)[inside].max() / np.abs(exact).max()

    stress = model.element_stress(np.ones(grid.n_elements), u)
    expected_stress = material_matrix_3d(E, NU) @ strain_of_gradient(gradient)
    stress_error = np.abs(stress - expected_stress).max() / np.abs(expected_stress).max()
    steps = model.last_solve_report()["steps"]
    return [
        checked(
            12,
            "3D patch test: a linear displacement field and its uniform stress",
            CLOSED_FORM,
            max(displacement_error, stress_error),
            1e-8,
            f"relative error, the larger of the inside displacements ({g3(displacement_error)}) and the stress "
            f"({g3(stress_error)}); 4 x 3 x 5 elements, {steps} conjugate-gradient steps",
        )
    ]


# ---------------------------------------------------------------------------------------------------------
# Row 13. Limit: in pure shear in each of the three coordinate planes, the element shear stress equals
# G * gamma to 1e-8 (relative), every other component zero to 1e-8 of G * gamma. The shear field is put on
# the boundary of a 4 x 4 x 4 block: u_a = gamma / 2 * x_b and u_b = gamma / 2 * x_a in the plane (a, b).

ROW_13_PLANES = [("yz", 1, 2, 3), ("zx", 2, 0, 4), ("xy", 0, 1, 5)]  # name, axis a, axis b, stress entry


def row_13_pure_shear() -> list[Row]:
    gamma = 1e-3
    g_gamma = E / (2 * (1 + NU)) * gamma
    grid = Grid3d(4, 4, 4, 1.0)
    rows = []
    for name, a, b, entry in ROW_13_PLANES:
        gradient = np.zeros((3, 3))
        gradient[a, b] = gradient[b, a] = gamma / 2
        model, u, _, _ = solve_with_boundary_field_3d(grid, gradient, np.zeros(3))
        stress = model.element_stress(np.ones(grid.n_elements), u)
        shear_error = np.abs(stress[:, entry] - g_gamma).max() / g_gamma
        others = np.delete(stress, entry, axis=1)
        other_error = np.abs(others).max() / g_gamma
        rows.append(
            checked(
                13,
                f"3D pure shear in the {name} plane: shear stress equal to G gamma, other components zero",
                CLOSED_FORM,
                max(shear_error, other_error),
                1e-8,
                f"relative to G gamma, the larger of the shear stress error ({g3(shear_error)}) and the other "
                f"five components ({g3(other_error)}); 4 x 4 x 4 elements",
            )
        )
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 14. Limit: the matrix-free product K(rho) x equals the product with the matrix assembled independently
# with SciPy from the reference element matrix, to a relative 1e-12 (largest entry difference over the
# largest entry of the assembled product). Set-up: 7 x 5 x 3 elements; the elements with i >= 5 and j >= 3
# outside the part; random densities between 0 and 1 (E_min = 1e-9 E); the face x = 0 clamped and the z
# displacement held on the face z = 3; a random vector x. Held degrees of freedom and those that touch no
# active element are identity rows and columns in both.


def assembled_matrix(grid, active, rho, held, scipy_sparse):
    """K(rho) assembled element by element in SciPy, with identity rows and columns for the held degrees
    of freedom and for those that touch no active element."""
    k_unit = element_stiffness_3d(1.0, NU, grid.h)
    e_min = 1e-9 * E
    steps = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    rows, cols, values = [], [], []
    touched = np.zeros(grid.n_dofs, dtype=bool)
    for k, j, i in itertools.product(range(grid.nz), range(grid.ny), range(grid.nx)):
        element = i + grid.nx * (j + grid.ny * k)
        if not active[element]:
            continue
        nodes = [(i + di) + (grid.nx + 1) * ((j + dj) + (grid.ny + 1) * (k + dk)) for di, dj, dk in steps]
        dofs = np.array([3 * n + c for n in nodes for c in range(3)])
        touched[dofs] = True
        modulus = e_min + rho[element] ** 3 * (E - e_min)
        rows.append(np.repeat(dofs, 24))
        cols.append(np.tile(dofs, 24))
        values.append((modulus * k_unit).ravel())
    shape = (grid.n_dofs, grid.n_dofs)
    k = scipy_sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape)
    fixed = held | ~touched
    free = scipy_sparse.diags((~fixed).astype(float))
    return (free @ k.tocsr() @ free + scipy_sparse.diags(fixed.astype(float))).tocsr()


def row_14_product() -> list[Row]:
    name = "3D matrix-free product K(rho) x against the matrix assembled with SciPy"
    try:
        from scipy import sparse
    except ImportError:
        return [not_run(14, name, INDEPENDENT_COMPUTATION, 1e-12, "SciPy not installed")]
    nx, ny, nz = 7, 5, 3
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))  # x fastest within each layer
    layer = ~((i >= 5) & (j >= 3)).ravel()
    active = np.tile(layer, nz)
    grid = Grid3d(nx, ny, nz, 1.0, active=active)
    random = np.random.default_rng(14)
    rho = random.uniform(0.0, 1.0, grid.n_elements)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k, j_, i_ in itertools.product(range(nz + 1), range(ny + 1), range(nx + 1)):
        node = grid.node(i_, j_, k)
        if i_ == 0:
            held[3 * node : 3 * node + 3] = True
        if k == nz:
            held[3 * node + 2] = True
    model = Elasticity3d(grid, held, youngs_modulus=E, poisson_ratio=NU, e_min=1e-9 * E, penalty=3.0)
    x = random.standard_normal(grid.n_dofs)

    ours = model.product(rho, x)
    expected = assembled_matrix(grid, active, rho, held, sparse) @ x
    error = np.abs(ours - expected).max() / np.abs(expected).max()
    return [
        checked(
            14,
            name,
            INDEPENDENT_COMPUTATION,
            error,
            1e-12,
            f"largest entry difference relative to the largest entry; {nx} x {ny} x {nz} elements, "
            f"{int((~active).sum())} outside the part, random densities, {int(held.sum())} held degrees of freedom",
        )
    ]


# ---------------------------------------------------------------------------------------------------------
# Row 15. Limit: runs on 1, 2, 5 and 12 threads, and a repeated run on 12 threads, give bit-for-bit the same
# numbers: the largest absolute difference from the run on 1 thread is exactly 0, and every array is equal
# byte for byte. Compared: the product K(rho) x; a complete solve to 1e-10 (its displacements, its final
# relative residual and its step count: the solver's sums decide every step length, so a sum that differed
# would change them); the per-element strain energy, sensitivity and stress of that solution; the forward
# and the backward map of the Helmholtz filter (radius 2.5 h, built on the same thread count) on the
# densities and on the strain energy. Set-up: 19 x
# 15 x 15 elements, so that the sums span more blocks than the largest thread count and every thread gets
# work; the elements with i >= 13 and j >= 10 outside the part; random densities between 0.3 and 1 (E_min =
# 1e-9 E); the face x = 0 held at a prescribed displacement (so the held values enter the right-hand side)
# and a load spread over the far face of the part (x = 19, j <= 10); a random vector x.

ROW_15_THREADS = (1, 2, 5, 12)
# The length of the blocks the engine's sums run over (`BLOCK_LENGTH` in crates/michell/src/threads.rs),
# used here only to state in the note how many blocks the set-up spans.
SUM_BLOCK_LENGTH = 1024


def row_15_thread_count() -> list[Row]:
    nx, ny, nz = 19, 15, 15
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))  # x fastest within each layer
    active = np.tile(~((i >= 13) & (j >= 10)).ravel(), nz)
    grid = Grid3d(nx, ny, nz, 1.0, active=active)
    random = np.random.default_rng(15)
    rho = random.uniform(0.3, 1.0, grid.n_elements)
    x = random.standard_normal(grid.n_dofs)
    held = np.zeros(grid.n_dofs, dtype=bool)
    held_values = np.zeros(grid.n_dofs)
    loads = np.zeros(grid.n_dofs)
    for k, j_ in itertools.product(range(nz + 1), range(ny + 1)):
        node = grid.node(0, j_, k)
        held[3 * node : 3 * node + 3] = True
        held_values[3 * node : 3 * node + 3] = [1e-4 * j_, -2e-4 * k, 5e-5]  # a stretch and a shift
        if j_ <= 10:  # the far face, where it belongs to the part
            loads[3 * grid.node(nx, j_, k) + 2] = -1.0 / (11 * (nz + 1))

    def outputs(model, pde_filter) -> dict[str, np.ndarray]:
        u = model.solve(rho, loads, held_values=held_values)
        report = model.last_solve_report()
        energy = model.element_strain_energy(rho, u)
        return {
            "product": model.product(rho, x),
            "displacements": u,
            "relative residual": np.array([report["relative_residual"]]),
            "steps": np.array([float(report["steps"])]),
            "strain energy": energy,
            "sensitivity": model.sensitivity(rho, u, u),
            "stress": model.element_stress(rho, u),
            "filter forward": pde_filter.forward(rho),
            "filter backward": pde_filter.backward(rho, energy),
        }

    runs = []
    for threads in ROW_15_THREADS:
        model = Elasticity3d(
            grid, held, youngs_modulus=E, poisson_ratio=NU, e_min=1e-9 * E, penalty=3.0,
            tolerance=SOLVE_TOLERANCE, max_steps=5000, threads=threads,
        )  # fmt: skip
        pde_filter = HelmholtzFilter(grid, 2.5, threads=threads)
        runs.append(outputs(model, pde_filter))
    runs.append(outputs(model, pde_filter))  # the same objects on 12 threads, called again

    first = runs[0]
    largest, identical = 0.0, True
    for run in runs[1:]:
        for name, values in first.items():
            largest = max(largest, float(np.abs(run[name] - values).max()))
            identical = identical and run[name].tobytes() == values.tobytes()
    blocks = -(-grid.n_dofs // SUM_BLOCK_LENGTH)
    threads = ", ".join(str(t) for t in ROW_15_THREADS[:-1]) + f" and {ROW_15_THREADS[-1]}"
    return [
        Row(
            15,
            f"3D analysis and Helmholtz filter give the same numbers on {threads} threads and on a repeated run",
            REPEATABILITY,
            largest,
            0.0,
            bool(identical and largest == 0.0),
            f"largest absolute difference from the run on 1 thread, over the product, a complete solve "
            f"({int(first['steps'][0])} steps to {SOLVE_TOLERANCE:g}: displacements, final residual, step count) "
            f"the strain energy, sensitivity and stress of its solution, and the Helmholtz filter forward and "
            f"backward; every array also compared byte for "
            f"byte; {nx} x {ny} x {nz} elements, {grid.n_dofs} degrees of freedom ({blocks} blocks of a sum), "
            f"{int((~active).sum())} outside the part, random densities, held displacements",
        )
    ]
