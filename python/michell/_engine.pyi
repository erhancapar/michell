"""Type hints for the compiled Rust module (source: crates/michell-py/src/).

Every array argument must be a 1-D, contiguous float64 NumPy array (``Mma.update``'s ``d_limits`` and the
bodies of ``voxel_occupancy``: 2-D, C-contiguous float64 arrays); anything else is refused with
``EngineError`` (code ``invalid_input``), never converted. Every result is a new array.
"""

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]

class EngineError(Exception):
    """An error from the engine.

    ``code`` is stable (``invalid_input``, ``singular_system``, ``solver_not_converged``,
    ``internal_error``), ``message`` is for
    people, ``details`` holds the numbers involved. The checks written in Python raise it as well, with code
    ``invalid_input`` (see ``michell.errors``), and a refused problem file with a code of its own per kind
    of fault (``unknown_key``, ``missing_key``, ...; ``michell.errors.ProblemFileError``). An ``EngineError`` that leaves ``michell.optimise``
    also has ``partial_result``, the result of the passes completed before it (see ``optimise``).
    """

    code: str
    message: str
    details: dict[str, int | float]

def engine_version() -> str:
    """Return the version of the Rust engine crate."""

def element_stiffness(
    youngs_modulus: float, poisson_ratio: float, h: float, t: float
) -> npt.NDArray[np.float64]:
    """The 8 x 8 stiffness matrix of one square plane-stress element of side h and thickness t.

    Nodes counter-clockwise from the lower left; degrees of freedom u0, v0, u1, v1, u2, v2, u3, v3.
    """

def dot(a: Vector, b: Vector) -> float:
    """a . b of two arrays of the same length, summed in index order on one thread.

    Use it instead of ``a @ b`` for long vectors in the optimisation loop: the OpenBLAS build that NumPy
    ships runs a product of more than 10000 entries on every core, and its threads then keep the processor
    busy between calls.
    """

class Grid:
    """A grid of nx by ny square elements of side h mm and thickness t mm, y up.

    Element (i, j) has index i + nx*j; node (i, j) has index i + (nx+1)*j; node n has its x
    displacement at degree of freedom 2n and its y displacement at 2n+1. ``active`` (one entry per
    element, 1.0 inside the part, 0.0 outside) defaults to all ones.
    """

    nx: int
    ny: int
    h: float
    t: float
    voxel_volume_mm3: float
    """Volume of one element, h * h * t."""
    n_elements: int
    n_nodes: int
    n_dofs: int
    active: Vector

    def __init__(self, nx: int, ny: int, h: float, t: float, active: Vector | None = None) -> None: ...
    def node(self, i: int, j: int) -> int:
        """Index of node (i, j), 0 <= i <= nx, 0 <= j <= ny."""

    def element(self, i: int, j: int) -> int:
        """Index of element (i, j), 0 <= i < nx, 0 <= j < ny."""

class Elasticity:
    """Plane-stress elasticity on a grid, E(rho) = e_min + rho^penalty (youngs_modulus - e_min).

    ``held`` has one entry per degree of freedom, 1.0 to hold it. A degree of freedom that touches
    no active element is held at zero as well. A part that is not held against every rigid motion
    is refused with code ``singular_system``.
    """

    grid: Grid

    def __init__(
        self,
        grid: Grid,
        held: Vector,
        *,
        youngs_modulus: float,
        poisson_ratio: float,
        e_min: float,
        penalty: float,
    ) -> None: ...
    def solve(self, rho: Vector, loads: Vector, held_values: Vector | None = None) -> Vector:
        """Displacements u with K(rho) u = loads; loads on held degrees of freedom are ignored.

        ``held_values`` (one entry per degree of freedom) sets the displacement of the held ones;
        without it they stay at zero.
        """

    def adjoint(self, rho: Vector, rhs: Vector) -> Vector:
        """lambda with K(rho) lambda = rhs, every held degree of freedom at zero."""

    def sensitivity(self, rho: Vector, u: Vector, lam: Vector) -> Vector:
        """Per element, -lam_e^T (dK/drho_e) u_e; for compliance (lam = u) the whole gradient."""

    def element_strain_energy(self, rho: Vector, u: Vector) -> Vector:
        """Per element, 1/2 u_e^T K_e(rho) u_e in N mm; the compliance is twice the sum."""

    def element_stress(self, rho: Vector, u: Vector) -> npt.NDArray[np.float64]:
        """Stress at each element's centre with modulus E(rho_e), MPa; shape (n_elements, 3):
        sx, sy, txy. Inactive elements give zeros."""

def element_stiffness_3d(youngs_modulus: float, poisson_ratio: float, h: float) -> npt.NDArray[np.float64]:
    """The 24 x 24 stiffness matrix of one cube element of side h (8 nodes, 2 x 2 x 2 Gauss points).

    Nodes: the face z = 0 counter-clockwise from (0, 0, 0) seen from above, then the face z = h in the
    same order; degrees of freedom u, v, w of node 0, then of node 1, and so on.
    """

class Grid3d:
    """A grid of nx by ny by nz cube elements of side h mm.

    Element (i, j, k) has index i + nx*(j + ny*k); node (i, j, k) has index i + (nx+1)*(j + (ny+1)*k);
    node n has its x, y and z displacement at degrees of freedom 3n, 3n+1 and 3n+2. ``active`` (one entry
    per element, 1.0 inside the part, 0.0 outside) defaults to all ones.
    """

    nx: int
    ny: int
    nz: int
    h: float
    voxel_volume_mm3: float
    """Volume of one element, h ** 3."""
    n_elements: int
    n_nodes: int
    n_dofs: int
    active: Vector

    def __init__(self, nx: int, ny: int, nz: int, h: float, active: Vector | None = None) -> None: ...
    def node(self, i: int, j: int, k: int) -> int:
        """Index of node (i, j, k), 0 <= i <= nx, 0 <= j <= ny, 0 <= k <= nz."""

    def element(self, i: int, j: int, k: int) -> int:
        """Index of element (i, j, k), 0 <= i < nx, 0 <= j < ny, 0 <= k < nz."""

class Elasticity3d:
    """Linear elasticity on a 3D grid, E(rho) = e_min + rho^penalty (youngs_modulus - e_min), isotropic.

    ``held`` has one entry per degree of freedom (three per node), 1.0 to hold it. A degree of freedom
    that touches no active element is held at zero as well. Building the object raises ``EngineError``
    with code ``singular_system`` when a body of face-joined active elements is not held against every
    rigid motion. The stiffness matrix is never assembled; a solve runs conjugate gradients
    preconditioned by one multigrid V-cycle until the relative residual |b - K u| / |b| is at most
    ``tolerance``, and raises ``EngineError`` with code ``solver_not_converged`` (details: steps,
    relative_residual) after ``max_steps`` steps. Every call runs on ``threads`` threads (at least 1);
    the numbers do not depend on the thread count.
    """

    grid: Grid3d
    tolerance: float
    max_steps: int
    threads: int
    multigrid_levels: int
    """The number of multigrid levels, the finest included (1: the grid is solved directly)."""
    multigrid_dropped: int
    """The number of degrees of freedom of the coarsest multigrid level that its direct solve leaves out."""

    def __init__(
        self,
        grid: Grid3d,
        held: Vector,
        *,
        youngs_modulus: float,
        poisson_ratio: float,
        e_min: float,
        penalty: float,
        tolerance: float = 1e-6,
        max_steps: int = 5000,
        threads: int = 1,
    ) -> None: ...
    def solve(
        self, rho: Vector, loads: Vector, held_values: Vector | None = None, start: Vector | None = None
    ) -> Vector:
        """Displacements u with K(rho) u = loads; loads on held degrees of freedom are ignored.

        ``held_values`` (one entry per degree of freedom) sets the displacement of the held ones; without
        it they stay at zero. ``start`` is the first guess of u (its held entries are ignored); without it
        the solve starts from zero.
        """

    def adjoint(self, rho: Vector, rhs: Vector, start: Vector | None = None) -> Vector:
        """lambda with K(rho) lambda = rhs, every held degree of freedom at zero."""

    def last_solve_report(self) -> dict[str, int | float] | None:
        """``{"steps": ..., "relative_residual": ..., "products": ...}`` of the last ``solve`` or
        ``adjoint`` (``products``: the products with the fine stiffness matrix, the multigrid smoother's
        included); None before the first one and after one that raised an error."""

    def product(self, rho: Vector, x: Vector) -> Vector:
        """K(rho) x without assembling K; held degrees of freedom and those that touch no active element
        act as identity rows and columns."""

    def sensitivity(self, rho: Vector, u: Vector, lam: Vector) -> Vector:
        """Per element, -lam_e^T (dK/drho_e) u_e; for compliance (lam = u) the whole gradient."""

    def element_strain_energy(self, rho: Vector, u: Vector) -> Vector:
        """Per element, 1/2 u_e^T K_e(rho) u_e in N mm; the compliance is twice the sum."""

    def element_stress(self, rho: Vector, u: Vector) -> npt.NDArray[np.float64]:
        """Stress at each element's centre with modulus E(rho_e), MPa; shape (n_elements, 6):
        sxx, syy, szz, syz, szx, sxy (shear components with G times the engineering shear strain).
        Inactive elements give zeros."""

class DensityFilter:
    """The density filter with cone weights, over the active elements of a grid:
    rho_e = sum_j w_ej x_j / sum_j w_ej, w_ej = max(0, radius - distance between centres), radius in mm.
    On a ``Grid`` it runs on one thread (``threads`` must be 1); on a ``Grid3d`` on ``threads`` threads,
    as a long call. ``outside`` ("closed" or "empty", for every face or as a dict by face "x-", "x+",
    "y-", "y+", "z-", "z+"; faces left out are closed) and ``inactive`` ("closed" or "empty") say what lies
    beyond each face and in the inactive elements.
    """

    radius: float
    element_size: float
    is_3d: bool
    threads: int
    outside: dict[str, str]
    inactive: str

    def __init__(
        self,
        grid: Grid | Grid3d,
        radius: float,
        threads: int = 1,
        *,
        outside: str | dict[str, str] = "closed",
        inactive: str = "closed",
    ) -> None: ...
    def forward(self, x: Vector) -> Vector:
        """rho, one value per element (0 on inactive elements)."""

    def backward(self, x: Vector, grad: Vector) -> Vector:
        """dg/dx from dg/drho; x is the field the filter was applied to (only its length is checked)."""

class HelmholtzFilter:
    """The PDE (Helmholtz) filter on a 3D grid (Lazarov and Sigmund 2011): rho solves
    -r^2 lap(psi) + psi = x on the active elements with a zero-flux boundary, r = radius / (2 sqrt 3),
    radius in mm; solved by conjugate gradients to the relative residual ``tolerance`` on ``threads``
    threads."""

    radius: float
    length: float
    element_size: float
    tolerance: float
    threads: int

    def __init__(self, grid: Grid3d, radius: float, threads: int = 1, *, tolerance: float = 1e-10) -> None: ...
    def forward(self, x: Vector) -> Vector:
        """rho, one value per element (0 on inactive elements); a long call."""

    def backward(self, x: Vector, grad: Vector) -> Vector:
        """dg/dx from dg/drho: the same symmetric map applied to grad; x is only checked for its length."""

class Projection:
    """The projection (Wang, Lazarov and Sigmund 2011), applied to each value on its own:
    P(x) = (tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))), with
    beta > 0 and 0 <= eta <= 1. P(0) = 0 and P(1) = 1."""

    beta: float
    eta: float

    def __init__(self, beta: float, eta: float = 0.5) -> None: ...
    def forward(self, x: Vector) -> Vector:
        """P of each entry of x."""

    def backward(self, x: Vector, grad: Vector) -> Vector:
        """dg/dx from dg/dP: each entry of grad times P' at the same entry of x."""

def member_size(
    grid: Grid | Grid3d, filtered: Vector, projection: Projection, thresholds: Vector, c: float
) -> tuple[float, Vector]:
    """The member-size constraint on the solid phase (Zhou, Lazarov, Wang and Sigmund 2015, eqs. 8 and 10),
    g = (1/n) sum_e P(f_e) exp(-c |grad f|_e^2) [min(f_e - t_e, 0)]^2 over the active elements, with the
    gradient per element side, n the number of active elements and t_e the entry of ``thresholds`` (one per
    element, each between 0 and 1) for element e; returns g and dg/df."""

def voxel_occupancy(
    counts: tuple[int, int, int],
    voxel_mm: float,
    origin_mm: tuple[float, float, float],
    adding: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    subtracting: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = ...,
    rays_per_side: int = 3,
    threads: int = 1,
) -> Vector:
    """The occupancy of every voxel of a grid of ``counts`` (nx, ny, nz) voxels of side ``voxel_mm``, the
    corner of voxel (0, 0, 0) at ``origin_mm`` in the meshes' coordinates: the share of the voxel inside the
    union of the ``adding`` bodies minus the union of the ``subtracting`` bodies, one entry per voxel in the
    order of ``Grid3d``. Each body is a pair (vertices, triangles), float64 arrays of shape (n, 3) in mm and
    (m, 3) of whole-number vertex indices, closed and facing outward. Rays along x through every voxel
    column, ``rays_per_side`` x ``rays_per_side`` of them at the centres of its sub-squares; on each ray the
    length inside every voxel is exact. A ray through an edge or a vertex counts it once (see
    ``docs/design/geometry.md``); triangles parallel to the rays are ignored. Runs on ``threads`` threads;
    the numbers do not depend on the thread count or on the order of the triangles. A body that a ray finds
    not closed is refused with code ``invalid_input``, naming the body and the ray."""

def iso_surface(
    counts: tuple[int, int, int],
    spacing_mm: float,
    origin_mm: tuple[float, float, float],
    values: Vector,
    level: float,
    threads: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The closed surface where ``values`` equal ``level`` on a lattice of ``counts`` (nx, ny, nz) points
    ``spacing_mm`` apart, point (0, 0, 0) at ``origin_mm``; ``values`` holds one sample per point, x fastest.
    Marching tetrahedra on the split of every cell into six tetrahedra around the same body diagonal; a
    sample is inside when it is at least the level, and every vertex keeps a thousandth of its edge from
    both ends (see ``docs/design/geometry.md``). Returns the vertices, shape (n, 3) in mm, and the triangles,
    shape (m, 3) of whole-number vertex indices as float64, counter-clockwise seen from outside (the side of
    larger values is inside). Every sample on the lattice's outer layer must be below the level; a field
    whose outer layer reaches it, or that reaches it nowhere, is refused with code ``invalid_input``. Runs
    on ``threads`` threads; the result does not depend on the thread count."""

class Oc:
    """The optimality-criteria update of the 88-line code: one goal, one limit linear in the density."""

    move_limit: float

    def __init__(self, grid: Grid, move_limit: float = 0.2) -> None: ...
    def update(
        self,
        x: Vector,
        d_goal: Vector,
        d_limit: Vector,
        limit_weights: Vector,
        limit_bound: float,
        filter: DensityFilter | None = None,
    ) -> Vector:
        """The new design variables. d_goal and d_limit are gradients with respect to x; the limit,
        sum(limit_weights * rho) <= limit_bound, is measured on rho = filter.forward(x_new) inside the
        bisection (rho = x_new without a filter). Inactive elements keep their value."""

class Mma:
    """The method of moving asymptotes (Svanberg 1987, in the form of Svanberg's 2007 note) for variables
    between ``x_min`` and ``x_max`` and ``n_limits`` limits written f_i(x) <= 0.

    The object remembers the two previous designs and the asymptotes, so one ``Mma`` belongs to one run.
    Every update runs on ``threads`` threads (default 1, at least 1); the designs do not depend on how many
    there are. ``move_limit`` is the largest move of a variable in one update, as a fraction of
    ``x_max - x_min``: a number above 0 and at most 1, by default 0.5 (the value of Svanberg's note);
    another value raises ``EngineError`` with code ``invalid_input``.
    """

    n_variables: int
    n_limits: int
    threads: int
    """The number of threads every update runs on."""
    move_limit: float
    """The largest move of a variable in one update, as a fraction of ``x_max - x_min``."""
    iteration: int
    """How many updates have been made."""

    def __init__(
        self, x_min: Vector, x_max: Vector, n_limits: int, threads: int = 1, move_limit: float = 0.5
    ) -> None: ...
    def asymptotes(self) -> tuple[Vector, Vector]:
        """The lower and upper asymptotes of the last update."""

    def update(
        self, x: Vector, goal: float, d_goal: Vector, limits: Vector, d_limits: npt.NDArray[np.float64]
    ) -> Vector:
        """The next design. ``goal`` and ``d_goal``: f_0(x) and its gradient; ``limits``: f_i(x), one
        per limit; ``d_limits``: their gradients, a C-contiguous array of shape (n_limits, n_variables).
        A subproblem the solver cannot finish raises ``EngineError`` with code ``solver_not_converged``."""
