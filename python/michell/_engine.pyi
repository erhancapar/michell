"""Type hints for the compiled Rust module (source: crates/michell-py/src/).

Every array argument must be a 1-D, contiguous float64 NumPy array (``Mma.update``'s ``d_limits``: a 2-D,
C-contiguous float64 array); anything else is refused with ``EngineError`` (code ``invalid_input``), never
converted. Every result is a new array.
"""

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]

class EngineError(Exception):
    """An error from the engine.

    ``code`` is stable (``invalid_input``, ``singular_system``, ``solver_not_converged``,
    ``internal_error``), ``message`` is for
    people, ``details`` holds the numbers involved.
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

class DensityFilter:
    """The density filter with cone weights, over the active elements of a grid:
    rho_e = sum_j w_ej x_j / sum_j w_ej, w_ej = max(0, radius - distance between centres), radius in mm.
    """

    radius: float

    def __init__(self, grid: Grid, radius: float) -> None: ...
    def forward(self, x: Vector) -> Vector:
        """rho, one value per element (0 on inactive elements)."""

    def backward(self, x: Vector, grad: Vector) -> Vector:
        """dg/dx from dg/drho; x is the field the filter was applied to (only its length is checked)."""

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
    """

    n_variables: int
    n_limits: int
    iteration: int
    """How many updates have been made."""

    def __init__(self, x_min: Vector, x_max: Vector, n_limits: int) -> None: ...
    def asymptotes(self) -> tuple[Vector, Vector]:
        """The lower and upper asymptotes of the last update."""

    def update(
        self, x: Vector, goal: float, d_goal: Vector, limits: Vector, d_limits: npt.NDArray[np.float64]
    ) -> Vector:
        """The next design. ``goal`` and ``d_goal``: f_0(x) and its gradient; ``limits``: f_i(x), one
        per limit; ``d_limits``: their gradients, a C-contiguous array of shape (n_limits, n_variables).
        A subproblem the solver cannot finish raises ``EngineError`` with code ``solver_not_converged``."""
