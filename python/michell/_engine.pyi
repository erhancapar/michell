"""Type hints for the compiled Rust module (source: crates/michell-py/src/).

Every array argument must be a 1-D, contiguous float64 NumPy array; anything else is refused with
``EngineError`` (code ``invalid_input``), never converted. Every result is a new array.
"""

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]

class EngineError(Exception):
    """An error from the engine.

    ``code`` is stable (``invalid_input``, ``singular_system``, ``internal_error``), ``message`` is for
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
