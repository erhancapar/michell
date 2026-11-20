"""The engine's 3D analysis objects, as the package offers them: masks may be boolean arrays.

As in ``michell.analysis`` for 2D, these subclasses turn a boolean mask into float64 0.0 / 1.0 before the
engine sees it; any other argument is passed on unchanged.
"""

from __future__ import annotations

from michell import _engine
from michell.analysis import as_mask


class Grid3d(_engine.Grid3d):
    """A grid of nx by ny by nz cube elements (voxels) of side h mm.

    Element (i, j, k) has index ``i + nx * (j + ny * k)``; node (i, j, k) has index
    ``i + (nx + 1) * (j + (ny + 1) * k)``; node n has its x, y and z displacement at degrees of freedom
    3n, 3n + 1 and 3n + 2. ``active`` (one entry per element) may be a boolean array or float64 0.0 / 1.0;
    all active if omitted.
    """

    def __new__(cls, nx, ny, nz, h, active=None):
        return super().__new__(cls, nx, ny, nz, h, as_mask(active))


class Elasticity3d(_engine.Elasticity3d):
    """Linear elasticity on a 3D grid, with the calls of the 2D ``Elasticity``.

    ``held`` (one entry per degree of freedom, three per node) may be a boolean array or float64
    0.0 / 1.0. The stiffness matrix is never assembled: ``solve`` and ``adjoint`` use conjugate gradients
    with a diagonal (Jacobi) preconditioner, stopped at the relative residual ``tolerance`` (default
    1e-6); more than ``max_steps`` steps (default 500) raise ``EngineError`` with code
    ``solver_not_converged``. Both take a first guess ``start=``; nothing is kept from one call to the
    next except ``last_solve_report()``. ``element_stress`` returns six components per element: xx, yy,
    zz, yz, zx, xy.

    Every call runs on ``threads`` threads (default 1, at least 1; more threads than processor cores is
    allowed). The work is shared out so that the numbers do not depend on how many threads there are.

    Whether the part is held against every rigid motion is not checked: a part that is not held usually
    ends in ``solver_not_converged``.
    """

    def __new__(
        cls, grid, held, *, youngs_modulus, poisson_ratio, e_min, penalty, tolerance=1e-6, max_steps=500, threads=1
    ):
        return super().__new__(
            cls,
            grid,
            as_mask(held),
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            e_min=e_min,
            penalty=penalty,
            tolerance=tolerance,
            max_steps=max_steps,
            threads=threads,
        )
