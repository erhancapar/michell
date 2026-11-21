"""The filters the package provides. Each one is an engine object with a name and its own verification rows."""

from __future__ import annotations

import numpy as np

from michell import _engine
from michell.analysis import Grid
from michell.analysis3d import Grid3d
from michell.row import Row, compared


class DensityFilter(_engine.DensityFilter):
    """The density filter with cone weights (Bruns and Tortorelli 2001, Bourdin 2001):

        rho_e = sum_j w_ej x_j / sum_j w_ej,    w_ej = max(0, radius - distance between centres),

    over the active elements, radius in mm, on a 2D ``Grid`` or a 3D ``Grid3d``.
    ``DensityFilter(grid, radius, threads=1, *, outside="closed", inactive="closed")``; ``forward(x)`` and
    ``backward(x, grad)`` run in the engine. On a 2D grid the weights are stored per element and every call
    runs on one thread (``threads`` must be 1). On a 3D grid one stencil of weights by offset serves every
    element, and every call runs on ``threads`` threads, with numbers that do not depend on the thread
    count; it copies its input and releases Python's lock, like a solve.

    What lies beyond the faces of the grid is part of the model, and ``outside`` states it, either as one
    word for every face or as a dict by face (``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, and on a 3D grid
    ``"z-"``, ``"z+"``; faces left out are closed):

    - ``"closed"`` (the default): only elements inside the grid are averaged, and each average is
      normalised over them, so the face acts like a mirror. That is right for a symmetry plane, and for a
      face that continues into a wall or a support. A member lying along a closed face is filtered as if
      it were twice as wide, mirrored across the face.
    - ``"empty"``: the space beyond the face is void. Its weights stay in the normalising sum with a
      density of 0, so material at the face is averaged with nothing, as next to void elements. That is
      right for a free face of the part: a thin member flush against it is filtered like the same member in
      the interior, and a limit on member size sees it.

    ``inactive`` makes the same choice for the inactive elements inside the grid (the parts of the grid
    that are not part of the design): ``"closed"`` (the default) leaves them out of every sum, ``"empty"``
    averages them in as void. A uniform field stays uniform only next to closed faces and closed inactive
    elements; next to empty ones it drops, to a little over half at an empty face (0.56 for a radius of 8
    element sides).

    An empty face pads the filter's domain with void out to the radius, and an offset counts as void only
    when every face it crosses is empty (the padding stops at the plane of a closed face). Only the filter
    sees the padding: the analysis, the mass and the limits are taken over the grid alone, so the filtered
    density the padding would hold beyond the face is dropped (Clausen and Andreassen, "On filter boundary
    conditions in topology optimization", Struct. Multidisc. Optim. 56, 2017, section 3.2.2, call this a
    boundary padding on the filter and ask that it be stated; their domain extension also extends the
    analysis, which this filter does not). Two consequences follow. An element at a corner where two empty
    faces meet holds at most about a third of the full density (0.31 for a radius of 8 element sides), so
    it cannot become solid under a projection at 0.5; a support or load placed there needs solid elements
    around it (``KeepSolid``). And a
    member thinner than the radius cannot lie flush against an empty face and still project fully solid.
    """

    name = "density filter"

    def self_test(self) -> list[Row]:
        """The weights on a row of three elements worked by hand, a uniform field kept uniform on a 2D and a
        3D grid, and the backward map equal to the transpose of the forward one (verification row 25 has the
        3D checks with their limits)."""
        rows = []
        # Side 2 mm, radius 3 mm: own weight 3, direct neighbour 1, two away 0.
        row_of_three = DensityFilter(Grid(3, 1, 2.0, 1.0), 3.0)
        rho = row_of_three.forward(np.array([1.0, 0.0, 0.0]))
        rows.append(compared("weights by hand, rho_0 = 3 / (3 + 1)", float(rho[0]), 0.75, 1e-15))
        rows.append(compared("weights by hand, rho_1 = 1 / (1 + 3 + 1)", float(rho[1]), 0.2, 1e-15))

        grid = Grid(9, 5, 1.0, 1.0)
        density_filter = DensityFilter(grid, 2.4)
        uniform = density_filter.forward(np.full(grid.n_elements, 0.37))
        rows.append(compared("a uniform field stays uniform", float(np.abs(uniform - 0.37).max()), 0.0, 1e-14))

        x = np.linspace(0.0, 1.0, grid.n_elements)
        g = np.cos(np.arange(grid.n_elements))
        left = float(density_filter.forward(x) @ g)
        right = float(x @ density_filter.backward(x, g))
        rows.append(compared("backward is the transpose: <F x, g> = <x, F^T g>", right, left, 1e-12))

        grid_3d = Grid3d(7, 6, 5, 1.5)
        filter_3d = DensityFilter(grid_3d, 3.2)
        uniform = filter_3d.forward(np.full(grid_3d.n_elements, 0.37))
        rows.append(compared("3D grid: a uniform field stays uniform", float(np.abs(uniform - 0.37).max()), 0.0, 1e-14))
        x = np.linspace(0.0, 1.0, grid_3d.n_elements)
        g = np.cos(np.arange(grid_3d.n_elements))
        left = float(filter_3d.forward(x) @ g)
        right = float(x @ filter_3d.backward(x, g))
        rows.append(compared("3D grid: backward is the transpose", right, left, 1e-12))
        return rows


class HelmholtzFilter(_engine.HelmholtzFilter):
    """The PDE filter on a 3D grid (Lazarov and Sigmund, "Filters in topology optimization based on
    Helmholtz-type differential equations", IJNME 86, 2011):

        -r^2 lap(psi) + psi = x   on the active elements, zero flux across their boundary,   r = R / (2 sqrt 3),

    with the radius R in mm, solved on 8-node elements of the same grid; the element values go to the nodes
    and the node values are averaged back to the elements. ``HelmholtzFilter(grid, radius, threads=1)``;
    ``forward(x)`` and ``backward(x, grad)`` run in the engine, each one conjugate-gradient solve on
    ``threads`` threads (to the relative residual ``tolerance``, 1e-10 unless given). The map is linear and
    symmetric, so ``backward`` is the same solve. The zero flux makes each face of the grid, and the boundary
    of the inactive elements, behave like a closed face of the density filter (a mirror); this filter has no
    ``outside`` choice. The mass matrix is lumped, which makes the map free of negative entries: a field of
    values in [0, 1] stays in [0, 1] for any radius.
    """

    name = "Helmholtz filter"

    def self_test(self) -> list[Row]:
        """A uniform field kept uniform, the sum of the field kept, the backward map equal to the transpose
        of the forward one, and the response of a long bar to one element against the solution of the 1D
        equation, h / (2 r) exp(-|x| / r) (verification row 20 has the same checks with their limits)."""
        rows = []
        grid = Grid3d(9, 6, 5, 1.0)
        pde_filter = HelmholtzFilter(grid, 2.5, tolerance=1e-12)
        uniform = pde_filter.forward(np.full(grid.n_elements, 0.37))
        rows.append(compared("a uniform field stays uniform", float(np.abs(uniform - 0.37).max()), 0.0, 1e-10))

        random = np.random.default_rng(20)
        x = random.uniform(0.0, 1.0, grid.n_elements)
        y = random.uniform(0.0, 1.0, grid.n_elements)
        rows.append(compared("the sum of the field is kept", float(pde_filter.forward(x).sum()), float(x.sum()), 1e-10))
        left = float(pde_filter.forward(x) @ y)
        right = float(x @ pde_filter.backward(x, y))
        rows.append(compared("backward is the transpose: <F x, y> = <x, F^T y>", right, left, 1e-10))

        # A bar of 61 x 1 x 1 elements of side 1 mm, radius such that r = 4 mm: the element at the middle set
        # to 1. Across the bar the solution is uniform, so the 3D equation is the 1D one.
        h, r, n = 1.0, 4.0, 61
        bar = HelmholtzFilter(Grid3d(n, 1, 1, h), 2 * np.sqrt(3) * r, tolerance=1e-12)
        x = np.zeros(n)
        x[n // 2] = 1.0
        rho = bar.forward(x)
        distance = r  # one length scale away: four elements
        expected = h / (2 * r) * np.exp(-distance / r)
        rows.append(
            compared("a long bar against the 1D solution h / (2 r) exp(-|x| / r)", float(rho[n // 2 + 4]), expected, 0.02)
        )
        return rows


class Projection(_engine.Projection):
    """The projection, a smoothed step towards 0 or 1 around the threshold ``eta`` (Wang, Lazarov and
    Sigmund, "On projection methods, convergence and robust formulations in topology optimization",
    Struct. Multidisc. Optim. 43, 2011):

        P(x) = (tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))),

    applied to each value on its own, so it works on a 2D or a 3D field and needs no grid.
    ``Projection(beta, eta=0.5)`` with the sharpness ``beta > 0`` and ``0 <= eta <= 1``; ``forward(x)`` and
    ``backward(x, grad)`` (each entry of ``grad`` times P' at that entry of ``x``) run in the engine.
    P(0) = 0 and P(1) = 1 for every beta and eta.
    """

    name = "projection"

    def self_test(self) -> list[Row]:
        """The two ends kept, the threshold mapped to 1/2 when eta = 1/2 (the curve is then symmetric about
        that point), and the derivative against central differences (verification row 21 has the full
        check with its limits)."""
        rows = []
        ends = self.forward(np.array([0.0, 1.0]))
        rows.append(compared("P(0) = 0", float(ends[0]), 0.0, 1e-15))
        rows.append(compared("P(1) = 1", float(ends[1]), 1.0, 1e-15))
        symmetric = Projection(8.0, 0.5)
        rows.append(compared("P(1/2) = 1/2 when eta = 1/2", float(symmetric.forward(np.array([0.5]))[0]), 0.5, 1e-15))
        x = np.linspace(0.05, 0.95, 19)
        step = 1e-6
        difference = (self.forward(x + step) - self.forward(x - step)) / (2 * step)
        derivative = self.backward(x, np.ones_like(x))
        largest = float(np.abs(difference - derivative).max() / np.abs(derivative).max())
        rows.append(compared("the derivative against central differences", largest, 0.0, 1e-7))
        return rows


class KeepSolid:
    """Elements kept solid: ``forward`` sets the density of the marked elements to 1 and leaves the others as
    they are; ``backward`` passes the gradient through for the others and gives 0 for the marked ones, whose
    density no longer depends on x.

    ``KeepSolid(mask, grid=None)``: ``mask`` is a boolean or 0/1 array with one entry per element. With
    ``grid`` (a ``Grid`` or a ``Grid3d``) its length is checked and a mask that marks an inactive element is
    refused; without it the length is checked at each call. It belongs at the end of a design's chain, after
    the projection, so that the marked elements are exactly solid in the design that is solved:
    ``designs={"nominal": [projection, KeepSolid(pad)]}``. Supports and load points need solid material
    around them when the density filter's faces are empty (Clausen and Andreassen, "On filter boundary
    conditions in topology optimization", Struct. Multidisc. Optim. 56, 2017, section 3.3).
    """

    name = "keep solid"

    def __init__(self, mask, grid: Grid | Grid3d | None = None):
        values = np.asarray(mask)
        if values.ndim != 1:
            raise ValueError("the mask of KeepSolid needs one entry per element, as a flat array")
        if values.dtype != bool:
            if not np.all((values == 0) | (values == 1)):
                raise ValueError("the mask of KeepSolid holds only True/False or 0/1")
            values = values == 1
        if grid is not None:
            if values.size != grid.n_elements:
                raise ValueError(f"the mask of KeepSolid needs {grid.n_elements} entries (got {values.size})")
            inactive = values & (np.asarray(grid.active) == 0)
            if inactive.any():
                raise ValueError(
                    f"KeepSolid marks {int(inactive.sum())} inactive elements (first {int(np.flatnonzero(inactive)[0])}); "
                    f"only elements of the part can be kept solid"
                )
        self.mask = values.copy()
        self.mask.flags.writeable = False

    def _check(self, values: np.ndarray, label: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != self.mask.shape:
            raise ValueError(f"{label} needs {self.mask.size} entries, one per element (got {values.size})")
        return values

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self._check(x, "x").copy()
        out[self.mask] = 1.0
        return out

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        self._check(x, "x")
        grad = self._check(grad_out, "grad").copy()
        grad[self.mask] = 0.0
        return grad

    def self_test(self) -> list[Row]:
        """The marked elements set to 1 and the others kept, on a small mask of its own, and the backward map
        against central differences of <forward(x), g>."""
        test = KeepSolid(np.array([False, True, False, True, False]))
        x = np.array([0.2, 0.3, 0.9, 0.0, 0.6])
        kept = test.forward(x)
        expected = np.array([0.2, 1.0, 0.9, 1.0, 0.6])
        rows = [compared("marked elements set to 1, the others kept", float(np.abs(kept - expected).max()), 0.0, 0.0)]
        g = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
        step = 1e-6
        differences = np.array([
            (test.forward(x + step * e) @ g - test.forward(x - step * e) @ g) / (2 * step) for e in np.eye(5)
        ])  # fmt: skip
        largest = float(np.abs(differences - test.backward(x, g)).max())
        rows.append(compared("backward against central differences of <forward(x), g>", largest, 0.0, 1e-9))
        return rows
