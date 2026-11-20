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

    over the active elements, radius in mm. ``DensityFilter(grid, radius)``; ``forward(x)`` and
    ``backward(x, grad)`` run in the engine.
    """

    name = "density filter"

    def self_test(self) -> list[Row]:
        """The weights on a row of three elements worked by hand, a uniform field kept uniform, and the
        backward map equal to the transpose of the forward one."""
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
        return rows


class HelmholtzFilter(_engine.HelmholtzFilter):
    """The PDE filter on a 3D grid (Lazarov and Sigmund, "Filters in topology optimization based on
    Helmholtz-type differential equations", IJNME 86, 2011):

        -r^2 lap(psi) + psi = x   on the active elements, zero flux across their boundary,   r = R / (2 sqrt 3),

    with the radius R in mm, solved on 8-node elements of the same grid; the element values go to the nodes
    and the node values are averaged back to the elements. ``HelmholtzFilter(grid, radius, threads=1)``;
    ``forward(x)`` and ``backward(x, grad)`` run in the engine, each one conjugate-gradient solve on
    ``threads`` threads (to the relative residual ``tolerance``, 1e-10 unless given). The map is linear and
    symmetric, so ``backward`` is the same solve. The mass matrix is lumped, which makes the map free of
    negative entries: a field of values in [0, 1] stays in [0, 1] for any radius.
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
