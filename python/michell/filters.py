"""The filters the package provides. Each one is an engine object with a name and its own verification rows."""

from __future__ import annotations

import numpy as np

from michell import _engine
from michell.analysis import Grid
from michell.modules import Row


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
        rows.append(Row("weights by hand, rho_0 = 3 / (3 + 1)", float(rho[0]), 0.75, 1e-15))
        rows.append(Row("weights by hand, rho_1 = 1 / (1 + 3 + 1)", float(rho[1]), 0.2, 1e-15))

        grid = Grid(9, 5, 1.0, 1.0)
        density_filter = DensityFilter(grid, 2.4)
        uniform = density_filter.forward(np.full(grid.n_elements, 0.37))
        rows.append(Row("a uniform field stays uniform", float(np.abs(uniform - 0.37).max()), 0.0, 1e-14))

        x = np.linspace(0.0, 1.0, grid.n_elements)
        g = np.cos(np.arange(grid.n_elements))
        left = float(density_filter.forward(x) @ g)
        right = float(x @ density_filter.backward(x, g))
        rows.append(Row("backward is the transpose: <F x, g> = <x, F^T g>", right, left, 1e-12))
        return rows
