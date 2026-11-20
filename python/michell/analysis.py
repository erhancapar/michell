"""The engine's analysis objects, as the package offers them: masks may be boolean arrays.

The engine takes masks (active elements, held degrees of freedom) only as float64 arrays of 0.0 and 1.0
and converts nothing (``docs/design/engine-boundary.md``). These subclasses turn a boolean array into that
form before the engine sees it; any other argument is passed on unchanged.
"""

from __future__ import annotations

import numpy as np

from michell import _engine


def as_mask(values):
    """A boolean NumPy array becomes float64 0.0 / 1.0; anything else is returned as it is."""
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        return values.astype(np.float64)
    return values


class Grid(_engine.Grid):
    """A grid of nx by ny square elements of side h mm and thickness t mm, y up.

    ``active`` (one entry per element) may be a boolean array or float64 0.0 / 1.0; all active if omitted.
    """

    def __new__(cls, nx, ny, h, t, active=None):
        return super().__new__(cls, nx, ny, h, t, as_mask(active))


class Elasticity(_engine.Elasticity):
    """Plane-stress elasticity on a grid. ``held`` (one entry per degree of freedom) may be a boolean
    array or float64 0.0 / 1.0."""

    def __new__(cls, grid, held, *, youngs_modulus, poisson_ratio, e_min, penalty):
        return super().__new__(
            cls,
            grid,
            as_mask(held),
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            e_min=e_min,
            penalty=penalty,
        )
