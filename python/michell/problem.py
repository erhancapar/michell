"""A 2D plane-stress problem: the grid, its supports, the material, the load cases and the SIMP law."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from michell.analysis import Elasticity


@dataclass(frozen=True)
class Material:
    """The constants of the solid material. Units: MPa and g/mm^3."""

    youngs_modulus: float
    poisson_ratio: float
    density_g_mm3: float


@dataclass(frozen=True)
class LoadCase:
    """One load case: a force per degree of freedom (N) and the weight of the case in a weighted sum."""

    name: str
    loads: np.ndarray
    weight: float = 1.0

    def __post_init__(self):
        loads = np.array(self.loads, dtype=np.float64)  # a copy, so the case cannot change afterwards
        loads.flags.writeable = False
        object.__setattr__(self, "loads", loads)


class Problem:
    """A 2D plane-stress problem.

    ``held`` has one entry per degree of freedom (a boolean array or float64 0.0 / 1.0); a held degree of
    freedom stays at zero. The modulus of an element of density rho is
    E(rho) = e_min + rho^penalty (E_0 - e_min) (modified SIMP); ``e_min`` defaults to 1e-9 E_0.
    """

    def __init__(
        self,
        grid,
        held,
        material: Material,
        cases: Sequence[LoadCase],
        *,
        penalty: float = 3.0,
        e_min: float | None = None,
    ):
        names = [case.name for case in cases]
        if len(set(names)) != len(names):
            raise ValueError(f"load case names must differ; got {names}")
        for case in cases:
            if case.loads.shape != (grid.n_dofs,):
                raise ValueError(
                    f"load case {case.name!r} needs one load per degree of freedom ({grid.n_dofs}); "
                    f"got shape {case.loads.shape}"
                )
        self.grid = grid
        self.material = material
        self.cases = tuple(cases)
        self.penalty = penalty
        self.e_min = 1e-9 * material.youngs_modulus if e_min is None else e_min
        self.elasticity = Elasticity(
            grid,
            held,
            youngs_modulus=material.youngs_modulus,
            poisson_ratio=material.poisson_ratio,
            e_min=self.e_min,
            penalty=penalty,
        )

    def case(self, name: str) -> LoadCase:
        """The load case called ``name``."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"no load case called {name!r}; the cases are {[c.name for c in self.cases]}")
