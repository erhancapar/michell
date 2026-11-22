"""A problem: the grid, its supports, the material, the load cases and the SIMP law, with the analysis that
belongs to the grid (2D plane stress on a ``Grid``, 3D elasticity on a ``Grid3d``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from michell import _engine
from michell.analysis import Elasticity
from michell.analysis3d import Elasticity3d
from michell.errors import InvalidValue, UnknownName


@dataclass(frozen=True)
class Material:
    """The constants of the solid material: the modulus in MPa, the density in g/cm^3 (steel 7.85).

    The lengths of michell are in mm; ``mass_g`` is the one place where the density meets them."""

    youngs_modulus: float
    poisson_ratio: float
    density_g_cm3: float

    def mass_g(self, volume_mm3):
        """The mass in g of ``volume_mm3`` (a number or an array) of solid material."""
        return self.density_g_cm3 / 1000.0 * volume_mm3


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
    """A problem on a 2D or a 3D grid; the analysis follows from the grid's type.

    On a ``Grid`` the analysis is plane stress, solved directly (``michell.Elasticity``); on a ``Grid3d`` it is
    3D elasticity, solved by conjugate gradients with a multigrid preconditioner (``michell.Elasticity3d``).
    Either is kept as ``problem.elasticity``.

    ``held`` has one entry per degree of freedom (a boolean array or float64 0.0 / 1.0); a held degree of
    freedom stays at zero, and a load case that loads one is refused. The modulus of an element of density
    rho is E(rho) = e_min + rho^penalty (E_0 - e_min) (modified SIMP); ``e_min`` defaults to 1e-9 E_0.

    ``threads`` (the number of threads every solve and every MMA update runs on), ``solver_tolerance`` (the
    relative residual at which a solve stops) and ``solver_max_steps`` (the steps allowed before
    ``solver_not_converged``) are settings of the 3D analysis; on a 2D grid they must keep their defaults,
    since the direct solver has none of them. A solve of a design whose solid is joined to the supports
    takes tens of steps, about a
    hundred on an intricate 0-and-1 design; solid joined to the rest only through void (detached islands,
    whose stiffness is ``e_min``) raises the count by a large factor (verification row 17 gives the
    counts), and the default of 5000 leaves room for that. A load on void alone does not (same row).
    ``solver_not_converged`` names the steps made, the residual reached and the usual cause: material not
    connected to the supports.
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
        threads: int = 1,
        solver_tolerance: float = 1e-6,
        solver_max_steps: int = 5000,
    ):
        names = [case.name for case in cases]
        if len(set(names)) != len(names):
            raise InvalidValue(f"load case names must differ; got {names}")
        for case in cases:
            if case.loads.shape != (grid.n_dofs,):
                raise InvalidValue(
                    f"load case {case.name!r} needs one load per degree of freedom ({grid.n_dofs}); "
                    f"got shape {case.loads.shape}"
                )
        self.grid = grid
        self.material = material
        self.cases = tuple(cases)
        self.penalty = penalty
        self.e_min = 1e-9 * material.youngs_modulus if e_min is None else e_min
        self.is_3d = isinstance(grid, _engine.Grid3d)
        constants = dict(
            youngs_modulus=material.youngs_modulus,
            poisson_ratio=material.poisson_ratio,
            e_min=self.e_min,
            penalty=penalty,
        )
        if self.is_3d:
            self.elasticity = Elasticity3d(
                grid, held, **constants, tolerance=solver_tolerance, max_steps=solver_max_steps, threads=threads
            )
        else:
            given = {
                name: value
                for name, value, default in (
                    ("threads", threads, 1),
                    ("solver_tolerance", solver_tolerance, 1e-6),
                    ("solver_max_steps", solver_max_steps, 5000),
                )
                if value != default
            }
            if given:
                raise InvalidValue(
                    f"{', '.join(given)} only apply to a 3D grid (Grid3d): the 2D analysis solves directly "
                    f"on one thread; got {given}"
                )
            self.elasticity = Elasticity(grid, held, **constants)
        self.threads = threads
        held_mask = np.asarray(held, dtype=np.float64) != 0.0
        for case in self.cases:
            loaded = np.flatnonzero(held_mask & (case.loads != 0.0))
            if loaded.size:
                first = ", ".join(str(int(dof)) for dof in loaded[:5])
                raise InvalidValue(
                    f"load case {case.name!r} loads {loaded.size} held degrees of freedom (first: {first}); a "
                    f"held degree of freedom stays at zero, so its load would be lost",
                    {"held_loaded_dofs": int(loaded.size), "first_dof": int(loaded[0])},
                )

    def solve(self, density: np.ndarray, loads: np.ndarray, start: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """The displacements u with K(density) u = loads, and the number of conjugate-gradient steps the
        solve took. ``start`` is a first guess of u (a warm start, such as the state of the previous pass):
        the 3D solver starts from it, the direct 2D solver has no use for it and reports 0 steps."""
        if not self.is_3d:
            return self.elasticity.solve(density, loads), 0
        u = self.elasticity.solve(density, loads, start=start)
        return u, int(self.elasticity.last_solve_report()["steps"])

    def adjoint(self, density: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, int]:
        """lambda with K(density) lambda = rhs (every held degree of freedom at zero), and the number of
        conjugate-gradient steps the solve took (0 for the direct 2D solver)."""
        if not self.is_3d:
            return self.elasticity.adjoint(density, rhs), 0
        lam = self.elasticity.adjoint(density, rhs)
        return lam, int(self.elasticity.last_solve_report()["steps"])

    def case(self, name: str) -> LoadCase:
        """The load case called ``name``."""
        for case in self.cases:
            if case.name == name:
                return case
        raise UnknownName(f"no load case called {name!r}; the cases are {[c.name for c in self.cases]}")
