"""The two basic responses: compliance (low compliance = high stiffness) and mass."""

from __future__ import annotations

import numpy as np

from michell._engine import dot
from michell.analysis import Grid
from michell.evaluation import evaluate
from michell.modules import Partials
from michell.problem import LoadCase, Material, Problem
from michell.row import Row, compared


class Compliance:
    """The weighted sum over the load cases of f . u, in N mm: twice the strain energy stored in the
    part under each load case (and twice the work of loads applied slowly from zero).

    Its derivative with respect to u is the load vector, so its adjoint is the state itself and no extra
    solve is needed.
    """

    name = "compliance"
    unit = "N mm"
    reads = ("elasticity",)

    def value(self, ev) -> float:
        total = 0.0
        for case in ev.cases:
            total += case.weight * dot(ev.loads("elasticity", case.name), ev.state("elasticity", case.name))
        return float(total)

    def partials(self, ev) -> Partials:
        d_state = {}
        for case in ev.cases:
            d_state[("elasticity", case.name)] = case.weight * ev.loads("elasticity", case.name)
        return Partials(d_density=None, d_state=d_state)

    def self_test(self) -> list[Row]:
        """A bar in uniform tension: compliance = P^2 L / (E A), exact for this element (the field is
        linear). Source: the elongation of a bar, PL / (EA), in any text on strength of materials."""
        nx, ny, h, t = 8, 2, 2.5, 1.5
        e, force = 210000.0, 1000.0
        grid = Grid(nx, ny, h, t)
        held = np.zeros(grid.n_dofs, dtype=bool)
        for j in range(ny + 1):
            held[2 * grid.node(0, j)] = True  # x held on the left edge
        held[2 * grid.node(0, 0) + 1] = True  # y held at one node
        loads = np.zeros(grid.n_dofs)
        for j in range(ny + 1):  # a uniform traction on the right edge, as consistent nodal forces
            share = 0.5 if j in (0, ny) else 1.0
            loads[2 * grid.node(nx, j)] = force / ny * share
        material = Material(youngs_modulus=e, poisson_ratio=0.3, density_g_mm3=7.85e-3)
        problem = Problem(grid, held, material, [LoadCase("tension", loads)])
        ev, _ = evaluate(problem, np.ones(grid.n_elements), [], self.reads)
        length, area = nx * h, ny * h * t
        expected = force**2 * length / (e * area)
        return [compared("compliance of a bar in tension, P^2 L / (E A)", self.value(ev), expected, 1e-10)]


class Mass:
    """The mass of the part in g: material density times element volume times the sum of rho over the
    active elements. Linear in rho, so its partials are constant."""

    name = "mass"
    unit = "g"
    reads = ()

    def value(self, ev) -> float:
        return dot(self._per_element(ev), ev.density)

    def partials(self, ev) -> Partials:
        return Partials(d_density=self._per_element(ev), d_state={})

    @staticmethod
    def _per_element(ev) -> np.ndarray:
        """The mass of each element when solid; 0 for an element outside the part."""
        return ev.material.density_g_mm3 * ev.grid.voxel_volume_mm3 * ev.grid.active

    def self_test(self) -> list[Row]:
        """A solid block and the same block at half density: mass = density x volume."""
        nx, ny, h, t, density = 4, 3, 2.0, 5.0, 7.85e-3
        grid = Grid(nx, ny, h, t)
        held = np.zeros(grid.n_dofs, dtype=bool)
        for j in range(ny + 1):
            held[2 * grid.node(0, j)] = held[2 * grid.node(0, j) + 1] = True
        material = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_mm3=density)
        problem = Problem(grid, held, material, [LoadCase("none", np.zeros(grid.n_dofs))])
        solid = density * (nx * h) * (ny * h) * t
        rows = []
        for fraction in (1.0, 0.5):
            ev, _ = evaluate(problem, np.full(grid.n_elements, fraction), [], self.reads)
            rows.append(compared(f"mass of a block at density {fraction}", self.value(ev), fraction * solid, 1e-12))
        return rows
