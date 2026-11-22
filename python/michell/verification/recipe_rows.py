"""Verification row 28: the two-step procedure of ``michell.recipes.stiff_part`` on a 3D bridge.

Sources: Wang, Lazarov and Sigmund, "On projection methods, convergence and robust formulations in topology
optimization", Struct. Multidisc. Optim. 43 (2011), for the robust step; Zhou, Lazarov, Wang and Sigmund,
"Minimum length scale in topology optimization by geometric constraints", Comput. Methods Appl. Mech.
Engrg. 293 (2015) 266-282, for the member-size step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from michell import select
from michell._engine import dot
from michell.analysis3d import Grid3d
from michell.filters import KeepSolid
from michell.problem import LoadCase, Material, Problem
from michell.recipes import StiffPart, stiff_part
from michell.row import DESIGN_PROPERTY, Row, checked, g3, not_run

# ---------------------------------------------------------------------------------------------------------
# Row 28. Limits, each on its own line, fixed before any run:
# (a) the relative excess of the mass of the returned nominal design over the asked mass (the first over the
#     second, minus 1) is at most 0.001;
# (b) the thresholded design (nominal density at least 0.5) is one body, joined through element faces:
#     the number of bodies is at most 1 (an empty design counts as infinitely many);
# (c) the grey share, the mean of 4 rho (1 - rho) over the design elements (active and not held), is at
#     most 0.05;
# (d) a morphological opening with a ball of diameter 0.9 times the member size removes at most 5 % of the
#     solid of the thresholded design (held elements included);
# (e) the compliance of the returned nominal design is at most 1.10 times that of the nominal design at the
#     end of the robust step: the member-size step costs at most 10 % of stiffness.
#
# Measure of (d) (SciPy's `ndimage`, "not run" without SciPy): row 27's opening in 3D. The ball is the set
# of offsets (di, dj, dk) with di^2 + dj^2 + dk^2 <= (0.9 size / 2)^2 element sides squared (2.25^2 here,
# a ball 5 elements across); the share is the number of solid elements that `binary_opening` with that ball
# removes, over the number of solid elements. Beyond each face of the grid the field is continued by the
# rule fixed for that face before any run: beyond the two clamped faces (x- and x+) lies the support, so it
# is solid; beyond the four free faces the bridge is empty.
#
# Set-up: a block of 50 x 24 x 10 cube elements of side 1 mm (x along the span, z up), steel (210000 MPa,
# 0.3, 7.85 g/cm^3); every degree of freedom held on the faces x = 0 and x = 50; 1000 N down (-z) on a pad
# of 6 x 6 elements in the middle of the top face, as the consistent nodal forces of a uniform pressure;
# held solid (KeepSolid): an end plate 2 elements thick at each clamped face and the pad, 6 x 6 x 2
# elements under the load; the member size 5 mm (the filter radius 5 element sides, eps 1e-7 from figure 8
# at r = 5); the asked mass 30 % of the block's; the filter's x faces closed (they continue into the
# support), its y and z faces empty (free faces); every other setting the default of `stiff_part`.

ROW_28_DIMS = (50, 24, 10)
ROW_28_SIZE = 5.0
ROW_28_FRACTION = 0.30
ROW_28_LOAD_N = 1000.0
ROW_28_PLATE = 2
ROW_28_PAD = (6, 6, 2)
ROW_28_OUTSIDE = {"x-": "closed", "x+": "closed", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}
ROW_28_BEYOND = {"x-": "solid", "x+": "solid", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}


@dataclass(frozen=True)
class Bridge:
    """The part of row 28: the problem, the held elements and the asked mass in g."""

    problem: Problem
    keep_solid: KeepSolid
    mass_g: float


def bridge(nx: int = 50, ny: int = 24, nz: int = 10, threads: int = 1) -> Bridge:
    """The bridge of row 28 (see the set-up above the row) on nx x ny x nz elements of side 1 mm."""
    grid = Grid3d(nx, ny, nz, 1.0)
    held = select.held(grid, np.concatenate([select.face_nodes(grid, "x-"), select.face_nodes(grid, "x+")]))
    pad_x, pad_y, pad_z = ROW_28_PAD
    i0, j0 = (nx - pad_x) // 2, (ny - pad_y) // 2
    pad = ((i0, i0 + pad_x), (j0, j0 + pad_y), (nz - pad_z, nz))  # in mm: the elements have a side of 1 mm
    # The consistent nodal forces of a uniform pressure: each 1 mm square of the pad gives a quarter of its
    # force to each of its corners.
    loads = select.load(grid, select.face_nodes(grid, "z+", within=pad), (0.0, 0.0, -ROW_28_LOAD_N))
    everywhere = (-math.inf, math.inf)
    kept = (
        select.elements_in(grid, ((0, ROW_28_PLATE), everywhere, everywhere))
        | select.elements_in(grid, ((nx - ROW_28_PLATE, nx), everywhere, everywhere))
        | select.elements_in(grid, pad)
    )
    steel = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    problem = Problem(grid, held, steel, [LoadCase("pad", loads)], threads=threads)
    mass_g = ROW_28_FRACTION * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
    return Bridge(problem, KeepSolid(kept, grid), mass_g)


def face_connected_bodies(solid: np.ndarray, dims: tuple[int, int, int]) -> int:
    """The number of bodies of ``solid`` (one boolean per element, x fastest) joined through element faces.
    Needs SciPy."""
    from scipy import ndimage

    nx, ny, nz = dims
    _, count = ndimage.label(np.asarray(solid, dtype=bool).reshape(nz, ny, nx))  # default: faces only
    return int(count)


def opened_share_3d(solid: np.ndarray, dims: tuple[int, int, int], size: float, beyond: dict[str, str]) -> float:
    """The share of ``solid`` (one boolean per element, x fastest, elements of side 1) that a morphological
    opening with a ball of diameter 0.9 ``size`` removes. ``beyond`` says how the field continues past each
    face (``"x-"`` to ``"z+"``): ``"empty"``, ``"solid"`` or ``"mirror"`` (mirrored about the face, as row
    27 does at its symmetry line). Needs SciPy."""
    from scipy import ndimage

    nx, ny, nz = dims
    field = np.asarray(solid, dtype=bool).reshape(nz, ny, nx)
    radius = 0.9 * size / 2
    reach = int(np.floor(radius))
    offsets = np.arange(-reach, reach + 1)
    dk, dj, di = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    ball = di**2 + dj**2 + dk**2 <= radius**2
    margin = reach + 1
    padded = field
    for axis, name in ((2, "x"), (1, "y"), (0, "z")):  # array axis of each grid direction
        for side, width in (("-", (margin, 0)), ("+", (0, margin))):
            rule = beyond[name + side]
            pad = [(0, 0)] * 3
            pad[axis] = width
            if rule == "mirror":
                padded = np.pad(padded, pad, mode="symmetric")
            elif rule in ("empty", "solid"):
                padded = np.pad(padded, pad, constant_values=rule == "solid")
            else:
                raise ValueError(f"face {name + side}: unknown rule {rule!r}")
    inner = (slice(margin, -margin),) * 3
    opened = ndimage.binary_opening(padded, structure=ball)[inner]
    return float((field & ~opened).sum() / field.sum())


def compliance_of(problem: Problem, density: np.ndarray) -> float:
    loads = problem.cases[0].loads
    u, _ = problem.solve(density, loads)
    return dot(loads, u)


def bridge_run(threads: int = 1) -> tuple[Bridge, StiffPart]:
    """The bridge of row 28 and the outcome of ``stiff_part`` on it with the row's settings."""
    part = bridge(*ROW_28_DIMS, threads=threads)
    outcome = stiff_part(part.problem, part.mass_g, ROW_28_SIZE, part.keep_solid, outside=ROW_28_OUTSIDE)
    return part, outcome


def row_28_recipe_bridge() -> list[Row]:
    nx, ny, nz = ROW_28_DIMS
    head = f"Two-step recipe on a 3D bridge {nx} x {ny} x {nz}"
    names = (
        f"{head}: nominal mass over the asked mass, minus 1",
        f"{head}: bodies of the design thresholded at 0.5, joined through faces",
        f"{head}: grey share, mean of 4 rho (1 - rho) over the design elements",
        f"{head}: share of the solid removed by an opening of 0.9 times the member size",
        f"{head}: compliance over that at the end of the robust step",
    )
    limits = (0.001, 1.0, 0.05, 0.05, 1.10)
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError:
        return [
            not_run(28, name, DESIGN_PROPERTY, limit, "SciPy is not installed") for name, limit in zip(names, limits)
        ]
    part, outcome = bridge_run()
    problem = part.problem
    grid = problem.grid
    nominal = outcome.nominal
    element_mass = problem.material.mass_g(grid.voxel_volume_mm3)
    mass = element_mass * float(nominal.sum())
    solid = nominal >= 0.5
    bodies = face_connected_bodies(solid, ROW_28_DIMS)
    design = ~part.keep_solid.mask
    grey = float((4.0 * nominal * (1.0 - nominal))[design].mean())
    share = opened_share_3d(solid, ROW_28_DIMS, ROW_28_SIZE, ROW_28_BEYOND)
    compliance = compliance_of(problem, nominal)
    compliance_first = compliance_of(problem, outcome.first_step_nominal)
    result = outcome.result
    g = result.limits[1]
    passes = len(outcome.passes)
    set_up = (
        f"steel block of {nx} x {ny} x {nz} elements of 1 mm, clamped at x = 0 and x = {nx}, 1000 N down on a "
        f"pad of 6 x 6 elements in the middle of the top; held solid: end plates {ROW_28_PLATE} elements thick "
        f"and the pad, 6 x 6 x 2; member size {g3(ROW_28_SIZE)} mm, asked mass 30 % of the block "
        f"({g3(part.mass_g)} g); filter x faces closed, y and z faces empty; default settings ({passes} passes); "
        f"member-size value {g3(g)} against eps {g3(outcome.eps)} (g / eps {g3(g / outcome.eps)}), limits met "
        f"{result.limits_met}, stop reason {result.stop_reason}; limits fixed before any run"
    )
    return [
        checked(28, names[0], DESIGN_PROPERTY, mass / part.mass_g - 1.0, limits[0], f"mass {g3(mass)} g; {set_up}"),
        checked(
            28,
            names[1],
            DESIGN_PROPERTY,
            float(bodies) if bodies > 0 else float("inf"),
            limits[1],
            f"{int(solid.sum())} solid elements of {grid.n_elements}",
        ),
        checked(28, names[2], DESIGN_PROPERTY, grey, limits[2], f"over {int(design.sum())} design elements"),
        checked(
            28,
            names[3],
            DESIGN_PROPERTY,
            share,
            limits[3],
            f"a ball of diameter {g3(0.9 * ROW_28_SIZE)} elements; beyond the clamped faces solid (the support), "
            f"beyond the free faces empty, the rule fixed before any run",
        ),
        checked(
            28,
            names[4],
            DESIGN_PROPERTY,
            compliance / compliance_first,
            limits[4],
            f"compliance {g3(compliance)} N mm, at the end of the robust step {g3(compliance_first)} N mm",
        ),
    ]


__all__ = [
    "Bridge",
    "bridge",
    "bridge_run",
    "face_connected_bodies",
    "opened_share_3d",
    "row_28_recipe_bridge",
]
