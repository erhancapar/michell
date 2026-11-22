"""Verification row 29: elements held solid (``michell.KeepSolid``), on a 2D and on a 3D grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from michell import select
from michell.analysis import Grid
from michell.analysis3d import Grid3d
from michell.designs import design_chains, design_gradient, evaluate_designs
from michell.filters import DensityFilter, KeepSolid
from michell.mma import MovingAsymptotes
from michell.optimise import Limit, Result, optimise
from michell.problem import LoadCase, Material, Problem
from michell.responses import Compliance, Mass
from michell.row import CLOSED_FORM, DESIGN_PROPERTY, GRADIENT, Row, checked, g3

# ---------------------------------------------------------------------------------------------------------
# Row 29. Limits, each on its own line, fixed before any run, on each of the two parts below:
# (a) after KeepSolid the held elements are exactly 1 and the others unchanged: the largest difference from
#     that, over every element, is 0;
# (b) the derivative of the compliance with respect to the design variables of the held elements is exactly
#     0 (the largest absolute value is 0), and for the other elements KeepSolid's backward is the transpose
#     of its forward: the inner-product test |<J v, g> - <v, backward(x, g)>| / |<v, backward(x, g)>| is at
#     most 1e-12;
# (c) in a short optimisation (20 passes of MMA, the filters KeepSolid, the density filter and KeepSolid
#     again, as the examples chain them) the held elements of the returned density are exactly 1 (the
#     largest difference is 0), and the mass limit is met by Result.limits_met: its relative excess
#     (value - at_most) / at_most is at most the run's limit_tolerance, 1e-3, the measure limits_met judges;
# (d) the mass of the returned density is at least the mass of the held elements: the held mass over the
#     returned mass is at most 1.
#
# OC takes the density filter as its only filter and refuses any other in the chain (michell.oc), so the 2D
# run of (c) uses MMA as the 3D run does.
#
# In (b) the derivative is the one the loop uses (michell.designs.design_gradient) through the chain of (c),
# at design variables drawn at random from [0.1, 0.9] (seed 29); J v is forward(x + v) - forward(x), exact
# for a map that is affine, with v and g drawn at random from [-1, 1] on every element.
#
# Set-up, 2D: a plate of 30 x 10 square elements of side 1 mm, 1 mm thick, steel (210000 MPa, 0.3,
# 7.85 g/cm^3), clamped on its left edge; 1000 N down spread over the nodes of the right edge between y = 4
# and y = 6 mm; held solid: a strip 2 elements wide along the clamped edge and a pad of 4 x 4 elements at
# the right edge around the load. 3D: a block of 16 x 6 x 4 cube elements of side 1 mm, steel, clamped on its
# face x = 0; 1000 N down spread over the nodes of the face x = 16 between y = 2 and 4 mm and z = 1 and
# 3 mm; held solid: a plate 2 elements thick at the clamped face and a pad of 2 x 2 x 2 elements at the far
# face around the load. In both the density filter's radius is 1.5 mm, its faces and inactive elements
# closed (the default); the mass limit is 40 % of the part's mass; the run starts from the held elements at 1
# and the others at the share that gives that mass.

ROW_29_PASSES = 20
ROW_29_FRACTION = 0.40
ROW_29_RADIUS = 1.5
ROW_29_LOAD_N = 1000.0


@dataclass(frozen=True)
class HeldPart:
    """A part of row 29: the problem, the held elements and the filters of its optimisation."""

    label: str
    problem: Problem
    keep_solid: KeepSolid
    filters: tuple


def held_part(dimension: int) -> HeldPart:
    """The 2D (``dimension`` 2) or the 3D part of row 29 (see the set-up above the row)."""
    steel = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    everywhere = (-math.inf, math.inf)
    if dimension == 2:
        nx, ny = 30, 10
        grid = Grid(nx, ny, 1.0, 1.0)
        held = select.held(grid, select.face_nodes(grid, "x-"))
        loads = select.load(grid, select.face_nodes(grid, "x+", within=(everywhere, (4.0, 6.0))), (0.0, -ROW_29_LOAD_N))
        kept = select.elements_in(grid, ((0.0, 2.0), everywhere)) | select.elements_in(
            grid, ((nx - 4.0, nx), (3.0, 7.0))
        )
        label = f"2D plate {nx} x {ny}"
    else:
        nx, ny, nz = 16, 6, 4
        grid = Grid3d(nx, ny, nz, 1.0)
        held = select.held(grid, select.face_nodes(grid, "x-"))
        pad = (everywhere, (2.0, 4.0), (1.0, 3.0))
        loads = select.load(grid, select.face_nodes(grid, "x+", within=pad), (0.0, 0.0, -ROW_29_LOAD_N))
        kept = select.elements_in(grid, ((0.0, 2.0), everywhere, everywhere)) | select.elements_in(
            grid, ((nx - 2.0, nx), (2.0, 4.0), (1.0, 3.0))
        )
        label = f"3D block {nx} x {ny} x {nz}"
    problem = Problem(grid, held, steel, [LoadCase("load", loads)])
    keep_solid = KeepSolid(kept, grid)
    filters = (keep_solid, DensityFilter(grid, ROW_29_RADIUS), keep_solid)
    return HeldPart(label, problem, keep_solid, filters)


def held_part_run(part: HeldPart) -> tuple[Result, float, float]:
    """The short optimisation of (c) on ``part``: the ``Result``, the bound of the mass limit (g) and the mass
    of one solid element (g)."""
    problem = part.problem
    grid = problem.grid
    held = part.keep_solid.mask
    design = ~held
    element_mass = float(problem.material.mass_g(grid.voxel_volume_mm3))
    bound = ROW_29_FRACTION * element_mass * grid.n_elements
    share = (bound / element_mass - int(held.sum())) / int(design.sum())
    x = np.where(held, 1.0, share)
    result = optimise(
        problem, Compliance(), [Limit(Mass(), at_most=bound)], list(part.filters), MovingAsymptotes(), x,
        passes=ROW_29_PASSES,
    )  # fmt: skip
    return result, bound, element_mass


def row_29_keep_solid() -> list[Row]:
    rows = []
    for dimension in (2, 3):
        part = held_part(dimension)
        problem, keep = part.problem, part.keep_solid
        grid = problem.grid
        held = keep.mask
        random = np.random.default_rng(29)
        x = random.uniform(0.1, 0.9, grid.n_elements)
        head = f"KeepSolid on a {part.label}"
        what = f"{int(held.sum())} of {grid.n_elements} elements held"

        # (a) forward: the held elements 1, the others unchanged.
        out = keep.forward(x)
        difference = max(float(np.abs(out[held] - 1.0).max()), float(np.abs(out[~held] - x[~held]).max()))
        rows.append(
            checked(
                29, f"{head}: forward, largest difference from 1 on the held elements and from x elsewhere",
                CLOSED_FORM, difference, 0.0, f"{what}; x drawn at random from [0.1, 0.9]",
            )
        )  # fmt: skip

        # (b) the derivative of the compliance through the chain of (c), and the inner-product test.
        chains = design_chains(None)
        response = Compliance()
        base_fields, evaluated = evaluate_designs(problem, x, part.filters, chains, {None: set(response.reads)})
        gradient = design_gradient(problem, response, None, part.filters, chains, base_fields, evaluated)
        on_held = float(np.abs(gradient[held]).max())
        rows.append(
            checked(
                29, f"{head}: derivative of the compliance on the held elements, largest absolute value",
                GRADIENT, on_held, 0.0,
                f"{what}; the loop's derivative through KeepSolid, the density filter and KeepSolid; on the "
                f"other elements up to {g3(float(np.abs(gradient[~held]).max()))} N mm",
            )
        )  # fmt: skip
        v = random.uniform(-1.0, 1.0, grid.n_elements)
        g = random.uniform(-1.0, 1.0, grid.n_elements)
        jv = keep.forward(x + v) - keep.forward(x)
        transposed = float(np.dot(v, keep.backward(x, g)))
        inner = abs(float(np.dot(jv, g)) - transposed) / abs(transposed)
        rows.append(
            checked(
                29, f"{head}: backward against the transpose of forward, inner-product test",
                GRADIENT, inner, 1e-12,
                "|<J v, g> - <v, backward(x, g)>| / |<v, backward(x, g)>|, with J v = forward(x + v) - "
                "forward(x); v and g drawn at random from [-1, 1]",
            )
        )  # fmt: skip

        # (c) and (d): a short optimisation.
        result, bound, element_mass = held_part_run(part)
        density = result.density
        held_off = float(np.abs(density[held] - 1.0).max())
        mass = element_mass * float((np.asarray(grid.active) * density).sum())
        excess = (result.limits[0] - bound) / bound
        run = (
            f"{len(result.passes)} passes of MMA, filters KeepSolid, density filter (radius "
            f"{g3(ROW_29_RADIUS)} mm), KeepSolid; mass limit {g3(bound)} g, 40 % of the part"
        )
        rows.append(
            checked(
                29, f"{head}: short optimisation, largest difference from 1 on the held elements",
                DESIGN_PROPERTY, held_off, 0.0, f"{what}; {run}",
            )
        )  # fmt: skip
        rows.append(
            checked(
                29, f"{head}: short optimisation, relative excess of the mass over its bound",
                DESIGN_PROPERTY, excess, 1e-3,
                f"mass {g3(result.limits[0])} g; limits met {result.limits_met}, the run's limit_tolerance 1e-3; "
                f"stop reason {result.stop_reason}",
            )
        )  # fmt: skip
        held_mass = element_mass * int(held.sum())
        rows.append(
            checked(
                29, f"{head}: short optimisation, mass of the held elements over the mass returned",
                DESIGN_PROPERTY, held_mass / mass, 1.0,
                f"held {g3(held_mass)} g, returned density {g3(mass)} g",
            )
        )  # fmt: skip
    return rows


__all__ = ["HeldPart", "held_part", "held_part_run", "row_29_keep_solid"]
