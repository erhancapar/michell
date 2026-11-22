"""Verification row 32: supports and loads on the voxel model of a part (``michell.attach``).

The bar of lines (a) and (b) has closed-form answers that the trilinear elements reproduce exactly: under a
uniform traction on its end, with rollers on three orthogonal faces, the stress is uniform and the
displacement linear; under a uniform body force with Poisson's ratio 0 the displacement is quadratic in x and
the consistent nodal forces make the nodal values exact, as for linear bar elements. Both are therefore
checked to the solver's tolerance.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from michell import mesh, select
from michell.analysis3d import Grid3d
from michell.attach import Case, Load, LoadOnSupport, SetUp, Support, set_up
from michell.errors import InvalidValue
from michell.filters import KeepSolid
from michell.mesh import smallest_joined
from michell.problem import LoadCase, Material, Problem
from michell.recipes import stiff_part
from michell.regions import voxel_model
from michell.row import CLOSED_FORM, DESIGN_PROPERTY, INDEPENDENT_COMPUTATION, Row, checked, g3

# ---------------------------------------------------------------------------------------------------------
# Row 32. Limits, each on its own line, fixed before any run:
# (a) a bar pulled by a face load: the mean x displacement of the loaded face equals F L / (E A) to 1e-8
#     relative, and the largest deviation of the loaded nodes' x displacement from it is below 1e-8 relative;
# (b) the same bar with Poisson's ratio 0 under a region load: the mean x displacement of the face x = L
#     equals F L / (2 E A) to 1e-8 relative;
# (c) for every load of (a), (b), (d), (e) and (f), the sum of the nodal forces plus the reported share dropped
#     on held degrees of freedom equals the stated force to 1e-12 relative (the largest component of the
#     difference over the force's length);
# (d) a bore without a solid ring, loaded through a pin, with the held skin: a short stiff_part run
#     completes and meets its limits (Result.limits_met; counted 0 when met, 1 otherwise); the nominal
#     design at 0.5 is one body joined through voxel faces (the number of bodies is at most 1, an empty
#     design counting as infinitely many); every active voxel that has a loaded node as a corner is in it
#     (the number outside it is 0); with skin=False the set-up is refused (counted 0 when refused, 1
#     otherwise);
# (e) the example bridge (examples/bridge_3d_member_size.py) set up through michell.attach: the number of
#     entries where the active voxels, the held-solid mask, the held degrees of freedom and the load vector
#     differ from the example's, each 0; the number of faces where the density filter's faces differ, 0;
#     and (slow) the compliance of the full stiff_part run equals the example's to 1e-9 relative, on one
#     thread;
# (f) the bar of (a) with its loaded end face half-way through a voxel layer, and with it a quarter through
#     one, pulled through a slab that overhangs the face by three voxels on every side: the number of loaded
#     faces that are not end faces is 0; and the mean x displacement of the loaded face against F L / (E A)
#     with L the bar's true length differs by at most 1.25 % relative. The voxel model's end lies within half
#     a voxel of the true end, and the voxel bar is solved exactly (lines (a) and (b)), so the difference is
#     the voxel model's error in length, at most h / (2 L) = 0.5 / 40.25 here; the limit, half a voxel over
#     40 mm, states that bound (fixed before either case was run). This line shows the size of that error,
#     which a finer grid shrinks; it does not hide it.
#
# Set-up of (f): the box from the origin to (41, 10, 10) mm cut by a void box from x = 40.5 mm (half-way:
# the last layer is filled to one half, a tie, so it is active and the voxel bar is 41 mm long) or from
# x = 40.25 mm (a quarter: the last layer is inactive and the voxel bar is 40 mm long); supports as in (a),
# for the box of 41 mm; the slab from the true end face, 1 mm thick, overhanging it by 3 mm on every side.
#
# Set-up of (a) and (b): the box from the origin to (40, 10, 10) mm, all design, voxels of 1 mm, steel
# (210000 MPa, Poisson's ratio 0.3 in (a) and 0 in (b), 7.85 g/cm^3); rollers through marker slabs 1 mm
# thick beyond the faces x = 0 (fix x), y = 0 (fix y) and z = 0 (fix z), each overhanging the face by 1 mm
# on every side; (a) 1000 N along x through a marker slab beyond the face x = 40; (b) the whole bar a solid
# region (the same box) loaded with 1000 N along x, held as (a) (the rollers in y and z hold the bar against
# rigid motion and, with Poisson's ratio 0, change nothing in x). The bar is solved at density 1 to a
# relative residual of 1e-12. In (b) the x shares of the nodes on x = 0 (1/80 of the force) fall on the
# support: set_up warns (LoadOnSupport) and reports them; the row records the share in its note.
#
# Set-up of (d): a design block from the origin to (40, 20, 10) mm, voxels of 1 mm, steel; a void cylinder
# (the bore) of radius 3 mm on 32 sides, axis along z through (30, 10), 14 mm long; a support through a
# marker slab beyond the face x = 0 (all directions; it covers that face, so the face is closed); the load,
# 1000 N along -y, through a pin: a cylinder of radius 2.99 mm (a hundredth of a voxel smaller than the
# bore) on 32 sides, coaxial with it and as long as the block is thick. No solid region. Member size 5 mm,
# asked mass 30 % of the block's. The schedule of stiff_part shortened: beta 1, 2, 4, 8, 16, 32 with 6, 4,
# 4, 4, 4 and 6 passes, a new bound every 4 passes, 20 passes of the member-size step: 48 passes in all.
# This schedule was set after a trial run with that of the recipe's tests (7 passes in the first step, 3 in
# the second), which ends with the member-size value at 1.03e-6 against its bound of 1e-7.
#
# Set-up of (e), as the example's: a design box 50 x 24 x 10 mm from the origin, voxels of 1 mm, steel; the
# end plates (2 mm thick at x = 0 and x = 50) and the pad (x from 22 to 28, y from 9 to 15, z from 8 to 10
# mm) as solid regions; a marker slab 1 mm thick beyond each end face, overhanging by 1 mm, as the supports
# (all directions); a marker box 1 mm thick on top of the pad, of the pad's size, as the load of 1000 N
# along -z; member size 5 mm, the asked mass 30 % of the block's, as the example computes it. The example
# leaves the density filter's inactive voxels at its default ("closed"), set_up gives "empty": the bridge has
# no inactive voxel, so the two are the same filter.

ROW_32_VOXEL = 1.0
ROW_32_FORCE = 1000.0
ROW_32_BAR = (40.0, 10.0, 10.0)
ROW_32_SOLVE_TOLERANCE = 1e-12
ROW_32_ENDS = (40.5, 40.25)  # (f): half-way and a quarter through the layer from 40 to 41 mm
ROW_32_OVERHANG = 3.0
ROW_32_LENGTH_LIMIT = 0.0125
ROW_32_STEEL = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
ROW_32_BLOCK = (40.0, 20.0, 10.0)
ROW_32_BORE = ((30.0, 10.0), 3.0, 32)  # axis (x, y), radius, sides
ROW_32_PIN_SMALLER = 0.01  # voxel sides
ROW_32_SIZE = 5.0
ROW_32_FRACTION = 0.30
ROW_32_SCHEDULE = ((1.0, 6), (2.0, 4), (4.0, 4), (8.0, 4), (16.0, 4), (32.0, 6))
ROW_32_PASSES_PER_BOUND = 4
ROW_32_SECOND_STEP_PASSES = 20


def _slab(face: str, size, depth: float = 1.0, overhang: float = 1.0) -> tuple:
    """A marker box ``depth`` thick beyond ``face`` of the box from the origin to ``size``, overhanging it by
    ``overhang`` on every side."""
    low = [-overhang] * 3
    high = [s + overhang for s in size]
    axis = "xyz".index(face[0])
    if face[1] == "-":
        low[axis], high[axis] = -depth, 0.0
    else:
        low[axis], high[axis] = size[axis], size[axis] + depth
    return tuple(low), tuple(high)


def bar(poisson_ratio: float, region_load: bool, end: float | None = None) -> SetUp:
    """The bar of (a) (``region_load`` False), (b) (True), or (f) with its end face at x = ``end`` mm (a box
    of 41 mm cut by a void box from ``end`` on)."""
    box = mesh.box((0.0, 0.0, 0.0), ROW_32_BAR)
    size, void = ROW_32_BAR, None
    if end is not None:
        size = (float(math.ceil(end)), ROW_32_BAR[1], ROW_32_BAR[2])
        box = mesh.box((0.0, 0.0, 0.0), size)
        void = {"cut": mesh.box((end, -1.0, -1.0), (size[0] + 1.0, size[1] + 1.0, size[2] + 1.0))}
    solid = {"bar": box} if region_load else None
    model = voxel_model(box, solid, void, voxel_mm=ROW_32_VOXEL)
    supports = [Support(_slab(face, size), fix=(face[0],)) for face in ("x-", "y-", "z-")]
    on = "bar" if region_load else _slab("x+", ROW_32_BAR)
    if end is not None:
        on = ((end, -ROW_32_OVERHANG, -ROW_32_OVERHANG), (end + 1.0, size[1] + ROW_32_OVERHANG, size[2] + ROW_32_OVERHANG))
    material = Material(ROW_32_STEEL.youngs_modulus, poisson_ratio, ROW_32_STEEL.density_g_cm3)
    cases = [Case("pull", [Load(on, (ROW_32_FORCE, 0.0, 0.0))])]
    return set_up(model, material, supports, cases, ROW_32_SIZE, solver_tolerance=ROW_32_SOLVE_TOLERANCE)


def bore(skin: bool = True) -> SetUp:
    """The block of (d) with its bore and the pin."""
    (cx, cy), radius, sides = ROW_32_BORE
    block = mesh.box((0.0, 0.0, 0.0), ROW_32_BLOCK)
    hole = mesh.cylinder((cx, cy, ROW_32_BLOCK[2] / 2), radius, ROW_32_BLOCK[2] + 4.0, axis="z", segments=sides)
    model = voxel_model(block, None, {"bore": hole}, voxel_mm=ROW_32_VOXEL)
    pin = mesh.cylinder(
        (cx, cy, ROW_32_BLOCK[2] / 2), radius - ROW_32_PIN_SMALLER * ROW_32_VOXEL, ROW_32_BLOCK[2], axis="z",
        segments=sides,
    )  # fmt: skip
    supports = [Support(_slab("x-", ROW_32_BLOCK))]
    cases = [Case("pin", [Load(pin, (0.0, -ROW_32_FORCE, 0.0), skin=skin)])]
    return set_up(model, ROW_32_STEEL, supports, cases, ROW_32_SIZE)


ROW_32_BRIDGE = (50.0, 24.0, 10.0)
ROW_32_PAD = ((22.0, 9.0, 8.0), (28.0, 15.0, 10.0))


@dataclass(frozen=True)
class ExampleBridge:
    """The arrays of ``examples/bridge_3d_member_size.py``, built as the example builds them."""

    grid: Grid3d
    held: np.ndarray
    loads: np.ndarray
    keep_solid: KeepSolid
    outside: dict
    mass_g: float
    problem: Problem


def example_bridge(threads: int = 1) -> ExampleBridge:
    nx, ny, nz = 50, 24, 10
    grid = Grid3d(nx, ny, nz, 1.0)
    everywhere = (-math.inf, math.inf)
    pad = ((22.0, 28.0), (9.0, 15.0), (8.0, 10.0))
    supports = np.concatenate([select.face_nodes(grid, "x-"), select.face_nodes(grid, "x+")])
    held = select.held(grid, supports)
    loads = select.load(grid, select.face_nodes(grid, "z+", within=pad), (0.0, 0.0, -1000.0))
    kept = (
        select.elements_in(grid, ((0.0, 2.0), everywhere, everywhere))
        | select.elements_in(grid, ((nx - 2.0, nx), everywhere, everywhere))
        | select.elements_in(grid, pad)
    )
    keep_solid = KeepSolid(kept, grid)
    steel = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    problem = Problem(grid, held, steel, [LoadCase("pad", loads)], threads=threads)
    mass = 0.30 * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
    outside = {"x-": "closed", "x+": "closed", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}
    return ExampleBridge(grid, held, loads, keep_solid, outside, mass, problem)


def attached_bridge(threads: int = 1) -> tuple[SetUp, float]:
    """The bridge of (e) set up through ``michell.attach``, and its asked mass in g."""
    size = ROW_32_BRIDGE
    design = mesh.box((0.0, 0.0, 0.0), size)
    solid = {
        "plate_low": mesh.box((0.0, 0.0, 0.0), (2.0, size[1], size[2])),
        "plate_high": mesh.box((size[0] - 2.0, 0.0, 0.0), size),
        "pad": mesh.box(*ROW_32_PAD),
    }
    model = voxel_model(design, solid, voxel_mm=ROW_32_VOXEL, threads=threads)
    supports = [Support(_slab("x-", size)), Support(_slab("x+", size))]
    (x0, y0, _), (x1, y1, z1) = ROW_32_PAD
    cases = [Case("pad", [Load(((x0, y0, z1), (x1, y1, z1 + 1.0)), (0.0, 0.0, -ROW_32_FORCE))])]
    attached = set_up(model, ROW_32_STEEL, supports, cases, ROW_32_SIZE, threads=threads)
    grid = attached.problem.grid
    mass = 0.30 * ROW_32_STEEL.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
    return attached, mass


def _x_displacement(attached: SetUp, face_x: float) -> np.ndarray:
    """The x displacement, at density 1, of the nodes on the plane x = ``face_x`` (mm, grid coordinates)."""
    problem = attached.problem
    grid = problem.grid
    u, _ = problem.solve(np.ones(grid.n_elements), problem.cases[0].loads)
    nodes = select.nodes_in(grid, ((face_x, face_x), (-math.inf, math.inf), (-math.inf, math.inf)))
    return u[3 * nodes]


def _force_sum_error(load) -> float:
    total = np.array([math.fsum(load.forces_n[:, a]) for a in range(3)]) + np.array(load.dropped_n)
    return float(np.abs(total - np.array(load.force_n)).max() / np.linalg.norm(load.force_n))


def _counted_refusal(build) -> tuple[float, str]:
    try:
        build()
    except InvalidValue as error:
        return 0.0, error.message
    return 1.0, "not refused"


def row_32_attach() -> list[Row]:
    """Lines (a), (b), (f), (c) and the arrays of (e)."""
    rows = []
    length, area = ROW_32_BAR[0], ROW_32_BAR[1] * ROW_32_BAR[2]
    modulus = ROW_32_STEEL.youngs_modulus
    # (a)
    pulled = bar(ROW_32_STEEL.poisson_ratio, region_load=False)
    expected = ROW_32_FORCE * length / (modulus * area)
    u = _x_displacement(pulled, length)
    rows.append(checked(
        32, "Supports and loads: bar pulled by a face load, mean x displacement of the loaded face against F L / (E A)",
        CLOSED_FORM, abs(float(np.mean(u)) / expected - 1.0), 1e-8,
        f"F L / (E A) = {g3(expected)} mm; {pulled.loads[0].nodes.size} loaded nodes on {len(pulled.loads[0].faces)} "
        f"contact faces; rollers through marker slabs on x = 0, y = 0, z = 0",
    ))  # fmt: skip
    rows.append(checked(
        32, "Supports and loads: bar pulled by a face load, largest deviation of a loaded node from F L / (E A)",
        CLOSED_FORM, float(np.abs(u - expected).max()) / expected, 1e-8,
        "relative to F L / (E A); solved to a relative residual of 1e-12", strict=True,
    ))  # fmt: skip
    # (b)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LoadOnSupport)
        body = bar(0.0, region_load=True)
    expected = ROW_32_FORCE * length / (2.0 * modulus * area)
    u = _x_displacement(body, length)
    share = body.loads[0].dropped_share
    rows.append(checked(
        32, "Supports and loads: bar with Poisson's ratio 0 under a region load, mean x displacement of the face "
        "x = L against F L / (2 E A)", CLOSED_FORM, abs(float(np.mean(u)) / expected - 1.0), 1e-8,
        f"F L / (2 E A) = {g3(expected)} mm; {body.loads[0].voxels.size} voxels; {g3(100 * share)} % of the force "
        f"on held degrees of freedom ({len([w for w in caught if w.category is LoadOnSupport])} LoadOnSupport "
        f"warning)",
    ))  # fmt: skip
    # (f)
    ends = []
    for end, label in zip(ROW_32_ENDS, ("half-way", "a quarter")):
        off = bar(ROW_32_STEEL.poisson_ratio, region_load=False, end=end)
        ends.append(off)
        load = off.loads[0]
        voxel_end = float(np.asarray(load.faces.centres_mm())[0, 0])  # grid coordinates; the origin is 0 here
        not_end = int(np.count_nonzero(load.faces.sides != 1))
        rows.append(checked(
            32, f"Supports and loads: bar with its loaded end face {label} through a voxel layer, pulled through a "
            "slab overhanging it by three voxels, loaded faces that are not end faces", CLOSED_FORM, float(not_end), 0.0,
            f"{len(load.faces)} contact faces, {load.nodes.size} loaded nodes; the voxel bar ends at {g3(voxel_end)} mm, "
            f"the true one at {end:g} mm",
        ))  # fmt: skip
        expected = ROW_32_FORCE * end / (modulus * area)
        u = _x_displacement(off, voxel_end)
        rows.append(checked(
            32, f"Supports and loads: bar with its loaded end face {label} through a voxel layer, mean x displacement "
            "of the loaded face against F L / (E A) for the true length", CLOSED_FORM,
            abs(float(np.mean(u)) / expected - 1.0), ROW_32_LENGTH_LIMIT,
            f"L = {end:g} mm; the voxel bar is {g3(voxel_end)} mm long, so its exact extension differs by "
            f"{g3(voxel_end / end - 1.0)}: the voxel model's error in length, at most half a voxel over L",
        ))  # fmt: skip
    # (c), with the set-ups of (d), (e) and (f) (their loads; the run of (d) is in row_32_bore)
    attached, _ = attached_bridge()
    loads = [*pulled.loads, *body.loads, *bore().loads, *attached.loads, *(off.loads[0] for off in ends)]
    errors = [_force_sum_error(load) for load in loads]
    rows.append(checked(
        32, "Supports and loads: largest difference of the nodal forces plus the dropped share from the stated force",
        CLOSED_FORM, max(errors), 1e-12,
        f"relative to the force; {len(loads)} loads (the face and region loads of the bar, the pin, the bridge's "
        f"pad, the two bars of (f)); dropped shares {', '.join(g3(load.dropped_share) for load in loads)}",
    ))  # fmt: skip
    # (e), the arrays
    example = example_bridge()
    active = np.asarray(attached.problem.grid.active) == 1.0
    lines = [
        ("active voxels", np.count_nonzero(active != (np.asarray(example.grid.active) == 1.0))),
        ("held-solid voxels", np.count_nonzero(attached.keep_solid.mask != example.keep_solid.mask)),
        ("held degrees of freedom", np.count_nonzero(attached.held != example.held)),
        ("entries of the load vector", np.count_nonzero(attached.problem.cases[0].loads != example.loads)),
    ]
    for label, count in lines:
        rows.append(checked(
            32, f"Supports and loads: the example bridge set up from regions and marker bodies, {label} that differ "
            "from the example's", INDEPENDENT_COMPUTATION, float(count), 0.0,
            f"{attached.problem.grid.n_elements} voxels; {int(attached.keep_solid.mask.sum())} held solid, "
            f"{int(attached.held.sum())} held degrees of freedom",
        ))  # fmt: skip
    differ = [face for face in example.outside if attached.outside[face] != example.outside[face]]
    rows.append(checked(
        32, "Supports and loads: the example bridge, faces of the density filter that differ from the example's",
        INDEPENDENT_COMPUTATION, float(len(differ)), 0.0,
        f"set_up gives {', '.join(f'{face} {word}' for face, word in attached.outside.items())}; inactive "
        f"{attached.inactive!r} (the bridge has no inactive voxel)",
    ))  # fmt: skip
    return rows


def _bodies(solid: np.ndarray, counts) -> int:
    nx, ny, nz = counts
    if not solid.any():
        return 0
    index = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    field = solid.reshape(nz, ny, nx)
    first, second = [], []
    for axis in (2, 1, 0):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        both = field[tuple(low)] & field[tuple(high)]
        first.append(index[tuple(low)][both])
        second.append(index[tuple(high)][both])
    label = smallest_joined(nx * ny * nz, np.concatenate(first), np.concatenate(second))
    return int(np.unique(label[solid.ravel()]).size)


def row_32_bore() -> list[Row]:
    """Line (d): the bore loaded through a pin, a short stiff_part run."""
    rows = []
    attached = bore()
    load = attached.loads[0]
    grid = attached.problem.grid
    block_mass = ROW_32_STEEL.mass_g(ROW_32_BLOCK[0] * ROW_32_BLOCK[1] * ROW_32_BLOCK[2])
    mass = ROW_32_FRACTION * block_mass
    part = attached.stiff_part(
        mass, schedule=ROW_32_SCHEDULE, passes_per_bound=ROW_32_PASSES_PER_BOUND,
        second_step_passes=ROW_32_SECOND_STEP_PASSES,
    )  # fmt: skip
    result = part.result
    rows.append(checked(
        32, "Supports and loads: bore loaded through a pin with the held skin, a short stiff_part run meets its "
        "limits (0 when met)", DESIGN_PROPERTY, 0.0 if result.limits_met else 1.0, 0.0,
        f"mass {g3(result.limits[0])} g of {g3(mass)} g, member-size value {g3(result.limits[1])} (at most "
        f"{g3(part.eps)}); {load.faces.area_mm2:g} mm2 of contact on {len(load.faces)} faces; skin "
        f"{load.skin.size} voxels, {g3(load.skin_mass_g)} g; {len(part.passes)} passes",
    ))  # fmt: skip
    solid = part.nominal >= 0.5
    bodies = _bodies(solid, attached.model.counts)
    rows.append(checked(
        32, "Supports and loads: bore loaded through a pin, bodies of the nominal design at 0.5 (joined through "
        "voxel faces)", DESIGN_PROPERTY, float(bodies) if bodies else math.inf, 1.0,
        f"{int(solid.sum())} solid voxels of {int(np.count_nonzero(np.asarray(grid.active)))} active",
    ))  # fmt: skip
    loaded = np.zeros(grid.n_nodes, dtype=bool)
    loaded[load.nodes[np.any(load.forces_n != 0.0, axis=1)]] = True
    nx, ny, nz = attached.model.counts
    k, j, i = np.indices((nz, ny, nx)).reshape(3, -1)
    touches = np.zeros(grid.n_elements, dtype=bool)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                touches |= loaded[(i + a) + (nx + 1) * ((j + b) + (ny + 1) * (k + c))]
    touches &= attached.model.active
    rows.append(checked(
        32, "Supports and loads: bore loaded through a pin, active voxels at a loaded node outside the nominal "
        "design at 0.5", DESIGN_PROPERTY, float(np.count_nonzero(touches & ~solid)), 0.0,
        f"{int(loaded.sum())} loaded nodes, {int(touches.sum())} active voxels at them",
    ))  # fmt: skip
    value, message = _counted_refusal(lambda: bore(skin=False))
    rows.append(checked(
        32, "Supports and loads: the same pin load with skin=False is refused", DESIGN_PROPERTY, value, 0.0, message,
    ))  # fmt: skip
    return rows


def row_32_bridge_compliance(threads: int = 1) -> list[Row]:
    """The slow line of (e): the full stiff_part run of the example bridge, from the example's arrays and
    from michell.attach, on ``threads`` threads."""
    example = example_bridge(threads)
    reference = stiff_part(example.problem, example.mass_g, ROW_32_SIZE, example.keep_solid, outside=example.outside)
    attached, mass = attached_bridge(threads)
    part = attached.stiff_part(mass)
    compliance, expected = part.result.goal, reference.result.goal
    return [
        checked(
            32, "Supports and loads: the example bridge set up through michell.attach, compliance of the full "
            "stiff_part run against the example's", INDEPENDENT_COMPUTATION, abs(compliance / expected - 1.0), 1e-9,
            f"{g3(compliance)} against {g3(expected)} N mm, {threads} thread(s), {len(part.passes)} passes",
        )
    ]  # fmt: skip


__all__ = [
    "ExampleBridge",
    "attached_bridge",
    "bar",
    "bore",
    "example_bridge",
    "row_32_attach",
    "row_32_bore",
    "row_32_bridge_compliance",
]
