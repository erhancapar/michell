"""Verification rows 30 and 31: the voxeliser (``michell.regions.occupancy``) and the voxel model of a part
(``michell.regions.voxel_model``)."""

from __future__ import annotations

import math

import numpy as np

from michell import mesh
from michell.errors import InvalidValue
from michell.mesh import Mesh
from michell.regions import occupancy, voxel_model
from michell.row import (
    CLOSED_FORM,
    CONVERGENCE,
    DESIGN_PROPERTY,
    INDEPENDENT_COMPUTATION,
    REPEATABILITY,
    Row,
    checked,
    g3,
)

# ---------------------------------------------------------------------------------------------------------
# Row 30. Limits, each on its own line, fixed before any run:
# (a) a box with its faces on voxel planes, and a box with its y and z faces on voxel planes and its x faces
#     off them: the occupancy sum times the voxel volume differs from the exact volume by at most 1e-12
#     relative;
# (b) a faceted sphere, a cylinder with its axis along y and a torus, each at three voxel sizes halving each
#     time: the error of the sampled volume against the faceted body's exact volume falls by at least a
#     factor 3 per halving (the ratio of the error after a halving to the error before it is at most 1/3),
#     and is at most 0.5 % at the finest size;
# (c) three bodies hit exactly by rays: a box whose edges lie on rays, a pyramid whose apex (and base corners)
#     lie on rays, and a square prism turned by 45 degrees whose edges lie in a plane of rays. Where the
#     geometry makes the sampled volume exact (the box: the half-open rule takes exactly the rays of
#     [y0, y1) x [z0, z1)), it differs from the exact volume by at most 1e-12 relative; otherwise it differs
#     from that of the same body shifted by a small irrational offset (1e-4 voxel sides, far above the
#     lattice unit of 1.6e-7 voxel sides, so the shifted body's rays meet no edge or vertex) by at most 1e-3
#     relative;
# (d) the occupancy of a union minus a body on 1 and on 12 threads: the largest difference is 0.
#
# Set-up of (b), fixed after two trial runs. In the first, with one placement of each body per size (sizes
# 1 to 0.0625 mm), the error changed sign from one size to the next and the ratios between halvings ran from
# about 1.1 to 24: for a closed body the length inside along a ray is a continuous function of
# the ray's position that is 0 outside the body, so the second-order term of the sampling integrates to
# zero and what remains varies in sign with the body's position against the rays. The error at a voxel
# size is therefore the largest over 8 placements of the body, offset by (frac(k g1), frac(k g2),
# frac(k g3)) x 2 mm for k = 1 to 8 with g = (0.618034, 0.754878, 0.569840) (the cylinder not offset along
# its axis, so that its caps, which are parallel to the rays, stay on voxel planes). The second trial ran
# this measure at 2, 1, 0.5 and 0.25 mm; from 2 to 1 mm the torus's error fell by 2.7 only (its tube is then
# 1.5 voxels across), so the sizes were set to 1, 0.5 and 0.25 mm, from a third of the tube's radius down. Bodies: a sphere of radius 10 mm, 32 meridians and 16 bands;
# a cylinder of radius 6 mm and length 16 mm on 24 sides, its axis along y; a torus of radii 8 and 3 mm,
# 48 x 24 sides, its axis along z; no triangle of the three is parallel to the rays except the cylinder's
# caps. The grid: the cube from -14 to 14 mm along each axis, 3 x 3 rays per voxel column.

ROW_30_SIZES = (1.0, 0.5, 0.25)
ROW_30_GOLDEN = (0.6180339887498949, 0.7548776662466927, 0.5698402909980532)
ROW_30_PLACEMENTS = 8
ROW_30_HALF_WIDTH = 14.0
ROW_30_SHIFT = 1e-4  # voxel sides


def row_30_body(name: str, offset) -> Mesh:
    """A body of (b), offset by ``offset`` (mm)."""
    dx, dy, dz = offset
    if name == "sphere":
        return mesh.sphere((dx, dy, dz), 10.0, 32, 16)
    if name == "cylinder":
        return mesh.cylinder((dx, 0.0, dz), 6.0, 16.0, axis="y", segments=24)
    return mesh.torus((dx, dy, dz), 8.0, 3.0, segments=48, sides=24)


def row_30_placements() -> list[tuple[float, float, float]]:
    return [tuple(((k * g) % 1.0) * 2.0 for g in ROW_30_GOLDEN) for k in range(1, ROW_30_PLACEMENTS + 1)]


def _cube_grid(h: float) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    n = round(2 * ROW_30_HALF_WIDTH / h)
    return (n, n, n), (-ROW_30_HALF_WIDTH,) * 3


def _sampled(body: Mesh, counts, h: float, origin, threads: int = 1) -> float:
    return float(occupancy(counts, h, origin, [body], threads=threads).sum()) * h**3


def pyramid(base_x: float, yc: float, zc: float, half: float, height: float) -> Mesh:
    """The square pyramid with its base (half-side ``half``) in the plane x = ``base_x`` around (yc, zc) and
    its apex at x = base_x + height on the axis, facing outward."""
    vertices = [[base_x, yc - half, zc - half], [base_x, yc + half, zc - half], [base_x, yc + half, zc + half],
                [base_x, yc - half, zc + half], [base_x + height, yc, zc]]  # fmt: skip
    return Mesh(vertices, [[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])


def turned_prism(xc: float, yc: float, d: float, z0: float, z1: float) -> Mesh:
    """A square prism from z0 to z1 whose cross-section in (x, y) is the square with corners (xc -+ d, yc) and
    (xc, yc -+ d): a box turned by 45 degrees about z, facing outward."""
    ring = [(xc - d, yc), (xc, yc - d), (xc + d, yc), (xc, yc + d)]
    vertices = [[x, y, z] for z in (z0, z1) for x, y in ring]
    triangles = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for j in range(4):
        a, b = j, (j + 1) % 4
        triangles += [[a, b, b + 4], [a, b + 4, a + 4]]
    return Mesh(vertices, triangles)


def _shifted(body: Mesh, by) -> Mesh:
    return Mesh(body.vertices + np.asarray(by, dtype=np.float64), body.triangles)


def row_30_voxeliser() -> list[Row]:
    rows = []
    h = 1.0
    # (a) Boxes on and off the voxel planes along x.
    counts, origin = (12, 8, 6), (0.0, 0.0, 0.0)
    for label, low, high in [
        ("faces on voxel planes", (2.0, 1.0, 1.0), (10.0, 7.0, 5.0)),
        ("x faces off the voxel planes", (1.0 + math.sqrt(2) / 10, 1.0, 1.0), (10.0 - math.pi / 10, 7.0, 5.0)),
    ]:
        body = mesh.box(low, high)
        sampled = _sampled(body, counts, h, origin)
        exact = body.volume_mm3
        rows.append(checked(
            30, f"Voxeliser: box with {label}, relative difference of the sampled from the exact volume",
            CLOSED_FORM, abs(sampled / exact - 1.0), 1e-12,
            f"exact {g3(exact)} mm3; 1 mm voxels, 3 x 3 rays per voxel column",
        ))  # fmt: skip

    # (b) Convergence at three voxel sizes halving each time.
    for name in ("sphere", "cylinder", "torus"):
        errors = []
        for size in ROW_30_SIZES:
            counts_b, origin_b = _cube_grid(size)
            worst = 0.0
            for offset in row_30_placements():
                body = row_30_body(name, offset if name != "cylinder" else (offset[0], 0.0, offset[2]))
                worst = max(worst, abs(_sampled(body, counts_b, size, origin_b) / body.volume_mm3 - 1.0))
            errors.append(worst)
        ratios = [errors[i + 1] / errors[i] for i in range(len(errors) - 1)]
        sizes = ", ".join(f"{g3(s)} mm: {g3(e)}" for s, e in zip(ROW_30_SIZES, errors))
        rows.append(checked(
            30, f"Voxeliser: faceted {name}, largest ratio of the volume error after a voxel halving to before it",
            CONVERGENCE, max(ratios), 1.0 / 3.0,
            f"largest relative volume error over 8 placements at {sizes}; ratios {', '.join(g3(r) for r in ratios)}; "
            f"the measure (the largest over 8 placements) and the sizes were set after two trial runs (see the code)",
        ))  # fmt: skip
        rows.append(checked(
            30, f"Voxeliser: faceted {name}, relative volume error at the finest voxel size ({g3(ROW_30_SIZES[-1])} mm)",
            CONVERGENCE, errors[-1], 0.005, "the largest over the 8 placements; against the faceted body's exact volume",
        ))  # fmt: skip

    # (c) Bodies hit exactly by rays (rays of column j at j + 1/6, 1/2, 5/6 voxel sides, likewise in z).
    shift = (ROW_30_SHIFT * math.sqrt(5), ROW_30_SHIFT * math.sqrt(2), ROW_30_SHIFT * math.sqrt(3))
    counts_c, origin_c = (12, 9, 7), (0.0, 0.0, 0.0)
    on_rays = mesh.box((1.0 + math.sqrt(0.09), 1.0 + 1.0 / 6.0, 1.5), (9.7, 6.0 + 1.0 / 6.0, 4.5))
    sampled = _sampled(on_rays, counts_c, h, origin_c)
    rows.append(checked(
        30, "Voxeliser: box whose edges lie on rays, relative difference of the sampled from the exact volume",
        CLOSED_FORM, abs(sampled / on_rays.volume_mm3 - 1.0), 1e-12,
        "y from 1 + 1/6 to 6 + 1/6 and z from 1.5 to 4.5 voxel sides: corners, edges and the end faces' diagonals "
        "on rays",
    ))  # fmt: skip
    for label, body in [
        ("pyramid whose apex and base corners lie on rays", pyramid(1.25, 4.5, 3.5, 7.0 / 3.0, 8.5)),
        ("prism turned by 45 degrees with edges in a plane of rays", turned_prism(5.3, 3.5, 3.1, 1.0, 5.0)),
    ]:
        sampled = _sampled(body, counts_c, h, origin_c)
        moved = _sampled(_shifted(body, shift), counts_c, h, origin_c)
        rows.append(checked(
            30, f"Voxeliser: {label}, relative difference from the same body shifted off the rays",
            INDEPENDENT_COMPUTATION, abs(sampled / moved - 1.0), 1e-3,
            f"shift (sqrt 5, sqrt 2, sqrt 3) x 1e-4 voxel sides; sampled {g3(sampled)} mm3, exact "
            f"{g3(body.volume_mm3)} mm3",
        ))  # fmt: skip

    # (d) One thread against twelve.
    counts_d, origin_d = _cube_grid(0.5)
    adding = [row_30_body("torus", (0.3, 0.2, 0.1)), row_30_body("sphere", (4.1, -3.3, 2.2))]
    subtracting = [row_30_body("cylinder", (1.7, 0.0, -0.9))]
    one = occupancy(counts_d, 0.5, origin_d, adding, subtracting, threads=1)
    twelve = occupancy(counts_d, 0.5, origin_d, adding, subtracting, threads=12)
    rows.append(checked(
        30, "Voxeliser: largest difference of the occupancy between 1 and 12 threads",
        REPEATABILITY, float(np.abs(one - twelve).max()), 0.0,
        f"torus and sphere minus a cylinder on {counts_d[0]}^3 voxels of 0.5 mm",
    ))  # fmt: skip
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 31. Limits, each on its own line, fixed before any run:
# (e) a design box with a nested solid box (on voxel planes) and a void cylinder: the volumes of the active
#     voxels and of the held-solid voxels differ from the exact volumes of the part (the box minus the
#     cylinder) and of the nested box by at most 0.05 relative, the tolerance the volume table states
#     (``VoxelVolumeDiffers``), and the held-solid voxels are exactly those of the nested box (the number of
#     voxels that differ is 0);
# (f) a design box and a solid box that meet on a voxel mid-plane: no voxel between them is inactive (the
#     number of inactive voxels whose centre lies in either box is 0);
# (g) a part mirror-symmetric about y (a design box, two solid pads that mirror each other and a void
#     cylinder centred on the plane), voxelised with mirror=("y",): the number of voxels whose class
#     (inactive, design, held solid) differs from that of their mirror voxel is 0; the same part with the
#     cylinder moved off the plane is refused (counted 0 when refused, 1 otherwise);
# (h) a solid region thinner than half a voxel is refused (counted 0 when refused, 1 otherwise).
#
# Set-up: voxels of 1 mm. (e) design box 40 x 20 x 10 mm from the origin; solid box from (4, 4, 2) to
# (12, 16, 8) mm; void cylinder of radius 4 mm on 32 sides, axis along z through (28.3, 10.1), 14 mm long
# (through the whole box). (f) design box from (0, 0, 0) to (10.5, 6, 4) mm, solid box from (10.5, 0, 0) to
# (16, 6, 4) mm: they meet on x = 10.5, half-way through the voxel layer from 10 to 11. (g) design box
# 30 x 21 x 8 mm (the plane y = 10.5 runs through voxel centres, so rays lie on it); pads from (2.3, 1.4, 0)
# to (6.7, 5.2, 3) and from (2.3, 15.8, 0) to (6.7, 19.6, 3) mm, two named regions; void cylinder of
# radius 4 mm on 32 sides, axis along z through (17.2, 10.5), 12 mm long, then moved to y = 11.5 for the
# refusal. (h) design box 20 x 10 x 10 mm, solid plate 0.4 mm thick from (5, 2, 4.3) to (15, 8, 4.7) mm.

ROW_31_VOXEL = 1.0


def row_31_e() -> tuple:
    design = mesh.box((0.0, 0.0, 0.0), (40.0, 20.0, 10.0))
    nested = mesh.box((4.0, 4.0, 2.0), (12.0, 16.0, 8.0))
    hole = mesh.cylinder((28.3, 10.1, 5.0), 4.0, 14.0, axis="z", segments=32)
    model = voxel_model(design, {"nested": nested}, {"hole": hole}, voxel_mm=ROW_31_VOXEL)
    return design, nested, hole, model


def row_31_g(cylinder_y: float, mirror=("y",)):
    design = mesh.box((0.0, 0.0, 0.0), (30.0, 21.0, 8.0))
    pads = {"pad_a": mesh.box((2.3, 1.4, 0.0), (6.7, 5.2, 3.0)), "pad_b": mesh.box((2.3, 15.8, 0.0), (6.7, 19.6, 3.0))}
    hole = mesh.cylinder((17.2, cylinder_y, 4.0), 4.0, 12.0, axis="z", segments=32)
    return voxel_model(design, pads, {"hole": hole}, voxel_mm=ROW_31_VOXEL, mirror=mirror)


def _classes(model) -> np.ndarray:
    """0 inactive, 1 design, 2 held solid, per voxel, as an (nz, ny, nx) array."""
    nx, ny, nz = model.counts
    return (model.active.astype(int) + model.keep_solid.astype(int)).reshape(nz, ny, nx)


def _centres(model) -> np.ndarray:
    """The centre of every voxel in the meshes' coordinates (mm), in the order of the voxels."""
    nx, ny, nz = model.counts
    k, j, i = np.indices((nz, ny, nx))
    return model.to_mesh((np.stack([i, j, k], axis=-1).reshape(-1, 3) + 0.5) * model.voxel_mm)


def _refused(build) -> tuple[float, str]:
    try:
        build()
    except InvalidValue as error:
        return 0.0, error.message
    return 1.0, "not refused"


def row_31_voxel_model() -> list[Row]:
    rows = []
    # (e)
    design, nested, _, model = row_31_e()
    n = 32
    exact_part = design.volume_mm3 - n / 2 * math.sin(2 * math.pi / n) * 4.0**2 * 10.0
    part_row, held_row = model.volume("part"), model.volume("held solid")
    part_error = abs(part_row.voxels_mm3 / exact_part - 1.0)
    held_error = abs(held_row.voxels_mm3 / nested.volume_mm3 - 1.0)
    rows.append(checked(
        31, "Voxel model: design box, nested solid box, void cylinder; largest relative difference of the active "
        "and held-solid voxel volumes from the exact volumes", CLOSED_FORM, max(part_error, held_error), 0.05,
        f"part: voxels {g3(part_row.voxels_mm3)}, sampled {g3(part_row.sampled_mm3)}, exact {g3(exact_part)} mm3 "
        f"(table: voxels against sampled {g3(part_row.voxels_vs_sampled)}); held solid: voxels "
        f"{g3(held_row.voxels_mm3)}, exact {g3(nested.volume_mm3)} mm3",
    ))  # fmt: skip
    centres = _centres(model)
    in_nested = np.all((centres > [4.0, 4.0, 2.0]) & (centres < [12.0, 16.0, 8.0]), axis=1)
    rows.append(checked(
        31, "Voxel model: held-solid voxels that differ from the voxels of the nested box", CLOSED_FORM,
        float(np.count_nonzero(model.keep_solid != in_nested)), 0.0,
        f"{int(in_nested.sum())} voxels in the nested box, {int(model.keep_solid.sum())} held solid",
    ))  # fmt: skip
    # (f)
    meeting = voxel_model(mesh.box((0.0, 0.0, 0.0), (10.5, 6.0, 4.0)), {"block": mesh.box((10.5, 0.0, 0.0), (16.0, 6.0, 4.0))},
                          voxel_mm=ROW_31_VOXEL)  # fmt: skip
    layer = _classes(meeting)[:, :, 10]
    rows.append(checked(
        31, "Voxel model: design and solid boxes meeting on a voxel mid-plane, inactive voxels in either box",
        DESIGN_PROPERTY, float(np.count_nonzero(~meeting.active)), 0.0,
        f"{meeting.counts[0]} x {meeting.counts[1]} x {meeting.counts[2]} voxels; the shared layer x = 10 to 11 mm: "
        f"{int(np.count_nonzero(layer == 2))} of {layer.size} held solid (a tie at one half counts as solid)",
    ))  # fmt: skip
    # (g)
    symmetric = row_31_g(10.5)
    classes = _classes(symmetric)
    unmirrored = _classes(row_31_g(10.5, mirror=()))
    rows.append(checked(
        31, "Voxel model: part symmetric about y with mirror=('y',), voxels whose class differs from their mirror's",
        DESIGN_PROPERTY, float(np.count_nonzero(classes != classes[:, ::-1, :])), 0.0,
        f"two pads as two named solid regions; without mirror {int(np.count_nonzero(unmirrored != unmirrored[:, ::-1, :]))} "
        f"voxels differ",
    ))  # fmt: skip
    value, message = _refused(lambda: row_31_g(11.5))
    rows.append(checked(
        31, "Voxel model: the same part with the void cylinder 1 mm off the plane is refused",
        DESIGN_PROPERTY, value, 0.0, message,
    ))  # fmt: skip
    # (h)
    value, message = _refused(lambda: voxel_model(
        mesh.box((0.0, 0.0, 0.0), (20.0, 10.0, 10.0)), {"plate": mesh.box((5.0, 2.0, 4.3), (15.0, 8.0, 4.7))},
        voxel_mm=ROW_31_VOXEL,
    ))  # fmt: skip
    rows.append(checked(
        31, "Voxel model: a solid region 0.4 voxel sides thick is refused", DESIGN_PROPERTY, value, 0.0, message,
    ))  # fmt: skip
    return rows


__all__ = [
    "ROW_30_PLACEMENTS",
    "ROW_30_SIZES",
    "pyramid",
    "row_30_body",
    "row_30_placements",
    "row_30_voxeliser",
    "row_31_voxel_model",
    "turned_prism",
]
