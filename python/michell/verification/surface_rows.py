"""Verification rows 34 and 35: the iso-surface of a field (``michell.surface.iso_surface``), and Taubin
smoothing (``michell.surface.smooth``) with the 3MF file (``michell.mesh.write_3mf``, ``read_3mf``)."""

from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from michell.mesh import Mesh, close_pairs, read_3mf, write_3mf
from michell.row import CLOSED_FORM, CONVERGENCE, DESIGN_PROPERTY, REPEATABILITY, Row, checked, g3
from michell.surface import ScalarField, iso_surface, smooth

# ---------------------------------------------------------------------------------------------------------
# Row 34. Limits, each on its own line, fixed before any run:
# (a) the distance field of a sphere of radius 10 mm at three lattice spacings halving each time (1, 0.5 and
#     0.25 mm): the surface is closed (0 defects), one body, genus 0, at every spacing; the error of its
#     volume against the exact sphere's falls by at least a factor 3 per halving (the ratio of the error after
#     a halving to the error before it is at most 1/3) and is below 0.2 % at the finest spacing; the error of
#     its area is below 1 % at the finest spacing;
# (b) the field of a torus of major radius 12 mm and minor radius 4 mm at 0.5 mm: closed, one body, genus 1;
#     volume within 0.5 % of the exact torus's;
# (c) two spheres of radius 6 mm with centres 11, 12 and 13 mm apart (joined, touching, apart), the field the
#     larger of their two fields, at 0.5 mm: closed in all three; one body joined, two apart, and one or two
#     touching (reported);
# (d) fields with whole layers of samples exactly at the level: a box's 0/1 field with the level at 0.5,
#     turned into steps at 0, 0.5 and 1 (its outermost layer of points at 0.5), a sheet one cell thick (one
#     layer of points at 1 between layers at 0) and the same sheet at 0.5: closed with no triangle without
#     area (0 defects), and no two vertices closer than 1e-9 mm (0 pairs);
# (e) the surface of the torus of (b) at 0.25 mm on 1 and on 12 threads: vertices and triangles identical
#     (0 differences).
#
# Set-up, fixed before the row was written; a trial run of (a) with the sphere centred on a point and off
# it gave the same convergence. The fields are signed distances, positive inside: R - |x - c| for a sphere
# of radius R around c, r - sqrt((rho - R)^2 + z^2) for the torus (rho the distance from the z axis), level
# 0. (a) the lattice spans the cube from -12 to 12 mm, the sphere's centre at (0.123, 0.271, 0.0577) mm, off
# every lattice point. (b, e) the lattice spans -18 to 18 mm in x and y and -6 to 6 mm in z, the torus
# centred on the origin. (c) the lattice spans -14 to 14 mm; centres at (-d/2, 0, 0) and (d/2, 0, 0), so the
# touching spheres meet at the origin, a lattice point where the field is exactly 0. (d) 20 x 20 x 20 points
# 0.5 mm apart from the origin: the box holds the points 4 to 15 along each axis, 0.5 on its outermost points
# and 1 inside; each sheet holds the points 4 to 15 along x and y on the layer k = 10.

ROW_34_SPACINGS = (1.0, 0.5, 0.25)
ROW_34_SPHERE_CENTRE = (0.123, 0.271, 0.0577)
ROW_34_GAP_MM = 1e-9


def lattice_field(low, high, spacing: float, function) -> ScalarField:
    """The field ``function(x, y, z)`` (arrays of shape (nz, ny, nx), mm) sampled on the lattice from the
    point ``low`` to the point ``high`` at ``spacing``."""
    counts = [int(round((h - lo) / spacing)) + 1 for lo, h in zip(low, high)]
    x, y, z = (lo + spacing * np.arange(n) for lo, n in zip(low, counts))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return ScalarField(function(xx, yy, zz), spacing, tuple(low))


def sphere_field(spacing: float, centre=ROW_34_SPHERE_CENTRE, radius: float = 10.0, half: float = 12.0) -> ScalarField:
    cx, cy, cz = centre
    return lattice_field((-half,) * 3, (half,) * 3, spacing,
                         lambda x, y, z: radius - np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))  # fmt: skip


def torus_field(spacing: float, major: float = 12.0, minor: float = 4.0) -> ScalarField:
    return lattice_field((-18.0, -18.0, -6.0), (18.0, 18.0, 6.0), spacing,
                         lambda x, y, z: minor - np.sqrt((np.sqrt(x**2 + y**2) - major) ** 2 + z**2))  # fmt: skip


def two_spheres_field(distance: float, spacing: float = 0.5, radius: float = 6.0) -> ScalarField:
    def field(x, y, z):
        first = radius - np.sqrt((x + distance / 2) ** 2 + y**2 + z**2)
        second = radius - np.sqrt((x - distance / 2) ** 2 + y**2 + z**2)
        return np.maximum(first, second)

    return lattice_field((-14.0,) * 3, (14.0,) * 3, spacing, field)


def stepped_fields() -> dict[str, ScalarField]:
    """The fields of (d), each at the level 0.5."""
    box = np.zeros((20, 20, 20))
    box[4:16, 4:16, 4:16] = 0.5
    box[5:15, 5:15, 5:15] = 1.0
    sheet = np.zeros((20, 20, 20))
    sheet[10, 4:16, 4:16] = 1.0
    return {
        "box stepped 0, 0.5, 1": ScalarField(box, 0.5),
        "sheet one cell thick at 1": ScalarField(sheet, 0.5),
        "sheet one cell thick at the level": ScalarField(sheet / 2, 0.5),
    }


def faults(mesh: Mesh) -> int:
    """The sum of the counts of ``Mesh.defects()``."""
    return sum(mesh.defects().values())


def smallest_gap(mesh: Mesh, within: float) -> float:
    """The smallest distance between two vertices, among the pairs no farther apart than ``within`` (mm);
    infinity when there is no such pair."""
    first, second = close_pairs(mesh.vertices, within)
    if len(first) == 0:
        return math.inf
    return float(np.linalg.norm(mesh.vertices[first] - mesh.vertices[second], axis=1).min())


def row_34_iso_surface() -> list[Row]:
    rows = []
    # (a)
    volume_exact, area_exact = 4.0 / 3.0 * math.pi * 10.0**3, 4.0 * math.pi * 10.0**2
    volume_errors, area_errors, shape_faults, notes = [], [], 0, []
    for spacing in ROW_34_SPACINGS:
        surface = iso_surface(sphere_field(spacing), 0.0)
        bodies = surface.bodies()
        genus = bodies[0].genus() if len(bodies) == 1 else -1
        shape_faults += faults(surface) + abs(len(bodies) - 1) + abs(genus)
        volume_errors.append(abs(surface.volume_mm3 / volume_exact - 1.0))
        area_errors.append(abs(surface.area_mm2 / area_exact - 1.0))
        notes.append(f"{g3(spacing)} mm: {len(surface.triangles)} triangles, {len(bodies)} body, genus {genus}")
    rows.append(checked(
        34, "Iso-surface: sphere's distance field at 1, 0.5 and 0.25 mm, defects plus bodies other than one plus "
        "genus other than 0", DESIGN_PROPERTY, float(shape_faults), 0.0, "; ".join(notes),
    ))  # fmt: skip
    ratios = [volume_errors[i + 1] / volume_errors[i] for i in range(len(volume_errors) - 1)]
    sizes = ", ".join(f"{g3(s)} mm: {g3(e)}" for s, e in zip(ROW_34_SPACINGS, volume_errors))
    rows.append(checked(
        34, "Iso-surface: sphere, largest ratio of the volume error after a halving of the spacing to before it",
        CONVERGENCE, max(ratios), 1.0 / 3.0,
        f"relative volume errors {sizes}; ratios {', '.join(g3(r) for r in ratios)}",
    ))  # fmt: skip
    rows.append(checked(
        34, "Iso-surface: sphere, relative volume error at 0.25 mm", CLOSED_FORM, volume_errors[-1], 0.002,
        "against 4/3 pi r^3, r = 10 mm", strict=True,
    ))  # fmt: skip
    rows.append(checked(
        34, "Iso-surface: sphere, relative area error at 0.25 mm", CLOSED_FORM, area_errors[-1], 0.01,
        f"against 4 pi r^2; at 1, 0.5 mm: {g3(area_errors[0])}, {g3(area_errors[1])}", strict=True,
    ))  # fmt: skip

    # (b)
    torus = iso_surface(torus_field(0.5), 0.0)
    bodies = torus.bodies()
    genus = bodies[0].genus() if len(bodies) == 1 else -1
    rows.append(checked(
        34, "Iso-surface: torus's field at 0.5 mm, defects plus bodies other than one plus genus other than 1",
        DESIGN_PROPERTY, float(faults(torus) + abs(len(bodies) - 1) + abs(genus - 1)), 0.0,
        f"{len(torus.triangles)} triangles, {len(bodies)} body, genus {genus}, Euler characteristic "
        f"{torus.euler_characteristic()}",
    ))  # fmt: skip
    exact = 2.0 * math.pi**2 * 12.0 * 4.0**2
    rows.append(checked(
        34, "Iso-surface: torus at 0.5 mm, relative volume error", CLOSED_FORM, abs(torus.volume_mm3 / exact - 1.0),
        0.005, f"against 2 pi^2 R r^2 = {g3(exact)} mm3; surface {g3(torus.volume_mm3)} mm3",
    ))  # fmt: skip

    # (c)
    for distance, label, allowed in [(11.0, "joined", (1,)), (12.0, "touching", (1, 2)), (13.0, "apart", (2,))]:
        surface = iso_surface(two_spheres_field(distance), 0.0)
        count = len(surface.bodies())
        rows.append(checked(
            34, f"Iso-surface: two spheres {g3(distance)} mm apart ({label}), defects plus a body count other than "
            f"{' or '.join(str(a) for a in allowed)}", DESIGN_PROPERTY,
            float(faults(surface) + (0 if count in allowed else 1)), 0.0,
            f"{count} {'body' if count == 1 else 'bodies'}, {len(surface.triangles)} triangles"
            + ("; the spheres meet at a lattice point exactly at the level, which counts as inside"
               if label == "touching" else ""),
        ))  # fmt: skip

    # (d)
    total_faults, close, notes = 0, 0, []
    for name, field in stepped_fields().items():
        surface = iso_surface(field, 0.5)
        total_faults += faults(surface)
        close += len(close_pairs(surface.vertices, ROW_34_GAP_MM)[0])
        within = field.spacing_mm / 4
        gap = smallest_gap(surface, within)
        gap_text = f"{g3(gap)} mm" if math.isfinite(gap) else f"above {g3(within)} mm"
        notes.append(f"{name}: {len(surface.triangles)} triangles, smallest gap between vertices {gap_text}")
    rows.append(checked(
        34, "Iso-surface: fields with whole layers of samples at the level, defects (no triangle without area)",
        DESIGN_PROPERTY, float(total_faults), 0.0, "box stepped 0, 0.5, 1 and two sheets one cell thick, 0.5 mm",
    ))  # fmt: skip
    rows.append(checked(
        34, "Iso-surface: fields with whole layers of samples at the level, pairs of vertices closer than 1e-9 mm",
        DESIGN_PROPERTY, float(close), 0.0, "; ".join(notes),
    ))  # fmt: skip

    # (e)
    field = torus_field(0.25)
    one, twelve = iso_surface(field, 0.0, threads=1), iso_surface(field, 0.0, threads=12)
    same_shape = one.vertices.shape == twelve.vertices.shape and one.triangles.shape == twelve.triangles.shape
    differences = (
        int(np.count_nonzero(one.vertices != twelve.vertices))
        + int(np.count_nonzero(one.triangles != twelve.triangles))
        if same_shape else max(one.vertices.size, twelve.vertices.size)
    )  # fmt: skip
    rows.append(checked(
        34, "Iso-surface: torus at 0.25 mm on 1 and 12 threads, entries of the vertices and triangles that differ",
        REPEATABILITY, float(differences), 0.0,
        f"{field.counts[0]} x {field.counts[1]} x {field.counts[2]} points; {len(one.vertices)} vertices, "
        f"{len(one.triangles)} triangles",
    ))  # fmt: skip
    return rows


# ---------------------------------------------------------------------------------------------------------
# Row 35. Limits, each on its own line, fixed before any run:
# (f) Taubin smoothing (lam 0.5, mu -0.53), 50 iterations, of the torus surface of row 34 (b): still closed
#     (0 defects) and genus 1, and the volume changes by less than 2 %; the roughness (the mean angle between
#     the normals of neighbouring triangles) is reported before and after, not judged;
# (g) the same torus surface written as a 3MF file and read back: the vertices and the triangles identical
#     (0 entries differ), and the package lists exactly the three required parts ([Content_Types].xml,
#     _rels/.rels, 3D/3dmodel.model; the count of parts missing or extra is 0).

ROW_35_ITERATIONS = 50
ROW_35_PARTS = ("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model")


def row_35_smoothing_and_3mf() -> list[Row]:
    rows = []
    torus = iso_surface(torus_field(0.5), 0.0)
    # (f)
    result = smooth(torus, ROW_35_ITERATIONS)
    smoothed = result.mesh
    bodies = smoothed.bodies()
    genus = bodies[0].genus() if len(bodies) == 1 else -1
    rows.append(checked(
        35, "Taubin smoothing: torus surface, 50 iterations, defects plus genus other than 1", DESIGN_PROPERTY,
        float(faults(smoothed) + abs(genus - 1)), 0.0,
        f"{len(bodies)} body, genus {genus}; largest move of a vertex {g3(result.largest_move_mm)} mm",
    ))  # fmt: skip
    rows.append(checked(
        35, "Taubin smoothing: torus surface, 50 iterations, relative change of the volume", DESIGN_PROPERTY,
        abs(result.volume_change), 0.02,
        f"roughness (mean angle between neighbouring normals) {g3(torus.roughness_deg())} degrees before, "
        f"{g3(smoothed.roughness_deg())} after; lam 0.5, mu -0.53", strict=True,
    ))  # fmt: skip
    # (g)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "torus.3mf"
        write_3mf(path, torus)
        back = read_3mf(path)
        with zipfile.ZipFile(path) as package:
            names = package.namelist()
        size = path.stat().st_size
    same_shape = back.vertices.shape == torus.vertices.shape and back.triangles.shape == torus.triangles.shape
    differences = (
        int(np.count_nonzero(back.vertices != torus.vertices))
        + int(np.count_nonzero(back.triangles != torus.triangles))
        if same_shape else max(back.vertices.size, torus.vertices.size)
    )  # fmt: skip
    rows.append(checked(
        35, "3MF: torus surface written and read back, entries of the vertices and triangles that differ",
        DESIGN_PROPERTY, float(differences), 0.0,
        f"{len(torus.vertices)} vertices, {len(torus.triangles)} triangles; file {g3(size / 1e6)} MB",
    ))  # fmt: skip
    rows.append(checked(
        35, "3MF: parts of the package missing from or beyond the three required", DESIGN_PROPERTY,
        float(len(set(names) ^ set(ROW_35_PARTS)) + len(names) - len(set(names))), 0.0, ", ".join(names),
    ))  # fmt: skip
    return rows


__all__ = [
    "ROW_34_SPACINGS",
    "faults",
    "lattice_field",
    "row_34_iso_surface",
    "row_35_smoothing_and_3mf",
    "smallest_gap",
    "sphere_field",
    "stepped_fields",
    "torus_field",
    "two_spheres_field",
]
