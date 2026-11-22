"""Verification rows 36 and 37: the part's closed mesh (``michell.part_mesh.finish``). Row 36 on synthetic
results (a field built by hand, no optimisation): three arrangements of the solid regions, the exact
interfaces and the refusals. Row 37 on optimised parts: the held skin of row 32's bore, and the example
bridge through ``michell run``."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from michell import mesh
from michell.attach import Case, Load, SetUp, Support, set_up
from michell.mesh import read_3mf, read_stl
from michell.optimise import Result
from michell.part_mesh import DELIVERED_TOLERANCE, INTERFACE_MM, LEVEL_TOLERANCE, MeshRefused, finish
from michell.problem import Material
from michell.recipes import StiffPart
from michell.regions import voxel_model
from michell.row import DESIGN_PROPERTY, INDEPENDENT_COMPUTATION, Row, checked, g3, not_run
from michell.verification.attach_rows import (
    ROW_32_BLOCK,
    ROW_32_FRACTION,
    ROW_32_PASSES_PER_BOUND,
    ROW_32_SCHEDULE,
    ROW_32_SECOND_STEP_PASSES,
    ROW_32_STEEL,
    bore,
)
from michell.verification.problem_file_rows import _example

# ---------------------------------------------------------------------------------------------------------
# Row 36. Limits, each on its own line, fixed before any run:
# (a) three arrangements of two solid boxes against the design box: nested inside it, overlapping it, and
#     touching it on a face: gates failed 0 in each; the organic volume within 0.2 % of its target in each;
# (b) a solid box whose outer face lies on the envelope, and a void cylinder (32 sides) through the part:
#     the largest distance of the mesh's vertices on that face from its plane at most 1e-6 mm; the number of
#     the mesh's vertices on the cylinder's wall that are not vertices of the void mesh: 0;
# (e) refusals: a result with a detached lump above 1 % of the whole is refused by the gate "one body" (0
#     when refused); a lump below 1 % is dropped and listed (0 when it is); a solid region apart from the
#     design region, which cannot be joined, is refused by the gate "envelope" (0 when refused).
#
# Set-up, fixed before the row was written. Voxels of 1 mm on a 40 x 20 x 12 mm part, steel. The field is
# built by hand: two bars of radius 3 mm through the centre (20, 10, 6) in the xy plane, with slopes +0.4
# and -0.4 (so they cross, and their ends are cut by the part's faces y = 0 and y = 20), the field
# clip(0.5 + (3 - d) / 2, 0, 1) at distance d from a bar's axis (the larger of the two); the nominal design
# is the field at 0.5 or more, and 1 on the voxels held solid. (a) nested: design box [0, 40] along x, solid
# boxes [0, 4] and [36, 40]; overlapping: design [3, 37], solids [0, 5] and [35, 40]; touching: design
# [4, 36], solids [0, 4] and [36, 40]; every box spans the part in y and z. Support on the low solid box, a
# load of 1000 N downward on the high one. (b) design [0, 40]; a solid box [30, 40] x [4, 16] x [0, 12]
# whose face x = 40 lies on the envelope; a solid plate [0, 4] (the support); a void cylinder of radius 3 mm,
# 32 sides, axis z through (20, 10), from face z = 0 to face z = 12; the field of the bars joined with a boss
# clip(0.5 + (6 - r) / 2) at distance r from the cylinder's axis, so that the part surrounds the hole. A
# vertex is on the box's face when it belongs to a triangle the mesh took from that face; a vertex is on the
# cylinder's wall when it lies within 1e-6 mm of one of the wall's facets, between them. (e) the nested
# arrangement of (a) with a ball of radius 3 mm (above 1 %) or 1.5 mm (below 1 %) added to the field at
# (20, 3, 3), away from the bars; and the nested arrangement with a third solid box [40.5, 44] x [0, 20] x
# [0, 12] beside the part, 0.5 mm apart (the grid then spans it).

ROW_36_SIZE = (40.0, 20.0, 12.0)
ROW_36_VOXEL = 1.0
ROW_36_STEEL = Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
ROW_36_BARS = ((20.0, 10.0, 6.0), 3.0, 2.0, (0.4, -0.4))  # centre, radius, ramp, slopes
ROW_36_ARRANGEMENTS = {
    "nested": ((0.0, 40.0), (0.0, 4.0), (36.0, 40.0)),
    "overlapping": ((3.0, 37.0), (0.0, 5.0), (35.0, 40.0)),
    "touching": ((4.0, 36.0), (0.0, 4.0), (36.0, 40.0)),
}
ROW_36_BLOCK = ((30.0, 4.0, 0.0), (40.0, 16.0, 12.0))
ROW_36_HOLE = ((20.0, 10.0), 3.0, 32, 6.0)  # axis (x, y), radius, sides, radius of the boss
ROW_36_LUMPS = {"above": 3.0, "below": 1.5}
ROW_36_LUMP_CENTRE = (20.0, 3.0, 3.0)
ROW_36_APART = ((40.5, 0.0, 0.0), (44.0, 20.0, 12.0))
ROW_36_WALL_MM = 1e-6


def _centres(model) -> np.ndarray:
    nx, ny, nz = model.counts
    k, j, i = np.indices((nz, ny, nx)).reshape(3, -1)
    return model.to_mesh((np.stack([i, j, k], axis=1) + 0.5) * model.voxel_mm)


def bars_field(model) -> np.ndarray:
    """The field of the two crossing bars on the voxel centres of ``model``."""
    centre, radius, ramp, slopes = ROW_36_BARS
    p = _centres(model) - np.array(centre)
    field = np.zeros(len(p))
    for slope in slopes:
        axis = np.array([1.0, slope, 0.0]) / np.hypot(1.0, slope)
        distance = np.linalg.norm(p - np.outer(p @ axis, axis), axis=1)
        field = np.maximum(field, np.clip(0.5 + (radius - distance) / ramp, 0.0, 1.0))
    return field


def ball_field(model, centre, radius: float) -> np.ndarray:
    ramp = ROW_36_BARS[2]
    distance = np.linalg.norm(_centres(model) - np.array(centre), axis=1)
    return np.clip(0.5 + (radius - distance) / ramp, 0.0, 1.0)


def synthetic_part(setup: SetUp, field: np.ndarray) -> StiffPart:
    """A ``StiffPart`` made from a field built by hand: the field as the filtered design, its voxels at 0.5
    or more (and every held voxel) as the nominal design, no passes."""
    held = np.asarray(setup.keep_solid.mask) if setup.keep_solid is not None else np.zeros(len(field), bool)
    active = np.asarray(setup.model.active)
    nominal = np.where(held, 1.0, (field >= 0.5).astype(np.float64)) * active
    result = Result(x=field, density=field, passes=())
    return StiffPart(
        x=field, filtered=field, nominal=nominal, result=result, passes=(), first_step_nominal=nominal,
        member_size=None, threshold=0.5, eps=0.0, value_at_switch=0.0, scale=1.0, returned_pass=1,
    )  # fmt: skip


def _set_up(design, solids, voids=None, voxel_mm: float = ROW_36_VOXEL) -> SetUp:
    model = voxel_model(design, solids, voids, voxel_mm=voxel_mm)
    high = next(name for name in solids if name != "low")
    cases = [Case("pull", [Load(high, (0.0, 0.0, -1000.0))])]
    return set_up(model, ROW_36_STEEL, [Support("low")], cases, 5.0)


def arrangement(name: str, apart: bool = False, voxel_mm: float = ROW_36_VOXEL) -> SetUp:
    """The set-up of one arrangement of (a); with ``apart``, a third solid box beside the part (e); on voxels
    of ``voxel_mm`` (the tests use larger ones)."""
    (d0, d1), (l0, l1), (h0, h1) = ROW_36_ARRANGEMENTS[name]
    length, width, height = ROW_36_SIZE
    design = mesh.box((d0, 0.0, 0.0), (d1, width, height))
    solids = {
        "low": mesh.box((l0, 0.0, 0.0), (l1, width, height)),
        "high": mesh.box((h0, 0.0, 0.0), (h1, width, height)),
    }
    if apart:
        solids["apart"] = mesh.box(*ROW_36_APART)
    return _set_up(design, solids, voxel_mm=voxel_mm)


def interfaces_set_up(voxel_mm: float = ROW_36_VOXEL) -> tuple[SetUp, mesh.Mesh]:
    """The set-up of (b) and its void cylinder."""
    length, width, height = ROW_36_SIZE
    (cx, cy), radius, sides, _ = ROW_36_HOLE
    design = mesh.box((0.0, 0.0, 0.0), (length, width, height))
    hole = mesh.cylinder((cx, cy, height / 2.0), radius, height, axis="z", segments=sides)
    solids = {"block": mesh.box(*ROW_36_BLOCK), "low": mesh.box((0.0, 0.0, 0.0), (4.0, width, height))}
    return _set_up(design, solids, {"hole": hole}, voxel_mm), hole


def interfaces_field(setup: SetUp) -> np.ndarray:
    (cx, cy), _, _, boss = ROW_36_HOLE
    p = _centres(setup.model)
    ramp = ROW_36_BARS[2]
    around = np.clip(0.5 + (boss - np.hypot(p[:, 0] - cx, p[:, 1] - cy)) / ramp, 0.0, 1.0)
    return np.maximum(bars_field(setup.model), around)


def _quiet_finish(part, setup):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return finish(part, setup)


def _refused(build) -> tuple[MeshRefused | None, object]:
    try:
        return None, build()
    except MeshRefused as error:
        return error, None


def row_36_arrangements() -> list[Row]:
    """Line (a): the three arrangements."""
    rows = []
    for name in ROW_36_ARRANGEMENTS:
        setup = arrangement(name)
        finished = _quiet_finish(synthetic_part(setup, bars_field(setup.model)), setup)
        failed = sum(not gate.passed for gate in finished.gates)
        rows.append(checked(
            36, f"Part mesh: solid regions {name} with the design region, gates failed", DESIGN_PROPERTY,
            float(failed), 0.0,
            f"{len(finished.gates)} gates; {len(finished.mesh.triangles)} triangles, one body of genus "
            f"{finished.genus} (voxel body {finished.voxel_genus}); envelope "
            f"{g3(finished.gate('envelope').numbers['volume_mm3'])} mm3",
        ))  # fmt: skip
        rows.append(checked(
            36, f"Part mesh: solid regions {name} with the design region, relative difference of the organic "
            "volume from its target", DESIGN_PROPERTY, abs(finished.organic_difference), LEVEL_TOLERANCE,
            f"organic {g3(finished.organic_mm3)} mm3, target {g3(finished.target_mm3)} mm3 at level "
            f"{finished.level:.4f} ({len(finished.level_steps)} levels tried); delivered compliance "
            f"{100 * finished.delivered.goal_difference:+.2f} %",
        ))  # fmt: skip
    return rows


def row_36_interfaces() -> list[Row]:
    """Line (b): the exact face of a solid box and the wall of a void cylinder."""
    setup, hole = interfaces_set_up()
    finished = _quiet_finish(synthetic_part(setup, interfaces_field(setup)), setup)
    numbers = finished.gate("exact interfaces").numbers
    v, t = finished.mesh.vertices, finished.mesh.triangles
    # The vertices on the box's face x = 40: those of the triangles lying in that face's rectangle.
    (x0, y0, z0), (x1, y1, z1) = ROW_36_BLOCK
    a, b, c = (v[t[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    centroid = (a + b + c) / 3.0
    on_face = (normal[:, 0] > 1.0 - 1e-9) & (np.abs(centroid[:, 0] - x1) < 1e-3)
    on_face &= (centroid[:, 1] > y0) & (centroid[:, 1] < y1) & (centroid[:, 2] > z0) & (centroid[:, 2] < z1)
    face_vertices = np.unique(t[on_face])
    largest = float(np.abs(v[face_vertices, 0] - x1).max()) if len(face_vertices) else np.inf
    # The vertices on the cylinder's wall: within ROW_36_WALL_MM of a side facet's plane, between its ends.
    (cx, cy), radius, sides, _ = ROW_36_HOLE
    angle = np.arctan2(v[:, 1] - cy, v[:, 0] - cx)
    step = 2.0 * np.pi / sides
    middle = (np.floor(angle / step) + 0.5) * step
    apothem = radius * np.cos(step / 2.0)
    along = (v[:, 0] - cx) * np.cos(middle) + (v[:, 1] - cy) * np.sin(middle)
    near = (np.abs(along - apothem) <= ROW_36_WALL_MM) & (v[:, 2] >= -ROW_36_WALL_MM)
    near &= v[:, 2] <= ROW_36_SIZE[2] + ROW_36_WALL_MM
    own = {row.tobytes() for row in hole.vertices}
    not_own = int(sum(1 for i in np.flatnonzero(near) if v[i].tobytes() not in own))
    return [
        checked(
            36, "Part mesh: a solid box's outer face on the envelope, largest distance of the mesh's vertices on "
            "that face from its plane (mm)", DESIGN_PROPERTY, largest, INTERFACE_MM,
            f"{len(face_vertices)} vertices on the face; the gate's largest distance over every region "
            f"{g3(max(numbers[k] for k in numbers if k.endswith('(mm)')))} mm",
        ),
        checked(
            36, "Part mesh: a void cylinder (32 sides) through the part, vertices of the mesh on its wall that are "
            "not the void mesh's", DESIGN_PROPERTY, float(not_own), 0.0,
            f"{int(near.sum())} vertices on the wall, the void mesh has {len(hole.vertices)}; gate "
            f"{'passed' if finished.gate('exact interfaces').passed else 'failed'}",
        ),
    ]  # fmt: skip


def row_36_refusals() -> list[Row]:
    """Line (e): a detached lump above and below 1 %, and a solid region that cannot be joined."""
    rows = []
    for which, radius in ROW_36_LUMPS.items():
        setup = arrangement("nested")
        field = np.maximum(bars_field(setup.model), ball_field(setup.model, ROW_36_LUMP_CENTRE, radius))
        error, finished = _refused(lambda: _quiet_finish(synthetic_part(setup, field), setup))
        if which == "above":
            share = error.details.get("second_share") if error is not None else None
            rows.append(checked(
                36, "Part mesh: a result with a detached lump above 1 % of the whole is refused by the gate "
                "'one body' (0 when refused)", DESIGN_PROPERTY,
                0.0 if error is not None and error.gate == "one body" else 1.0, 0.0,
                f"lump {g3(100 * share)} % of the whole: {error.message}" if error is not None else "not refused",
            ))  # fmt: skip
        else:
            listed = finished is not None and len(finished.loose_pieces) == 1 and finished.gate("one body").passed
            note = (
                f"{len(finished.loose_pieces)} piece dropped, {g3(100 * finished.loose_pieces[0].share)} % of the whole"
                if listed else (error.message if error is not None else "not dropped")
            )  # fmt: skip
            rows.append(checked(
                36, "Part mesh: a detached lump below 1 % of the whole is dropped and listed (0 when it is)",
                DESIGN_PROPERTY, 0.0 if listed else 1.0, 0.0, note,
            ))  # fmt: skip
    setup = arrangement("nested", apart=True)
    error, _ = _refused(lambda: _quiet_finish(synthetic_part(setup, bars_field(setup.model)), setup))
    rows.append(checked(
        36, "Part mesh: a solid region apart from the design region cannot be joined and is refused by the gate "
        "'envelope' (0 when refused)", DESIGN_PROPERTY, 0.0 if error is not None and error.gate == "envelope" else 1.0,
        0.0, error.message if error is not None else "not refused",
    ))  # fmt: skip
    return rows


def row_36_part_mesh() -> list[Row]:
    """Row 36: lines (a), (b) and (e)."""
    return row_36_arrangements() + row_36_interfaces() + row_36_refusals()


# ---------------------------------------------------------------------------------------------------------
# Row 37. Limits, each on its own line, fixed before any run:
# (c) row 32's bore loaded through a pin (design block, void cylinder, held skin, no solid region), after
#     the short stiff_part run of row 32: held-solid voxel centres outside the mesh: 0;
# (d) the example bridge (examples/bridge_file/bridge.toml, which asks for part.stl and part.3mf) through
#     michell run: gates failed 0; triangles that differ between part.stl and part.3mf, the 3MF's vertices
#     rounded to float32 as STL stores them: 0; the relative difference of the compliance of the mesh
#     voxelised back from the result's at most 0.05. Reported with no limit: the share of the returned
#     design's solid voxels that change side.
#
# Set-up, fixed before the row was written: (c) the set-up and run of row 32's line (d) with the held skin;
# (d) the command `michell run bridge.toml --out <folder> --json`, its result.json and its two mesh files.

ROW_37_THREADS = 1


def row_37_bore() -> list[Row]:
    """Line (c): the held skin of the bore stays inside the mesh."""
    setup = bore()
    mass = ROW_32_FRACTION * ROW_32_STEEL.mass_g(ROW_32_BLOCK[0] * ROW_32_BLOCK[1] * ROW_32_BLOCK[2])
    part = setup.stiff_part(
        mass, schedule=ROW_32_SCHEDULE, passes_per_bound=ROW_32_PASSES_PER_BOUND,
        second_step_passes=ROW_32_SECOND_STEP_PASSES,
    )  # fmt: skip
    error, finished = _refused(lambda: _quiet_finish(part, setup))
    if error is not None:
        gate = next((g for g in error.gates if g.name == "held solid inside"), None)
        outside = gate.numbers["centres_outside"] if gate is not None else np.inf
        note = f"refused: {error.message}"
    else:
        gate = finished.gate("held solid inside")
        outside = gate.numbers["centres_outside"]
        note = (
            f"{int(gate.numbers['held_voxels'])} held voxels (the skin); level {finished.level:.4f}, "
            f"{len(finished.mesh.triangles)} triangles, delivered compliance "
            f"{100 * finished.delivered.goal_difference:+.2f} %"
        )
    return [checked(
        37, "Part mesh: row 32's bore loaded through a pin after a short stiff_part run, held-solid voxel centres "
        "outside the mesh", DESIGN_PROPERTY, float(outside), 0.0, note,
    )]  # fmt: skip


def _same_mesh(stl: Path, three_mf: Path) -> tuple[int, str]:
    """The triangles that differ between the two files: the 3MF's vertices rounded to float32, each
    triangle as its three corners from its smallest one on (the files may start a triangle at another
    corner), compared as sets."""
    a, b = read_stl(stl), read_3mf(three_mf)

    def corners(m):
        c = m.vertices.astype(np.float32)[m.triangles]  # (n, 3, 3)
        keyed = []
        for tri in c:
            rows = [tuple(r) for r in tri]
            k = rows.index(min(rows))
            keyed.append(tuple(rows[k:] + rows[:k]))
        return set(keyed), len(keyed)

    sa, na = corners(a)
    sb, nb = corners(b)
    differ = len(sa ^ sb) + abs(na - len(sa)) + abs(nb - len(sb))
    return differ, f"{na} triangles in part.stl, {nb} in part.3mf"


def row_37_bridge(examples: Path | str | None = None, threads: int = ROW_37_THREADS) -> list[Row]:
    """Line (d): michell run on the example bridge writes part.stl and part.3mf."""
    file = _example(examples)
    names = [
        "gates failed",
        "triangles that differ between part.stl and part.3mf (float32)",
        "relative difference of the compliance of the mesh voxelised back from the result's",
    ]
    limits = [0.0, 0.0, DELIVERED_TOLERANCE]
    if file is None:
        return [
            not_run(37, f"Part mesh: michell run on the example bridge, {name}", INDEPENDENT_COMPUTATION, limit,
                    "example files not found")
            for name, limit in zip(names, limits)
        ]  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        command = [sys.executable, "-m", "michell.cli", "run", str(file), "--out", folder, "--threads", str(threads),
                   "--json"]  # fmt: skip
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        out = Path(folder)
        if done.returncode not in (0, 1) or not (out / "part.stl").is_file() or not (out / "part.3mf").is_file():
            note = f"michell run exited with {done.returncode}: {(done.stdout + done.stderr).strip()[-300:]}"
            return [checked(37, f"Part mesh: michell run on the example bridge, {name}", DESIGN_PROPERTY, np.inf,
                            limit, note) for name, limit in zip(names, limits)]  # fmt: skip
        record = json.loads((out / "result.json").read_text())["mesh"]
        differ, files_note = _same_mesh(out / "part.stl", out / "part.3mf")
    gates = record["gates"]
    failed = sum(not gate["passed"] for gate in gates)
    delivered = record["delivered"]
    return [
        checked(
            37, f"Part mesh: michell run on the example bridge, {names[0]}", DESIGN_PROPERTY, float(failed), 0.0,
            f"exit status {done.returncode}; level {record['level']:.4f}, {record['triangles']} triangles; mesh "
            f"{g3(record['mass_g'])} g, mass limit {g3(record['mass_limit_g'])} g, design "
            f"{g3(record['design_mass_g'])} g, voxels {g3(record['voxel_mass_g'])} g; organic volume "
            f"{100 * record['organic_difference']:+.3f} % from its target",
        ),
        checked(37, f"Part mesh: michell run on the example bridge, {names[1]}", INDEPENDENT_COMPUTATION, float(differ),
                0.0, files_note),
        checked(
            37, f"Part mesh: michell run on the example bridge, {names[2]}", DESIGN_PROPERTY,
            abs(delivered["goal_difference"]), DELIVERED_TOLERANCE,
            f"compliance {g3(delivered['goal_delivered'])} against {g3(delivered['goal_result'])} "
            f"({100 * delivered['goal_difference']:+.2f} %); voxels changing side "
            f"{100 * delivered['changed_share']:.2f} % of {delivered['solid_voxels']} ({delivered['gained']} gained, "
            f"{delivered['lost']} lost), reported with no limit",
        ),
    ]  # fmt: skip


__all__ = [
    "ROW_36_ARRANGEMENTS",
    "arrangement",
    "ball_field",
    "bars_field",
    "interfaces_field",
    "interfaces_set_up",
    "row_36_arrangements",
    "row_36_interfaces",
    "row_36_part_mesh",
    "row_36_refusals",
    "row_37_bore",
    "row_37_bridge",
    "synthetic_part",
]
