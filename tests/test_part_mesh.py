"""The part's closed mesh (``michell.part_mesh.finish``): each step on small cases, each gate made to fail
on purpose, the refusals and the warnings."""

from __future__ import annotations

import dataclasses
import json
import math
import warnings

import manifold3d
import numpy as np
import pytest

import michell
from michell import errors, mesh, part_mesh
from michell.part_mesh import (
    DeliveredPartDiffers,
    FinishedPart,
    GenusDiffers,
    LevelOffTarget,
    MeshRefused,
    finish,
)
from michell.verification.attach_rows import bore
from michell.verification.part_mesh_rows import (
    arrangement,
    ball_field,
    bars_field,
    interfaces_field,
    interfaces_set_up,
    synthetic_part,
)

VOXEL = 2.0  # mm: the row's part on voxels twice as large, so each test takes a fraction of a second


def small(name: str = "nested"):
    setup = arrangement(name, voxel_mm=VOXEL)
    return setup, synthetic_part(setup, bars_field(setup.model))


def quiet(part, setup, *args, **kwargs) -> FinishedPart:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return finish(part, setup, *args, **kwargs)


def body(manifold_body: manifold3d.Manifold) -> manifold3d.Manifold:
    """A manifold3d body whose triangles carry a face id beyond every region's, so that the gate of exact
    interfaces does not measure them."""
    data = manifold_body.to_mesh64()
    m = mesh.Mesh(np.asarray(data.vert_properties)[:, :3], np.asarray(data.tri_verts).reshape(-1, 3))
    return part_mesh._manifold(m, np.full(len(m.triangles), 10**9))


# ---------------------------------------------------------------------------------------------------------
# The route


def test_the_field_continues_into_the_solid_regions_and_beyond():
    setup, part = small()
    source = part_mesh._source(setup, None)
    field = part_mesh._field(source, part.filtered)
    pad = part_mesh.LAYERS_BEYOND + 1
    values = field.values
    inner = values[pad:-pad, pad:-pad, pad:-pad]
    design = source.active & ~source.held
    assert np.array_equal(inner[design], part.filtered.reshape(source.active.shape)[design])
    # The outer layer is below every level, and the lattice's origin is half a voxel in from the grid's.
    for face in (values[0], values[-1], values[:, 0], values[:, -1], values[:, :, 0], values[:, :, -1]):
        assert face.max() == 0.0
    assert np.allclose(field.origin_mm, source.origin + VOXEL / 2 - pad * VOXEL)
    # Deep inside a solid region (more than LAYERS_INTO_SOLID layers from a design voxel) the field is 0.
    region = source.region_held
    assert region.any() and inner[region].min() == 0.0


def test_a_held_skin_is_one_in_the_field():
    setup = bore()
    source = part_mesh._source(setup, None)
    skin = source.held & ~source.region_held
    assert skin.any() and not source.region_held.any()
    pad = part_mesh.LAYERS_BEYOND + 1
    values = part_mesh._field(source, np.full(skin.size, 0.25)).values[pad:-pad, pad:-pad, pad:-pad]
    assert np.all(values[skin] == 1.0) and np.all(values[source.active & ~skin] == 0.25)


def test_the_finished_part_meets_its_level_and_every_gate():
    setup, part = small()
    finished = quiet(part, setup)
    assert all(gate.passed for gate in finished.gates)
    assert [gate.name for gate in finished.gates] == [
        "envelope", "closed", "no zero-area triangle", "one body", "held solid inside", "void outside",
        "exact interfaces",
    ]  # fmt: skip
    assert abs(finished.organic_difference) <= part_mesh.LEVEL_TOLERANCE
    assert finished.exact and finished.mesh.closed() is not None and len(finished.mesh.bodies()) == 1
    assert finished.volume_mm3 == pytest.approx(finished.organic_mm3 + finished.exact_mm3, rel=1e-12)
    assert finished.mass_g == pytest.approx(7.85e-3 * finished.volume_mm3)
    assert finished.voxel_mass_g == pytest.approx(7.85e-3 * finished.delivered.solid_voxels * VOXEL**3)
    assert finished.design_mass_g is None and finished.mass_limit_g is None  # built by hand: no passes
    assert finished.smoothing["flipped_triangles"] == 0 and finished.smoothing["iterations"] == 10
    assert finished.genus == finished.voxel_genus
    assert len(finished.delivered.cases) == 1 and finished.delivered.gained_inactive == 0
    summary = finished.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert summary["triangles"] == len(finished.mesh.triangles) and summary["gates"][0]["name"] == "envelope"
    with pytest.raises(errors.InvalidValue):
        finished.gate("watertight")


@pytest.mark.parametrize("name", ["nested", "overlapping", "touching"])
def test_the_three_arrangements_join(name):
    setup, part = small(name)
    finished = quiet(part, setup)
    assert all(gate.passed for gate in finished.gates)
    assert finished.gate("envelope").numbers["bodies"] == 1


def test_a_region_touching_part_of_a_face_joins():
    design = mesh.box((4.0, 0.0, 0.0), (36.0, 20.0, 12.0))
    solids = {
        "low": mesh.box((0.0, 2.0, 2.0), (4.0, 18.0, 10.0)),
        "high": mesh.box((36.0, 2.0, 2.0), (40.0, 18.0, 10.0)),
    }
    model = michell.voxel_model(design, solids, voxel_mm=VOXEL)
    setup = michell.set_up(model, michell.Material(210000.0, 0.3, 7.85), [michell.Support("low")],
                           [michell.Case("pull", [michell.Load("high", (0.0, 0.0, -1000.0))])], 5.0)  # fmt: skip
    finished = quiet(synthetic_part(setup, bars_field(model)), setup)
    assert all(gate.passed for gate in finished.gates)


def test_the_faces_of_the_regions_are_exact():
    setup, hole = interfaces_set_up(voxel_mm=VOXEL)
    finished = quiet(synthetic_part(setup, interfaces_field(setup)), setup)
    gate = finished.gate("exact interfaces")
    assert gate.passed
    assert max(value for key, value in gate.numbers.items() if key.endswith("(mm)")) <= part_mesh.INTERFACE_MM
    assert gate.numbers["void hole: vertices inside its faces not its own"] == 0
    # Every vertex on the plane x = 40 of the solid box lies on it exactly.
    v = finished.mesh.vertices
    assert np.abs(v[np.abs(v[:, 0] - 40.0) < 1e-3, 0] - 40.0).max() == 0.0
    assert finished.gate("void outside").numbers["void_voxels"] > 0


def test_without_region_meshes_the_cut_is_the_grid_box_and_exactness_is_not_claimed():
    setup, part = small()
    finished = quiet(part, setup.problem, setup.keep_solid)
    assert not finished.exact
    gate = finished.gate("exact interfaces")
    assert gate.passed and "not claimed" in gate.message
    assert finished.gate("held solid inside").passed
    with pytest.raises(errors.InvalidType):
        finish(part, setup.model)
    with pytest.raises(errors.InvalidValue):
        finish(part, setup, setup.keep_solid)


def test_the_inputs_are_checked():
    setup, part = small()
    with pytest.raises(errors.InvalidType):
        finish(part.result, setup)
    short = dataclasses.replace(part, filtered=part.filtered[:-1])
    with pytest.raises(errors.InvalidValue):
        finish(short, setup)
    with pytest.raises(errors.InvalidValue):
        finish(part, setup, threads=0)
    empty = synthetic_part(setup, np.zeros(len(part.filtered)))
    with pytest.raises(MeshRefused) as caught:
        finish(empty, setup)
    assert caught.value.gate == "level"


# ---------------------------------------------------------------------------------------------------------
# Loose pieces and the refusals


def test_a_small_lump_is_dropped_and_a_large_one_refused():
    setup = arrangement("nested", voxel_mm=1.0)
    bars = bars_field(setup.model)
    finished = quiet(synthetic_part(setup, np.maximum(bars, ball_field(setup.model, (20.0, 3.0, 3.0), 1.5))), setup)
    (piece,) = finished.loose_pieces
    assert 0 < piece.share < part_mesh.LOOSE_SHARE and finished.gate("one body").numbers["loose_pieces"] == 1
    assert np.all(np.abs(np.array(piece.low) + np.array(piece.high) - 2 * np.array([20.0, 3.0, 3.0])) < 1.0)
    with pytest.raises(MeshRefused) as caught:
        finish(synthetic_part(setup, np.maximum(bars, ball_field(setup.model, (20.0, 3.0, 3.0), 3.0))), setup)
    error = caught.value
    assert error.gate == "one body" and error.code == "mesh_refused" and error.details["second_share"] > 0.01
    assert [gate.name for gate in error.gates][-1] == "one body" and not error.gates[-1].passed
    assert error.as_dict()["gate"] == "one body" and isinstance(error, errors.EngineError)


def test_a_region_apart_cannot_be_joined():
    setup = arrangement("nested", apart=True, voxel_mm=VOXEL)
    with pytest.raises(MeshRefused) as caught:
        finish(synthetic_part(setup, bars_field(setup.model)), setup)
    assert caught.value.gate == "envelope" and caught.value.details["bodies"] == 2
    assert "overlap" in caught.value.message


def test_a_closed_cavity_is_kept_and_counted():
    outer = mesh.box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    inner = mesh.box((3.0, 3.0, 3.0), (6.0, 6.0, 6.0))
    hollow = mesh.Mesh(np.vstack([outer.vertices, inner.vertices]),
                       np.vstack([outer.triangles, inner.triangles[:, ::-1] + len(outer.vertices)]))  # fmt: skip
    chosen, kept, loose, cavities, whole, volumes = part_mesh._clean(hollow)
    assert kept == 1 and cavities == 1 and not loose and chosen.all() and whole == pytest.approx(1000.0)
    assert volumes == [pytest.approx(1000.0)]


# ---------------------------------------------------------------------------------------------------------
# Each gate made to fail: the output of the cut and join changed before the gates see it


def changed(monkeypatch, change):
    original = part_mesh._finished

    def finished(organic, exact):
        mesh_, ids, flips = original(organic, exact)
        return (*change(mesh_, ids), flips)

    monkeypatch.setattr(part_mesh, "_finished", finished)


def refused_gate(monkeypatch, change, name="nested") -> MeshRefused:
    setup, part = small(name)
    changed(monkeypatch, change)
    with pytest.raises(MeshRefused) as caught:
        quiet(part, setup)
    return caught.value


def test_an_open_mesh_is_refused(monkeypatch):
    error = refused_gate(monkeypatch, lambda m, ids: (mesh.Mesh(m.vertices, m.triangles[1:]), ids[1:]))
    assert error.gate == "closed" and error.details["open_edges"] == 3


def test_a_triangle_without_area_is_refused(monkeypatch):
    def degenerate(m, ids):
        v, t = m.vertices, m.triangles.copy()
        a, b, c = t[0]
        middle = len(v)
        t[0] = (a, middle, c)
        extra = np.array([(middle, b, c), (b, middle, a)])
        return mesh.Mesh(np.vstack([v, (v[a] + v[b]) / 2]), np.vstack([t, extra])), np.concatenate([ids, ids[:2]])

    error = refused_gate(monkeypatch, degenerate)
    assert error.gate == "no zero-area triangle" and error.details["zero_area_triangles"] == 1


def test_a_held_voxel_outside_is_refused(monkeypatch):
    setup, _ = small()
    source = part_mesh._source(setup, None)
    k, j, i = np.argwhere(source.region_held)[0]
    centre = source.origin + (np.array([i, j, k]) + 0.5) * source.h
    notch = body(manifold3d.Manifold.cube([source.h] * 3, True).translate(list(centre)))

    def cut(m, ids):
        return part_mesh._from_manifold(part_mesh._manifold(m, ids) - notch)

    error = refused_gate(monkeypatch, cut)
    assert error.gate == "held solid inside" and error.details["centres_outside"] >= 1


def test_a_void_voxel_inside_is_refused(monkeypatch):
    setup, hole = interfaces_set_up(voxel_mm=VOXEL)
    part = synthetic_part(setup, interfaces_field(setup))
    plug = body(manifold3d.Manifold.cylinder(12.0, 3.5, 3.5, 16).translate([20.0, 10.0, 0.0]))
    changed(monkeypatch, lambda m, ids: part_mesh._from_manifold(part_mesh._manifold(m, ids) + plug))
    with pytest.raises(MeshRefused) as caught:
        quiet(part, setup)
    assert caught.value.gate == "void outside" and caught.value.details["centres_inside"] >= 1


def test_a_vertex_off_a_region_face_is_refused(monkeypatch):
    organic_id = part_mesh._exact(part_mesh._source(small()[0], None)).organic_id

    def lifted(m, ids):
        v = np.array(m.vertices)
        plate = (ids >= 0) & (ids < organic_id)
        corner = m.triangles[np.flatnonzero(plate)[0], 0]
        v[corner] += 1e-4 * np.sign(v[corner] - v.mean(axis=0))
        return mesh.Mesh(v, m.triangles), ids

    error = refused_gate(monkeypatch, lifted)
    assert error.gate == "exact interfaces"
    assert max(value for key, value in error.details.items() if key.endswith("(mm)")) > part_mesh.INTERFACE_MM


def test_a_new_vertex_in_a_void_wall_is_refused(monkeypatch):
    setup, hole = interfaces_set_up(voxel_mm=VOXEL)
    part = synthetic_part(setup, interfaces_field(setup))
    exact = part_mesh._exact(part_mesh._source(setup, None))
    (wall,) = [region for region in exact.regions if region.role == "void"]

    def split(m, ids):
        inside = np.flatnonzero((ids >= wall.first) & (ids < wall.last))[0]
        a, b, c = m.triangles[inside]
        centre = len(m.vertices)
        t = np.array(m.triangles)
        t[inside] = (a, b, centre)
        extra = np.array([(b, c, centre), (c, a, centre)])
        v = np.vstack([m.vertices, m.vertices[[a, b, c]].mean(axis=0)])
        return mesh.Mesh(v, np.vstack([t, extra])), np.concatenate([ids, [ids[inside]] * 2])

    changed(monkeypatch, split)
    with pytest.raises(MeshRefused) as caught:
        quiet(part, setup)
    assert caught.value.gate == "exact interfaces"
    assert caught.value.details["void hole: vertices inside its faces not its own"] == 1


def test_a_triangle_without_area_from_the_cut_is_flipped_away():
    box = mesh.box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    v, t = box.vertices, box.triangles.copy()
    a, b, c = t[0]
    middle = len(v)
    t[0] = (a, middle, c)
    t = np.vstack([t, [(middle, b, c), (b, middle, a)]])
    broken = mesh.Mesh(np.vstack([v, (v[a] + v[b]) / 2]), t)
    assert broken.defects()["zero_area_triangles"] == 1
    ids = np.arange(len(t))
    fixed, new_ids, flips = part_mesh._without_slivers(broken, ids)
    assert flips == 1 and all(value == 0 for value in fixed.defects().values())
    assert fixed.volume_mm3 == pytest.approx(2.0, rel=1e-12) and len(new_ids) == len(fixed.triangles)
    assert np.array_equal(fixed.vertices, broken.vertices)


# ---------------------------------------------------------------------------------------------------------
# The warnings


def test_the_delivered_part_is_checked(monkeypatch):
    setup, part = small()
    monkeypatch.setattr(part_mesh, "DELIVERED_TOLERANCE", 0.0)
    with pytest.warns(DeliveredPartDiffers, match="delivered part differs"):
        finished = finish(part, setup)
    assert not finished.delivered.within and finished.delivered.tolerance == 0.0
    case = finished.delivered.cases[0]
    assert case.difference == pytest.approx(case.delivered / case.result - 1.0)
    assert finished.delivered.goal_difference == pytest.approx(case.difference)
    assert finished.delivered.changed_share == pytest.approx(
        (finished.delivered.gained + finished.delivered.lost) / finished.delivered.solid_voxels
    )


def test_a_genus_that_differs_warns(monkeypatch):
    drill = body(manifold3d.Manifold.cylinder(20.0, 1.0, 1.0, 16).translate([20.0, 10.0, -4.0]))
    setup, part = small()
    monkeypatch.setattr(part_mesh, "DELIVERED_TOLERANCE", math.inf)  # 2 mm voxels: the bars are 3 voxels thick
    changed(monkeypatch, lambda m, ids: part_mesh._from_manifold(part_mesh._manifold(m, ids) - drill))
    with pytest.warns(GenusDiffers, match="through holes"):
        finished = finish(part, setup)
    assert finished.genus == finished.voxel_genus + 1


def test_a_level_off_target_warns(monkeypatch):
    setup, part = small()
    monkeypatch.setattr(part_mesh, "LEVEL_TOLERANCE", 1e-15)
    monkeypatch.setattr(part_mesh, "LEVEL_STEPS", 4)
    monkeypatch.setattr(part_mesh, "DELIVERED_TOLERANCE", math.inf)
    with pytest.warns(LevelOffTarget, match="level search ended"):
        finished = finish(part, setup)
    assert len(finished.level_steps) <= 5 and not math.isnan(finished.level)


def test_the_names_are_exported():
    assert michell.finish is finish and michell.FinishedPart is FinishedPart
    for name in ("MeshRefused", "DeliveredPartDiffers", "GenusDiffers", "LevelOffTarget", "StlRounding"):
        assert getattr(michell, name) is getattr(part_mesh, name) and name in michell.__all__
    assert "part_mesh" in michell.__all__ and "finish" in michell.__all__
