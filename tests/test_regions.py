"""michell.regions: the voxeliser's Python face and the voxel model of a part (grid, classes, volume table,
bodies, mirror, warning and refusals)."""

import math
import warnings

import numpy as np
import pytest

import michell
from michell import mesh, select
from michell.mesh import Mesh
from michell.regions import VoxelVolumeDiffers, occupancy, voxel_model


def test_occupancy_of_a_box_union_and_difference():
    left = mesh.box((0.0, 0.0, 0.0), (2.5, 1.0, 1.0))
    right = mesh.box((1.5, 0.0, 0.0), (3.0, 1.0, 1.0))
    hole = mesh.box((0.5, 0.0, 0.0), (1.25, 1.0, 1.0))
    values = occupancy((4, 1, 1), 1.0, (0.0, 0.0, 0.0), [left, right], [hole])
    assert values.dtype == np.float64 and values.tolist() == [0.5, 0.75, 1.0, 0.0]
    assert occupancy((4, 1, 1), 1.0, (0.0, 0.0, 0.0), [left], rays_per_side=1, threads=3).tolist() == [1, 1, 0.5, 0]


def test_occupancy_does_not_depend_on_threads_or_triangle_order():
    body = mesh.torus((0.1, 0.2, 0.3), 4.0, 1.5, segments=24, sides=12)
    counts, origin = (24, 24, 10), (-6.0, -6.0, -2.5)
    one = occupancy(counts, 0.5, origin, [body])
    order = np.random.default_rng(3).permutation(len(body.triangles))
    shuffled = Mesh(body.vertices, np.roll(body.triangles[order], 1, axis=1))
    for threads in (2, 5, 12):
        assert np.array_equal(occupancy(counts, 0.5, origin, [shuffled], threads=threads), one)


def test_the_engine_refuses_open_bodies_and_wrong_arguments():
    box = mesh.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    opened = Mesh(box.vertices, box.triangles[1:])
    with pytest.raises(michell.EngineError) as error:
        occupancy((2, 2, 2), 1.0, (0.0, 0.0, 0.0), [box], [opened])
    assert error.value.code == "invalid_input" and "subtracting body 0 is not closed" in error.value.message
    for counts, h, origin in [
        ((0, 2, 2), 1.0, (0, 0, 0)),
        ((2, 2, 2), -1.0, (0, 0, 0)),
        ((2, 2, 2), 1.0, (0, math.inf, 0)),
    ]:
        with pytest.raises(michell.EngineError):
            occupancy(counts, h, origin, [box])
    with pytest.raises(michell.EngineError):
        occupancy((2, 2, 2), 1.0, (0, 0, 0), [box], rays_per_side=0)
    with pytest.raises(ValueError):
        occupancy((2, 2), 1.0, (0, 0, 0), [box])
    with pytest.raises(TypeError):
        occupancy((2, 2, 2), 1.0, (0, 0, 0), [np.zeros((3, 3))])
    # The engine's own check of the arrays.
    with pytest.raises(michell.EngineError, match="whole-number"):
        michell._engine.voxel_occupancy((2, 2, 2), 1.0, (0.0, 0.0, 0.0), [(box.vertices, box.triangles + 0.5)])
    with pytest.raises(michell.EngineError, match="names vertex"):
        michell._engine.voxel_occupancy((2, 2, 2), 1.0, (0.0, 0.0, 0.0), [(box.vertices, box.triangles * 2.0)])


def test_the_grid_covers_the_design_and_solid_regions_centred():
    design = mesh.box((1.0, 2.0, 3.0), (11.0, 6.0, 5.0))
    pad = mesh.box((10.0, 2.0, 0.5), (12.0, 4.0, 3.5))
    model = voxel_model(design, {"pad": pad}, voxel_mm=1.0)
    # The box from (1, 2, 0.5) to (12, 6, 5): sizes 11, 4, 4.5 give 11, 4, 5 voxels, centred on it.
    assert model.counts == (11, 4, 5)
    assert model.origin_mm == pytest.approx((1.0, 2.0, 0.25))
    assert isinstance(model.grid, michell.Grid3d) and (model.grid.nx, model.grid.ny, model.grid.nz) == (11, 4, 5)
    # A whole number of voxels given in decimal millimetres is not rounded up.
    assert voxel_model(mesh.box((0, 0, 0), (0.3, 0.6, 0.9)), voxel_mm=0.1).counts == (3, 6, 9)
    # Points move between the meshes' coordinates and the grid's, so michell.select works on the model.
    assert model.to_grid([1.0, 2.0, 0.25]).tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(model.to_mesh(model.to_grid([[3.0, 4.0, 5.0]])), [[3.0, 4.0, 5.0]])
    low, high = model.to_grid([10.0, 2.0, 0.5]), model.to_grid([12.0, 4.0, 3.5])
    in_pad = select.elements_in(model.grid, tuple(zip(low, high)))
    assert np.array_equal(in_pad & model.active, model.keep_solid) and int(model.keep_solid.sum()) == 12


def test_a_nested_solid_region_overrides_the_design_and_keep_solid_takes_the_mask():
    design = mesh.box((0.0, 0.0, 0.0), (10.0, 6.0, 4.0))
    nested = mesh.box((2.0, 1.0, 1.0), (5.0, 4.0, 3.0))
    model = voxel_model(design, {"core": nested}, voxel_mm=1.0)
    assert int(model.keep_solid.sum()) == 3 * 3 * 2
    assert int(model.design.sum()) == 240 - 18
    assert model.active.all()
    michell.KeepSolid(model.keep_solid, model.grid)
    assert model.volume("part").exact_mm3 is None  # a design and a solid region: no one mesh is the part
    assert model.volume("held solid").exact_mm3 == 18.0 and model.volume("held solid").voxels_mm3 == 18.0
    assert model.volume("core", "solid").sampled_vs_exact == 0.0
    assert model.bodies == 1 and model.body_sizes == ()
    assert set(model.region_occupancy) == {"design", "core"}
    with pytest.raises(ValueError):
        model.keep_solid[0] = True
    with pytest.raises(ValueError):
        model.volume("nothing")


@pytest.mark.parametrize("meet, held", [(4.5, True), (4.3, True), (4.7, False)])
def test_regions_that_meet_inside_a_voxel_lose_no_layer(meet, held):
    # The part fills the shared layer x = 4 to 5 completely; it is held solid when the solid region fills at
    # least half of it (a tie at 4.5 counts as solid).
    design = mesh.box((0.0, 0.0, 0.0), (meet, 2.0, 2.0))
    block = mesh.box((meet, 0.0, 0.0), (8.0, 2.0, 2.0))
    model = voxel_model(design, {"block": block}, voxel_mm=1.0)
    assert model.active.all()
    assert np.all(model.part_occupancy == 1.0)
    assert np.all(model.keep_solid.reshape(2, 2, 8)[:, :, 4] == held)
    assert model.region_occupancy["design"].reshape(2, 2, 8)[0, 0, 4] == pytest.approx(meet - 4.0)


def test_a_void_region_cuts_the_part_and_the_table_lists_every_region():
    design = mesh.box((0.0, 0.0, 0.0), (12.0, 8.0, 4.0))
    hole = mesh.cylinder((6.1, 4.05, 2.0), 2.5, 6.0, axis="z", segments=32)
    model = voxel_model(design, void={"hole": hole}, voxel_mm=0.5)
    roles = [(row.name, row.role) for row in model.volumes]
    assert roles == [("part", "part"), ("held solid", "held solid"), ("design", "design"), ("hole", "void")]
    part = model.volume("part")
    assert part.exact_mm3 is None
    exact = 12 * 8 * 4 - 16 * math.sin(math.pi / 16) * 2.5**2 * 4
    assert part.sampled_mm3 == pytest.approx(exact, rel=5e-3)
    assert abs(part.voxels_vs_sampled) < 0.05
    # The void region alone is clipped to the grid: 4 of its 6 mm lie inside.
    assert model.volume("hole").sampled_mm3 == pytest.approx(hole.volume_mm3 * 4 / 6, rel=1e-2)
    assert not model.keep_solid.any()


def test_the_bodies_of_the_active_set_are_counted_through_faces():
    design = mesh.box((0.0, 0.0, 0.0), (9.0, 3.0, 3.0))
    wall = mesh.box((3.0, -1.0, -1.0), (4.0, 4.0, 4.0))
    model = voxel_model(design, void={"wall": wall}, voxel_mm=1.0)
    assert model.bodies == 2 and model.body_sizes == (45, 27)
    # Two blocks that touch only along an edge are two bodies.
    cut = {"a": mesh.box((-1.0, 2.0, -1.0), (2.0, 5.0, 3.0)), "b": mesh.box((2.0, -1.0, -1.0), (5.0, 2.0, 3.0))}
    model = voxel_model(mesh.box((0.0, 0.0, 0.0), (4.0, 4.0, 2.0)), void=cut, voxel_mm=1.0)
    assert int(model.active.sum()) == 16
    assert model.bodies == 2 and model.body_sizes == (8, 8)


def test_a_coarse_voxel_size_warns_and_the_model_is_returned():
    design = mesh.box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    # The void takes 2.6 of the first 5 mm voxel layer: the part keeps 0.48 of it, which is not active.
    cut = {"cut": mesh.box((-1.0, -1.0, -1.0), (2.6, 11.0, 11.0))}
    with pytest.warns(VoxelVolumeDiffers, match="more than 5 %"):
        model = voxel_model(design, void=cut, voxel_mm=5.0)
    assert int(model.active.sum()) == 4
    assert model.volume("part").voxels_vs_sampled == pytest.approx(500 / 740 - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        voxel_model(design, void=cut, voxel_mm=0.5)


def test_a_region_too_thin_for_the_voxels_is_refused_by_name():
    design = mesh.box((0.0, 0.0, 0.0), (20.0, 10.0, 10.0))
    plate = mesh.box((5.0, 2.0, 4.3), (15.0, 8.0, 4.7))
    with pytest.raises(ValueError) as error:
        voxel_model(design, {"plate": plate}, voxel_mm=1.0)
    assert "'plate'" in error.value.message and "0.4 mm" in error.value.message
    assert error.value.details == {"smallest_extent_mm": pytest.approx(0.4), "voxel_mm": 1.0}
    slot = mesh.box((5.0, -1.0, 4.3), (15.0, 11.0, 4.7))
    with pytest.raises(ValueError, match="void region 'slot'"):
        voxel_model(design, void={"slot": slot}, voxel_mm=1.0)
    with pytest.raises(ValueError, match="fills no voxel"):
        voxel_model(design, void={"all": mesh.box((-1, -1, -1), (21, 11, 11))}, voxel_mm=1.0)


def test_wrong_arguments_are_refused():
    design = mesh.box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    pad = mesh.box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="its own name"):
        voxel_model(design, {"a": pad}, {"a": pad}, voxel_mm=1.0)
    with pytest.raises(ValueError, match="its own name"):
        voxel_model(design, {"design": pad}, voxel_mm=1.0)
    with pytest.raises(TypeError):
        voxel_model(np.zeros((3, 3)), voxel_mm=1.0)
    with pytest.raises(TypeError):
        voxel_model(design, [pad], voxel_mm=1.0)
    for bad in (0.0, -1.0, math.nan, True, "1"):
        with pytest.raises(ValueError):
            voxel_model(design, voxel_mm=bad)
    for bad in ("y", ("w",), ("y", "y"), "xy"):
        with pytest.raises(ValueError):
            voxel_model(design, voxel_mm=1.0, mirror=bad)
    with pytest.raises(ValueError):
        voxel_model(design, voxel_mm=1.0, threads=0)
    opened = Mesh(pad.vertices, pad.triangles[1:])
    with pytest.raises(ValueError, match="the solid region 'pad': the mesh is not closed"):
        voxel_model(design, {"pad": opened}, voxel_mm=1.0)
    inside_out = Mesh(pad.vertices, pad.triangles[:, ::-1])
    assert voxel_model(design, {"pad": inside_out}, voxel_mm=1.0).keep_solid.sum() == 8


def mirrored_part(hole_y, pads=True, mirror=("y",), hole_x=17.2, threads=1):
    design = mesh.box((0.0, 0.0, 0.0), (30.0, 21.0, 8.0))
    solid = {"pad_a": mesh.box((2.3, 1.4, 0.0), (6.7, 5.2, 3.0)), "pad_b": mesh.box((2.3, 15.8, 0.0), (6.7, 19.6, 3.0))}
    hole = mesh.cylinder((hole_x, hole_y, 4.0), 4.0, 12.0, axis="z", segments=32)
    return voxel_model(design, solid if pads else None, {"hole": hole}, voxel_mm=1.0, mirror=mirror, threads=threads)


def test_mirror_makes_the_classes_mirror_exactly_and_pairs_regions_that_mirror_each_other():
    model = mirrored_part(10.5)
    nx, ny, nz = model.counts
    classes = (model.active.astype(int) + model.keep_solid).reshape(nz, ny, nx)
    assert np.array_equal(classes, classes[:, ::-1, :])
    a = model.region_occupancy["pad_a"].reshape(nz, ny, nx)
    b = model.region_occupancy["pad_b"].reshape(nz, ny, nx)
    assert np.array_equal(a, b[:, ::-1, :])
    part = model.part_occupancy.reshape(nz, ny, nx)
    assert np.array_equal(part, part[:, ::-1, :])
    assert model.mirror == ("y",)
    # Mirrored about x and y together.
    both = mirrored_part(10.5, pads=False, mirror=("x", "y"), hole_x=15.0)
    classes = (both.active.astype(int) + both.keep_solid).reshape(both.counts[::-1])
    assert np.array_equal(classes, classes[:, :, ::-1]) and np.array_equal(classes, classes[:, ::-1, :])


def test_mirror_refuses_a_region_off_the_plane_by_name():
    with pytest.raises(ValueError) as error:
        mirrored_part(11.5)
    assert "void region 'hole'" in error.value.message and "plane y = 10.5 mm" in error.value.message
    assert error.value.details["differs_mm3"] > 0.005 * error.value.details["sampled_mm3"]
    # A single pad has no partner.
    design = mesh.box((0.0, 0.0, 0.0), (30.0, 21.0, 8.0))
    with pytest.raises(ValueError, match="solid region 'pad'"):
        voxel_model(design, {"pad": mesh.box((2.3, 1.4, 0.0), (6.7, 5.2, 3.0))}, voxel_mm=1.0, mirror=("y",))


def test_the_model_does_not_depend_on_the_thread_count():
    one, five = mirrored_part(10.5), mirrored_part(10.5, threads=5)
    assert np.array_equal(one.part_occupancy, five.part_occupancy)
    assert np.array_equal(one.keep_solid, five.keep_solid) and five.threads == 5
