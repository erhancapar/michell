"""Supports and loads on the voxel model (`michell.attach`): the faces in contact with a body in the cases that
decide the rule, the nodal shares, the held skin, the filter faces, the refusals and the warning."""

import math

import numpy as np
import pytest

import michell
from michell import attach, mesh, select
from michell.attach import Case, Load, LoadOnSupport, Support, set_up
from michell.errors import InvalidType, InvalidValue, UnknownName
from michell.regions import voxel_model

STEEL = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)


def by_side(faces):
    """The number of faces on each side, by the side's name."""
    return {attach.FACES[s]: int(np.count_nonzero(faces.sides == s)) for s in range(6) if np.any(faces.sides == s)}


def bar(length=40.0):
    return voxel_model(mesh.box((0.0, 0.0, 0.0), (length, 10.0, 10.0)), voxel_mm=1.0)


def bar_ending(end):
    """A bar of 10 x 10 mm from x = 0 to x = ``end`` on 1 mm voxels whose end face lies at the given point of
    the last voxel layer: on a grid plane (40), half-way (40.5) or a quarter (40.25) through the layer from
    40 to 41 mm (a box of 41 mm cut by a void box from ``end`` on; the grid covers the design box), or at
    three quarters (40.75: a box of 40.5 mm, on its grid of 41 voxels centred on it, from -0.25 to 40.75 mm;
    the end then lies at 40.5 mm in the meshes' coordinates)."""
    if end == 40.75:
        return voxel_model(mesh.box((0.0, 0.0, 0.0), (40.5, 10.0, 10.0)), voxel_mm=1.0)
    design = mesh.box((0.0, 0.0, 0.0), (math.ceil(end), 10.0, 10.0))
    if end == math.ceil(end):
        return voxel_model(design, voxel_mm=1.0)
    cut = mesh.box((end, -1.0, -1.0), (math.ceil(end) + 1.0, 11.0, 11.0))
    return voxel_model(design, None, {"cut": cut}, voxel_mm=1.0)


def bored_block():
    block = mesh.box((0.0, 0.0, 0.0), (40.0, 20.0, 10.0))
    hole = mesh.cylinder((30.0, 10.0, 5.0), 3.0, 14.0, axis="z", segments=32)
    return voxel_model(block, None, {"bore": hole}, voxel_mm=1.0)


def l_marker():
    """A marker of L-shaped section, 1 mm thick, wrapped around the edge x = 40, y = 0 of ``bar()`` from
    outside: it touches the end face for y from 0 to 3 mm and the face y = 0 for x from 37 to 40 mm."""
    ring = [(37.0, -1.0), (41.0, -1.0), (41.0, 3.0), (40.0, 3.0), (40.0, 0.0), (37.0, 0.0)]
    vertices = [[x, y, z] for z in (0.0, 10.0) for x, y in ring]
    triangles = [[4, 0, 5], [4, 1, 0], [4, 2, 1], [4, 3, 2], [10, 11, 6], [10, 6, 7], [10, 7, 8], [10, 8, 9]]
    for a in range(6):
        b = (a + 1) % 6
        triangles += [[a, b, b + 6], [a, b + 6, a + 6]]
    return mesh.Mesh(vertices, triangles)


# ---------------------------------------------------------------------------------------------------------
# The cases of the contact rule


@pytest.mark.parametrize("end", [40.0, 40.5, 40.25, 40.75])
@pytest.mark.parametrize("depth", [1.0, 0.25])
@pytest.mark.parametrize("overhang", [0.0, 3.0])
def test_a_slab_catches_the_end_faces_only_wherever_the_face_lies_in_its_voxel_layer(end, depth, overhang):
    # The end face on a grid plane, half-way, a quarter and three quarters through the last voxel layer; a
    # slab 1 mm thick or a marker a quarter of a voxel thick (lying inside the layer when the face is inside
    # it); flush with the face's sides or overhanging them by three voxels on every side.
    model = bar_ending(end)
    face = 40.5 if end == 40.75 else end
    contact = attach.contact_faces(model, ((face, -overhang, -overhang), (face + depth, 10 + overhang, 10 + overhang)))
    assert by_side(contact.faces) == {"x+": 100}
    last = np.flatnonzero(model.active.reshape(10, 10, model.counts[0]).any(axis=(0, 1)))[-1]
    assert set(np.unique(contact.faces.voxels % model.counts[0])) == {last}


@pytest.mark.parametrize("end, face, covered", [(40.75, 40.5, 4), (40.5, 40.5, 8)])
def test_the_side_faces_of_the_last_layer_see_at_most_half_of_an_overhanging_slab(end, face, covered):
    # Filled to three quarters, the side faces see a quarter of the slab; in the tie at one half, exactly
    # half, which "more than half" leaves out.
    contact = attach.contact_faces(bar_ending(end), ((face, -3.0, -3.0), (face + 1.0, 13.0, 13.0)))
    sides = contact.rays[contact.boundary.sides >= 2]
    last_layer = contact.boundary.voxels[contact.boundary.sides >= 2] % 41 == 40
    assert set(sides[last_layer]) == {covered} and set(sides[~last_layer]) == {0}


@pytest.mark.parametrize("change", [-0.01, 0.01])
def test_a_pin_in_a_bore_with_a_clearance_or_an_interference_catches_the_whole_bore(change):
    model = bored_block()
    boundary = attach.boundary_faces(model)
    outer = np.zeros(len(boundary), dtype=bool)
    for face in attach.FACES:
        outer |= boundary.on_grid_face(face)
    pin = mesh.cylinder((30.0, 10.0, 5.0), 3.0 + change, 10.0, axis="z", segments=32)
    contact = attach.contact_faces(model, pin)
    assert np.array_equal(contact.faces.keys(), boundary.keys()[~outer])  # every face of the bore, nothing else
    assert len(contact.faces) == 240 and contact.rays[~outer].min() == 12  # 12 or 16 of 16 rays
    assert contact.rays[outer].max() <= 1  # the rims of the bore's ends, with the interference


def test_a_pin_of_half_the_bore_s_diameter_is_refused():
    pin = mesh.cylinder((30.0, 10.0, 5.0), 1.5, 10.0, axis="z", segments=32)
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 21.0, 11.0)))]
    assert len(attach.contact_faces(bored_block(), pin).faces) == 0
    with pytest.raises(InvalidValue, match="touches no boundary face"):
        set_up(bored_block(), STEEL, supports, [Case("pin", [Load(pin, (0.0, -1.0, 0.0))])], 5.0)


def test_a_body_that_wraps_around_an_edge_catches_both_faces():
    assert by_side(attach.contact_faces(bar(), l_marker()).faces) == {"x+": 30, "y-": 30}


def test_a_box_that_overlaps_the_part_deeply_also_covers_the_faces_it_passes_inside():
    # Seen along z, a box through the part's whole depth covers the windows of the top and bottom faces it
    # lies under; marker bodies are meant to touch the part or overlap it slightly.
    contact = attach.contact_faces(bar(), ((37.0, -2.0, 0.0), (42.0, 2.0, 10.0)))
    assert by_side(contact.faces) == {"x+": 20, "y-": 30, "z-": 6, "z+": 6}


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_slab_a_twentieth_of_a_voxel_thick_is_seen_and_one_a_twenty_fifth_thick_is_not(axis):
    model = bar()
    size = (40.0, 10.0, 10.0)

    def slab(thickness):
        low, high = [0.0, 0.0, 0.0], list(size)
        low[axis], high[axis] = size[axis], size[axis] + thickness
        return tuple(low), tuple(high)

    faces = {0: 100, 1: 400, 2: 400}[axis]
    assert by_side(attach.contact_faces(model, slab(0.05)).faces) == {attach.FACES[2 * axis + 1]: faces}
    assert len(attach.contact_faces(model, slab(0.04)).faces) == 0


# ---------------------------------------------------------------------------------------------------------
# The pieces on their own


def test_the_boundary_faces_of_a_box():
    faces = attach.boundary_faces(bar())
    assert len(faces) == 2 * (40 * 10 + 10 * 10 + 40 * 10)
    assert faces.area_mm2 == len(faces)
    assert np.all(np.diff(faces.keys()) > 0)


def test_a_traction_on_a_patch_of_a_grid_face_equals_select_load():
    model = bar()
    contact = attach.contact_faces(model, ((40.0, 2.0, 3.0), (41.0, 7.0, 9.0)))
    loads = attach.traction_loads(model.grid, contact.faces, (300.0, -20.0, 7.0))
    nodes = select.face_nodes(model.grid, "x+", within=((40.0, 40.0), (2.0, 7.0), (3.0, 9.0)))
    assert np.array_equal(loads, select.load(model.grid, nodes, (300.0, -20.0, 7.0)))


def test_a_body_force_gives_each_voxel_s_nodes_an_eighth():
    model = voxel_model(mesh.box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0)), voxel_mm=1.0)
    loads = attach.body_force_loads(model.grid, model.active, (0.0, 0.0, -27.0))
    assert math.fsum(loads[2::3]) == -27.0 and not loads[0::3].any() and not loads[1::3].any()
    centre = model.grid.node(1, 1, 1)
    corner = model.grid.node(0, 0, 0)
    assert loads[3 * centre + 2] == pytest.approx(-1.0, rel=1e-15)  # eight voxels, an eighth of 1 N each
    assert loads[3 * corner + 2] == pytest.approx(-1.0 / 8.0, rel=1e-15)


def test_the_held_skin_reaches_half_the_member_size_and_at_least_two_voxels():
    # Voxel centres lie 0.5, 1.5, 2.5, ... mm behind the end face; a centre at exactly the depth is inside.
    model = bar()
    end = attach.contact_faces(model, ((40.0, 0.0, 0.0), (41.0, 10.0, 10.0))).faces
    for size, layers in ((2.0, 2), (5.0, 3), (6.0, 3), (7.0, 4)):
        skin = attach.held_skin(model, end, size).reshape(10, 10, 40)
        assert np.flatnonzero(skin.any(axis=(0, 1))).tolist() == list(range(40 - layers, 40))
        assert int(skin.sum()) == 100 * layers
        assert attach.skin_depth_mm(size, 1.0) == max(size / 2, 2.0)


def test_the_held_skin_against_every_distance_measured_directly():
    model = bored_block()
    pin = mesh.cylinder((30.0, 10.0, 5.0), 2.99, 10.0, axis="z", segments=32)
    faces = attach.contact_faces(model, pin).faces
    nx, ny, nz = model.counts
    k, j, i = np.indices((nz, ny, nx)).reshape(3, -1)
    centres = (np.stack([i, j, k], axis=1) + 0.5) * model.voxel_mm
    distance = np.linalg.norm(centres[:, None, :] - faces.centres_mm()[None, :, :], axis=2).min(axis=1)
    expected = (distance <= 2.5 + 1e-9) & model.active
    assert np.array_equal(attach.held_skin(model, faces, 5.0), expected)


def test_the_voxels_of_a_solid_region():
    design = mesh.box((0.0, 0.0, 0.0), (10.0, 4.0, 4.0))
    model = voxel_model(design, {"pad": mesh.box((0.0, 0.0, 0.0), (2.0, 4.0, 4.0))}, voxel_mm=1.0)
    assert int(attach.region_voxels(model, "pad").sum()) == 32
    with pytest.raises(UnknownName, match="pad"):
        attach.region_voxels(model, "pads")
    with pytest.raises(InvalidValue, match="design region"):
        attach.region_voxels(model, "design")


def test_mirror_images_of_bodies_give_mirror_images_of_contact_faces():
    # Nothing in the contact faces stands in the way of comparing a body with the mirror image of another.
    model = voxel_model(mesh.box((0.0, 0.0, 0.0), (20.0, 10.0, 6.0)), voxel_mm=1.0, mirror=("y",))
    low = attach.contact_faces(model, ((3.0, -1.0, 6.0), (7.0, 3.5, 7.0))).faces
    high = attach.contact_faces(model, ((3.0, 6.5, 6.0), (7.0, 11.0, 7.0))).faces
    nx, ny, _ = model.counts
    i, j, k = low.voxels % nx, (low.voxels // nx) % ny, low.voxels // (nx * ny)
    mirrored = np.sort(6 * (i + nx * ((ny - 1 - j) + ny * k)) + low.sides)  # z+ faces keep their side
    assert np.array_equal(mirrored, high.keys())


# ---------------------------------------------------------------------------------------------------------
# set_up


def pulled_bar(**load):
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    cases = [Case("pull", [Load(((40.0, -1.0, -1.0), (41.0, 11.0, 11.0)), (1000.0, 0.0, 0.0), **load)])]
    return set_up(bar(), STEEL, supports, cases, 5.0)


def test_set_up_reports_each_support_and_load():
    attached = pulled_bar()
    support, load = attached.report
    assert (support.role, support.rule, len(support.faces), support.nodes.size) == ("support", "contact", 100, 121)
    assert support.closed_faces == ("x-",) and support.skin.size == 0
    assert (load.role, load.case, load.rule, load.area_mm2) == ("load", "pull", "contact", 100.0)
    assert load.point_mm == pytest.approx((40.0, 5.0, 5.0), abs=1e-12)
    assert load.dropped_share == 0.0 and load.closed_faces == ()
    # The load's end face is free, so the skin holds the voxels whose centres lie within 2.5 mm of it (half
    # the member size of 5 mm): three layers.
    assert load.skin.size == 300 and load.skin_mass_g == pytest.approx(STEEL.mass_g(300.0))
    assert int(attached.keep_solid.mask.sum()) == 300
    assert attached.outside == {"x-": "closed", "x+": "empty", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}
    assert attached.inactive == "empty"
    assert math.fsum(attached.problem.cases[0].loads[0::3]) == 1000.0


def test_a_face_held_in_part_or_by_two_supports_stays_free():
    halves = [Support(((-1.0, -1.0, -1.0), (0.0, 5.0, 11.0))), Support(((-1.0, 5.0, -1.0), (0.0, 11.0, 11.0)))]
    cases = [Case("pull", [Load(((40.0, -1.0, -1.0), (41.0, 11.0, 11.0)), (1000.0, 0.0, 0.0))])]
    attached = set_up(bar(), STEEL, halves, cases, 5.0)
    assert attached.outside["x-"] == "empty"
    first, second = attached.supports
    assert first.skin.size == second.skin.size > 150  # three layers behind each half and a little beyond
    assert np.isin(first.skin, second.skin).any()


def test_the_caller_overrides_a_filter_face_and_the_skin_follows_it():
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    cases = [Case("pull", [Load(((40.0, -1.0, -1.0), (41.0, 11.0, 11.0)), (1000.0, 0.0, 0.0))])]
    attached = set_up(bar(), STEEL, supports, cases, 5.0, outside={"x+": "closed", "x-": "empty"})
    assert attached.outside["x+"] == "closed" and attached.outside["x-"] == "empty"
    assert attached.loads[0].skin.size == 0 and attached.supports[0].skin.size == 300


def test_no_skin_on_voxels_already_held_solid():
    design = mesh.box((0.0, 0.0, 0.0), (40.0, 10.0, 10.0))
    model = voxel_model(design, {"end": mesh.box((37.0, 0.0, 0.0), (40.0, 10.0, 10.0))}, voxel_mm=1.0)
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    cases = [Case("pull", [Load(((40.0, -1.0, -1.0), (41.0, 11.0, 11.0)), (1000.0, 0.0, 0.0), skin=False)])]
    attached = set_up(model, STEEL, supports, cases, 5.0)
    assert attached.loads[0].skin.size == 0
    assert np.array_equal(attached.keep_solid.mask, model.keep_solid)


def test_a_region_load_and_a_support_on_a_region():
    design = mesh.box((0.0, 0.0, 0.0), (20.0, 6.0, 6.0))
    solid = {"root": mesh.box((0.0, 0.0, 0.0), (2.0, 6.0, 6.0)), "tip": mesh.box((18.0, 0.0, 0.0), (20.0, 6.0, 6.0))}
    model = voxel_model(design, solid, voxel_mm=1.0)
    attached = set_up(model, STEEL, [Support("root")], [Case("down", [Load("tip", (0.0, 0.0, -50.0))])], 5.0)
    support, load = attached.report
    assert (support.rule, support.voxels.size, support.nodes.size) == ("region", 72, 3 * 7 * 7)
    assert (load.rule, load.volume_mm3, load.skin.size) == ("region", 72.0, 0)
    assert load.point_mm == pytest.approx((19.0, 3.0, 3.0), abs=1e-12)
    assert attached.outside["x-"] == "empty"  # a region support closes no face
    loads = attached.problem.cases[0].loads
    assert math.fsum(loads[2::3]) == -50.0 and not loads[0::3].any()


# ---------------------------------------------------------------------------------------------------------
# Refusals and the warning


def test_a_body_that_touches_no_face_is_refused():
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    inside = [Case("c", [Load(((10.0, 2.0, 2.0), (20.0, 8.0, 8.0)), (1.0, 0.0, 0.0))])]
    with pytest.raises(InvalidValue, match="touches no boundary face.*solid region") as refused:
        set_up(bar(), STEEL, supports, inside, 5.0)
    assert refused.value.details == {"most_rays": 0, "volume_mm3": pytest.approx(360.0)}  # inside the part
    away = [Case("c", [Load(((50.0, 0.0, 0.0), (51.0, 10.0, 10.0)), (1.0, 0.0, 0.0))])]
    with pytest.raises(InvalidValue, match="touches no boundary face"):
        set_up(bar(), STEEL, supports, away, 5.0)


def test_a_contact_on_design_voxels_without_a_skin_is_refused():
    with pytest.raises(InvalidValue, match="skin=False"):
        pulled_bar(skin=False)


def test_a_load_mostly_on_held_degrees_of_freedom_is_refused():
    slab = ((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0))
    with pytest.raises(InvalidValue, match="falls on degrees of freedom that a support holds"):
        set_up(bar(), STEEL, [Support(slab)], [Case("c", [Load(slab, (10.0, 0.0, 0.0))])], 5.0)


def test_a_load_partly_on_held_degrees_of_freedom_warns():
    model = voxel_model(mesh.box((0.0, 0.0, 0.0), (40.0, 10.0, 10.0)), {"bar": mesh.box((0.0, 0.0, 0.0), (40.0, 10.0, 10.0))},
                        voxel_mm=1.0)  # fmt: skip
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    with pytest.warns(LoadOnSupport, match="1.25 %"):
        attached = set_up(model, STEEL, supports, [Case("c", [Load("bar", (1000.0, 0.0, 0.0))])], 5.0)
    load = attached.loads[0]
    assert load.dropped_share == pytest.approx(1.0 / 80.0, rel=1e-12)
    assert math.fsum(attached.problem.cases[0].loads[0::3]) + load.dropped_n[0] == pytest.approx(1000.0, rel=1e-15)


@pytest.mark.parametrize(
    "build, error, match",
    [
        (lambda: Support(((0, 0, 0), (1, 1, 1)), fix=("w",)), InvalidValue, "fix"),
        (lambda: Support(((0, 0, 0), (1, 1, 1)), fix=()), InvalidValue, "fix"),
        (lambda: Support(((0, 0, 0), (1, 1, 0)),), InvalidValue, "above"),
        (lambda: Support(3.0), InvalidType, "solid region"),
        (lambda: Support(((0, 0, 0), (1, 1, 1)), skin="yes"), InvalidType, "skin"),
        (lambda: Load("pad", (1.0, 2.0)), InvalidValue, "three"),
        (lambda: Load("pad", (1.0, math.nan, 0.0)), InvalidValue, "finite"),
        (lambda: Load("", (1.0, 0.0, 0.0)), InvalidValue, "non-empty"),
        (lambda: Case("c", []), InvalidValue, "non-empty"),
        (lambda: Case("c", [Load("pad", (1.0, 0.0, 0.0))], weight=-1.0), InvalidValue, "weight"),
    ],
)
def test_bad_supports_loads_and_cases_are_refused(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_bad_arguments_of_set_up_are_refused():
    model = bar()
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 11.0, 11.0)))]
    cases = [Case("pull", [Load(((40.0, -1.0, -1.0), (41.0, 11.0, 11.0)), (1000.0, 0.0, 0.0))])]
    with pytest.raises(InvalidValue, match="faces of the grid"):
        set_up(model, STEEL, supports, cases, 5.0, outside={"w+": "closed"})
    with pytest.raises(InvalidValue, match="'closed' or 'empty'"):
        set_up(model, STEEL, supports, cases, 5.0, outside={"x+": "open"})
    with pytest.raises(InvalidValue, match="member size"):
        set_up(model, STEEL, supports, cases, 0.0)
    with pytest.raises(InvalidType, match="VoxelModel"):
        set_up("model", STEEL, supports, cases, 5.0)
    with pytest.raises(UnknownName, match="no region 'tip'"):
        set_up(model, STEEL, supports, [Case("c", [Load("tip", (1.0, 0.0, 0.0))])], 5.0)


@pytest.mark.filterwarnings("ignore::michell.LimitNotMet")  # two passes per step do not meet the limits
def test_the_set_up_runs_stiff_part_in_one_call():
    model = voxel_model(mesh.box((0.0, 0.0, 0.0), (12.0, 6.0, 4.0)), voxel_mm=1.0)
    supports = [Support(((-1.0, -1.0, -1.0), (0.0, 7.0, 5.0)))]
    cases = [Case("tip", [Load(((12.0, 2.0, -1.0), (13.0, 4.0, 5.0)), (0.0, 0.0, -100.0))])]
    attached = set_up(model, STEEL, supports, cases, 2.0)
    with pytest.warns(michell.GridTooCoarse):  # a member size of 2 voxels keeps the run small
        part = attached.stiff_part(
            STEEL.mass_g(0.5 * 12 * 6 * 4), schedule=((1.0, 2), (32.0, 2)), passes_per_bound=2, second_step_passes=2
        )
    assert np.all(part.nominal[attached.keep_solid.mask] == 1.0)
