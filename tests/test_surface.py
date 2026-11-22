"""michell.surface: the field on its lattice, the iso-surface and its level rule, refining a field, and Taubin
smoothing."""

import math

import numpy as np
import pytest

import michell
from michell.errors import InvalidType, InvalidValue
from michell.mesh import Mesh, close_pairs
from michell.surface import ScalarField, iso_surface, refine, smooth
from michell.verification.surface_rows import lattice_field, sphere_field, torus_field

END_GAP = 1e-3  # the engine's END_GAP


def random_field(shape, values, seed):
    """A field of the given values at random, 0 on the outer layer."""
    rng = np.random.default_rng(seed)
    field = rng.choice(np.asarray(values, dtype=np.float64), size=shape)
    field[[0, -1], :, :] = field[:, [0, -1], :] = field[:, :, [0, -1]] = 0.0
    return field


def check_surface(mesh: Mesh, spacing: float) -> None:
    """Closed, facing outward as it comes from the engine, no two vertices within END_GAP spacing / 4."""
    assert mesh.volume_mm3 > 0.0  # before closed(), which would turn an inward mesh the right way out
    assert mesh.defects() == {"open_edges": 0, "edges_on_more_than_two": 0, "edges_same_direction": 0,
                              "zero_area_triangles": 0}  # fmt: skip
    assert mesh.closed() is mesh
    assert len(close_pairs(mesh.vertices, END_GAP * spacing / 4)[0]) == 0


def test_a_field_keeps_its_values_read_only_with_x_fastest():
    values = np.arange(24.0).reshape(2, 3, 4)
    field = ScalarField(values, 0.5, (1.0, 2.0, 3.0))
    assert field.counts == (4, 3, 2)
    x, y, z = field.coordinates()
    assert x.tolist() == [1.0, 1.5, 2.0, 2.5] and y.tolist() == [2.0, 2.5, 3.0] and z.tolist() == [3.0, 3.5]
    values[0, 0, 0] = 99.0
    assert field.values[0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        field.values[0, 0, 0] = 1.0
    assert "4 x 3 x 2 points" in repr(field)


@pytest.mark.parametrize(
    "values, spacing, origin, error",
    [
        (np.zeros((3, 3)), 1.0, (0, 0, 0), InvalidValue),
        (np.zeros((3, 0, 3)), 1.0, (0, 0, 0), InvalidValue),
        (np.zeros((3, 3, 3), dtype=bool), 1.0, (0, 0, 0), InvalidType),
        (np.full((3, 3, 3), np.nan), 1.0, (0, 0, 0), InvalidValue),
        (np.zeros((3, 3, 3)), 0.0, (0, 0, 0), InvalidValue),
        (np.zeros((3, 3, 3)), math.inf, (0, 0, 0), InvalidValue),
        (np.zeros((3, 3, 3)), 1.0, (0, 0), InvalidValue),
        (np.zeros((3, 3, 3)), 1.0, (0, math.nan, 0), InvalidValue),
    ],
    ids=["2-D", "empty axis", "booleans", "not finite", "zero spacing", "infinite spacing", "two coordinates",
         "origin not finite"],  # fmt: skip
)
def test_a_field_is_refused_with_wrong_values_or_lattice(values, spacing, origin, error):
    with pytest.raises(error):
        ScalarField(values, spacing, origin)


def test_a_single_sample_above_the_level_gives_a_closed_body_around_it():
    values = np.zeros((3, 3, 3))
    values[1, 1, 1] = 1.0
    mesh = iso_surface(ScalarField(values, 2.0, (10.0, 0.0, 0.0)), 0.25)
    # 14 edges from the centre (6 along the axes, 6 face diagonals, 2 body diagonals), one triangle in each of
    # the 24 tetrahedra around it; each vertex three quarters of its edge from the centre.
    assert len(mesh.vertices) == 14 and len(mesh.triangles) == 24
    check_surface(mesh, 2.0)
    assert mesh.genus() == 0
    steps = np.abs(mesh.vertices - [12.0, 2.0, 2.0]).max(axis=1) / 2.0
    np.testing.assert_allclose(steps, 0.75, rtol=0, atol=1e-12)


def test_a_sample_exactly_at_the_level_is_inside_and_the_vertices_keep_the_end_gap():
    values = np.zeros((3, 3, 3))
    values[1, 1, 1] = 0.5
    mesh = iso_surface(ScalarField(values, 1.0), 0.5)
    check_surface(mesh, 1.0)
    np.testing.assert_allclose(np.abs(mesh.vertices - 1.0).max(axis=1), END_GAP, rtol=1e-12)
    # Just below the level the sample is outside, and there is no surface.
    values[1, 1, 1] = np.nextafter(0.5, 0.0)
    with pytest.raises(michell.EngineError, match="no sample of the field reaches the level"):
        iso_surface(ScalarField(values, 1.0), 0.5)


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.5, 1.0),
        (0.0, 0.5),
        (0.0, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0)),
        (0.0, 0.25, 0.5, 0.75, 1.0),
    ],
    ids=["0, level, 1", "0, level", "one unit in the last place either side", "quarters"],
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_fields_with_many_samples_at_the_level_give_closed_surfaces(values, seed):
    mesh = iso_surface(ScalarField(random_field((9, 11, 10), values, seed), 0.5), 0.5)
    check_surface(mesh, 0.5)
    # Every body of the surface is closed on its own, with a genus.
    for body in mesh.bodies():
        assert body.genus() >= 0


@pytest.mark.parametrize("value", [1.0, 0.5], ids=["at 1", "at the level"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_sheet_one_cell_thick_gives_a_closed_slab(value, axis):
    values = np.zeros((12, 12, 12))
    sheet = [slice(3, 9)] * 3
    sheet[axis] = 6
    values[tuple(sheet)] = value
    mesh = iso_surface(ScalarField(values, 0.5), 0.5)
    check_surface(mesh, 0.5)
    assert len(mesh.bodies()) == 1 and mesh.genus() == 0
    # Half a cell either side at 1, a thousandth of an edge either side at the level.
    thickness = np.ptp(mesh.vertices[:, 2 - axis])
    assert thickness == pytest.approx(0.5 if value == 1.0 else 2 * END_GAP * 0.5, rel=1e-9)


@pytest.mark.parametrize(
    "step, bodies",
    [((1, 1, 0), 1), ((1, 0, 1), 1), ((0, 1, 1), 1), ((1, 1, 1), 1), ((-1, -1, -1), 1), ((1, -1, 0), 2),
     ((1, 0, -1), 2), ((-1, 1, 1), 2)],
    ids=["+x+y", "+x+z", "+y+z", "+x+y+z", "-x-y-z", "+x-y", "+x-z", "-x+y+z"],
)  # fmt: skip
def test_two_samples_that_meet_across_a_diagonal_are_joined_only_along_the_split(step, bodies):
    # The first sample at (2, 2, 2); the second one step away along a face or body diagonal.
    values = np.zeros((5, 5, 5))
    values[2, 2, 2] = 1.0
    i, j, k = (2 + d for d in step)
    values[k, j, i] = 1.0
    mesh = iso_surface(ScalarField(values, 1.0), 0.5)
    check_surface(mesh, 1.0)
    assert len(mesh.bodies()) == bodies


def test_the_surface_is_the_same_on_every_thread_count():
    field = ScalarField(random_field((30, 41, 37), (0.0, 0.5, 1.0, 0.3), 5), 0.5)
    one = iso_surface(field, 0.5, threads=1)
    for threads in (2, 7, 12):
        many = iso_surface(field, 0.5, threads=threads)
        assert np.array_equal(many.vertices, one.vertices) and np.array_equal(many.triangles, one.triangles)


def test_surfaces_of_known_genus():
    assert iso_surface(sphere_field(1.0), 0.0).genus() == 0
    assert iso_surface(torus_field(1.0), 0.0).genus() == 1

    # Two tori side by side, overlapping: one body with two through holes.
    def two_tori(x, y, z):
        first = 2.0 - np.sqrt((np.sqrt((x + 5.0) ** 2 + y**2) - 5.0) ** 2 + z**2)
        second = 2.0 - np.sqrt((np.sqrt((x - 5.0) ** 2 + y**2) - 5.0) ** 2 + z**2)
        return np.maximum(first, second)

    mesh = iso_surface(lattice_field((-14.0, -9.0, -4.0), (14.0, 9.0, 4.0), 0.5, two_tori), 0.0)
    assert len(mesh.bodies()) == 1 and mesh.genus() == 2 and mesh.euler_characteristic() == -2


def test_a_hollow_ball_gives_two_bodies_the_cavity_facing_inward():
    def shell(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.minimum(8.0 - r, r - 5.0)

    mesh = iso_surface(lattice_field((-10.0,) * 3, (10.0,) * 3, 0.5, shell), 0.0)
    outer, cavity = mesh.bodies()
    assert outer.volume_mm3 > 0 > cavity.volume_mm3
    assert outer.genus() == cavity.genus() == 0
    exact = 4.0 / 3.0 * math.pi * (8.0**3 - 5.0**3)
    assert mesh.volume_mm3 == pytest.approx(exact, rel=0.01)


def test_iso_surface_refusals():
    values = np.zeros((5, 5, 5))
    values[1:4, 1:4, 1:4] = 1.0
    field = ScalarField(values, 1.0)
    touching = values.copy()
    touching[4, 2, 3] = 0.5
    with pytest.raises(michell.EngineError, match=r"point \(3, 2, 4\) on the lattice's outer layer") as refused:
        iso_surface(ScalarField(touching, 1.0), 0.5)
    assert refused.value.code == "invalid_input" and refused.value.details["k"] == 4
    with pytest.raises(michell.EngineError, match="no sample of the field reaches the level"):
        iso_surface(field, 1.5)
    with pytest.raises(michell.EngineError, match="threads"):
        iso_surface(field, 0.5, threads=0)
    with pytest.raises(InvalidValue):
        iso_surface(field, "half")
    with pytest.raises(InvalidValue):
        iso_surface(field, True)
    with pytest.raises(InvalidType):
        iso_surface(values, 0.5)
    with pytest.raises(michell.EngineError, match="level"):
        iso_surface(field, math.nan)


# ---------------------------------------------------------------------------------------------------------
# Refining


@pytest.mark.parametrize("order", ["linear", "cubic"])
def test_refining_keeps_every_old_sample_and_reproduces_linear_fields(order):
    rng = np.random.default_rng(3)
    field = ScalarField(rng.random((4, 5, 6)), 0.6, (1.0, -2.0, 0.5))
    fine = refine(field, 3, order)
    assert fine.counts == (16, 13, 10) and fine.spacing_mm == pytest.approx(0.2) and fine.origin_mm == field.origin_mm
    assert np.array_equal(fine.values[::3, ::3, ::3], field.values)
    linear = lattice_field((0.0, 0.0, 0.0), (3.0, 2.0, 2.5), 0.5, lambda x, y, z: 2.0 * x - 3.0 * y + 0.5 * z + 1.0)
    fine = refine(linear, 4, order)
    x, y, z = np.meshgrid(*fine.coordinates(), indexing="ij")
    expected = (2.0 * x - 3.0 * y + 0.5 * z + 1.0).transpose(2, 1, 0)
    np.testing.assert_allclose(fine.values, expected, rtol=0, atol=1e-12)
    assert refine(field, 1, order).values.tolist() == field.values.tolist()


def test_the_cubic_order_is_more_accurate_than_the_linear_one_on_a_smooth_field():
    # Monotone along every axis, so that no slope is set to 0 at an extremum.
    function = lambda x, y, z: np.exp(0.4 * x) + np.tanh(0.5 * y - 1.0) + 0.2 * z**3  # noqa: E731
    coarse = lattice_field((0.0, 0.0, 0.0), (4.0, 4.0, 2.0), 0.5, function)
    errors = {}
    for order in ("linear", "cubic"):
        fine = refine(coarse, 4, order)
        x, y, z = np.meshgrid(*fine.coordinates(), indexing="ij")
        exact = function(x, y, z).transpose(2, 1, 0)
        # Away from the end intervals of the lines, whose end slopes are only first-order accurate.
        errors[order] = np.abs(fine.values - exact)[4:-4, 4:-4, 4:-4].max()
    assert errors["cubic"] < errors["linear"] / 5


def test_at_a_step_the_cubic_order_gives_the_smooth_step_and_does_not_overshoot():
    line = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    values = np.broadcast_to(line, (3, 3, 6))
    fine = refine(ScalarField(values, 1.0), 8, "cubic").values[1, 1]
    t = np.arange(9) / 8
    np.testing.assert_allclose(fine[16:25], 3 * t**2 - 2 * t**3, rtol=0, atol=1e-15)
    assert np.all(fine[:16] == 0.0) and np.all(fine[24:] == 1.0)
    assert fine[20] == 0.5
    # A random 0/1 field stays within [0, 1] in both orders; its outer layer stays 0.
    rng = np.random.default_rng(8)
    noise = (rng.random((7, 8, 9)) > 0.5).astype(float)
    for order in ("linear", "cubic"):
        refined = refine(ScalarField(noise, 1.0), 5, order).values
        assert refined.min() >= 0.0 and refined.max() <= 1.0
    padded = random_field((7, 8, 9), (0.0, 1.0), 9)
    refined = refine(ScalarField(padded, 1.0), 3, "cubic")
    assert np.all(refined.values[[0, -1], :, :] == 0) and np.all(refined.values[:, :, [0, -1]] == 0)
    check_surface(iso_surface(refined, 0.5), 1.0 / 3.0)


def test_a_refined_field_gives_a_closed_surface_closer_to_the_sphere():
    coarse = sphere_field(1.0)
    exact = 4.0 / 3.0 * math.pi * 10.0**3
    error = abs(iso_surface(coarse, 0.0).volume_mm3 / exact - 1.0)
    for order in ("linear", "cubic"):
        mesh = iso_surface(refine(coarse, 2, order), 0.0)
        check_surface(mesh, 0.5)
        assert abs(mesh.volume_mm3 / exact - 1.0) < error


@pytest.mark.parametrize("factor, order", [(0, "cubic"), (2.0, "cubic"), (True, "cubic"), (2, "quadratic")])
def test_refine_refusals(factor, order):
    with pytest.raises(InvalidValue):
        refine(ScalarField(np.zeros((3, 3, 3)), 1.0), factor, order)
    with pytest.raises(InvalidType):
        refine(np.zeros((3, 3, 3)), 2)


# ---------------------------------------------------------------------------------------------------------
# Smoothing


def test_smoothing_moves_vertices_only_and_keeps_a_closed_mesh_closed():
    mesh = iso_surface(torus_field(1.0), 0.0)
    result = smooth(mesh, 20)
    assert np.array_equal(result.mesh.triangles, mesh.triangles)
    assert result.mesh.defects() == mesh.defects() and result.mesh.genus() == 1
    moved = np.linalg.norm(result.mesh.vertices - mesh.vertices, axis=1)
    assert result.largest_move_mm == moved.max() > 0
    assert result.volume_before_mm3 == mesh.volume_mm3 and result.volume_after_mm3 == result.mesh.volume_mm3
    assert result.volume_change == pytest.approx(result.volume_after_mm3 / result.volume_before_mm3 - 1.0)
    assert abs(result.volume_change) < 0.02
    assert result.iterations == 20
    assert smooth(mesh, 0).largest_move_mm == 0.0


def test_smoothing_takes_out_noise_and_fixed_vertices_stay():
    sphere = michell.mesh.sphere((0.0, 0.0, 0.0), 10.0, 48, 24)
    rng = np.random.default_rng(4)
    normals = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1)[:, None]
    noisy = Mesh(sphere.vertices + 0.3 * rng.standard_normal(len(sphere.vertices))[:, None] * normals, sphere.triangles)
    fixed = sphere.vertices[:, 2] > 8.0
    result = smooth(noisy, 30, fixed=fixed)
    assert result.mesh.roughness_deg() < noisy.roughness_deg() / 2
    assert np.array_equal(result.mesh.vertices[fixed], noisy.vertices[fixed])
    assert not np.array_equal(result.mesh.vertices[~fixed], noisy.vertices[~fixed])
    assert abs(result.volume_change) < 0.05


def test_a_vertex_no_triangle_uses_does_not_move():
    box = michell.mesh.box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    loose = Mesh(np.vstack([box.vertices, [[5.0, 5.0, 5.0]]]), box.triangles)
    assert smooth(loose, 3).mesh.vertices[-1].tolist() == [5.0, 5.0, 5.0]


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"iterations": -1}, InvalidValue),
        ({"iterations": 2.0}, InvalidValue),
        ({"iterations": 2, "lam": 0.0}, InvalidValue),
        ({"iterations": 2, "mu": -0.4}, InvalidValue),
        ({"iterations": 2, "mu": math.nan}, InvalidValue),
        ({"iterations": 2, "fixed": np.zeros(8)}, InvalidValue),
        ({"iterations": 2, "fixed": np.zeros(7, dtype=bool)}, InvalidValue),
    ],
    ids=["negative iterations", "iterations not whole", "lam 0", "mu above -lam", "mu not a number",
         "fixed not boolean", "fixed of the wrong length"],  # fmt: skip
)
def test_smooth_refusals(arguments, error):
    with pytest.raises(error):
        smooth(michell.mesh.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), **arguments)


def test_smooth_refuses_what_is_not_a_mesh():
    with pytest.raises(InvalidType):
        smooth(np.zeros((3, 3)), 2)


@pytest.mark.slow
def test_a_large_lattice_on_four_threads_is_closed_and_equal_to_one_thread():
    field = lattice_field((-15.0, -10.0, -10.0), (14.9, 9.9, 9.9), 0.1,
                          lambda x, y, z: 9.0 - np.sqrt((x / 1.4) ** 2 + y**2 + z**2))  # fmt: skip
    assert field.counts == (300, 200, 200)
    four = iso_surface(field, 0.0, threads=4)
    one = iso_surface(field, 0.0, threads=1)
    assert np.array_equal(four.vertices, one.vertices) and np.array_equal(four.triangles, one.triangles)
    check_surface(four, 0.1)
    assert len(four.bodies()) == 1 and four.genus() == 0
