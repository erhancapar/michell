"""michell.mesh: the Mesh and its check, STL files in both forms (and a binary file whose header starts with
"solid"), joining vertices, and the exact volumes of the bodies made from their dimensions."""

import math

import numpy as np
import pytest

import michell
from michell import mesh
from michell.mesh import Mesh, read_stl, write_stl


def sphere_volume(radius, segments, rings):
    theta = np.pi * np.arange(rings + 1) / rings
    rho, z = radius * np.sin(theta), radius * np.cos(theta)
    bands = (z[:-1] - z[1:]) * (rho[:-1] ** 2 + rho[:-1] * rho[1:] + rho[1:] ** 2)
    return segments / 6 * math.sin(2 * math.pi / segments) * math.fsum(bands.tolist())


@pytest.mark.parametrize(
    "made, expected",
    [
        (lambda: mesh.box((0.5, -1.0, 2.0), (3.0, 1.0, 2.25)), 2.5 * 2.0 * 0.25),
        (lambda: mesh.sphere((1.0, 2.0, 3.0), 5.0, 32, 16), sphere_volume(5.0, 32, 16)),
        (lambda: mesh.sphere((0.0, 0.0, 0.0), 2.0, 7, 5), sphere_volume(2.0, 7, 5)),
        (lambda: mesh.cylinder((0.0, 1.0, 0.0), 2.0, 7.0, axis="y", segments=24), 12 * math.sin(math.pi / 12) * 28),
        (lambda: mesh.cylinder((0.0, 0.0, 0.0), 1.0, 3.0, axis="x", segments=5), 2.5 * math.sin(2 * math.pi / 5) * 3),
        (lambda: mesh.cylinder((0.0, 0.0, 0.0), 1.0, 3.0, segments=6), 3 * math.sin(math.pi / 3) * 3),
        (
            lambda: mesh.torus((0.0, 0.0, 0.0), 5.0, 1.5, segments=48, sides=24),
            48 * math.sin(math.pi / 24) * 5.0 * 12 * math.sin(math.pi / 12) * 1.5**2,
        ),
        (
            lambda: mesh.torus((1.0, 0.0, 0.0), 3.0, 1.0, axis="x", segments=12, sides=8),
            12 * math.sin(math.pi / 6) * 3.0 * 4 * math.sin(math.pi / 4) * 1.0,
        ),
    ],
    ids=["box", "sphere", "coarse sphere", "cylinder y", "cylinder x", "cylinder z", "torus", "torus x"],
)
def test_the_bodies_are_closed_outward_and_have_the_stated_volume(made, expected):
    body = made()
    assert body.closed() is body
    assert body.defects() == {
        "open_edges": 0,
        "edges_on_more_than_two": 0,
        "edges_same_direction": 0,
        "zero_area_triangles": 0,
    }
    assert body.volume_mm3 == pytest.approx(expected, rel=1e-13)


def test_the_bodies_of_row_30_have_no_triangle_parallel_to_the_rays_except_the_cylinder_caps():
    def normal_x(body):
        v, t = body.vertices, body.triangles
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        return np.abs(n[:, 0]) / np.linalg.norm(n, axis=1)

    assert normal_x(mesh.sphere((0, 0, 0), 10.0, 32, 16)).min() > 1e-3
    assert normal_x(mesh.torus((0, 0, 0), 8.0, 3.0, segments=48, sides=24)).min() > 1e-3
    cylinder = mesh.cylinder((0, 0, 0), 6.0, 16.0, axis="y", segments=24)
    sides = np.ptp(cylinder.vertices[cylinder.triangles][:, :, 1], axis=1) > 0
    assert normal_x(cylinder)[sides].min() > 1e-3
    assert np.all(normal_x(cylinder)[~sides] == 0.0)


def test_a_mesh_is_read_only_and_its_measures():
    body = mesh.box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        body.vertices[0, 0] = 5.0
    with pytest.raises(ValueError):
        body.triangles[0, 0] = 1
    assert body.area_mm2 == pytest.approx(2 * (2 + 3 + 6))
    assert body.bounds.tolist() == [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    assert "8 vertices, 12 triangles" in repr(body)
    # The volume is taken about the box's centre, so a far offset does not cost digits.
    far = Mesh(body.vertices + 1e6, body.triangles)
    assert far.volume_mm3 == pytest.approx(6.0, rel=1e-9)


@pytest.mark.parametrize(
    "vertices, triangles, kind",
    [
        ([[0, 0, 0]], [[0, 0]], ValueError),
        ([[0, 0]], [[0, 0, 0]], ValueError),
        ([[0, 0, math.nan]], np.zeros((0, 3), dtype=int), ValueError),
        ([[0, 0, 0]], [[0, 0, 1]], ValueError),
        ([[0, 0, 0]], [[0.0, 0.0, 0.0]], TypeError),
        ([[True, False, True]], [[0, 0, 0]], TypeError),
    ],
    ids=["triangle of two", "2D vertex", "NaN", "index out of range", "float indices", "bool vertices"],
)
def test_wrong_arrays_are_refused(vertices, triangles, kind):
    with pytest.raises(kind) as error:
        Mesh(vertices, triangles)
    assert isinstance(error.value, michell.EngineError) and error.value.code == "invalid_input"


def test_closed_refuses_with_the_numbers_of_what_is_wrong():
    body = mesh.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    v, t = body.vertices, body.triangles
    cases = {
        "open_edges": Mesh(v, t[1:]),  # one triangle missing: 3 open edges
        "edges_on_more_than_two": Mesh(v, np.vstack([t, t[:1]])),  # one triangle twice
        "edges_same_direction": Mesh(v, np.vstack([t[:1, [0, 2, 1]], t[1:]])),  # one triangle turned round
        "zero_area_triangles": Mesh(v, np.vstack([t, [[0, 1, 1], [0, 1, 1]]])),
    }
    expected = {"open_edges": 3, "edges_on_more_than_two": 3, "edges_same_direction": 3, "zero_area_triangles": 2}
    for name, broken in cases.items():
        with pytest.raises(ValueError) as error:
            broken.closed()
        assert error.value.details[name] == expected[name], (name, error.value.details)
        assert "not closed" in error.value.message
    with pytest.raises(ValueError, match="no triangles"):
        Mesh(v, np.zeros((0, 3), dtype=int)).closed()
    flat = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 1]])
    with pytest.raises(ValueError):
        flat.closed()


def test_a_closed_mesh_facing_inward_is_turned_round():
    body = mesh.sphere((1.0, 0.0, 0.0), 2.0, 12, 6)
    inside_out = Mesh(body.vertices, body.triangles[:, ::-1])
    assert inside_out.volume_mm3 == pytest.approx(-body.volume_mm3)
    fixed = inside_out.closed()
    assert fixed.volume_mm3 == pytest.approx(body.volume_mm3)
    assert fixed is not inside_out


def float32_box():
    return mesh.box((0.5, -1.25, 2.0), (3.0, 1.0, 2.25))  # coordinates exact in float32


def test_binary_stl_round_trip(tmp_path):
    body = mesh.sphere((0.25, 0.5, -1.0), 2.0, 16, 8)
    path = tmp_path / "sphere.stl"
    write_stl(path, body)
    assert path.stat().st_size == 84 + 50 * len(body.triangles)
    assert not path.read_bytes().startswith(b"solid")
    back = read_stl(path)
    assert back.closed() is back
    assert len(back.vertices) == len(body.vertices) and len(back.triangles) == len(body.triangles)
    assert np.allclose(back.vertices[back.triangles], body.vertices[body.triangles], atol=1e-6)
    exact = float32_box()
    write_stl(path, exact)
    again = read_stl(path)
    assert np.array_equal(again.vertices[again.triangles], exact.vertices[exact.triangles])
    assert again.volume_mm3 == exact.volume_mm3


def ascii_stl(body, name="part"):
    lines = [f"solid {name}"]
    for a, b, c in body.vertices[body.triangles]:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {float(p[0])!r} {float(p[1])!r} {float(p[2])!r}" for p in (a, b, c)]
        lines += ["    endloop", "  endfacet"]
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def test_ascii_stl_is_read(tmp_path):
    body = float32_box()
    path = tmp_path / "box.stl"
    path.write_text(ascii_stl(body), encoding="ascii")
    back = read_stl(path)
    assert back.closed() is back
    assert len(back.vertices) == 8
    assert np.array_equal(back.vertices[back.triangles], body.vertices[body.triangles])
    # Windows line ends are read too.
    path.write_bytes(ascii_stl(body).replace("\n", "\r\n").encode("ascii"))
    assert read_stl(path).volume_mm3 == body.volume_mm3


def test_a_binary_stl_whose_header_starts_with_solid_is_read_as_binary(tmp_path):
    body = float32_box()
    path = tmp_path / "trap.stl"
    write_stl(path, body)
    data = bytearray(path.read_bytes())
    data[:80] = b"solid part made by a program that writes facet here".ljust(80, b" ")
    path.write_bytes(bytes(data))
    back = read_stl(path)
    assert np.array_equal(back.vertices[back.triangles], body.vertices[body.triangles])


def test_files_that_are_not_stl_are_refused(tmp_path):
    path = tmp_path / "not.stl"
    for content in [b"", b"hello world", b"solid only a name\n", b"\x00" * 100]:
        path.write_bytes(content)
        with pytest.raises(ValueError):
            read_stl(path)
    body = float32_box()
    path.write_text(ascii_stl(body).replace("vertex 3.0", "vertex", 1), encoding="ascii")
    with pytest.raises(ValueError):
        read_stl(path)


def test_vertices_are_joined_exactly_or_within_a_tolerance(tmp_path):
    body = mesh.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    corners = body.vertices[body.triangles]
    # Move one corner of one triangle by 1e-4 mm: an open mesh until the corners are joined within 1e-3.
    corners[0, 0] += 1e-4
    path = tmp_path / "gap.stl"
    path.write_text(ascii_stl(Mesh(corners.reshape(-1, 3), np.arange(36).reshape(12, 3))), encoding="ascii")
    loose = read_stl(path)
    assert len(loose.vertices) == 9
    with pytest.raises(ValueError, match="not closed"):
        loose.closed()
    joined = read_stl(path, merge_mm=1e-3)
    assert len(joined.vertices) == 8
    assert joined.closed() is joined
    # Each group takes the coordinates of its first corner in the file.
    assert np.allclose(joined.vertices[joined.triangles[0, 0]], corners[0, 0])
    with pytest.raises(ValueError):
        read_stl(path, merge_mm=-1.0)


def test_chains_of_close_vertices_join_and_far_ones_do_not():
    points = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.95]])
    groups = mesh._close_groups(points, 1.0)
    assert groups.tolist() == [0, 0, 0, 3, 3]
    assert mesh.smallest_joined(5, np.array([4, 3]), np.array([2, 4])).tolist() == [0, 1, 2, 2, 2]


@pytest.mark.parametrize(
    "call",
    [
        lambda: mesh.box((0, 0, 0), (1, 0, 1)),
        lambda: mesh.sphere((0, 0, 0), -1.0),
        lambda: mesh.sphere((0, 0, 0), 1.0, segments=2),
        lambda: mesh.cylinder((0, 0, 0), 1.0, 1.0, axis="w"),
        lambda: mesh.torus((0, 0, 0), 1.0, 2.0),
        lambda: mesh.torus((0, 0), 2.0, 1.0),
    ],
    ids=["flat box", "negative radius", "two segments", "unknown axis", "minor above major", "2D centre"],
)
def test_bodies_with_wrong_dimensions_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_write_stl_takes_a_mesh(tmp_path):
    with pytest.raises(TypeError):
        write_stl(tmp_path / "x.stl", np.zeros((3, 3)))
