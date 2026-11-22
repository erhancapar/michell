"""michell.mesh: the bodies of a mesh, its Euler characteristic, genus and roughness, and pairs of close
vertices."""

import numpy as np
import pytest

from michell import mesh
from michell.errors import InvalidValue
from michell.mesh import Mesh, close_pairs


def joined(*meshes: Mesh) -> Mesh:
    """The meshes as one, their vertices and triangles one after the other."""
    vertices, triangles, offset = [], [], 0
    for part in meshes:
        vertices.append(part.vertices)
        triangles.append(part.triangles + offset)
        offset += len(part.vertices)
    return Mesh(np.vstack(vertices), np.vstack(triangles))


def test_bodies_come_largest_first_each_with_its_own_vertices():
    small = mesh.box((10.0, 0.0, 0.0), (11.0, 1.0, 1.0))
    large = mesh.sphere((0.0, 0.0, 0.0), 3.0)
    ring = mesh.torus((0.0, 20.0, 0.0), 6.0, 1.5)
    bodies = joined(small, large, ring).bodies()
    assert [len(b.triangles) for b in bodies] == [len(ring.triangles), len(large.triangles), len(small.triangles)]
    assert bodies[0].volume_mm3 > bodies[1].volume_mm3 > bodies[2].volume_mm3
    # Each body keeps its triangles and vertices in their order, renumbered from 0.
    assert np.array_equal(bodies[2].vertices, small.vertices) and np.array_equal(bodies[2].triangles, small.triangles)
    assert [b.genus() for b in bodies] == [1, 0, 0]
    assert len(small.bodies()) == 1 and small.bodies()[0].triangles.tolist() == small.triangles.tolist()


def test_two_boxes_that_share_only_a_corner_are_two_bodies_and_equal_ones_keep_their_order():
    first = mesh.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    second = mesh.box((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    # Share the corner vertex (1, 1, 1): one vertex in both boxes.
    shared = Mesh(np.vstack([first.vertices, second.vertices[1:]]),
                  np.vstack([first.triangles, np.where(second.triangles == 0, 7, second.triangles + 7)]))  # fmt: skip
    assert len(np.unique(shared.vertices, axis=0)) == len(shared.vertices) == 15
    a, b = shared.bodies()
    assert a.volume_mm3 == b.volume_mm3 == 1.0
    assert a.bounds.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]  # the lower first triangle index first
    assert a.genus() == b.genus() == 0
    # As one mesh it is closed edge by edge but not one body.
    with pytest.raises(InvalidValue, match="2"):
        shared.genus()


def test_a_cavity_is_a_body_of_its_own_facing_inward_after_the_solid():
    outer = mesh.box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    inner = mesh.box((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    hollow = joined(Mesh(inner.vertices, inner.triangles[:, [0, 2, 1]]), outer)
    solid, cavity = hollow.bodies()
    assert solid.volume_mm3 == 64.0 and cavity.volume_mm3 == -1.0
    assert hollow.volume_mm3 == pytest.approx(63.0)


@pytest.mark.parametrize(
    "made, chi, genus",
    [
        (lambda: mesh.box((0, 0, 0), (1, 2, 3)), 2, 0),
        (lambda: mesh.sphere((0, 0, 0), 2.0, 12, 6), 2, 0),
        (lambda: mesh.cylinder((0, 0, 0), 1.0, 2.0, segments=9), 2, 0),
        (lambda: mesh.torus((0, 0, 0), 3.0, 1.0, segments=12, sides=8), 0, 1),
    ],
    ids=["box", "sphere", "cylinder", "torus"],
)
def test_euler_characteristic_and_genus(made, chi, genus):
    body = made()
    assert body.euler_characteristic() == chi
    assert body.genus() == genus


def test_the_genus_of_an_open_mesh_is_refused():
    box = mesh.box((0, 0, 0), (1, 1, 1))
    with pytest.raises(InvalidValue, match="3 open edges"):
        Mesh(box.vertices, box.triangles[1:]).genus()
    with pytest.raises(InvalidValue, match="no triangles"):
        Mesh(box.vertices, np.zeros((0, 3), dtype=int)).genus()
    # An open mesh still has an Euler characteristic: a box without one face is a disc.
    open_box = Mesh(box.vertices, box.triangles[[i for i in range(12) if i not in (0, 6)]])
    assert open_box.euler_characteristic() == 1


def test_roughness():
    # A box: 12 edges between faces at 90 degrees and 6 diagonals between halves of one face at 0.
    assert mesh.box((0, 0, 0), (1, 2, 3)).roughness_deg() == pytest.approx(60.0)
    flat = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]])
    assert flat.roughness_deg() == 0.0
    assert Mesh(flat.vertices, flat.triangles[:1]).roughness_deg() == 0.0
    fine = mesh.sphere((0, 0, 0), 1.0, 64, 32).roughness_deg()
    assert fine < mesh.sphere((0, 0, 0), 1.0, 16, 8).roughness_deg()
    assert 0 < fine < 10


def test_close_pairs_finds_each_pair_once():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.5], [3.0, 0.0, 0.0], [3.0, 0.0, 0.9]])
    first, second = close_pairs(points, 1.0)
    assert sorted(zip(first.tolist(), second.tolist())) == [(0, 1), (3, 4)]
    assert len(close_pairs(points[:1], 1.0)[0]) == 0
