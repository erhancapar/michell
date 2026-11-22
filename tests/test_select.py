"""michell.select: every function against indices found by brute force from node and element coordinates on
small 2D and 3D grids; the load vector against a consistent load worked by hand on a 2 x 2 patch and against
the loads of the bridge of row 28 built by index arithmetic (bit for bit); refused arguments."""

import itertools
import math

import numpy as np
import pytest

import michell
from michell import select

INF = math.inf
TOL = 1e-6  # element sides, as in michell.select


def grids():
    return [michell.Grid(4, 3, 0.5, 2.0), michell.Grid3d(3, 2, 4, 0.5)]


def node_points(grid):
    """Every node as (index, coordinates in mm), from grid.node."""
    if isinstance(grid, michell.Grid3d):
        ranges = (range(grid.nx + 1), range(grid.ny + 1), range(grid.nz + 1))
    else:
        ranges = (range(grid.nx + 1), range(grid.ny + 1))
    for ijk in itertools.product(*ranges):
        yield grid.node(*ijk), tuple(grid.h * a for a in ijk)


def element_points(grid):
    """Every element as (index, centre in mm), from grid.element."""
    if isinstance(grid, michell.Grid3d):
        ranges = (range(grid.nx), range(grid.ny), range(grid.nz))
    else:
        ranges = (range(grid.nx), range(grid.ny))
    for ijk in itertools.product(*ranges):
        yield grid.element(*ijk), tuple(grid.h * (a + 0.5) for a in ijk)


def inside(point, box, h):
    return all(low - TOL * h <= p <= high + TOL * h for p, (low, high) in zip(point, box))


def boxes(grid):
    """Boxes with bounds on grid lines, between them, computed with rounding, open and degenerate."""
    h = grid.h
    dims = 3 if isinstance(grid, michell.Grid3d) else 2
    picks = [
        (0.0, h),
        (0.3 * h, 2.7 * h),
        (h * (1 - 1e-7), 2 * h * (1 + 1e-7)),  # bounds within the tolerance of grid lines
        (h * (1 + 1e-5), 3 * h),  # a bound beyond the tolerance
        (-INF, INF),
        (h, h),
        (-1.0, 0.0),
        (2 * h, 1e9),
    ]
    return list(itertools.product(picks, repeat=dims))


@pytest.mark.parametrize("grid", grids(), ids=["2d", "3d"])
def test_nodes_and_elements_in_a_box_match_brute_force(grid):
    for box in boxes(grid):
        expected_nodes = sorted(n for n, p in node_points(grid) if inside(p, box, grid.h))
        assert select.nodes_in(grid, box).tolist() == expected_nodes, box
        expected = np.zeros(grid.n_elements, dtype=bool)
        for e, c in element_points(grid):
            expected[e] = inside(c, box, grid.h)
        mask = select.elements_in(grid, box)
        assert mask.dtype == np.bool_ and mask.shape == (grid.n_elements,)
        assert np.array_equal(mask, expected), box


@pytest.mark.parametrize("grid", grids(), ids=["2d", "3d"])
def test_face_nodes_match_brute_force(grid):
    is_3d = isinstance(grid, michell.Grid3d)
    sizes = (grid.nx, grid.ny, grid.nz) if is_3d else (grid.nx, grid.ny)
    faces = select.FACES_3D if is_3d else select.FACES_2D
    some_boxes = boxes(grid)[::5]
    for face in faces:
        axis = "xyz".index(face[0])
        plane = 0.0 if face[1] == "-" else sizes[axis] * grid.h
        on = sorted(n for n, p in node_points(grid) if p[axis] == plane)
        assert select.face_nodes(grid, face).tolist() == on
        for box in some_boxes:
            expected = sorted(n for n, p in node_points(grid) if p[axis] == plane and inside(p, box, grid.h))
            assert select.face_nodes(grid, face, within=box).tolist() == expected


@pytest.mark.parametrize("grid", grids(), ids=["2d", "3d"])
def test_dofs_and_held_match_brute_force(grid):
    per_node = 3 if isinstance(grid, michell.Grid3d) else 2
    letters = "xyz"[:per_node]
    nodes = np.array([0, 5, grid.n_nodes - 1, 7, 5])
    for count in range(1, per_node + 1):
        for directions in itertools.permutations(letters, count):
            directions = "".join(directions)
            expected = sorted({per_node * n + letters.index(d) for n in nodes.tolist() for d in directions})
            assert select.dofs(grid, nodes, directions).tolist() == expected
            mask = select.held(grid, nodes, directions)
            assert mask.dtype == np.bool_ and mask.shape == (grid.n_dofs,)
            assert np.flatnonzero(mask).tolist() == expected
    every = sorted({per_node * n + a for n in nodes.tolist() for a in range(per_node)})
    assert select.dofs(grid, nodes).tolist() == every


def test_load_on_a_2_by_2_patch_is_the_consistent_load_worked_by_hand():
    # 3 x 3 nodes on the face z+ bound 4 squares; each square carries F / 4, a quarter of it to each corner:
    # the corners take F / 16, the edge nodes F / 8 and the middle node F / 4. Exact in binary for this F.
    grid = michell.Grid3d(4, 3, 2, 1.0)
    nodes = [grid.node(i, j, 2) for i in (1, 2, 3) for j in (0, 1, 2)]
    force = (8.0, -4.0, -16.0)
    loads = select.load(grid, nodes, force)
    expected = np.zeros(grid.n_dofs)
    for i in (1, 2, 3):
        for j in (0, 1, 2):
            weight = (1 if i in (1, 3) else 2) * (1 if j in (0, 2) else 2) / 16
            n = grid.node(i, j, 2)
            expected[3 * n : 3 * n + 3] = [weight * f for f in force]
    assert np.array_equal(loads, expected)  # bit for bit
    # An arbitrary force: the same weights, to 1e-15 relative.
    force = (1000.0 / 3, math.pi, -math.e * 100)
    loads = select.load(grid, nodes, force)
    for i in (1, 2, 3):
        for j in (0, 1, 2):
            weight = (1 if i in (1, 3) else 2) * (1 if j in (0, 2) else 2) / 16
            n = grid.node(i, j, 2)
            assert loads[3 * n : 3 * n + 3] == pytest.approx([weight * f for f in force], rel=1e-15, abs=0)
    for axis in range(3):
        assert math.fsum(loads[axis::3]) == force[axis]


def test_load_on_a_2d_face_and_on_an_l_shaped_patch():
    grid = michell.Grid(5, 4, 0.5, 1.0)
    nodes = [grid.node(i, 4) for i in (1, 2, 3)]  # two element sides on the face y+: a half to each end
    loads = select.load(grid, nodes, (0.0, -8.0))
    assert loads[[2 * n + 1 for n in nodes]].tolist() == [-2.0, -4.0, -2.0]
    assert np.count_nonzero(loads) == 3

    grid = michell.Grid3d(3, 3, 2, 1.0)
    squares = [(0, 0), (1, 0), (0, 1)]  # an L of three squares on the face x+
    nodes = sorted({grid.node(3, j + a, k + b) for j, k in squares for a in (0, 1) for b in (0, 1)})
    loads = select.load(grid, nodes, (-12.0, 0.0, 0.0))
    shares = {n: 0 for n in nodes}
    for j, k in squares:
        for a in (0, 1):
            for b in (0, 1):
                shares[grid.node(3, j + a, k + b)] += 1
    assert {n: loads[3 * n] for n in nodes} == {n: -1.0 * s for n, s in shares.items()}


def test_load_falls_back_to_equal_shares():
    grid = michell.Grid3d(3, 3, 3, 1.0)
    edge = [grid.node(3, j, 3) for j in range(4)]  # a line on two faces: no face area
    inner = [grid.node(1, 1, 1), grid.node(2, 1, 1)]  # off the faces
    scattered = [grid.node(0, 0, 0), grid.node(3, 3, 0)]  # on one face, but bounding no square
    for nodes in ([grid.node(1, 1, 3)], edge, inner, scattered):
        loads = select.load(grid, nodes, (0.0, 0.0, -10.0))
        assert loads[[3 * n + 2 for n in nodes]] == pytest.approx([-10.0 / len(nodes)] * len(nodes), rel=1e-15)
        assert np.count_nonzero(loads) == len(nodes)
        assert math.fsum(loads) == -10.0


def test_load_sums_to_the_force_exactly():
    grid = michell.Grid3d(9, 7, 5, 1.0)
    rng = np.random.default_rng(3)
    for _ in range(40):
        i0, j0 = rng.integers(0, 6), rng.integers(0, 4)
        a, b = rng.integers(1, 4), rng.integers(1, 4)
        nodes = [grid.node(i, j, 5) for i in range(i0, i0 + a + 1) for j in range(j0, j0 + b + 1)]
        force = tuple(rng.normal(size=3) * 1000.0 / 7)
        loads = select.load(grid, nodes, force)
        for axis in range(3):
            assert math.fsum(loads[axis::3]) == force[axis]
        count = len(nodes) + 1
        loads = select.load(grid, [*nodes, grid.node(0, 0, 0)][:count], force)  # equal shares
        for axis in range(3):
            assert math.fsum(loads[axis::3]) == force[axis]


def test_load_matches_the_bridge_loads_built_by_index_arithmetic():
    # The pad of row 28 and examples/bridge_3d_member_size.py, as it was built before michell.select.
    nx, ny, nz = 50, 24, 10
    grid = michell.Grid3d(nx, ny, nz, 1.0)
    i0, j0 = (nx - 6) // 2, (ny - 6) // 2
    by_hand = np.zeros(grid.n_dofs)
    for i in range(i0, i0 + 6):
        for j in range(j0, j0 + 6):
            for a in (0, 1):
                for b in (0, 1):
                    by_hand[3 * grid.node(i + a, j + b, nz) + 2] -= 1000.0 / 36 / 4
    held_by_hand = np.zeros(grid.n_dofs, dtype=bool)
    for i in (0, nx):
        for k in range(nz + 1):
            for j in range(ny + 1):
                node = grid.node(i, j, k)
                held_by_hand[3 * node : 3 * node + 3] = True
    kept_by_hand = np.zeros((nz, ny, nx), dtype=bool)
    kept_by_hand[:, :, :2] = True
    kept_by_hand[:, :, nx - 2 :] = True
    kept_by_hand[nz - 2 :, j0 : j0 + 6, i0 : i0 + 6] = True

    pad = ((22.0, 28.0), (9.0, 15.0), (8.0, 10.0))
    loads = select.load(grid, select.face_nodes(grid, "z+", within=pad), (0.0, 0.0, -1000.0))
    assert np.array_equal(loads, by_hand)  # bit for bit
    supports = np.concatenate([select.face_nodes(grid, "x-"), select.face_nodes(grid, "x+")])
    assert np.array_equal(select.held(grid, supports), held_by_hand)
    everywhere = (-INF, INF)
    kept = (
        select.elements_in(grid, ((0.0, 2.0), everywhere, everywhere))
        | select.elements_in(grid, ((nx - 2.0, nx), everywhere, everywhere))
        | select.elements_in(grid, pad)
    )
    assert np.array_equal(kept, kept_by_hand.ravel())

    from michell.verification.recipe_rows import bridge

    for dims in ((50, 24, 10), (16, 8, 6)):  # row 28 and its small version in tests/test_recipes.py
        part = bridge(*dims)
        nx, ny, nz = dims
        g = part.problem.grid
        i0, j0 = (nx - 6) // 2, (ny - 6) // 2
        expected = np.zeros(g.n_dofs)
        for i in range(i0, i0 + 6):
            for j in range(j0, j0 + 6):
                for a in (0, 1):
                    for b in (0, 1):
                        expected[3 * g.node(i + a, j + b, nz) + 2] -= 1000.0 / 36 / 4
        assert np.array_equal(part.problem.cases[0].loads, expected)
        expected_kept = np.zeros((nz, ny, nx), dtype=bool)
        expected_kept[:, :, :2] = True
        expected_kept[:, :, nx - 2 :] = True
        expected_kept[nz - 2 :, j0 : j0 + 6, i0 : i0 + 6] = True
        assert np.array_equal(part.keep_solid.mask, expected_kept.ravel())


def refused(call, *args, **kwargs):
    with pytest.raises(michell.EngineError) as caught:
        call(*args, **kwargs)
    assert caught.value.code == "invalid_input"


def test_refused_arguments():
    flat, block = michell.Grid(4, 3, 0.5, 2.0), michell.Grid3d(3, 2, 4, 0.5)
    refused(select.face_nodes, flat, "z+")
    refused(select.face_nodes, block, "top")
    refused(select.face_nodes, "grid", "x-")
    refused(select.nodes_in, flat, ((0, 1), (0, 1), (0, 1)))
    refused(select.nodes_in, block, ((0, 1), (0, 1)))
    refused(select.nodes_in, block, ((1, 0), (0, 1), (0, 1)))
    refused(select.nodes_in, block, ((0, math.nan), (0, 1), (0, 1)))
    refused(select.elements_in, block, ((0, "1"), (0, 1), (0, 1)))
    refused(select.elements_in, block, [(0, 1, 2), (0, 1), (0, 1)])
    refused(select.dofs, block, [0, block.n_nodes])
    refused(select.dofs, block, [-1])
    refused(select.dofs, block, [])
    refused(select.dofs, block, np.array([True, False]))
    refused(select.dofs, block, [0.0, 1.0])
    refused(select.dofs, block, [[0, 1]])
    refused(select.dofs, block, [0], "xx")
    refused(select.dofs, block, [0], "w")
    refused(select.dofs, block, [0], "")
    refused(select.dofs, flat, [0], "xz")
    refused(select.held, flat, [0], "xyz")
    refused(select.load, block, [0, 0], (0.0, 0.0, 1.0))
    refused(select.load, block, [0], (0.0, 1.0))
    refused(select.load, flat, [0], (0.0, 0.0, 1.0))
    refused(select.load, block, [0], (0.0, math.inf, 1.0))
    refused(select.load, block, [0], ("a", "b", "c"))
    with pytest.raises(ValueError):  # a refused value is also a ValueError
        select.face_nodes(flat, "z-")
    with pytest.raises(TypeError):  # a wrong type is also a TypeError
        select.nodes_in("grid", ((0, 1), (0, 1)))
