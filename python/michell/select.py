"""Nodes, elements and degrees of freedom picked by position, and loads built from them, on a 2D ``Grid`` or a
3D ``Grid3d``, so that a problem is set up without computing indices by hand.

Positions are in mm, with the grid's corner at the origin: node (i, j, k) lies at (i h, j h, k h), element
(i, j, k) has its centre at ((i + 1/2) h, (j + 1/2) h, (k + 1/2) h), for the element side h (in 2D the same
without k). The faces of the grid are named as ``DensityFilter`` names them: ``"x-"`` is the face x = 0 and
``"x+"`` the face x = nx h, and likewise ``"y-"``, ``"y+"`` and, on a 3D grid, ``"z-"``, ``"z+"``.

A box is a pair (low, high) of bounds in mm per direction: two pairs on a 2D grid, three on a 3D grid, such as
``((22, 28), (9, 15), (10, 10))``; a bound may be ``math.inf`` or ``-math.inf``. A box is closed on all sides:
a point lies inside when every coordinate lies between its bounds or within 1e-6 h of one of them, so a box
whose bounds fall on grid lines takes in the nodes on them however the bounds were computed.

- ``face_nodes(grid, face, within=None)``: the nodes on a face, optionally only those inside a box;
- ``nodes_in(grid, box)``: the nodes inside a box;
- ``elements_in(grid, box)``: the elements whose centre lies inside a box, as a boolean mask with one entry
  per element (for ``KeepSolid`` and ``active=``);
- ``dofs(grid, nodes, directions=None)``: the degrees of freedom of nodes in the given directions;
- ``held(grid, nodes, directions=None)``: the same as a boolean mask with one entry per degree of freedom (for
  ``Problem``);
- ``load(grid, nodes, force)``: a load vector (for ``LoadCase``) that spreads a force over nodes, as the
  consistent nodal load of a uniform traction when the nodes cover a patch of element faces on a face of the
  grid, in equal shares otherwise.

Node lists are index arrays in ascending order; masks combine with ``|`` and ``&``. Refused arguments raise
``EngineError`` with code ``invalid_input`` (``michell.errors``).
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue

# A point within this many element sides of a bound of a box counts as on the bound.
TOLERANCE_IN_ELEMENTS = 1e-6

FACES_2D = ("x-", "x+", "y-", "y+")
FACES_3D = ("x-", "x+", "y-", "y+", "z-", "z+")


def _dims(grid) -> tuple[int, ...]:
    if isinstance(grid, _engine.Grid3d):
        return (grid.nx, grid.ny, grid.nz)
    if isinstance(grid, _engine.Grid):
        return (grid.nx, grid.ny)
    raise InvalidType(f"a michell.Grid or michell.Grid3d is needed; got {type(grid).__name__}")


def _node_ijk(grid) -> np.ndarray:
    """The whole-number position (i, j[, k]) of every node, in node order, shape (n_nodes, 2 or 3)."""
    counts = [n + 1 for n in _dims(grid)]
    grids = np.meshgrid(*[np.arange(n) for n in reversed(counts)], indexing="ij")
    return np.stack([g.ravel() for g in reversed(grids)], axis=1)


def _element_ijk(grid) -> np.ndarray:
    """The whole-number position (i, j[, k]) of every element, in element order."""
    grids = np.meshgrid(*[np.arange(n) for n in reversed(_dims(grid))], indexing="ij")
    return np.stack([g.ravel() for g in reversed(grids)], axis=1)


def _box(grid, box) -> np.ndarray:
    dims = _dims(grid)
    if not isinstance(box, (list, tuple)) or len(box) != len(dims):
        raise InvalidValue(
            f"a box on a {len(dims)}D grid is {len(dims)} pairs (low, high) of bounds in mm, one per direction; "
            f"got {box!r}"
        )
    bounds = []
    for axis, pair in zip("xyz", box):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InvalidValue(f"the {axis} bounds of a box are a pair (low, high) in mm; got {pair!r}")
        for value in pair:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise InvalidType(f"the {axis} bounds of a box must be numbers; got {pair!r}")
            if math.isnan(value):
                raise InvalidValue(f"the {axis} bounds of a box must not be NaN; got {pair!r}")
        low, high = float(pair[0]), float(pair[1])
        if low > high:
            raise InvalidValue(f"the {axis} bounds of a box run from low to high; got ({low:g}, {high:g})")
        bounds.append((low, high))
    return np.array(bounds)


def _inside(grid, points_mm: np.ndarray, box) -> np.ndarray:
    bounds = _box(grid, box)
    tolerance = TOLERANCE_IN_ELEMENTS * float(grid.h)
    return np.all((points_mm >= bounds[:, 0] - tolerance) & (points_mm <= bounds[:, 1] + tolerance), axis=1)


def _face(grid, face) -> tuple[int, int]:
    """The axis of ``face`` and the whole-number position of its plane along it."""
    dims = _dims(grid)
    names = FACES_3D if len(dims) == 3 else FACES_2D
    if face not in names:
        raise InvalidValue(f"the faces of a {len(dims)}D grid are {', '.join(names)}; got {face!r}")
    axis = "xyz".index(face[0])
    return axis, 0 if face[1] == "-" else dims[axis]


def _nodes(grid, nodes) -> np.ndarray:
    """``nodes`` as an array of node indices, checked; duplicates are kept."""
    _dims(grid)
    values = np.asarray(nodes)
    if values.ndim != 1:
        raise InvalidValue(f"nodes are given as a flat array of node indices; got shape {values.shape}")
    if values.dtype == np.bool_ or not (values.size == 0 or np.issubdtype(values.dtype, np.integer)):
        raise InvalidType(f"nodes are given as whole-number node indices; got an array of {values.dtype}")
    if values.size == 0:
        raise InvalidValue("no nodes were given")
    bad = (values < 0) | (values >= grid.n_nodes)
    if bad.any():
        raise InvalidValue(
            f"node {int(values[bad][0])} is not a node of the grid (0 to {grid.n_nodes - 1})",
            {"node": int(values[bad][0]), "n_nodes": int(grid.n_nodes)},
        )
    return values.astype(np.int64)


def _directions(grid, directions) -> list[int]:
    letters = "xyz"[: len(_dims(grid))]
    if not isinstance(directions, str) or not directions:
        raise InvalidValue(f"directions are a string of the letters {letters!r}, such as 'x' or {letters!r}")
    if any(d not in letters for d in directions) or len(set(directions)) != len(directions):
        raise InvalidValue(
            f"directions are distinct letters of {letters!r} on a {len(letters)}D grid; got {directions!r}"
        )
    return sorted(letters.index(d) for d in directions)


def face_nodes(grid, face: str, within=None) -> np.ndarray:
    """The nodes on ``face`` of the grid (``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, in 3D ``"z-"``, ``"z+"``),
    in ascending order; with ``within`` (a box, see the module), only those inside it."""
    axis, plane = _face(grid, face)
    ijk = _node_ijk(grid)
    on = ijk[:, axis] == plane
    if within is not None:
        on &= _inside(grid, ijk * float(grid.h), within)
    return np.flatnonzero(on).astype(np.int64)


def nodes_in(grid, box) -> np.ndarray:
    """The nodes inside ``box`` (closed, with a tolerance of 1e-6 h; see the module), in ascending order."""
    return np.flatnonzero(_inside(grid, _node_ijk(grid) * float(grid.h), box)).astype(np.int64)


def elements_in(grid, box) -> np.ndarray:
    """A boolean mask with one entry per element, true for the elements whose centre lies inside ``box``
    (closed, with a tolerance of 1e-6 h; see the module). For ``KeepSolid`` and for ``active=``."""
    return _inside(grid, (_element_ijk(grid) + 0.5) * float(grid.h), box)


def dofs(grid, nodes, directions: str | None = None) -> np.ndarray:
    """The degrees of freedom of ``nodes`` in ``directions`` (letters of ``"xy"`` on a 2D grid, of ``"xyz"``
    on a 3D grid, such as ``"x"`` or ``"xz"``; every direction if omitted), in ascending order, each once:
    node n moves in x at degree of freedom d n, in y at d n + 1 and in z at d n + 2, with d = 2 in 2D and 3
    in 3D."""
    nodes = _nodes(grid, nodes)
    per_node = len(_dims(grid))
    if directions is None:
        directions = "xyz"[:per_node]
    axes = np.array(_directions(grid, directions))
    return np.unique((per_node * nodes[:, None] + axes[None, :]).ravel())


def held(grid, nodes, directions: str | None = None) -> np.ndarray:
    """A boolean mask with one entry per degree of freedom, true for the degrees of freedom of ``nodes`` in
    ``directions`` (as in ``dofs``): the ``held`` argument of ``Problem``. Masks of several supports combine
    with ``|``."""
    chosen = dofs(grid, nodes, directions)
    mask = np.zeros(grid.n_dofs, dtype=bool)
    mask[chosen] = True
    return mask


def _patch(grid, nodes: np.ndarray) -> np.ndarray | None:
    """The element faces (2D: element sides) on a face of the grid whose corners are all among ``nodes``, as
    an array of corner node indices with one row per face, if every node is a corner of one of them; else
    None."""
    dims = _dims(grid)
    ijk = _node_ijk(grid)[nodes]
    chosen = np.zeros(grid.n_nodes, dtype=bool)
    chosen[nodes] = True
    strides = np.cumprod([1, *[n + 1 for n in dims[:-1]]])
    for face in FACES_3D[: 2 * len(dims)]:
        axis, plane = _face(grid, face)
        if not np.all(ijk[:, axis] == plane):
            continue
        across = [a for a in range(len(dims)) if a != axis]
        # Every element face on this grid face, by its lowest corner, with its corners.
        lows = np.meshgrid(*[np.arange(dims[a]) for a in across], indexing="ij")
        lows = np.stack([g.ravel() for g in lows], axis=1)
        corner_offsets = np.array(np.meshgrid(*[[0, 1]] * len(across), indexing="ij")).reshape(len(across), -1).T
        base = plane * strides[axis] + lows @ strides[across]
        corners = base[:, None] + corner_offsets @ strides[across]
        patch = corners[np.all(chosen[corners], axis=1)]
        if patch.size and np.isin(nodes, patch).all():
            return patch
    return None


def load(grid, nodes, force) -> np.ndarray:
    """A load vector (N, one entry per degree of freedom, for ``LoadCase``) that spreads ``force`` (N, a vector
    of 2 components on a 2D grid and 3 on a 3D grid) over ``nodes`` (distinct node indices).

    When the nodes lie on one face of the grid and every one of them is a corner of an element face (on a 2D
    grid an element side) of that face whose corners are all among them, they cover a patch of such faces,
    and the load is the consistent nodal load of a uniform traction on the patch: each of its n faces carries
    force / n, split equally among its corners (a quarter each in 3D, a half in 2D). A node then takes one
    share for every patch face it bounds: an interior node of a rectangular patch four quarters, a node on its
    edge two, a corner one. Otherwise (a single node, nodes along a line on a 3D grid, nodes off the faces)
    every node takes force / (number of nodes).

    The shares add up to one, and the vector sums to ``force`` exactly, component by component, summed with
    ``math.fsum``: should rounding leave a residue, the largest entry of that component takes it, a change of
    a unit or so in its last place.
    """
    dims = _dims(grid)
    nodes = _nodes(grid, nodes)
    if np.unique(nodes).size != nodes.size:
        raise InvalidValue("load takes each node once; the nodes given repeat some")
    values = np.asarray(force)
    if values.shape != (len(dims),) or not np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_:
        raise InvalidValue(f"the force on a {len(dims)}D grid is a vector of {len(dims)} numbers (N); got {force!r}")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise InvalidValue(f"the force must be finite; got {force!r}")
    per_node = len(dims)
    loads = np.zeros(grid.n_dofs)
    patch = _patch(grid, nodes)
    for axis, component in enumerate(values):
        if component == 0.0:
            continue
        if patch is not None:
            share = component / patch.shape[0] / patch.shape[1]
            np.add.at(loads, per_node * patch.ravel() + axis, share)
        else:
            loads[per_node * nodes + axis] += component / nodes.size
        # Rounding may leave the sum a unit or so in its last place off the force: the largest entry takes the
        # residue, then single steps of its own last place while the sum is still off (as after a tie).
        entries = per_node * nodes + axis
        largest = entries[np.argmax(np.abs(loads[entries]))]
        loads[largest] += component - math.fsum(loads[entries])
        for _ in range(64):
            residue = component - math.fsum(loads[entries])
            if residue == 0.0:
                break
            loads[largest] = np.nextafter(loads[largest], math.copysign(math.inf, residue))
    return loads


__all__ = [
    "FACES_2D",
    "FACES_3D",
    "TOLERANCE_IN_ELEMENTS",
    "dofs",
    "elements_in",
    "face_nodes",
    "held",
    "load",
    "nodes_in",
]
