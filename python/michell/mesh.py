"""Closed triangle meshes in mm: ``Mesh``, STL files, and a few bodies made from their dimensions.

- ``Mesh(vertices, triangles)``: vertices as an (n, 3) array in mm, triangles as an (m, 3) array of vertex
  indices, both kept read-only; ``volume_mm3``, ``area_mm2`` and ``bounds``. ``Mesh.closed()`` returns the
  mesh ready for the voxeliser or refuses it with the numbers of what is wrong.
- ``read_stl(path, merge_mm=0.0)`` reads a binary or an ASCII STL file and joins equal vertices;
  ``write_stl(path, mesh)`` writes a binary one.
- ``box``, ``sphere``, ``cylinder`` and ``torus`` make closed, outward-facing meshes. Each docstring states
  the exact volume of the faceted body (not of the smooth shape it approximates): that is the volume a
  voxelisation of it converges to.

A triangle's corners run counter-clockwise seen from outside the body, so its normal (b - a) x (c - a)
points out of it. Refused arguments and meshes raise ``EngineError`` with code ``invalid_input``
(``michell.errors``).
"""

from __future__ import annotations

import math
import numbers
import re
from pathlib import Path

import numpy as np

from michell.errors import InvalidType, InvalidValue

# A triangle counts as having no area when twice its area is at most this share of its longest side squared.
ZERO_AREA = 1e-12

# The 50 bytes of one triangle of a binary STL file, after its 80-byte header and its 4-byte count.
_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
_STL_HEADER = b"binary STL written by michell".ljust(80, b" ")


def _read_only(values: np.ndarray) -> np.ndarray:
    values = np.array(values, copy=True)
    values.flags.writeable = False
    return values


class Mesh:
    """A triangle mesh in mm: ``vertices`` (n, 3), float64, and ``triangles`` (m, 3), whole-number indices
    into the vertices (int64). Both are copied and kept read-only, so a mesh never changes once made.

    ``volume_mm3`` is the signed volume the triangles enclose (positive when they face outward),
    ``area_mm2`` the area of the surface and ``bounds`` the corners (low, high) of the box around the
    vertices. ``closed()`` checks that the mesh bounds a body and returns it facing outward.
    """

    def __init__(self, vertices, triangles):
        v = np.asarray(vertices)
        if v.dtype == np.bool_ or not np.issubdtype(v.dtype, np.number):
            raise InvalidType(f"the vertices of a mesh are numbers (mm); got an array of {v.dtype}")
        if v.ndim != 2 or v.shape[1] != 3:
            raise InvalidValue(f"the vertices of a mesh are an array of shape (n, 3); got shape {v.shape}")
        v = v.astype(np.float64) + 0.0  # + 0.0 turns -0.0 into 0.0
        if not np.all(np.isfinite(v)):
            bad = int(np.flatnonzero(~np.all(np.isfinite(v), axis=1))[0])
            raise InvalidValue(f"vertex {bad} of the mesh is not finite", {"vertex": bad})
        t = np.asarray(triangles)
        if t.size == 0:
            t = t.reshape(0, 3).astype(np.int64)
        if t.dtype == np.bool_ or not np.issubdtype(t.dtype, np.integer):
            raise InvalidType(f"the triangles of a mesh are whole-number vertex indices; got an array of {t.dtype}")
        if t.ndim != 2 or t.shape[1] != 3:
            raise InvalidValue(f"the triangles of a mesh are an array of shape (m, 3); got shape {t.shape}")
        if t.size and (t.min() < 0 or t.max() >= len(v)):
            bad = int(np.flatnonzero(np.any((t < 0) | (t >= len(v)), axis=1))[0])
            raise InvalidValue(
                f"triangle {bad} names a vertex the mesh does not have (it has {len(v)})",
                {"triangle": bad, "vertices": len(v)},
            )
        self._vertices = _read_only(v)
        self._triangles = _read_only(t.astype(np.int64))

    @property
    def vertices(self) -> np.ndarray:
        """The vertices, (n, 3), mm; read-only."""
        return self._vertices

    @property
    def triangles(self) -> np.ndarray:
        """The triangles, (m, 3) vertex indices; read-only."""
        return self._triangles

    def _corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v, t = self._vertices, self._triangles
        return v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]

    @property
    def volume_mm3(self) -> float:
        """The signed volume enclosed by the triangles, mm^3: the sum of the tetrahedra each triangle makes
        with the centre of the bounding box, added with ``math.fsum``. It is exact for the faceted body up
        to rounding, positive when the triangles face outward."""
        if len(self._triangles) == 0:
            return 0.0
        low, high = self.bounds
        centre = (low + high) / 2.0
        a, b, c = (corner - centre for corner in self._corners())
        return math.fsum(np.einsum("ij,ij->i", a, np.cross(b, c)).tolist()) / 6.0

    @property
    def area_mm2(self) -> float:
        """The area of the surface, mm^2."""
        if len(self._triangles) == 0:
            return 0.0
        a, b, c = self._corners()
        return math.fsum(np.linalg.norm(np.cross(b - a, c - a), axis=1).tolist()) / 2.0

    @property
    def bounds(self) -> np.ndarray:
        """The box around the vertices: row 0 its low corner, row 1 its high corner, mm (a new array)."""
        if len(self._vertices) == 0:
            raise InvalidValue("a mesh without vertices has no bounds")
        return np.array([self._vertices.min(axis=0), self._vertices.max(axis=0)])

    def defects(self) -> dict[str, int]:
        """The counts that ``closed()`` checks: ``open_edges`` (edges on one triangle only),
        ``edges_on_more_than_two`` (edges shared by three or more triangles), ``edges_same_direction`` (edges
        whose two triangles run along them the same way, so that one of the two faces the wrong way) and
        ``zero_area_triangles`` (triangles with a repeated corner or with twice their area at most
        ``ZERO_AREA`` times their longest side squared)."""
        t = self._triangles
        n = max(len(self._vertices), 1)
        directed = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        low, high = directed.min(axis=1), directed.max(axis=1)
        _, undirected_count = np.unique(low * n + high, return_counts=True)
        directed_keys, directed_count = np.unique(directed[:, 0] * n + directed[:, 1], return_counts=True)
        # An edge on exactly two triangles runs the same way in both when its directed key occurs twice.
        reverse = (directed_keys % n) * n + directed_keys // n
        reverse_present = np.isin(reverse, directed_keys)
        same_direction = int(np.count_nonzero((directed_count == 2) & ~reverse_present))
        a, b, c = self._corners()
        twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        longest = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                   np.linalg.norm(a - c, axis=1)]), axis=0) if len(t) else np.zeros(0)  # fmt: skip
        repeated = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
        zero_area = repeated | (twice_area <= ZERO_AREA * longest**2)
        return {
            "open_edges": int(np.count_nonzero(undirected_count == 1)),
            "edges_on_more_than_two": int(np.count_nonzero(undirected_count > 2)),
            "edges_same_direction": same_direction,
            "zero_area_triangles": int(np.count_nonzero(zero_area)),
        }

    def closed(self) -> Mesh:
        """The mesh, checked to bound a body: every edge on exactly two triangles that run along it in
        opposite directions, and no triangle without area (``defects()``). It is refused with
        ``InvalidValue`` and those counts otherwise; nothing is repaired, with one exception: a closed mesh
        whose volume is negative (its triangles all face inward, as some programs write STL) is returned
        turned inside out. A closed mesh that encloses no volume is refused as well."""
        if len(self._triangles) == 0:
            raise InvalidValue("the mesh has no triangles, so it bounds no body")
        found = self.defects()
        if any(found.values()):
            listed = ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in found.items() if count)
            raise InvalidValue(f"the mesh is not closed: {listed}", found)
        volume = self.volume_mm3
        low, high = self.bounds
        if abs(volume) <= 1e-12 * float(np.prod(high - low)):
            raise InvalidValue(f"the mesh is closed but encloses no volume ({volume:g} mm3)", {"volume_mm3": volume})
        if volume < 0.0:
            return Mesh(self._vertices, self._triangles[:, [0, 2, 1]])
        return self

    def __repr__(self) -> str:
        return f"Mesh({len(self._vertices)} vertices, {len(self._triangles)} triangles)"


# ---------------------------------------------------------------------------------------------------------
# STL files


def read_stl(path, merge_mm: float = 0.0) -> Mesh:
    """The mesh of an STL file, binary or ASCII.

    The file is binary when its size is 84 + 50 n bytes, with n the count stored in bytes 80 to 83, even if
    its header starts with "solid" (some programs write that); otherwise it is ASCII when it starts with
    "solid" and holds "facet"; anything else is refused. An STL file stores every triangle with its own
    three corners, so corners with exactly equal coordinates are joined into one vertex. With ``merge_mm``
    above 0, corners closer than that (in mm) are joined too, in chains: each group takes the coordinates
    of its first corner in the file. The mesh is returned as read; ``closed()`` checks it.
    """
    data = Path(path).read_bytes()
    tolerance = _non_negative(merge_mm, "merge_mm")
    if len(data) >= 84 and len(data) == 84 + 50 * int.from_bytes(data[80:84], "little"):
        count = int.from_bytes(data[80:84], "little")
        records = np.frombuffer(data, dtype=_STL_RECORD, count=count, offset=84)
        corners = records["corners"].astype(np.float64).reshape(-1, 3)
    elif data.startswith(b"solid") and b"facet" in data:
        corners = _ascii_corners(data.decode("latin-1"), path)
    else:
        raise InvalidValue(
            f"{path} is not an STL file: its size does not match a binary STL and it does not start as an ASCII one"
        )
    if not np.all(np.isfinite(corners)):
        raise InvalidValue(f"{path} holds coordinates that are not finite numbers")
    vertices, triangles = _join_corners(corners + 0.0, tolerance)
    return Mesh(vertices, triangles)


_NUMBER = r"([-+0-9.eEinfINFaA]+)"
_VERTEX = re.compile(r"\bvertex\s+" + _NUMBER + r"\s+" + _NUMBER + r"\s+" + _NUMBER, re.IGNORECASE)
_FACET = re.compile(r"\bfacet\b", re.IGNORECASE)


def _ascii_corners(text: str, path) -> np.ndarray:
    found = _VERTEX.findall(text)
    facets = len(_FACET.findall(text))
    if len(found) != 3 * facets:
        raise InvalidValue(
            f"{path}: an ASCII STL file has three vertices per facet; found {len(found)} vertices in {facets} facets"
        )
    try:
        return np.array(found, dtype=np.float64).reshape(-1, 3)
    except ValueError as error:
        raise InvalidValue(f"{path}: a vertex coordinate is not a number ({error})") from None


def _join_corners(corners: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (in the order of their first corner) and triangles from the corners of an STL file."""
    unique, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    vertices = unique[order]
    index = rank[inverse.ravel()]
    if tolerance > 0.0 and len(vertices) > 1:
        group = _close_groups(vertices, tolerance)
        kept, renumbered = np.unique(group, return_inverse=True)
        vertices = vertices[kept]
        index = renumbered.ravel()[index]
    return vertices, index.reshape(-1, 3)


# The cell offsets that, with the cell itself, reach each pair of neighbouring cells once.
_HALF_NEIGHBOURS = [o for o in np.ndindex(3, 3, 3) if (o[0] - 1, o[1] - 1, o[2] - 1) > (0, 0, 0)]


def _close_groups(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """For each vertex the smallest index of the vertices joined to it by chains of steps no longer than
    ``tolerance``. Vertices closer than the tolerance lie in the same or in neighbouring cells of side
    ``tolerance``, so only those pairs are measured."""
    cells = np.floor((vertices - vertices.min(axis=0)) / tolerance).astype(np.int64)
    keys, cell_of = np.unique(cells, axis=0, return_inverse=True)
    cell_of = cell_of.ravel()
    members = np.argsort(cell_of, kind="stable")
    counts = np.bincount(cell_of, minlength=len(keys))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first_parts, second_parts = [], []
    offsets = [np.zeros(3, dtype=np.int64)] + [np.array(o, dtype=np.int64) - 1 for o in _HALF_NEIGHBOURS]
    for offset in offsets:
        target = keys + offset
        joint, where = np.unique(np.concatenate([keys, target]), axis=0, return_inverse=True)
        where = where.ravel()
        cell_at = np.full(len(joint), -1, dtype=np.int64)
        cell_at[where[: len(keys)]] = np.arange(len(keys))
        found = cell_at[where[len(keys) :]]
        cells_a = np.flatnonzero(found >= 0)
        cells_b = found[cells_a]
        pairs = counts[cells_a] * counts[cells_b]
        if pairs.sum() == 0:
            continue
        owner = np.repeat(np.arange(len(cells_a)), pairs)
        within = np.arange(int(pairs.sum())) - np.repeat(np.cumsum(pairs) - pairs, pairs)
        width = counts[cells_b][owner]
        first = members[starts[cells_a][owner] + within // width]
        second = members[starts[cells_b][owner] + within % width]
        keep = np.linalg.norm(vertices[first] - vertices[second], axis=1) <= tolerance
        if not offset.any():
            keep &= first < second
        first_parts.append(first[keep])
        second_parts.append(second[keep])
    first = np.concatenate(first_parts) if first_parts else np.zeros(0, dtype=np.int64)
    second = np.concatenate(second_parts) if second_parts else np.zeros(0, dtype=np.int64)
    return smallest_joined(len(vertices), first, second)


def smallest_joined(n: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """For each of ``n`` items the smallest item joined to it through the pairs (first[i], second[i]): the
    groups of a graph, labelled by their smallest member. Roots are hooked under the smaller root of each
    pair and the labels then followed to their root (``label = label[label]``) until nothing changes, so
    the number of rounds grows with the logarithm of a group's size, not with its length."""
    label = np.arange(n, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    while first.size:
        a, b = label[first], label[second]
        if np.array_equal(a, b):
            break
        np.minimum.at(label, np.maximum(a, b), np.minimum(a, b))
        while True:
            followed = label[label]
            if np.array_equal(followed, label):
                break
            label = followed
    return label


def write_stl(path, mesh: Mesh) -> None:
    """Writes ``mesh`` as a binary STL file (coordinates rounded to float32, as the format stores them), with
    the unit normal of each triangle (zero for a triangle without area) and a header that does not start
    with "solid"."""
    if not isinstance(mesh, Mesh):
        raise InvalidType(f"write_stl writes a michell.mesh.Mesh; got {type(mesh).__name__}")
    a, b, c = mesh._corners()
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal), where=length[:, None] > 0)
    records = np.zeros(len(mesh.triangles), dtype=_STL_RECORD)
    records["normal"] = normal
    records["corners"] = mesh.vertices[mesh.triangles]
    Path(path).write_bytes(_STL_HEADER + len(records).to_bytes(4, "little") + records.tobytes())


# ---------------------------------------------------------------------------------------------------------
# Bodies made from their dimensions

# For each axis of revolution, the two axes across it, so that first x second = the axis.
_ACROSS = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


def _point(value, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.shape != (3,) or values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
        raise InvalidValue(f"{name} is a point of three numbers (mm); got {value!r}")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise InvalidValue(f"{name} must be finite; got {value!r}")
    return values


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise InvalidValue(f"{name} must be a positive finite number; got {value!r}")
    return float(value)


def _non_negative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
        raise InvalidValue(f"{name} must be a finite number of at least 0; got {value!r}")
    return float(value)


def _count(value, name: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < least:
        raise InvalidValue(f"{name} must be a whole number of at least {least}; got {value!r}")
    return int(value)


def _axis(value) -> str:
    if value not in _ACROSS:
        raise InvalidValue(f"the axis is 'x', 'y' or 'z'; got {value!r}")
    return value


def box(low, high) -> Mesh:
    """The box from corner ``low`` to corner ``high`` (mm, each coordinate of ``high`` above that of
    ``low``): 8 vertices, 12 triangles. Its volume is the product of its sides."""
    low, high = _point(low, "low"), _point(high, "high")
    if not np.all(high > low):
        raise InvalidValue(f"each coordinate of high must be above that of low; got {low.tolist()}, {high.tolist()}")
    corners = np.array([[(low, high)[i][0], (low, high)[j][1], (low, high)[k][2]]
                        for k in (0, 1) for j in (0, 1) for i in (0, 1)])  # fmt: skip
    # Corner (i, j, k) is vertex i + 2 j + 4 k; each face as two triangles, counter-clockwise from outside.
    faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]]
    triangles = [[a, b, c] for a, b, c, d in faces] + [[a, c, d] for a, b, c, d in faces]
    return Mesh(corners, triangles)


def sphere(centre, radius: float, segments: int = 32, rings: int = 16) -> Mesh:
    """A sphere of ``radius`` (mm) around ``centre`` with its poles along z: ``segments`` meridians (at
    angles 2 pi j / segments from the x axis) and ``rings`` bands from pole to pole (vertices at polar
    angles pi i / rings). The bands are planar trapezoids, cut in two; the caps are triangles.

    The faceted body is a stack of frusta of regular pyramids, so its volume is, with n = segments,
    rho_i = radius sin(pi i / rings) and z_i = radius cos(pi i / rings),

        V = (n / 6) sin(2 pi / n) sum_{i=0}^{rings-1} (z_i - z_{i+1}) (rho_i^2 + rho_i rho_{i+1} + rho_{i+1}^2).

    With ``segments`` a multiple of 4 no triangle is parallel to the x axis.
    """
    centre, radius = _point(centre, "centre"), _positive(radius, "radius")
    n, rings = _count(segments, "segments", 3), _count(rings, "rings", 2)
    phi = 2.0 * np.pi * np.arange(n) / n
    theta = np.pi * np.arange(1, rings) / rings
    ring_points = np.stack([np.outer(np.sin(theta), np.cos(phi)), np.outer(np.sin(theta), np.sin(phi)),
                            np.outer(np.cos(theta), np.ones(n))], axis=-1).reshape(-1, 3)  # fmt: skip
    vertices = np.vstack([[0.0, 0.0, 1.0], ring_points, [0.0, 0.0, -1.0]]) * radius + centre
    south = len(vertices) - 1

    def at(i: int, j: int) -> int:  # vertex of ring i (1 .. rings-1) and meridian j
        return 1 + (i - 1) * n + j % n

    triangles = [[0, at(1, j), at(1, j + 1)] for j in range(n)]
    for i in range(1, rings - 1):
        for j in range(n):
            a, b, c, d = at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)
            triangles += [[a, b, c], [a, c, d]]
    triangles += [[south, at(rings - 1, j + 1), at(rings - 1, j)] for j in range(n)]
    return Mesh(vertices, triangles)


def cylinder(centre, radius: float, length: float, axis: str = "z", segments: int = 32) -> Mesh:
    """A cylinder of ``radius`` and ``length`` (mm) around ``centre``, its axis along ``axis`` ("x", "y" or
    "z"): a prism on a regular polygon of ``segments`` corners inscribed in the circle, closed by two flat
    caps. Across the axis the corners lie at angles 2 pi j / segments from the first axis of the pair
    (y, z), (z, x) or (x, y) for an axis x, y or z, towards the second.

    Its volume is that of the prism: V = (segments / 2) sin(2 pi / segments) radius^2 length. With an even
    number of segments and the axis along y, no side triangle is parallel to the x axis (the caps are).
    """
    centre, radius = _point(centre, "centre"), _positive(radius, "radius")
    length, n, axis = _positive(length, "length"), _count(segments, "segments", 3), _axis(axis)
    first, second = _ACROSS[axis]
    along = "xyz".index(axis)
    angle = 2.0 * np.pi * np.arange(n) / n
    ring = np.zeros((n, 3))
    ring[:, first], ring[:, second] = radius * np.cos(angle), radius * np.sin(angle)
    half = np.zeros(3)
    half[along] = length / 2.0
    vertices = np.vstack([ring - half, ring + half, [-half, half]]) + centre
    bottom_centre, top_centre = 2 * n, 2 * n + 1
    triangles = []
    for j in range(n):
        b0, b1, t0, t1 = j, (j + 1) % n, n + j, n + (j + 1) % n
        triangles += [[b0, b1, t1], [b0, t1, t0], [top_centre, t0, t1], [bottom_centre, b1, b0]]
    return Mesh(vertices, triangles)


def torus(
    centre, major_radius: float, minor_radius: float, axis: str = "z", segments: int = 48, sides: int = 24
) -> Mesh:
    """A torus around ``centre`` with its axis of revolution along ``axis``: a regular polygon of ``sides``
    corners inscribed in the tube's circle of ``minor_radius`` (mm), at ``major_radius`` from the axis (it
    must be the larger), carried round the axis in ``segments`` steps. Round the axis the corners lie at
    angles 2 pi j / segments from the first axis of the pair (y, z), (z, x) or (x, y) for an axis x, y or z;
    round the tube at angles 2 pi i / sides from the outward direction. Every face is a planar trapezoid,
    cut in two.

    The faceted body is the cross-section polygon swept linearly between the meridian planes; its volume is,
    with n = segments and m = sides,

        V = n sin(2 pi / n) * major_radius * (m / 2) sin(2 pi / m) minor_radius^2,

    the polygon's area times its centre's distance from the axis times n sin(2 pi / n). With both counts
    multiples of 4 and the axis along z, no triangle is parallel to the x axis.
    """
    centre = _point(centre, "centre")
    major, minor = _positive(major_radius, "major_radius"), _positive(minor_radius, "minor_radius")
    if minor >= major:
        raise InvalidValue(f"the minor radius must be below the major radius; got {minor:g} and {major:g}")
    n, m, axis = _count(segments, "segments", 3), _count(sides, "sides", 3), _axis(axis)
    first, second = _ACROSS[axis]
    along = "xyz".index(axis)
    phi = 2.0 * np.pi * np.arange(n) / n
    theta = 2.0 * np.pi * np.arange(m) / m
    distance = major + minor * np.cos(theta)  # from the axis, per corner of the tube
    vertices = np.zeros((n, m, 3))
    vertices[:, :, first] = np.outer(np.cos(phi), distance)
    vertices[:, :, second] = np.outer(np.sin(phi), distance)
    vertices[:, :, along] = minor * np.sin(theta)[None, :]
    triangles = []
    for j in range(n):
        for i in range(m):
            a, b = j * m + i, ((j + 1) % n) * m + i
            c, d = ((j + 1) % n) * m + (i + 1) % m, j * m + (i + 1) % m
            triangles += [[a, b, c], [a, c, d]]
    return Mesh(vertices.reshape(-1, 3) + centre, triangles)


__all__ = ["ZERO_AREA", "Mesh", "box", "cylinder", "read_stl", "smallest_joined", "sphere", "torus", "write_stl"]
