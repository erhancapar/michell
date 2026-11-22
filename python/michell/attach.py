"""Supports and loads on the voxel model of a part: where the part is held, where it is loaded, what is held
solid under those places and which faces of the grid the density filter treats as closed.

- ``Support(on, fix=("x", "y", "z"), skin=True)``, ``Load(on, force_n, skin=True)`` and
  ``Case(name, loads, weight=1.0)``: what a support or a load acts on and how.
- ``set_up(model, material, supports, cases, size_mm, *, threads=1, outside=None, ...)``: the ``Problem``,
  the ``KeepSolid`` mask, the density filter's faces and a report, as a ``SetUp``.
- The pieces it is built from, each usable on its own: ``boundary_faces`` (the part's boundary faces),
  ``contact_faces`` (those in contact with a body), ``region_voxels`` (the voxels of a solid region),
  ``traction_loads`` and ``body_force_loads`` (the nodal shares of a uniform traction on faces and of a
  uniform body force on voxels) and ``held_skin`` (the voxels held solid behind contact faces).
- ``LoadOnSupport``: the warning ``set_up`` gives when more than 1 % of a load falls on held degrees of
  freedom.

What a support or a load acts ``on``:

1. A solid region of the part, by its name in the model: a load is a uniform body force over the region's
   voxels (the voxels the region fills by majority that are active), each voxel giving one eighth of its
   share to each of its eight nodes; a support holds every node of those voxels.
2. Any other closed body (a ``Mesh``), or a box given as its low and high corners ``((x0, y0, z0),
   (x1, y1, z1))`` in the meshes' coordinates: a load is a uniform traction over the part's boundary faces
   in contact with the body, each face giving one quarter of its share to each of its four nodes; a support
   holds the nodes of those faces. A boundary face lies between an active voxel and a voxel that is not
   active, or the grid's edge. It is in contact when, seen along the face's normal, the body covers more
   than half of the face within a window two voxel sides deep centred on the face (the boundary voxel and
   the position across it): of 16 rays along the normal through the face, at 1/8, 3/8, 5/8 and 7/8 of its
   sides, more than 8 run inside the body for at least one twentieth of a voxel side within the window.

The rules in full, with what they give in the cases that decide them, are in ``docs/design/geometry.md``.
Refused arguments raise ``EngineError`` with code ``invalid_input`` (``michell.errors``).
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from michell import mesh as _mesh
from michell.errors import InvalidType, InvalidValue, UnknownName
from michell.filters import KeepSolid
from michell.mesh import Mesh
from michell.problem import LoadCase, Material, Problem
from michell.regions import MAJORITY, VoxelModel, occupancy

# The faces of a voxel and of the grid, in this order: side s is normal to axis s // 2 and faces towards
# -1 for even s, +1 for odd s.
FACES = ("x-", "x+", "y-", "y+", "z-", "z+")

# The contact test: RAYS_PER_SIDE x RAYS_PER_SIDE rays along a boundary face's normal, at the centres of the
# face's sub-squares; a ray counts when the body's length on it within the window (the boundary voxel and the
# position across it) is at least RAY_LENGTH voxel sides; the face is in contact when more than CONTACT_RAYS
# of its rays count.
RAYS_PER_SIDE = 4
RAY_LENGTH = 1.0 / 20.0
CONTACT_RAYS = 8

# The fine grid of the contact test is computed in pieces of at most this many fine voxels (float64 each).
FINE_VOXELS_PER_PIECE = 1 << 22

# The length on a ray is compared with RAY_LENGTH with this relative slack, so that a marker exactly a
# twentieth of a voxel thick, given in decimal millimetres, counts.
LENGTH_SLACK = 1e-9

# A load's share on held degrees of freedom: a warning above the first, a refusal above the second.
DROPPED_WARNING = 0.01
DROPPED_REFUSAL = 0.5

# The held skin reaches half the member size behind a contact face, and at least this many voxel sides.
SKIN_LEAST_VOXELS = 2

_AXES = "xyz"


class LoadOnSupport(UserWarning):
    """Issued by ``set_up`` when more than 1 % of a load falls on degrees of freedom that a support holds.
    Those shares act on the support directly and do not load the part; the message names the load and the
    share. Above one half the load is refused instead."""


# ---------------------------------------------------------------------------------------------------------
# What a support or a load acts on


def _box_corners(value) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """``value`` as a box ((x0, y0, z0), (x1, y1, z1)) of finite numbers with each high above its low, or
    None when it does not have the shape of one."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        return None
    corners = []
    for corner in value:
        if not isinstance(corner, (list, tuple, np.ndarray)) or len(corner) != 3:
            return None
        if any(isinstance(c, bool) or not isinstance(c, numbers.Real) for c in corner):
            return None
        corners.append(tuple(float(c) for c in corner))
    low, high = corners
    if not all(math.isfinite(c) for c in low + high):
        raise InvalidValue(f"the corners of a box must be finite numbers (mm); got {value!r}")
    if not all(h > lo for lo, h in zip(low, high)):
        raise InvalidValue(f"each coordinate of a box's high corner must be above that of its low corner; got {value!r}")
    return low, high


def _target(on):
    """``on`` checked: a region name (a non-empty string), a ``Mesh``, or a box as a pair of tuples."""
    if isinstance(on, str):
        if not on:
            raise InvalidValue("a support or a load acts on a solid region named by a non-empty string")
        return on
    if isinstance(on, Mesh):
        return on
    box = _box_corners(on)
    if box is None:
        raise InvalidType(
            "a support or a load acts on a solid region (its name), a michell.Mesh, or a box "
            f"((x0, y0, z0), (x1, y1, z1)) in mm; got {on!r}"
        )
    return box


def _describe(on) -> str:
    if isinstance(on, str):
        return f"the solid region {on!r}"
    if isinstance(on, Mesh):
        low, high = on.bounds
        return f"a body from ({', '.join(f'{c:g}' for c in low)}) to ({', '.join(f'{c:g}' for c in high)}) mm"
    low, high = on
    return f"the box from ({', '.join(f'{c:g}' for c in low)}) to ({', '.join(f'{c:g}' for c in high)}) mm"


def _flag(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise InvalidType(f"{name} is True or False; got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class Support:
    """Where the part is held: ``on`` a solid region (its name in the model), a closed body (a ``Mesh``) or a
    box ((x0, y0, z0), (x1, y1, z1)) in the meshes' coordinates; ``fix``, the directions held (any of
    ``"x"``, ``"y"``, ``"z"``, as a tuple such as ``("z",)`` or a string such as ``"xy"``; all three by
    default); ``skin``, whether the voxels behind a contact patch on design voxels are held solid (see
    ``set_up``)."""

    on: object
    fix: tuple[str, ...] = ("x", "y", "z")
    skin: bool = True

    def __post_init__(self):
        object.__setattr__(self, "on", _target(self.on))
        fix = self.fix
        letters = tuple(fix) if isinstance(fix, (str, list, tuple)) else None
        if not letters or any(d not in _AXES for d in letters) or len(set(letters)) != len(letters):
            raise InvalidValue(f"fix holds distinct directions among 'x', 'y', 'z', such as ('z',); got {fix!r}")
        object.__setattr__(self, "fix", tuple(sorted(letters, key=_AXES.index)))
        object.__setattr__(self, "skin", _flag(self.skin, "skin"))


@dataclass(frozen=True)
class Load:
    """A force ``force_n`` (three numbers, N) on ``on``: a solid region (its name in the model; a uniform
    body force over its voxels), a closed body (a ``Mesh``) or a box ((x0, y0, z0), (x1, y1, z1)) in the
    meshes' coordinates (a uniform traction over the part's faces in contact with it); ``skin`` as for a
    ``Support``."""

    on: object
    force_n: tuple[float, float, float]
    skin: bool = True

    def __post_init__(self):
        object.__setattr__(self, "on", _target(self.on))
        values = self.force_n
        if (
            not isinstance(values, (list, tuple, np.ndarray))
            or len(values) != 3
            or any(isinstance(v, bool) or not isinstance(v, numbers.Real) for v in values)
            or not all(math.isfinite(v) for v in values)
        ):
            raise InvalidValue(f"force_n is three finite numbers (N); got {values!r}")
        object.__setattr__(self, "force_n", tuple(float(v) for v in values))
        object.__setattr__(self, "skin", _flag(self.skin, "skin"))


@dataclass(frozen=True)
class Case:
    """A load case given by what its loads act on: a ``name``, its ``loads`` (``Load`` objects, acting
    together) and its ``weight`` in the weighted sum of the cases. ``set_up`` turns it into a ``LoadCase``
    (one force per degree of freedom)."""

    name: str
    loads: tuple[Load, ...]
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise InvalidValue(f"a load case is named by a non-empty string; got {self.name!r}")
        if isinstance(self.loads, Load) or not isinstance(self.loads, (list, tuple)) or not self.loads:
            raise InvalidValue(f"load case {self.name!r} holds a non-empty list of michell.attach.Load")
        for load in self.loads:
            if not isinstance(load, Load):
                raise InvalidType(f"load case {self.name!r} holds michell.attach.Load objects; got {type(load).__name__}")
        object.__setattr__(self, "loads", tuple(self.loads))
        if isinstance(self.weight, bool) or not isinstance(self.weight, numbers.Real):
            raise InvalidValue(f"the weight of load case {self.name!r} is a number; got {self.weight!r}")
        if not (math.isfinite(self.weight) and self.weight >= 0):
            raise InvalidValue(f"the weight of load case {self.name!r} must be a finite number at least 0")
        object.__setattr__(self, "weight", float(self.weight))


# ---------------------------------------------------------------------------------------------------------
# Faces


def _model(model) -> VoxelModel:
    if not isinstance(model, VoxelModel):
        raise InvalidType(f"a michell.regions.VoxelModel is needed; got {type(model).__name__}")
    return model


def _voxel_ijk(voxels: np.ndarray, counts) -> np.ndarray:
    nx, ny, _ = counts
    voxels = np.asarray(voxels, dtype=np.int64)
    return np.stack([voxels % nx, (voxels // nx) % ny, voxels // (nx * ny)], axis=1)


def _node_index(ijk: np.ndarray, counts) -> np.ndarray:
    nx, ny, _ = counts
    return ijk[..., 0] + (nx + 1) * (ijk[..., 1] + (ny + 1) * ijk[..., 2])


@dataclass(frozen=True, eq=False)
class Faces:
    """Voxel faces of a grid of ``counts`` voxels of side ``voxel_mm``: face f is side ``sides[f]`` (an index
    into ``FACES``: 0 for x-, 1 for x+, ..., 5 for z+) of voxel ``voxels[f]``, ordered by voxel, then side.

    ``nodes()`` gives the four corner nodes of every face, ``centres_mm()`` their centres in the grid's
    coordinates (``michell.select``'s: the grid's corner at the origin), ``across()`` the whole-number
    position (i, j, k) of the voxel across each face (-1 or n beyond the grid's edge), ``keys()`` one number
    per face (6 voxel + side) to compare sets of faces, and ``on_grid_face(name)`` which faces lie on a face
    of the grid's box. ``area_mm2`` is their area."""

    counts: tuple[int, int, int]
    voxel_mm: float
    voxels: np.ndarray
    sides: np.ndarray

    def __len__(self) -> int:
        return int(self.voxels.size)

    @property
    def area_mm2(self) -> float:
        return len(self) * self.voxel_mm**2

    def keys(self) -> np.ndarray:
        return 6 * self.voxels + self.sides

    def subset(self, chosen: np.ndarray) -> Faces:
        """The faces where ``chosen`` (a boolean per face) is true."""
        return Faces(self.counts, self.voxel_mm, self.voxels[chosen], self.sides[chosen])

    def _normals(self) -> np.ndarray:
        normals = np.zeros((len(self), 3), dtype=np.int64)
        normals[np.arange(len(self)), self.sides // 2] = np.where(self.sides % 2 == 1, 1, -1)
        return normals

    def across(self) -> np.ndarray:
        return _voxel_ijk(self.voxels, self.counts) + self._normals()

    def centres_mm(self) -> np.ndarray:
        return (_voxel_ijk(self.voxels, self.counts) + 0.5 + 0.5 * self._normals()) * self.voxel_mm

    def nodes(self) -> np.ndarray:
        """The corner nodes of every face, shape (faces, 4), in the node numbering of ``Grid3d``."""
        base = _voxel_ijk(self.voxels, self.counts)
        rows = np.arange(len(self))
        axis = self.sides // 2
        base[rows, axis] += self.sides % 2
        first, second = (axis + 1) % 3, (axis + 2) % 3
        corners = []
        for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
            corner = base.copy()
            corner[rows, first] += d1
            corner[rows, second] += d2
            corners.append(_node_index(corner, self.counts))
        return np.stack(corners, axis=1).reshape(len(self), 4)

    def on_grid_face(self, name: str) -> np.ndarray:
        """A boolean per face: true for the faces that lie on the face ``name`` (``"x-"`` to ``"z+"``) of the
        grid's box."""
        if name not in FACES:
            raise InvalidValue(f"the faces of the grid are {', '.join(FACES)}; got {name!r}")
        side = FACES.index(name)
        axis = side // 2
        position = _voxel_ijk(self.voxels, self.counts)[:, axis]
        return (self.sides == side) & (position == (self.counts[axis] - 1 if side % 2 else 0))


def boundary_faces(model: VoxelModel) -> Faces:
    """The boundary faces of the part: every face between an active voxel and a voxel that is not active, or
    the grid's edge, as ``Faces``."""
    model = _model(model)
    nx, ny, nz = model.counts
    padded = np.zeros((nz + 2, ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = model.active.reshape(nz, ny, nx)
    inner = padded[1:-1, 1:-1, 1:-1]
    voxels, sides = [], []
    for side in range(6):
        step = [0, 0, 0]
        step[2 - side // 2] = 1 if side % 2 else -1  # the array axes run z, y, x
        neighbour = padded[1 + step[0] : nz + 1 + step[0], 1 + step[1] : ny + 1 + step[1], 1 + step[2] : nx + 1 + step[2]]
        found = np.flatnonzero((inner & ~neighbour).ravel())
        voxels.append(found)
        sides.append(np.full(found.size, side, dtype=np.int64))
    voxels, sides = np.concatenate(voxels).astype(np.int64), np.concatenate(sides)
    order = np.lexsort((sides, voxels))
    return Faces(model.counts, model.voxel_mm, voxels[order], sides[order])


@dataclass(frozen=True, eq=False)
class Contact:
    """What ``contact_faces`` returns: ``faces``, the boundary faces in contact with the body; ``boundary``,
    every boundary face of the part; ``rays``, for each face of ``boundary``, how many of its 16 rays run
    inside the body for at least a twentieth of a voxel side within the face's window (more than 8 make
    the face a contact face)."""

    faces: Faces
    boundary: Faces
    rays: np.ndarray


def _body_mesh(body, label: str) -> Mesh:
    if isinstance(body, Mesh):
        try:
            return body.closed()
        except InvalidValue as error:
            raise InvalidValue(f"{label}: {error.message}", error.details) from None
    box = _box_corners(body)
    if box is None:
        raise InvalidType(f"{label} is a michell.Mesh or a box ((x0, y0, z0), (x1, y1, z1)) in mm; got {body!r}")
    return _mesh.box(*box)


def _window_rays(model: VoxelModel, body: Mesh, axis: int, threads: int) -> np.ndarray:
    """For the normal ``axis`` (0, 1, 2 for x, y, z): an array over the extended grid (``counts`` + 2 voxels,
    indexed [K, J, I] with I along ``axis``, J and K along the next two axes in cyclic order) whose entry at
    I is the number of the 16 rays through the face between positions I and I + 1 along ``axis`` that run
    inside ``body`` for at least RAY_LENGTH voxel sides within those two positions.

    The voxeliser's rays run along x, so the body is handed to it with its coordinates turned cyclically
    (which keeps its triangles facing outward) so that ``axis`` becomes x, on a grid four times finer than
    the model's with one ray per fine column: each fine column is then one of the 16 rays of the coarse
    column, and each fine voxel's occupancy times its side is the body's exact length on that ray inside it.
    Only the box around the body (widened by one position along ``axis`` for the window) is computed, in
    pieces of at most FINE_VOXELS_PER_PIECE fine voxels."""
    order = [axis, (axis + 1) % 3, (axis + 2) % 3]
    h = model.voxel_mm
    extended = np.array(model.counts)[order] + 2
    origin = (np.array(model.origin_mm) - h)[order]
    turned = Mesh(np.ascontiguousarray(body.vertices[:, order]), body.triangles)
    low, high = turned.bounds
    first = np.clip(np.floor((low - origin) / h).astype(np.int64), 0, extended)
    last = np.clip(np.ceil((high - origin) / h).astype(np.int64), 0, extended)
    first[0], last[0] = max(first[0] - 1, 0), min(last[0] + 1, extended[0])
    rays = np.zeros((extended[2], extended[1], extended[0]), dtype=np.int8)
    if np.any(last - first < 1) or last[0] - first[0] < 2:
        return rays
    n = RAYS_PER_SIDE
    along = int(last[0] - first[0])
    rows = max(1, min(int(last[1] - first[1]), FINE_VOXELS_PER_PIECE // (n**3 * along)))
    layers = max(1, FINE_VOXELS_PER_PIECE // (n**3 * along * rows))
    least = n * RAY_LENGTH * (1.0 - LENGTH_SLACK)  # in fine voxel sides
    for k0 in range(int(first[2]), int(last[2]), layers):
        k1 = min(k0 + layers, int(last[2]))
        for j0 in range(int(first[1]), int(last[1]), rows):
            j1 = min(j0 + rows, int(last[1]))
            corner = origin + np.array([first[0], j0, k0]) * h
            fine = occupancy((n * along, n * (j1 - j0), n * (k1 - k0)), h / n, tuple(corner), [turned],
                             rays_per_side=1, threads=threads)  # fmt: skip
            # The length on each ray inside each coarse position, then inside each window of two positions.
            per_position = fine.reshape(n * (k1 - k0), n * (j1 - j0), along, n).sum(axis=3)
            window = per_position[:, :, :-1] + per_position[:, :, 1:]
            counted = (window >= least).reshape(k1 - k0, n, j1 - j0, n, along - 1).sum(axis=(1, 3))
            rays[k0:k1, j0:j1, int(first[0]) : int(last[0]) - 1] = counted
    return rays


def _contact(model: VoxelModel, body: Mesh, boundary: Faces, threads: int) -> Contact:
    counted = np.zeros(len(boundary), dtype=np.int64)
    ijk = _voxel_ijk(boundary.voxels, model.counts) + 1  # in the extended grid
    for axis in range(3):
        here = boundary.sides // 2 == axis
        if not here.any():
            continue
        rays = _window_rays(model, body, axis, threads)
        position = ijk[here][:, [axis, (axis + 1) % 3, (axis + 2) % 3]]
        # A face towards + lies between its voxel and the next position; one towards - between the one before.
        start = position[:, 0] - (boundary.sides[here] % 2 == 0)
        counted[here] = rays[position[:, 2], position[:, 1], start]
    return Contact(boundary.subset(counted > CONTACT_RAYS), boundary, counted)


def contact_faces(model: VoxelModel, body, *, threads: int | None = None) -> Contact:
    """The boundary faces of the part in contact with ``body`` (a closed ``Mesh``, or a box ((x0, y0, z0),
    (x1, y1, z1)) in the meshes' coordinates): those that the body covers by more than half, seen along the
    face's normal within a window two voxel sides deep centred on the face. Of 16 rays along the normal
    through the face, at 1/8, 3/8, 5/8 and 7/8 of its sides, more than 8 must run inside the body for at
    least one twentieth of a voxel side within the window (the boundary voxel and the position across it).
    The lengths come from the voxeliser, exact along each ray. On ``threads`` threads (the model's by
    default). Returns a ``Contact``, whose ``faces`` may be empty."""
    model = _model(model)
    return _contact(model, _body_mesh(body, "the body"), boundary_faces(model), _threads(threads, model.threads))


def region_voxels(model: VoxelModel, name: str) -> np.ndarray:
    """The voxels of the solid region ``name``: the active voxels it fills by majority, as a boolean mask
    with one entry per voxel. A name that is not a solid region of the model is refused."""
    model = _model(model)
    roles = {row.name: row.role for row in model.volumes if row.role in ("design", "solid", "void")}
    if name not in roles:
        solids = [n for n, role in roles.items() if role == "solid"]
        raise UnknownName(f"the model has no region {name!r}; its solid regions are {solids}")
    if roles[name] != "solid":
        raise InvalidValue(
            f"{name!r} is the model's {roles[name]} region; a support or a load acts on a solid region, a body or a box"
        )
    return (np.asarray(model.region_occupancy[name]) >= MAJORITY) & model.active


# ---------------------------------------------------------------------------------------------------------
# Nodal shares


def _force(force_n) -> np.ndarray:
    values = np.asarray(force_n)
    if values.shape != (3,) or values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
        raise InvalidValue(f"a force is three numbers (N); got {force_n!r}")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise InvalidValue(f"a force must be finite; got {force_n!r}")
    return values


def _spread(n_nodes: int, corners: np.ndarray, force: np.ndarray) -> np.ndarray:
    """The load vector of ``force`` spread over the ``corners`` (one row of nodes per face or voxel): each row
    carries force / rows, split equally among its nodes, as ``michell.select.load`` does on a patch of a
    grid face. The components then sum to ``force`` exactly (with ``math.fsum``): should rounding leave a
    residue, the largest entry of that component takes it."""
    loads = np.zeros(3 * n_nodes)
    nodes = np.unique(corners)
    rows, per_row = corners.shape
    for axis, component in enumerate(force):
        if component == 0.0:
            continue
        share = component / rows / per_row
        np.add.at(loads, 3 * corners.ravel() + axis, share)
        entries = 3 * nodes + axis
        largest = entries[np.argmax(np.abs(loads[entries]))]
        loads[largest] += component - math.fsum(loads[entries])
        for _ in range(64):
            residue = component - math.fsum(loads[entries])
            if residue == 0.0:
                break
            loads[largest] = np.nextafter(loads[largest], math.copysign(math.inf, residue))
    return loads


def _grid_counts(grid) -> tuple[int, int, int]:
    try:
        return int(grid.nx), int(grid.ny), int(grid.nz)
    except AttributeError:
        raise InvalidType(f"a michell.Grid3d is needed; got {type(grid).__name__}") from None


def traction_loads(grid, faces: Faces, force_n) -> np.ndarray:
    """The load vector (N, one entry per degree of freedom of ``grid``) of ``force_n`` as a uniform traction
    over ``faces``: each face carries force / faces, one quarter of it at each of its four corners. The
    components sum to the force exactly (``math.fsum``)."""
    if not isinstance(faces, Faces):
        raise InvalidType(f"faces are michell.attach.Faces; got {type(faces).__name__}")
    if faces.counts != _grid_counts(grid):
        raise InvalidValue(f"the faces belong to a grid of {faces.counts} voxels, not of {_grid_counts(grid)}")
    if len(faces) == 0:
        raise InvalidValue("a traction needs at least one face")
    return _spread(int(grid.n_nodes), faces.nodes(), _force(force_n))


def _voxel_corners(voxels: np.ndarray, counts) -> np.ndarray:
    ijk = _voxel_ijk(voxels, counts)
    corners = [_node_index(ijk + np.array([a, b, c]), counts) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
    return np.stack(corners, axis=1)


def _voxel_list(grid, voxels) -> np.ndarray:
    values = np.asarray(voxels)
    n = int(grid.n_elements)
    if values.dtype == np.bool_:
        if values.shape != (n,):
            raise InvalidValue(f"a voxel mask needs one entry per voxel ({n}); got shape {values.shape}")
        return np.flatnonzero(values).astype(np.int64)
    if values.ndim != 1 or not (values.size == 0 or np.issubdtype(values.dtype, np.integer)):
        raise InvalidValue("voxels are a boolean mask or a flat array of voxel indices")
    if values.size and (values.min() < 0 or values.max() >= n):
        raise InvalidValue(f"a voxel index lies outside the grid (0 to {n - 1})")
    return np.unique(values.astype(np.int64))


def body_force_loads(grid, voxels, force_n) -> np.ndarray:
    """The load vector (N, one entry per degree of freedom of ``grid``) of ``force_n`` as a uniform body force
    over ``voxels`` (a boolean mask or voxel indices): each voxel carries force / voxels, one eighth of it at
    each of its eight corners. The components sum to the force exactly (``math.fsum``)."""
    counts = _grid_counts(grid)
    chosen = _voxel_list(grid, voxels)
    if chosen.size == 0:
        raise InvalidValue("a body force needs at least one voxel")
    return _spread(int(grid.n_nodes), _voxel_corners(chosen, counts), _force(force_n))


# ---------------------------------------------------------------------------------------------------------
# The held skin


def skin_depth_mm(size_mm: float, voxel_mm: float) -> float:
    """How far behind a contact face the skin reaches: half the member size, at least two voxel sides."""
    return max(size_mm / 2.0, SKIN_LEAST_VOXELS * voxel_mm)


def held_skin(model: VoxelModel, faces: Faces, size_mm: float) -> np.ndarray:
    """The held skin behind ``faces``: one boolean per voxel, true for the active voxels whose centre lies
    within ``skin_depth_mm(size_mm, h)`` (half the member size, at least two voxel sides) of the centre of
    one of the faces. Distances are compared in half voxel sides, where every face centre and voxel centre
    lies on whole numbers."""
    model = _model(model)
    size_mm = _positive(size_mm, "the member size")
    if not isinstance(faces, Faces) or faces.counts != model.counts:
        raise InvalidValue("the faces must be michell.attach.Faces of the model's grid")
    nx, ny, nz = model.counts
    reach = 2.0 * skin_depth_mm(size_mm, model.voxel_mm) / model.voxel_mm  # in half voxel sides
    limit = reach * reach * (1.0 + 1e-9)
    span = int(math.ceil(reach / 2.0)) + 1
    steps = np.arange(-span, span + 1)
    grid_steps = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    mask = np.zeros(nx * ny * nz, dtype=bool)
    ijk = _voxel_ijk(faces.voxels, model.counts)
    for side in range(6):
        normal = np.zeros(3, dtype=np.int64)
        normal[side // 2] = 1 if side % 2 else -1
        # A voxel `step` away from the face's voxel has its centre 2 step - normal half sides from the face's.
        offsets = grid_steps[np.sum((2 * grid_steps - normal) ** 2, axis=1) <= limit]
        here = ijk[faces.sides == side]
        block = max(1, 2_000_000 // max(1, len(offsets)))
        for start in range(0, len(here), block):
            reached = (here[start : start + block, None, :] + offsets[None, :, :]).reshape(-1, 3)
            inside = np.all((reached >= 0) & (reached < np.array([nx, ny, nz])), axis=1)
            reached = reached[inside]
            mask[reached[:, 0] + nx * (reached[:, 1] + ny * reached[:, 2])] = True
    return mask & model.active


# ---------------------------------------------------------------------------------------------------------
# The set-up


@dataclass(frozen=True, eq=False)
class Attachment:
    """One line of the report of ``set_up``: a support, or a load of a load case.

    - ``role`` (``"support"`` or ``"load"``), ``case`` (the load case's name, None for a support),
      ``index`` (the position in the supports or in the case's loads, from 0) and ``on`` (what it acts on,
      in words).
    - ``rule``: ``"region"`` (a solid region: its ``voxels`` and ``volume_mm3``) or ``"contact"`` (a body or a
      box: its contact ``faces`` and ``area_mm2``).
    - ``nodes``: the nodes it holds or loads (ascending); ``point_mm``: the point of action, the centroid of
      its nodal shares (each face's quarters or each voxel's eighths), in the meshes' coordinates.
    - For a load: ``force_n`` (as given), ``forces_n`` (the force on each of ``nodes``, after the shares on
      held degrees of freedom are set to zero), ``dropped_n`` (the sum of those shares, a vector) and
      ``dropped_share`` (its length over the force's). For a support: ``fix``.
    - ``skin``: the voxels held solid as its skin that the model did not already hold (indices), with
      ``skin_mass_g``; ``closed_faces``: the faces of the grid's box that are closed for the density filter
      and on which some of its contact faces lie.
    """

    role: str
    case: str | None
    index: int
    on: str
    rule: str
    nodes: np.ndarray
    faces: Faces | None
    area_mm2: float
    voxels: np.ndarray | None
    volume_mm3: float
    point_mm: tuple[float, float, float]
    fix: tuple[str, ...] | None
    force_n: tuple[float, float, float] | None
    forces_n: np.ndarray | None
    dropped_n: tuple[float, float, float]
    dropped_share: float
    skin: np.ndarray
    skin_mass_g: float
    closed_faces: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.role == "support":
            return f"support {self.index} (on {self.on})"
        return f"load {self.index} of load case {self.case!r} (on {self.on})"


@dataclass(frozen=True, eq=False)
class SetUp:
    """What ``set_up`` returns.

    - ``problem``: the ``Problem`` (the model's grid, the held degrees of freedom, the material, one
      ``LoadCase`` per case); ``held``: its held degrees of freedom, a boolean per degree of freedom.
    - ``keep_solid``: the ``KeepSolid`` of the voxels held solid, the model's solid regions and the skins.
    - ``outside`` and ``inactive``: what the density filter sees beyond each face of the grid and in the
      inactive voxels, for ``stiff_part`` (``outside`` is a new dict at each read).
    - ``size_mm``: the member size the skins were set for.
    - ``supports`` and ``loads``: the report, one ``Attachment`` per support and per load (``report`` is the
      two together).

    ``stiff_part(mass_g, **settings)`` runs ``michell.stiff_part(problem, mass_g, size_mm, keep_solid,
    outside=outside, inactive=inactive, **settings)``.
    """

    model: VoxelModel
    problem: Problem
    held: np.ndarray
    keep_solid: KeepSolid
    filter_faces: Mapping[str, str]
    inactive: str
    size_mm: float
    supports: tuple[Attachment, ...]
    loads: tuple[Attachment, ...]

    @property
    def outside(self) -> dict[str, str]:
        return dict(self.filter_faces)

    @property
    def report(self) -> tuple[Attachment, ...]:
        return self.supports + self.loads

    def stiff_part(self, mass_g: float, **settings):
        from michell.recipes import stiff_part

        return stiff_part(
            self.problem, mass_g, self.size_mm, self.keep_solid, outside=self.outside, inactive=self.inactive, **settings
        )


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (math.isfinite(value) and value > 0):
        raise InvalidValue(f"{name} must be a positive finite number; got {value!r}")
    return float(value)


def _threads(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise InvalidValue(f"threads must be a whole number of at least 1; got {value!r}")
    return int(value)


def _overrides(outside) -> dict[str, str]:
    if outside is None:
        return {}
    if not isinstance(outside, Mapping):
        raise InvalidType(f"outside is a dict of grid face to 'closed' or 'empty'; got {type(outside).__name__}")
    for face, word in outside.items():
        if face not in FACES:
            raise InvalidValue(f"the faces of the grid are {', '.join(FACES)}; got {face!r}")
        if word not in ("closed", "empty"):
            raise InvalidValue(f"a face of the density filter is 'closed' or 'empty'; got {word!r} for {face}")
    return dict(outside)


@dataclass
class _Placed:
    """A support or a load placed on the model, before the report is written."""

    role: str
    case: str | None
    index: int
    item: Support | Load
    rule: str
    corners: np.ndarray  # one row of nodes per face (4) or voxel (8)
    faces: Faces | None = None
    voxels: np.ndarray | None = None

    @property
    def label(self) -> str:
        if self.role == "support":
            return f"support {self.index} (on {_describe(self.item.on)})"
        return f"load {self.index} of load case {self.case!r} (on {_describe(self.item.on)})"


def set_up(
    model: VoxelModel,
    material: Material,
    supports: Sequence[Support],
    cases: Sequence[Case],
    size_mm: float,
    *,
    threads: int = 1,
    outside: Mapping[str, str] | None = None,
    solver_tolerance: float = 1e-6,
    solver_max_steps: int = 5000,
) -> SetUp:
    """The problem an optimisation runs on, from the voxel model of a part, the material, the supports, the
    load cases (``Case``), the member size in mm and the number of threads.

    1. **What each support and load acts on** (``Support``, ``Load``): a solid region of the part gives a
       uniform body force over its voxels (``body_force_loads``) or holds all their nodes; a body or a box
       gives a uniform traction over the boundary faces in contact with it (``contact_faces``,
       ``traction_loads``) or holds their nodes, in the directions of ``fix``. A body that touches no
       boundary face is refused, with what was measured.
    2. **Shares on held degrees of freedom.** A load's shares on degrees of freedom that a support holds act
       on the support directly: they are set to zero, not spread over the other nodes. Their sum is reported
       (``dropped_n``, and ``dropped_share``, its length over the force's); above 1 % ``set_up`` warns
       (``LoadOnSupport``), above one half it refuses the load. The rest of the force sums to it exactly.
    3. **Filter faces.** Every boundary of the part is a free surface for the density filter: ``outside``
       is ``"empty"`` on every face of the grid, and ``inactive`` is ``"empty"``; except that a face of the
       grid's box whose whole active area (every boundary face of an active voxel on it) is in contact with
       one support is ``"closed"``. ``outside`` (a dict by face, ``"x-"`` to ``"z+"``) overrides that per
       face.
    4. **The held skin.** Where a contact face of a support or a load lies on a design voxel (active, not
       held solid by the model) and not on a face of the grid that is closed (after the overrides), the
       active voxels whose centre lies within half the member size (at least two voxel sides) of the face's
       centre are held solid (``held_skin``). With ``skin=False`` on that support or load such a face is
       refused instead. Region supports and loads need no skin.

    ``solver_tolerance`` and ``solver_max_steps`` are passed to ``Problem``. Returns a ``SetUp``; the
    rules, and what they give in the cases that decide them, are in ``docs/design/geometry.md``.
    """
    model = _model(model)
    if not isinstance(material, Material):
        raise InvalidType(f"the material is a michell.Material; got {type(material).__name__}")
    if not isinstance(supports, (list, tuple)) or any(not isinstance(s, Support) for s in supports):
        raise InvalidType("supports are a list of michell.attach.Support")
    if not isinstance(cases, (list, tuple)) or not cases or any(not isinstance(c, Case) for c in cases):
        raise InvalidType("cases are a non-empty list of michell.attach.Case")
    size_mm = _positive(size_mm, "the member size")
    threads = _threads(threads, 1)
    overrides = _overrides(outside)
    grid, counts, h = model.grid, model.counts, model.voxel_mm
    n_nodes = int(grid.n_nodes)
    boundary = boundary_faces(model)
    found: dict[object, tuple] = {}  # what each target gives, computed once

    def place(role: str, case: str | None, index: int, item) -> _Placed:
        target = item.on
        key = id(target) if isinstance(target, Mesh) else target
        if key not in found:
            if isinstance(target, str):
                voxels = np.flatnonzero(region_voxels(model, target))
                if voxels.size == 0:
                    raise InvalidValue(f"the solid region {target!r} has no active voxel: the void regions remove it")
                found[key] = ("region", voxels, None)
            else:
                contact = _contact(model, _body_mesh(target, "the body"), boundary, threads)
                found[key] = ("contact", None, contact)
        rule, voxels, contact = found[key]
        placed = _Placed(role, case, index, item, rule, np.zeros((0, 4), dtype=np.int64))
        if rule == "region":
            placed.voxels, placed.corners = voxels, _voxel_corners(voxels, counts)
            return placed
        if len(contact.faces) == 0:
            most = int(contact.rays.max()) if contact.rays.size else 0
            volume = _body_mesh(target, "the body").volume_mm3
            raise InvalidValue(
                f"{placed.label} touches no boundary face of the part: of the {RAYS_PER_SIDE**2} rays along the "
                f"normal of any boundary face, at most {most} run inside it for {RAY_LENGTH:g} of a voxel side within "
                f"the face's window (contact needs more than {CONTACT_RAYS}); the body's volume is {volume:.4g} mm3. "
                f"A marker body should touch the part's face or overlap it slightly; to load or hold a volume of the "
                f"part, make it a solid region and name that region",
                {"most_rays": most, "volume_mm3": volume},
            )
        placed.faces, placed.corners = contact.faces, contact.faces.nodes()
        return placed

    placed_supports = [place("support", None, i, support) for i, support in enumerate(supports)]
    placed_loads = [place("load", case.name, i, load) for case in cases for i, load in enumerate(case.loads)]

    # The held degrees of freedom.
    held = np.zeros(3 * n_nodes, dtype=bool)
    for placed in placed_supports:
        nodes = np.unique(placed.corners)
        for direction in placed.item.fix:
            held[3 * nodes + _AXES.index(direction)] = True

    # Filter faces: closed where one support's contact covers the whole active area of a grid face.
    filter_faces = {}
    for name in FACES:
        on_face = boundary.keys()[boundary.on_grid_face(name)]
        closed = on_face.size > 0 and any(
            placed.faces is not None and np.isin(on_face, placed.faces.keys()).all() for placed in placed_supports
        )
        filter_faces[name] = "closed" if closed else "empty"
    filter_faces.update(overrides)
    closed_names = [name for name in FACES if filter_faces[name] == "closed"]

    # The held skins.
    kept = model.keep_solid.copy()
    skins, closed_under = {}, {}
    for placed in placed_supports + placed_loads:
        skins[id(placed)] = np.zeros(0, dtype=np.int64)
        closed_under[id(placed)] = ()
        if placed.faces is None:
            continue
        on_closed = np.zeros(len(placed.faces), dtype=bool)
        under = []
        for name in closed_names:
            here = placed.faces.on_grid_face(name)
            if here.any():
                under.append(name)
            on_closed |= here
        closed_under[id(placed)] = tuple(under)
        needs = ~model.keep_solid[placed.faces.voxels] & ~on_closed
        if not needs.any():
            continue
        if not placed.item.skin:
            raise InvalidValue(
                f"{placed.label} has skin=False, and {int(needs.sum())} of its {len(placed.faces)} contact faces lie "
                f"on design voxels that are neither held solid nor on a closed face of the grid: the filtered "
                f"design cannot become solid there. Hold the voxels behind it solid (a solid region), or leave the "
                f"skin on",
                {"faces_on_design_voxels": int(needs.sum())},
            )
        skin = held_skin(model, placed.faces.subset(needs), size_mm) & ~model.keep_solid
        skins[id(placed)] = np.flatnonzero(skin)
        kept |= skin

    # The loads, with their shares on held degrees of freedom set to zero.
    load_vectors = {case.name: np.zeros(3 * n_nodes) for case in cases}
    load_results = {}
    warned = []
    for placed in placed_loads:
        force = np.array(placed.item.force_n)
        vector = _spread(n_nodes, placed.corners, force)
        on_held = held & (vector != 0.0)
        dropped = np.array([math.fsum(vector[a::3][on_held[a::3]]) for a in range(3)])
        vector[on_held] = 0.0
        size = float(np.linalg.norm(force))
        share = float(np.linalg.norm(dropped)) / size if size > 0.0 else 0.0
        if share > DROPPED_REFUSAL:
            raise InvalidValue(
                f"{placed.label}: {100 * share:.3g} % of the force falls on degrees of freedom that a support holds "
                f"(more than {100 * DROPPED_REFUSAL:g} %), so it would act on the support, not on the part",
                {"dropped_share": share},
            )
        if share > DROPPED_WARNING:
            warned.append(
                f"{placed.label}: {100 * share:.3g} % of the force falls on degrees of freedom that a support holds "
                f"and acts on the support directly (dropped {', '.join(f'{c:.4g}' for c in dropped)} N)"
            )
        load_vectors[placed.case] += vector
        load_results[id(placed)] = (vector, dropped, share)

    load_cases = [LoadCase(case.name, load_vectors[case.name], case.weight) for case in cases]
    problem = Problem(
        grid, held, material, load_cases, threads=threads, solver_tolerance=solver_tolerance,
        solver_max_steps=solver_max_steps,
    )  # fmt: skip
    for message in warned:
        warnings.warn(message, LoadOnSupport, stacklevel=2)

    nx, ny, _ = counts

    def node_positions(nodes: np.ndarray) -> np.ndarray:
        ijk = np.stack([nodes % (nx + 1), (nodes // (nx + 1)) % (ny + 1), nodes // ((nx + 1) * (ny + 1))], axis=1)
        return model.to_mesh(ijk * h)

    def report(placed: _Placed) -> Attachment:
        nodes = np.unique(placed.corners)
        weights = np.bincount(np.searchsorted(nodes, placed.corners.ravel()), minlength=nodes.size) / placed.corners.size
        point = tuple(float(c) for c in weights @ node_positions(nodes))
        skin = skins[id(placed)]
        if placed.role == "load":
            vector, dropped, share = load_results[id(placed)]
            forces = np.stack([vector[3 * nodes + a] for a in range(3)], axis=1)
            force_n, dropped_n = placed.item.force_n, tuple(float(c) for c in dropped)
        else:
            forces, force_n, dropped_n, share = None, None, (0.0, 0.0, 0.0), 0.0
        return Attachment(
            role=placed.role, case=placed.case, index=placed.index, on=_describe(placed.item.on), rule=placed.rule,
            nodes=nodes, faces=placed.faces, area_mm2=placed.faces.area_mm2 if placed.faces is not None else 0.0,
            voxels=placed.voxels, volume_mm3=placed.voxels.size * h**3 if placed.voxels is not None else 0.0,
            point_mm=point, fix=placed.item.fix if placed.role == "support" else None, force_n=force_n,
            forces_n=forces, dropped_n=dropped_n, dropped_share=share, skin=skin,
            skin_mass_g=float(material.mass_g(skin.size * h**3)), closed_faces=closed_under[id(placed)],
        )  # fmt: skip

    return SetUp(
        model=model,
        problem=problem,
        held=held,
        keep_solid=KeepSolid(kept, grid),
        filter_faces=dict(filter_faces),
        inactive="empty",
        size_mm=size_mm,
        supports=tuple(report(p) for p in placed_supports),
        loads=tuple(report(p) for p in placed_loads),
    )


__all__ = [
    "CONTACT_RAYS",
    "DROPPED_REFUSAL",
    "DROPPED_WARNING",
    "FACES",
    "FINE_VOXELS_PER_PIECE",
    "LENGTH_SLACK",
    "RAYS_PER_SIDE",
    "RAY_LENGTH",
    "SKIN_LEAST_VOXELS",
    "Attachment",
    "Case",
    "Contact",
    "Faces",
    "Load",
    "LoadOnSupport",
    "SetUp",
    "Support",
    "body_force_loads",
    "boundary_faces",
    "contact_faces",
    "held_skin",
    "region_voxels",
    "set_up",
    "skin_depth_mm",
    "traction_loads",
]
