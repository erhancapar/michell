"""From the result of ``stiff_part`` to the part's closed triangle mesh, with exact interfaces and checks.

- ``finish(part, setup, keep_solid=None, *, threads=None)``: the mesh of an optimised part, as a
  ``FinishedPart``. ``part`` is the ``StiffPart`` that ``stiff_part`` returned and ``setup`` the ``SetUp`` it
  ran on (``SetUp.stiff_part``), whose voxel model keeps the regions' meshes: the organic surface of the design is cut by the envelope (the
  design and solid regions together), joined with the exact solid regions and cut by the exact void regions.
  ``setup`` may instead be the bare ``Problem`` of a result built in Python without region meshes (with its
  ``keep_solid``): the cut is then the grid's box and exactness is not claimed.
- ``FinishedPart``: the mesh with the level of the surface, the volumes and masses, the loose pieces dropped,
  the closed cavities, every gate with its numbers, the genus against the voxel body's, and the check of the
  delivered part (``DeliveredCheck``: the mesh voxelised back on the run's grid and solved once per load case).
- ``MeshRefused``: the refusal of a mesh that fails a gate (code ``mesh_refused``), naming the gate and the
  numbers. ``DeliveredPartDiffers``, ``GenusDiffers`` and ``LevelOffTarget``: the warnings ``finish`` gives;
  ``StlRounding``: the warning ``michell run`` gives when float32 STL coordinates would join vertices, and
  ``StlRefused`` the one it gives when ``write_stl`` refuses the mesh and ``part.stl`` is not written.

The route, and the reason for each step, are in ``docs/design/geometry.md`` ("From the result to the mesh").
The cut and join use manifold3d (Apache-2.0) through its 64-bit mesh interface, so that the regions'
vertices keep their float64 coordinates.
"""

from __future__ import annotations

import math
import numbers
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import manifold3d
import numpy as np

from michell._engine import dot
from michell.analysis3d import Grid3d
from michell.attach import SetUp
from michell.errors import InvalidType, InvalidValue, _Refused
from michell.filters import KeepSolid
from michell.mesh import ZERO_AREA, Mesh, box
from michell.problem import Problem
from michell.recipes import StiffPart
from michell.regions import MAJORITY, VoxelModel, occupancy
from michell.surface import ScalarField, iso_surface, refine, smooth

# The field: how many layers of voxels the design's field is continued into the solid regions and beyond
# the part (outside the envelope, into the void regions and beyond the grid), by repetition.
LAYERS_INTO_SOLID = 3
LAYERS_BEYOND = 3

# The surface: the field refined by this factor and order, the iso-surface smoothed by this many Taubin
# iterations (michell.surface.smooth with its default weights).
REFINE_FACTOR = 2
REFINE_ORDER = "cubic"
SMOOTHING_ITERATIONS = 10

# The level search stops when the organic volume is within this share of its target, after at most
# LEVEL_STEPS evaluations; the level is searched between LEVEL_LOW and LEVEL_HIGH, starting at LEVEL_START.
LEVEL_TOLERANCE = 0.002
LEVEL_STEPS = 24
LEVEL_START = 0.5
LEVEL_LOW, LEVEL_HIGH = 0.02, 0.98

# Bodies of positive volume below this share of the whole are dropped and listed; a larger second body is
# refused.
LOOSE_SHARE = 0.01

# Exact interfaces: a vertex the mesh takes from a region lies on the plane of the region's triangle it
# came from within this distance (mm).
INTERFACE_MM = 1e-6

# The delivered-part check: the compliance of the mesh voxelised back is within this share of the result's.
DELIVERED_TOLERANCE = 0.05

# The point test: a point within this many voxel sides of the surface, along the ray, counts as on it.
ON_SURFACE_VOXELS = 1e-6

# The mesh voxelised back on a grid this many times finer (odd, so that the middle fine column runs through
# each voxel's centre), one ray per fine column; computed in pieces of at most this many fine voxels.
BACK_SUBDIVISION = 3
BACK_PIECE = 1 << 22


class MeshRefused(_Refused, ValueError):
    """The refusal of ``finish``: the mesh failed a gate and is not returned. ``gate`` names the gate (such
    as ``"closed"``, ``"one body"``, ``"held solid inside"``), ``message`` says what was measured, and
    ``details`` holds the numbers; ``gates`` holds every gate measured before the refusal (``Gate``). Code
    ``mesh_refused``; also an ``EngineError`` and a ``ValueError``."""

    def __init__(self, gate: str, message: str, details=None, gates=()):
        super().__init__(f"{gate}: {message}", details)
        self.code = "mesh_refused"
        self.gate = gate
        self.gates = tuple(gates)
        self.args = (str(self),)

    def as_dict(self) -> dict:
        return {"code": self.code, "gate": self.gate, "message": self.message, "details": dict(self.details)}


class DeliveredPartDiffers(UserWarning):
    """Issued by ``finish`` when the mesh voxelised back on the run's grid has a compliance (in a load case,
    or the weighted goal) more than 5 % from the result's: the delivered part is not the optimised one. The
    numbers are in ``FinishedPart.delivered``; ``michell run`` then exits with status 1."""


class GenusDiffers(UserWarning):
    """Issued by ``finish`` when the mesh's main body has a different number of through holes from the
    returned design's voxel body (``FinishedPart.genus`` and ``voxel_genus``)."""


class StlRounding(UserWarning):
    """Issued by ``michell run`` when ``part.stl``, which stores coordinates as float32, would join vertices
    of the mesh that round to the same float32 point; ``part.3mf`` keeps every float64 coordinate. ``finish``
    joins such vertices itself where it can (``FinishedPart.smoothing``, ``float32_vertices_joined``), and
    ``write_stl`` never writes a closed mesh that would not read back closed."""


class StlRefused(UserWarning):
    """Issued by ``michell run`` when ``write_stl`` refuses the mesh because, as the float32 coordinates an
    STL file stores, it would not read back closed: ``part.stl`` is not written, while ``part.3mf`` (which
    keeps every float64 coordinate) and the result are written all the same, and the reason is in the
    ``mesh`` section of ``result.json`` (``stl_refused``)."""


class LevelOffTarget(UserWarning):
    """Issued by ``finish`` when the level search ends with the organic volume more than 0.2 % from its
    target (the volume of the returned design's design voxels), as it does where the volume jumps at a
    level (a member that closes or opens)."""


@dataclass(frozen=True)
class Gate:
    """One gate of ``finish``: ``name``, whether it ``passed``, the ``numbers`` measured (plain numbers by
    name) and a sentence (``message``)."""

    name: str
    passed: bool
    numbers: Mapping[str, float]
    message: str

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "numbers": dict(self.numbers), "message": self.message}


@dataclass(frozen=True)
class LoosePiece:
    """A body of the finished mesh dropped as loose: its ``volume_mm3``, its ``share`` of the whole and the
    corners ``low`` and ``high`` of the box around it (mm)."""

    volume_mm3: float
    share: float
    low: tuple[float, float, float]
    high: tuple[float, float, float]


@dataclass(frozen=True)
class CaseCheck:
    """One load case of the delivered-part check: the compliance (N mm) of the returned design
    (``result``, its nominal design) and of the mesh voxelised back (``delivered``), and ``difference``,
    delivered over result minus 1."""

    case: str
    weight: float
    result: float
    delivered: float
    difference: float


@dataclass(frozen=True)
class DeliveredCheck:
    """The mesh voxelised back on the run's grid (a voxel is solid when the mesh fills at least half of it,
    by the voxeliser with 3 x 3 rays per voxel) against the returned design (its nominal design at 0.5 or
    more): ``solid_voxels`` of the returned design, the voxels ``gained`` and ``lost`` and their shares of
    ``solid_voxels`` (``changed_share`` the two together), ``gained_inactive`` (voxels the mesh fills that
    are not active; they are not solved). Each load case solved once on the 0/1 design (``cases``) and the
    weighted goal (``goal_result``, ``goal_delivered``, ``goal_difference``); ``within``: every difference
    at most ``tolerance``. ``goal_reported`` is the result's own goal (``Result.goal``) where it has one."""

    solid_voxels: int
    gained: int
    lost: int
    gained_inactive: int
    cases: tuple[CaseCheck, ...]
    goal_result: float
    goal_delivered: float
    goal_reported: float | None
    tolerance: float

    @property
    def gained_share(self) -> float:
        return self.gained / self.solid_voxels if self.solid_voxels else math.inf

    @property
    def lost_share(self) -> float:
        return self.lost / self.solid_voxels if self.solid_voxels else math.inf

    @property
    def changed_share(self) -> float:
        return (self.gained + self.lost) / self.solid_voxels if self.solid_voxels else math.inf

    @property
    def goal_difference(self) -> float:
        return self.goal_delivered / self.goal_result - 1.0 if self.goal_result else math.inf

    @property
    def largest_difference(self) -> float:
        return max([abs(self.goal_difference), *(abs(c.difference) for c in self.cases)])

    @property
    def within(self) -> bool:
        return bool(self.largest_difference <= self.tolerance)

    def as_dict(self) -> dict:
        return {
            "solid_voxels": self.solid_voxels,
            "gained": self.gained,
            "lost": self.lost,
            "gained_share": self.gained_share,
            "lost_share": self.lost_share,
            "changed_share": self.changed_share,
            "gained_inactive": self.gained_inactive,
            "cases": [
                {"case": c.case, "weight": c.weight, "result": c.result, "delivered": c.delivered,
                 "difference": c.difference}
                for c in self.cases
            ],  # fmt: skip
            "goal_result": self.goal_result,
            "goal_delivered": self.goal_delivered,
            "goal_reported": self.goal_reported,
            "goal_difference": self.goal_difference,
            "tolerance": self.tolerance,
            "within": self.within,
        }


@dataclass(frozen=True, eq=False)
class FinishedPart:
    """What ``finish`` returns.

    - ``mesh``: the closed mesh of the part (mm, float64, facing outward); ``exact``: whether its interfaces
      are the regions' own (a set-up with region meshes), else it was cut by the grid's box.
    - ``level``: the level of the organic surface on the filtered field; ``organic_mm3`` and ``target_mm3``:
      the volume of the mesh outside the exact solid and void regions and its target, the volume of the
      returned design's design voxels; ``level_steps``: each level tried, with its organic volume.
    - ``volume_mm3``, ``mass_g``; ``mass_limit_g`` (the run's mass bound, None when unknown),
      ``design_mass_g`` (the returned design's mass as the optimisation measured it, None when unknown) and
      ``voxel_mass_g`` (its voxels at 0.5 or more).
    - ``loose_pieces`` (``LoosePiece``, dropped), ``cavities`` (closed cavities kept in the mesh).
    - ``gates``: every gate (``Gate``), all passed; ``genus`` and ``voxel_genus`` (reported, a warning when
      they differ); ``delivered``: the ``DeliveredCheck``.
    - ``smoothing``: the Taubin smoothing of the last surface (iterations, largest move in mm, triangles
      whose normal turned over, the surface's triangles before the cut, the triangles without area the cut
      left that were removed by an edge flip, and the vertices joined because they round to one float32
      point); ``settings``: the route's settings; ``seconds``: time per step.

    ``summary()`` gives all of it as plain data (the ``mesh`` section of ``result.json``).
    """

    mesh: Mesh
    exact: bool
    level: float
    organic_mm3: float
    target_mm3: float
    level_steps: tuple[tuple[float, float], ...]
    volume_mm3: float
    mass_g: float
    mass_limit_g: float | None
    design_mass_g: float | None
    voxel_mass_g: float
    exact_mm3: float
    loose_pieces: tuple[LoosePiece, ...]
    cavities: int
    gates: tuple[Gate, ...]
    genus: int
    voxel_genus: int
    delivered: DeliveredCheck
    smoothing: Mapping[str, float]
    settings: Mapping[str, float]
    seconds: Mapping[str, float]

    @property
    def organic_difference(self) -> float:
        """The organic volume over its target, minus 1."""
        return self.organic_mm3 / self.target_mm3 - 1.0 if self.target_mm3 else math.inf

    def gate(self, name: str) -> Gate:
        for gate in self.gates:
            if gate.name == name:
                return gate
        raise InvalidValue(f"no gate called {name!r}; the gates are {[g.name for g in self.gates]}")

    def summary(self) -> dict:
        """Everything above but the mesh itself, as plain data."""
        return {
            "exact": self.exact,
            "level": self.level,
            "level_steps": [list(step) for step in self.level_steps],
            "organic_mm3": self.organic_mm3,
            "target_mm3": self.target_mm3,
            "organic_difference": self.organic_difference,
            "triangles": int(len(self.mesh.triangles)),
            "vertices": int(len(self.mesh.vertices)),
            "volume_mm3": self.volume_mm3,
            "exact_mm3": self.exact_mm3,
            "mass_g": self.mass_g,
            "mass_limit_g": self.mass_limit_g,
            "design_mass_g": self.design_mass_g,
            "voxel_mass_g": self.voxel_mass_g,
            "loose_pieces": [
                {"volume_mm3": p.volume_mm3, "share": p.share, "low": list(p.low), "high": list(p.high)}
                for p in self.loose_pieces
            ],
            "cavities": self.cavities,
            "gates": [gate.as_dict() for gate in self.gates],
            "genus": self.genus,
            "voxel_genus": self.voxel_genus,
            "delivered": self.delivered.as_dict(),
            "smoothing": dict(self.smoothing),
            "settings": dict(self.settings),
            "seconds": dict(self.seconds),
        }


# ---------------------------------------------------------------------------------------------------------
# What the mesh is made from


@dataclass(frozen=True, eq=False)
class _Source:
    """The inputs of the route: the grid, the classes and the regions' meshes."""

    problem: Problem
    counts: tuple[int, int, int]
    h: float
    origin: np.ndarray
    active: np.ndarray  # (nz, ny, nx) booleans, as the arrays below
    held: np.ndarray  # every voxel held solid (regions and skins)
    region_held: np.ndarray  # the voxels the solid regions hold
    void_voxels: np.ndarray  # the voxels void regions fill by majority
    design_mesh: Mesh | None
    solids: Mapping[str, Mesh]
    voids: Mapping[str, Mesh]
    exact: bool


def _source(setup, keep_solid) -> _Source:
    if isinstance(setup, SetUp):
        if keep_solid is not None:
            raise InvalidValue("keep_solid is the set-up's own; pass it only with a bare Problem")
        model: VoxelModel = setup.model
        shape = model.counts[::-1]
        roles = model.roles
        if "design" not in model.meshes:
            raise InvalidValue("the set-up's voxel model keeps no region meshes (it was not made by voxel_model)")
        solids = {name: model.meshes[name] for name, role in roles.items() if role == "solid"}
        voids = {name: model.meshes[name] for name, role in roles.items() if role == "void"}
        void_voxels = np.zeros(math.prod(model.counts), dtype=bool)
        for name in voids:
            void_voxels |= np.asarray(model.region_occupancy[name]) >= MAJORITY
        return _Source(
            problem=setup.problem, counts=model.counts, h=model.voxel_mm, origin=np.array(model.origin_mm),
            active=model.active.reshape(shape), held=np.asarray(setup.keep_solid.mask).reshape(shape),
            region_held=np.asarray(model.keep_solid).reshape(shape), void_voxels=void_voxels.reshape(shape),
            design_mesh=model.meshes["design"], solids=solids, voids=voids, exact=True,
        )  # fmt: skip
    if isinstance(setup, Problem):
        grid = setup.grid
        if not isinstance(grid, Grid3d):
            raise InvalidValue("finish makes the mesh of a part on a 3D grid (Grid3d); the problem is 2D")
        counts = (int(grid.nx), int(grid.ny), int(grid.nz))
        shape = counts[::-1]
        active = (np.asarray(grid.active) == 1.0).reshape(shape)
        if keep_solid is None:
            held = np.zeros(shape, dtype=bool)
        elif isinstance(keep_solid, KeepSolid):
            held = np.asarray(keep_solid.mask, dtype=bool).reshape(shape)
        else:
            raise InvalidType(f"keep_solid is a michell.KeepSolid or None; got {type(keep_solid).__name__}")
        return _Source(
            problem=setup, counts=counts, h=float(grid.h), origin=np.zeros(3), active=active, held=held,
            region_held=np.zeros(shape, dtype=bool), void_voxels=np.zeros(shape, dtype=bool), design_mesh=None,
            solids={}, voids={}, exact=False,
        )  # fmt: skip
    raise InvalidType(f"finish takes the SetUp the part ran on, or a bare Problem; got {type(setup).__name__}")


# ---------------------------------------------------------------------------------------------------------
# Step 1: the field


def _shifted(values: np.ndarray, axis: int, step: int) -> np.ndarray:
    """``values`` moved by one place along ``axis`` (towards +1 or -1), the emptied layer set to zero: entry
    i of the result holds entry i - step of ``values``."""
    out = np.zeros_like(values)
    target = [slice(None)] * 3
    source = [slice(None)] * 3
    if step > 0:
        target[axis], source[axis] = slice(1, None), slice(None, -1)
    else:
        target[axis], source[axis] = slice(None, -1), slice(1, None)
    out[tuple(target)] = values[tuple(source)]
    return out


def _continued(values: np.ndarray, known: np.ndarray, into: np.ndarray, layers: int) -> None:
    """Continues ``values`` from the ``known`` voxels into the voxels of ``into`` by repetition, ``layers``
    layers deep, in place: in each layer every voxel of ``into`` that is not known but has a known face
    neighbour takes the mean of its known face neighbours and becomes known. Across a flat face this is
    the value of the voxel behind it, repeated."""
    for _ in range(layers):
        total = np.zeros_like(values)
        count = np.zeros(values.shape, dtype=np.int64)
        weighted = np.where(known, values, 0.0)
        for axis in range(3):
            for step in (1, -1):
                total += _shifted(weighted, axis, step)
                count += _shifted(known, axis, step)
        new = into & ~known & (count > 0)
        if not new.any():
            return
        values[new] = total[new] / count[new]
        known |= new


def _field(source: _Source, filtered: np.ndarray) -> ScalarField:
    """The field the surface is drawn from, on the lattice of the voxel centres padded beyond the grid."""
    pad = LAYERS_BEYOND + 1
    shape = tuple(n + 2 * pad for n in source.active.shape)
    inner = (slice(pad, -pad),) * 3

    def padded(values, fill=False):
        out = np.full(shape, fill, dtype=np.asarray(values).dtype)
        out[inner] = values
        return out

    active, held, region_held = padded(source.active), padded(source.held), padded(source.region_held)
    values = np.zeros(shape)
    known = np.zeros(shape, dtype=bool)
    design = active & ~held
    values[design] = padded(filtered.reshape(source.active.shape), 0.0)[design]
    skin = held & ~region_held
    values[skin] = 1.0
    known |= design | skin
    # Members run on into the solid regions, and deep inside them the field is low again.
    _continued(values, known, region_held, LAYERS_INTO_SOLID)
    known |= region_held
    # Beyond the part: into the inactive voxels and beyond the grid (with exact regions), or beyond the grid
    # only (cut by the grid's box).
    beyond = ~active if source.exact else ~padded(np.ones(source.active.shape, dtype=bool))
    _continued(values, known, beyond, LAYERS_BEYOND)
    origin = source.origin + source.h / 2.0 - pad * source.h
    return ScalarField(values, source.h, tuple(origin))


# ---------------------------------------------------------------------------------------------------------
# Step 3: cut and join (manifold3d, 64-bit)


def _planes(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The planes of a region's triangles: for each triangle the index of its plane, and for each plane its
    unit normal and a point on it. Triangles whose unit normals and offsets agree to 1e-9 (mm) share a
    plane, as the two triangles of a box's face do."""
    v, t = mesh.vertices, mesh.triangles
    a = v[t[:, 0]]
    normal = np.cross(v[t[:, 1]] - a, v[t[:, 2]] - a)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    offset = np.einsum("ij,ij->i", normal, a)
    keys = np.round(np.column_stack([normal, offset]), 9) + 0.0
    _, first, label = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return label.ravel(), normal[first], a[first]


def _manifold(mesh: Mesh, ids: np.ndarray) -> manifold3d.Manifold:
    """``mesh`` as a manifold3d body (float64), each triangle with the face id ``ids`` gives it, so that the
    triangles of the result can be traced to the region and the plane they came from. Triangles with one id
    may be joined into one face by manifold3d; triangles with different ids keep the edges between them."""
    data = manifold3d.Mesh64(
        np.array(mesh.vertices, dtype=np.float64, order="C"),
        np.array(mesh.triangles, dtype=np.uint64, order="C"),
        face_id=np.asarray(ids, dtype=np.uint64),
    )
    body = manifold3d.Manifold(data)
    if body.status() != manifold3d.Error.NoError:
        raise InvalidValue(f"manifold3d does not take the mesh as a closed body: {body.status()}")
    return body


def _from_manifold(body: manifold3d.Manifold) -> tuple[Mesh, np.ndarray]:
    data = body.to_mesh64()
    vertices = np.asarray(data.vert_properties, dtype=np.float64)[:, :3]
    triangles = np.asarray(data.tri_verts, dtype=np.int64).reshape(-1, 3)
    ids = np.asarray(data.face_id, dtype=np.int64)
    if len(ids) != len(triangles):
        ids = np.full(len(triangles), -1, dtype=np.int64)
    return Mesh(vertices, triangles), ids


@dataclass(frozen=True)
class _Region:
    """A region as the cut and join sees it: its name, role and mesh, the first face id of its planes and,
    for each plane, its unit normal and a point on it (face id ``first + i`` is plane i)."""

    name: str
    role: str
    mesh: Mesh
    first: int
    normals: np.ndarray
    points: np.ndarray

    @property
    def last(self) -> int:
        return self.first + len(self.normals)


@dataclass(frozen=True, eq=False)
class _Exact:
    """The exact regions as manifold3d bodies, made once: the envelope (design and solid regions joined),
    the solid regions joined, the void regions joined, the regions' planes and face ids, and the face id
    of the organic surface."""

    envelope: manifold3d.Manifold
    solid: manifold3d.Manifold | None
    void: manifold3d.Manifold | None
    exact_mm3: float  # the solid regions minus the void regions
    regions: tuple[_Region, ...]
    organic_id: int
    envelope_numbers: Mapping[str, float]


def _exact(source: _Source) -> _Exact:
    if source.exact:
        meshes = [("design", "design", source.design_mesh)]
        meshes += [(name, "solid", m) for name, m in source.solids.items()]
        meshes += [(name, "void", m) for name, m in source.voids.items()]
    else:
        low = source.origin
        high = source.origin + np.array(source.counts) * source.h
        meshes = [("grid box", "design", box(low, high))]
    regions, bodies = [], []
    first = 0
    for name, role, mesh in meshes:
        label, normals, points = _planes(mesh)
        try:
            bodies.append((role, _manifold(mesh, first + label)))
        except InvalidValue as error:
            raise MeshRefused(
                "cut and join", f"the {role} region {name!r} cannot be joined: {error.message}", {"region": name}
            ) from error
        regions.append(_Region(name, role, mesh, first, normals, points))
        first += len(normals)
    joined = [b for role, b in bodies if role in ("design", "solid")]
    solids = [b for role, b in bodies if role == "solid"]
    voids = [b for role, b in bodies if role == "void"]
    envelope = manifold3d.Manifold.batch_boolean(joined, manifold3d.OpType.Add) if len(joined) > 1 else joined[0]
    solid = manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add) if solids else None
    void = manifold3d.Manifold.batch_boolean(voids, manifold3d.OpType.Add) if voids else None
    exact_mm3 = 0.0
    if solid is not None:
        exact_mm3 = float((solid - void).volume() if void is not None else solid.volume())
    # The envelope must be one body: two regions that only touch are joined across their common face (the
    # union's symbolic perturbation does so), never left as two bodies.
    envelope_numbers = {"bodies": float(len(envelope.decompose())), "volume_mm3": float(envelope.volume())}
    return _Exact(envelope, solid, void, exact_mm3, tuple(regions), first, envelope_numbers)


def _finished(organic: Mesh, exact: _Exact) -> tuple[Mesh, np.ndarray, int]:
    ids = np.full(len(organic.triangles), exact.organic_id, dtype=np.uint64)
    body = _manifold(organic, ids) ^ exact.envelope
    if exact.solid is not None:
        body = body + exact.solid
    if exact.void is not None:
        body = body - exact.void
    if body.status() != manifold3d.Error.NoError:
        raise MeshRefused("cut and join", f"manifold3d failed on the cut and join: {body.status()}")
    mesh, ids = _from_manifold(body)
    return _without_slivers(mesh, ids)


def _slivers(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The triangles without area, as ``Mesh.defects`` counts them."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    longest = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                               np.linalg.norm(a - c, axis=1)]), axis=0)  # fmt: skip
    return np.flatnonzero(twice_area <= ZERO_AREA * longest**2)


def _without_slivers(mesh: Mesh, ids: np.ndarray, rounds: int = 8) -> tuple[Mesh, np.ndarray, int]:
    """Removes triangles without area by flipping their longest edge. manifold3d can leave a triangle whose
    three corners lie on one line where the organic body's outline meets an edge of a region's face (its
    corner on that line splits the edge). The sliver (a, b, c), with c on its longest side a-b, and its
    neighbour (b, a, d) across that side become (c, a, d) and (c, d, b): the same surface, with no vertex
    moved, and both new triangles lie in the neighbour's plane (c is on a-b) and take its face id. Returns
    the mesh, the face ids and the number of flips."""
    v = mesh.vertices
    t = np.array(mesh.triangles, dtype=np.int64)
    ids = np.array(ids)
    n = len(v)
    flips = 0
    for _ in range(rounds):
        bad = _slivers(v, t)
        if not len(bad):
            break
        keys = np.concatenate([t[:, 0] * n + t[:, 1], t[:, 1] * n + t[:, 2], t[:, 2] * n + t[:, 0]])
        order = np.argsort(keys)
        sorted_keys = keys[order]
        touched = np.zeros(len(t), dtype=bool)
        done = 0
        for i in bad:
            if touched[i]:
                continue
            corners = t[i]
            lengths = [np.linalg.norm(v[corners[(k + 1) % 3]] - v[corners[k]]) for k in range(3)]
            k = int(np.argmax(lengths))
            a, b, c = corners[k], corners[(k + 1) % 3], corners[(k + 2) % 3]
            at = np.searchsorted(sorted_keys, b * n + a)
            if at >= len(sorted_keys) or sorted_keys[at] != b * n + a:
                continue
            j = int(order[at] % len(t))
            if touched[j] or j == i:
                continue
            other = t[j]
            d = [x for x in other if x != a and x != b]
            if len(d) != 1:
                continue
            d = d[0]
            t[i] = (c, a, d)
            t[j] = (c, d, b)
            ids[i] = ids[j]
            touched[i] = touched[j] = True
            done += 1
        flips += done
        if not done:
            break
    return Mesh(v, t), ids, flips


def _plane_distance(points: np.ndarray, triangles: np.ndarray, ids: np.ndarray, exact: _Exact) -> np.ndarray:
    """For each point (one per triangle), its distance from the plane of the region's face the triangle
    came from (0 for a triangle of the organic surface)."""
    out = np.zeros(len(triangles))
    for region in exact.regions:
        own = (ids >= region.first) & (ids < region.last)
        if own.any():
            plane = ids[own] - region.first
            out[own] = np.abs(np.einsum("ij,ij->i", region.normals[plane], points[own] - region.points[plane]))
    return out


def _single_precision(mesh: Mesh, ids: np.ndarray, exact: _Exact) -> tuple[Mesh, np.ndarray, int]:
    """Joins vertices that round to the same float32 point, as an STL file stores them, so that the mesh is
    the same closed mesh in float32. Two such vertices are the ends of a sliver's short edge, left where a
    region's edge crosses another region's face a fraction of a micrometre from one of its corners. A group
    is joined only when its vertices are joined by edges, onto the vertex that keeps every triangle of the
    group on the plane of its region's face within INTERFACE_MM (so no vertex of a region's face leaves its
    plane); the triangles that lose their area are removed. If the joined mesh would not be closed, the mesh
    is returned unchanged. Returns the mesh, the face ids and the number of vertices joined."""
    rounded = mesh.vertices.astype(np.float32)
    _, group, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    group = group.ravel()
    if not np.any(counts > 1):
        return mesh, ids, 0
    v, t = mesh.vertices, np.array(mesh.triangles)
    target = np.arange(len(v))
    joined = 0
    for g in np.flatnonzero(counts > 1):
        members = np.flatnonzero(group == g)
        touching = np.flatnonzero(np.isin(t, members).any(axis=1))
        inside = np.isin(t[touching], members).sum(axis=1)
        if np.count_nonzero(inside >= 2) < len(members) - 1:
            continue  # not joined by edges
        best, worst = None, math.inf
        for keep in members:
            moved = np.where(np.isin(t[touching], members), keep, t[touching])
            corners = v[moved]
            far = max(_plane_distance(corners[:, k], t[touching], ids[touching], exact).max() for k in range(3))
            if far < worst:
                best, worst = keep, far
        if worst > INTERFACE_MM:
            continue
        target[members] = best
        joined += len(members) - 1
    if not joined:
        return mesh, ids, 0
    t = target[t]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
    used, renumbered = np.unique(t[keep], return_inverse=True)
    result = Mesh(v[used], renumbered.reshape(-1, 3))
    found = result.defects()
    if found["open_edges"] + found["edges_on_more_than_two"] + found["edges_same_direction"]:
        return mesh, ids, 0
    return result, ids[keep], joined


# ---------------------------------------------------------------------------------------------------------
# Points inside a mesh


def _ray_inside(mesh: Mesh, points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """For each point, whether it lies inside the closed ``mesh`` and whether it lies on its surface (within
    ``tolerance`` mm along the ray): the winding number along a ray in +x. The ray is moved off the point in
    y and z by a step of 1e-9 of the mesh's size in directions that no mesh of whole or decimal numbers
    shares, so that it passes through no edge or vertex."""
    v, t = mesh.vertices, mesh.triangles
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    normal = np.cross(b - a, c - a)
    keep = normal[:, 0] != 0.0
    a, b, c, normal = a[keep], b[keep], c[keep], normal[keep]
    low_y = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1])
    high_y = np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])
    low_z = np.minimum(np.minimum(a[:, 2], b[:, 2]), c[:, 2])
    high_z = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
    size = float(np.max(mesh.bounds[1] - mesh.bounds[0])) if len(v) else 1.0
    nudge = np.array([0.0, 0.7071067811865476, 0.5772156649015329]) * 1e-9 * size
    inside = np.zeros(len(points), dtype=bool)
    on = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(np.asarray(points, dtype=np.float64)):
        y, z = point[1] + nudge[1], point[2] + nudge[2]
        near = (low_y <= y) & (y <= high_y) & (low_z <= z) & (z <= high_z)
        if not near.any():
            continue
        pa, pb, pc, n = a[near], b[near], c[near], normal[near]

        def edge(p, q):  # twice the signed area of (p, q, ray) in the yz plane
            return (q[:, 1] - p[:, 1]) * (z - p[:, 2]) - (q[:, 2] - p[:, 2]) * (y - p[:, 1])

        e0, e1, e2 = edge(pa, pb), edge(pb, pc), edge(pc, pa)
        hit = ((e0 > 0) & (e1 > 0) & (e2 > 0)) | ((e0 < 0) & (e1 < 0) & (e2 < 0))
        if not hit.any():
            continue
        pa, n = pa[hit], n[hit]
        # The plane n . (p - a) = 0 crossed by the ray at (x, y, z).
        x = pa[:, 0] - (n[:, 1] * (y - pa[:, 1]) + n[:, 2] * (z - pa[:, 2])) / n[:, 0]
        ahead = x > point[0]
        inside[index] = int(np.sum(np.sign(n[ahead, 0]))) > 0
        on[index] = bool(np.any(np.abs(x - point[0]) <= tolerance))
    return inside, on


def _back(mesh: Mesh, source: _Source, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """The mesh voxelised back on the run's grid: the occupancy of every voxel (3 x 3 rays per voxel, as
    ``occupancy`` computes it by default) and the share of the middle third of each voxel's central ray
    that lies inside (where it is 0 or 1 the voxel's centre is outside or inside the mesh), each of shape
    (nz, ny, nx)."""
    nx, ny, nz = source.counts
    s, h = BACK_SUBDIVISION, source.h
    occupied = np.zeros((nz, ny, nx))
    middle = np.zeros((nz, ny, nx))
    layers = max(1, BACK_PIECE // (s**3 * nx * ny))
    for k0 in range(0, nz, layers):
        k1 = min(nz, k0 + layers)
        corner = source.origin + np.array([0.0, 0.0, k0 * h])
        fine = occupancy((s * nx, s * ny, s * (k1 - k0)), h / s, tuple(corner), [mesh], rays_per_side=1,
                         threads=threads).reshape(k1 - k0, s, ny, s, nx, s)  # fmt: skip
        occupied[k0:k1] = fine.mean(axis=(1, 3, 5))
        middle[k0:k1] = fine[:, s // 2, :, s // 2, :, s // 2]
    return occupied, middle


def _centres_inside(mesh: Mesh, source: _Source, middle: np.ndarray, voxels: np.ndarray) -> np.ndarray:
    """For each voxel of the mask ``voxels``: +1 when its centre lies inside the mesh, -1 outside, 0 on the
    surface (within ON_SURFACE_VOXELS voxel sides along the ray). Where the middle third of the voxel's
    central ray is wholly inside or outside the answer is read from it; elsewhere a ray is cast."""
    shares = middle[voxels]
    answer = np.where(shares >= 1.0, 1, np.where(shares <= 0.0, -1, 0)).astype(np.int64)
    unsure = (shares > 0.0) & (shares < 1.0)
    if unsure.any():
        k, j, i = np.nonzero(voxels)
        centres = source.origin + (np.stack([i, j, k], axis=1)[unsure] + 0.5) * source.h
        inside, on = _ray_inside(mesh, centres, ON_SURFACE_VOXELS * source.h)
        answer[unsure] = np.where(on, 0, np.where(inside, 1, -1))
    return answer


# ---------------------------------------------------------------------------------------------------------
# Bodies


def _clean(mesh: Mesh) -> tuple[np.ndarray, int, list[LoosePiece], int, float, list[float]]:
    """The bodies of ``mesh`` sorted out: the bodies of positive volume at least LOOSE_SHARE of the whole
    are kept, those below it are loose pieces (dropped), and a closed cavity (a body of negative volume) is
    kept when it lies inside a kept body. Returns a boolean per triangle (kept), the number of kept bodies,
    the loose pieces, the number of cavities kept, the whole volume (of the bodies of positive volume) and
    the kept bodies' volumes, largest first."""
    labels = mesh._body_labels()
    names, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.ravel()
    bodies = []
    for index in range(len(names)):
        triangles = mesh.triangles[inverse == index]
        used = np.unique(triangles)
        bodies.append(Mesh(mesh.vertices[used], np.searchsorted(used, triangles)))
    volumes = np.array([b.volume_mm3 for b in bodies])
    whole = math.fsum(v for v in volumes if v > 0.0)
    keep = np.zeros(len(bodies), dtype=bool)
    loose = []
    for index, body in enumerate(bodies):
        if volumes[index] > 0.0 and volumes[index] >= LOOSE_SHARE * whole:
            keep[index] = True
        elif volumes[index] > 0.0:
            low, high = body.bounds
            loose.append(LoosePiece(float(volumes[index]), float(volumes[index] / whole), tuple(map(float, low)),
                                    tuple(map(float, high))))  # fmt: skip
    solids = np.flatnonzero(keep)
    cavities = 0
    for index in np.flatnonzero(volumes <= 0.0):
        point = bodies[index].vertices[:1]
        for owner in solids:
            if _ray_inside(bodies[owner], point, 0.0)[0][0]:
                keep[index] = True
                cavities += 1
                break
    kept_volumes = sorted((float(volumes[i]) for i in solids), reverse=True)
    loose.sort(key=lambda piece: -piece.volume_mm3)
    return keep[inverse], len(solids), loose, cavities, whole, kept_volumes


def _subset(mesh: Mesh, chosen: np.ndarray) -> Mesh:
    """The mesh of the triangles where ``chosen`` is true, with the vertices they use."""
    if chosen.all():
        return mesh
    triangles = mesh.triangles[chosen]
    used = np.unique(triangles)
    return Mesh(mesh.vertices[used], np.searchsorted(used, triangles))


def _merged(mesh: Mesh) -> Mesh:
    """The mesh with vertices of equal coordinates joined into one."""
    unique, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    return Mesh(unique, inverse.ravel()[mesh.triangles])


def _largest_genus(mesh: Mesh) -> int:
    bodies = [b for b in mesh.bodies() if b.volume_mm3 > 0.0]
    return bodies[0].genus() if bodies else 0


# ---------------------------------------------------------------------------------------------------------
# finish


def _threads(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise InvalidValue(f"threads must be a whole number of at least 1; got {value!r}")
    return int(value)


def _surface(fine: ScalarField, level: float, threads: int, exact: _Exact):
    raw = iso_surface(fine, level, threads=threads)
    smoothed = smooth(raw, SMOOTHING_ITERATIONS)
    a, b, c = (raw.vertices[raw.triangles[:, i]] for i in range(3))
    before = np.cross(b - a, c - a)
    a, b, c = (smoothed.mesh.vertices[raw.triangles[:, i]] for i in range(3))
    after = np.cross(b - a, c - a)
    flipped = int(np.count_nonzero(np.einsum("ij,ij->i", before, after) <= 0.0))
    mesh, ids, edge_flips = _finished(smoothed.mesh, exact)
    report = {
        "iterations": float(SMOOTHING_ITERATIONS),
        "largest_move_mm": smoothed.largest_move_mm,
        "flipped_triangles": float(flipped),
        "surface_triangles": float(len(raw.triangles)),
        "slivers_flipped": float(edge_flips),
    }
    return mesh, ids, report


def finish(part: StiffPart, setup: SetUp | Problem, keep_solid: KeepSolid | None = None, *,
           threads: int | None = None) -> FinishedPart:  # fmt: skip
    """The closed mesh of the optimised part ``part`` (the ``StiffPart`` of ``stiff_part``), run on
    ``setup`` (the ``SetUp`` of ``michell.attach.set_up``, whose voxel model keeps the regions' meshes).

    1. **The field**, on the voxel centres: the result's filtered field on the design voxels; 1 on the
       voxels of a held skin (held solid without an exact region, so the surface itself must hold them);
       continued by repetition (each new voxel the mean of its known face neighbours) LAYERS_INTO_SOLID
       layers into the solid regions' voxels, and 0 deeper inside them; continued LAYERS_BEYOND layers into
       the inactive voxels and beyond the grid, and 0 beyond.
    2. **The surface**: the field refined by REFINE_FACTOR (monotone cubic), ``iso_surface`` at the level,
       SMOOTHING_ITERATIONS Taubin iterations, all before the cut and join.
    3. **Cut and join** (manifold3d, float64): the organic body intersected with the envelope (the design
       and solid regions joined), joined with the solid regions, minus the void regions. A triangle without
       area that the cut leaves (three corners on one line) is removed by flipping its longest edge.
    4. **The level**: a root search (bracket, then false position) on the volume of the finished body outside
       the exact solid and void regions, against the volume of the returned design's design voxels (its
       nominal design at 0.5 or more, not held by a solid region; skin voxels count), within 0.2 %.
    5. **Loose pieces**: bodies of positive volume below 1 % of the whole are dropped and listed; a larger
       second body is refused. Closed cavities inside the part are kept and counted.
    6. **Float32**: vertices that round to the same float32 point (the ends of a sliver's short edge, as an STL
       file would join them) are joined onto the one that keeps every face on its region's plane, and the
       triangles that lose their area are removed.
       **Gates**: closed (after merging equal vertices), no triangle without area, one body, every held
       voxel's centre inside and every void voxel's centre outside, exact interfaces (every vertex of a
       triangle taken from a region on the plane of the region's face within 1e-6 mm, and no vertex inside a
       void region's wall that is not the void mesh's own). A mesh that fails one is refused (``MeshRefused``).
       The genus is compared with the voxel body's (``GenusDiffers`` warns).
    7. **The delivered part**: the mesh voxelised back on the run's grid, each load case solved once on that
       0/1 design and compared with the returned design's (``DeliveredPartDiffers`` warns beyond 5 %).

    ``setup`` may instead be the bare ``Problem`` of a result built in Python without region meshes, with
    the ``keep_solid`` it ran with: the held voxels are then held by the surface itself, the cut is the
    grid's box, and exactness is not claimed (``FinishedPart.exact`` is False). Runs on ``threads`` threads
    (the problem's by default). Returns a ``FinishedPart``.
    """
    started = time.perf_counter()
    if not isinstance(part, StiffPart):
        raise InvalidType(f"finish takes the StiffPart that stiff_part returned; got {type(part).__name__}")
    source = _source(setup, keep_solid)
    threads = _threads(threads, int(source.problem.threads))
    n = math.prod(source.counts)
    filtered = np.asarray(part.filtered, dtype=np.float64)
    nominal = np.asarray(part.nominal, dtype=np.float64)
    if filtered.shape != (n,) or nominal.shape != (n,):
        raise InvalidValue(f"the part's fields need one value per voxel of the set-up's grid ({n})")
    seconds = {}
    shape = source.active.shape
    voxel_volume = source.h**3
    material = source.problem.material
    returned = (nominal >= 0.5).reshape(shape) & source.active
    target = float(np.count_nonzero(returned & ~source.region_held)) * voxel_volume
    if target <= 0.0:
        raise MeshRefused("level", "the returned design has no design voxel at 0.5 or more to make a surface of")

    # Steps 1 and 2: the field, refined once.
    fine = refine(_field(source, filtered), REFINE_FACTOR, REFINE_ORDER)
    highest = float(fine.values.max())
    exact = _exact(source)
    seconds["field and regions"] = time.perf_counter() - started

    # Step 4: the level, by a bracket and false position on the organic volume.
    tried: dict[float, tuple] = {}

    def organic(level: float):
        if level not in tried:
            mesh, ids, report = _surface(fine, level, threads, exact)
            chosen = _clean(mesh)[0]
            tried[level] = (_subset(mesh, chosen).volume_mm3 - exact.exact_mm3, mesh, ids, report)
        return tried[level][0] - target

    search_started = time.perf_counter()
    low_level, high_level = LEVEL_LOW, min(LEVEL_HIGH, highest * (1.0 - 1e-9))
    level = min(LEVEL_START, high_level)
    excess = organic(level)
    bracket = None
    step = 0.05
    while abs(excess) > LEVEL_TOLERANCE * target and bracket is None:
        other = level + step if excess > 0 else level - step
        other = min(max(other, low_level), high_level)
        if other == level:
            break
        other_excess = organic(other)
        if (other_excess > 0) != (excess > 0) or abs(other_excess) <= LEVEL_TOLERANCE * target:
            bracket = (level, excess, other, other_excess)
        level, excess = other, other_excess
        step *= 2.0
    if bracket is not None and abs(excess) > LEVEL_TOLERANCE * target:
        a, fa, b, fb = bracket
        side = 0
        for _ in range(LEVEL_STEPS):
            c = b - fb * (b - a) / (fb - fa)
            if not a < c < b and not b < c < a:
                c = (a + b) / 2.0
            fc = organic(c)
            level, excess = c, fc
            if abs(fc) <= LEVEL_TOLERANCE * target or len(tried) >= LEVEL_STEPS:
                break
            if (fc > 0) == (fb > 0):  # Illinois: the end that stays has its value halved
                b, fb = c, fc
                if side == -1:
                    fa /= 2.0
                side = -1
            else:
                a, fa = c, fc
                if side == 1:
                    fb /= 2.0
                side = 1
    # The level whose organic volume is closest to the target.
    level = min(tried, key=lambda key: abs(tried[key][0] - target))
    organic_mm3, mesh, ids, smoothing = tried[level]
    seconds["level search"] = time.perf_counter() - search_started
    if bracket is None and abs(organic_mm3 - target) > LEVEL_TOLERANCE * target:
        raise MeshRefused(
            "level",
            f"no level between {low_level:g} and {high_level:.4g} gives the organic volume {target:.6g} mm3: from "
            f"{min(v[0] for v in tried.values()):.6g} to {max(v[0] for v in tried.values()):.6g} mm3 over the "
            f"levels tried",
            {"target_mm3": target},
        )
    if abs(organic_mm3 / target - 1.0) > LEVEL_TOLERANCE:
        warnings.warn(
            f"the level search ended at {level:.6g} with the organic volume {organic_mm3:.6g} mm3, "
            f"{100 * (organic_mm3 / target - 1):+.3f} % from its target {target:.6g} mm3 (the volume jumps there)",
            LevelOffTarget,
            stacklevel=2,
        )

    # Step 5 and 6: loose pieces and the gates.
    gate_started = time.perf_counter()
    gates: list[Gate] = []

    def refuse(gate: Gate):
        gates.append(gate)
        if not gate.passed:
            raise MeshRefused(gate.name, gate.message, dict(gate.numbers), gates)

    chosen, kept, loose, cavities, whole, kept_volumes = _clean(mesh)
    cleaned, cleaned_ids = _subset(mesh, chosen), ids[chosen]
    cleaned, cleaned_ids, joined = _single_precision(cleaned, cleaned_ids, exact)
    smoothing = {**smoothing, "float32_vertices_joined": float(joined)}
    gates.append(Gate("envelope", exact.envelope_numbers["bodies"] == 1.0, exact.envelope_numbers,
                      f"the envelope (design and solid regions joined) is {int(exact.envelope_numbers['bodies'])} "
                      "body"))  # fmt: skip
    if not gates[-1].passed:
        raise MeshRefused(
            "envelope",
            f"the design and solid regions joined make {int(exact.envelope_numbers['bodies'])} bodies, not one: "
            "a region that neither overlaps nor shares a face with the others cannot be joined; make the regions "
            "overlap",
            dict(exact.envelope_numbers),
            gates,
        )
    merged = _merged(cleaned)
    found = merged.defects()
    edge_faults = found["open_edges"] + found["edges_on_more_than_two"] + found["edges_same_direction"]
    refuse(Gate("closed", edge_faults == 0, {k: float(found[k]) for k in
                ("open_edges", "edges_on_more_than_two", "edges_same_direction")},
                f"after merging equal vertices, {edge_faults} edges not on exactly two triangles in opposite "
                "directions"))  # fmt: skip
    refuse(Gate("no zero-area triangle", found["zero_area_triangles"] == 0,
                {"zero_area_triangles": float(found["zero_area_triangles"])},
                f"{found['zero_area_triangles']} triangles without area"))  # fmt: skip
    second = kept_volumes[1] if kept > 1 else 0.0
    refuse(Gate("one body", kept == 1,
                {"bodies": float(kept), "second_mm3": second, "second_share": second / whole if whole else 0.0,
                 "loose_pieces": float(len(loose))},
                f"{kept} bodies of at least {100 * LOOSE_SHARE:g} % of the whole ({whole:.6g} mm3); the second "
                f"holds {second:.6g} mm3; {len(loose)} loose pieces below it dropped"))  # fmt: skip
    occupied, middle = _back(cleaned, source, threads)
    held_answer = _centres_inside(cleaned, source, middle, source.held)
    void_answer = _centres_inside(cleaned, source, middle, source.void_voxels)
    held_out = int(np.count_nonzero(held_answer < 0))
    void_in = int(np.count_nonzero(void_answer > 0))
    refuse(Gate("held solid inside", held_out == 0,
                {"held_voxels": float(np.count_nonzero(source.held)), "centres_outside": float(held_out),
                 "centres_on_surface": float(np.count_nonzero(held_answer == 0))},
                f"{held_out} of {int(np.count_nonzero(source.held))} held-solid voxels have their centre outside "
                "the mesh"))  # fmt: skip
    refuse(Gate("void outside", void_in == 0,
                {"void_voxels": float(np.count_nonzero(source.void_voxels)), "centres_inside": float(void_in),
                 "centres_on_surface": float(np.count_nonzero(void_answer == 0))},
                f"{void_in} of {int(np.count_nonzero(source.void_voxels))} void voxels have their centre inside "
                "the mesh"))  # fmt: skip
    if source.exact:
        refuse(_interface_gate(cleaned, cleaned_ids, exact))
    else:
        gates.append(Gate("exact interfaces", True, {}, "not claimed: the part was built without region meshes "
                          "and cut by the grid's box"))  # fmt: skip
    genus = _largest_genus(cleaned)
    padded = np.zeros(tuple(n + 2 for n in shape))
    padded[1:-1, 1:-1, 1:-1] = returned
    voxel_body = iso_surface(ScalarField(padded, source.h, tuple(source.origin - source.h / 2.0)), 0.5,
                             threads=threads)  # fmt: skip
    voxel_genus = _largest_genus(voxel_body)
    if genus != voxel_genus:
        warnings.warn(
            f"the mesh's main body has {genus} through holes, the returned design's voxel body {voxel_genus} "
            "(counted on the surface of its voxels at 0.5 drawn by the same tetrahedra)",
            GenusDiffers,
            stacklevel=2,
        )
    seconds["gates"] = time.perf_counter() - gate_started

    # Step 7: the delivered part.
    check_started = time.perf_counter()
    back = occupied >= MAJORITY
    gained = back & ~returned
    lost = returned & ~back
    delivered_density = (back & source.active).astype(np.float64).ravel()
    problem = source.problem
    cases = []
    for case in problem.cases:
        u_result, _ = problem.solve(nominal, case.loads)
        u_back, _ = problem.solve(delivered_density, case.loads)
        c_result = float(dot(case.loads, u_result))
        c_back = float(dot(case.loads, u_back))
        cases.append(CaseCheck(case.name, float(case.weight), c_result, c_back,
                               c_back / c_result - 1.0 if c_result else math.inf))  # fmt: skip
    goal = part.result.goal if part.result is not None else None
    delivered = DeliveredCheck(
        solid_voxels=int(np.count_nonzero(returned)),
        gained=int(np.count_nonzero(gained & source.active)),
        lost=int(np.count_nonzero(lost)),
        gained_inactive=int(np.count_nonzero(gained & ~source.active)),
        cases=tuple(cases),
        goal_result=math.fsum(c.weight * c.result for c in cases),
        goal_delivered=math.fsum(c.weight * c.delivered for c in cases),
        goal_reported=None if goal is None else float(goal),
        tolerance=DELIVERED_TOLERANCE,
    )
    if not delivered.within:
        warnings.warn(
            f"the delivered part differs from the optimised one: the mesh voxelised back has a compliance "
            f"{100 * delivered.goal_difference:+.2f} % from the result's (the largest difference over the load "
            f"cases and the goal {100 * delivered.largest_difference:.2f} %, more than "
            f"{100 * DELIVERED_TOLERANCE:g} %); {delivered.gained} voxels gained, {delivered.lost} lost of "
            f"{delivered.solid_voxels}",
            DeliveredPartDiffers,
            stacklevel=2,
        )
    seconds["delivered part"] = time.perf_counter() - check_started
    seconds["total"] = time.perf_counter() - started

    bounds = part.passes[-1].bounds if part.passes else ()
    limits = part.result.limits if part.result is not None else None
    volume = cleaned.volume_mm3
    return FinishedPart(
        mesh=cleaned,
        exact=source.exact,
        level=float(level),
        organic_mm3=float(organic_mm3),
        target_mm3=target,
        level_steps=tuple((float(k), float(v[0])) for k, v in tried.items()),
        volume_mm3=volume,
        mass_g=float(material.mass_g(volume)),
        mass_limit_g=float(bounds[0]) if bounds else None,
        design_mass_g=float(limits[0]) if limits else None,
        voxel_mass_g=float(material.mass_g(np.count_nonzero(returned) * voxel_volume)),
        exact_mm3=exact.exact_mm3,
        loose_pieces=tuple(loose),
        cavities=cavities,
        gates=tuple(gates),
        genus=int(genus),
        voxel_genus=int(voxel_genus),
        delivered=delivered,
        smoothing=MappingProxyType(smoothing),
        settings=MappingProxyType({
            "refine_factor": float(REFINE_FACTOR), "smoothing_iterations": float(SMOOTHING_ITERATIONS),
            "layers_into_solid": float(LAYERS_INTO_SOLID), "layers_beyond": float(LAYERS_BEYOND),
            "level_tolerance": LEVEL_TOLERANCE, "loose_share": LOOSE_SHARE, "interface_mm": INTERFACE_MM,
            "delivered_tolerance": DELIVERED_TOLERANCE,
        }),  # fmt: skip
        seconds=MappingProxyType(seconds),
    )


def _interface_gate(mesh: Mesh, ids: np.ndarray, exact: _Exact) -> Gate:
    """Exact interfaces. Every vertex of a triangle the mesh takes from a region lies on the plane of the
    region's face it came from within INTERFACE_MM. And a void region's wall in the mesh is made of the
    void mesh's own vertices: a vertex used only by the void region's triangles (inside its wall, not on the
    edge where the wall meets another face) is a vertex of the void mesh. The same count is reported for
    the other regions without a limit: where a solid region's face and the envelope's lie in one plane, the
    outline of the organic body's cut is drawn onto that face, its vertices exactly in the plane."""
    numbers_: dict[str, float] = {}
    worst = 0.0
    void_new = 0
    for region in exact.regions:
        own = (ids >= region.first) & (ids < region.last)
        if not own.any():
            continue
        plane = ids[own] - region.first
        corners = mesh.vertices[mesh.triangles[own]]  # (m, 3, 3)
        offsets = corners - region.points[plane][:, None, :]
        largest = float(np.abs(np.einsum("ij,ikj->ik", region.normals[plane], offsets)).max())
        inner = np.setdiff1d(np.unique(mesh.triangles[own]), np.unique(mesh.triangles[~own]))
        own_vertices = {row.tobytes() for row in region.mesh.vertices}
        new = sum(1 for v in inner if mesh.vertices[v].tobytes() not in own_vertices)
        numbers_[f"{region.role} {region.name}: largest distance (mm)"] = largest
        numbers_[f"{region.role} {region.name}: vertices inside its faces not its own"] = float(new)
        worst = max(worst, largest)
        if region.role == "void":
            void_new += new
    passed = worst <= INTERFACE_MM and void_new == 0
    return Gate(
        "exact interfaces", passed, numbers_,
        f"the largest distance of a vertex taken from a region from its face's plane is {worst:.3g} mm (at most "
        f"{INTERFACE_MM:g}); {void_new} vertices inside a void region's wall are not the void mesh's own",
    )  # fmt: skip


__all__ = [
    "BACK_SUBDIVISION",
    "DELIVERED_TOLERANCE",
    "INTERFACE_MM",
    "LAYERS_BEYOND",
    "LAYERS_INTO_SOLID",
    "LEVEL_TOLERANCE",
    "LOOSE_SHARE",
    "REFINE_FACTOR",
    "REFINE_ORDER",
    "SMOOTHING_ITERATIONS",
    "CaseCheck",
    "DeliveredCheck",
    "DeliveredPartDiffers",
    "FinishedPart",
    "Gate",
    "GenusDiffers",
    "LevelOffTarget",
    "LoosePiece",
    "MeshRefused",
    "StlRefused",
    "StlRounding",
    "finish",
]
