"""Regions as voxels: closed meshes in mm turned into the voxel model an optimisation runs on.

- ``occupancy(counts, voxel_mm, origin_mm, adding, subtracting=(), ...)``: the share of every voxel of a
  grid that lies inside the union of some meshes minus the union of others (the engine's voxeliser; the
  rule is in ``docs/design/geometry.md``).
- ``voxel_model(design, solid=None, void=None, *, voxel_mm, mirror=(), threads=1)``: the grid around the
  design and solid regions, the active voxels, the voxels held solid, a volume table and the bodies of the
  active set, as a ``VoxelModel``.
- ``VoxelVolumeDiffers``: the warning ``voxel_model`` gives when the active voxels' volume differs from the
  part's sampled volume by more than 5 %.

The classes are taken from the union of the regions, not from each region alone: the part is (design or
solid) minus void; a voxel is active when the part fills at least half of it, and an active voxel is held
solid when the solid regions (minus void) fill at least half of what the part fills of it. A solid region
nested inside the design region therefore overrides it, and two regions that meet inside a voxel lose no
layer of voxels between them.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from michell import _engine
from michell.analysis3d import Grid3d
from michell.errors import InvalidType, InvalidValue
from michell.mesh import Mesh, smallest_joined

# A voxel belongs to a class when the class fills at least this share of it (or, for the held-solid class,
# of what the part fills of it); a tie counts as inside.
MAJORITY = 0.5

# ``voxel_model`` warns when the active voxels' volume differs from the part's sampled volume by more than this.
VOLUME_TOLERANCE = 0.05

# With ``mirror``, a body whose volume without a mirror image is more than this share of its sampled volume
# is refused.
MIRROR_TOLERANCE = 0.005

# The counts of the grid are ceil(size / voxel_mm - COUNT_SLACK), so that a size of a whole number of voxels
# given in decimal millimetres is not rounded up by one.
COUNT_SLACK = 1e-9


class VoxelVolumeDiffers(UserWarning):
    """Issued by ``voxel_model`` when the volume of the active voxels differs from the part's sampled volume
    (the sum of its occupancy times the voxel volume) by more than 5 %: the voxels are too coarse for the
    part's thinnest features. The message gives both volumes; the model is returned."""


def _mesh(value, label: str) -> Mesh:
    if not isinstance(value, Mesh):
        raise InvalidType(f"{label} must be a michell.mesh.Mesh; got {type(value).__name__}")
    try:
        return value.closed()
    except InvalidValue as error:
        raise InvalidValue(f"{label}: {error.message}", error.details) from None


def _bodies(values, label: str) -> list[tuple[np.ndarray, np.ndarray]]:
    bodies = []
    for index, mesh in enumerate(values):
        if not isinstance(mesh, Mesh):
            raise InvalidType(f"body {index} of {label} must be a michell.mesh.Mesh; got {type(mesh).__name__}")
        bodies.append((np.ascontiguousarray(mesh.vertices), np.ascontiguousarray(mesh.triangles, dtype=np.float64)))
    return bodies


def occupancy(
    counts: Sequence[int],
    voxel_mm: float,
    origin_mm: Sequence[float],
    adding: Sequence[Mesh],
    subtracting: Sequence[Mesh] = (),
    *,
    rays_per_side: int = 3,
    threads: int = 1,
) -> np.ndarray:
    """The occupancy of every voxel of a grid of ``counts`` (nx, ny, nz) voxels of side ``voxel_mm`` (mm),
    with the corner of voxel (0, 0, 0) at ``origin_mm`` in the meshes' coordinates: the share of the voxel
    inside the union of the ``adding`` meshes minus the union of the ``subtracting`` meshes, one float64
    between 0 and 1 per voxel, indexed as in ``Grid3d`` (x fastest).

    The meshes must be closed and face outward (``Mesh.closed()`` returns them so). Rays run along x through
    every voxel column, ``rays_per_side`` x ``rays_per_side`` of them at the centres of the column's
    sub-squares; on each ray the length inside every voxel is exact, so the occupancy is exact in x and
    sampled in y and z. The numbers do not depend on ``threads`` or on the order of the triangles.
    """
    if len(counts) != 3:
        raise InvalidValue(f"counts are three whole numbers (nx, ny, nz); got {counts!r}")
    if len(origin_mm) != 3:
        raise InvalidValue(f"the origin is a point of three numbers (mm); got {origin_mm!r}")
    return _engine.voxel_occupancy(
        tuple(int(c) for c in counts), float(voxel_mm), tuple(float(c) for c in origin_mm),
        _bodies(adding, "adding"), _bodies(subtracting, "subtracting"), rays_per_side, threads,
    )  # fmt: skip


@dataclass(frozen=True)
class VolumeRow:
    """One line of the volume table of a ``VoxelModel``, volumes in mm^3.

    ``role`` is ``"part"`` (design or solid, minus void), ``"held solid"`` (the class held solid), or the
    role of a named region (``"design"``, ``"solid"``, ``"void"``). ``exact_mm3`` is the volume of the mesh
    where one mesh defines the line (a region; the part when it is the design region alone; the held-solid
    class when it is one solid region and there is no void), else None. ``sampled_mm3`` is the sum of the
    occupancy times the voxel volume; ``voxels_mm3`` the volume of the voxels of the class (for a region:
    the voxels it fills by majority on its own). The ``*_vs_*`` properties are relative differences, the
    first over the second minus 1 (None where the second is missing or 0).
    """

    name: str
    role: str
    exact_mm3: float | None
    sampled_mm3: float
    voxels_mm3: float

    @staticmethod
    def _relative(value: float, reference: float | None) -> float | None:
        if reference is None or reference == 0.0:
            return None
        return value / reference - 1.0

    @property
    def sampled_vs_exact(self) -> float | None:
        return self._relative(self.sampled_mm3, self.exact_mm3)

    @property
    def voxels_vs_sampled(self) -> float | None:
        return self._relative(self.voxels_mm3, self.sampled_mm3)

    @property
    def voxels_vs_exact(self) -> float | None:
        return self._relative(self.voxels_mm3, self.exact_mm3)


@dataclass(frozen=True)
class VoxelModel:
    """The voxel model of a part, as ``voxel_model`` returns it.

    - ``grid``: a ``Grid3d`` of ``counts`` voxels of side ``voxel_mm``, its ``active`` voxels those of the
      part; ``origin_mm`` is the grid's corner (node (0, 0, 0)) in the meshes' coordinates. ``to_grid`` and
      ``to_mesh`` move points between the two, so that ``michell.select`` (which puts the grid's corner at
      the origin) works on the model.
    - ``keep_solid``: one boolean per voxel, true for the voxels held solid, the mask ``KeepSolid`` takes;
      ``design``: the active voxels that are not held.
    - ``part_occupancy``, ``solid_occupancy`` (the solid regions minus void) and ``region_occupancy`` (each
      named region alone, by name; the design region is ``"design"``): the occupancy fields the classes and
      the table come from (after mirroring, with ``mirror``).
    - ``volumes``: the volume table, a ``VolumeRow`` for the part, the held-solid class and each region.
    - ``bodies``: the number of bodies of the active set joined through voxel faces, and ``body_sizes``
      their sizes in voxels, largest first, when there is more than one (else empty).
    - ``mirror`` and ``threads``: as given.
    """

    grid: Grid3d
    voxel_mm: float
    counts: tuple[int, int, int]
    origin_mm: tuple[float, float, float]
    keep_solid: np.ndarray
    part_occupancy: np.ndarray
    solid_occupancy: np.ndarray
    region_occupancy: Mapping[str, np.ndarray]
    volumes: tuple[VolumeRow, ...]
    bodies: int
    body_sizes: tuple[int, ...]
    mirror: tuple[str, ...]
    threads: int

    @property
    def active(self) -> np.ndarray:
        """One boolean per voxel, true for the voxels of the part."""
        return np.asarray(self.grid.active) == 1.0

    @property
    def design(self) -> np.ndarray:
        """One boolean per voxel, true for the active voxels that are not held solid."""
        return self.active & ~self.keep_solid

    def to_grid(self, points_mm) -> np.ndarray:
        """Points (mm, the last axis of length 3) in the meshes' coordinates, moved to the grid's, whose
        origin is the grid's corner (the positions ``michell.select`` takes)."""
        return np.asarray(points_mm, dtype=np.float64) - np.array(self.origin_mm)

    def to_mesh(self, points_mm) -> np.ndarray:
        """Points (mm) in the grid's coordinates, moved to the meshes'."""
        return np.asarray(points_mm, dtype=np.float64) + np.array(self.origin_mm)

    def volume(self, name: str, role: str | None = None) -> VolumeRow:
        """The line of the volume table with this name (and role, where a name is used twice)."""
        for row in self.volumes:
            if row.name == name and (role is None or row.role == role):
                return row
        raise InvalidValue(f"the volume table has no line {name!r}")


def _named(regions, role: str) -> dict[str, Mesh]:
    if regions is None:
        return {}
    if not isinstance(regions, Mapping):
        raise InvalidType(f"the {role} regions are a dict of name to michell.mesh.Mesh; got {type(regions).__name__}")
    named = {}
    for name, mesh in regions.items():
        if not isinstance(name, str) or not name:
            raise InvalidValue(f"the {role} regions are named by non-empty strings; got {name!r}")
        named[name] = _mesh(mesh, f"the {role} region {name!r}")
    return named


def _mirror_planes(mirror) -> tuple[str, ...]:
    if isinstance(mirror, str) or not isinstance(mirror, (tuple, list)):
        raise InvalidValue(f"mirror is a tuple of planes, any of 'x', 'y', 'z', such as ('y',); got {mirror!r}")
    if any(plane not in ("x", "y", "z") for plane in mirror) or len(set(mirror)) != len(mirror):
        raise InvalidValue(f"mirror holds distinct planes among 'x', 'y', 'z'; got {mirror!r}")
    return tuple(mirror)


def _mirrored(field: np.ndarray, counts: tuple[int, int, int], plane: str) -> np.ndarray:
    """``field`` (one value per voxel) reflected in the plane across ``plane`` through the grid's centre."""
    nx, ny, nz = counts
    axis = {"x": 2, "y": 1, "z": 0}[plane]
    return np.flip(field.reshape(nz, ny, nx), axis=axis).ravel()


def _face_joined_bodies(active: np.ndarray, counts: tuple[int, int, int]) -> list[int]:
    """The sizes in voxels of the bodies of ``active`` joined through voxel faces, largest first."""
    nx, ny, nz = counts
    index = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    solid = active.reshape(nz, ny, nx)
    first, second = [], []
    for axis in (2, 1, 0):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        both = solid[tuple(low)] & solid[tuple(high)]
        first.append(index[tuple(low)][both])
        second.append(index[tuple(high)][both])
    label = smallest_joined(nx * ny * nz, np.concatenate(first), np.concatenate(second))
    sizes = np.bincount(label[active.ravel()])
    return sorted((int(s) for s in sizes[sizes > 0]), reverse=True)


def voxel_model(
    design: Mesh,
    solid: Mapping[str, Mesh] | None = None,
    void: Mapping[str, Mesh] | None = None,
    *,
    voxel_mm: float,
    mirror: Sequence[str] = (),
    threads: int = 1,
) -> VoxelModel:
    """The voxel model of a part: ``design`` (a ``Mesh``, where material may be), ``solid`` (a dict of name to
    ``Mesh``, where material must be) and ``void`` (likewise, where it must not be), on cube voxels of side
    ``voxel_mm`` (mm), on ``threads`` threads.

    Every mesh must pass ``Mesh.closed()``. The grid covers the box around the design and solid meshes:
    ceil(size / voxel_mm - 1e-9) voxels along each axis, centred on that box. The occupancies come from the
    voxeliser (``occupancy``): the part, (design or solid) minus void; the solid regions minus void; and
    each region alone. A voxel is active when the part fills at least half of it; an active voxel is held
    solid when the solid regions fill at least half of what the part fills of it, else it is a design
    voxel (ties at exactly one half count as inside).

    With ``mirror`` (any of "x", "y", "z", such as ``("y",)``; planes through the centre of the box), each
    region's occupancy is first compared with its mirror image: the volume it has without a mirror image
    (half the sum of |f - M f| times the voxel volume) must be at most 0.5 % of its sampled volume, or the
    mirror image must match another region of the same role as closely (two regions that mirror each other).
    A region that passes neither is refused, naming it, the plane and the volume that differs. Then every
    occupancy field is averaged with its mirror image (a region with a partner with its partner's), so that
    the classes mirror exactly.

    Refused (``InvalidValue``): a region that fills no voxel by majority on its own (naming it, its smallest
    extent and the voxel size), a part that fills no voxel, names used twice (``"design"`` is the design
    region's). Warned (``VoxelVolumeDiffers``): the active voxels' volume more than 5 % off the part's
    sampled volume. Returns a ``VoxelModel``.
    """
    design = _mesh(design, "the design region")
    solids, voids = _named(solid, "solid"), _named(void, "void")
    names = ["design", *solids, *voids]
    if len(set(names)) != len(names):
        raise InvalidValue(f"each region needs its own name (the design region is 'design'); got {names}")
    if (
        isinstance(voxel_mm, bool)
        or not isinstance(voxel_mm, numbers.Real)
        or not (math.isfinite(voxel_mm) and voxel_mm > 0)
    ):
        raise InvalidValue(f"voxel_mm must be a positive finite number; got {voxel_mm!r}")
    h = float(voxel_mm)
    planes = _mirror_planes(mirror)
    if isinstance(threads, bool) or not isinstance(threads, numbers.Integral) or threads < 1:
        raise InvalidValue(f"threads must be a whole number of at least 1; got {threads!r}")
    threads = int(threads)

    # The grid: the box around the design and solid meshes, a whole number of voxels, centred on it.
    boxes = np.array([mesh.bounds for mesh in [design, *solids.values()]])
    low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    counts = tuple(max(1, math.ceil((hi - lo) / h - COUNT_SLACK)) for lo, hi in zip(low, high))
    origin = tuple(float(c) for c in (low + high) / 2.0 - np.array(counts) * h / 2.0)

    def field(adding, subtracting=()):
        return occupancy(counts, h, origin, list(adding), list(subtracting), threads=threads)

    roles = {"design": "design", **{n: "solid" for n in solids}, **{n: "void" for n in voids}}
    meshes = {"design": design, **solids, **voids}
    regions = {name: field([mesh]) for name, mesh in meshes.items()}
    part = field([design, *solids.values()], voids.values())
    held_field = field(solids.values(), voids.values()) if solids else np.zeros(math.prod(counts))

    for name, values in regions.items():
        if not np.any(values >= MAJORITY):
            extent = meshes[name].bounds
            smallest = float(np.min(extent[1] - extent[0]))
            raise InvalidValue(
                f"the {roles[name]} region {name!r} fills no voxel by majority on its own: its smallest extent is "
                f"{smallest:g} mm, the voxel side {h:g} mm; use smaller voxels or a thicker region",
                {"smallest_extent_mm": smallest, "voxel_mm": h},
            )

    voxel_volume = h**3
    for plane in planes:
        centre = float((low + high)["xyz".index(plane)] / 2.0)
        partner: dict[str, str] = {}
        for name, values in regions.items():
            if name in partner:
                continue
            own_volume = float(values.sum()) * voxel_volume
            candidates = [name] + [other for other in regions if other != name and roles[other] == roles[name]
                                   and other not in partner]  # fmt: skip
            differs = []
            for other in candidates:
                difference = float(np.abs(values - _mirrored(regions[other], counts, plane)).sum()) / 2.0 * voxel_volume
                differs.append(difference)
                if difference <= MIRROR_TOLERANCE * own_volume:
                    partner[name], partner[other] = other, name
                    break
            else:
                raise InvalidValue(
                    f"the {roles[name]} region {name!r} is not mirror-symmetric about the plane {plane} = {centre:g} mm: "
                    f"{differs[0]:.4g} mm3 of its {own_volume:.4g} mm3 has no mirror image (more than "
                    f"{100 * MIRROR_TOLERANCE:g} %)",
                    {"plane_mm": centre, "differs_mm3": differs[0], "sampled_mm3": own_volume},
                )
        regions = {name: (values + _mirrored(regions[partner[name]], counts, plane)) / 2.0
                   for name, values in regions.items()}  # fmt: skip
        part = (part + _mirrored(part, counts, plane)) / 2.0
        held_field = (held_field + _mirrored(held_field, counts, plane)) / 2.0

    active = part >= MAJORITY
    if not active.any():
        raise InvalidValue("the part (design or solid, minus void) fills no voxel by majority", {"voxel_mm": h})
    keep_solid = active & (held_field >= MAJORITY * part)

    exact_part = design.volume_mm3 if not solids and not voids else None
    exact_held = next(iter(solids.values())).volume_mm3 if len(solids) == 1 and not voids else None
    volumes = [
        VolumeRow("part", "part", exact_part, float(part.sum()) * voxel_volume, int(active.sum()) * voxel_volume),
        VolumeRow("held solid", "held solid", exact_held, float(held_field.sum()) * voxel_volume,
                  int(keep_solid.sum()) * voxel_volume),
    ]  # fmt: skip
    for name, values in regions.items():
        volumes.append(VolumeRow(name, roles[name], meshes[name].volume_mm3, float(values.sum()) * voxel_volume,
                                 int(np.count_nonzero(values >= MAJORITY)) * voxel_volume))  # fmt: skip
    difference = volumes[0].voxels_vs_sampled
    if difference is not None and abs(difference) > VOLUME_TOLERANCE:
        warnings.warn(
            f"the active voxels hold {volumes[0].voxels_mm3:.4g} mm3, the part's sampled volume is "
            f"{volumes[0].sampled_mm3:.4g} mm3 ({100 * difference:+.2f} %, more than {100 * VOLUME_TOLERANCE:g} %): "
            f"the voxels of {h:g} mm are coarse for the part's thinnest features",
            VoxelVolumeDiffers,
            stacklevel=2,
        )

    sizes = _face_joined_bodies(active, counts)

    def frozen(values: np.ndarray) -> np.ndarray:
        values = np.array(values, copy=True)
        values.flags.writeable = False
        return values

    return VoxelModel(
        grid=Grid3d(counts[0], counts[1], counts[2], h, active),
        voxel_mm=h,
        counts=counts,
        origin_mm=origin,
        keep_solid=frozen(keep_solid),
        part_occupancy=frozen(part),
        solid_occupancy=frozen(held_field),
        region_occupancy=MappingProxyType({name: frozen(values) for name, values in regions.items()}),
        volumes=tuple(volumes),
        bodies=len(sizes),
        body_sizes=tuple(sizes) if len(sizes) > 1 else (),
        mirror=planes,
        threads=threads,
    )


__all__ = [
    "MAJORITY",
    "MIRROR_TOLERANCE",
    "VOLUME_TOLERANCE",
    "VolumeRow",
    "VoxelModel",
    "VoxelVolumeDiffers",
    "occupancy",
    "voxel_model",
]
