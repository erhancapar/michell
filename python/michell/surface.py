"""From a scalar field sampled on a lattice to a smooth closed surface.

- ``ScalarField(values, spacing_mm, origin_mm=(0, 0, 0))``: samples on a regular lattice of points, the
  values as an array of shape (nz, ny, nx), so that x runs fastest as in ``Grid3d``.
- ``iso_surface(field, level, threads=1)``: the closed ``Mesh`` where the field equals ``level``, the side
  of larger values inside (the engine's marching tetrahedra; the rule is in ``docs/design/geometry.md``).
- ``refine(field, factor, order="cubic")``: the field resampled on a lattice ``factor`` times finer, by
  trilinear interpolation or by a monotone cubic one.
- ``smooth(mesh, iterations, lam=0.5, mu=-0.53, fixed=None)``: Taubin smoothing, which moves the vertices
  and never changes the triangles, with the largest move and the change of volume, as a ``Smoothing``.

Refused arguments raise ``EngineError`` with code ``invalid_input`` (``michell.errors``).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue
from michell.mesh import Mesh


def _read_only(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=True)
    values.flags.writeable = False
    return values


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise InvalidValue(f"{name} must be a positive finite number; got {value!r}")
    return float(value)


def _whole(value, name: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < least:
        raise InvalidValue(f"{name} must be a whole number of at least {least}; got {value!r}")
    return int(value)


@dataclass(frozen=True, eq=False)
class ScalarField:
    """Samples of a scalar field on a regular lattice of points ``spacing_mm`` apart.

    ``values`` has shape (nz, ny, nx): ``values[k, j, i]`` is the sample at the point
    ``origin_mm + (i, j, k) * spacing_mm``, so that in memory x runs fastest, as in ``Grid3d``. The values
    are copied as float64 and kept read-only; they must be finite. ``counts`` is (nx, ny, nz).
    """

    values: np.ndarray
    spacing_mm: float
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
            raise InvalidType(f"the values of a field are numbers; got an array of {values.dtype}")
        if values.ndim != 3 or min(values.shape) < 1:
            raise InvalidValue(
                f"the values of a field are an array of shape (nz, ny, nx), each at least 1; got shape {values.shape}"
            )
        values = _read_only(values)
        if not np.all(np.isfinite(values)):
            k, j, i = (int(c) for c in np.argwhere(~np.isfinite(values))[0])
            raise InvalidValue(f"the sample at point ({i}, {j}, {k}) is not finite", {"i": i, "j": j, "k": k})
        origin = np.asarray(self.origin_mm)
        if origin.shape != (3,) or origin.dtype == np.bool_ or not np.issubdtype(origin.dtype, np.number):
            raise InvalidValue(f"the origin is a point of three numbers (mm); got {self.origin_mm!r}")
        if not np.all(np.isfinite(origin)):
            raise InvalidValue(f"the origin must be finite; got {self.origin_mm!r}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "spacing_mm", _positive(self.spacing_mm, "spacing_mm"))
        object.__setattr__(self, "origin_mm", tuple(float(c) for c in origin))

    @property
    def counts(self) -> tuple[int, int, int]:
        """The number of points along x, y and z."""
        nz, ny, nx = self.values.shape
        return nx, ny, nz

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The coordinates of the lattice's points along x, y and z (mm), as three 1-D arrays."""
        return tuple(self.origin_mm[a] + self.spacing_mm * np.arange(n) for a, n in enumerate(self.counts))

    def __repr__(self) -> str:
        nx, ny, nz = self.counts
        return f"ScalarField({nx} x {ny} x {nz} points, {self.spacing_mm:g} mm apart, origin {self.origin_mm})"


def _field(value) -> ScalarField:
    if not isinstance(value, ScalarField):
        raise InvalidType(f"expected a michell.surface.ScalarField; got {type(value).__name__}")
    return value


def iso_surface(field: ScalarField, level: float, *, threads: int = 1) -> Mesh:
    """The closed surface where ``field`` equals ``level``, as a ``Mesh`` facing outward, with the side of
    larger values inside: marching tetrahedra (Doi and Koide 1991) on the split of every lattice cell into
    six tetrahedra around the same body diagonal.

    A sample is inside when it is at least the level. Every vertex lies on an edge of the tetrahedra whose
    ends are on different sides, by linear interpolation, kept at least a thousandth of the edge from both
    ends, and is made once for its edge; so the surface has no triangle without area and no two vertices
    at the same place, and it passes ``Mesh.closed()`` whatever the field. Every sample of the lattice's
    outer layer must be below the level (pad the field); a field whose outer layer reaches the level is
    refused, naming the sample, and so is a field that reaches the level nowhere. Runs on ``threads``
    threads; the vertices, the triangles and their order do not depend on the thread count.
    """
    field = _field(field)
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise InvalidValue(f"the level must be a number; got {level!r}")
    vertices, triangles = _engine.iso_surface(
        field.counts, field.spacing_mm, field.origin_mm, field.values.ravel(), float(level), threads
    )
    return Mesh(vertices, triangles.astype(np.int64))


# ---------------------------------------------------------------------------------------------------------
# Refining a field


def refine(field: ScalarField, factor: int, order: str = "cubic") -> ScalarField:
    """The field resampled on a lattice ``factor`` times finer: the spacing divided by ``factor``, the same
    origin, and (n - 1) * factor + 1 points along an axis of n points, so that every old point is a new one
    and keeps its sample exactly. Between the old points the new samples are interpolated one axis after
    the other (x, then y, then z), each axis by one of two orders:

    - ``"linear"``: linear interpolation along each axis, which makes the field trilinear in each cell;
    - ``"cubic"``: piecewise cubic Hermite interpolation with the slopes of Fritsch and Butland (1984) at
      the samples, which meet the condition of Fritsch and Carlson (1980) for a monotone interpolant. The
      slope at a sample is the harmonic mean 2 / (1 / d0 + 1 / d1) of the differences d0 and d1 to its two
      neighbours, and 0 where they differ in sign or one is 0; at the two ends of a line it is the
      difference to the one neighbour. It reproduces fields that are linear along an axis, and between two
      samples it never leaves the range of those two samples. Its error falls with the cube of the spacing
      where the field is monotone along the line, and with the square in the two end intervals of a line
      and next to a sample where the field turns (where the slope is 0), as the linear order's does.

    Neither order overshoots. At a step of a 0/1 field (0, 0, 1, 1 along an axis) the cubic order gives
    3 t^2 - 2 t^3 between the 0 and the 1, which crosses 0.5 half-way, and exactly 0 and 1 on either side;
    a 0/1 field stays within [0, 1] everywhere, and an outer layer below a level stays below it, so the
    refined field can be handed to ``iso_surface`` whenever the field could. Each new sample is also held
    within the range of the two samples it lies between, which removes the rounding of the formula.
    """
    field = _field(field)
    factor = _whole(factor, "factor", 1)
    if order not in ("linear", "cubic"):
        raise InvalidValue(f"the order is 'linear' or 'cubic'; got {order!r}")
    values = np.array(field.values)
    for axis in (2, 1, 0):  # x, y, z in an array of shape (nz, ny, nx)
        values = _refine_axis(values, axis, factor, order)
    return ScalarField(values, field.spacing_mm / factor, field.origin_mm)


def _refine_axis(values: np.ndarray, axis: int, factor: int, order: str) -> np.ndarray:
    lines = np.moveaxis(values, axis, -1)
    n = lines.shape[-1]
    if factor == 1 or n == 1:
        return values
    low, high = lines[..., :-1], lines[..., 1:]
    difference = high - low
    out = np.empty(lines.shape[:-1] + ((n - 1) * factor + 1,))
    out[..., -1] = lines[..., -1]
    if order == "cubic":
        slope = np.zeros(lines.shape)
        before, after = difference[..., :-1], difference[..., 1:]
        same_sign = before * after > 0
        with np.errstate(divide="ignore"):
            harmonic = 2.0 / (1.0 / np.where(same_sign, before, 1.0) + 1.0 / np.where(same_sign, after, 1.0))
        slope[..., 1:-1] = np.where(same_sign, harmonic, 0.0)
        slope[..., 0], slope[..., -1] = difference[..., 0], difference[..., -1]
        slope_low, slope_high = slope[..., :-1], slope[..., 1:]
    least, most = np.minimum(low, high), np.maximum(low, high)
    for s in range(factor):
        t = s / factor
        if order == "linear":
            new = low + t * difference
        else:
            new = low + (3.0 - 2.0 * t) * t * t * difference + t * (1.0 - t) ** 2 * slope_low
            new += t * t * (t - 1.0) * slope_high
        out[..., s : (n - 1) * factor : factor] = np.minimum(np.maximum(new, least), most)
    return np.moveaxis(out, -1, axis)


# ---------------------------------------------------------------------------------------------------------
# Taubin smoothing


@dataclass(frozen=True)
class Smoothing:
    """The result of ``smooth``: the smoothed ``mesh`` (the same triangles, moved vertices), the number of
    ``iterations``, the largest distance a vertex moved (mm) and the volume before and after (mm^3).
    ``volume_change`` is the relative change, after over before minus 1 (None when the volume before is 0).
    """

    mesh: Mesh
    iterations: int
    largest_move_mm: float
    volume_before_mm3: float
    volume_after_mm3: float

    @property
    def volume_change(self) -> float | None:
        if self.volume_before_mm3 == 0.0:
            return None
        return self.volume_after_mm3 / self.volume_before_mm3 - 1.0


def smooth(mesh: Mesh, iterations: int, *, lam: float = 0.5, mu: float = -0.53, fixed=None) -> Smoothing:
    """Taubin smoothing (Taubin 1995, "A signal processing approach to fair surface design"): ``iterations``
    times, a step x += lam L(x) followed by a step x += mu L(x), where L(x) at a vertex is the mean of its
    neighbours (the vertices it shares an edge with) minus the vertex itself. The step with lam > 0 smooths
    and shrinks; the step with mu < -lam swells the surface back, so that the shape keeps its size while
    short ripples (such as the facets of an iso-surface) die out. One iteration is one step of each.

    ``fixed`` is an optional boolean array with one entry per vertex: those vertices do not move (they still
    pull their neighbours). The triangles are never changed, so a closed mesh stays closed. Vertices that no
    triangle uses do not move.
    """
    if not isinstance(mesh, Mesh):
        raise InvalidType(f"smooth smooths a michell.mesh.Mesh; got {type(mesh).__name__}")
    iterations = _whole(iterations, "iterations", 0)
    for name, value in (("lam", lam), ("mu", mu)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise InvalidValue(f"{name} must be a finite number; got {value!r}")
    if not (lam > 0 and mu < -lam):
        raise InvalidValue(f"Taubin smoothing needs lam > 0 and mu < -lam; got lam {lam!r}, mu {mu!r}")
    n = len(mesh.vertices)
    moves = np.ones(n)
    if fixed is not None:
        mask = np.asarray(fixed)
        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise InvalidValue(
                f"fixed is a boolean array with one entry per vertex ({n}); got {mask.dtype} of shape {mask.shape}"
            )
        moves[mask] = 0.0
    # Every pair of neighbours once (as low * n + high), then in both directions: vertex `source` gets the
    # coordinates of vertex `target`.
    t = mesh.triangles
    sides = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    keys = np.unique(sides.min(axis=1) * max(n, 1) + sides.max(axis=1))
    low, high = keys // max(n, 1), keys % max(n, 1)
    source, target = np.concatenate([low, high]), np.concatenate([high, low])
    degree = np.bincount(source, minlength=n)
    moves[degree == 0] = 0.0
    weight = moves / np.maximum(degree, 1)

    def step(x: np.ndarray, factor: float) -> np.ndarray:
        # x is (3, n): each coordinate contiguous, which makes the gathers and sums about twice as fast.
        change = np.empty_like(x)
        for c in range(3):
            change[c] = weight * np.bincount(source, weights=x[c][target], minlength=n) - moves * x[c]
        return x + factor * change

    start = np.array(mesh.vertices)
    x = np.ascontiguousarray(start.T)
    for _ in range(iterations):
        x = step(step(x, lam), mu)
    x = np.ascontiguousarray(x.T)
    smoothed = Mesh(x, mesh.triangles)
    largest = float(np.linalg.norm(x - start, axis=1).max()) if n else 0.0
    return Smoothing(smoothed, iterations, largest, mesh.volume_mm3, smoothed.volume_mm3)


__all__ = ["ScalarField", "Smoothing", "iso_surface", "refine", "smooth"]
