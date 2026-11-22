"""The minimum member size as a limit, and the threshold that a wanted member size gives.

Source: Zhou, Lazarov, Wang and Sigmund, "Minimum length scale in topology optimization by geometric
constraints", Comput. Methods Appl. Mech. Engrg. 293 (2015) 266-282, doi:10.1016/j.cma.2015.05.003
(equation and figure numbers below are those of that paper). The constraint on the solid phase (eqs. 8,
10 and 12),

    g = (1/n) sum_e I_e [min(rho~_e - eta_e, 0)]^2 <= eps,      I_e = rho_bar_e exp(-c |grad rho~|_e^2),

asks that the filtered field rho~ reach at least ``eta_e`` wherever it has a ridge inside the solid, so
that every member is at least as wide as a member whose filtered field just reaches ``eta_e``. It is
computed in the engine (``crates/michell/src/member_size.rs``, where the gradient and its units are
described): ``rho_bar`` is the projection of the filtered field, the gradient is taken per element side
between neighbouring active elements, and n is the number of active elements.

On a part. The paper states the constraint, with one threshold, for a box that is all design space. On a
part it can be impossible to meet: the solid the user holds fixed counts against it, and where the part
itself is thinner than the asked size (between two empty faces, beside an empty hole) no design reaches the
threshold. The threshold is therefore given per element, built from the asked one by two rules, and the
fully solid part gives g = 0 exactly, whatever its shape, so that the bound can always be met:

- The threshold is capped by what the part can reach: ``eta_e = min(threshold, reach_e)``, where ``reach``
  is the filtered field of the fully solid part. The filter is linear with weights that are not negative,
  so no design gives an element more than its reach. With closed faces and closed inactive elements the
  reach is 1 everywhere and nothing changes.
- Elements the user holds solid (``keep_solid``) are not judged, ``eta_e = 0``: the optimiser cannot change
  them. The design elements attached to them are judged as before, so they still take filtered density
  from the held solid; a void design next to a held wall gives a small g, not 0 (1.6e-5 for a wall one
  element thick at beta 8 and R = 3 elements, 1e-12 to 1e-11 at beta 32).

What the cap costs. In a region of the part thinner than the asked size the limit asks that the solid
fill the part locally, to within the filter radius, which is stricter than asking for a member of the
asked width. Near a flat empty face of a thick part it changes nothing for members: at the depth d below
such a face the fully solid part reaches ``1/2 + member_threshold(2 d) / 2``, which is below the threshold
only in a layer shallower than half the asked size (for the cone filter in 3D at the size R, threshold
0.8125, the layer is 0.347 R deep), while the ridge of a member at least the asked size thick lies at
least half the size deep.

How it is used. The limit reads the filtered field, so it sits on a design whose own filter chain is
empty, beside the design the goal reads:

    density_filter = michell.DensityFilter(grid, radius)
    projection = michell.Projection(32.0)
    designs = {"filtered": [], "nominal": [projection]}
    limit = michell.Limit(michell.on("filtered", michell.MemberSize(density_filter, projection,
                          michell.member_threshold(density_filter, size))), at_most=eps)

It reads no physics, so the filtered design is never solved. The paper switches the limit on only after
a topology has formed, at a high, fixed sharpness of the projection, and takes ``eps`` from its figure 8
for the filter radius in elements; ``examples/mbb_beam_member_size.py`` and verification row 27 follow a
procedure written out in full. The constraint on holes (the void phase, eqs. 9 and 11) is not provided.
Scope: the limit is switched on after a layout has formed and reshapes its members; where the asked size
cannot be reached without changing the layout, it does not get there (in trials on the MBB beam of row 27
at a mass fraction of 0.3 with sizes of 7 and 8 elements, the bound was missed by a factor of 2 to 3 and the
compliance was several times higher), so a run must check that the bound is met.

The limit is supported with the density filter (the cone filter of the paper, eq. 6). With the Helmholtz
filter it is experimental: the paper derives ``c`` and the values of ``eps`` for the cone filter only.

Two properties to know:

- What lies beyond each face of the grid is the density filter's ``outside`` (see ``DensityFilter``). At
  a closed face (the default) the filter averages over the elements inside the grid only, which acts like
  a mirror: right at a symmetry plane, but a member that lies along a free face then looks to the filter
  like a member twice as wide, so the limit holds it to about half the size. A strip 0.75 times the size
  thick gives g of 1e-16 against a closed face; against an empty face it gives more than 10000 eps, as it
  does in the interior (verification row 26). Free faces of the part should therefore be empty. The gradient of the
  limit is taken inside the grid only, one-sided at a face, whichever the face's choice.
- ``member_threshold`` is the value of the continuous filter. On the grid a member exactly ``size_mm``
  thick reaches slightly less (0.777 against 0.780 for a strip 8 elements thick and R = 8 elements), so the
  limit asks for members a fraction of an element thicker than the size.
"""

from __future__ import annotations

import math

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue
from michell.filters import DensityFilter, HelmholtzFilter, KeepSolid, Projection
from michell.modules import Partials
from michell.row import Row, compared


def member_threshold(filter, size_mm: float) -> float:
    """The value of the filtered field on the mid-plane of an infinite plate (3D grid) or on the centre line
    of an infinite strip (2D grid) of thickness ``size_mm``, solid inside and empty outside, filtered by
    ``filter``. A member thinner than ``size_mm`` cannot reach it, so it is the threshold ``eta_e`` of
    ``MemberSize`` for that member size. The value is that of the continuous filter; on the grid it is
    approached as the radius grows in elements.

    The density filter weighs a point at distance s by w(s) = R - s inside the radius R. With
    t = size_mm / (2 R), the value is the share of the filter's weight that lies within the distance
    t R of a plane (3D) or of a line (2D), and 1 for t >= 1:

    - 3D (a ball): the weight of the layer at height z is 2 pi (R^3 / 6 - R z^2 / 2 + |z|^3 / 3) (integrate
      w over the disc of the layer, with u = sqrt(rho^2 + z^2)); the whole weight is pi R^4 / 3. The share
      within |z| <= t R is

          2 (t - t^3 + t^4 / 2),          0.8125 at size_mm = R.

    - 2D (a disc): the weight of the line at height y is R X - y^2 arcosh(R / |y|) with X = sqrt(R^2 - y^2)
      (integrate w along the line); the whole weight is pi R^3 / 3. Integrating the line weight by parts
      gives the share within |y| <= t R,

          (2 / pi) (2 t sqrt(1 - t^2) + arcsin t - t^3 arcosh(1 / t)),      0.780 at size_mm = R.

    The Helmholtz filter (3D grid) solves -r^2 psi'' + psi = 1 inside the plate and 0 outside, with
    r = R / (2 sqrt 3) (``filter.length``); its Green's function across the plate is exp(-|z| / r) / (2 r),
    whose weight within |z| <= size_mm / 2 is

          1 - exp(-size_mm / (2 r)).
    """
    size = float(size_mm)
    if not (math.isfinite(size) and size >= 0.0):
        raise InvalidValue(f"the member size must be a number at least 0; got {size_mm!r}")
    if isinstance(filter, HelmholtzFilter):
        return 1.0 - math.exp(-size / (2.0 * filter.length))
    if not isinstance(filter, _engine.DensityFilter):
        raise InvalidType(f"member_threshold takes a DensityFilter or a HelmholtzFilter; got {type(filter).__name__}")
    t = size / (2.0 * filter.radius)
    if t >= 1.0:
        return 1.0
    if filter.is_3d:
        return 2.0 * (t - t**3 + t**4 / 2.0)
    if t == 0.0:
        return 0.0
    return 2.0 / math.pi * (2.0 * t * math.sqrt(1.0 - t * t) + math.asin(t) - t**3 * math.acosh(1.0 / t))


class MemberSize:
    """The minimum member size as a limit (Zhou, Lazarov, Wang and Sigmund 2015, eqs. 8, 10 and 12), with one
    threshold per element:

        g = (1/n) sum_e I_e [min(rho~_e - eta_e, 0)]^2,     I_e = P(rho~_e) exp(-c |grad rho~|_e^2),

    a response that reads no physics and is meant as a limit ``g <= eps`` on a design whose filter chain is
    empty, so that ``rho~`` is the filtered field (see the module's docstring for an example).

    ``MemberSize(filter, projection, threshold, c=None, keep_solid=None)``: ``filter`` is the filter that
    makes the filtered field (its radius R and element size h give the default ``c = (R / h)^4``, the paper's
    choice, with the gradient taken per element side); ``projection`` is the projection P of the design the
    goal reads; ``threshold`` is the value a member of the asked size reaches, usually
    ``member_threshold(filter, size_mm)``, a number between 0 and 1; ``keep_solid`` (a ``KeepSolid``, or a
    boolean array with one entry per element) marks the elements the user holds solid. The value is
    dimensionless.

    The thresholds ``eta_e`` are built from these on the first evaluation, when the grid is known (the
    filter does not hold it), and kept in ``thresholds`` (``None`` before then):

    - ``reach = filter.forward(grid.active)`` is the filtered field of the fully solid part, and
      ``eta_e = min(threshold, reach_e)``. The filter is linear with weights that are not negative, so no
      design gives an element more than its reach; where the part cannot hold a member of the asked size,
      the limit asks for what the fully solid part reaches there.
    - ``eta_e = 0`` on the elements of ``keep_solid``: the optimiser cannot change them, so they are not
      judged. The design elements attached to them are judged as before.

    The fully solid part therefore gives g = 0 exactly, whatever its shape, and the bound can always be met.
    The module's docstring states what this costs.

    Supported with the ``DensityFilter`` (the cone filter the paper uses, eq. 6). With the
    ``HelmholtzFilter`` it is experimental (``experimental`` is then true): the paper's ``c`` and its values
    of ``eps`` (figure 8) were found with the cone filter only.
    """

    name = "member size"
    unit = "1"
    reads = ()

    def __init__(
        self, filter, projection: Projection, threshold: float, c: float | None = None, keep_solid=None
    ):
        if not isinstance(filter, (_engine.DensityFilter, HelmholtzFilter)):
            raise InvalidType(f"MemberSize takes a DensityFilter or a HelmholtzFilter; got {type(filter).__name__}")
        if not isinstance(projection, _engine.Projection):
            raise InvalidType(f"MemberSize takes a Projection; got {type(projection).__name__}")
        threshold = float(threshold)
        if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            raise InvalidValue(f"the threshold of MemberSize must lie between 0 and 1; got {threshold!r}")
        if keep_solid is None or isinstance(keep_solid, KeepSolid):
            held = None if keep_solid is None else keep_solid.mask
        else:
            held = np.asarray(keep_solid)
            if held.ndim != 1 or held.dtype != bool:
                raise InvalidType(
                    "keep_solid takes a KeepSolid or a boolean array with one entry per element; got "
                    f"{type(keep_solid).__name__} of {held.ndim} dimensions and type {held.dtype}"
                )
            held = held.copy()
        self.filter = filter
        self.radius = float(filter.radius)
        self.element_size = float(filter.element_size)
        self.projection = projection
        self.threshold = threshold
        self.keep_solid = held
        self.c = (self.radius / self.element_size) ** 4 if c is None else float(c)
        self.experimental = isinstance(filter, HelmholtzFilter)
        self._thresholds = None
        self._active = None

    @property
    def thresholds(self) -> np.ndarray | None:
        """eta_e, one per element (read-only), as built on the first evaluation; ``None`` before it."""
        return self._thresholds

    def _thresholds_on(self, grid) -> np.ndarray:
        active = np.asarray(grid.active, dtype=float)
        if self._thresholds is None or not np.array_equal(active, self._active):
            if self.keep_solid is not None and self.keep_solid.size != active.size:
                raise InvalidValue(
                    f"keep_solid of MemberSize needs {active.size} entries, one per element "
                    f"(got {self.keep_solid.size})"
                )
            reach = np.asarray(self.filter.forward(active), dtype=float)
            thresholds = np.minimum(self.threshold, reach)
            if self.keep_solid is not None:
                thresholds[self.keep_solid] = 0.0
            thresholds.flags.writeable = False
            self._thresholds, self._active = thresholds, active
        return self._thresholds

    def evaluate(self, grid, filtered: np.ndarray) -> tuple[float, np.ndarray]:
        """g and dg/d(filtered) on ``grid`` (a ``Grid`` or a ``Grid3d``), computed in the engine."""
        return _engine.member_size(grid, filtered, self.projection, self._thresholds_on(grid), self.c)

    def value(self, ev) -> float:
        return float(self.evaluate(ev.grid, ev.density)[0])

    def partials(self, ev) -> Partials:
        return Partials(d_density=self.evaluate(ev.grid, ev.density)[1], d_state={})

    def self_test(self) -> list[Row]:
        """A uniform field (no slope anywhere, so I = P(v)) against g = P(v) (v - threshold)^2, a field that
        reaches the threshold everywhere against g = 0, and the two thresholds at size = R against their
        closed forms (verification rows 25 and 26 have the full checks with their limits)."""
        from michell.analysis import Grid
        from michell.analysis3d import Grid3d

        rows = []
        grid = Grid3d(5, 4, 3, 1.0)
        projection = Projection(8.0, 0.5)
        constraint = MemberSize(DensityFilter(grid, 3.0), projection, 0.8)
        value = 0.6
        expected = float(projection.forward(np.array([value]))[0]) * (value - 0.8) ** 2
        g, _ = constraint.evaluate(grid, np.full(grid.n_elements, value))
        rows.append(compared("a uniform field: g = P(v) (v - threshold)^2", float(g), expected, 1e-14))
        g, _ = constraint.evaluate(grid, np.full(grid.n_elements, 0.85))
        rows.append(compared("a field above the threshold everywhere: g = 0", float(g), 0.0, 0.0))
        cone_3d = member_threshold(DensityFilter(grid, 2.0), 2.0)
        rows.append(compared("threshold of the 3D cone filter at size = R: 13/16", cone_3d, 0.8125, 1e-15))
        cone_2d = member_threshold(DensityFilter(Grid(4, 4, 1.0, 1.0), 2.0), 2.0)
        expected_2d = 2.0 / math.pi * (math.sqrt(3.0) / 2.0 + math.pi / 6.0 - math.log(2.0 + math.sqrt(3.0)) / 8.0)
        rows.append(compared("threshold of the 2D cone filter at size = R", cone_2d, expected_2d, 1e-15))
        return rows
