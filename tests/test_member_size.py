"""The density filter on a 3D grid, the member-size limit and its threshold, a limit's scale, and the
member-size limit in the optimisation loop (on a design that is never solved)."""

import math

import numpy as np
import pytest

import michell
from michell.gradient_check import check_gradient, small_problem
from michell.mma import limit_scale
from michell.verification.member_size_rows import held_wall_part


def test_the_density_filter_takes_a_3d_grid_and_threads():
    grid = michell.Grid3d(6, 5, 4, 1.5)
    density_filter = michell.DensityFilter(grid, 3.0, threads=3)
    assert (density_filter.is_3d, density_filter.threads, density_filter.radius) == (True, 3, 3.0)
    assert density_filter.element_size == 1.5
    rows = density_filter.self_test()
    assert rows and all(row.passed for row in rows), [(row.name, row.note) for row in rows if not row.passed]
    x = np.random.default_rng(1).uniform(0.0, 1.0, grid.n_elements)
    single = michell.DensityFilter(grid, 3.0)
    assert np.array_equal(density_filter.forward(x), single.forward(x))
    assert np.array_equal(density_filter.backward(x, x), single.backward(x, x))
    plane = michell.DensityFilter(michell.Grid(4, 3, 2.0, 1.0), 3.0)
    assert (plane.is_3d, plane.threads, plane.element_size) == (False, 1, 2.0)


def test_the_density_filter_on_a_2d_grid_refuses_more_than_one_thread_and_oc_refuses_a_3d_filter():
    with pytest.raises(michell.EngineError, match="one thread") as error:
        michell.DensityFilter(michell.Grid(4, 3, 1.0, 1.0), 2.0, threads=2)
    assert error.value.code == "invalid_input"
    with pytest.raises(TypeError, match="Grid"):
        michell.DensityFilter("not a grid", 2.0)
    grid = michell.Grid(4, 3, 1.0, 1.0)
    oc = michell.Oc(grid, 0.2)
    n = grid.n_elements
    with pytest.raises(michell.EngineError, match="2D grid"):
        oc.update(np.full(n, 0.5), -np.ones(n), np.ones(n), np.ones(n), 6.0, michell.DensityFilter(michell.Grid3d(4, 3, 1, 1.0), 2.0))


def cone_share_by_quadrature(t: float, dimension: int, n: int = 2000) -> float:
    """The share of the cone's weight (w = 1 - s inside the unit radius) within |z| <= t, by the midpoint rule
    over the whole disc (2D) or ball (3D, in cylinder coordinates), without the layer formula."""
    side = (np.arange(n) + 0.5) / n
    if dimension == 3:  # rho from 0 to 1, z from 0 to 1 (both halves alike), weight 2 pi rho
        rho, z = np.meshgrid(side, side)
        w = np.maximum(0.0, 1.0 - np.hypot(rho, z)) * 2 * np.pi * rho
    else:  # x and y from 0 to 1 (the four quarters alike)
        x, z = np.meshgrid(side, side)
        w = np.maximum(0.0, 1.0 - np.hypot(x, z))
    inside = w * (z <= t)
    return float(inside.sum() / w.sum())


@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_the_cone_thresholds_equal_a_numerical_integration(t):
    radius = 4.0
    filter_3d = michell.DensityFilter(michell.Grid3d(2, 2, 2, 1.0), radius)
    filter_2d = michell.DensityFilter(michell.Grid(2, 2, 1.0, 1.0), radius)
    size = 2 * t * radius
    assert michell.member_threshold(filter_3d, size) == pytest.approx(cone_share_by_quadrature(t, 3), abs=2e-6)
    assert michell.member_threshold(filter_2d, size) == pytest.approx(cone_share_by_quadrature(t, 2), abs=2e-6)
    assert michell.member_threshold(filter_3d, 2 * radius) == 1.0
    assert michell.member_threshold(filter_2d, 0.0) == 0.0


def test_the_helmholtz_threshold_equals_a_numerical_solution_of_the_1d_equation():
    # -r^2 psi'' + psi = 1 for |z| < size / 2, 0 outside, on a long interval, by finite differences (Thomas
    # algorithm); psi at the mid-plane against 1 - exp(-size / (2 r)).
    helmholtz = michell.HelmholtzFilter(michell.Grid3d(2, 2, 2, 1.0), 3.0)
    r, size = helmholtz.length, 1.5
    step, half_length = 0.002, 12.0
    z = np.arange(-half_length, half_length + step / 2, step)
    source = (np.abs(z) < size / 2).astype(float)
    off = -(r**2) / step**2
    diagonal = np.full(z.size, 1.0 + 2 * r**2 / step**2)
    c_prime, d_prime = np.zeros(z.size), np.zeros(z.size)
    c_prime[0], d_prime[0] = off / diagonal[0], source[0] / diagonal[0]
    for i in range(1, z.size):
        denominator = diagonal[i] - off * c_prime[i - 1]
        c_prime[i] = off / denominator
        d_prime[i] = (source[i] - off * d_prime[i - 1]) / denominator
    psi = np.zeros(z.size)
    psi[-1] = d_prime[-1]
    for i in range(z.size - 2, -1, -1):
        psi[i] = d_prime[i] - c_prime[i] * psi[i + 1]
    assert michell.member_threshold(helmholtz, size) == pytest.approx(psi[z.size // 2], abs=1e-4)


def test_the_discrete_cone_filter_approaches_its_threshold_from_below():
    # A plate 12 elements thick, filter radius 12 elements: the elements next to the mid-plane, where the
    # cone is not cut off by the end of the grid, fall short of the continuous value by less than 0.5 %.
    radius, thickness = 12, 12
    side, depth = 2 * radius + 1, thickness + 2 * radius
    grid = michell.Grid3d(side, side, depth, 1.0)
    x = np.zeros((depth, side, side))
    x[radius : radius + thickness] = 1.0
    density_filter = michell.DensityFilter(grid, float(radius), threads=4)
    centre = density_filter.forward(x.ravel()).reshape(depth, side, side)[radius + thickness // 2, radius, radius]
    expected = michell.member_threshold(density_filter, float(thickness))
    assert expected * 0.995 < centre < expected


def test_member_size_takes_c_from_the_filter_and_marks_the_helmholtz_filter_experimental():
    grid = michell.Grid3d(4, 4, 4, 0.5)
    projection = michell.Projection(32.0)
    cone = michell.MemberSize(michell.DensityFilter(grid, 1.5), projection, 0.8)
    assert cone.c == pytest.approx(3.0**4) and not cone.experimental
    assert cone.reads == () and cone.name == "member size"
    pde = michell.MemberSize(michell.HelmholtzFilter(grid, 1.5), projection, 0.8, c=10.0)
    assert pde.c == 10.0 and pde.experimental
    rows = cone.self_test()
    assert rows and all(row.passed for row in rows), [(row.name, row.note) for row in rows if not row.passed]
    with pytest.raises(TypeError):
        michell.MemberSize(projection, projection, 0.8)
    with pytest.raises(TypeError):
        michell.MemberSize(michell.DensityFilter(grid, 1.5), michell.DensityFilter(grid, 1.5), 0.8)
    with pytest.raises(michell.EngineError):
        cone.evaluate(grid, np.full(grid.n_elements - 1, 0.5))


def test_member_size_depends_on_the_grid_in_elements_not_in_mm():
    # The gradient is taken per element side and c = (R / h)^4, so the same field on elements of 1 mm and of
    # 0.5 mm, with the radius scaled alike, gives the same value.
    values = []
    field = michell.DensityFilter(michell.Grid(20, 12, 1.0, 1.0), 3.0).forward(
        np.random.default_rng(4).uniform(0.0, 1.0, 240)
    )
    for h in (1.0, 0.5):
        grid = michell.Grid(20, 12, h, 1.0)
        constraint = michell.MemberSize(michell.DensityFilter(grid, 3.0 * h), michell.Projection(8.0), 0.6)
        values.append(constraint.evaluate(grid, field)[0])
    assert values[0] > 0.0
    assert values[1] == values[0]


def test_a_limit_takes_a_scale():
    mass = michell.Mass()
    assert limit_scale(michell.Limit(mass, at_most=-4.0)) == 4.0
    assert limit_scale(michell.Limit(mass, at_most=0.0)) == 1.0
    assert limit_scale(michell.Limit(mass, at_most=2e-7, scale=3e-4)) == 3e-4
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="scale"):
            michell.Limit(mass, at_most=1.0, scale=scale)


def member_size_run(problem, x, density_filter, with_member_size, passes=3):
    projection = michell.Projection(4.0)
    limits = [michell.Limit(michell.on("nominal", michell.Mass()), at_most=0.4 * 7.85e-3 * 1.5 * 1.5 * 2.0 * 72)]
    if with_member_size:
        member_size = michell.MemberSize(density_filter, projection, michell.member_threshold(density_filter, 3.0))
        limits.append(michell.Limit(michell.on("filtered", member_size), at_most=1e-6, scale=1e-3))
    return michell.optimise(
        problem,
        michell.on("nominal", michell.Compliance()),
        limits,
        [density_filter],
        michell.MovingAsymptotes(),
        x,
        passes=passes,
        designs={"filtered": [], "nominal": [projection]},
    )


def test_a_member_size_limit_on_a_design_nothing_else_reads_causes_no_solve():
    problem, x, filters = small_problem()
    calls = []
    solve, adjoint = problem.solve, problem.adjoint

    def counted_solve(*args, **kwargs):
        calls.append("solve")
        return solve(*args, **kwargs)

    def counted_adjoint(*args, **kwargs):
        calls.append("adjoint")
        return adjoint(*args, **kwargs)

    problem.solve, problem.adjoint = counted_solve, counted_adjoint
    # Three passes meet neither the mass nor the member-size limit, and optimise says so.
    with pytest.warns(michell.LimitNotMet):
        result = member_size_run(problem, x, filters[0], with_member_size=True)
    # One solve per load case per pass and one for the returned design, all on the nominal design;
    # compliance needs no adjoint solve.
    assert calls == ["solve"] * (len(problem.cases) * (len(result.passes) + 1))
    assert all(set(p.design_solver_steps) == {"nominal"} for p in result.passes)
    assert all(len(p.limits) == 2 and p.limits[1] > 0.0 for p in result.passes)
    assert np.array_equal(result.designs["filtered"], result.density)
    calls.clear()
    plain = member_size_run(problem, x, filters[0], with_member_size=False)
    assert calls == ["solve"] * (len(problem.cases) * (len(plain.passes) + 1))


def test_the_gradient_of_the_member_size_limit_through_the_loop_matches_finite_differences():
    problem, x, filters = small_problem(seed=2)
    density_filter = filters[0]
    projection = michell.Projection(4.0)
    member_size = michell.MemberSize(density_filter, projection, 0.7, c=4.0)
    check = check_gradient(
        problem,
        michell.on("filtered", member_size),
        x,
        filters,
        step=1e-6,
        designs={"filtered": [], "nominal": [projection]},
    )
    assert check.largest_difference < 1e-5
    assert math.isfinite(check.largest_element_difference)


def test_keep_solid_refuses_bad_masks_and_holds_its_elements_in_the_loop():
    grid = michell.Grid(4, 3, 1.0, 1.0, active=np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=bool))
    with pytest.raises(ValueError, match="inactive"):
        michell.KeepSolid(np.eye(1, 12, 5, dtype=bool).ravel(), grid)
    with pytest.raises(ValueError, match="entries"):
        michell.KeepSolid(np.zeros(11, dtype=bool), grid)
    with pytest.raises(ValueError, match="0/1"):
        michell.KeepSolid(np.full(12, 0.5))
    keep = michell.KeepSolid(np.eye(1, 12, 11).ravel(), grid)  # 0/1 mask
    with pytest.raises(ValueError, match="entries"):
        keep.forward(np.zeros(5))
    x = np.full(12, 0.3)
    assert keep.forward(x)[11] == 1.0 and keep.forward(x)[0] == 0.3
    assert keep.backward(x, np.ones(12))[11] == 0.0


# A part with empty faces, a hole (a block of inactive elements, empty to the filter) and a thin wall of
# elements held solid beside the hole: `held_wall_part` of the verification rows. Radius 3 element sides,
# threshold for a size of 3 (0.8125 in 3D), beta 8.


def held_wall_limit(dimension, keep):
    grid, wall = held_wall_part(dimension)
    density_filter = michell.DensityFilter(grid, 3.0, outside="empty", inactive="empty")
    threshold = michell.member_threshold(density_filter, 3.0)
    keep_solid = michell.KeepSolid(wall, grid) if keep else None
    constraint = michell.MemberSize(density_filter, michell.Projection(8.0), threshold, keep_solid=keep_solid)
    return grid, wall, density_filter, constraint


@pytest.mark.parametrize("dimension", [2, 3])
def test_the_fully_solid_part_meets_the_limit_exactly(dimension):
    for keep in (False, True):
        grid, wall, density_filter, constraint = held_wall_limit(dimension, keep)
        assert constraint.thresholds is None
        solid = density_filter.forward(grid.active)
        assert constraint.evaluate(grid, solid)[0] == 0.0
        reach = density_filter.forward(grid.active)
        expected = np.minimum(constraint.threshold, reach)
        if keep:
            expected[wall] = 0.0
        assert np.array_equal(constraint.thresholds, expected)
        assert not constraint.thresholds.flags.writeable
    # Not vacuous: the fully solid part falls short of the threshold somewhere, and with that one threshold
    # on every element it would not meet the limit.
    active = grid.active == 1.0
    assert (reach[active] < constraint.threshold).sum() > 0
    one_threshold = np.full(grid.n_elements, constraint.threshold)
    assert michell._engine.member_size(grid, reach, constraint.projection, one_threshold, constraint.c)[0] > 0.0


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_design_void_but_for_the_held_wall_is_judged_unless_the_wall_is_held(dimension):
    grid, wall, density_filter, constraint = held_wall_limit(dimension, keep=False)
    assert constraint.evaluate(grid, density_filter.forward(wall.astype(float)))[0] > 0.0


@pytest.mark.parametrize("dimension", [2, 3])
def test_holding_the_wall_leaves_only_what_its_neighbours_take_from_it(dimension):
    # The design elements next to a held wall take filtered density from it and are still judged, so a design
    # that is void but for the wall keeps a small g: less than with the wall judged, and negligible at the
    # sharp projection the limit is used with (beta 32).
    grid, wall, density_filter, held = held_wall_limit(dimension, keep=True)
    wall_only = density_filter.forward(wall.astype(float))
    g_held = held.evaluate(grid, wall_only)[0]
    grid_judged, wall_judged, filter_judged, judged = held_wall_limit(dimension, keep=False)
    g_judged = judged.evaluate(grid_judged, filter_judged.forward(wall_judged.astype(float)))[0]
    assert 0.0 < g_held < g_judged
    sharp = michell.MemberSize(
        density_filter, michell.Projection(32.0), held.threshold, keep_solid=michell.KeepSolid(wall, grid)
    )
    assert sharp.evaluate(grid, wall_only)[0] < 1e-9


def test_closed_faces_leave_the_threshold_unchanged_and_bad_arguments_are_refused():
    grid = michell.Grid3d(6, 5, 4, 1.0)
    constraint = michell.MemberSize(michell.DensityFilter(grid, 2.0), michell.Projection(8.0), 0.8)
    constraint.evaluate(grid, np.full(grid.n_elements, 0.5))
    assert np.array_equal(constraint.thresholds, np.full(grid.n_elements, 0.8)) and constraint.threshold == 0.8
    density_filter, projection = michell.DensityFilter(grid, 2.0), michell.Projection(8.0)
    for threshold in (1.2, -0.1, float("nan")):
        with pytest.raises(michell.errors.InvalidValue):
            michell.MemberSize(density_filter, projection, threshold)
    for keep_solid in ("wall", np.zeros(grid.n_elements), np.zeros((2, 60), dtype=bool)):
        with pytest.raises(michell.errors.InvalidType):
            michell.MemberSize(density_filter, projection, 0.8, keep_solid=keep_solid)
    short = np.zeros(grid.n_elements - 1, dtype=bool)
    short = michell.MemberSize(density_filter, projection, 0.8, keep_solid=short)
    with pytest.raises(michell.errors.InvalidValue, match="keep_solid"):
        short.evaluate(grid, np.full(grid.n_elements, 0.5))


def test_near_a_flat_empty_face_the_reach_falls_below_the_threshold_only_in_a_shallow_layer():
    # At the depth t R below a flat empty face of a solid half-space the 3D cone filter reaches
    # 1/2 + member_threshold(2 t R) / 2: the share of the ball's weight on the solid side. It equals the
    # threshold at size = R, 13/16, at t = 0.347, shallower than the ridge of a member R thick (t = 0.5). On
    # the grid (R = 12 element sides, the face z- empty, the other faces closed and the column in the middle,
    # where the ball is not cut off), the elements with their centre less deep than 0.347 R are those below
    # the threshold.
    radius = 12
    side, depth = 2 * radius + 1, 3 * radius
    grid = michell.Grid3d(side, side, depth, 1.0)
    density_filter = michell.DensityFilter(grid, float(radius), threads=4, outside={"z-": "empty"})
    threshold = michell.member_threshold(density_filter, float(radius))
    low, high = 0.0, 1.0
    for _ in range(60):
        t = (low + high) / 2
        reach = 0.5 + michell.member_threshold(density_filter, 2 * t * radius) / 2
        low, high = (t, high) if reach < threshold else (low, t)
    assert threshold == 0.8125 and low == pytest.approx(0.347, abs=5e-4)
    column = density_filter.forward(np.ones(grid.n_elements)).reshape(depth, side, side)[:, radius, radius]
    below = np.flatnonzero(column < threshold)
    assert np.array_equal(below, np.arange(int(np.ceil(low * radius - 0.5))))
