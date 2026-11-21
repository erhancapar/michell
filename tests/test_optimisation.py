"""Tests for the module interface, the optimisation loop and the boundary rules of the new engine calls."""

import numpy as np
import pytest

import michell
from michell import _engine
from michell.adjoint import multiple_of_loads
from michell.evaluation import apply_filters, evaluate
from michell.gradient_check import small_problem
from michell.row import compared


class Displacement:
    """A test response: one displacement of one load case. Its d_state is a unit vector, not a multiple of
    the loads, so its gradient needs the real adjoint solve."""

    unit = "mm"
    reads = ("elasticity",)

    def __init__(self, dof, case):
        self.dof, self.case = dof, case
        self.name = f"displacement {dof} in {case}"

    def value(self, ev):
        return float(ev.state("elasticity", self.case)[self.dof])

    def partials(self, ev):
        d_state = np.zeros(ev.grid.n_dofs)
        d_state[self.dof] = 1.0
        return michell.Partials(d_density=None, d_state={("elasticity", self.case): d_state})

    def self_test(self):
        return []


def test_a_response_with_a_real_adjoint_solve_passes_the_gradient_check():
    problem, _, _ = small_problem()
    dof = 2 * problem.grid.node(12, 6)  # x displacement of the top-right corner, under the tip load
    tip_loads = problem.case("tip").loads
    assert multiple_of_loads(np.eye(problem.grid.n_dofs)[dof], tip_loads) is None
    check = michell.check_on_small_problem(Displacement(dof, "tip"))
    assert check.largest_difference <= 1e-5


def test_compliance_uses_the_state_as_its_adjoint():
    loads = np.array([0.0, 2.0, -1.0])
    assert multiple_of_loads(3.0 * loads, loads) == pytest.approx(3.0)
    assert multiple_of_loads(np.array([0.0, 2.0, 1.0]), loads) is None


@pytest.mark.parametrize(
    "module",
    [
        michell.Compliance(),
        michell.Mass(),
        michell.DensityFilter(michell.Grid(2, 2, 1.0, 1.0), 1.0),
        michell.KeepSolid(np.array([True, False, False, False])),
    ],
    ids=["compliance", "mass", "density filter", "keep solid"],
)
def test_every_module_passes_its_own_rows(module):
    rows = module.self_test()
    assert rows
    for row in rows:
        print(f"{module.name}: {row.name}: measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed


def test_a_row_computes_pass_or_fail():
    assert compared("x", 1.0 + 1e-9, 1.0, 1e-8).passed is True
    assert compared("x", 1.1, 1.0, 1e-8).passed is False
    assert compared("zero", 1e-13, 0.0, 1e-12).passed is True
    assert compared("not a number", float("nan"), 1.0, 1e-8).passed is False


def test_the_evaluation_is_read_only():
    problem, x, filters = small_problem()
    ev, _ = evaluate(problem, x, filters, ("elasticity",))
    with pytest.raises(ValueError):
        ev.density[0] = 0.0
    with pytest.raises(ValueError):
        ev.state("elasticity", "tip")[0] = 0.0
    with pytest.raises(ValueError):
        ev.loads("elasticity", "tip")[0] = 0.0
    with pytest.raises(KeyError, match="reads"):
        ev.state("elasticity", "no such case")


def test_boolean_masks_give_the_same_model_as_float_masks():
    active = np.ones(12, dtype=bool)
    active[5] = False
    grid_bool = michell.Grid(4, 3, 1.0, 1.0, active=active)
    grid_float = michell.Grid(4, 3, 1.0, 1.0, active=active.astype(float))
    assert np.array_equal(grid_bool.active, grid_float.active)
    held = np.zeros(grid_bool.n_dofs, dtype=bool)
    for j in range(4):
        held[2 * grid_bool.node(0, j)] = held[2 * grid_bool.node(0, j) + 1] = True
    loads = np.zeros(grid_bool.n_dofs)
    loads[2 * grid_bool.node(4, 1) + 1] = -1.0
    material = dict(youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0)
    u_bool = michell.Elasticity(grid_bool, held, **material).solve(np.ones(12), loads)
    u_float = michell.Elasticity(grid_float, held.astype(float), **material).solve(np.ones(12), loads)
    assert np.array_equal(u_bool, u_float)
    # Other types are still refused by the engine, not converted.
    with pytest.raises(michell.EngineError, match="float64"):
        michell.Grid(4, 3, 1.0, 1.0, active=np.ones(12, dtype=np.int64))


def test_long_calls_accept_read_only_inputs_and_leave_them_alone():
    problem, x, filters = small_problem()
    rho = filters[0].forward(x)
    rho.flags.writeable = False
    loads = problem.case("tip").loads  # read-only
    u = problem.elasticity.solve(rho, loads)
    lam = problem.elasticity.adjoint(rho, loads)
    assert np.array_equal(u, lam)
    assert np.array_equal(rho, filters[0].forward(x))


def test_the_density_filter_backward_checks_the_length_of_x():
    grid = michell.Grid(3, 2, 1.0, 1.0)
    density_filter = michell.DensityFilter(grid, 1.5)
    with pytest.raises(michell.EngineError) as caught:
        density_filter.backward(np.ones(5), np.ones(6))
    assert caught.value.code == "invalid_input"


def oc_inputs():
    problem, x, filters = small_problem()
    return problem, x, filters, [michell.Limit(michell.Mass(), at_most=0.5)]


def test_oc_refuses_what_it_cannot_handle():
    problem, x, filters, limits = oc_inputs()
    oc = michell.OptimalityCriteria()
    with pytest.raises(ValueError, match="exactly one limit"):
        oc.start(problem, x, limits * 2, filters)
    with pytest.raises(ValueError, match="reads no physics"):
        oc.start(problem, x, [michell.Limit(michell.Compliance(), 1.0)], filters)
    with pytest.raises(ValueError, match="only the density filter"):
        oc.start(problem, x, limits, filters * 2)

    class SquaredMass(michell.Mass):
        name = "squared mass"

        def value(self, ev):
            return super().value(ev) ** 2

    with pytest.raises(ValueError, match="linear"):
        oc.start(problem, x, [michell.Limit(SquaredMass(), 1.0)], filters)


def test_the_loop_meets_the_limit_keeps_inactive_elements_empty_and_stops_on_small_change():
    problem, x, filters, _ = oc_inputs()
    # The limit is the starting mass, so the material is only moved around and the compliance must fall.
    start_mass = michell.Mass().value(evaluate(problem, x, filters, ())[0])
    limits = [michell.Limit(michell.Mass(), at_most=start_mass)]
    result = michell.optimise(
        problem, michell.Compliance(), limits, filters, michell.OptimalityCriteria(), x, passes=200,
        stop_change=0.01,
    )
    assert len(result.passes) < 200
    assert result.passes[-1].change <= 0.01
    assert all(p.change > 0.01 for p in result.passes[:-1])
    goals = [p.goal for p in result.passes]
    assert goals[-1] < goals[0]
    inactive = problem.grid.active == 0.0
    assert np.all(result.x[inactive] == 0.0) and np.all(result.density[inactive] == 0.0)
    mass = michell.Mass().value(evaluate(problem, result.x, filters, ())[0])
    assert mass == pytest.approx(start_mass, rel=1e-2)


def test_the_engine_oc_update_takes_the_filter_of_its_own_grid_only():
    grid = michell.Grid(4, 2, 1.0, 1.0)
    active = np.ones(8, dtype=bool)
    active[3] = False
    other = michell.DensityFilter(michell.Grid(4, 2, 1.0, 1.0, active=active), 1.5)
    oc = _engine.Oc(grid, 0.2)
    ones = np.ones(8)
    with pytest.raises(michell.EngineError) as caught:
        oc.update(np.full(8, 0.5), -ones, ones, ones, 4.0, other)
    assert caught.value.code == "invalid_input"


class FailingOnPass:
    """MMA, except that on one pass it hands the engine a goal that is not a number, so the engine raises
    there."""

    name = "MMA failing on one pass"

    def __init__(self, failing_pass):
        self.mma = michell.MovingAsymptotes()
        self.failing_pass = failing_pass
        self.updates = []

    def start(self, problem, x, limits, filters):
        self.mma.start(problem, x, limits, filters)

    def update(self, x, goal, d_goal, limits, d_limits):
        if len(self.updates) + 1 == self.failing_pass:
            goal = float("nan")
        x_new = self.mma.update(x, goal, d_goal, limits, d_limits)
        self.updates.append(x_new)
        return x_new


def test_an_engine_error_in_the_loop_carries_the_passes_completed_before_it():
    problem, x, filters, limits = oc_inputs()
    run = (problem, michell.Compliance(), limits, filters)
    optimiser = FailingOnPass(3)
    with pytest.raises(michell.EngineError) as caught:
        michell.optimise(*run, optimiser, x, passes=10)
    assert caught.value.code == "invalid_input"
    partial = caught.value.partial_result
    assert [p.number for p in partial.passes] == [1, 2]
    np.testing.assert_array_equal(partial.x, optimiser.updates[-1])
    np.testing.assert_array_equal(partial.density, apply_filters(partial.x, filters)[-1])
    # The same as a run that stops after those two passes.
    two_passes = michell.optimise(*run, michell.MovingAsymptotes(), x, passes=2)
    assert partial.passes == two_passes.passes
    np.testing.assert_array_equal(partial.x, two_passes.x)
    np.testing.assert_array_equal(partial.density, two_passes.density)

    with pytest.raises(michell.EngineError) as caught:
        michell.optimise(*run, FailingOnPass(1), x, passes=10)
    partial = caught.value.partial_result
    assert partial.passes == ()
    np.testing.assert_array_equal(partial.x, np.where(problem.grid.active == 1.0, x, 0.0))
