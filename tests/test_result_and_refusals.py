"""The values a run returns for its own design, why it stopped and whether its limits hold; the error contract of
the checks written in Python; and the inputs refused before a run starts."""

import warnings

import numpy as np
import pytest

import michell
from michell.errors import InvalidValue
from michell.evaluation import apply_filters, evaluate
from michell.gradient_check import small_problem, small_problem_3d


def solid_mass(problem):
    return michell.Mass().value(evaluate(problem, np.ones(problem.grid.n_elements), [], ())[0])


def independent_values(problem, x, filters, goal, limits, designs=None):
    """The goal and limit values of x, from fresh solves (no warm start) of each design they read."""
    density = apply_filters(x, filters)[-1]

    def value(response):
        chain = []
        if isinstance(response, michell.On):
            chain, response = designs[response.design], response.response
        return response.value(evaluate(problem, density, chain, response.reads)[0])

    return value(goal), tuple(value(limit.response) for limit in limits)


def test_oc_returns_the_values_of_the_design_it_returns_and_stops_on_change():
    problem, x, filters = small_problem()
    limits = [michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))]
    result = michell.optimise(
        problem, michell.Compliance(), limits, filters, michell.OptimalityCriteria(), x, passes=200, stop_change=0.01
    )
    assert result.stop_reason == "change" and len(result.passes) < 200
    goal, limit_values = independent_values(problem, result.x, filters, michell.Compliance(), limits)
    assert result.goal == goal and result.limits == limit_values
    assert result.goal != result.passes[-1].goal  # the record is one update behind
    assert result.limits_met == (True,) and result.feasible


def test_mma_returns_the_values_of_the_design_it_returns_and_stops_on_the_pass_count():
    problem, x, filters = small_problem()
    limits = [michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))]
    result = michell.optimise(problem, michell.Compliance(), limits, filters, michell.MovingAsymptotes(), x, passes=4)
    assert result.stop_reason == "passes" and len(result.passes) == 4
    goal, limit_values = independent_values(problem, result.x, filters, michell.Compliance(), limits)
    assert result.goal == goal and result.limits == limit_values
    assert isinstance(result.goal, float) and all(isinstance(met, bool) for met in result.limits_met)


def test_a_3d_run_returns_the_values_of_the_design_it_returns():
    problem, x, filters = small_problem_3d()
    limits = [michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))]
    result = michell.optimise(problem, michell.Compliance(), limits, filters, michell.MovingAsymptotes(), x, passes=2)
    goal, limit_values = independent_values(problem, result.x, filters, michell.Compliance(), limits)
    # The run's last solve starts from the states of its last pass, the independent one from zero.
    assert result.goal == pytest.approx(goal, rel=1e-9)
    assert result.limits == pytest.approx(limit_values, rel=1e-12)


def test_a_run_on_named_designs_returns_the_values_of_each_design_read():
    problem, x, filters = small_problem()
    designs = michell.robust_designs(4.0, 0.2)
    goal = michell.on("eroded", michell.Compliance())
    limits = [
        michell.Limit(michell.on("dilated", michell.Mass()), at_most=0.5 * solid_mass(problem)),
        michell.Limit(michell.on("nominal", michell.Compliance()), at_most=1e6),
    ]
    result = michell.optimise(
        problem, goal, limits, filters, michell.MovingAsymptotes(), x, passes=3, designs=designs
    )
    assert (result.goal, result.limits) == independent_values(problem, result.x, filters, goal, limits, designs)


def test_a_limit_that_is_not_met_is_reported_and_warned_about():
    problem, _, filters = small_problem()
    bound = 0.1 * solid_mass(problem)
    limits = [michell.Limit(michell.Mass(), at_most=bound)]
    start = np.ones(problem.grid.n_elements)
    # One MMA update cannot move the full block down to a tenth of its mass.
    with pytest.warns(michell.LimitNotMet, match=r"mass = [0-9.e+-]+ g, at most ") as caught:
        result = michell.optimise(problem, michell.Compliance(), limits, filters, michell.MovingAsymptotes(), start, 1)
    assert f"{bound:.6g} g" in str(caught[0].message)
    assert result.limits_met == (False,) and not result.feasible
    assert result.limits[0] > bound
    # A tolerance that covers the excess makes the same design meet the limit.
    excess = (result.limits[0] - bound) / bound
    with warnings.catch_warnings():
        warnings.simplefilter("error", michell.LimitNotMet)
        loose = michell.optimise(
            problem, michell.Compliance(), limits, filters, michell.MovingAsymptotes(), start, 1,
            limit_tolerance=2 * excess,
        )  # fmt: skip
    assert loose.feasible and loose.goal == result.goal


def test_a_run_that_meets_its_limits_does_not_warn_and_a_run_without_limits_is_feasible():
    problem, x, filters = small_problem()
    limits = [michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", michell.LimitNotMet)
        result = michell.optimise(
            problem, michell.Compliance(), limits, filters, michell.OptimalityCriteria(), x, passes=30
        )
        free = michell.optimise(problem, michell.Mass(), [], filters, michell.MovingAsymptotes(), x, passes=1)
    assert result.feasible
    assert free.limits == () and free.limits_met == () and free.feasible


class FailingUpdate:
    """MMA, except that on one pass it hands the engine a goal that is not a number, so the engine raises."""

    def __init__(self, failing_pass):
        self.mma, self.failing_pass, self.count = michell.MovingAsymptotes(), failing_pass, 0

    def start(self, problem, x, limits, filters):
        self.mma.start(problem, x, limits, filters)

    def update(self, x, goal, d_goal, limits, d_limits):
        self.count += 1
        return self.mma.update(x, float("nan") if self.count == self.failing_pass else goal, d_goal, limits, d_limits)


class FailingCompliance(michell.Compliance):
    """Compliance whose value raises an ``EngineError`` from its call number ``failing_call`` on."""

    def __init__(self, failing_call):
        self.failing_call, self.calls = failing_call, 0

    def value(self, ev):
        self.calls += 1
        if self.calls >= self.failing_call:
            raise InvalidValue("a failing test response")
        return super().value(ev)


def test_a_partial_result_says_error_and_holds_the_values_of_its_design_when_they_were_measured():
    problem, x, filters = small_problem()
    limits = [michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))]
    with pytest.raises(michell.EngineError) as caught:
        michell.optimise(problem, michell.Compliance(), limits, filters, FailingUpdate(3), x, passes=10)
    partial = caught.value.partial_result
    assert partial.stop_reason == "error" and len(partial.passes) == 2
    # The failing pass measured its x before the update failed: those are the values of the partial design.
    goal, limit_values = independent_values(problem, partial.x, filters, michell.Compliance(), limits)
    assert partial.goal == goal and partial.limits == limit_values and partial.limits_met is not None

    # Two passes, then the evaluation of the returned design raises: the error leaves with the two passes.
    with pytest.raises(michell.EngineError) as caught:
        michell.optimise(problem, FailingCompliance(3), limits, filters, michell.MovingAsymptotes(), x, passes=2)
    partial = caught.value.partial_result
    assert partial.stop_reason == "error" and len(partial.passes) == 2
    assert partial.goal is None and partial.limits is None and partial.limits_met is None
    assert not partial.feasible


def test_the_checks_written_in_python_raise_engine_errors_with_a_published_code():
    problem, _, _ = small_problem()
    cases = [
        (ValueError, lambda: michell.Limit(michell.Mass(), at_most=1.0, scale=-1.0)),
        (ValueError, lambda: michell.robust_designs(4.0, 0.7)),
        (TypeError, lambda: michell.on(3, michell.Mass())),
        (KeyError, lambda: problem.case("missing")),
    ]
    for kind, call in cases:
        with pytest.raises(michell.EngineError) as caught:
            call()
        assert isinstance(caught.value, kind)
        assert caught.value.code == "invalid_input"
        assert str(caught.value) == f"invalid_input: {caught.value.message}"
        assert isinstance(caught.value.details, dict)


def test_a_load_on_a_held_degree_of_freedom_is_refused():
    grid = michell.Grid(6, 2, 1.0, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    left = [2 * grid.node(0, j) + d for j in range(3) for d in (0, 1)]
    held[left] = True
    loads = np.zeros(grid.n_dofs)
    loads[left] = 1.0
    steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    free = np.zeros(grid.n_dofs)
    free[2 * grid.node(6, 1) + 1] = -1.0
    with pytest.raises(michell.EngineError) as caught:
        michell.Problem(grid, held.astype(float), steel, [michell.LoadCase("free", free), michell.LoadCase("wall", loads)])
    error = caught.value
    assert error.code == "invalid_input" and isinstance(error, ValueError)
    assert "'wall'" in error.message and "6 held degrees of freedom" in error.message
    assert "first: 0, 1, " in error.message
    assert error.details == {"held_loaded_dofs": 6, "first_dof": 0}
    michell.Problem(grid, held, steel, [michell.LoadCase("free", free)])  # a load off the supports is taken


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -float("inf")])
def test_a_bound_that_is_not_a_finite_number_is_refused(bound):
    with pytest.raises(michell.EngineError, match=r"finite number; got at_most=(nan|inf|-inf)") as caught:
        michell.Limit(michell.Mass(), at_most=bound)
    assert caught.value.code == "invalid_input"


def test_a_bound_or_a_scale_of_several_numbers_is_refused():
    for settings, words in (({"at_most": np.array([1.0, 2.0])}, "finite number"),
                            ({"at_most": 1.0, "scale": np.array([1.0, 2.0])}, "positive number")):  # fmt: skip
        with pytest.raises(michell.EngineError, match=words) as caught:
            michell.Limit(michell.Mass(), **settings)
        assert caught.value.code == "invalid_input" and isinstance(caught.value, ValueError)


def run_refused(response, bound):
    problem, x, filters = small_problem()
    designs = {"nominal": [michell.Projection(4.0)]} if isinstance(response, michell.On) else None
    goal = michell.on("nominal", michell.Mass()) if designs else michell.Mass()
    optimiser = michell.MovingAsymptotes()
    with pytest.raises(michell.EngineError) as caught:
        michell.optimise(problem, goal, [michell.Limit(response, bound)], filters, optimiser, x, 5, designs=designs)
    error = caught.value
    assert error.code == "invalid_input" and error.partial_result is None
    assert "limit 0" in error.message and "is refused" in error.message
    assert error.details == {"limit": 0, "at_most": bound}
    return error


@pytest.mark.parametrize(
    ("response", "bound", "words"),
    [
        (michell.Mass(), -2.5, "below 0 g can never be met (got -2.5)"),
        (michell.on("nominal", michell.Compliance()), -1.0, "below 0 N mm can never be met (got -1)"),
    ],
)
def test_a_bound_below_0_that_no_design_can_meet_is_refused_before_the_first_pass(response, bound, words):
    assert words in run_refused(response, bound).message


@pytest.mark.parametrize(
    ("response", "words"),
    [
        (michell.Mass(), "met only by the empty part, so the bound must be above 0 g (got 0)"),
        (michell.Compliance(), "met only by a part without load, so the bound must be above 0 N mm (got 0)"),
    ],
)
def test_a_bound_of_0_is_refused_before_the_first_pass_without_saying_it_can_never_be_met(response, words):
    message = run_refused(response, 0.0).message
    assert words in message and "never be met" not in message


def test_a_refusal_by_the_optimisers_start_comes_before_any_solve(monkeypatch):
    solves = []
    solve = michell.Problem.solve

    def counted(self, *arguments, **settings):
        solves.append(1)
        return solve(self, *arguments, **settings)

    monkeypatch.setattr(michell.Problem, "solve", counted)
    problem, x, filters = small_problem()
    limit = michell.Limit(michell.Mass(), at_most=0.4 * solid_mass(problem))
    with pytest.raises(michell.EngineError, match="exactly one limit") as caught:
        michell.optimise(problem, michell.Compliance(), [limit, limit], filters, michell.OptimalityCriteria(), x, 5)
    assert caught.value.code == "invalid_input" and caught.value.partial_result is None
    assert solves == []
    problem_3d, x_3d, filters_3d = small_problem_3d()
    with pytest.raises(michell.EngineError, match="2D grid only") as caught:
        michell.optimise(problem_3d, michell.Compliance(), [], filters_3d, michell.OptimalityCriteria(), x_3d, 5)
    assert caught.value.partial_result is None and solves == []
    # The counter sees the solves of a run that is not refused.
    michell.optimise(problem, michell.Compliance(), [], filters, michell.MovingAsymptotes(), x, 1)
    assert len(solves) > 0


def test_a_mass_bound_below_what_keep_solid_holds_is_refused():
    problem, x, filters = small_problem()
    grid = problem.grid
    pad = np.zeros(grid.n_elements, dtype=bool)
    pad[:4] = True
    kept = 4 * problem.material.mass_g(grid.voxel_volume_mm3)
    designs = {"nominal": [michell.Projection(4.0), michell.KeepSolid(pad, grid)]}
    goal = michell.on("nominal", michell.Compliance())
    for bound, refused in ((0.9 * kept, True), (1.1 * kept, False)):
        limits = [michell.Limit(michell.on("nominal", michell.Mass()), at_most=bound)]
        run = (problem, goal, limits, filters, michell.MovingAsymptotes(), x, 1)
        if refused:
            with pytest.raises(michell.EngineError, match=rf"KeepSolid holds 4 elements solid, {kept:.6g} g"):
                michell.optimise(*run, designs=designs)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", michell.LimitNotMet)
                michell.optimise(*run, designs=designs)


@pytest.mark.parametrize("tolerance", [-1e-3, float("nan"), np.array([1e-3, 1e-2])])
def test_a_limit_tolerance_below_0_or_not_a_number_is_refused(tolerance):
    problem, x, filters = small_problem()
    with pytest.raises(michell.EngineError, match="limit_tolerance") as caught:
        michell.optimise(problem, michell.Mass(), [], filters, michell.MovingAsymptotes(), x, 1, limit_tolerance=tolerance)
    assert caught.value.code == "invalid_input"


def test_density_is_given_in_grams_per_cubic_centimetre():
    steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    assert steel.mass_g(1000.0) == pytest.approx(7.85, rel=1e-15)  # 1 cm^3 of steel
    assert not hasattr(steel, "density_g_mm3")


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), "heavy"])
def test_a_load_case_weight_below_0_or_not_finite_is_refused(weight):
    with pytest.raises(michell.EngineError) as refused:
        michell.LoadCase("case", np.zeros(4), weight)
    assert refused.value.code == "invalid_input"
    assert michell.LoadCase("case", np.zeros(4), 0.0).weight == 0.0
