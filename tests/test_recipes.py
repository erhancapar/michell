"""The two-step procedure of `michell.recipes`: one call, consistent fields, repeatable on any thread count,
a callback per pass, refused arguments, the bound of figure 8; and verification row 28 (slow)."""

import warnings

import numpy as np
import pytest

import michell
from michell import verification
from michell.recipes import EPS_OF_FIGURE_8, member_size_bound, stiff_part
from michell.verification.recipe_rows import bridge

SHORT = ((1.0, 3), (8.0, 2), (32.0, 2))

# A few passes do not meet the member-size limit; the warning that says so is expected here.
pytestmark = pytest.mark.filterwarnings("ignore::michell.LimitNotMet")


def small_bridge(threads=1):
    """The bridge of row 28 on 16 x 8 x 6 elements, with half the block's mass (a third of it is held)."""
    part = bridge(16, 8, 6, threads=threads)
    return type(part)(part.problem, part.keep_solid, part.mass_g / 0.3 * 0.5)


def run_small(threads=1, **settings):
    part = small_bridge(threads)
    settings = {"schedule": SHORT, "passes_per_bound": 2, "second_step_passes": 3, **settings}
    outside = {"x-": "closed", "x+": "closed", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}
    # The member size of 2 elements keeps these runs small; stiff_part warns that it is under five.
    with pytest.warns(michell.GridTooCoarse):
        return part, stiff_part(part.problem, part.mass_g, 2.0, part.keep_solid, outside=outside, **settings)


def test_the_recipe_runs_and_its_fields_agree():
    part, outcome = run_small()
    keep = part.keep_solid
    projection = michell.Projection(32.0, 0.5)
    assert np.array_equal(outcome.nominal, keep.forward(projection.forward(outcome.filtered)))
    assert np.all(outcome.nominal[keep.mask] == 1.0)
    assert np.array_equal(outcome.x, outcome.result.x)
    assert np.array_equal(outcome.filtered, outcome.member_size.filter.forward(keep.forward(outcome.x)))
    assert outcome.result.stop_reason == "passes" and len(outcome.result.limits_met) == 2
    steps = [record.step for record in outcome.passes]
    assert steps == ["robust"] * 7 + ["member size"] * 3
    assert [record.beta for record in outcome.passes[:7]] == [1.0, 1.0, 1.0, 8.0, 8.0, 32.0, 32.0]
    assert [record.number for record in outcome.passes] == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
    assert (
        outcome.threshold == outcome.member_size.threshold == michell.member_threshold(outcome.member_size.filter, 2.0)
    )
    assert outcome.eps == member_size_bound(2.0) and outcome.scale == max(outcome.value_at_switch, outcome.eps)
    assert outcome.passes[-1].bounds == (part.mass_g, outcome.eps)
    # The records of the last call are those `optimise` kept.
    for record, kept in zip(outcome.passes[-3:], outcome.result.passes):
        assert (record.goal, record.limits, record.change) == (kept.goal, kept.limits, kept.change)
    assert outcome.first_step_nominal.shape == outcome.nominal.shape
    assert np.all(outcome.first_step_nominal[keep.mask] == 1.0)


def test_the_recipe_gives_the_same_numbers_on_1_and_4_threads():
    _, one = run_small(threads=1)
    _, four = run_small(threads=4)
    assert np.array_equal(one.x, four.x)
    assert np.array_equal(one.nominal, four.nominal)
    assert [r.goal for r in one.passes] == [r.goal for r in four.passes]
    assert [r.limits for r in one.passes] == [r.limits for r in four.passes]


def test_the_callback_is_called_once_per_pass_with_the_records():
    seen = []
    _, outcome = run_small(callback=lambda step, number, record: seen.append((step, number, record)))
    assert len(seen) == len(outcome.passes) == 10
    assert [record for _, _, record in seen] == list(outcome.passes)
    assert all((step, number) == (record.step, record.number) for step, number, record in seen)


def test_the_move_limits_reach_both_steps():
    _, outcome = run_small(move_limit=0.05, second_step_move_limit=0.02)
    robust = [record.change for record in outcome.passes if record.step == "robust"]
    member = [record.change for record in outcome.passes if record.step == "member size"]
    assert max(robust) <= 0.05 * (1.0 + 1e-12) and max(member) <= 0.02 * (1.0 + 1e-12)
    _, wide = run_small()
    assert max(record.change for record in wide.passes) > 0.05


def test_the_best_design_that_meets_its_limits_can_be_returned():
    # With eps 1e-3, five member-size passes and a move limit of 0.5 the last design is 0.6 % over its mass;
    # the designs before it meet both limits, and the fifth pass's design has the lowest compliance of those.
    # (At the default 0.1 every design of this step meets both limits and the last is the best.)
    settings = {"eps": 1e-3, "second_step_passes": 5, "second_step_move_limit": 0.5}
    with pytest.warns(michell.LimitNotMet):
        part, last = run_small(second_step_returns="last", **settings)
    assert not last.result.feasible and last.returned_pass == 6
    with warnings.catch_warnings():
        warnings.simplefilter("error", michell.LimitNotMet)  # the returned design meets its limits: no warning
        _, best = run_small(second_step_returns="best", **settings)
    assert best.passes == last.passes and best.result.passes == last.result.passes
    member = [record for record in best.passes if record.step == "member size"]
    feasible = [r for r in member if r.limits[0] <= part.mass_g * 1.001 and r.limits[1] <= best.eps * 1.001]
    assert best.returned_pass == min(feasible, key=lambda r: r.goal).number == 5
    # The goal again to the solver's accuracy (its solves start from other states), the limits exactly.
    assert best.result.feasible and best.result.goal == pytest.approx(member[4].goal, rel=1e-6)
    assert best.result.limits == pytest.approx(member[4].limits, rel=1e-9)
    # Every field describes the returned design: a fresh evaluation of its x gives the same.
    assert np.array_equal(best.x, best.result.x)
    keep = part.keep_solid
    filtered = best.member_size.filter.forward(keep.forward(best.x))
    assert np.array_equal(best.filtered, filtered) and np.array_equal(best.result.designs["filtered"], filtered)
    assert np.array_equal(best.nominal, keep.forward(michell.Projection(32.0, 0.5).forward(filtered)))
    loads = part.problem.cases[0].loads
    compliance = michell.dot(loads, part.problem.solve(best.nominal, loads)[0])
    assert best.result.goal == pytest.approx(compliance, rel=1e-6)
    assert best.result.stop_reason == "passes"


def test_the_last_design_is_returned_with_a_warning_when_none_meets_its_limits():
    with pytest.warns(michell.LimitNotMet):
        _, last = run_small(second_step_passes=3, second_step_returns="last")
    with pytest.warns(michell.LimitNotMet):
        _, best = run_small(second_step_passes=3, second_step_returns="best")
    assert best.returned_pass == last.returned_pass == 4
    assert np.array_equal(best.x, last.x) and best.result.goal == last.result.goal


def test_the_recipe_runs_on_a_2d_grid():
    grid = michell.Grid(16, 8, 1.0, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for j in range(9):
        held[2 * grid.node(0, j) : 2 * grid.node(0, j) + 2] = True
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(16, 0) + 1] = -1.0
    problem = michell.Problem(grid, held, michell.Material(1.0, 0.3, 1000.0), [michell.LoadCase("tip", loads)])
    with pytest.warns(michell.GridTooCoarse):  # a member size of 2 elements, as in run_small
        outcome = stiff_part(
            problem, 0.4 * 16 * 8, 2.0, outside={"x+": "empty", "y-": "empty", "y+": "empty"},
            schedule=((1.0, 2), (32.0, 2)), second_step_passes=2,
        )  # fmt: skip
    assert outcome.nominal.shape == (grid.n_elements,) and len(outcome.passes) == 6


def test_an_engine_error_names_its_step_and_pass():
    part = small_bridge()

    class Failing:
        name = "failing"

        def __init__(self):
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 7:  # the evaluation of the third pass, the first of the second call
                raise michell.EngineError("invalid_input: stop here")
            return x

        def backward(self, x, grad):
            return grad

    # The member size of 2 elements gives GridTooCoarse before the first pass, as in run_small.
    with pytest.warns(michell.GridTooCoarse), pytest.raises(michell.EngineError) as error:
        stiff_part(
            part.problem, part.mass_g, 2.0, part.keep_solid, [Failing()], schedule=SHORT, passes_per_bound=2,
            second_step_passes=3,
        )  # fmt: skip
    assert (error.value.step, error.value.pass_number) == ("robust", 3)
    partial = error.value.partial_result
    assert partial.stop_reason == "error" and partial.passes == ()


@pytest.mark.parametrize(
    "change, match",
    [
        ({"mass_g": 0.0}, "asked mass"),
        ({"mass_g": 1e9}, "asked mass"),
        ({"mass_g": 1e-6}, "held elements"),
        ({"size_mm": -1.0}, "member size"),
        ({"size_mm": "5"}, "number"),
        ({"keep_solid": np.zeros(3, dtype=bool)}, "KeepSolid"),
        ({"filters": "x"}, "list"),
        ({"schedule": ()}, "schedule"),
        ({"schedule": ((0.0, 3),)}, "beta"),
        ({"schedule": ((1.0, 0),)}, "passes"),
        ({"schedule": ((1.0, 2.5),)}, "whole number"),
        ({"spread": 0.5}, "spread"),
        ({"passes_per_bound": 0}, "passes_per_bound"),
        ({"second_step_passes": True}, "whole number"),
        ({"move_limit": 0.0}, "move limit"),
        ({"second_step_move_limit": 1.5}, "move limit"),
        ({"second_step_move_limit": "0.2"}, "move limit"),
        ({"second_step_returns": "first"}, "second_step_returns"),
        ({"eps": -1e-7}, "eps"),
        ({"callback": 3}, "callable"),
    ],
)
def test_bad_arguments_are_refused(change, match):
    part = small_bridge()
    arguments = {"problem": part.problem, "mass_g": part.mass_g, "size_mm": 2.0, "keep_solid": part.keep_solid}
    arguments.update({"schedule": SHORT, **change})
    with pytest.raises(michell.EngineError, match=match) as error:
        stiff_part(**arguments)
    assert error.value.code == "invalid_input"
    assert isinstance(error.value, (ValueError, TypeError))


def test_a_problem_is_required():
    with pytest.raises(TypeError, match="Problem"):
        stiff_part("not a problem", 1.0, 2.0)


def test_the_bound_of_figure_8():
    for r, eps in EPS_OF_FIGURE_8.items():
        assert member_size_bound(r) == eps
    assert member_size_bound(9.0) == member_size_bound(10.0) == 5e-8
    assert member_size_bound(9.5) == pytest.approx(5e-8, rel=1e-12)
    assert member_size_bound(8.5) == pytest.approx(np.sqrt(5.5e-8 * 5e-8), rel=1e-12)
    assert member_size_bound(2.0) == member_size_bound(1.0) == 1e-6
    assert member_size_bound(12.0) == member_size_bound(40.0) == 5e-8
    middle = member_size_bound(4.5)
    assert middle == pytest.approx(np.sqrt(3e-7 * 1e-7), rel=1e-12)
    with pytest.raises(michell.EngineError):
        member_size_bound(0.0)


@pytest.mark.slow
def test_row_28_the_recipe_on_a_3d_bridge():
    pytest.importorskip("scipy")
    rows = verification.row_28_recipe_bridge()
    assert len(rows) == 5
    # The mass is checked as its relative excess over the asked mass, so the table shows the limit itself.
    assert rows[0].name.endswith("nominal mass over the asked mass, minus 1") and rows[0].limit == 0.001
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"
