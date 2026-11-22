"""Tests for the MMA optimiser: the boundary rules of `michell.Mma`, a part with inactive elements, and a
problem with two limits."""

import numpy as np
import pytest

import michell
from michell.evaluation import evaluate
from michell.gradient_check import small_problem


def test_mma_refuses_wrong_input_and_keeps_its_state():
    mma = michell.Mma(np.zeros(3), np.ones(3), 1)
    x = np.full(3, 0.5)
    with pytest.raises(michell.EngineError, match="shape") as error:
        mma.update(x, 1.0, np.ones(3), np.zeros(1), np.ones((1, 2)))
    assert error.value.code == "invalid_input"
    two_limits = michell.Mma(np.zeros(3), np.ones(3), 2)
    with pytest.raises(michell.EngineError, match="C-contiguous"):
        two_limits.update(x, 1.0, np.ones(3), np.zeros(2), np.asfortranarray(np.ones((2, 3))))
    with pytest.raises(michell.EngineError, match="bounds"):
        mma.update(np.array([0.5, 1.5, 0.5]), 1.0, np.ones(3), np.zeros(1), np.ones((1, 3)))
    with pytest.raises(michell.EngineError):
        mma.update(x.astype(np.float32), 1.0, np.ones(3), np.zeros(1), np.ones((1, 3)))
    with pytest.raises(michell.EngineError):
        michell.Mma(np.zeros(2), np.zeros(2), 1)
    assert mma.iteration == 0
    mma.update(x, 1.0, np.ones(3), np.zeros(1), np.ones((1, 3)))
    assert mma.iteration == 1


def test_the_move_limit_bounds_one_update_and_is_refused_outside_0_to_1():
    # A steep goal from the middle of the range: the default lets a variable move by more than 0.1.
    x, d_goal = np.full(3, 0.5), np.array([100.0, -100.0, 50.0])
    default = michell.Mma(np.zeros(3), np.ones(3), 0)
    assert default.move_limit == 0.5
    assert np.abs(default.update(x, 1.0, d_goal, np.zeros(0), np.zeros((0, 3))) - x).max() > 0.3
    narrow = michell.Mma(np.zeros(3), np.ones(3), 0, move_limit=0.1)
    assert narrow.move_limit == 0.1
    moved = np.abs(narrow.update(x, 1.0, d_goal, np.zeros(0), np.zeros((0, 3))) - x).max()
    assert 0.09 < moved <= 0.1 * (1.0 + 1e-12)
    for refused in (0.0, -0.2, 1.01, float("nan"), float("inf")):
        with pytest.raises(michell.EngineError, match="move limit") as error:
            michell.Mma(np.zeros(3), np.ones(3), 0, move_limit=refused)
        assert error.value.code == "invalid_input"


# Eight variables of an update captured from stiff_part (the 16 x 8 x 6 bridge of row 28, the member-size
# step at move 0.1, its second update), those of the steepest goal gradient, about 4e6; exact values.
# fmt: off
CAPTURED_UPDATE = {
    "previous": ["0x1.1daa340de3b93p-1", "0x1.1daa32f9b89ccp-1", "0x1.1daa340cabd32p-1", "0x1.1daa32eb93017p-1",
                 "0x1.2a16334c22742p-1", "0x1.2a1631e7866d7p-1", "0x1.2a16334ace6a9p-1", "0x1.2a1631d8ce789p-1"],
    "x": ["0x1.d4ee01b566a95p-2", "0x1.d4edff8d10707p-2", "0x1.d4ee01b2f6dd3p-2", "0x1.d4edff70c539dp-2",
          "0x1.edc60031e681ap-2", "0x1.edc5fd68ae744p-2", "0x1.edc6002f3e6e8p-2", "0x1.edc5fd4b3e8a8p-2"],
    "goal": ["0x1.54055b83e8041p+19"],
    "d_goal": ["-0x1.d3a9248c94d4fp+21", "-0x1.d3a9ad74a55edp+21", "-0x1.d3a924a10f397p+21", "-0x1.d3a9ac00658eep+21",
               "-0x1.b8794fa62e6abp+22", "-0x1.b879d098531acp+22", "-0x1.b8794fbf0358cp+22", "-0x1.b879ced8b109dp+22"],
    "limits": ["-0x1.07c50b2a7a016p-2", "0x1.3f879bd056706p-2"],
    "d_mass": ["0x1.a3655bc571673p-7", "0x1.a3655f89cdba7p-7", "0x1.a3655bc347349p-7", "0x1.a3655f74356e4p-7",
               "0x1.5eafc8da20d91p-7", "0x1.5eafcbeaba3bdp-7", "0x1.5eafc8d9428a6p-7", "0x1.5eafcbe2d0838p-7"],
    "d_member_size": ["0x1.68f0f80983f6bp-4", "0x1.68f0fbf189366p-4", "0x1.68f0f80836cf4p-4", "0x1.68f0fbe4b84ffp-4",
                      "0x1.0f8cbbbea8007p-4", "0x1.0f8cbe51d4d6bp-4", "0x1.0f8cbbbdfccdap-4", "0x1.0f8cbe4b742c5p-4"],
}
# fmt: on


def test_an_update_whose_bounds_are_pressed_on_hard_gives_a_finite_design():
    """CAPTURED_UPDATE: the subproblem held some variables at a distance from their move bound below the
    spacing of the numbers there, and the design returned was not a number."""
    data = {name: np.array([float.fromhex(v) for v in values]) for name, values in CAPTURED_UPDATE.items()}
    mma = michell.Mma(np.zeros(8), np.ones(8), 2, move_limit=0.1)
    # The first update only sets the state the captured one followed (the asymptotes of the first two
    # updates depend on x alone).
    mma.update(data["previous"], 1.0, np.zeros(8), np.full(2, -1.0), np.zeros((2, 8)))
    d_limits = np.array([data["d_mass"], data["d_member_size"]])
    design = mma.update(data["x"], data["goal"][0], data["d_goal"], data["limits"], d_limits)
    assert np.all(np.isfinite(design))
    assert np.all(np.abs(design - data["x"]) <= 0.1 + 1e-12) and np.all((design >= 0.0) & (design <= 1.0))


@pytest.mark.filterwarnings("ignore::michell.LimitNotMet")  # one pass does not reach half the mass
def test_the_move_limit_of_moving_asymptotes_reaches_the_engine():
    problem, x, filters = small_problem()
    x = np.where(problem.grid.active == 1.0, x, 0.0)
    ev, _ = evaluate(problem, x, filters, ("elasticity",))
    limits = [michell.Limit(michell.Mass(), 0.5 * michell.Mass().value(ev))]

    def one_pass(optimiser):
        result = michell.optimise(
            problem, goal=michell.Compliance(), limits=limits, filters=filters, optimiser=optimiser, x=x, passes=1
        )
        return np.abs(result.x - x).max()

    assert one_pass(michell.MovingAsymptotes()) > 0.05
    assert one_pass(michell.MovingAsymptotes(move_limit=0.05)) <= 0.05 * (1.0 + 1e-12)
    assert michell.MovingAsymptotes().move_limit == 0.5
    with pytest.raises(michell.errors.InvalidValue, match="move limit"):
        michell.MovingAsymptotes(move_limit=1.5)
    with pytest.raises(michell.errors.InvalidType, match="move limit"):
        michell.MovingAsymptotes(move_limit="0.2")


class CaseCompliance:
    """The compliance of one load case: a limit per case gives a problem with two limits."""

    unit = "N mm"
    reads = ("elasticity",)

    def __init__(self, case):
        self.case = case
        self.name = f"compliance of {case}"

    def value(self, ev):
        return float(ev.loads("elasticity", self.case) @ ev.state("elasticity", self.case))

    def partials(self, ev):
        return michell.Partials(d_density=None, d_state={("elasticity", self.case): ev.loads("elasticity", self.case)})

    def self_test(self):
        return []


def test_least_mass_under_two_stiffness_limits_on_a_part_with_a_hole():
    problem, _, filters = small_problem()
    active = problem.grid.active == 1.0
    x = np.where(active, 0.5, 0.0)
    ev, _ = evaluate(problem, x, filters, ("elasticity",))
    tip, pull = CaseCompliance("tip"), CaseCompliance("pull")
    bounds = {"tip": tip.value(ev), "pull": pull.value(ev)}
    start_mass = michell.Mass().value(ev)
    result = michell.optimise(
        problem,
        goal=michell.Mass(),
        limits=[michell.Limit(tip, bounds["tip"]), michell.Limit(pull, bounds["pull"])],
        filters=filters,
        optimiser=michell.MovingAsymptotes(),
        x=x,
        passes=60,
    )
    last = result.passes[-1]
    excess = [last.limits[0] / bounds["tip"] - 1.0, last.limits[1] / bounds["pull"] - 1.0]
    print(f"mass {last.goal:.4g} g from {start_mass:.4g} g; limit excess {excess[0]:.1e}, {excess[1]:.1e}")
    assert np.all(result.x[~active] == 0.0)
    assert last.goal < 0.9 * start_mass
    assert max(excess) <= 1e-3
