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
