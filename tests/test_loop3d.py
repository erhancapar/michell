"""The optimisation loop on a 3D grid: the problem, warm starts, the record of a pass, the optimisers and the
Helmholtz filter, and the 3D cantilever example (slow)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import michell
from michell.evaluation import evaluate
from michell.gradient_check import small_problem_3d

ROOT = Path(__file__).parent.parent
STEEL = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)


def cantilever_3d(nx=12, ny=4, nz=4, threads=1, **settings):
    """A block clamped on x = 0 with a load down at the middle of its lower far edge."""
    grid = michell.Grid3d(nx, ny, nz, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(nz + 1):
        for j in range(ny + 1):
            node = grid.node(0, j, k)
            held[3 * node : 3 * node + 3] = True
    loads = np.zeros(grid.n_dofs)
    loads[3 * grid.node(nx, ny // 2, 0) + 2] = -100.0
    return michell.Problem(grid, held, STEEL, [michell.LoadCase("tip", loads)], threads=threads, **settings)


def test_the_problem_picks_the_analysis_from_the_grid():
    problem = cantilever_3d(threads=2, solver_tolerance=1e-8, solver_max_steps=200)
    assert isinstance(problem.elasticity, michell.Elasticity3d)
    assert (problem.elasticity.threads, problem.elasticity.tolerance, problem.elasticity.max_steps) == (2, 1e-8, 200)
    assert problem.is_3d and problem.threads == 2


@pytest.mark.parametrize("setting", [{"threads": 2}, {"solver_tolerance": 1e-8}, {"solver_max_steps": 100}])
def test_3d_settings_are_refused_on_a_2d_grid(setting):
    grid = michell.Grid(4, 2, 1.0, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    held[: 2 * 3] = True
    case = michell.LoadCase("none", np.zeros(grid.n_dofs))
    with pytest.raises(ValueError, match="only apply to a 3D grid"):
        michell.Problem(grid, held, STEEL, [case], **setting)
    assert not michell.Problem(grid, held, STEEL, [case]).is_3d


def test_a_warm_start_is_taken_from_the_states_handed_in_and_nothing_is_kept():
    problem = cantilever_3d(solver_tolerance=1e-10)
    x = np.full(problem.grid.n_elements, 0.5)
    first, _ = evaluate(problem, x, [], ("elasticity",))
    assert first.solver_steps > 0
    # Starting from the solution of the same design, the solve has nothing left to do.
    again, _ = evaluate(problem, x, [], ("elasticity",), start=first.states)
    assert again.solver_steps == 0
    # Without a start, the same steps as the first time: the analysis kept nothing.
    fresh, _ = evaluate(problem, x, [], ("elasticity",))
    assert fresh.solver_steps == first.solver_steps
    assert np.array_equal(fresh.state("elasticity", "tip"), first.state("elasticity", "tip"))
    # A start for a state that is not solved is ignored; a state missing from the start is solved from zero.
    other, _ = evaluate(problem, x, [], ("elasticity",), start={("elasticity", "elsewhere"): np.ones(3)})
    assert other.solver_steps == first.solver_steps


def test_the_record_holds_the_solver_steps_and_the_threads():
    problem = cantilever_3d(threads=2)
    result = michell.optimise(
        problem,
        goal=michell.Compliance(),
        limits=[michell.Limit(michell.Mass(), at_most=0.3 * STEEL.mass_g(problem.grid.n_elements))],
        filters=[michell.HelmholtzFilter(problem.grid, 2.0, threads=2)],
        optimiser=michell.MovingAsymptotes(),
        x=np.full(problem.grid.n_elements, 0.3),
        passes=8,
    )
    assert result.threads == 2
    assert all(p.solver_steps > 0 for p in result.passes)
    assert result.passes[-1].goal < result.passes[0].goal


def test_a_2d_run_records_no_solver_steps_and_one_thread():
    problem, x, filters = michell.gradient_check.small_problem()
    mass = michell.Mass()
    limit = 0.4 * mass.value(evaluate(problem, np.ones(problem.grid.n_elements), [], ())[0])
    result = michell.optimise(
        problem, michell.Compliance(), [michell.Limit(mass, limit)], filters, michell.MovingAsymptotes(), x, passes=2
    )
    assert result.threads == 1
    assert [p.solver_steps for p in result.passes] == [0, 0]


def test_oc_refuses_a_3d_grid_and_any_filter_but_the_density_filter_naming_mma():
    problem = cantilever_3d()
    limit = [michell.Limit(michell.Mass(), at_most=1.0)]
    x = np.full(problem.grid.n_elements, 0.3)
    with pytest.raises(ValueError, match="MMA"):
        michell.OptimalityCriteria().start(problem, x, limit, [])
    problem_2d, x_2d, _ = michell.gradient_check.small_problem()
    pde_filter = michell.HelmholtzFilter(michell.Grid3d(2, 2, 2, 1.0), 2.0)
    with pytest.raises(ValueError, match="MMA"):
        michell.OptimalityCriteria().start(problem_2d, x_2d, limit, [pde_filter])


def test_the_helmholtz_filter_passes_its_self_test_and_takes_only_a_3d_grid():
    pde_filter = michell.HelmholtzFilter(michell.Grid3d(4, 3, 2, 1.0), 2.5)
    rows = pde_filter.self_test()
    assert rows and all(row.passed for row in rows), [(row.name, row.note) for row in rows if not row.passed]
    assert pde_filter.name == "Helmholtz filter"
    assert pde_filter.length == pytest.approx(2.5 / (2 * np.sqrt(3)))
    with pytest.raises(TypeError):
        michell.HelmholtzFilter(michell.Grid(4, 3, 1.0, 1.0), 2.5)


@pytest.mark.parametrize("radius", [0.3, 1.0, 1.5])
def test_the_helmholtz_filter_keeps_a_sharp_field_in_range_for_a_small_radius(radius):
    grid = michell.Grid3d(9, 9, 9, 1.0)
    x = np.zeros(grid.n_elements)
    x[grid.element(4, 4, 4)] = 1.0
    pde_filter = michell.HelmholtzFilter(grid, radius)
    # The linear map itself (backward, which applies no clip) stays in [0, 1] to the solver's tolerance.
    values = pde_filter.backward(x, x)
    assert values.min() >= -1e-10 and values.max() <= 1.0 + 1e-10
    assert abs(values.sum() - 1.0) < 1e-9
    rho = pde_filter.forward(x)
    assert rho.min() >= 0.0 and rho.max() <= 1.0


def test_the_gradient_check_runs_unchanged_on_a_3d_problem():
    problem, x, filters = small_problem_3d(seed=1)
    check = michell.check_gradient(problem, michell.Compliance(), x, filters, elements=[0, 17, 101])
    assert check.largest_difference < 1e-5


@pytest.mark.slow
def test_the_3d_cantilever_example_meets_its_mass_limit_and_halves_the_compliance(tmp_path):
    picture = tmp_path / "cantilever.png"
    command = [sys.executable, str(ROOT / "examples" / "cantilever_3d.py"), "--nx", "32", "--ny", "16", "--nz", "16"]
    command += ["--passes", "50", "--threads", "4", "--out", str(picture)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"compliance {summary['compliance_first']:.4g} -> {summary['compliance_last']:.4g} N mm, ", end="")
    print(f"mass {summary['mass']:.6g} of at most {summary['mass_limit']:.6g} g, {summary['seconds']:.1f} s")
    # The mass of the final design (after the last update), within a relative 1e-6 of the limit at most.
    assert summary["mass"] <= summary["mass_limit"] * (1 + 1e-6)
    assert summary["compliance_last"] <= 0.5 * summary["compliance_first"]
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.slow
def test_the_3d_cantilever_example_runs_with_the_robust_formulation(tmp_path):
    picture = tmp_path / "cantilever_robust.png"
    command = [sys.executable, str(ROOT / "examples" / "cantilever_3d.py"), "--nx", "24", "--ny", "12", "--nz", "12"]
    command += ["--passes", "36", "--threads", "4", "--robust", "--out", str(picture)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["robust"] is True and summary["passes"] == 36
    for name in ("eroded", "nominal", "dilated"):
        assert f"{name:8}: compliance" in result.stdout
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
