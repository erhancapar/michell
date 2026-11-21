"""Several designs from one set of design variables: the projection, `on`, `robust_designs`, and the loop
with `designs=` (only the designs read are solved, steps and warm starts per design, the final fields)."""

import numpy as np
import pytest

import michell
from michell.designs import design_chains, evaluate_designs
from michell.evaluation import apply_filters
from michell.gradient_check import small_problem
from michell.verification.robust_rows import robust_mbb


def test_the_projection_passes_its_self_test_and_needs_no_grid():
    projection = michell.Projection(8.0, 0.75)
    rows = projection.self_test()
    assert rows and all(row.passed for row in rows), [(row.name, row.note) for row in rows if not row.passed]
    assert (projection.name, projection.beta, projection.eta) == ("projection", 8.0, 0.75)
    assert michell.Projection(2.0).eta == 0.5
    values = projection.forward(np.linspace(0.0, 1.0, 7))
    assert values[0] == 0.0 and values[-1] == 1.0 and np.all(np.diff(values) > 0)


@pytest.mark.parametrize("beta, eta", [(0.0, 0.5), (-2.0, 0.5), (8.0, -0.01), (8.0, 1.01), (float("nan"), 0.5)])
def test_the_projection_refuses_a_wrong_sharpness_or_threshold(beta, eta):
    with pytest.raises(michell.EngineError) as error:
        michell.Projection(beta, eta)
    assert error.value.code == "invalid_input"


def test_the_robust_set_has_three_thresholds_around_one_half():
    designs = michell.robust_designs(16.0, 0.2)
    assert list(designs) == ["eroded", "nominal", "dilated"]
    thresholds = [chain[0].eta for chain in designs.values()]
    assert thresholds == pytest.approx([0.7, 0.5, 0.3])
    assert all(len(chain) == 1 and chain[0].beta == 16.0 for chain in designs.values())
    for spread in (0.0, 0.5, -0.1):
        with pytest.raises(ValueError, match="spread"):
            michell.robust_designs(8.0, spread)


def test_on_names_a_design_and_reads_like_its_response():
    response = michell.on("eroded", michell.Compliance())
    assert (response.design, response.name, response.unit, tuple(response.reads)) == (
        "eroded",
        "compliance",
        "N mm",
        ("elasticity",),
    )
    with pytest.raises(ValueError, match="already reads"):
        michell.on("nominal", response)


def run_2d(goal, limit_response, designs, passes=2, optimiser=None):
    problem, x, filters = small_problem()
    return michell.optimise(
        problem,
        goal,
        [michell.Limit(limit_response, at_most=0.4 * 7.85e-3 * 1.5 * 1.5 * 2.0 * problem.grid.n_elements)],
        filters,
        optimiser or michell.MovingAsymptotes(),
        x,
        passes=passes,
        designs=designs,
    )


def test_with_designs_every_goal_and_limit_must_name_a_known_design():
    designs = michell.robust_designs(4.0, 0.25)
    with pytest.raises(ValueError, match="must name the design"):
        run_2d(michell.Compliance(), michell.on("dilated", michell.Mass()), designs)
    with pytest.raises(ValueError, match="must name the design"):
        run_2d(michell.on("eroded", michell.Compliance()), michell.Mass(), designs)
    with pytest.raises(ValueError, match="'thin'"):
        run_2d(michell.on("thin", michell.Compliance()), michell.on("dilated", michell.Mass()), designs)
    with pytest.raises(ValueError, match="no designs were given"):
        run_2d(michell.on("eroded", michell.Compliance()), michell.Mass(), None)
    with pytest.raises(TypeError, match="must be a list"):
        run_2d(michell.on("a", michell.Compliance()), michell.on("a", michell.Mass()), {"a": michell.Projection(2.0)})


def test_oc_refuses_a_limit_on_a_named_design_naming_mma():
    with pytest.raises(ValueError, match="MMA"):
        run_2d(
            michell.on("eroded", michell.Compliance()),
            michell.on("dilated", michell.Mass()),
            michell.robust_designs(4.0, 0.25),
            optimiser=michell.OptimalityCriteria(),
        )


def test_one_design_without_filters_of_its_own_gives_the_run_without_designs():
    plain = run_2d(michell.Compliance(), michell.Mass(), None, passes=3)
    named = run_2d(michell.on("only", michell.Compliance()), michell.on("only", michell.Mass()), {"only": []}, passes=3)
    assert np.array_equal(plain.x, named.x)
    assert [p.goal for p in plain.passes] == [p.goal for p in named.passes]
    assert plain.designs == {} and all(p.design_solver_steps == {} for p in plain.passes)
    assert np.array_equal(named.designs["only"], named.density)


def cantilever_3d_problem(nx=8, ny=4, nz=4):
    """A block of nx by ny by nz elements of side 1.5 mm clamped on x = 0, a load down at the middle of its
    lower far edge, a Helmholtz filter of radius 2.5 element sides, and x drawn from [0.3, 0.7]."""
    grid = michell.Grid3d(nx, ny, nz, 1.5)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(nz + 1):
        for j in range(ny + 1):
            node = grid.node(0, j, k)
            held[3 * node : 3 * node + 3] = True
    loads = np.zeros(grid.n_dofs)
    loads[3 * grid.node(nx, ny // 2, 0) + 2] = -1000.0
    steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    problem = michell.Problem(grid, held, steel, [michell.LoadCase("down", loads)], solver_tolerance=1e-8)
    x = np.random.default_rng(3).uniform(0.3, 0.7, grid.n_elements)
    return problem, x, [michell.HelmholtzFilter(grid, 3.75)]


def test_only_the_designs_read_are_solved_and_every_design_is_returned():
    problem, x, filters = cantilever_3d_problem()
    grid = problem.grid
    designs = michell.robust_designs(8.0, 0.25)
    result = michell.optimise(
        problem,
        michell.on("eroded", michell.Compliance()),
        [michell.Limit(michell.on("dilated", michell.Mass()), at_most=0.4 * 7.85e-3 * 1.5**3 * grid.n_elements)],
        filters,
        michell.MovingAsymptotes(),
        x,
        passes=2,
        designs=designs,
    )
    # Mass reads no physics, so the dilated design is projected but never solved; nominal is not read at all.
    assert all(set(p.design_solver_steps) == {"eroded"} for p in result.passes)
    assert all(p.solver_steps == p.design_solver_steps["eroded"] for p in result.passes)
    assert list(result.designs) == ["eroded", "nominal", "dilated"]
    for name, chain in designs.items():
        assert np.array_equal(result.designs[name], apply_filters(result.density, chain)[-1])
    assert np.all(result.designs["eroded"] <= result.designs["nominal"])
    assert np.all(result.designs["nominal"] <= result.designs["dilated"])


def test_two_solved_designs_keep_their_own_steps():
    problem, x, filters = cantilever_3d_problem(16, 8, 8)  # large enough for a multigrid of two levels
    assert problem.elasticity.multigrid_levels > 1
    designs = michell.robust_designs(4.0, 0.2)
    first = 7.85e-3 * 1.5**3 * problem.grid.n_elements
    result = michell.optimise(
        problem,
        michell.on("eroded", michell.Compliance()),
        [
            michell.Limit(michell.on("nominal", michell.Compliance()), at_most=1e6),
            michell.Limit(michell.on("dilated", michell.Mass()), at_most=0.4 * first),
        ],
        filters,
        michell.MovingAsymptotes(),
        x,
        passes=3,
        designs=designs,
    )
    for p in result.passes:
        assert set(p.design_solver_steps) == {"eroded", "nominal"}
        assert p.solver_steps == sum(p.design_solver_steps.values())


def test_each_design_takes_its_warm_start_from_its_own_states():
    problem, x, filters = cantilever_3d_problem(16, 8, 8)
    chains = design_chains(michell.robust_designs(4.0, 0.2))
    reads = {"eroded": {"elasticity"}, "nominal": {"elasticity"}}
    _, first = evaluate_designs(problem, x, filters, chains, reads)
    assert all(ev.solver_steps > 1 for ev, _ in first.values())
    # Started from its own solution, each design has nothing left to do ...
    own = {name: ev.states for name, (ev, _) in first.items()}
    _, again = evaluate_designs(problem, x, filters, chains, reads, start=own)
    assert all(ev.solver_steps == 0 for ev, _ in again.values())
    # ... and started from the other design's solution, it has.
    swapped = {"eroded": own["nominal"], "nominal": own["eroded"]}
    _, crossed = evaluate_designs(problem, x, filters, chains, reads, start=swapped)
    assert all(ev.solver_steps > 0 for ev, _ in crossed.values())


def test_the_robust_mbb_rescales_the_dilated_bound_before_each_step():
    problem, run = robust_mbb(30, 10, betas=(1.0, 2.0), passes_per_step=4)
    assert run.passes == 8 and len(run.bounds) == 2
    # The first bound comes from x = 0.5 everywhere: the dilated design weighs more than the nominal one.
    assert run.bounds[0] > run.asked_mass
    assert set(run.result.designs) == {"eroded", "nominal", "dilated"}
