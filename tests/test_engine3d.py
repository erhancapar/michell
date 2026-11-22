"""Tests for the Python face of the 3D analysis: the grid, the calls of Elasticity3d, its solver settings
and report, and the rules of the Rust boundary."""

import numpy as np
import pytest

import michell
from michell import _engine


def clamped_block(nx=4, ny=2, nz=2, **settings):
    """A grid clamped on its face x = 0, and its elasticity model (unit modulus)."""
    grid = michell.Grid3d(nx, ny, nz, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(nz + 1):
        for j in range(ny + 1):
            node = grid.node(0, j, k)
            held[3 * node : 3 * node + 3] = True
    model = michell.Elasticity3d(grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0, **settings)
    return grid, model


def tip_load(grid):
    loads = np.zeros(grid.n_dofs)
    loads[3 * grid.node(grid.nx, grid.ny // 2, grid.nz) + 2] = -1.0
    return loads


def test_the_package_offers_the_engine_classes():
    assert issubclass(michell.Grid3d, _engine.Grid3d)
    assert issubclass(michell.Elasticity3d, _engine.Elasticity3d)
    assert michell.element_stiffness_3d is _engine.element_stiffness_3d


def test_grid_numbering_runs_x_then_y_then_z():
    grid = michell.Grid3d(3, 2, 4, 0.5)
    assert (grid.n_elements, grid.n_nodes, grid.n_dofs) == (24, 60, 180)
    assert grid.element(2, 1, 3) == 2 + 3 * (1 + 2 * 3)
    assert grid.node(3, 2, 4) == 59
    assert grid.voxel_volume_mm3 == 0.125
    assert grid.active.dtype == np.float64 and grid.active.sum() == 24
    mask = np.ones(24, dtype=bool)
    mask[5] = False
    assert michell.Grid3d(3, 2, 4, 0.5, active=mask).active.sum() == 23
    with pytest.raises(michell.EngineError) as error:
        grid.node(4, 0, 0)
    assert error.value.code == "invalid_input"


def test_solve_reports_its_steps_and_a_start_at_the_solution_takes_none():
    # 16 x 8 x 8 elements: two multigrid levels.
    grid, model = clamped_block(16, 8, 8, tolerance=1e-10)
    assert (model.tolerance, model.max_steps, model.multigrid_levels) == (1e-10, 5000, 2)
    assert model.last_solve_report() is None
    rho = np.ones(grid.n_elements)
    loads = tip_load(grid)
    u = model.solve(rho, loads)
    report = model.last_solve_report()
    assert report["steps"] > 1 and report["relative_residual"] <= 1e-10
    # Each step: its own product and one V-cycle with one smoothing sweep before and after.
    assert report["products"] == 1 + 3 * report["steps"]
    model.solve(rho, loads, start=u)
    assert model.last_solve_report()["steps"] == 0
    # The adjoint of the compliance is the displacement itself.
    lam = model.adjoint(rho, loads, start=None)
    assert np.allclose(lam, u, rtol=0, atol=1e-9 * np.abs(u).max())


def test_the_step_cap_defaults_to_that_of_problem_in_the_binding_too():
    grid, model = clamped_block()
    held = np.zeros(grid.n_dofs)
    for j, k in ((0, 0), (2, 0), (0, 2)):  # three corners of the face x = 0, every degree of freedom
        held[3 * grid.node(0, j, k) : 3 * grid.node(0, j, k) + 3] = 1.0
    bare = _engine.Elasticity3d(grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0)
    assert model.max_steps == bare.max_steps == 5000


def test_a_grid_that_is_its_own_coarsest_level_is_solved_in_one_step():
    grid, model = clamped_block(tolerance=1e-10)
    assert model.multigrid_levels == 1
    model.solve(np.ones(grid.n_elements), tip_load(grid))
    assert model.last_solve_report()["steps"] == 1


def test_too_few_steps_raise_solver_not_converged_with_the_numbers():
    grid, model = clamped_block(16, 8, 8, tolerance=1e-10, max_steps=3)
    rho = np.ones(grid.n_elements)
    with pytest.raises(michell.EngineError) as error:
        model.solve(rho, tip_load(grid))
    assert error.value.code == "solver_not_converged"
    assert error.value.details["steps"] == 3
    assert error.value.details["relative_residual"] > 1e-10
    assert model.last_solve_report() is None


def test_per_element_quantities_have_their_shapes_and_compliance_is_twice_the_energy():
    grid, model = clamped_block(tolerance=1e-12, max_steps=1000)
    rho = np.full(grid.n_elements, 0.6)
    loads = tip_load(grid)
    u = model.solve(rho, loads)
    energy = model.element_strain_energy(rho, u)
    assert energy.shape == (grid.n_elements,)
    assert abs(loads @ u - 2 * energy.sum()) < 1e-10 * (loads @ u)
    sensitivity = model.sensitivity(rho, u, u)
    # For compliance: -p rho^(p-1) (E_0 - E_min) u_e^T K_unit u_e = -(p / rho) * 2 * energy_e at E_min -> 0.
    assert np.allclose(sensitivity, -3.0 / 0.6 * 2 * energy, rtol=1e-6)
    stress = model.element_stress(rho, u)
    assert stress.shape == (grid.n_elements, 6) and stress.dtype == np.float64
    product = model.product(rho, u)
    free = loads != 0
    assert np.allclose(product[free], loads[free])


def test_wrong_input_is_refused_never_converted():
    grid, model = clamped_block()
    rho = np.ones(grid.n_elements)
    loads = tip_load(grid)
    for wrong in (rho.astype(np.float32), list(rho), rho[:-1]):
        with pytest.raises(michell.EngineError) as error:
            model.solve(wrong, loads)
        assert error.value.code == "invalid_input"
    with pytest.raises(michell.EngineError):
        model.solve(rho, loads, start=np.zeros(5))
    with pytest.raises(michell.EngineError):
        clamped_block(tolerance=0.0)
    with pytest.raises(michell.EngineError):
        clamped_block(max_steps=-1)


def test_threads_is_an_explicit_setting_and_does_not_change_the_numbers():
    # 16 x 8 x 8 elements: 4131 degrees of freedom, enough for the work to be shared out.
    grid, one = clamped_block(16, 8, 8, tolerance=1e-10)
    assert one.threads == 1
    rho = np.full(grid.n_elements, 0.7)
    loads = tip_load(grid)
    u = one.solve(rho, loads)
    # More threads than this machine may have cores: the threads take turns.
    _, many = clamped_block(16, 8, 8, tolerance=1e-10, threads=12)
    assert many.threads == 12
    assert many.solve(rho, loads).tobytes() == u.tobytes()
    assert many.last_solve_report() == one.last_solve_report()
    with pytest.raises(michell.EngineError) as error:
        clamped_block(threads=0)
    assert error.value.code == "invalid_input" and error.value.details["threads"] == 0
    with pytest.raises(michell.EngineError) as error:
        clamped_block(threads=-1)
    assert error.value.code == "invalid_input"


def test_element_stiffness_3d_is_24_by_24_and_symmetric():
    k = michell.element_stiffness_3d(1.0, 0.3, 1.0)
    assert k.shape == (24, 24)
    assert np.allclose(k, k.T, rtol=0, atol=1e-15)
    with pytest.raises(michell.EngineError):
        michell.element_stiffness_3d(1.0, 0.5, 1.0)


def held_model(grid, held):
    return michell.Elasticity3d(grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0)


def held_face(grid):
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(grid.nz + 1):
        for j in range(grid.ny + 1):
            node = grid.node(0, j, k)
            held[3 * node : 3 * node + 3] = True
    return held


def assert_not_held(grid, held, elements=None):
    with pytest.raises(michell.EngineError) as error:
        held_model(grid, held)
    assert error.value.code == "singular_system"
    if elements is not None:
        assert error.value.details["elements"] == elements


def test_a_part_without_support_or_held_at_one_node_is_refused():
    grid = michell.Grid3d(6, 3, 3, 1.0)
    assert_not_held(grid, np.zeros(grid.n_dofs, dtype=bool), elements=54)
    one = np.zeros(grid.n_dofs, dtype=bool)
    node = grid.node(0, 1, 1)
    one[3 * node : 3 * node + 3] = True
    assert_not_held(grid, one)


def test_a_loose_second_body_and_a_block_joined_along_an_edge_are_refused():
    # 8 x 3 x 3: a clamped bar in the elements i < 4, and a loose 2 x 3 x 3 block at i >= 6.
    active = np.zeros((3, 3, 8), dtype=bool)  # [k, j, i]
    active[:, :, :4] = True
    active[:, :, 6:] = True
    grid = michell.Grid3d(8, 3, 3, 1.0, active=active.ravel())
    assert_not_held(grid, held_face(grid), elements=18)
    # A 2 x 3 x 3 bar clamped at x = 0 and one more element at (2, 3, 3) of a 3 x 4 x 4 grid, touching the
    # bar only along the edge x = 2, y = 3, z = 3.
    active = np.zeros((4, 4, 3), dtype=bool)
    active[:3, :3, :2] = True
    active[3, 3, 2] = True
    grid = michell.Grid3d(3, 4, 4, 1.0, active=active.ravel())
    assert_not_held(grid, held_face(grid), elements=1)
    # Joined through a face instead, the same block is held.
    active[3, 3, 2] = False
    active[2, 2, 2] = True
    held_model(michell.Grid3d(3, 4, 4, 1.0, active=active.ravel()), held_face(grid))


def test_a_thin_plate_builds_its_levels_and_solves():
    # 64 x 64 x 2: the depth stops halving at one element while the plane goes on (three levels).
    grid = michell.Grid3d(64, 64, 2, 1.0)
    model = michell.Elasticity3d(
        grid, held_face(grid), youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0,
        tolerance=1e-8, threads=4,
    )  # fmt: skip
    assert model.multigrid_levels == 3
    loads = np.zeros(grid.n_dofs)
    loads[3 * grid.node(64, 32, 2) + 2] = -1.0
    model.solve(np.ones(grid.n_elements), loads)
    report = model.last_solve_report()
    assert report["relative_residual"] <= 1e-8 and report["steps"] < 100
