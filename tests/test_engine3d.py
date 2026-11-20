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
    grid, model = clamped_block(tolerance=1e-10)
    assert (model.tolerance, model.max_steps) == (1e-10, 500)
    assert model.last_solve_report() is None
    rho = np.ones(grid.n_elements)
    loads = tip_load(grid)
    u = model.solve(rho, loads)
    report = model.last_solve_report()
    assert report["steps"] > 1 and report["relative_residual"] <= 1e-10
    model.solve(rho, loads, start=u)
    assert model.last_solve_report()["steps"] == 0
    # The adjoint of the compliance is the displacement itself.
    lam = model.adjoint(rho, loads, start=None)
    assert np.allclose(lam, u, rtol=0, atol=1e-9 * np.abs(u).max())


def test_too_few_steps_raise_solver_not_converged_with_the_numbers():
    grid, model = clamped_block(tolerance=1e-10, max_steps=3)
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


def test_element_stiffness_3d_is_24_by_24_and_symmetric():
    k = michell.element_stiffness_3d(1.0, 0.3, 1.0)
    assert k.shape == (24, 24)
    assert np.allclose(k, k.T, rtol=0, atol=1e-15)
    with pytest.raises(michell.EngineError):
        michell.element_stiffness_3d(1.0, 0.5, 1.0)
