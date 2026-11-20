"""Tests for the Python face of michell: import, versions and the rules of the Rust boundary."""

import subprocess
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import michell
from michell import _engine
from michell.cli import main


def small_cantilever(nx=4, ny=2):
    """A grid clamped on its left edge, and its elasticity model (unit modulus)."""
    grid = michell.Grid(nx, ny, 1.0, 1.0)
    held = np.zeros(grid.n_dofs)
    for j in range(ny + 1):
        node = grid.node(0, j)
        held[2 * node] = held[2 * node + 1] = 1.0
    model = michell.Elasticity(grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0)
    return grid, model


def test_package_version_matches_metadata():
    assert michell.__version__ == version("michell") == "0.0.1"


def test_engine_version_is_the_rust_crate_version():
    assert _engine.engine_version() == "0.0.1"


def test_user_code_sees_the_engine_through_the_package():
    # Grid and Elasticity are subclasses that also accept boolean masks; the work is the engine's.
    assert issubclass(michell.Grid, _engine.Grid)
    assert issubclass(michell.Elasticity, _engine.Elasticity)
    assert issubclass(michell.DensityFilter, _engine.DensityFilter)
    assert michell.Oc is _engine.Oc
    assert michell.EngineError is _engine.EngineError
    assert issubclass(michell.EngineError, Exception)


def test_grid_numbering_runs_x_fastest():
    grid = michell.Grid(3, 2, 0.5, 2.0)
    assert (grid.n_elements, grid.n_nodes, grid.n_dofs) == (6, 12, 24)
    assert grid.element(2, 1) == 5
    assert grid.node(3, 2) == 11
    assert grid.active.dtype == np.float64 and grid.active.sum() == 6


def test_outputs_are_new_float64_arrays():
    grid, model = small_cantilever()
    rho = np.ones(grid.n_elements)
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(4, 1) + 1] = -1.0
    u = model.solve(rho, loads)
    assert u.dtype == np.float64 and u.shape == (grid.n_dofs,)
    # New memory (handed over from Rust, so numpy reports its owner as a capsule, not the array itself).
    assert u.flags.writeable and u.flags.c_contiguous
    assert not np.shares_memory(u, loads) and not np.shares_memory(u, rho)
    assert model.element_stress(rho, u).shape == (grid.n_elements, 3)


def test_a_non_contiguous_array_is_refused():
    grid, model = small_cantilever()
    every_second = np.ones(2 * grid.n_elements)[::2]
    assert not every_second.flags.c_contiguous
    with pytest.raises(michell.EngineError, match="contiguous") as caught:
        model.solve(every_second, np.zeros(grid.n_dofs))
    assert caught.value.code == "invalid_input"
    # The message's advice works.
    model.solve(np.ascontiguousarray(every_second), np.zeros(grid.n_dofs))


def test_other_dtypes_are_refused_not_converted():
    grid, model = small_cantilever()
    with pytest.raises(michell.EngineError, match="float64") as caught:
        model.solve(np.ones(grid.n_elements, dtype=np.int64), np.zeros(grid.n_dofs))
    assert caught.value.code == "invalid_input"
    with pytest.raises(michell.EngineError, match="float64"):
        model.solve([1.0] * grid.n_elements, np.zeros(grid.n_dofs))


def test_a_wrong_length_is_refused_with_the_numbers():
    grid, model = small_cantilever()
    with pytest.raises(michell.EngineError) as caught:
        model.solve(np.ones(grid.n_elements - 1), np.zeros(grid.n_dofs))
    error = caught.value
    assert error.code == "invalid_input"
    assert "rho" in error.message
    assert error.details == {"expected": grid.n_elements, "got": grid.n_elements - 1}
    assert isinstance(error.details["expected"], int)


def test_a_density_outside_0_to_1_is_refused():
    grid, model = small_cantilever()
    rho = np.ones(grid.n_elements)
    rho[3] = 1.5
    with pytest.raises(michell.EngineError) as caught:
        model.solve(rho, np.zeros(grid.n_dofs))
    assert caught.value.details == {"element": 3, "rho": 1.5}


def test_a_mask_holds_only_zeros_and_ones():
    with pytest.raises(michell.EngineError, match="0.0 and 1.0") as caught:
        michell.Grid(2, 2, 1.0, 1.0, active=np.array([1.0, 0.5, 1.0, 1.0]))
    assert caught.value.code == "invalid_input"


def test_a_part_that_is_not_held_is_a_singular_system():
    grid = michell.Grid(6, 3, 1.0, 1.0)
    with pytest.raises(michell.EngineError) as caught:
        michell.Elasticity(
            grid, np.zeros(grid.n_dofs), youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0
        )
    error = caught.value
    assert error.code == "singular_system"
    assert set(error.details) == {"dof", "node", "pivot", "diagonal"}
    assert str(error).startswith("singular_system: ")


def test_compliance_is_twice_the_strain_energy_and_the_adjoint_of_compliance_is_u():
    grid, model = small_cantilever(6, 3)
    rho = np.linspace(0.2, 1.0, grid.n_elements)
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(6, 3) + 1] = -1.0
    u = model.solve(rho, loads)
    compliance = loads @ u
    assert compliance == pytest.approx(2 * model.element_strain_energy(rho, u).sum(), rel=1e-12)
    assert np.allclose(model.adjoint(rho, loads), u, rtol=0, atol=1e-12 * np.abs(u).max())


def test_sensitivity_of_compliance_matches_central_differences():
    grid, model = small_cantilever(6, 3)
    rho = np.linspace(0.2, 0.9, grid.n_elements)
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(6, 3) + 1] = -1.0
    u = model.solve(rho, loads)
    gradient = model.sensitivity(rho, u, u)
    step = 1e-6
    for element in (0, 9, 17):
        up, down = rho.copy(), rho.copy()
        up[element] += step
        down[element] -= step
        difference = (loads @ model.solve(up, loads) - loads @ model.solve(down, loads)) / (2 * step)
        assert gradient[element] == pytest.approx(difference, rel=1e-6)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "michell 0.0.1 (engine 0.0.1)"


def test_installed_console_script():
    script = "michell.exe" if sys.platform == "win32" else "michell"
    command = Path(sysconfig.get_path("scripts")) / script
    result = subprocess.run([command, "--version"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "michell 0.0.1 (engine 0.0.1)"


def test_a_grid_too_large_to_number_is_refused_not_a_crash():
    with pytest.raises(michell.EngineError, match="too large") as caught:
        michell.Grid(2**40, 2**40, 1.0, 1.0)
    assert caught.value.code == "invalid_input"


def test_dot_sums_in_index_order_and_refuses_wrong_input():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, -5.0, 6.0])
    assert michell.dot(a, b) == 12.0
    # In index order 1e16 + 1 rounds back to 1e16, so the sum is exactly 0.
    assert michell.dot(np.array([1e16, 1.0, -1e16]), np.ones(3)) == 0.0
    read_only = np.array(a)
    read_only.flags.writeable = False  # as the arrays of an Evaluation
    assert michell.dot(read_only, b) == 12.0
    with pytest.raises(michell.EngineError) as caught:
        michell.dot(a, b[:2])
    assert caught.value.code == "invalid_input"
    assert caught.value.details == {"length_a": 3, "length_b": 2}
    with pytest.raises(michell.EngineError, match="float64"):
        michell.dot(a.astype(np.int64), b)
