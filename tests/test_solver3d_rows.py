"""Verification rows 16 to 18: the multigrid-preconditioned 3D solver against a sparse direct solution, its
step counts, and a 3D cantilever against beam theory.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured. The verification table runs every
row at its stated size; here row 16 runs at reduced sizes (its direct solutions at about 20 000 elements
take tens of seconds), and the full sizes of rows 16 and 18 and the 96 x 48 x 48 case of row 17 are marked
slow: `pytest -m slow` runs them.
"""

import pytest

from michell import verification
from michell.verification.solver3d_rows import ROW_17_UNIFORM_SIZES

ROW_16_REDUCED = {
    "uniform": (20, 8, 8),
    "lattice": (20, 8, 8),
    "irregular": (20, 8, 8),
    "L-shaped": (15, 9, 11),
}


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


@pytest.mark.parametrize("case", list(ROW_16_REDUCED))
def test_row_16_solver_equals_a_sparse_direct_solution_at_reduced_size(case):
    assert_rows_pass(verification.row_16_direct_solve(case, ROW_16_REDUCED[case]))


@pytest.mark.slow
@pytest.mark.parametrize("case", list(verification.ROW_16_CASES))
def test_row_16_solver_equals_a_sparse_direct_solution(case):
    assert_rows_pass(verification.row_16_direct_solve(case))


def test_row_17_uniform_step_counts_at_24_and_48():
    assert_rows_pass(verification.row_17_uniform(ROW_17_UNIFORM_SIZES[:2]))


@pytest.mark.slow
def test_row_17_uniform_step_counts_at_24_48_and_96():
    assert_rows_pass(verification.row_17_uniform())


@pytest.mark.parametrize("case", ["lattice", "irregular"])
def test_row_17_steps_on_a_0_and_1_design(case):
    assert_rows_pass(verification.row_17_design(case))


@pytest.mark.parametrize("case", verification.ROW_17_DETACHED_CASES)
def test_row_17_steps_with_detached_material_or_the_load_on_void(case):
    assert_rows_pass(verification.row_17_detached(case))


def test_row_18_cantilever_against_timoshenko_at_4_and_8_through_the_depth():
    assert_rows_pass(verification.row_18_cantilever((4, 8)))


@pytest.mark.slow
def test_row_18_cantilever_against_timoshenko():
    assert_rows_pass(verification.row_18_cantilever())
