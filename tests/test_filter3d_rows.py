"""Verification rows 19 and 20: the gradients of the 3D loop through the Helmholtz filter, and the filter.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured.
"""

from michell import verification
from michell.gradient_check import small_problem_3d


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_19_problem_is_small_enough_for_a_direct_solve():
    # Row 19 states that its elasticity solves are direct: one multigrid level, one conjugate-gradient step.
    problem, x, filters = small_problem_3d(seed=0)
    assert problem.elasticity.multigrid_levels == 1
    density = filters[0].forward(x)
    _, steps = problem.solve(density, problem.cases[0].loads)
    assert steps == 1


def test_row_19_gradients_through_the_helmholtz_filter_match_finite_differences():
    rows = verification.row_19_gradients()
    assert len(rows) == 2
    assert_rows_pass(rows)


def test_row_20_helmholtz_filter_keeps_the_sum_is_symmetric_and_matches_the_1d_solution():
    rows = verification.row_20_filter()
    assert len(rows) == 3
    assert_rows_pass(rows)
