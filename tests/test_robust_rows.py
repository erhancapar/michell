"""Verification rows 21 to 23: the projection, the gradients of the robust formulation in 3D, and the robust
MBB beam.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured. Row 23 (300 passes) is marked slow:
`pytest -m slow` runs it.
"""

import pytest

from michell import verification


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_21_projection_derivative_and_ends():
    rows = verification.row_21_projection()
    assert len(rows) == 2
    assert_rows_pass(rows)


def test_row_22_robust_gradients_through_the_helmholtz_filter_and_the_projection():
    rows = verification.row_22_robust_gradients()
    assert len(rows) == 2
    assert_rows_pass(rows)


@pytest.mark.slow
def test_row_23_robust_mbb_beam_mass_discreteness_and_order():
    rows = verification.row_23_robust_mbb()
    assert len(rows) == 3
    assert_rows_pass(rows)
