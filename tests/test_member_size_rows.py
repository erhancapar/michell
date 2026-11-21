"""Verification rows 25 to 27: the density filter on a 3D grid, and the minimum member size as a limit.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured. Row 27 (1200 passes of a 2D beam)
is marked slow: `pytest -m slow` runs it.
"""

import pytest

from michell import verification


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_25_density_filter_3d_against_a_direct_computation():
    rows = verification.row_25_filter()
    assert len(rows) == 7
    assert_rows_pass(rows)


def test_row_25_member_size_derivative_in_2d_and_3d():
    rows = verification.row_25_member_size_gradients()
    assert len(rows) == 2
    assert_rows_pass(rows)


def test_row_26_member_size_tells_thin_from_thick():
    rows = verification.row_26_thin_and_thick()
    assert len(rows) == 8
    assert_rows_pass(rows)


@pytest.mark.slow
def test_row_27_mbb_beam_with_the_member_size_limit():
    pytest.importorskip("scipy")
    rows = verification.row_27_member_size_mbb()
    assert len(rows) == 3
    assert_rows_pass(rows)
