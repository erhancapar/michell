"""Verification rows 7 and 8: the gradients through the filter, and the MBB beam against the 88-line code.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured. Row 8 on the 150 x 50 beam takes
the longest and is marked slow: `pytest -m slow` runs it.
"""

from pathlib import Path

import pytest

from michell import verification

REFERENCE = Path(__file__).parent / "reference"


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_7_gradients_through_the_filter_match_finite_differences():
    rows = verification.row_7_gradients()
    assert len(rows) == 2
    assert_rows_pass(rows)


@pytest.mark.parametrize(
    "name",
    [
        "mbb_60x20_r2.4_density",
        pytest.param("mbb_150x50_r6_density", marks=pytest.mark.slow),
    ],
)
def test_row_8_mbb_beam_with_oc_follows_top88(name):
    rows = verification.row_8_mbb_with_oc(REFERENCE, name)
    assert len(rows) == 2
    assert_rows_pass(rows)
