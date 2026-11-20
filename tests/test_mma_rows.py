"""Verification rows 9 and 10: the method of moving asymptotes on two published problems and on the MBB beam.

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured.
"""

from pathlib import Path

from michell import verification

REFERENCE = Path(__file__).parent / "reference"


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_9_mma_reaches_published_optima():
    rows = verification.row_9_published_optima()
    assert len(rows) == 4
    assert_rows_pass(rows)


def test_row_10_mbb_beam_with_mma_matches_oc():
    rows = verification.row_10_mbb_with_mma(REFERENCE)
    assert len(rows) == 2
    assert_rows_pass(rows)
