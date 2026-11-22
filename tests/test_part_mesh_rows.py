"""Verification rows 36 and 37: the part's closed mesh on results built by hand and on optimised parts.
`pytest -s -k row` prints what each line measured."""

from pathlib import Path

import pytest

from michell import verification

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_36_the_part_mesh_on_results_built_by_hand():
    rows = verification.row_36_part_mesh()
    report(rows)
    assert len(rows) == 11
    assert all(row.number == 36 and row.passed for row in rows)


def test_row_37_without_the_examples_is_not_run(tmp_path):
    rows = verification.row_37_bridge(tmp_path)
    assert len(rows) == 3
    assert all(row.passed is None and row.note == "not run: example files not found" for row in rows)


@pytest.mark.slow
def test_row_37_the_held_skin_of_the_bore_stays_inside():
    rows = verification.row_37_bore()
    report(rows)
    assert len(rows) == 1 and rows[0].number == 37 and rows[0].passed


@pytest.mark.slow
def test_row_37_michell_run_on_the_example_bridge_writes_the_mesh():
    rows = verification.row_37_bridge(EXAMPLES)
    report(rows)
    assert len(rows) == 3
    assert all(row.number == 37 and row.passed for row in rows)
