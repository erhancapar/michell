"""Verification row 33: the problem file and `michell run` on the example bridge. `pytest -s -k row` prints
what each line measured."""

import warnings
from pathlib import Path

import pytest

from michell import verification

EXAMPLES = Path(__file__).parent.parent / "examples"


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_33_the_example_bridge_from_its_problem_file_and_one_refusal_per_fault():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = verification.row_33_problem_file(EXAMPLES)
    report(rows)
    assert len(rows) == 16
    assert all(row.number == 33 and row.passed for row in rows)


def test_row_33_without_the_example_files_reports_not_run(tmp_path):
    rows = verification.row_33_problem_file(tmp_path) + verification.row_33_run(None)
    assert len(rows) == 19
    assert all(row.passed is None and row.note == "not run: example files not found" for row in rows)


def test_row_33_finds_the_examples_beside_the_reference_folder():
    assert verification.examples_beside(EXAMPLES.parent / "tests" / "reference") == EXAMPLES
    assert verification.examples_beside(None) is None


@pytest.mark.slow
def test_row_33_michell_run_on_the_example_bridge():
    rows = verification.row_33_run(EXAMPLES)
    report(rows)
    assert len(rows) == 3
    assert all(row.number == 33 and row.passed for row in rows)
