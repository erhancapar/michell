"""Verification row 32: supports and loads on the voxel model. `pytest -s -k row` prints what each line
measured."""

import warnings

import pytest

from michell import verification


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_32_supports_and_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the row catches the one warning it expects (LoadOnSupport) itself
        rows = verification.row_32_attach()
    report(rows)
    assert len(rows) == 13
    assert all(row.number == 32 and row.passed for row in rows)


@pytest.mark.slow
def test_row_32_a_bore_loaded_through_a_pin():
    rows = verification.row_32_bore()
    report(rows)
    assert len(rows) == 4
    assert all(row.number == 32 and row.passed for row in rows)


@pytest.mark.slow
def test_row_32_the_example_bridge_s_compliance():
    rows = verification.row_32_bridge_compliance()
    report(rows)
    assert len(rows) == 1
    assert rows[0].passed, rows[0].note
