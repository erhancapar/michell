"""Verification rows 34 and 35: the iso-surface of a field, Taubin smoothing and the 3MF file. `pytest -s -k
row` prints what each line measured."""

from michell import verification


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_34_the_iso_surface():
    rows = verification.row_34_iso_surface()
    report(rows)
    assert len(rows) == 12
    assert all(row.number == 34 and row.passed for row in rows)


def test_row_35_smoothing_and_the_3mf_file():
    rows = verification.row_35_smoothing_and_3mf()
    report(rows)
    assert len(rows) == 4
    assert all(row.number == 35 and row.passed for row in rows)
