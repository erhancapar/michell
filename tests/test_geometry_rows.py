"""Verification rows 30 and 31: the voxeliser and the voxel model of a part. `pytest -s -k row` prints what each
line measured."""

import warnings

from michell import verification


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_30_the_voxeliser():
    rows = verification.row_30_voxeliser()
    report(rows)
    assert len(rows) == 12
    assert all(row.number == 30 and row.passed for row in rows)


def test_row_31_the_voxel_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the row's models must not need a VoxelVolumeDiffers warning
        rows = verification.row_31_voxel_model()
    report(rows)
    assert len(rows) == 6
    assert all(row.number == 31 and row.passed for row in rows)
