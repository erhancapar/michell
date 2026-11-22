"""Verification row 29: KeepSolid on a 2D and a 3D part, its forward and backward maps, the derivative of the
compliance on the held elements and a short optimisation. `pytest -s -k row` prints what each line measured."""

import numpy as np

from michell import verification
from michell.verification.keep_solid_rows import held_part, held_part_run


def report(rows):
    for row in rows:
        print(f"row {row.number}: {row.name}: measured {row.measured:.3g}, limit {row.limit:g} ({row.note})")


def test_row_29_keep_solid_in_2d_and_3d():
    rows = verification.row_29_keep_solid()
    report(rows)
    assert len(rows) == 12
    assert all(row.number == 29 and row.passed for row in rows)


def test_the_short_runs_of_row_29_meet_their_limit_and_return_the_held_elements_solid():
    for dimension in (2, 3):
        part = held_part(dimension)
        result, _, _ = held_part_run(part)
        assert result.limits_met == (True,) and result.feasible
        assert np.all(result.density[part.keep_solid.mask] == 1.0)
