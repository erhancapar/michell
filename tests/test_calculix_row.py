"""Verification row 24: the 3D analysis against CalculiX on the same voxels.

The row itself (set-up, measure, limit) is in `michell.verification.calculix_rows`; this test runs it on the
CalculiX output kept in `tests/reference/calculix` and asserts that it passes. `pytest -s -k row` prints what
it measured.
"""

from pathlib import Path

from michell import verification
from michell.verification.calculix_rows import calculix_design, design_fingerprint, read_calculix_reference

REFERENCE = Path(__file__).parent / "reference"


def test_the_design_built_here_is_the_one_calculix_solved():
    stored = read_calculix_reference(REFERENCE)
    _, solid = calculix_design()
    assert stored.case["solid_elements"] == int(solid.sum())
    assert stored.case["solid_mask_sha256"] == design_fingerprint(solid)
    assert len(stored.nodes) == stored.case["loaded_nodes"] == 25 * 25


def test_row_24_compliance_equals_calculix():
    rows = verification.row_24_calculix(REFERENCE)
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"
