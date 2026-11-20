"""Verification rows 1 to 6: the 2D analysis against closed forms, beam theory, Kirsch and the 88-line code.

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


def test_row_1_element_matrix_equals_the_closed_form():
    assert_rows_pass(verification.row_1_element_matrix())


def test_row_2_patch_test_reproduces_a_linear_field():
    assert_rows_pass(verification.row_2_patch_test())


def test_row_3_pure_shear_gives_g_times_gamma():
    assert_rows_pass(verification.row_3_pure_shear())


def test_row_4_cantilever_converges_to_timoshenko_beam_theory():
    assert_rows_pass(verification.row_4_cantilever())


def test_row_5_plate_with_a_hole_matches_kirsch():
    assert_rows_pass(verification.row_5_plate_with_a_hole())


def test_row_6_first_pass_compliance_equals_top88():
    rows = verification.row_6_first_pass(REFERENCE)
    assert len(rows) == 2
    assert_rows_pass(rows)
