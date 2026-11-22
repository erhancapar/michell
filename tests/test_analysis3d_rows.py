"""Verification rows 11 to 15: the 3D analysis against an exact integration, closed forms and an
independently assembled stiffness matrix, and the same numbers on any thread count (the MMA update
included).

The rows themselves (set-up, measure, limit) are in `michell.verification`; these tests run them and
assert that they pass. `pytest -s -k row` prints what each row measured.
"""

from michell import verification


def assert_rows_pass(rows):
    assert rows
    for row in rows:
        print(f"ROW {row.number} ({row.name}): measured {row.measured!r}, limit {row.limit!r}; {row.note}")
        assert row.passed is True, f"{row.name}: {row.note}"


def test_row_11_element_matrix_equals_an_exact_integration():
    rows = verification.row_11_element_matrix()
    assert len(rows) == 3
    assert_rows_pass(rows)


def test_row_12_patch_test_reproduces_a_linear_field_in_3d():
    assert_rows_pass(verification.row_12_patch_test())


def test_row_13_pure_shear_gives_g_times_gamma_in_each_plane():
    rows = verification.row_13_pure_shear()
    assert len(rows) == 3
    assert_rows_pass(rows)


def test_row_14_matrix_free_product_equals_the_assembled_matrix():
    assert_rows_pass(verification.row_14_product())


def test_row_15_the_same_numbers_on_1_2_5_and_12_threads_and_on_a_repeated_run():
    assert_rows_pass(verification.row_15_thread_count())


def test_row_15_the_mma_update_gives_the_same_designs_on_1_2_5_and_12_threads():
    assert_rows_pass(verification.row_15_mma_thread_count())
