"""The verification rows of the engine, as functions that return their results.

Each function runs one row of the verification table (``docs/verification.md``) and returns ``Row``
objects: what was measured, the limit fixed before any result was computed, and whether the row passed.
The test suite calls these functions and asserts ``passed``; ``python -m michell.verification`` runs them
all and writes the table.

Rows 6, 8 and 10 compare with the output of the 88-line code kept in the repository under
``tests/reference``; they take that folder as an argument and report "not run" when it is absent. Row 9
solves its problems a second time with SciPy and row 14 assembles the 3D stiffness matrix with SciPy; both
report "not run" when SciPy is not installed. Row 15 runs the 3D analysis on several thread counts. Row 16
compares the 3D solver with a sparse direct solution computed by SciPy (also "not run" without it), row 17
counts its steps, and row 18 compares a 3D cantilever with beam theory. Row 19 checks the gradients of the 3D
loop through the Helmholtz filter against finite differences, and row 20 the filter itself against its
properties and a 1D closed form. Row 21 checks the projection, row 22 the gradients of the robust
formulation (several designs from one set of variables) in 3D, and row 23 runs the robust MBB beam and
checks the mass, the discreteness and the order of the compliances of its three designs.
"""

from __future__ import annotations

from pathlib import Path

from michell.row import (
    CLOSED_FORM,
    CONVERGENCE,
    DESIGN_PROPERTY,
    GRADIENT,
    INDEPENDENT_COMPUTATION,
    PUBLISHED_OPTIMUM,
    REFERENCE_CODE,
    REPEATABILITY,
    Row,
)
from michell.verification.analysis3d_rows import (
    row_11_element_matrix,
    row_12_patch_test,
    row_13_pure_shear,
    row_14_product,
    row_15_thread_count,
)
from michell.verification.analysis_rows import (
    row_1_element_matrix,
    row_2_patch_test,
    row_3_pure_shear,
    row_4_cantilever,
    row_5_plate_with_a_hole,
    row_6_first_pass,
)
from michell.verification.filter3d_rows import row_19_gradients, row_20_filter
from michell.verification.mbb import MBB_CASES
from michell.verification.solver3d_rows import (
    ROW_16_CASES,
    row_16_direct_solve,
    row_17_design,
    row_17_uniform,
    row_18_cantilever,
)
from michell.verification.mma_rows import row_9_published_optima, row_10_mbb_with_mma
from michell.verification.optimisation_rows import row_7_gradients, row_8_mbb_with_oc
from michell.verification.robust_rows import row_21_projection, row_22_robust_gradients, row_23_robust_mbb


def all_rows(reference: Path | str | None = None) -> list[Row]:
    """Runs rows 1 to 23 in order. ``reference`` is the folder that holds ``top88/`` (in the repository:
    ``tests/reference``)."""
    rows = []
    rows += row_1_element_matrix()
    rows += row_2_patch_test()
    rows += row_3_pure_shear()
    rows += row_4_cantilever()
    rows += row_5_plate_with_a_hole()
    rows += row_6_first_pass(reference)
    rows += row_7_gradients()
    for name, _, _, _ in MBB_CASES:
        rows += row_8_mbb_with_oc(reference, name)
    rows += row_9_published_optima()
    rows += row_10_mbb_with_mma(reference)
    rows += row_11_element_matrix()
    rows += row_12_patch_test()
    rows += row_13_pure_shear()
    rows += row_14_product()
    rows += row_15_thread_count()
    for case in ROW_16_CASES:
        rows += row_16_direct_solve(case)
    rows += row_17_uniform()
    rows += row_17_design("lattice")
    rows += row_17_design("irregular")
    rows += row_18_cantilever()
    rows += row_19_gradients()
    rows += row_20_filter()
    rows += row_21_projection()
    rows += row_22_robust_gradients()
    rows += row_23_robust_mbb()
    return rows


__all__ = [
    "CLOSED_FORM",
    "CONVERGENCE",
    "DESIGN_PROPERTY",
    "GRADIENT",
    "INDEPENDENT_COMPUTATION",
    "MBB_CASES",
    "PUBLISHED_OPTIMUM",
    "REFERENCE_CODE",
    "REPEATABILITY",
    "ROW_16_CASES",
    "Row",
    "all_rows",
    "row_1_element_matrix",
    "row_2_patch_test",
    "row_3_pure_shear",
    "row_4_cantilever",
    "row_5_plate_with_a_hole",
    "row_6_first_pass",
    "row_7_gradients",
    "row_8_mbb_with_oc",
    "row_9_published_optima",
    "row_10_mbb_with_mma",
    "row_11_element_matrix",
    "row_12_patch_test",
    "row_13_pure_shear",
    "row_14_product",
    "row_15_thread_count",
    "row_16_direct_solve",
    "row_17_uniform",
    "row_17_design",
    "row_18_cantilever",
    "row_19_gradients",
    "row_20_filter",
    "row_21_projection",
    "row_22_robust_gradients",
    "row_23_robust_mbb",
]
