"""The verification rows of the engine, as functions that return their results.

Each function runs one row of the verification table (``docs/verification.md``) and returns ``Row``
objects: what was measured, the limit fixed before any result was computed, and whether the row passed.
The test suite calls these functions and asserts ``passed``; ``python -m michell.verification`` runs them
all and writes the table.

Rows 6, 8 and 10 compare with the output of the 88-line code kept in the repository under
``tests/reference``; they take that folder as an argument and report "not run" when it is absent. Row 9
solves its problems a second time with SciPy and row 14 assembles the 3D stiffness matrix with SciPy; both
report "not run" when SciPy is not installed. Row 15 runs the 3D analysis and the MMA update on several thread counts. Row 16
compares the 3D solver with a sparse direct solution computed by SciPy (also "not run" without it), row 17
counts its steps, and row 18 compares a 3D cantilever with beam theory. Row 19 checks the gradients of the 3D
loop through the Helmholtz filter against finite differences, and row 20 the filter itself against its
properties and a 1D closed form. Row 21 checks the projection, row 22 the gradients of the robust
formulation (several designs from one set of variables) in 3D, and row 23 runs the robust MBB beam and
checks the mass, the discreteness and the order of the compliances of its three designs. Row 24 compares
the 3D analysis with CalculiX on the same voxels; it reads CalculiX's numbers from
``tests/reference/calculix`` and reports "not run" when they are absent. Row 25 checks the density filter
on a 3D grid against a direct computation and the derivative of the member-size limit against finite
differences, row 26 checks that the limit tells a thin member from a thick one, and row 27 runs an MBB
beam with the limit and measures its members by a morphological opening (SciPy; "not run" without it).
Row 28 runs the two-step procedure of ``michell.recipes.stiff_part`` on a 3D bridge and measures its mass,
its discreteness, its members (the opening of row 27 in 3D) and what the member-size step cost in
stiffness (SciPy; "not run" without it). Row 29 checks ``KeepSolid`` (elements held solid) on a 2D and a 3D
part: its forward and backward maps, the derivative of the compliance on the held elements, and a short
optimisation whose held elements must come back solid within its mass limit. Row 30 checks the voxeliser
(``michell.regions.occupancy``): boxes against their exact volumes, the convergence of a faceted sphere,
cylinder and torus as the voxels are halved, bodies whose edges and vertices lie exactly on rays, and the
thread count. Row 31 checks the voxel model of a part (``michell.regions.voxel_model``): its class
volumes, a solid region nested in the design region, two regions that meet inside a voxel, the mirror option
and the refusal of a region too thin for the voxels.
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
    row_15_mma_thread_count,
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
from michell.verification.calculix_rows import row_24_calculix
from michell.verification.filter3d_rows import row_19_gradients, row_20_filter
from michell.verification.geometry_rows import row_30_voxeliser, row_31_voxel_model
from michell.verification.keep_solid_rows import row_29_keep_solid
from michell.verification.mbb import MBB_CASES
from michell.verification.member_size_rows import (
    row_25_filter,
    row_25_member_size_gradients,
    row_26_thin_and_thick,
    row_27_member_size_mbb,
)
from michell.verification.solver3d_rows import (
    ROW_16_CASES,
    ROW_17_DETACHED_CASES,
    row_16_direct_solve,
    row_17_design,
    row_17_detached,
    row_17_uniform,
    row_18_cantilever,
)
from michell.verification.mma_rows import row_9_published_optima, row_10_mbb_with_mma
from michell.verification.optimisation_rows import row_7_gradients, row_8_mbb_with_oc
from michell.verification.recipe_rows import row_28_recipe_bridge
from michell.verification.robust_rows import row_21_projection, row_22_robust_gradients, row_23_robust_mbb


def all_rows(reference: Path | str | None = None) -> list[Row]:
    """Runs rows 1 to 31 in order. ``reference`` is the folder that holds ``top88/`` and ``calculix/`` (in
    the repository: ``tests/reference``)."""
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
    rows += row_15_mma_thread_count()
    for case in ROW_16_CASES:
        rows += row_16_direct_solve(case)
    rows += row_17_uniform()
    rows += row_17_design("lattice")
    rows += row_17_design("irregular")
    for case in ROW_17_DETACHED_CASES:
        rows += row_17_detached(case)
    rows += row_18_cantilever()
    rows += row_19_gradients()
    rows += row_20_filter()
    rows += row_21_projection()
    rows += row_22_robust_gradients()
    rows += row_23_robust_mbb()
    rows += row_24_calculix(reference)
    rows += row_25_filter()
    rows += row_25_member_size_gradients()
    rows += row_26_thin_and_thick()
    rows += row_27_member_size_mbb()
    rows += row_28_recipe_bridge()
    rows += row_29_keep_solid()
    rows += row_30_voxeliser()
    rows += row_31_voxel_model()
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
    "ROW_17_DETACHED_CASES",
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
    "row_15_mma_thread_count",
    "row_15_thread_count",
    "row_16_direct_solve",
    "row_17_uniform",
    "row_17_design",
    "row_17_detached",
    "row_18_cantilever",
    "row_19_gradients",
    "row_20_filter",
    "row_21_projection",
    "row_22_robust_gradients",
    "row_23_robust_mbb",
    "row_24_calculix",
    "row_25_filter",
    "row_25_member_size_gradients",
    "row_26_thin_and_thick",
    "row_27_member_size_mbb",
    "row_28_recipe_bridge",
    "row_29_keep_solid",
    "row_30_voxeliser",
    "row_31_voxel_model",
]
