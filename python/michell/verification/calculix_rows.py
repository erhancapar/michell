"""Verification row 24: the 3D analysis against CalculiX on the same voxels.

CalculiX (G. Dhondt, www.calculix.de) is an independent finite-element program. Its element C3D8 is the
8-node brick with 2 x 2 x 2 Gauss integration, the element of the 3D analysis. The script
``tools/reference/make_calculix_reference.py`` writes the design below as a CalculiX input deck (one C3D8
element per solid voxel, no element for a void), runs CalculiX and keeps only numbers in
``tests/reference/calculix/``: the compliance, the displacements of the loaded nodes, and the grid and the
fingerprint of the design they belong to. This row builds the same design with the same functions and solves
it with the 3D analysis.

Set-up (fixed before any result was computed): the irregular 0-and-1 design of rows 16 and 17
(``designs3d.irregular``) on a 48 x 24 x 24 cantilever of unit cube voxels, E = 210000 MPa, Poisson ratio
0.3, clamped on the face x = 0, a total load of 1 N in -z spread over the face x = 48 by area. The voids are
left out of the analysis (inactive elements), as they have no element in CalculiX, and the solid has density
1. The 3D analysis solves to a relative residual of 1e-10. Limit: its compliance f^T u within 0.5 % of
CalculiX's.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from michell.analysis3d import Elasticity3d, Grid3d
from michell.row import REFERENCE_CODE, Row, checked, g3, not_run
from michell.verification.designs3d import E, NU, SetUp, irregular
from michell.verification.mbb import REFERENCE_NOT_FOUND
from michell.verification.solver3d_rows import ROW_THREADS

CALCULIX_SHAPE = (48, 24, 24)
CALCULIX_CASE = "cantilever_48x24x24_irregular"
ROW_24_LIMIT = 0.005


def calculix_design() -> tuple[SetUp, np.ndarray]:
    """The set-up of row 24 (the irregular design on the 48 x 24 x 24 cantilever) and its solid voxels,
    one bool per element. Raises ``ValueError`` if the solid does not join the two end faces."""
    setup, joined = irregular(CALCULIX_SHAPE)
    if not joined:
        raise ValueError("the irregular design does not join the held slab to the loaded slab")
    return setup, setup.rho > 0.5


def design_fingerprint(solid: np.ndarray) -> str:
    """The SHA-256 of the solid mask, one bit per element in element order (x fastest), as hexadecimal
    text. Two designs with the same fingerprint have the same voxels."""
    return hashlib.sha256(np.packbits(np.asarray(solid, dtype=bool)).tobytes()).hexdigest()


@dataclass(frozen=True)
class CalculixResult:
    """What the reference keeps of one CalculiX run: the case description (shape, material, design and its
    fingerprint, compliance) and the displacements of the loaded nodes, by michell node index."""

    case: dict
    nodes: np.ndarray  # michell node indices of the loaded nodes
    displacements: np.ndarray  # one row of ux, uy, uz per loaded node


def read_calculix_reference(reference: Path | str | None, name: str = CALCULIX_CASE) -> CalculixResult | None:
    """The stored CalculiX result ``name`` from ``reference/calculix/`` (``name.json`` and
    ``name_loaded_nodes.csv``), or ``None`` when the folder or a file is absent."""
    if reference is None:
        return None
    folder = Path(reference) / "calculix"
    case_file, nodes_file = folder / f"{name}.json", folder / f"{name}_loaded_nodes.csv"
    if not case_file.is_file() or not nodes_file.is_file():
        return None
    case = json.loads(case_file.read_text(encoding="utf-8"))
    with open(nodes_file, newline="") as file:
        lines = list(csv.DictReader(file))
    nodes = np.array([int(line["node"]) for line in lines])
    displacements = np.array([[float(line[c]) for c in ("ux", "uy", "uz")] for line in lines])
    return CalculixResult(case, nodes, displacements)


# ---------------------------------------------------------------------------------------------------------
# Row 24. Limit: the compliance of the 3D analysis (voids inactive, solved to 1e-10) within a relative 0.5 %
# of CalculiX's on the same voxels. The largest difference of a loaded node's displacement is reported, not
# judged.


def row_24_calculix(reference: Path | str | None) -> list[Row]:
    nx, ny, nz = CALCULIX_SHAPE
    name = f"3D cantilever {nx} x {ny} x {nz}, irregular 0-and-1 design: compliance against CalculiX"
    stored = read_calculix_reference(reference)
    if stored is None:
        return [not_run(24, name, REFERENCE_CODE, ROW_24_LIMIT, REFERENCE_NOT_FOUND)]
    setup, solid = calculix_design()
    case = stored.case
    fingerprint = design_fingerprint(solid)
    if (
        tuple(case["shape"]) != CALCULIX_SHAPE
        or case["solid_elements"] != int(solid.sum())
        or case["solid_mask_sha256"] != fingerprint
    ):
        note = (
            f"the design built here ({int(solid.sum())} solid elements, fingerprint {fingerprint[:12]}) is not the "
            f"one CalculiX solved ({case['solid_elements']} solid elements, fingerprint "
            f"{case['solid_mask_sha256'][:12]})"
        )
        return [Row(24, name, REFERENCE_CODE, None, ROW_24_LIMIT, False, note)]

    grid = Grid3d(nx, ny, nz, 1.0, active=solid)
    model = Elasticity3d(
        grid, setup.held, youngs_modulus=E, poisson_ratio=NU, e_min=1e-9 * E, penalty=3.0,
        tolerance=1e-10, max_steps=5000, threads=ROW_THREADS,
    )  # fmt: skip
    u = model.solve(np.ones(grid.n_elements), setup.loads)
    steps = model.last_solve_report()["steps"]
    compliance = float(setup.loads @ u)
    expected = float(case["compliance"])
    error = abs(compliance - expected) / expected

    ours = u.reshape(-1, 3)[stored.nodes]
    largest = float(np.abs(ours - stored.displacements).max() / np.abs(stored.displacements).max())
    note = (
        f"relative difference; compliance {g3(compliance)} N mm here, {g3(expected)} N mm by "
        f"{case['program']}; {int(solid.sum())} solid elements, one C3D8 element each in CalculiX; "
        f"{steps} steps to 1e-10; displacements of the {len(stored.nodes)} loaded nodes: largest difference "
        f"over the largest displacement {g3(largest)} (reported, not judged), CalculiX printing them to "
        f"{case['printed_digits']} significant digits"
    )
    return [checked(24, name, REFERENCE_CODE, error, ROW_24_LIMIT, note)]
