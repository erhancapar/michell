"""The MBB beam of the 88-line code, and its reference files.

Andreassen, Clausen, Schevenels, Lazarov and Sigmund, "Efficient topology optimization in MATLAB using 88
lines of code", Struct. Multidisc. Optim. 43 (2011). The reference files hold the compliance of every pass of
that code, run in GNU Octave (see ``tests/reference/top88/PROVENANCE.txt``).

Set-up of that code: half of the beam, square elements of unit side and thickness, unit modulus, E_min 1e-9,
Poisson 0.3, penalty 3, a unit load down on the top-left node, x held on the whole left edge (the symmetry
line), y held at the bottom-right node; volume fraction 0.5, here a mass limit of 0.5 nx ny g with a
material density of 1 g/mm^3 (1000 g/cm^3).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from michell.analysis import Grid
from michell.problem import LoadCase, Material, Problem

# The reference cases: file name, nx, ny, filter radius.
MBB_CASES = [
    ("mbb_60x20_r2.4_density", 60, 20, 2.4),
    ("mbb_150x50_r6_density", 150, 50, 6.0),
]

REFERENCE_NOT_FOUND = "reference data not found"


def mbb_case(name: str) -> tuple[int, int, float]:
    """nx, ny and the filter radius of the reference case ``name``."""
    for case_name, nx, ny, radius in MBB_CASES:
        if case_name == name:
            return nx, ny, radius
    raise KeyError(f"no MBB reference case called {name!r}; the cases are {[c[0] for c in MBB_CASES]}")


def mbb_problem(nx: int, ny: int) -> Problem:
    """The MBB beam of the 88-line code on nx by ny elements."""
    grid = Grid(nx, ny, 1.0, 1.0)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for j in range(ny + 1):
        held[2 * grid.node(0, j)] = True
    held[2 * grid.node(nx, 0) + 1] = True
    loads = np.zeros(grid.n_dofs)
    loads[2 * grid.node(0, ny) + 1] = -1.0
    material = Material(youngs_modulus=1.0, poisson_ratio=0.3, density_g_cm3=1000.0)
    return Problem(grid, held, material, [LoadCase("load", loads)], penalty=3.0, e_min=1e-9)


def read_reference(reference: Path | str | None, name: str) -> list[dict[str, str]] | None:
    """The passes of reference file ``name`` (columns pass, compliance, volume_fraction, max_change) from
    ``reference/top88/``, or ``None`` when the folder or the file is absent."""
    if reference is None:
        return None
    path = Path(reference) / "top88" / f"{name}.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as file:
        return list(csv.DictReader(file))
