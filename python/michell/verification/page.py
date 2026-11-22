"""The verification table as a Markdown page."""

from __future__ import annotations

from collections.abc import Sequence

from michell.row import Row, g3

INTRODUCTION = """\
This table checks the 2D and the 3D analysis, the 2D and the 3D optimisation, the density filter on a 3D grid,
the Helmholtz filter, the projection and the member-size limit against closed-form solutions, against an exact
integration and an independent assembly of the 3D stiffness matrix, against a sparse direct solution of the 3D
system, against central finite differences, against the published 88-line topology optimisation code
(Andreassen et al., Struct. Multidisc. Optim. 43, 2011, run in GNU Octave; see
`tests/reference/top88/PROVENANCE.txt`), against published optima (Svanberg, Int. J. Numer. Meth. Eng. 24,
1987) and against the finite-element program CalculiX on the same voxels (see
`tests/reference/calculix/PROVENANCE.txt`); it also runs the 3D analysis and the filters on several thread
counts and twice in a row and compares the numbers, counts the steps the 3D solver takes, checks the three
designs of a robust optimisation (Wang, Lazarov and Sigmund, Struct. Multidisc. Optim. 43, 2011) for their
mass, their discreteness (the measure of Sigmund, Struct. Multidisc. Optim. 33, 2007) and the order of their
compliances, and checks the minimum member size as a limit (Zhou, Lazarov, Wang and Sigmund, Comput. Methods
Appl. Mech. Engrg. 293, 2015) on thin and thick members and on an optimised beam, whose members it measures by
a morphological opening; and it runs the two-step procedure of `michell.recipes` (the robust formulation,
then the member-size limit) on a 3D bridge and measures the design it returns, and checks that elements held solid
(`KeepSolid`) stay solid and carry no derivative, also through a short optimisation; it checks the
voxeliser, which turns closed triangle meshes into the share of every voxel inside them, against exact
volumes, as the voxels are halved, on bodies whose edges and vertices lie exactly on its rays and on 1 and 12
threads, and the voxel model of a part built from design, solid and void regions; and it checks supports and
loads placed on that model by the bodies they touch against closed forms and against a part set up by hand. The measure, the set-up and the limit of every row were fixed before its first result
was computed, with four exceptions, stated in their notes: in row 27 the measure and its 5 % limit were fixed before
any run, while the member size, the mass fraction, the 10 % tolerance on the limit's bound and the 10 % guard
were set after trial runs; and in row 30 the convergence measure (the largest error over 8 placements of
each body) and the voxel sizes were set after two trial runs: one placement per size, whose error changed
sign from size to size, then this measure at 2 to 0.25 mm, where the torus's error fell by 2.7 only from 2
to 1 mm; its code states both; in row 32 the shortened schedule of the bore's run was set after a trial run
with that of the recipe's tests, which does not meet the member-size limit; and the fifth line of row 9 (the two-bar truss with its goal written 1e6 times
smaller) was added, with the limit of the row's first lines, after its problem was known to stop the subproblem
solver of an earlier version. Every limit is an upper bound on the measured value. Measured values are rounded
to three significant digits; values at the level of floating-point round-off may differ in those digits from
one machine to another. The code of every row, with its set-up, is in `python/michell/verification/`, and
continuous integration runs every row on each change."""

REGENERATING = """\
## Regenerating this page

From the repository, with the package and SciPy installed:

    python -m michell.verification --reference tests/reference --out docs/verification.md

Rows 6, 8 and 10 read the output of the 88-line code from `tests/reference`, and row 24 the output of
CalculiX (`tools/reference/make_calculix_reference.py` writes it; CalculiX is not needed to run the row);
row 9 solves its problems a second time with SciPy, row 14 assembles the 3D stiffness matrix with SciPy,
row 16 solves the 3D system directly with SciPy, and rows 27 and 28 measure their members with SciPy's
morphological opening. Without them those rows are listed as not run. The command
exits with status 1 when a row fails (and, with `--require-all`, when a row was not run). The test suite runs
the same functions."""


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|")


def result_text(row: Row) -> str:
    if row.passed is None:
        return "not run"
    return "pass" if row.passed else "**fail**"


def render(rows: Sequence[Row], version: str, engine_version: str, commit: str | None = None) -> str:
    """The page: what the table is, the version (and commit), the table, how to regenerate it."""
    lines = ["# Verification", "", INTRODUCTION, ""]
    lines.append(f"Version: michell {version} (engine {engine_version})" + (f", commit `{commit}`." if commit else "."))
    lines += [
        "",
        "| Row | Check | Kind | Measured | Limit | Result | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        measured = "" if row.measured is None else g3(row.measured)
        limit = ("< " if row.strict else "≤ ") + g3(row.limit)
        lines.append(
            f"| {row.number} | {cell(row.name)} | {row.kind} | {measured} | {limit} | {result_text(row)} | "
            f"{cell(row.note)} |"
        )
    passed = sum(1 for row in rows if row.passed is True)
    failed = sum(1 for row in rows if row.passed is False)
    not_run = sum(1 for row in rows if row.passed is None)
    lines += ["", f"{passed} passed, {failed} failed, {not_run} not run.", "", REGENERATING, ""]
    return "\n".join(lines)


def exit_status(rows: Sequence[Row], require_all: bool = False) -> int:
    """0 when every row that ran passed (and, with ``require_all``, every row ran); 1 otherwise."""
    if any(row.passed is False for row in rows):
        return 1
    if require_all and any(row.passed is None for row in rows):
        return 1
    return 0
