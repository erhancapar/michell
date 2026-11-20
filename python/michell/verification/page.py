"""The verification table as a Markdown page."""

from __future__ import annotations

from collections.abc import Sequence

from michell.row import Row, g3

INTRODUCTION = """\
This table checks the 2D and the 3D analysis, the 2D and the 3D optimisation and the Helmholtz filter against
closed-form solutions, against an exact integration and an independent assembly of the 3D stiffness matrix,
against a sparse direct solution of the 3D system, against central finite differences, against the published
88-line topology optimisation code (Andreassen et al., Struct. Multidisc. Optim. 43, 2011, run in GNU Octave; see
`tests/reference/top88/PROVENANCE.txt`) and against published optima (Svanberg, Int. J. Numer. Meth. Eng. 24,
1987); it also runs the 3D analysis and the Helmholtz filter on several thread counts and twice in a row and
compares the numbers, and
counts the steps the 3D solver takes. The measure, the set-up and the limit of every row were fixed before
its first result was computed; a limit is never changed to make a row pass. Every limit is an upper bound on
the measured value. Measured values are rounded to three significant digits; values at the level of
floating-point round-off may differ in those digits from one machine to another. The code of every row, with
its set-up, is in `python/michell/verification/`, and continuous integration runs every row on each change."""

REGENERATING = """\
## Regenerating this page

From the repository, with the package and SciPy installed:

    python -m michell.verification --reference tests/reference --out docs/verification.md

Rows 6, 8 and 10 read the output of the 88-line code from `tests/reference`; row 9 solves its problems a
second time with SciPy, row 14 assembles the 3D stiffness matrix with SciPy, and row 16 solves the 3D system
directly with SciPy. Without them those rows are listed as not run. The command exits with status 1 when a row fails (and, with `--require-all`, when a row
was not run). The test suite runs the same functions."""


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
