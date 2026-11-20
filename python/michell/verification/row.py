"""One line of the verification table."""

from __future__ import annotations

from dataclasses import dataclass

# The kinds of check, as the table names them.
CLOSED_FORM = "closed form"
GRADIENT = "gradient"
REFERENCE_CODE = "reference code"
PUBLISHED_OPTIMUM = "published optimum"


@dataclass(frozen=True)
class Row:
    """The result of one check of the verification table.

    Every limit is an upper bound on ``measured``: ``strict`` rows must stay below it, the others at or
    below it. ``passed`` is ``None`` and ``measured`` is ``None`` when the row was not run (its reference
    data or SciPy missing); ``note`` then says why.

    This differs from ``michell.Row``, a line of a module's own self-test, which compares a measured value
    with an expected one.
    """

    number: int
    name: str
    kind: str
    measured: float | None
    limit: float
    passed: bool | None
    note: str
    strict: bool = False

    @property
    def was_run(self) -> bool:
        return self.passed is not None


def checked(number: int, name: str, kind: str, measured: float, limit: float, note: str, strict: bool = False) -> Row:
    """A row whose result follows from its measured value and its limit alone."""
    passed = measured < limit if strict else measured <= limit
    return Row(number, name, kind, float(measured), limit, bool(passed), note, strict)


def not_run(number: int, name: str, kind: str, limit: float, reason: str, strict: bool = False) -> Row:
    """A row that could not be run, with the reason."""
    return Row(number, name, kind, None, limit, None, f"not run: {reason}", strict)


def g3(value: float) -> str:
    """A number rounded to three significant digits, the precision of the table: 1007.02 becomes 1010,
    2.2451e-16 becomes 2.25e-16."""
    return f"{float(f'{value:.3g}'):g}"
