"""One check of the code against a known answer: a line of a module's self-test or of the verification table.

Both use the same ``Row``. A module's ``self_test()`` (``docs/design/modules.md``) builds its rows with
``compared``; the verification table (``michell.verification``, ``docs/verification.md``) builds its rows
with ``checked`` and ``not_run``.
"""

from __future__ import annotations

from dataclasses import dataclass

# The kinds of check, as the verification table names them.
CLOSED_FORM = "closed form"
GRADIENT = "gradient"
REFERENCE_CODE = "reference code"
PUBLISHED_OPTIMUM = "published optimum"
INDEPENDENT_COMPUTATION = "independent computation"
REPEATABILITY = "repeatability"
CONVERGENCE = "convergence"
DESIGN_PROPERTY = "property of the design"


@dataclass(frozen=True)
class Row:
    """The result of one check: a measured number against a limit stated before it was measured.

    Every limit is an upper bound on ``measured``: ``strict`` rows must stay below it, the others at or
    below it. ``passed`` is ``None`` and ``measured`` is ``None`` when the row was not run (its reference
    data or SciPy missing); ``note`` then says why. ``number`` is the row's number in the verification
    table, ``None`` for a line of a module's self-test. Rows are built with ``compared``, ``checked`` or
    ``not_run``, which compute ``passed`` from the numbers; a row whose result needs more than one number
    (verification row 4: the error must also shrink) is built directly.
    """

    number: int | None
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


def compared(name: str, value: float, expected: float, tolerance: float, kind: str = CLOSED_FORM) -> Row:
    """A self-test row: ``value`` against a known ``expected`` value. The measured number is their
    difference relative to ``expected`` (absolute when ``expected`` is 0); the limit is ``tolerance``."""
    scale = abs(expected) if expected != 0 else 1.0
    difference = abs(value - expected) / scale
    note = f"value {value:.12g}, expected {expected:.12g}"
    return Row(None, name, kind, float(difference), tolerance, bool(difference <= tolerance), note)


def checked(number: int, name: str, kind: str, measured: float, limit: float, note: str, strict: bool = False) -> Row:
    """A verification row whose result follows from its measured value and its limit alone."""
    passed = measured < limit if strict else measured <= limit
    return Row(number, name, kind, float(measured), limit, bool(passed), note, strict)


def not_run(number: int, name: str, kind: str, limit: float, reason: str, strict: bool = False) -> Row:
    """A verification row that could not be run, with the reason."""
    return Row(number, name, kind, None, limit, None, f"not run: {reason}", strict)


def g3(value: float) -> str:
    """A number rounded to three significant digits, the precision of the table: 1007.02 becomes 1010,
    2.2451e-16 becomes 2.25e-16."""
    return f"{float(f'{value:.3g}'):g}"
