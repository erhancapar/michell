"""The errors that the checks written in Python raise: each one an ``EngineError`` with code
``invalid_input``, the same three fields as the engine's errors (``code``, ``message``, ``details``) and the
same text, ``"invalid_input: <message>"``.

Each is also the built-in exception that fits the check (``ValueError``, ``TypeError`` or ``KeyError``), so
``except ValueError`` catches a refused value as well.
Callers that want every refusal catch ``EngineError`` and read ``code``.

A problem file (``michell.problem_file``) is refused with a ``ProblemFileError``, one class and one stable
code per kind of fault, with ``path``, the key's place in the file (``supports[1].fix``; empty for the file
as a whole), and ``file``, the problem file:

========================  ====================  ==============================================================
class                     code                  the fault
========================  ====================  ==============================================================
``InvalidToml``           ``invalid_toml``      the file is not TOML (the message gives the line and column)
``UnknownSchema``         ``unknown_schema``    ``schema`` names a format other than ``"michell/0"``
``UnknownKey``            ``unknown_key``       a key the format does not have, anywhere in the file
``MissingKey``            ``missing_key``       a key the format needs is not there
``WrongType``             ``wrong_type``        a value of the wrong type or shape (a string for a number, two
                                                numbers for three)
``OutOfRange``            ``out_of_range``      a value outside its range (a negative size, a direction "w")
``UndefinedName``         ``unknown_name``      a name the file does not define, or a module michell does not
                                                have
``DuplicateName``         ``duplicate_name``    a name given twice (a region and a marker body, two load cases)
``MissingFile``           ``file_not_found``    the problem file or a geometry file it names does not exist, or
                                                cannot be read
``NotAvailableYet``       ``not_available_yet`` part of the format that is planned but not yet built (STEP
                                                geometry, symmetry, mesh output)
========================  ====================  ==============================================================

Each is also an ``InvalidValue`` (``WrongType`` an ``InvalidType``, ``UndefinedName`` an ``UnknownName``),
so the built-in exceptions catch it as above. ``as_dict()`` gives the error as plain data.
"""

from __future__ import annotations

import os

from michell._engine import EngineError


class _Refused(EngineError):
    # The key of a problem file the refusal belongs to, where one is known (see ProblemFileError).
    path = ""

    def __init__(self, message: str, details: dict[str, int | float] | None = None):
        super().__init__(f"invalid_input: {message}")
        self.code = "invalid_input"
        self.message = message
        self.details = dict(details or {})

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class InvalidValue(_Refused, ValueError):
    """A value michell refuses: ``EngineError`` with code ``invalid_input``, and a ``ValueError``."""


class InvalidType(_Refused, TypeError):
    """An argument of the wrong type: ``EngineError`` with code ``invalid_input``, and a ``TypeError``."""


class UnknownName(_Refused, KeyError):
    """A name michell does not know (a load case, a solved state): ``EngineError`` with code
    ``invalid_input``, and a ``KeyError``."""


class ProblemFileError(_Refused):
    """A problem file that michell refuses: ``code`` (one per kind of fault, the table above), ``path``
    (the key's place in the file, such as ``load_cases[0].loads[1].on``; empty for the file as a whole),
    ``file`` (the problem file), ``message`` (a sentence for people) and ``details`` (the numbers
    involved). ``str()`` gives ``"<code> at <path>: <message>"``."""

    CODE = "invalid_input"

    def __init__(
        self,
        message: str,
        path: str = "",
        file: str | os.PathLike | None = None,
        details: dict[str, int | float] | None = None,
    ):
        super().__init__(message, details)
        self.code = self.CODE
        self.path = path
        self.file = None if file is None else os.fspath(file)
        self.args = (str(self),)

    def __str__(self) -> str:
        where = f" at {self.path}" if self.path else ""
        return f"{self.code}{where}: {self.message}"

    def as_dict(self) -> dict:
        """The error as plain data: ``code``, ``path``, ``file``, ``message`` and ``details``."""
        return {"code": self.code, "path": self.path, "file": self.file, "message": self.message,
                "details": dict(self.details)}  # fmt: skip


class InvalidToml(ProblemFileError, InvalidValue):
    """The problem file is not TOML (or not UTF-8): code ``invalid_toml``."""

    CODE = "invalid_toml"


class UnknownSchema(ProblemFileError, InvalidValue):
    """``schema`` names a format michell does not read: code ``unknown_schema``."""

    CODE = "unknown_schema"


class UnknownKey(ProblemFileError, InvalidValue):
    """A key the format does not have: code ``unknown_key``. A typing mistake is never ignored."""

    CODE = "unknown_key"


class MissingKey(ProblemFileError, InvalidValue):
    """A key the format needs is missing: code ``missing_key``."""

    CODE = "missing_key"


class WrongType(ProblemFileError, InvalidType):
    """A value of the wrong type or shape: code ``wrong_type``."""

    CODE = "wrong_type"


class OutOfRange(ProblemFileError, InvalidValue):
    """A value outside its range: code ``out_of_range``."""

    CODE = "out_of_range"


class UndefinedName(ProblemFileError, UnknownName):
    """A name the file does not define (what a support or a load acts on), or a module michell does not
    have: code ``unknown_name``."""

    CODE = "unknown_name"


class DuplicateName(ProblemFileError, InvalidValue):
    """A name given twice: code ``duplicate_name``."""

    CODE = "duplicate_name"


class MissingFile(ProblemFileError, InvalidValue):
    """The problem file, or a geometry file it names, does not exist or cannot be read: code
    ``file_not_found``."""

    CODE = "file_not_found"


class NotAvailableYet(ProblemFileError, InvalidValue):
    """Part of the format that is planned but not built yet: code ``not_available_yet``. The message says
    what is planned."""

    CODE = "not_available_yet"


__all__ = [
    "DuplicateName",
    "InvalidToml",
    "InvalidType",
    "InvalidValue",
    "MissingFile",
    "MissingKey",
    "NotAvailableYet",
    "OutOfRange",
    "ProblemFileError",
    "UndefinedName",
    "UnknownKey",
    "UnknownName",
    "UnknownSchema",
    "WrongType",
]
