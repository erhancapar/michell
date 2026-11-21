"""The errors that the checks written in Python raise: each one an ``EngineError`` with code
``invalid_input``, the same three fields as the engine's errors (``code``, ``message``, ``details``) and the
same text, ``"invalid_input: <message>"``.

Each is also the built-in exception that fits the check (``ValueError``, ``TypeError`` or ``KeyError``), so
``except ValueError`` catches a refused value as well.
Callers that want every refusal catch ``EngineError`` and read ``code``.
"""

from __future__ import annotations

from michell._engine import EngineError


class _Refused(EngineError):
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
