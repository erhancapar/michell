"""The module interface: what a response (a goal or a limit) and a filter must provide.

See ``docs/design/modules.md``. A response turns the density and the solved state into one number and
writes down its partial derivatives; it never solves anything. The driver completes the gradient with the
adjoint method (``michell.adjoint``) and pushes it back through the filters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from michell.evaluation import Evaluation


@dataclass(frozen=True)
class Partials:
    """The partial derivatives of one response, all local.

    ``d_density``: dg/d(rho) with the state held fixed, one value per element, or ``None`` when g does
    not depend on the density directly. ``d_state``: dg/d(state) for each (physics, case) the response
    reads, one value per degree of freedom.
    """

    d_density: np.ndarray | None
    d_state: Mapping[tuple[str, str], np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Row:
    """One line of a module's own proof: a measured value against a known one.

    ``tolerance`` is relative to ``expected`` (absolute when ``expected`` is 0) and is stated before the
    value is measured. ``passed`` is computed from the numbers, never set by hand.
    """

    name: str
    measured: float
    expected: float
    tolerance: float

    @property
    def passed(self) -> bool:
        scale = abs(self.expected) if self.expected != 0 else 1.0
        return abs(self.measured - self.expected) <= self.tolerance * scale


class Response(Protocol):
    """A goal or a limit: one number from the density and the solved state."""

    name: str  # "compliance", "mass"
    unit: str  # "N mm", "g"
    reads: Sequence[str]  # physics that must be solved first: ("elasticity",) or ()

    def value(self, ev: Evaluation) -> float: ...

    def partials(self, ev: Evaluation) -> Partials: ...

    def self_test(self) -> Sequence[Row]: ...


class Filter(Protocol):
    """A map from the design variables (or the output of the previous filter) to a new field."""

    name: str

    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        """The chain rule back to x: dg/dx from dg/d(output), taken at x."""
        ...

    def self_test(self) -> Sequence[Row]: ...
