"""The gradient of a response with respect to the design variables x.

A response gives its partial derivatives only (``Partials``). This file completes them with the adjoint
method and then applies the filters backwards:

    K lambda = dg/du,        dg/drho_e = dg/drho_e (partial) - lambda^T (dK/drho_e) u,

for each (physics, case) the response reads (the adjoint method; see Bendsøe and Sigmund, "Topology
Optimization", 2003). The engine's ``sensitivity`` returns the second term with its minus sign, so it is added.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from michell.evaluation import Evaluation
from michell.modules import Filter, Response
from michell.problem import Problem

# d_state counts as a multiple of the load vector when it differs from c * loads by at most this fraction
# of its largest entry (rounding only).
LOAD_MULTIPLE_TOLERANCE = 1e-12


def multiple_of_loads(d_state: np.ndarray, loads: np.ndarray) -> float | None:
    """The number c with d_state = c * loads, or None if there is none."""
    norm = loads @ loads
    if norm == 0.0:
        return None
    c = (d_state @ loads) / norm
    if np.abs(d_state - c * loads).max() <= LOAD_MULTIPLE_TOLERANCE * np.abs(d_state).max():
        return c
    return None


def adjoint_state(
    problem: Problem, density: np.ndarray, u: np.ndarray, loads: np.ndarray, d_state: np.ndarray
) -> np.ndarray | None:
    """lambda with K(rho) lambda = d_state, or None when d_state is zero (no contribution).

    When d_state is a multiple c of the load vector, lambda is that multiple of the state, c * u, and the
    extra solve is skipped (compliance is the classic case).
    """
    if not np.any(d_state):
        return None
    c = multiple_of_loads(d_state, loads)
    if c is not None:
        return c * u
    return problem.elasticity.adjoint(density, d_state)


def backward_through_filters(
    filters: Sequence[Filter], fields: Sequence[np.ndarray], grad: np.ndarray
) -> np.ndarray:
    """dg/dx from dg/drho, by the chain rule through the filters in reverse order.
    ``fields[i]`` is the input of ``filters[i]`` (as returned by ``apply_filters``)."""
    for index in reversed(range(len(filters))):
        grad = filters[index].backward(fields[index], grad)
    return grad


def total_gradient(
    problem: Problem,
    response: Response,
    ev: Evaluation,
    fields: Sequence[np.ndarray],
    filters: Sequence[Filter],
) -> np.ndarray:
    """dg/dx of ``response`` at the evaluation ``ev`` (made from ``fields`` by ``filters``)."""
    partials = response.partials(ev)
    n_elements = problem.grid.n_elements
    if partials.d_density is None:
        d_density = np.zeros(n_elements)
    else:
        d_density = np.array(partials.d_density, dtype=np.float64)  # a copy: it is added to below
        if d_density.shape != (n_elements,):
            raise ValueError(
                f"{response.name}: d_density needs one value per element ({n_elements}); "
                f"got shape {d_density.shape}"
            )
    for (physics, case), d_state in partials.d_state.items():
        if physics not in response.reads:
            raise ValueError(f"{response.name}: d_state for {physics!r}, which is not in `reads`")
        d_state = np.ascontiguousarray(d_state, dtype=np.float64)
        u = ev.state(physics, case)
        lam = adjoint_state(problem, ev.density, u, ev.loads(physics, case), d_state)
        if lam is not None:
            d_density += problem.elasticity.sensitivity(ev.density, u, lam)
    return backward_through_filters(filters, fields, d_density)
