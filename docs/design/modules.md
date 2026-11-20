# Modules: how goals, limits and shape rules plug in

Status: design note, kept in step with the code. The interface, the `Compliance` and `Mass` responses, the
optimisation loop on a 2D or a 3D grid (a `Problem` picks the analysis from the grid's type), the density
filter (2D), the Helmholtz filter (3D) and both optimisers below exist.

## The picture

One optimisation pass is a chain, and its gradient runs the same chain backwards:

```
design variables x  ->  filters  ->  density rho  ->  physics solve  ->  state (e.g. displacements)
                                                                             |
                    optimiser  <-  gradients  <-  responses (goal and limits) <-
```

- **x**: one number between 0 and 1 per design element. These are what the optimiser changes.
- **Filters** turn x into the physical density **rho** (0 = empty, 1 = solid). Smoothing, sharpening and every
  manufacturing rule live here.
- **Physics** (in the Rust engine) solves for the state: displacements for elasticity, temperatures for heat.
- **Responses** turn density and state into single numbers: stiffness, mass, peak stress, a temperature.
  One response is the goal; the others are limits.

A module is a small Python class that fills one of two slots: a **response** or a **filter**. Physics is not a
Python module; it lives in the engine (see `engine-boundary.md`).

## A response: one number and its partial derivatives

```python
class Response(Protocol):
    name: str                  # "compliance", "mass"
    unit: str                  # "N mm", "g"
    reads: Sequence[str]       # physics that must be solved first: ("elasticity",) or ()

    def value(self, ev: Evaluation) -> float: ...
    def partials(self, ev: Evaluation) -> Partials: ...
    def self_test(self) -> Sequence[Row]: ...
```

`Evaluation` is read-only and holds what a response may look at:

| Field | What it is |
|---|---|
| `ev.grid` | grid shape, voxel size, which elements are active |
| `ev.density` | rho, one float per element |
| `ev.material` | the material's constants |
| `ev.cases` | the load cases, with their weights |
| `ev.state(physics, case)` | the solved state, one float per degree of freedom |
| `ev.loads(physics, case)` | the load vector of that case |
| `ev.engine` | heavy helpers that run in Rust (element strain energy, element stress, ...); with one physics, this is the elasticity model |

`Partials` is what the author writes down by hand, and it is all local:

```python
@dataclass(frozen=True)
class Partials:
    d_density: np.ndarray | None                    # dg/d(rho) with the state held fixed, per element
    d_state: Mapping[tuple[str, str], np.ndarray]   # dg/d(state) for each (physics, case), per degree of freedom
```

**A response never solves anything.** The driver completes the gradient with the adjoint method: for each state
the response reads, it solves one extra linear system with `d_state` as the right-hand side,

$$K\,\lambda = \frac{\partial g}{\partial u}, \qquad
\frac{dg}{d\rho_e} = \frac{\partial g}{\partial \rho_e} - \lambda^{T}\,\frac{\partial K}{\partial \rho_e}\,u ,$$

and then pushes the result back through the filters to get dg/dx. (The engine's `sensitivity` call returns the
second term with its minus sign, so the driver adds it.) When `d_state` is a multiple of the load
vector (compliance is the classic case) the adjoint is that multiple of the state and the extra solve is skipped.

## A filter: a map of the field and its reverse

```python
class Filter(Protocol):
    name: str
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray: ...   # chain rule, back to x
    def self_test(self) -> Sequence[Row]: ...
```

Filters are applied in the order the problem file lists them. The density filter (cone weights, 2D), the
Helmholtz filter (a PDE filter, 3D), the projection and symmetry are built into the engine; a manufacturing rule such as "no overhang steeper than 45 degrees" is one more filter.

## Two small examples

```python
class Mass:
    name, unit, reads = "mass", "g", ()

    def value(self, ev):                            # inactive elements carry no mass
        per_element = ev.material.density_g_mm3 * ev.grid.voxel_volume_mm3
        return per_element * (ev.grid.active * ev.density).sum()

    def partials(self, ev):
        per_element = ev.material.density_g_mm3 * ev.grid.voxel_volume_mm3
        return Partials(d_density=per_element * ev.grid.active, d_state={})


class Compliance:                                   # low compliance = high stiffness
    name, unit, reads = "compliance", "N mm", ("elasticity",)

    def value(self, ev):
        return sum(c.weight * michell.dot(ev.loads("elasticity", c.name), ev.state("elasticity", c.name))
                   for c in ev.cases)

    def partials(self, ev):
        return Partials(d_density=None,
                        d_state={("elasticity", c.name): c.weight * ev.loads("elasticity", c.name)
                                 for c in ev.cases})
```

## Goal, limits and the optimiser

A problem has one goal and any number of limits. A limit is a response with a bound,
`Limit(response, at_most)`, in the response's own unit. The loop hands the optimiser plain numbers and arrays:

```python
class Optimiser(Protocol):
    def start(self, problem, x, limits, filters) -> None: ...            # once, before the first pass
    def update(self, x, goal, d_goal, limits, d_limits) -> np.ndarray: ...   # the next design variables
```

Two optimisers implement it: the classic optimality-criteria update (`OptimalityCriteria`: one limit, kept to
reproduce the published 88-line result; it measures the limit on the filtered field inside its bisection, so it
takes only the 2D density filter, on a 2D grid) and the method of moving asymptotes (`MovingAsymptotes`: any
number of limits, any filter, 2D or 3D). MMA's constants are absolute numbers, so `MovingAsymptotes` hands it the goal divided by its value at
the first pass and each limit as its relative excess over its bound, `(value - at_most) / |at_most|`; both
scales stay fixed for the run. The goal's value first reaches the optimiser in `update`, so its scale is taken
there, not in `start`.

## What every module must bring, and what it gets for free

- **Its rows.** `self_test()` returns the module's own proof: at least one case with a known answer (a formula
  from a textbook, with the source named). A row is built with `compared(name, value, expected, tolerance)`
  from `michell.row`: the measured number is the difference between the value and the expected one, relative
  to the expected value (absolute when it is 0), the limit is the tolerance stated in advance, and pass or
  fail is computed from the two. The verification table (`docs/verification.md`) uses the same `Row` class;
  its rows add a number in the table, a kind of check (closed form, gradient, reference code, published
  optimum), a note, and the case of a row that could not be run.
- **A gradient check, for free.** The framework compares every module's gradient with finite differences on a
  small random problem. No module can skip it.
- **Its status is computed, not claimed.** A module is listed as *supported* only while all its rows pass in CI;
  otherwise it is *experimental*, and a report that uses it says so next to the number.

## Rules for module code

1. No hidden state: the same `Evaluation` gives the same answer. No files, no network, no randomness.
2. Inputs are read-only; return new arrays.
3. Units are fixed: mm, N, MPa, g. A module states the unit of its value.
4. Work on whole arrays with NumPy. If a module needs a loop over elements, the loop belongs in the engine, and
   it moves there once a measurement shows it is the slow part. A product of two long vectors is
   `michell.dot(a, b)`, not `a @ b`: NumPy hands `@` to its BLAS library, and the OpenBLAS build that NumPy
   ships runs a product of more than 10000 entries on every processor core and leaves its threads busy
   between the calls of the loop.
5. Third-party packages can add modules: they are found by name through the Python entry-point group
   `michell.modules`, so `pip install` of another package is enough.

## Open questions (to settle with real code)

- Stress and other local limits produce one value per element. Whether the aggregation into one number is part
  of the response or a shared helper is decided with the stress module ([milestone 4](../roadmap.md)).
- Responses that read two physics at once (thermal stress) need the adjoint of a coupled solve (milestone 6).
- Whether a filter written in Python is fast enough at one million elements is measured in milestone 2.
