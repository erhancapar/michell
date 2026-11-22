# Modules: how goals, limits and shape rules plug in

Status: design note, kept in step with the code. The interface, the `Compliance` and `Mass` responses, the
optimisation loop on a 2D or a 3D grid (a `Problem` picks the analysis from the grid's type), the density
filter (2D and 3D), the Helmholtz filter (3D), the projection (both), several designs from one set of
variables with the robust set, the minimum member size as a limit, and both optimisers below exist.

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

Filters are applied in the order the problem file lists them. The density filter (cone weights, 2D and 3D),
the Helmholtz filter (a PDE filter, 3D), the projection and symmetry are built into the engine; a manufacturing rule such as "no overhang steeper than 45 degrees" is one more filter.

What lies beyond a face of the grid is part of the model, and the density filter states it:
`DensityFilter(grid, radius, threads=1, *, outside="closed", inactive="closed")`, with `outside` one word for
every face or a mapping by face (`"x-"`, `"x+"`, `"y-"`, `"y+"`, and in 3D `"z-"`, `"z+"`; faces left out are
closed). A closed face (the default) averages over the elements inside the grid only, so it acts like a mirror:
right for a symmetry plane, and for a face that continues into a wall or a support. An empty face keeps the
weights that fall beyond it in each average with a density of 0, as if void elements lay there: right for a
free face of the part. `inactive` makes the same choice for the inactive elements. A uniform field stays
uniform only next to closed faces and closed inactive elements. The padding is seen by the filter alone: the
analysis, the mass and the limits are taken over the grid, so an element at a corner of two empty faces can
never become solid (it reaches about a third of the full density), and a support or a load there needs
solid elements around it. This is the boundary padding of the filter discussed by Clausen and
Andreassen ("On filter boundary conditions in topology optimization", Struct. Multidisc. Optim. 56, 2017),
not their domain extension, which extends the analysis too. The Helmholtz filter's zero-flux boundary acts
like a closed face everywhere.

Supports and load points need solid material around them when the filter's faces are empty (Clausen and
Andreassen 2017, section 3.3): `KeepSolid(mask, grid=None)` is the filter that provides it, setting the marked
elements to 1 at the end of a design's chain, after the projection (`designs={"nominal": [projection,
KeepSolid(pad)]}`), with a zero gradient for them.

## Several designs from one set of variables

One x can feed several density fields. `optimise(..., designs=...)` takes a mapping from a name to a list
of filters; each named design is the output of the problem's `filters` passed through its own list, and it
is a density field of its own, with its own solves:

```
x  ->  filters  ->  rho  -+->  design filters of "eroded"   ->  rho_eroded   ->  solve  ->  goal
                          +->  design filters of "nominal"  ->  rho_nominal
                          +->  design filters of "dilated"  ->  rho_dilated  ->  limit (mass: no solve)
```

- **`on(name, response)`** says which design a goal or a limit reads:
  `goal=on("eroded", Compliance())`, `Limit(on("dilated", Mass()), at_most=...)`. The response itself does
  not change: it sees the one `Evaluation` of its design and nothing else. When `designs` is given, every
  goal and limit must name its design (an error otherwise); when it is not, nothing may name one, and the
  loop works as before, the output of `filters` being the one design.
- **Only what is read is computed.** A design no goal or limit reads is not evaluated in the loop, and a
  design is solved only for the physics its responses read (a mass limit reads none). The final field of
  every design is in `Result.designs`.
- **Gradients** go back through the design's own filters, then through `filters`, so every design pulls on
  the same x. Warm starts are kept per design (each design's solves start from that design's states of the
  previous pass), and `Pass.design_solver_steps` holds the solver steps per design solved.
- **`robust_designs(beta, spread)`** is the set this was made for, the robust formulation of Wang, Lazarov
  and Sigmund ("On projection methods, convergence and robust formulations in topology optimization",
  Struct. Multidisc. Optim. 43, 2011): `eroded`, `nominal` and `dilated`, each one `Projection` of
  sharpness `beta` at the threshold 0.5 + spread, 0.5 and 0.5 - spread, standing for a part made uniformly
  too thin, as drawn and too thick. The robust goal is the worst of the three designs. For compliance the
  worst is always the eroded one: its density is nowhere higher than the nominal one's (the projection
  falls as the threshold rises), and stiffness only grows with density, so compliance only falls with it.
  The goal therefore reads `eroded` alone, and one solve per load case replaces three. The mass limit sits
  on `dilated`, the heaviest; its bound is rescaled between continuation steps so that `nominal` meets the
  asked mass (the rule as stated by Kumar and Langelaar, Mech. Mach. Theory, 2022). Continuation (raising `beta`) and that rescaling are a loop over
  `beta` in the user's script that calls `optimise` again from the last x; the library holds no schedule.

## The minimum member size: a limit on the filtered design

`MemberSize` is the geometric constraint on the solid phase of Zhou, Lazarov, Wang and Sigmund ("Minimum
length scale in topology optimization by geometric constraints", Comput. Methods Appl. Mech. Engrg. 293,
2015, eqs. 8, 10 and 12):

$$g = \frac{1}{n} \sum_e \bar\rho_e\, e^{-c\,|\nabla \tilde\rho|_e^2} \left[\min(\tilde\rho_e - \eta_e,\ 0)\right]^2 \le \varepsilon .$$

Inside every solid member the filtered field $\tilde\rho$ has a ridge, where its gradient vanishes; the
exponential picks out those ridges on the solid ($\bar\rho$ is the projected field), and the limit asks
the filtered field to reach $\eta_e$ there. `member_threshold(filter, size)` gives $\eta_e$ for a wanted
member size: the filtered value at the middle of an infinite plate (3D) or strip (2D) of that thickness,
which a thinner member cannot reach. The gradient is taken per element side between neighbouring active
elements, and $c = (R / h)^4$ by default, the paper's choice for a filter radius of $R / h$ elements.

It is an ordinary response that reads no physics. It must see the filtered field, so it is read on a
design whose own filter chain is empty, beside the design the goal reads; that design is never solved:

```python
designs = {"filtered": [], "nominal": [projection]}
goal = on("nominal", Compliance())
limits = [Limit(on("nominal", Mass()), at_most=m),
          Limit(on("filtered", MemberSize(density_filter, projection, member_threshold(density_filter, size))),
                at_most=eps, scale=value_at_switch)]
```

No change to the loop or the interface was needed. Two things are left to the user's script, in the open,
as the paper prescribes: the limit is switched on only after a topology has formed without it, at a high,
fixed sharpness of the projection (one more call of `optimise`); and its bound $\varepsilon$ is taken from
the paper's figure 8 for the filter radius in elements (about 5.5e-8 at 8 elements). Such a bound lies far
below the values the limit takes when it is switched on (of the order of 1e-4), so the limit's `scale` is
set to its value at the switch (see "Goal, limits and the optimiser"). `examples/mbb_beam_member_size.py`
writes the whole procedure out, and verification row 27 runs it.

- **Attainable on any part.** The paper's box is all design space; on a part the held solid and the regions
  thinner than the asked size could make the bound impossible. The threshold is therefore one per element:
  capped by `reach`, the filtered field of the fully solid part, and 0 on the elements given as
  `keep_solid` (a `KeepSolid` or a boolean mask), which are not judged. The fully solid part gives g = 0
  exactly, whatever its shape (rows 25 and 26). In a region thinner than the asked size the limit then asks
  that the solid fill the part locally, which is stricter than a member of the asked width.
- **Supported with the density filter**, the cone filter of the paper (rows 25 to 27).
- **Experimental with the Helmholtz filter** (`MemberSize.experimental` is then true): the paper derives
  $c$ and its values of $\varepsilon$ for the cone filter only, and no row checks the pair.
- **Free faces should be empty** (`outside` of the density filter, above). A closed face acts like a mirror,
  so a member lying along it looks to the filter like one twice as wide, and the limit holds it to about half
  the size: a strip 0.75 times the size thick gives g = 1e-16 against a closed face, and more than 10000
  eps against an empty face, as in the interior (row 26). At a symmetry plane the mirror is right. The
  limit's gradient is taken inside the grid, one-sided at a face, whichever the face's choice.
- **Scope:** the limit is switched on after a layout has formed and reshapes its members; where the asked
  size cannot be reached without changing the layout, it does not get there (trials on the MBB beam of row
  27 at a mass fraction of 0.3 with sizes of 7 and 8 elements missed the bound by a factor of 2 to 3, at a
  compliance several times higher), so a run must check that the bound is met: `Result.limits_met` says so,
  and `optimise` warns (`LimitNotMet`) when it is not.
- Only the solid phase is built; the paper's constraint on holes (the void phase) is not.

## Two small examples

```python
class Mass:
    name, unit, reads = "mass", "g", ()

    def value(self, ev):                            # inactive elements carry no mass
        per_element = ev.material.mass_g(ev.grid.voxel_volume_mm3)
        return per_element * (ev.grid.active * ev.density).sum()

    def partials(self, ev):
        per_element = ev.material.mass_g(ev.grid.voxel_volume_mm3)
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
the first pass and each limit as `(value - at_most) / scale`, with the limit's `scale`: by default
`|at_most|`, the relative excess over the bound; a limit whose bound is far below the values it takes before
it is met (the member-size limit) is given a larger scale, `Limit(..., scale=...)`. Both scales stay fixed
for the run. The goal's value first reaches the optimiser in `update`, so its scale is taken
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
