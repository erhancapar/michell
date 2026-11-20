# The engine boundary: what is Rust, what is Python

Status: design note, kept in step with the code. The 2D analysis, the density filter, the OC update and the
MMA update exist; the rest of the table below is planned.

## The rule

**Rust owns every loop over elements or nodes. Python owns everything a user or an agent would want to read
or change.**

| Rust engine (`crates/michell`) | Python (`python/michell`) |
|---|---|
| the grid and its numbering | reading the problem file and checking it |
| element stiffness and conductivity | reading STEP and STL, turning them into voxels |
| the linear solver (direct in 2D, multigrid in 3D) | the optimisation loop itself |
| the built-in filters (density, projection, symmetry) | response modules (goals and limits) |
| the optimisers' update step (OC, MMA) | manufacturing rules written as filters |
| heavy helpers (strain energy, stress per element) | the finished mesh, the report, the command line |

The optimisation loop is deliberately in Python: it is about thirty lines, each heavy step is one call into
the engine, and anyone can read it or write their own. The cost of the Python layer is a handful of calls per
pass, against a solve that takes most of the time.

## Two crates

- `crates/michell`: the engine as a plain Rust library. It knows nothing about Python, so a Rust or (later) C
  program can use it directly.
- `crates/michell-py`: the bindings. They convert arguments and errors and contain no engineering logic.
  Python sees them as `michell._engine`; user code should import `michell`, not `_engine`.

## What crosses the boundary

Only three things cross: small settings (numbers, strings), **NumPy arrays of float64**, and handles to engine
objects. Planned objects, with the stage that brings them:

| Object | Calls | Stage |
|---|---|---|
| `Grid` | built from shape, voxel size and an active-element mask | 1 |
| `Elasticity` | `solve(rho, loads, held_values=None) -> u`, `adjoint(rho, rhs) -> lambda`, `sensitivity(rho, u, lambda)` (the term $-\lambda^T (\partial K / \partial \rho_e)\, u$ per element, sign included), `element_strain_energy`, `element_stress` | 1 (2D), 2 (3D) |
| `DensityFilter`, `Projection` | `forward(x)`, `backward(x, grad)` | 1, 2 |
| `Oc` | `update(x, d_goal, d_limit, limit_weights, limit_bound, filter=None) -> x`: the classic update for one limit, measured on the filtered field | 1 |
| `Mma` | built from the bounds of the variables and the number of limits; `update(x, goal, d_goal, limits, d_limits) -> x`, with the limits written f_i(x) <= 0 and `d_limits` one row per limit; it remembers the two previous designs | 1 |
| `HeatConduction` | same shape as `Elasticity` | 4 |

Array rules:

1. **Numbering.** Element `(i, j, k)` has index `i + nx * (j + ny * k)`: x runs fastest. Nodes use the same rule
   on the grid of `(nx+1, ny+1, nz+1)` points. A node's degrees of freedom are stored together: `3n, 3n+1, 3n+2`
   for x, y, z displacement (in 2D: `2n, 2n+1`).
2. **Inputs are never converted.** The engine accepts a contiguous float64 array and refuses anything else
   (wrong type, wrong length, not contiguous) with a clear error; a 2-D input (the limits' gradients of `Mma`)
   must also be row-major (C order). A short call reads the array in place. A long
   call (a solve) copies its inputs once before it releases Python's lock, so that another Python thread cannot
   change them halfway; the copy costs far less than the solve. The engine never keeps a pointer to an input
   after the call returns.
3. **Outputs are new arrays** owned by Python.
4. Elements outside the design space are simply inactive: they take part in no equation. A degree of freedom
   that touches no active element is held at zero, and a load on it is refused.
5. Masks (which elements are active, which degrees of freedom are held) cross as float64 arrays of 0 and 1.
   The Python package accepts boolean arrays and converts them.

## Threads

A long call (a solve, an optimiser update) copies its inputs and then releases Python's global lock, so other
Python threads keep running; a short call (a filter, a per-element quantity) keeps the lock and reads in place.
The number of threads is an explicit input (problem file or argument), never a guess from the environment, and
it is written into every result.

## Same input, same output

The promise: **on the same machine, with the same thread count, the same input gives bit-for-bit the same
output.** To keep it, parallel sums are split into fixed blocks that are added in a fixed order, so the result
does not depend on which thread finishes first. Across different machines or thread counts the results agree
within a tolerance that the verification table states; they are not promised to be bit-identical, because
processors differ in how they round.

## Errors

The engine returns errors as values. The bindings turn them into `michell.EngineError` with three fields: a
stable `code` (for programs and agents), a `message` (for people), and `details` (the numbers involved).
Examples: `invalid_input` for a wrong array or setting; `singular_system` when a part is not held (found when
the `Elasticity` object is built, because it does not depend on the density); `solver_not_converged` with the
residual and step count (the MMA subproblem; the iterative solver in 3D).
A Rust panic is a bug in michell: it is caught at the boundary and reported as `internal_error`, and it never
brings down the Python process.

## Moving code across the line

A module starts in Python. It moves into Rust only when a measurement on a real problem shows it takes a
noticeable share of the time per pass; the measurement is kept with the change. The reverse also holds: nothing
goes into the engine for convenience.
