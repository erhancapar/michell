# The engine boundary: what is Rust, what is Python

Status: design note for stage 0. The engine is empty today; this note fixes where the line runs so that the
first engine code (stage 1) lands on the right side of it.

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
| `Elasticity` | `solve(rho, loads) -> u`, `adjoint(rho, rhs) -> lambda`, `sensitivity(rho, u, lambda) -> per element` | 1 (2D), 2 (3D) |
| `DensityFilter`, `Projection` | `forward(x)`, `backward(x, grad)` | 1, 2 |
| `Oc`, `Mma` | `update(x, goal, d_goal, limits, d_limits) -> x` | 1 |
| `HeatConduction` | same shape as `Elasticity` | 4 |

Array rules:

1. **Numbering.** Element `(i, j, k)` has index `i + nx * (j + ny * k)`: x runs fastest. Nodes use the same rule
   on the grid of `(nx+1, ny+1, nz+1)` points. A node's degrees of freedom are stored together: `3n, 3n+1, 3n+2`
   for x, y, z displacement (in 2D: `2n, 2n+1`).
2. **Inputs are borrowed, not copied.** The engine reads a contiguous float64 array in place and never keeps a
   pointer to it after the call returns. Anything else (wrong type, wrong length, not contiguous) is refused
   with a clear error, never silently converted.
3. **Outputs are new arrays** owned by Python.
4. Elements outside the design space are simply inactive: they take part in no equation.

## Threads

The engine releases Python's global lock for the length of every call, so other Python threads keep running.
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
Examples: `solver_not_converged` with the residual and pass count; `singular_system` when a part is not held.
A Rust panic is a bug in michell: it is caught at the boundary and reported as `internal_error`, and it never
brings down the Python process.

## Moving code across the line

A module starts in Python. It moves into Rust only when a measurement on a real problem shows it takes a
noticeable share of the time per pass; the measurement is kept with the change. The reverse also holds: nothing
goes into the engine for convenience.
