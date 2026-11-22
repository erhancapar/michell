# The engine boundary: what is Rust, what is Python

Status: design note, kept in step with the code. The 2D analysis, the density filter, the OC update, the
MMA update and `dot` exist. The 3D analysis exists (`Grid3d`, `Elasticity3d`) and runs on as many threads as
it is given: its solver is conjugate gradients preconditioned by one multigrid V-cycle (Galerkin coarse
matrices, damped Jacobi smoothing, a direct solve on the coarsest level), and it checks when it is built that
the part is held (see "Errors"). The Helmholtz filter (`HelmholtzFilter`) and the density filter on the 3D
grid exist, on threads like the 3D analysis; the projection (`Projection`) and the member-size limit
(`member_size`) work on a field of either grid. The voxeliser (`voxel_occupancy`), which turns closed
triangle meshes into the occupancy of every voxel of a grid, exists, on threads like the 3D analysis; the
meshes and the voxel model built from them are Python (`michell.mesh`, `michell.regions`; see
[geometry.md](geometry.md)). The iso-surface (`iso_surface`), which turns a field sampled on a lattice into a
closed triangle surface by marching tetrahedra, exists, on threads like the voxeliser; refining the field,
smoothing the surface, the measures of a mesh and 3MF files are Python (`michell.surface`, `michell.mesh`).
The rest of the table below is planned.

## The rule

**Rust owns every loop over elements or nodes. Python owns everything a user or an agent would want to read
or change.**

| Rust engine (`crates/michell`) | Python (`python/michell`) |
|---|---|
| the grid and its numbering | reading the problem file and checking it |
| element stiffness and conductivity | reading STEP and STL files, building the voxel model from the regions |
| the linear solver (direct in 2D, multigrid in 3D) | the optimisation loop itself |
| the built-in filters (density, projection, symmetry) | response modules (goals and limits) |
| the optimisers' update step (OC, MMA) | manufacturing rules written as filters |
| heavy helpers (strain energy, stress per element) | the finished mesh, the report, the command line |
| the voxeliser (the loop over rays and triangles) | refining a field, smoothing a surface (three sums per step) |
| the iso-surface (the loop over lattice cells) | the bodies and genus of a mesh, STL and 3MF files |

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
objects. Planned objects, with the [milestone](../roadmap.md) that brings them:

| Object | Calls | Milestone |
|---|---|---|
| `Grid` | built from shape, voxel size and an active-element mask | 1 |
| `Elasticity` | `solve(rho, loads, held_values=None) -> u`, `adjoint(rho, rhs) -> lambda` (in 3D both also take a first guess `start=None`, and `last_solve_report()` gives the steps of the last solve), `sensitivity(rho, u, lambda)` (the term $-\lambda^T (\partial K / \partial \rho_e)\, u$ per element, sign included), `element_strain_energy`, `element_stress` | 1 (2D), 2 (3D) |
| `DensityFilter`, `HelmholtzFilter`, `Projection` | `forward(x)`, `backward(x, grad)` | 1, 2 |
| `member_size` (a function) | `member_size(grid, filtered, projection, thresholds, c) -> (g, dg/dfiltered)`: the member-size limit on the filtered field, one threshold per element | 3 |
| `Oc` | `update(x, d_goal, d_limit, limit_weights, limit_bound, filter=None) -> x`: the classic update for one limit, measured on the filtered field | 1 |
| `Mma` | built from the bounds of the variables and the number of limits; `update(x, goal, d_goal, limits, d_limits) -> x`, with the limits written f_i(x) <= 0 and `d_limits` one row per limit; it remembers the two previous designs | 1 |
| `dot` (a function) | `dot(a, b) -> float`: the dot product of two vectors, summed in index order, for the products of the optimisation loop (see "Threads") | 1 |
| `voxel_occupancy` (a function) | `voxel_occupancy(counts, voxel_mm, origin_mm, adding, subtracting=[], rays_per_side=3, threads=1) -> occupancy`: the share of every voxel inside the union of the `adding` bodies minus the union of the `subtracting` ones, each body a pair of arrays (vertices (n, 3) in mm, triangles (m, 3) of whole-number indices, both float64) | 3 |
| `iso_surface` (a function) | `iso_surface(counts, spacing_mm, origin_mm, values, level, threads=1) -> (vertices, triangles)`: the closed surface where `values` (one sample per point of a lattice of `counts` points, x fastest) equal `level`, the side of larger values inside; vertices (n, 3) in mm and triangles (m, 3) of whole-number indices, both float64 | 3 |
| `HeatConduction` | same shape as `Elasticity` | 4 |

Array rules:

1. **Numbering.** Element `(i, j, k)` has index `i + nx * (j + ny * k)`: x runs fastest. Nodes use the same rule
   on the grid of `(nx+1, ny+1, nz+1)` points. A node's degrees of freedom are stored together: `3n, 3n+1, 3n+2`
   for x, y, z displacement (in 2D: `2n, 2n+1`).
2. **Inputs are never converted.** The engine accepts a contiguous float64 array and refuses anything else
   (wrong type, wrong length, not contiguous) with a clear error; a 2-D input (the limits' gradients of `Mma`,
   the bodies of `voxel_occupancy`) must also be row-major (C order). A short call reads the array in place. A long call (a solve, the
   Helmholtz filter, the density filter on a 3D grid) copies its inputs once before it releases Python's
   lock, so that another Python thread cannot change them halfway; the copy costs far less than the work.
   The engine never keeps a pointer to an input after the call returns.
3. **Outputs are new arrays** owned by Python.
4. Elements outside the design space are simply inactive: they take part in no equation. A degree of freedom
   that touches no active element is held at zero, and a load on it is refused.
5. Masks (which elements are active, which degrees of freedom are held) cross as float64 arrays of 0 and 1.
   The Python package accepts boolean arrays and converts them.

## Threads

A long call (a solve, an optimiser update, the Helmholtz filter, which solves a linear system, the density
filter on a 3D grid, the voxeliser, the iso-surface) copies its inputs and then releases Python's global lock, so other Python threads keep
running; a short call (the density filter on a 2D grid, the projection, the member-size limit, a per-element
quantity) keeps the lock and reads in place.
The number of threads is an explicit input (problem file or argument), never a guess from the environment, and
it is written into every result.

The same holds for the libraries michell calls. NumPy hands a product of two vectors (`a @ b`) to its BLAS
library; the OpenBLAS build that NumPy ships runs a product of more than 10000 entries on every processor
core and leaves its threads waiting at full load between calls. The optimisation loop therefore takes its
products from the engine's `dot`, on one thread; the measurement that moved them is in the header of
`crates/michell/src/dot.rs`.

The 3D analysis takes the thread count as an argument of the object, `Elasticity3d(..., threads=1)`: at least
1, and a count above the number of processor cores is allowed (the threads then take turns). The engine keeps
one pool of worker threads per distinct count (from the `rayon` crate), started the first time that count is
asked for and shared by every object that asks for the same count. The Helmholtz filter takes its thread
count the same way, `HelmholtzFilter(grid, radius, threads=1)`, as does the density filter on a 3D grid,
`DensityFilter(grid, radius, threads=1, *, outside="closed", inactive="closed")` (the last two say what lies
beyond each face and in the inactive elements; they are checked when the filter is built, and the default
gives the numbers of the filter without them, bit for bit), and so does a 3D `Problem`, which hands it to its analysis; the
thread count is written into the result of an optimisation (`Result.threads`). The voxeliser takes it as
an argument too, `voxel_occupancy(..., threads=1)`, and `voxel_model(..., threads=1)` hands it on; so does
the iso-surface, `iso_surface(..., threads=1)`. The 2D analysis, the density
filter on a 2D grid (which refuses a thread count other than 1), the projection, the member-size limit and
the OC update run on one thread. The MMA update takes its thread count the same way, `Mma(x_min, x_max,
n_limits, threads=1)`; `MovingAsymptotes` hands it the problem's.

## Same input, same output

The promise: **on the same machine, with the same thread count, the same input gives bit-for-bit the same
output.** To keep it, parallel sums are split into fixed blocks that are added in a fixed order, so the result
does not depend on which thread finishes first. Across different machines or thread counts the results agree
within a tolerance that the verification table states; they are not promised to be bit-identical, because
processors differ in how they round.

The 3D analysis is built so that its sums do not depend on which thread runs which block: every sum runs over
blocks of a fixed length that does not depend on the thread count, each block summed in index order and the
block sums added in order. Every other step writes separate pieces of its output (the entries of one node, the
value of one element), each computed with the same arithmetic whichever thread runs it. Verification row 15
compares the numbers of runs on several thread counts, the Helmholtz filter's and the 3D density filter's
included. The voxeliser computes every voxel column on its own from the inputs, so its numbers are the same on
every thread count (verification row 30). The iso-surface computes every lattice point's vertices and every
cell's triangles from the samples alone and joins them in the order of the points, so its vertices, its
triangles and their order are the same on every thread count (verification row 34).

The engine keeps no solution from one call to the next: a call's output depends only on its inputs. A warm
start is therefore an input. The 3D `solve` takes a first guess `start=`, and the optimisation loop hands
each solve the state of the same load case from the previous pass, in the open (`evaluate(..., start=)`);
the number of conjugate-gradient steps each pass took is written into its record (`Pass.solver_steps`).

## Errors

The engine returns errors as values. The bindings turn them into `michell.EngineError` with three fields: a
stable `code` (for programs and agents), a `message` (for people), and `details` (the numbers involved).
Examples: `invalid_input` for a wrong array or setting; `singular_system` when a part is not held (found when
the analysis object is built, because it does not depend on the density: in 2D by factoring the stiffness
matrix with every active element at the same modulus; in 3D by checking, for every body of active elements
joined through shared faces, that its held degrees of freedom stop all six rigid motions, with the size of the
body and one of its elements in the details; the 3D solve itself can also raise it, if the direct solve on the
coarsest multigrid level meets a pivot that is not positive for the density of that call); `solver_not_converged` with the residual and step count (the
MMA subproblem; the iterative solver in 3D). An `EngineError` that leaves `michell.optimise` carries the passes
completed before it as `partial_result`, a `Result` like the one a run that stopped there would return, with
`stop_reason` `"error"`. It is `None` when the arguments were refused before the first pass, a refusal by the
optimiser's `start` included (`start` is called among those checks, so nothing has been solved), and when the
filters cannot be applied to the `x` of the partial result.
The checks written in Python raise `EngineError` too, with the code `invalid_input` and the same three fields;
each is also the built-in exception it replaces (`ValueError`, `TypeError` or `KeyError`, from
`michell.errors`). Among them: a load case that loads a held degree of freedom (refused when the `Problem` is
built, since the load would be lost), a limit whose bound is not a finite number, and a bound that no design
can meet (a mass or compliance bound below 0, or a mass bound below the mass that a final `KeepSolid` holds
solid) or that only a degenerate case meets (a mass bound of 0, met only by the empty part, and a compliance
bound of 0, met only by a part without load), refused by `optimise` before the first pass.
The 3D check is stricter than the mathematics in one rare case: elements that meet only along an edge or at a
corner count as separate bodies, so a body restrained only through such a hinge to another, held body is
refused, although the hinge stops some of its motions.
A Rust panic is a bug in michell: it is caught at the boundary and reported as `internal_error`, and it never
brings down the Python process.

## Moving code across the line

A module starts in Python. It moves into Rust only when a measurement on a real problem shows it takes a
noticeable share of the time per pass; the measurement is kept with the change. The reverse also holds: nothing
goes into the engine for convenience.
