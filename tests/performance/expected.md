# Performance

This page gives the run time and the peak memory of `michell.recipes.stiff_part`, the library's whole procedure
from a uniform start to a design, on one 3D problem at 2 grid sizes. The problem: a steel block of 40 by 20 by
20 mm (Young's modulus 210000 MPa, Poisson ratio 0.3, 7.85 g/cm³), every degree of freedom held on its face
x = 0, and a force of 1000 N down spread over the nodes of its face x = 40 mm as a uniform traction.
`stiff_part` makes the stiffest part with at most 30 % of the block's mass and members at least 3 mm thick, with
its default settings: 100 passes of the robust formulation, then 40 with the member-size limit, 140 in all. For
a size m the grid has 2m by m by m cube elements.

One pass is one update of the design: the filters, the solve of the load case on the design whose compliance is
the goal (conjugate gradients preconditioned by a multigrid cycle), the gradients back through the filters and
the update by the method of moving asymptotes. A pass is timed from the end of the one before to the end of its
update, so the passes add up to nearly the whole run: the first pass also holds the setting up of the density
filter, and a pass that begins a new call of `optimise` (every 10 passes and at each change of sharpness in the
robust step, and at the switch to the member-size limit) also holds the evaluation of the design the call before
returned and the new bound on the mass, and starts its solve from zero instead of from the state of the pass
before. Each size runs in a Python process of its own, so its peak memory, the largest resident memory of that
process, is that size's own; it includes the interpreter and NumPy.

## Machine

- Processor: Example Processor 3000
- Logical cores: 8
- Memory: 33.6 GB
- Operating system: Linux 6.1.0
- Python 3.12.1, michell 0.0.1 (engine 0.0.1)
- Threads of the solves, the filter and the updates: 4

## Results

| Grid | Elements | Voxel (mm) | Filter radius (elements) | Passes | Seconds per pass | Slowest ten (s) | Peak memory (GB) | Solver steps per solve | Largest solve (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 x 20 x 20 | 16,000 | 1 | 3 | 140 | 0.312 | 0.619 | 0.0812 | 28.5 | 141 |
| 80 x 40 x 40 | 128,000 | 0.5 | 6 | 140 | 2.72 | 5.56 | 0.654 | 35 | 212 |

Seconds per pass is the mean over every pass of the run, and the slowest ten the mean of its ten slowest passes.
Solver steps per solve are the conjugate-gradient steps of one solve, as the mean over every solve of the run;
the largest solve is the one that took the most. The memory is in units of 10⁹ bytes.

| Grid | Robust step (s per pass) | Member-size step (s per pass) | Total (s) | Compliance (N mm) | Limits met | Stop reason |
|---|---:|---:|---:|---:|---|---|
| 40 x 20 x 20 | 0.3 | 0.343 | 44 | 61.2 | no | passes |
| 80 x 40 x 40 | 2.5 | 3.14 | 382 | 49.9 | yes | passes |

The compliance is that of the returned design, and its limits are its mass and its member-size value; the total
is the time of the whole call of `stiff_part`.

At 40 x 20 x 20 `stiff_part` warned: `GridTooCoarse`.

## Repeating the measurement

From the repository, with the package installed:

    python -m michell.performance run --sizes 20 40 --threads 4 --out docs/performance.json
    python -m michell.performance render docs/performance.json --out docs/performance.md

The first command writes the measurements, with the time and the solver steps of every pass, to the JSON file;
the second writes this page from it. Run on another machine, they give that machine's numbers.

## What the numbers do not say

- They come from one machine, the one above, on 4 threads; another processor, another memory or another thread
  count gives other numbers.
- The member size is fixed in millimetres (3 mm), so the filter radius counted in elements grows with the grid,
  from 3 to 6 elements here, and the density filter's share of a pass grows with it: the time of a pass does not
  follow the number of elements alone.
- Another problem gives other numbers: the steps of a solve, and with them the time of a pass, depend on the
  shape, the supports, the loads and the design as it forms.
