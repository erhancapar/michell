# Performance

This page gives the run time and the peak memory of `michell.recipes.stiff_part`, the library's whole procedure
from a uniform start to a design, on one 3D problem at 5 grid sizes. The problem: a steel block of 40 by 20 by
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

- Processor: AMD Ryzen 5 7600 6-Core Processor
- Logical cores: 12
- Memory: 16.8 GB
- Operating system: Arch Linux (Linux 7.3.5-arch1-1)
- Python 3.12.14, michell 0.0.1 (engine 0.0.1)
- Threads of the solves, the filter and the updates: 12

## Results

| Grid | Elements | Voxel (mm) | Filter radius (elements) | Passes | Seconds per pass | Slowest ten (s) | Peak memory (GB) | Solver steps per solve | Largest solve (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 x 36 x 36 | 93,312 | 0.556 | 5.4 | 140 | 1.18 | 2.22 | 0.265 | 35.1 | 57 |
| 92 x 46 x 46 | 194,672 | 0.435 | 6.9 | 140 | 2.45 | 4.63 | 0.514 | 33.6 | 56 |
| 116 x 58 x 58 | 390,224 | 0.345 | 8.7 | 140 | 6.67 | 12.6 | 0.964 | 44.7 | 79 |
| 144 x 72 x 72 | 746,496 | 0.278 | 10.8 | 140 | 14.1 | 26.8 | 1.54 | 45.4 | 89 |
| 180 x 90 x 90 | 1,458,000 | 0.222 | 13.5 | 140 | 31.6 | 59.1 | 2.48 | 42.1 | 85 |

Seconds per pass is the mean over every pass of the run, and the slowest ten the mean of its ten slowest passes.
Solver steps per solve are the conjugate-gradient steps of one solve, as the mean over every solve of the run;
the largest solve is the one that took the most. The memory is in units of 10⁹ bytes.

| Grid | Robust step (s per pass) | Member-size step (s per pass) | Total (s) | Compliance (N mm) | Limits met | Stop reason |
|---|---:|---:|---:|---:|---|---|
| 72 x 36 x 36 | 1.1 | 1.37 | 168 | 23.9 | yes | passes |
| 92 x 46 x 46 | 2.28 | 2.86 | 345 | 23.2 | yes | passes |
| 116 x 58 x 58 | 6.14 | 8 | 941 | 23.2 | yes | passes |
| 144 x 72 x 72 | 13.1 | 16.7 | 1990 | 23.2 | yes | passes |
| 180 x 90 x 90 | 29.9 | 36.1 | 4490 | 23.2 | yes | passes |

The compliance is that of the returned design, and its limits are its mass and its member-size value; the total
is the time of the whole call of `stiff_part`.

## Repeating the measurement

From the repository, with the package installed:

    python -m michell.performance run --sizes 36 46 58 72 90 --threads 12 --out docs/performance.json
    python -m michell.performance render docs/performance.json --out docs/performance.md

The first command writes the measurements, with the time and the solver steps of every pass, to the JSON file;
the second writes this page from it. Run on another machine, they give that machine's numbers.

## What the numbers do not say

- They come from one machine, the one above, on 12 threads; another processor, another memory or another thread
  count gives other numbers.
- The member size is fixed in millimetres (3 mm), so the filter radius counted in elements grows with the grid,
  from 5.4 to 13.5 elements here, and the density filter's share of a pass grows with it: the time of a pass
  does not follow the number of elements alone.
- Another problem gives other numbers: the steps of a solve, and with them the time of a pass, depend on the
  shape, the supports, the loads and the design as it forms.
