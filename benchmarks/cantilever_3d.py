"""Run time, solver steps and memory of the 3D cantilever of `examples/cantilever_3d.py`.

The same problem as the example (a block clamped on its face x = 0, a line load of 1000 N down along the lower
edge of its free end, least compliance with at most 15 % of the block's mass, the Helmholtz filter of radius
2.5 element sides, the method of moving asymptotes), run for a given number of passes on a given number of
threads. It prints one JSON line:

- ``setup_seconds``: building the problem (the analysis, its multigrid levels) and the filter;
- ``seconds_per_pass``: the wall-clock time of each pass (its solves, gradients, filter and update);
- ``solver_steps_per_pass``: the conjugate-gradient steps of each pass;
- ``peak_memory_mb``: the largest resident memory of the process so far, in MB (``null`` where the operating
  system does not report it).

Run from the repository:

    python benchmarks/cantilever_3d.py --nx 64 --ny 32 --nz 32 --threads 4 --passes 10
"""

import argparse
import json
import sys
import time

import numpy as np

import michell


class TimedPasses:
    """An optimiser that hands every call to ``inner`` and notes the time at the end of each update, which
    is the end of a pass."""

    def __init__(self, inner):
        self.inner = inner
        self.ends = []

    def start(self, problem, x, limits, filters):
        self.inner.start(problem, x, limits, filters)

    def update(self, *arguments):
        x = self.inner.update(*arguments)
        self.ends.append(time.perf_counter())
        return x


def peak_memory_mb():
    """The peak resident memory of this process in MB, or None where the ``resource`` module is missing."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux, bytes on macOS
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


parser = argparse.ArgumentParser(description="Measure the 3D cantilever: seconds and solver steps per pass, memory.")
parser.add_argument("--nx", type=int, default=64)
parser.add_argument("--ny", type=int, default=32)
parser.add_argument("--nz", type=int, default=32)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--passes", type=int, default=10)
args = parser.parse_args()
nx, ny, nz = args.nx, args.ny, args.nz

started = time.perf_counter()
grid = michell.Grid3d(nx, ny, nz, 1.0)
held = np.zeros(grid.n_dofs, dtype=bool)
for k in range(nz + 1):
    for j in range(ny + 1):
        node = grid.node(0, j, k)
        held[3 * node : 3 * node + 3] = True
loads = np.zeros(grid.n_dofs)
for j in range(ny + 1):
    loads[3 * grid.node(nx, j, 0) + 2] = -1000.0 / ny * (0.5 if j in (0, ny) else 1.0)
steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_mm3=7.85e-3)
problem = michell.Problem(grid, held, steel, [michell.LoadCase("tip", loads)], threads=args.threads)
mass_limit = 0.15 * steel.density_g_mm3 * grid.voxel_volume_mm3 * grid.n_elements
pde_filter = michell.HelmholtzFilter(grid, 2.5, threads=args.threads)
optimiser = TimedPasses(michell.MovingAsymptotes())
setup_seconds = time.perf_counter() - started

started = time.perf_counter()
result = michell.optimise(
    problem,
    goal=michell.Compliance(),
    limits=[michell.Limit(michell.Mass(), at_most=mass_limit)],
    filters=[pde_filter],
    optimiser=optimiser,
    x=np.full(grid.n_elements, 0.15),
    passes=args.passes,
)
seconds = np.diff([started] + optimiser.ends)
print(
    json.dumps(
        {
            "shape": [nx, ny, nz],
            "elements": grid.n_elements,
            "threads": result.threads,
            "passes": len(result.passes),
            "setup_seconds": round(setup_seconds, 4),
            "seconds_per_pass": [round(float(s), 4) for s in seconds],
            "solver_steps_per_pass": [p.solver_steps for p in result.passes],
            "peak_memory_mb": peak_memory_mb(),
        }
    )
)
