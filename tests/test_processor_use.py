"""A run of the optimisation loop keeps one processor core busy, not all of them.

The engine runs on one thread, so the processor time of a run should be about its wall time. NumPy's `@` on
vectors of more than 10000 entries starts the threads of its BLAS library, which then wait for work at full
load between the calls of the loop; the loop uses `michell.dot` for its products instead. This test runs a
short optimisation in a fresh Python process (so that threads started by other tests do not count), with no
thread-count setting in its environment, and compares the processor time of the process with its wall time.

On a loaded machine the ratio can only fall, because the process then waits for a core; so the test is not
sensitive to load. On fewer than three cores the threads could not be told apart from normal running, so it
is skipped there.
"""

import json
import os
import subprocess
import sys

import pytest

# The settings that limit the BLAS threads; removed so that they cannot hide the problem.
THREAD_SETTINGS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
)

# Processor time over wall time must stay below this. A run on one thread gives about 1.0; with the BLAS
# threads busy the ratio was close to the number of cores (15.7 on 16 cores).
LIMIT = 1.5

# The MBB beam of `examples/mbb_beam.py` (150 x 50 elements: 15402 degrees of freedom, above the 10000
# entries at which OpenBLAS starts its threads) for 15 passes; prints wall and processor time as JSON.
RUN = """
import json, time
import numpy as np
import michell

nx, ny = 150, 50
grid = michell.Grid(nx, ny, 1.0, 1.0)
held = np.zeros(grid.n_dofs, dtype=bool)
for j in range(ny + 1):
    held[2 * grid.node(0, j)] = True
held[2 * grid.node(nx, 0) + 1] = True
loads = np.zeros(grid.n_dofs)
loads[2 * grid.node(0, ny) + 1] = -1.0
material = michell.Material(youngs_modulus=1.0, poisson_ratio=0.3, density_g_cm3=1000.0)
problem = michell.Problem(grid, held, material, [michell.LoadCase("load", loads)], penalty=3.0, e_min=1e-9)

# OpenBLAS starts its threads when NumPy is imported, and they wait at full load for a moment before they
# sleep: a fixed cost of the import (about 1.3 s of processor time on 16 cores), not of the loop.
time.sleep(0.5)
wall, cpu = time.perf_counter(), time.process_time()
michell.optimise(
    problem,
    goal=michell.Compliance(),
    limits=[michell.Limit(michell.Mass(), at_most=0.5 * nx * ny)],
    filters=[michell.DensityFilter(grid, 6.0)],
    optimiser=michell.OptimalityCriteria(),
    x=np.full(grid.n_elements, 0.5),
    passes=15,
)
print(json.dumps({"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}))
"""


@pytest.mark.skipif((os.cpu_count() or 1) < 3, reason="fewer than three processor cores")
def test_a_run_uses_about_one_core(tmp_path):
    environment = {name: value for name, value in os.environ.items() if name not in THREAD_SETTINGS}
    result = subprocess.run(
        [sys.executable, "-c", RUN],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    times = json.loads(result.stdout.strip().splitlines()[-1])
    ratio = times["cpu"] / times["wall"]
    print(f"wall {times['wall']:.2f} s, processor {times['cpu']:.2f} s, ratio {ratio:.2f}")
    assert ratio < LIMIT, f"processor time {ratio:.2f} times the wall time; limit {LIMIT}"
