"""Estimates of the time and the peak memory of ``stiff_part`` on a problem, before it runs.

The estimate is a model fitted to the measurements of ``docs/performance.json`` (``michell.performance``:
``stiff_part`` on a 3D cantilever at five grid sizes, with the time and the solver steps of every pass), on
the machine those measurements name, the reference. The time of one pass, for a problem with ``n`` active
voxels (in millions), ``s`` conjugate-gradient steps in the pass and a density filter whose stencil holds
``k`` cells (thousands; ``(2 reach + 1)^3``, the cells its sums run over), is

    seconds per pass = n (a + b s + c k),

fitted by least squares on the relative error over every measured pass. The steps of a pass are those of
one solve per load case; their low, mean and high values per load case are the lowest, mean and highest of
the measured sizes' means, so the range of the time holds how the solver's work changes with the problem
and as the design forms. The peak memory is ``m n_v^p`` GB for ``n_v`` voxels of the grid (in millions),
fitted in logarithms. Each range is widened by the largest relative error of the fit at a measured size
left out of it: fitted to the other sizes and tested on that one (leave one size out). Each load case beyond
the first adds the memory of ``extra_load_case`` in the model, measured apart on the same cantilever
(``measure_extra_load_case``; ``python -m michell.estimate load-cases``).

On another machine, or another thread count, the estimate is scaled by a short timing: a fixed solve and a
fixed pass of the density filter, each timed on the problem's thread count and divided by the same workload
timed on the reference machine (``model.json`` beside this module holds both the fit and the reference timing,
with the day it was made and the versions it timed). The solve's ratio scales the terms ``a`` and ``b``, the
filter's ratio the term ``c``.

    python -m michell.estimate fit docs/performance.json --out python/michell/estimate/model.json [--timing]

writes the model from the measurements (``--timing`` also times the workload here and stores it as the
reference's; without it the stored timing is kept).
"""

from __future__ import annotations

import datetime
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from michell.errors import InvalidValue

FORMAT = 1
MODEL_PATH = Path(__file__).with_name("model.json")

# The timing: a solve and a density filter pass on a fixed small cantilever, each repeated within a budget.
TIMING_GRID = (48, 24, 24)
TIMING_VOXEL_MM = 1.0
TIMING_RADIUS_ELEMENTS = 4.5
TIMING_BUDGET_SECONDS = 0.3  # per workload; at least TIMING_LEAST repetitions whatever the budget
TIMING_LEAST = 2
TIMING_MOST = 50

# Units of the fit: active voxels in millions, filter stencil cells in thousands.
_MILLION = 1e6
_THOUSAND = 1e3


def stencil_cells(radius_elements: float, largest: int | None = None) -> int:
    """The cells a 3D density filter of ``radius_elements`` element sides sums over per voxel:
    ``(2 reach + 1)^3`` with ``reach = ceil(radius) - 1``, at most ``largest`` (the grid's largest count of
    voxels along an axis, when given), as ``michell.filters.DensityFilter`` builds it."""
    reach = max(math.ceil(radius_elements) - 1, 0)
    if largest is not None:
        reach = min(reach, int(largest))
    return (2 * reach + 1) ** 3


def _largest(size: dict) -> int | None:
    """The largest count of voxels along an axis of a measured size's grid, if recorded."""
    return max(size["grid"]) if size.get("grid") else None


def _passes(measurements: dict) -> int:
    problem = measurements["problem"]
    return sum(int(p) for _, p in problem["schedule"]) + int(problem["second_step_passes"])


_SIZE_KEYS = (
    "elements",
    "filter_radius_elements",
    "seconds_of_each_pass",
    "solver_steps_of_each_pass",
    "peak_memory_gb",
    "grid",
)


def _time_coefficients(sizes) -> tuple[float, float, float]:
    """``a``, ``b`` and ``c`` of the time per pass, by least squares on the relative error of every pass."""
    rows, seconds = [], []
    for one in sizes:
        n = one["elements"] / _MILLION
        k = stencil_cells(one["filter_radius_elements"], _largest(one)) / _THOUSAND
        for t, steps in zip(one["seconds_of_each_pass"], one["solver_steps_of_each_pass"], strict=True):
            rows.append((n, n * steps, n * k))
            seconds.append(t)
    a = np.array(rows, dtype=float)
    t = np.array(seconds, dtype=float)
    weight = 1.0 / t
    return tuple(float(c) for c in np.linalg.lstsq(a * weight[:, None], t * weight, rcond=None)[0])


def _memory_coefficients(sizes) -> tuple[float, float]:
    """``m`` and ``p`` of the peak memory ``m n_v^p``, fitted in logarithms."""
    elements = np.array([one["elements"] for one in sizes], dtype=float) / _MILLION
    memory = np.array([one["peak_memory_gb"] for one in sizes], dtype=float)
    exponent, log_at_one = (float(c) for c in np.polyfit(np.log(elements), np.log(memory), 1))
    return math.exp(log_at_one), exponent


def _seconds_at(coefficients, one: dict, steps: float) -> float:
    base, per_step, per_cell = coefficients
    n = one["elements"] / _MILLION
    cells = stencil_cells(one["filter_radius_elements"], _largest(one)) / _THOUSAND
    return n * (base + per_step * steps + per_cell * cells)


def _held_out(sizes) -> list[dict]:
    """For each size, the relative error of the fit to the other sizes there: the time per pass (at the
    size's own mean solver steps) and the peak memory."""
    out = []
    for index, one in enumerate(sizes):
        others = [other for i, other in enumerate(sizes) if i != index]
        steps = float(np.mean(one["solver_steps_of_each_pass"]))
        seconds = _seconds_at(_time_coefficients(others), one, steps)
        at_one, exponent = _memory_coefficients(others)
        memory = at_one * (one["elements"] / _MILLION) ** exponent
        out.append(
            {
                "elements": int(one["elements"]),
                "seconds_share_off": seconds / float(np.mean(one["seconds_of_each_pass"])) - 1.0,
                "memory_share_off": memory / float(one["peak_memory_gb"]) - 1.0,
            }
        )
    return out


def fit_sizes(sizes, reference: dict, passes: int) -> dict:
    """The model fitted to ``sizes`` (at least three measured sizes, each a dict with ``elements``,
    ``filter_radius_elements``, ``seconds_of_each_pass``, ``solver_steps_of_each_pass``, ``peak_memory_gb``
    and ``grid``), for the ``reference`` machine and ``passes`` passes. Each range is widened by the largest
    error of the fit at a size left out of it (``held_out``: fitted to the other sizes, tested on that one)."""
    sizes = [{key: one[key] for key in _SIZE_KEYS if key in one} for one in sizes]
    if len(sizes) < 3:
        raise InvalidValue(f"the fit needs at least three measured sizes, to leave one out; got {len(sizes)}")
    coefficients = _time_coefficients(sizes)
    base, per_step, per_cell = coefficients
    at_one, exponent = _memory_coefficients(sizes)
    held_out = _held_out(sizes)
    steps_mean = [float(np.mean(one["solver_steps_of_each_pass"])) for one in sizes]
    table = []
    for one, steps in zip(sizes, steps_mean):
        table.append(
            dict(
                one,
                solver_steps_per_pass=steps,
                seconds_per_pass=float(np.mean(one["seconds_of_each_pass"])),
                fitted_seconds_per_pass=_seconds_at(coefficients, one, steps),
                fitted_peak_memory_gb=float(at_one * (one["elements"] / _MILLION) ** exponent),
            )
        )
    elements = [one["elements"] for one in sizes]
    radii = [one["filter_radius_elements"] for one in sizes]
    return {
        "format": FORMAT,
        "reference": reference,
        "passes": passes,
        "seconds_per_pass": {
            "per_million_active": base,
            "per_million_active_and_solver_step": per_step,
            "per_million_active_and_thousand_filter_cells": per_cell,
            "largest_share_off_held_out": max(abs(row["seconds_share_off"]) for row in held_out),
        },
        "solver_steps_per_pass_and_case": {
            "low": min(steps_mean),
            "mean": float(np.mean(steps_mean)),
            "high": max(steps_mean),
        },
        "peak_memory_gb": {
            "at_one_million_voxels": at_one,
            "exponent": exponent,
            "largest_share_off_held_out": max(abs(row["memory_share_off"]) for row in held_out),
        },
        "measured": {
            "elements": [int(min(elements)), int(max(elements))],
            "filter_radius_elements": [float(min(radii)), float(max(radii))],
            "load_cases": 1,
        },
        "held_out": held_out,
        "sizes": table,
    }


def fit(measurements: dict) -> dict:
    """The model from the measurements of ``michell.performance`` (format 1, every size completed)."""
    if measurements.get("format") != 1:
        raise InvalidValue(f"the measurements must be of format 1; got {measurements.get('format')!r}")
    sizes = measurements["sizes"]
    if not sizes or any(one.get("error") for one in sizes):
        raise InvalidValue("the measurements must hold at least one size, each completed")
    machine = measurements["machine"]
    reference = {key: machine.get(key) for key in ("cpu", "logical_cores", "threads", "os")}
    return fit_sizes(sizes, reference, _passes(measurements))


def load_model(path: str | Path | None = None) -> dict:
    """The model shipped with the package (``model.json``), or the one at ``path``."""
    return json.loads(Path(path or MODEL_PATH).read_text(encoding="utf-8"))


def write_model(
    measurements_path, out, timing: dict | None = None, extra_load_case: dict | None = None
) -> dict:
    """Fits the measurements at ``measurements_path`` and writes the model to ``out`` with ``timing`` (the
    reference timing, from ``timing()``) and ``extra_load_case`` (the memory of each load case beyond the
    first, measured apart: ``{"gb_per_million_voxels": ..., "measured": ...}``); each not given is kept from
    ``out`` (from the shipped model when ``out`` does not exist)."""
    model = fit(json.loads(Path(measurements_path).read_text(encoding="utf-8")))
    stored = load_model(out if Path(out).exists() else None) if timing is None or extra_load_case is None else {}
    model["timing"] = stored["timing"] if timing is None else timing
    model["extra_load_case"] = stored["extra_load_case"] if extra_load_case is None else extra_load_case
    Path(out).write_text(json.dumps(model, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return model


class _Enough(Exception):
    pass


def load_case_memory(size: int, cases: int, passes: int, threads: int) -> dict:
    """The peak memory of this process over the first ``passes`` passes of ``stiff_part`` on the cantilever
    of ``michell.performance`` at ``size``, with ``cases`` load cases at its free end (1 to 3: the load of the
    measurements, then the same force along y, then along x). Run it in a fresh process: the peak is the
    process's. Returns ``{"size", "elements", "cases", "passes", "threads", "peak_after_each_pass_gb"}``."""
    from michell import select
    from michell.performance import LOAD_N, MEMBER_SIZE_MM, cantilever, peak_memory_gb
    from michell.problem import LoadCase, Problem
    from michell.recipes import stiff_part

    if cases not in (1, 2, 3):
        raise InvalidValue(f"cases must be 1, 2 or 3; got {cases!r}")
    problem, mass_g = cantilever(size, threads)
    if cases > 1:
        grid = problem.grid
        end = select.face_nodes(grid, "x+")
        forces = ((0.0, 0.0, -LOAD_N), (0.0, LOAD_N, 0.0), (LOAD_N, 0.0, 0.0))[:cases]
        load_cases = [LoadCase(f"case {i + 1}", select.load(grid, end, force)) for i, force in enumerate(forces)]
        held = select.held(grid, select.face_nodes(grid, "x-"))
        problem = Problem(grid, held, problem.material, load_cases, threads=threads)
    peaks = []

    def note(step, number, record):
        peaks.append(peak_memory_gb())
        if len(peaks) >= passes:
            raise _Enough

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a coarse grid's warning: the measurement is of memory alone
        try:
            stiff_part(problem, mass_g, MEMBER_SIZE_MM, callback=note)
        except _Enough:
            pass
    return {
        "size": size,
        "elements": int(problem.grid.n_elements),
        "cases": cases,
        "passes": passes,
        "threads": threads,
        "peak_after_each_pass_gb": peaks,
    }


_LOAD_CASE_CHILD = (
    "import json, sys\n"
    "from michell.estimate import load_case_memory\n"
    "print(json.dumps(load_case_memory(*(int(a) for a in sys.argv[1:]))))\n"
)


def measure_extra_load_case(size: int = 90, cases: int = 3, passes: int = 4, threads: int = 12) -> dict:
    """The memory of each load case beyond the first: ``load_case_memory`` with one load case and with
    ``cases``, each in a fresh process, and the difference of their peaks per further case and per million
    voxels of the grid. Returns ``{"gb_per_million_voxels", "measured", "runs"}``."""
    import subprocess
    import sys

    runs = []
    for count in (1, cases):
        child = subprocess.run(
            [sys.executable, "-c", _LOAD_CASE_CHILD, str(size), str(count), str(passes), str(threads)],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
        )
        runs.append(json.loads(child.stdout.strip().splitlines()[-1]))
    one, more = (max(run["peak_after_each_pass_gb"]) for run in runs)
    elements = runs[0]["elements"]
    per_case = (more - one) / (cases - 1) / (elements / _MILLION)
    how = (
        f"the difference of the peak memory over the first {passes} passes of stiff_part between 1 and {cases} load "
        f"cases on the cantilever of michell.performance at size {size} ({elements} voxels), on {threads} threads; "
        f"python -m michell.estimate load-cases --size {size} --cases {cases} --passes {passes} --threads {threads} "
        "measures it again"
    )
    return {"gb_per_million_voxels": per_case, "measured": how, "runs": runs}


def _timed(work, budget: float) -> float:
    """The shortest time of one call of ``work`` over repeated calls within ``budget`` seconds."""
    best, started, count = math.inf, time.perf_counter(), 0
    while count < TIMING_LEAST or (time.perf_counter() - started < budget and count < TIMING_MOST):
        before = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - before)
        count += 1
    return best


def timing(threads: int, budget: float = TIMING_BUDGET_SECONDS) -> dict:
    """Times the fixed workload on ``threads`` threads: one solve of a cantilever of
    ``TIMING_GRID`` voxels at a uniform density 0.5, and one pass of the density filter (forward and back)
    of radius ``TIMING_RADIUS_ELEMENTS`` element sides on the same grid; each the shortest of repeated runs
    within ``budget`` seconds. Returns ``{"threads", "solve_seconds", "filter_seconds", "seconds", "date",
    "michell", "engine"}`` (``seconds``: the whole timing, setting up included; ``date``: the day it was made;
    the last two: the versions of the package and of the engine timed)."""
    from michell import __version__, engine_version, select
    from michell.analysis3d import Elasticity3d, Grid3d
    from michell.filters import DensityFilter

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise InvalidValue(f"threads must be a whole number of at least 1; got {threads!r}")
    started = time.perf_counter()
    grid = Grid3d(*TIMING_GRID, TIMING_VOXEL_MM)
    held = select.held(grid, select.face_nodes(grid, "x-"))
    loads = select.load(grid, select.face_nodes(grid, "x+"), (0.0, 0.0, -1.0))
    model = Elasticity3d(
        grid, held, youngs_modulus=1.0, poisson_ratio=0.3, e_min=1e-9, penalty=3.0, threads=threads
    )
    rho = np.full(grid.n_elements, 0.5)
    density_filter = DensityFilter(grid, TIMING_RADIUS_ELEMENTS * TIMING_VOXEL_MM, threads)
    x = np.linspace(0.0, 1.0, grid.n_elements)

    def filter_pass():
        density_filter.backward(x, density_filter.forward(x))

    solve_seconds = _timed(lambda: model.solve(rho, loads), budget)
    filter_seconds = _timed(filter_pass, budget)
    return {
        "threads": threads,
        "solve_seconds": solve_seconds,
        "filter_seconds": filter_seconds,
        "seconds": time.perf_counter() - started,
        "date": datetime.date.today().isoformat(),
        "michell": __version__,
        "engine": engine_version(),
    }


@dataclass(frozen=True)
class Estimate:
    """The estimate of a run of ``stiff_part``: each range a ``(low, centre, high)`` triple.

    - ``seconds_per_pass``, ``total_seconds`` (the ``passes`` of the default schedule, the passes only: not
      the preparation, not the mesh of the part), ``peak_memory_gb``.
    - ``scaled``: whether a timing on this machine scaled it (``timing`` then holds it, with the ratios
      ``solve_ratio`` and ``filter_ratio`` to the reference); otherwise it is the reference machine's.
    - ``reference``: the reference machine (processor, logical cores, threads).
    - The problem: ``active`` and ``voxels``, ``filter_radius_elements``, ``load_cases``, ``threads``.
    - ``within_measured``: whether the problem lies inside what was measured: the sizes, a filter radius at
      most the measured, one load case and, without a timing, the reference's thread count (``outside`` says
      how not).
    """

    seconds_per_pass: tuple[float, float, float]
    total_seconds: tuple[float, float, float]
    peak_memory_gb: tuple[float, float, float]
    passes: int
    scaled: bool
    timing: dict | None
    reference: dict
    active: int
    voxels: int
    filter_radius_elements: float
    load_cases: int
    threads: int
    within_measured: bool
    outside: tuple[str, ...]

    def as_dict(self) -> dict:
        def triple(values):
            return {"low": values[0], "centre": values[1], "high": values[2]}

        return {
            "seconds_per_pass": triple(self.seconds_per_pass),
            "total_seconds": triple(self.total_seconds),
            "peak_memory_gb": triple(self.peak_memory_gb),
            "passes": self.passes,
            "scaled": self.scaled,
            "timing": self.timing,
            "reference": self.reference,
            "active": self.active,
            "voxels": self.voxels,
            "filter_radius_elements": self.filter_radius_elements,
            "load_cases": self.load_cases,
            "threads": self.threads,
            "within_measured": self.within_measured,
            "outside": list(self.outside),
            "covers": "the passes of stiff_part; not the preparation and not the mesh of the part",
        }


def estimate(
    active: int,
    voxels: int,
    filter_radius_elements: float,
    load_cases: int,
    threads: int,
    *,
    timed: bool | dict = False,
    model: dict | None = None,
    largest: int | None = None,
) -> Estimate:
    """The estimate for a problem of ``active`` active voxels on a grid of ``voxels`` (``largest`` along its
    longest axis, when given: it caps the filter's reach), a density filter of ``filter_radius_elements``
    element sides (the member size over the voxel side), ``load_cases`` load cases and ``threads`` threads.
    ``timed``: False for the reference machine's estimate, True to time the workload here (``timing``), or a
    timing already made. A problem outside what was measured (the sizes, a filter radius above the measured,
    more than one load case and, without a timing, a thread count other than the reference's) is estimated all
    the same; ``within_measured`` and ``outside`` say so (``michell check`` prints it as a warning)."""
    model = load_model() if model is None else model
    if timed is True:
        timed = timing(threads)
    solve_ratio = filter_ratio = 1.0
    if timed:
        reference = model["timing"]
        solve_ratio = timed["solve_seconds"] / reference["solve_seconds"]
        filter_ratio = timed["filter_seconds"] / reference["filter_seconds"]
        timed = dict(timed, solve_ratio=solve_ratio, filter_ratio=filter_ratio)
    time_fit, steps = model["seconds_per_pass"], model["solver_steps_per_pass_and_case"]
    n = active / _MILLION
    cells = stencil_cells(filter_radius_elements, largest) / _THOUSAND
    share = time_fit["largest_share_off_held_out"]
    base, per_step = time_fit["per_million_active"], time_fit["per_million_active_and_solver_step"]
    per_cell = time_fit["per_million_active_and_thousand_filter_cells"]

    def per_pass(steps_per_case: float) -> float:
        return n * (solve_ratio * (base + per_step * steps_per_case * load_cases) + filter_ratio * per_cell * cells)

    seconds = (per_pass(steps["low"]) * (1 - share), per_pass(steps["mean"]), per_pass(steps["high"]) * (1 + share))
    passes = model["passes"]
    memory_fit = model["peak_memory_gb"]
    memory = memory_fit["at_one_million_voxels"] * (voxels / _MILLION) ** memory_fit["exponent"]
    memory += (load_cases - 1) * model["extra_load_case"]["gb_per_million_voxels"] * voxels / _MILLION
    memory_share = memory_fit["largest_share_off_held_out"]
    measured = model["measured"]
    outside = []
    low, high = measured["elements"]
    if not low <= active <= high:
        outside.append(f"{active} active voxels, outside the measured {low} to {high}")
    high = measured["filter_radius_elements"][1]
    if filter_radius_elements > high:  # below the measured radii the filter's term only shrinks (a small share)
        outside.append(
            f"a filter radius of {filter_radius_elements:.3g} voxel sides, above the measured {high:.3g}"
        )
    if load_cases > measured["load_cases"]:
        outside.append(
            f"{load_cases} load cases, more than the measured {measured['load_cases']} (the memory of each further "
            f"case was measured apart: {model['extra_load_case']['measured']})"
        )
    reference_threads = model["reference"]["threads"]
    if not timed and threads != reference_threads:
        outside.append(
            f"no timing here, so the time is the reference machine's on its {reference_threads} threads, not on "
            f"the problem's {threads}"
        )
    return Estimate(
        seconds_per_pass=seconds,
        total_seconds=tuple(passes * s for s in seconds),
        peak_memory_gb=(memory * (1 - memory_share), memory, memory * (1 + memory_share)),
        passes=passes,
        scaled=bool(timed),
        timing=timed or None,
        reference=model["reference"],
        active=int(active),
        voxels=int(voxels),
        filter_radius_elements=float(filter_radius_elements),
        load_cases=int(load_cases),
        threads=int(threads),
        within_measured=not outside,
        outside=tuple(outside),
    )

