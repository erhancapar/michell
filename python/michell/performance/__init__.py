"""Run time and peak memory of ``michell.recipes.stiff_part`` on a 3D cantilever at several grid sizes.

    python -m michell.performance run --sizes 36 46 58 72 90 --threads 12 --out performance.json
    python -m michell.performance render performance.json --out docs/performance.md

The problem: a steel block of 40 x 20 x 20 mm on a grid of 2m x m x m cube elements for a size m, every
degree of freedom held on the face x = 0, and a force of 1000 N down (-z) spread over the nodes of the face
x = 40 mm as a uniform traction (``michell.select.load``). ``stiff_part`` runs with its default settings
and schedule, the asked mass 30 % of the block's and the member size 3.0 mm. ``measure`` runs one size in
the calling process; ``run`` starts a fresh Python process for each size, so that the peak memory it reads
is that size's own, and collects their results with a description of the machine. The machine is described
by its processor, core count, memory and operating system only: no host name, user name or path is
written. ``michell.performance.page.render`` writes the page from the result.

A size whose run fails still gives its record: what was measured up to the failure, with an ``"error"``
entry that says what was raised and where (``measure``), or, when its process ended without giving a record,
the process's exit status and the last lines of its error output (``run``). ``run`` writes the result
collected so far after every size, so a run that is stopped leaves the sizes it finished.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

from michell import EngineError, __version__, engine_version, select
from michell.analysis3d import Grid3d
from michell.errors import InvalidValue
from michell.problem import LoadCase, Material, Problem
from michell.recipes import (
    MEMBER_SIZE_STEP,
    PASSES_PER_BOUND,
    ROBUST_STEP,
    SCHEDULE,
    SECOND_STEP_PASSES,
    stiff_part,
)

FORMAT = 1
BLOCK_MM = (40.0, 20.0, 20.0)
YOUNGS_MODULUS = 210000.0  # MPa
POISSON_RATIO = 0.3
DENSITY_G_CM3 = 7.85
LOAD_N = 1000.0
MASS_SHARE = 0.30
MEMBER_SIZE_MM = 3.0
SIZES = (36, 46, 58, 72, 90)
THREADS = 12
SLOWEST = 10
ERROR_LINES = 20  # the last lines of a process's error output kept when it gives no record


class _CountingProblem(Problem):
    """A ``Problem`` that notes the conjugate-gradient steps of every solve and adjoint solve it makes."""

    def __init__(self, *arguments, **settings):
        super().__init__(*arguments, **settings)
        self.solver_steps: list[int] = []

    def solve(self, density, loads, start=None):
        u, steps = super().solve(density, loads, start=start)
        self.solver_steps.append(steps)
        return u, steps

    def adjoint(self, density, rhs):
        lam, steps = super().adjoint(density, rhs)
        self.solver_steps.append(steps)
        return lam, steps


def _check_size(size) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise InvalidValue(f"the size must be a whole number of at least 1; got {size!r}")


def cantilever(size: int, threads: int = 1) -> tuple[Problem, float]:
    """The problem at size ``size`` (a grid of 2 size x size x size elements) and the asked mass in g."""
    _check_size(size)
    h = BLOCK_MM[1] / size
    grid = Grid3d(2 * size, size, size, h)
    held = select.held(grid, select.face_nodes(grid, "x-"))
    loads = select.load(grid, select.face_nodes(grid, "x+"), (0.0, 0.0, -LOAD_N))
    steel = Material(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density_g_cm3=DENSITY_G_CM3)
    problem = _CountingProblem(grid, held, steel, [LoadCase("end", loads)], threads=threads)
    mass_g = MASS_SHARE * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
    return problem, mass_g


def scaled_passes(passes_scale: float) -> tuple[tuple[tuple[float, int], ...], int]:
    """The schedule of the robust step and the passes of the member-size step, each count multiplied by
    ``passes_scale`` and rounded, at least 1."""
    if isinstance(passes_scale, bool) or not isinstance(passes_scale, (int, float)):
        raise InvalidValue(f"the passes scale must be a number; got {passes_scale!r}")
    if not (math.isfinite(passes_scale) and passes_scale > 0.0):
        raise InvalidValue(f"the passes scale must be a finite number above 0; got {passes_scale!r}")
    schedule = tuple((beta, max(1, round(passes * passes_scale))) for beta, passes in SCHEDULE)
    return schedule, max(1, round(SECOND_STEP_PASSES * passes_scale))


def peak_memory_gb() -> float | None:
    """The largest resident memory of this process so far, in GB; ``None`` where the system does not report
    it. On Linux the value is ``VmHWM`` of ``/proc/self/status``: ``resource.getrusage`` reports there the
    larger of the process's own peak and that of the process that started it, which would hide a small
    run's own peak behind its parent's. Elsewhere it is ``ru_maxrss`` of ``resource.getrusage`` (bytes on
    macOS, kB on other systems)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024 / 1e9  # in kB
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def measure(size: int, threads: int = 1, passes_scale: float = 1.0) -> dict:
    """Runs ``stiff_part`` on the problem at ``size`` in this process and returns its measurements.

    The time of a pass runs from the end of the pass before it (for the first, from the call of
    ``stiff_part``) to the end of its update, as the callback sees it, so the passes add up to the whole run
    except the evaluation of the returned design. The solver steps are those of every solve made during the
    call. The peak memory is that of this process."""
    schedule, second_step_passes = scaled_passes(passes_scale)
    problem, mass_g = cantilever(size, threads)
    grid = problem.grid
    ends: list[tuple[str, float, int]] = []

    def note(step, number, record):
        ends.append((step, time.perf_counter(), len(problem.solver_steps)))

    part, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        try:
            part = stiff_part(
                problem, mass_g, MEMBER_SIZE_MM, schedule=schedule, second_step_passes=second_step_passes,
                callback=note,
            )  # fmt: skip
        except Exception as raised:  # every failure is recorded with the passes completed before it
            error = raised
        total = time.perf_counter() - started
    seconds, steps = [], []
    by_step: dict[str, list[float]] = {ROBUST_STEP: [], MEMBER_SIZE_STEP: []}
    before, solves_before = started, 0
    for step, end, solves in ends:
        seconds.append(end - before)
        by_step[step].append(end - before)
        steps.append(sum(problem.solver_steps[solves_before:solves]))
        before, solves_before = end, solves
    result = None if part is None else part.result
    return {
        "size": size,
        "grid": [grid.nx, grid.ny, grid.nz],
        "elements": grid.n_elements,
        "voxel_mm": float(grid.h),
        "filter_radius_elements": MEMBER_SIZE_MM / float(grid.h),
        "passes": len(seconds),
        "seconds_per_pass": {
            "mean": _mean(seconds),
            "first_step_mean": _mean(by_step[ROBUST_STEP]),
            "second_step_mean": _mean(by_step[MEMBER_SIZE_STEP]),
            "slowest_ten_mean": _mean(sorted(seconds)[-SLOWEST:]),
        },
        "total_seconds": total,
        "peak_memory_gb": peak_memory_gb(),
        "solver_steps_per_solve": {
            "mean": _mean(problem.solver_steps),
            "largest": max(problem.solver_steps, default=None),
            "solves": len(problem.solver_steps),
        },
        "goal": None if result is None else float(result.goal),
        "limits_met": None if result is None else all(result.limits_met),
        "stop_reason": None if result is None else result.stop_reason,
        "warnings": sorted({type(w.message).__name__ for w in caught}),
        "seconds_of_each_pass": seconds,
        "solver_steps_of_each_pass": steps,
        "error": None if error is None else _raised(error, len(seconds), sum(p for _, p in schedule)),
    }


def _finite_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None if value is None else str(value)
    return value if math.isfinite(value) else None


def _raised(error: Exception, completed: int, robust_passes: int) -> dict:
    """The ``"error"`` entry of a run that raised ``error`` after ``completed`` passes: the exception's class
    name, its code and details (an ``EngineError``'s; ``None`` and ``{}`` otherwise), its message, the step
    and the number in that step of the pass in which it was raised (as ``stiff_part`` gives them, or else
    counted from the passes completed), and the number of that pass counted through the whole run."""
    number = completed + 1
    counted_step = ROBUST_STEP if number <= robust_passes else MEMBER_SIZE_STEP
    counted_in_step = number if number <= robust_passes else number - robust_passes
    step = getattr(error, "step", None) if isinstance(error, EngineError) else None
    in_step = getattr(error, "pass_number", None) if isinstance(error, EngineError) else None
    if step is None or in_step is None:
        step, in_step = counted_step, counted_in_step
    details = getattr(error, "details", None)
    message = getattr(error, "message", None)
    return {
        "type": type(error).__name__,
        "code": getattr(error, "code", None) if isinstance(error, EngineError) else None,
        "message": _scrubbed(message if isinstance(message, str) else str(error)),
        "details": (
            {str(key): _finite_or_none(value) for key, value in details.items()} if isinstance(details, dict) else {}
        ),
        "step": step,
        "pass_in_step": in_step,
        "pass": number,
        "exit_status": None,
        "error_output": [],
    }


# A path: a run of /name or \name parts (after an optional drive letter or ~), not preceded by a letter, a
# digit or a dot, so that a unit such as N/mm is left alone.
_PATH = re.compile(r"(?<![\w.\\/])(?:[A-Za-z]:|~)?(?:[\\/][^\s\\/'\"(),:;]+)+[\\/]?")


def _scrubbed(text: str) -> str:
    """``text`` with every path cut to its file name and the machine's host name taken out."""
    text = _PATH.sub(lambda found: re.split(r"[\\/]", found.group(0).rstrip("\\/"))[-1], text)
    host = socket.gethostname()
    for name in sorted({host, host.split(".")[0]} - {""}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", "(host)", text)
    return text


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    key, _, value = line.partition(":")
                    if key.strip() in ("model name", "Model", "Hardware"):
                        return value.strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            found = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            )
            return found.stdout.strip() or None
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or None


def _memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "darwin":
        try:
            found = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            return int(found.stdout) / 1e9
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return None


def _distribution() -> str | None:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except (AttributeError, OSError):
        return None


def machine(threads: int) -> dict:
    """The machine: processor, logical cores, memory, operating system, Python and michell versions, and
    the threads used. Nothing that names the machine or its user."""
    return {
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] if sys.platform == "darwin" else platform.release(),
        "os_distribution": _distribution(),
        "python": platform.python_version(),
        "michell": __version__,
        "engine": engine_version(),
        "threads": threads,
    }


def problem_description(passes_scale: float) -> dict:
    """The fixed settings of the problem and of the procedure, as the page states them."""
    schedule, second_step_passes = scaled_passes(passes_scale)
    return {
        "block_mm": list(BLOCK_MM),
        "youngs_modulus_mpa": YOUNGS_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "density_g_cm3": DENSITY_G_CM3,
        "load_n": LOAD_N,
        "mass_share": MASS_SHARE,
        "member_size_mm": MEMBER_SIZE_MM,
        "schedule": [[beta, passes] for beta, passes in schedule],
        "second_step_passes": second_step_passes,
        "passes_per_bound": PASSES_PER_BOUND,
        "passes_scale": passes_scale,
    }


_CHILD = (
    "import json, sys\n"
    "from michell.performance import measure\n"
    "print(json.dumps(measure(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))))\n"
)


def _record_of(size: int, child: subprocess.CompletedProcess) -> dict:
    """The record the process for ``size`` printed last, or, when it printed none, the record of a run that
    ended without one: the grid, nothing measured, and its exit status and last lines of error output."""
    lines = (child.stdout or "").strip().splitlines()
    try:
        one = json.loads(lines[-1])
    except (IndexError, ValueError):
        one = None
    if isinstance(one, dict) and one.get("size") == size:
        return one
    voxel = BLOCK_MM[1] / size
    output = [_scrubbed(line.rstrip()) for line in (child.stderr or "").splitlines() if line.strip()]
    return {
        "size": size,
        "grid": [2 * size, size, size],
        "elements": 2 * size**3,
        "voxel_mm": voxel,
        "filter_radius_elements": MEMBER_SIZE_MM / voxel,
        "passes": None,
        "seconds_per_pass": {
            "mean": None, "first_step_mean": None, "second_step_mean": None, "slowest_ten_mean": None,
        },
        "total_seconds": None,
        "peak_memory_gb": None,
        "solver_steps_per_solve": {"mean": None, "largest": None, "solves": None},
        "goal": None,
        "limits_met": None,
        "stop_reason": None,
        "warnings": [],
        "seconds_of_each_pass": [],
        "solver_steps_of_each_pass": [],
        "error": {
            "type": None, "code": None, "message": None, "details": {}, "step": None, "pass_in_step": None,
            "pass": None, "exit_status": child.returncode, "error_output": output[-ERROR_LINES:],
        },
    }  # fmt: skip


def write_json(path, data: dict) -> None:
    """Writes ``data`` as JSON to ``path`` at once: to a temporary file beside it (``.<name>.part``), then
    renamed over it, so that the file is never left half written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(sizes, threads: int = THREADS, passes_scale: float = 1.0, report=None, out=None) -> dict:
    """Measures each size in a fresh Python process (``measure``) and returns the whole result:
    ``{"format": 1, "machine": ..., "problem": ..., "sizes": [...]}``. ``report``, if given, is called
    with each size's record as it arrives. ``out``, if given, is the JSON file the result collected so far
    is written to after every size (``write_json``).

    A size that fails does not stop the others: its record has an ``"error"`` entry (``None`` for a size
    that completed). A process that ends without printing its record (stopped, or out of memory) is recorded
    with its exit status and the last lines of its error output, every path in them cut to its file name."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise InvalidValue(f"threads must be a whole number of at least 1; got {threads!r}")
    for size in sizes:
        _check_size(size)
    data = {"format": FORMAT, "machine": machine(threads), "problem": problem_description(passes_scale), "sizes": []}
    for size in sizes:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, str(size), str(threads), repr(float(passes_scale))],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        one = _record_of(size, child)
        data["sizes"].append(one)
        if out is not None:
            write_json(out, data)
        if report is not None:
            report(one)
    return data


def failed(data: dict) -> list[dict]:
    """The records of the sizes of ``data`` (a result of ``run``) that did not complete."""
    return [one for one in data["sizes"] if one.get("error")]


__all__ = [
    "cantilever", "failed", "machine", "measure", "peak_memory_gb", "problem_description", "run", "scaled_passes",
    "write_json",
]  # fmt: skip
