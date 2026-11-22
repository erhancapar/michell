"""The performance page as Markdown, written from the result of ``michell.performance.run`` alone: every
number on the page comes from that result."""

from __future__ import annotations

import textwrap

from michell.row import g3

NOT_REPORTED = "not reported"
WIDTH = 112


def _wrap(text: str, first: str = "", rest: str = "") -> str:
    # An equation such as "x = 0" is kept on one line: its spaces are held as no-break spaces while wrapping.
    held = text.replace(" = ", "\u00a0=\u00a0")
    wrapped = textwrap.fill(
        held, WIDTH, initial_indent=first, subsequent_indent=rest, break_long_words=False, break_on_hyphens=False
    )
    return wrapped.replace("\u00a0", " ")


def _paragraph(text: str) -> str:
    return _wrap(text)


def _bullet(text: str) -> str:
    return _wrap(text, "- ", "  ")


def _number(value) -> str:
    return NOT_REPORTED if value is None else g3(value)


def _grid(one: dict) -> str:
    return " x ".join(str(count) for count in one["grid"])


def _introduction(data: dict) -> list[str]:
    problem = data["problem"]
    x, y, z = (f"{side:g}" for side in problem["block_mm"])
    robust = sum(passes for _, passes in problem["schedule"])
    second = problem["second_step_passes"]
    count = len(data["sizes"])
    if problem["passes_scale"] == 1:
        settings = "with its default settings"
    else:
        settings = (
            "with its default settings except the number of passes, each count of the schedule multiplied by "
            f"{problem['passes_scale']:g}"
        )
    what = (
        "This page gives the run time and the peak memory of `michell.recipes.stiff_part`, the library's whole "
        f"procedure from a uniform start to a design, on one 3D problem at {count} grid size"
        f"{'' if count == 1 else 's'}. The problem: a steel block of {x} by {y} by {z} mm (Young's modulus "
        f"{problem['youngs_modulus_mpa']:g} MPa, Poisson ratio {problem['poisson_ratio']:g}, "
        f"{problem['density_g_cm3']:g} g/cm³), every degree of freedom held on its face x = 0, and a force of "
        f"{problem['load_n']:g} N down spread over the nodes of its face x = {x} mm as a uniform traction. "
        f"`stiff_part` makes the stiffest part with at most {100 * problem['mass_share']:g} % of the block's mass "
        f"and members at least {problem['member_size_mm']:g} mm thick, {settings}: {robust} passes of the robust "
        f"formulation, then {second} with the member-size limit, {robust + second} in all. For a size m the grid "
        "has 2m by m by m cube elements."
    )
    how = (
        "One pass is one update of the design: the filters, the solve of the load case on the design whose "
        "compliance is the goal (conjugate gradients preconditioned by a multigrid cycle), the gradients back "
        "through the filters and the update by the method of moving asymptotes. A pass is timed from the end of "
        "the one before to the end of its update, so the passes add up to nearly the whole run: the first pass "
        "also holds the setting up of the density filter, and a pass that begins a new call of `optimise` (every "
        f"{problem['passes_per_bound']} passes and at each change of sharpness in the robust step, and at the "
        "switch to the member-size limit) also holds the evaluation of the design the call before returned and "
        "the new bound on the mass, and starts its solve from zero instead of from the state of the pass before. "
        "Each size runs in a Python process of its own, so its peak memory, the largest resident memory of that "
        "process, is that size's own; it includes the interpreter and NumPy."
    )
    return [_paragraph(what), "", _paragraph(how)]


def _machine(data: dict) -> list[str]:
    machine = data["machine"]
    system = " ".join(part for part in (machine["os"], machine["os_version"]) if part) or NOT_REPORTED
    if machine.get("os_distribution"):
        system = f"{machine['os_distribution']} ({system})"
    memory = NOT_REPORTED if machine["memory_gb"] is None else f"{g3(machine['memory_gb'])} GB"
    return [
        "## Machine",
        "",
        f"- Processor: {machine['cpu'] or NOT_REPORTED}",
        f"- Logical cores: {machine['logical_cores'] or NOT_REPORTED}",
        f"- Memory: {memory}",
        f"- Operating system: {system}",
        f"- Python {machine['python']}, michell {machine['michell']} (engine {machine['engine']})",
        f"- Threads of the solves, the filter and the updates: {machine['threads']}",
    ]


def _results(data: dict) -> list[str]:
    lines = [
        "## Results",
        "",
        "| Grid | Elements | Voxel (mm) | Filter radius (elements) | Passes | Seconds per pass | Slowest ten (s) "
        "| Peak memory (GB) | Solver steps per solve | Largest solve (steps) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for one in data["sizes"]:
        seconds, steps = one["seconds_per_pass"], one["solver_steps_per_solve"]
        lines.append(
            f"| {_grid(one)} | {one['elements']:,} | {g3(one['voxel_mm'])} | {g3(one['filter_radius_elements'])} "
            f"| {one['passes']} | {_number(seconds['mean'])} | {_number(seconds['slowest_ten_mean'])} "
            f"| {_number(one['peak_memory_gb'])} | {_number(steps['mean'])} | {_number(steps['largest'])} |"
        )
    lines += [
        "",
        _paragraph(
            "Seconds per pass is the mean over every pass of the run, and the slowest ten the mean of its ten "
            "slowest passes. Solver steps per solve are the conjugate-gradient steps of one solve, as the mean over "
            "every solve of the run; the largest solve is the one that took the most. The memory is in units of "
            "10⁹ bytes."
        ),
        "",
        "| Grid | Robust step (s per pass) | Member-size step (s per pass) | Total (s) | Compliance (N mm) "
        "| Limits met | Stop reason |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for one in data["sizes"]:
        seconds = one["seconds_per_pass"]
        lines.append(
            f"| {_grid(one)} | {_number(seconds['first_step_mean'])} | {_number(seconds['second_step_mean'])} "
            f"| {g3(one['total_seconds'])} | {g3(one['goal'])} | {'yes' if one['limits_met'] else 'no'} "
            f"| {one['stop_reason']} |"
        )
    lines += [
        "",
        _paragraph(
            "The compliance is that of the returned design, and its limits are its mass and its member-size value; "
            "the total is the time of the whole call of `stiff_part`."
        ),
    ]
    for one in data["sizes"]:
        if one.get("warnings"):
            names = ", ".join(f"`{name}`" for name in one["warnings"])
            lines += ["", f"At {_grid(one)} `stiff_part` warned: {names}."]
    return lines


def _command(data: dict) -> str:
    sizes = " ".join(str(one["size"]) for one in data["sizes"])
    scale = data["problem"]["passes_scale"]
    option = "" if scale == 1 else f" --passes-scale {scale:g}"
    return (
        f"    python -m michell.performance run --sizes {sizes} --threads {data['machine']['threads']}{option} "
        "--out docs/performance.json\n"
        "    python -m michell.performance render docs/performance.json --out docs/performance.md"
    )


def _caveats(data: dict) -> list[str]:
    radii = [one["filter_radius_elements"] for one in data["sizes"]]
    grows = f", from {g3(min(radii))} to {g3(max(radii))} elements here," if len(radii) > 1 else ""
    return [
        "## What the numbers do not say",
        "",
        _bullet(
            f"They come from one machine, the one above, on {data['machine']['threads']} threads; another "
            "processor, another memory or another thread count gives other numbers."
        ),
        _bullet(
            f"The member size is fixed in millimetres ({data['problem']['member_size_mm']:g} mm), so the filter "
            f"radius counted in elements grows with the grid{grows} and the density filter's share of a pass grows "
            "with it: the time of a pass does not follow the number of elements alone."
        ),
        _bullet(
            "Another problem gives other numbers: the steps of a solve, and with them the time of a pass, depend on "
            "the shape, the supports, the loads and the design as it forms."
        ),
    ]


def render(data: dict) -> str:
    """The page for ``data``, the result of ``michell.performance.run`` (as read from its JSON file)."""
    if data.get("format") != 1:
        raise ValueError(f"the performance result has format {data.get('format')!r}; this page reads format 1")
    lines = ["# Performance", "", *_introduction(data), ""]
    lines += [*_machine(data), "", *_results(data), ""]
    lines += [
        "## Repeating the measurement",
        "",
        "From the repository, with the package installed:",
        "",
        _command(data),
        "",
        _paragraph(
            "The first command writes the measurements, with the time and the solver steps of every pass, to the "
            "JSON file; the second writes this page from it. Run on another machine, they give that machine's "
            "numbers."
        ),
        "",
        *_caveats(data),
        "",
    ]
    return "\n".join(lines)


__all__ = ["render"]
