"""The ``michell`` command.

    michell check FILE [--threads N] [--json] [--no-timing] [--picture SVG]
    michell run FILE [--out DIR] [--threads N] [--json]
    michell --version

``check`` reads a problem file, builds the voxel model and places the supports and loads, and prints what it found
with an estimate of the run's time and peak memory; it solves nothing (``michell.problem_file.prepare``). The
estimate is scaled to this machine by a timing of about a second on the problem's threads, unless ``--no-timing``
asks for the reference machine's (``michell.estimate``). ``--picture`` writes a picture of the voxel model with
its supports and loads (``michell.picture``). ``run`` does the same (with the reference machine's estimate), then
the optimisation, and writes the outcome (``michell.problem_file.run``). With ``--json`` the standard output holds
one JSON object and nothing else; the line printed after every pass of ``run`` then goes to standard error.

Exit status: 0 when the file is accepted (``check``) or the run met its limits and delivered the mesh asked
for (``run``); 1 when the run finished but a limit is not met or the delivered mesh differs (the part's
mesh failed a gate and was not written, or the mesh voxelised back differs from the optimised design by more
than its tolerance, ``michell.part_mesh.DeliveredPartDiffers``); 2 when the input is refused: anything
refused before the first pass (the problem file, the geometry, the set-up, a part that is not held, an output
folder that cannot be made, and the command line itself, which ``argparse`` refuses with a usage line on
standard error and nothing on standard output); 3 when the engine fails during the passes (an ``EngineError`` of the engine, such as
``solver_not_converged``, rather than a refusal of ``michell.errors``) or anything else goes wrong (code
``internal_error``).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from michell import __version__, engine_version
from michell._engine import EngineError
from michell.errors import OutOfRange, _Refused
from michell.export import _plain
from michell.part_mesh import DELIVERED_TOLERANCE
from michell.picture import write_picture
from michell.problem_file import Prepared, RunOutcome, _optimise, _prepare

LIMITS_MET, LIMITS_NOT_MET, REFUSED, ENGINE_FAILED = 0, 1, 2, 3


def _error(error: EngineError, file: str) -> dict:
    if hasattr(error, "as_dict"):
        return error.as_dict()
    return {
        "code": getattr(error, "code", "internal_error"),
        "path": getattr(error, "path", ""),
        "file": file,
        "message": getattr(error, "message", str(error)),
        "details": dict(getattr(error, "details", {}) or {}),
    }


def _print_error(error: dict) -> None:
    where = f" at {error['path']}" if error["path"] else ""
    print(f"michell: refused, {error['code']}{where}: {error['message']}", file=sys.stderr)


def _numbers(values, unit: str = "") -> str:
    return "(" + ", ".join(f"{v:.4g}" for v in values) + ")" + (f" {unit}" if unit else "")


def _check_lines(summary: dict, with_estimate: bool = True) -> list[str]:
    """The prepared problem for people, one item per line (the estimate of time and memory too, for
    ``check``)."""
    grid, set_up = summary["grid"], summary["set_up"]
    nx, ny, nz = grid["counts"]
    lines = [
        f"{summary['problem_file']['name']}: read and checked (sha256 {summary['problem_file']['sha256'][:12]})",
        f"grid: {nx} x {ny} x {nz} voxels of {grid['voxel_mm']:g} mm, corner at {_numbers(grid['origin_mm'], 'mm')}; "
        f"{grid['active']} active, {grid['held_solid']} held solid ({grid['held_solid_regions']} by the solid "
        f"regions), {grid['design']} design",
        f"bodies of the part: {set_up['bodies']['count']}"
        + (f" (sizes in voxels {set_up['bodies']['sizes']})" if set_up["bodies"]["sizes"] else ""),
        "volumes (mm3): name, role, exact, sampled, voxels",
    ]
    for row in summary["volumes"]:
        exact = "-" if row["exact_mm3"] is None else f"{row['exact_mm3']:.6g}"
        lines.append(f"  {row['name']}, {row['role']}, {exact}, {row['sampled_mm3']:.6g}, {row['voxels_mm3']:.6g}")
    material = set_up["material"]
    lines.append(
        f"material: {material['name'] or 'unnamed'}, E {material['youngs_modulus_mpa']:g} MPa, Poisson's ratio "
        f"{material['poisson_ratio']:g}, {material['density_g_cm3']:g} g/cm3"
    )
    for line in set_up["supports"] + set_up["loads"]:
        where = (
            f"{line['faces']} contact faces, {line['area_mm2']:.4g} mm2"
            if line["rule"] == "contact"
            else f"{line['voxels']} voxels, {line['volume_mm3']:.4g} mm3"
        )
        if line["role"] == "support":
            what = f"holds {''.join(line['fix'])}"
        else:
            what = f"force {_numbers(line['force_n'], 'N')}, {100 * line['dropped_share']:.3g} % on held nodes"
        skin = f"; skin {line['skin_voxels']} voxels, {line['skin_mass_g']:.4g} g" if line["skin_voxels"] else ""
        closed = f"; on closed face {', '.join(line['closed_faces'])}" if line["closed_faces"] else ""
        lines.append(
            f"{line['key']} on {line['on']}: {where}, {line['nodes']} nodes, {what}, at "
            f"{_numbers(line['point_mm'], 'mm')}{skin}{closed}"
        )
    faces = ", ".join(f"{face} {word}" for face, word in set_up["filter_faces"].items())
    lines.append(f"density filter faces: {faces}; inactive voxels {set_up['inactive']}")
    goal, mass, size = set_up["goal"], set_up["mass_limit"], set_up["member_size"]
    lines.append(f"goal: {goal['module']}, {goal['sense']}")
    source = (
        f"max_fraction {mass['max_fraction']:g} of {mass['part_mass_g']:.6g} g" if "max_fraction" in mass else "max"
    )
    lines.append(f"mass bound: {mass['max_g']:.6g} g ({source}; held solid {mass['held_mass_g']:.6g} g)")
    given = "given" if size["given"] else "not given: five voxel sides"
    lines.append(f"member size: {size['size_mm']:g} mm, {size['voxels']:.3g} voxels ({given})")
    lines.append(f"threads: {set_up['threads']}")
    output = summary["output"]
    lines.append(f"run writes into {output['folder']}: {', '.join(output['files'])}")
    if with_estimate:
        lines.extend(_estimate_lines(summary["estimate"]))
    for item in summary["warnings"]:
        lines.append(f"warning ({item['category']}): {item['message']}")
    return lines


def _duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.2g} s"
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _estimate_lines(estimate: dict) -> list[str]:
    """The estimate of time and memory, with where it comes from."""
    per_pass, total, memory = estimate["seconds_per_pass"], estimate["total_seconds"], estimate["peak_memory_gb"]
    reference = estimate["reference"]
    machine = f"{reference['cpu']} on {reference['threads']} threads"
    if estimate["scaled"]:
        timing = estimate["timing"]
        source = (
            f"scaled to this machine by a {timing['seconds']:.2g} s timing on {timing['threads']} thread(s) "
            f"(solve x{timing['solve_ratio']:.3g}, filter x{timing['filter_ratio']:.3g} of the reference, {machine})"
        )
    else:
        source = f"for the reference machine ({machine}), not scaled to this one"
    lines = [
        f"estimate, {source}:",
        f"  {per_pass['low']:.3g} to {per_pass['high']:.3g} s per pass (centre {per_pass['centre']:.3g}); "
        f"{estimate['passes']} passes in {_duration(total['low'])} to {_duration(total['high'])} "
        f"(centre {_duration(total['centre'])}); not counting the mesh of the part",
        f"  peak memory {memory['low']:.3g} to {memory['high']:.3g} GB (centre {memory['centre']:.3g})",
    ]
    if not estimate["within_measured"]:
        lines.append(
            "warning (estimate): outside what was measured, so less sure than its range: "
            + "; ".join(estimate["outside"])
        )
    return lines


def _run_summary(outcome: RunOutcome) -> dict:
    result = outcome.part.result
    bounds = outcome.part.passes[-1].bounds if outcome.part.passes else None
    return _plain(
        {
            "goal": result.goal,
            "limits": result.limits,
            "bounds": bounds,
            "limits_met": result.limits_met,
            "feasible": result.feasible,
            "stop_reason": result.stop_reason,
            "passes": len(outcome.part.passes),
            "threads": result.threads,
            "seconds": {"prepare": outcome.prepared.seconds, "optimise": outcome.seconds},
            "files": {kind: str(path) for kind, path in outcome.paths.items()},
            "mesh": _mesh_summary(outcome),
            "warnings": [w.as_dict() for w in outcome.warnings],
        }
    )


def _mesh_summary(outcome: RunOutcome) -> dict | None:
    if not outcome.prepared.problem_file.mesh:
        return None
    if outcome.finished is None:
        return {"written": False, "delivered": False, "refused": outcome.mesh_refused.as_dict()}
    finished = outcome.finished
    return {
        "written": True,
        "delivered": finished.delivered.within,
        "level": finished.level,
        "triangles": len(finished.mesh.triangles),
        "mass_g": finished.mass_g,
        "compliance_difference": finished.delivered.goal_difference,
        "changed_share": finished.delivered.changed_share,
        "stl_refused": outcome.stl_refused,
        "seconds": outcome.mesh_seconds,
    }


def _pass_line(step: str, number: int, record) -> str:
    limits = ", ".join(f"{value:.4g} (at most {bound:.4g})" for value, bound in zip(record.limits, record.bounds))
    return f"{step:11} pass {number:3}, beta {record.beta:4g}: compliance {record.goal:.5g} N mm, limits {limits}"


def _prepared(args, timing: bool = False) -> Prepared:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # kept in Prepared.warnings and printed from there
        return _prepare(args.file, args.threads, timing)


def _check(args) -> int:
    try:
        prepared = _prepared(args, timing=not args.no_timing)
    except EngineError as error:
        return _refused(args, "check", error)
    summary = prepared.summary()
    if args.picture is not None:
        try:
            summary["picture"] = str(write_picture(prepared, args.picture))
        except OSError as error:
            refused = OutOfRange(
                f"the picture {args.picture!r} cannot be written: {error.strerror or error}", "", args.file
            )
            return _refused(args, "check", refused)
    if args.json:
        print(json.dumps({"command": "check", "exit": 0, "prepared": summary}, allow_nan=False))
    else:
        print("\n".join(_check_lines(summary)))
        if args.picture is not None:
            print(f"picture of the set-up written to {summary['picture']}")
    return 0


def _refused(args, command: str, error: EngineError, status: int = REFUSED) -> int:
    data = _error(error, args.file)
    if args.json:
        print(json.dumps({"command": command, "exit": status, "error": _plain(data)}, allow_nan=False))
    else:
        _print_error(data)
    return status


def _run(args) -> int:
    try:
        prepared = _prepared(args)
    except EngineError as error:
        return _refused(args, "run", error)
    out = sys.stderr if args.json else sys.stdout
    if not args.json:
        print("\n".join(_check_lines(prepared.summary(), with_estimate=False)), file=out)

    def show(step, number, record):
        print(_pass_line(step, number, record), file=out, flush=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # kept in RunOutcome.warnings and printed from there
            outcome = _optimise(prepared, args.out, show)
    except EngineError as error:
        status = REFUSED if isinstance(error, _Refused) else ENGINE_FAILED
        return _refused(args, "run", error, status)
    summary = _run_summary(outcome)
    status = LIMITS_MET if outcome.limits_met and outcome.mesh_delivered else LIMITS_NOT_MET
    if args.json:
        print(json.dumps({"command": "run", "exit": status, "result": summary}, allow_nan=False))
        return status
    new = outcome.warnings[len(prepared.warnings) :]
    for item in new:
        print(f"warning ({item.category.__name__}): {item.message}")
    limits = ", ".join(
        f"{value:.6g} (at most {bound:.6g}, {'met' if met else 'NOT met'})"
        for value, bound, met in zip(summary["limits"], summary["bounds"], summary["limits_met"])
    )
    print(f"compliance {summary['goal']:.8g} N mm; limits (mass g, member size) {limits}")
    print(
        f"{summary['passes']} passes in {summary['seconds']['optimise']:.1f} s on {summary['threads']} thread(s); "
        f"stopped by {summary['stop_reason']}"
    )
    mesh = summary["mesh"]
    if mesh is not None and mesh["written"]:
        print(
            f"mesh: level {mesh['level']:.4g}, {mesh['triangles']} triangles, {mesh['mass_g']:.6g} g; mesh voxelised "
            f"back: compliance {100 * mesh['compliance_difference']:+.2f} %, {100 * mesh['changed_share']:.2f} % of "
            f"the voxels change side ({'within' if mesh['delivered'] else 'NOT within'} "
            f"{100 * DELIVERED_TOLERANCE:g} %)"
        )
        if mesh["stl_refused"] is not None:
            kept = " (part.3mf is written)" if "3mf" in summary["files"] else ""
            print(f"part.stl not written: {mesh['stl_refused']}{kept}")
    elif mesh is not None:
        print(f"mesh refused, not written: {mesh['refused']['message']}")
    print("written: " + ", ".join(summary["files"].values()))
    return status


def main(argv: list[str] | None = None) -> int:
    """Run the ``michell`` command (``check``, ``run``, ``--version``) and return its exit status."""
    parser = argparse.ArgumentParser(
        prog="michell",
        description="Generative design by topology optimisation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"michell {__version__} (engine {engine_version()})",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    check = commands.add_parser("check", help="read and check a problem file and set it up; solve nothing")
    run = commands.add_parser("run", help="run a problem file and write the outcome")
    for command in (check, run):
        command.add_argument("file", help="the problem file (TOML)")
        command.add_argument("--threads", type=int, metavar="N", help="threads (default: the file's [run] threads)")
        command.add_argument("--json", action="store_true", help="print one JSON object, nothing else")
    check.add_argument(
        "--no-timing",
        action="store_true",
        help="skip the short timing that scales the estimate to this machine (give the reference machine's)",
    )
    check.add_argument("--picture", metavar="SVG", help="write a picture of the set-up to this SVG file")
    run.add_argument("--out", metavar="DIR", help="the output folder (default: the file's [output] folder)")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _check(args) if args.command == "check" else _run(args)
    except Exception as error:  # a fault of michell itself, or of the machine (a disk that is full)
        data = {
            "code": "internal_error",
            "path": "",
            "file": args.file,
            "message": f"{type(error).__name__}: {error}",
            "details": {},
        }
        if args.json:
            print(json.dumps({"command": args.command, "exit": ENGINE_FAILED, "error": data}))
        else:
            _print_error(data)
        return ENGINE_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
