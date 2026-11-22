"""``python -m michell.performance``: measures the run time and the peak memory of ``stiff_part``, and writes
the page from the measurements.

    python -m michell.performance run --sizes 36 46 58 72 90 --threads 12 --out performance.json
    python -m michell.performance render performance.json --out docs/performance.md

``run`` writes the JSON after every size and ends with status 0 when every size completed, 1 when any failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from michell.performance import SIZES, THREADS, failed, run
from michell.performance.page import render


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report(one: dict) -> None:
    error = one.get("error")
    if error:
        if error["pass"] is None:
            what = f"the process ended with status {error['exit_status']} without giving its measurements"
        else:
            name = error["code"] or error["type"]
            what = f"stopped in pass {error['pass']} ({error['step']} step) with {name}: {error['message']}"
        print(f"size {one['size']:>3} ({one['elements']:,} elements) failed: {what}", flush=True)
        return
    seconds = one["seconds_per_pass"]["mean"]
    print(
        f"size {one['size']:>3} ({one['elements']:,} elements): {one['passes']} passes in "
        f"{one['total_seconds']:.1f} s, {seconds:.3g} s per pass, peak memory {one['peak_memory_gb']} GB",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m michell.performance",
        description="Measure the run time and the peak memory of michell.recipes.stiff_part, and write the page.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("run", help="measure each size in a process of its own and write the JSON")
    measuring.add_argument(
        "--sizes", type=int, nargs="+", default=list(SIZES), help="sizes m: grids of 2m x m x m elements"
    )
    measuring.add_argument("--threads", type=int, default=THREADS, help="threads of the solves, filter and updates")
    measuring.add_argument(
        "--passes-scale", type=float, default=1.0, help="multiply every pass count of the schedule (for short runs)"
    )
    measuring.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    rendering = commands.add_parser("render", help="write the Markdown page from the JSON")
    rendering.add_argument("measurements", type=Path, help="the JSON file written by run")
    rendering.add_argument("--out", type=Path, help="write the page here (default: print it)")
    args = parser.parse_args(argv)

    if args.command == "run":
        data = run(args.sizes, args.threads, args.passes_scale, report=_report, out=args.out)
        print(f"written to {args.out}")
        return 1 if failed(data) else 0
    page = render(json.loads(args.measurements.read_text(encoding="utf-8")))
    if args.out is None:
        print(page, end="")
    else:
        _write(args.out, page)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
