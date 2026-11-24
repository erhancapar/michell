"""``python -m michell.estimate``: fits the estimate of time and memory to the measurements of
``python -m michell.performance`` and writes the model the package ships.

    python -m michell.estimate fit docs/performance.json --out python/michell/estimate/model.json [--timing]

``--timing`` also times the fixed workload here, on the measurements' thread count, and stores it as the
reference's timing; it belongs on the machine that made the measurements. Without it the stored timing is kept.

    python -m michell.estimate load-cases [--size 90] [--cases 3] [--passes 4] [--threads 12] [--store MODEL]

measures the memory of each load case beyond the first on the cantilever of ``michell.performance`` (one run
with one load case and one with ``--cases``, each in a fresh process) and prints it; ``--store`` writes it into
the model file as its ``extra_load_case``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from michell.estimate import MODEL_PATH, load_model, measure_extra_load_case, timing, write_model


def _fit(args) -> int:
    threads = json.loads(args.measurements.read_text(encoding="utf-8"))["machine"]["threads"]
    extra = None
    if args.extra_load_case:
        extra = {"gb_per_million_voxels": float(args.extra_load_case[0]), "measured": args.extra_load_case[1]}
    model = write_model(args.measurements, args.out, timing(threads) if args.timing else None, extra)
    for row in model["sizes"]:
        print(
            f"{row['elements']:>9} elements: {row['seconds_per_pass']:.3g} s per pass measured, "
            f"{row['fitted_seconds_per_pass']:.3g} fitted; {row['peak_memory_gb']:.3g} GB measured, "
            f"{row['fitted_peak_memory_gb']:.3g} fitted"
        )
    print(f"timing: {model['timing']}")
    print(f"extra load case: {model['extra_load_case']}")
    print(f"written to {args.out}")
    return 0


def _load_cases(args) -> int:
    extra = measure_extra_load_case(args.size, args.cases, args.passes, args.threads)
    for run in extra["runs"]:
        peaks = ", ".join(f"{gb:.4g}" for gb in run["peak_after_each_pass_gb"])
        print(f"{run['cases']} load case(s), {run['elements']} voxels: peak after each pass {peaks} GB")
    print(f"each further load case: {extra['gb_per_million_voxels']:.4g} GB per million voxels")
    if args.store is not None:
        if not extra["gb_per_million_voxels"] > 0:
            print("not stored: a size this small measures the interpreter, not the load cases; take a larger size")
            return 1
        model = load_model(args.store)
        model["extra_load_case"] = {key: extra[key] for key in ("gb_per_million_voxels", "measured")}
        args.store.write_text(json.dumps(model, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"stored in {args.store}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m michell.estimate", description="Fit the estimate of time and memory to the measurements."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="fit the model to michell.performance's measurements and write it")
    fitting.add_argument("measurements", type=Path, help="the JSON file of python -m michell.performance run")
    fitting.add_argument("--out", type=Path, default=MODEL_PATH, help="the model file to write")
    fitting.add_argument(
        "--timing", action="store_true", help="time the workload here as the reference's (else keep the stored one)"
    )
    fitting.add_argument(
        "--extra-load-case",
        nargs=2,
        metavar=("GB_PER_MILLION_VOXELS", "HOW"),
        help="the memory of each load case beyond the first and how it was measured (else keep the stored one)",
    )
    fitting.set_defaults(run=_fit)
    cases = commands.add_parser("load-cases", help="measure the memory of each load case beyond the first")
    cases.add_argument("--size", type=int, default=90, help="the cantilever's size (michell.performance)")
    cases.add_argument("--cases", type=int, default=3, choices=(2, 3), help="the load cases of the second run")
    cases.add_argument("--passes", type=int, default=4, help="the passes of stiff_part to run")
    cases.add_argument("--threads", type=int, default=12, help="the threads to run on")
    cases.add_argument("--store", type=Path, help="the model file to store the result in")
    cases.set_defaults(run=_load_cases)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
