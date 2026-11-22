"""Verification row 33: the problem file (``michell.problem_file``) and the ``michell run`` command, on the
example bridge written as a problem file (``examples/bridge_file/``) against the same bridge built by hand
(``examples/bridge_3d_member_size.py``, as row 32 builds it).

The example's folder is not part of the installed package: the rows take the folder that holds
``bridge_file/`` (in the repository: ``examples``) and report "not run" when it is absent.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from michell._engine import EngineError
from michell.export import read_result
from michell.problem_file import prepare
from michell.recipes import stiff_part
from michell.row import INDEPENDENT_COMPUTATION, DESIGN_PROPERTY, Row, checked, g3, not_run
from michell.verification.attach_rows import ROW_32_SIZE, example_bridge

# ---------------------------------------------------------------------------------------------------------
# Row 33. Limits, each on its own line, fixed before any run:
# (a) the example bridge prepared from its problem file (examples/bridge_file/bridge.toml, michell.prepare):
#     the number of entries where the active voxels, the held-solid mask, the held degrees of freedom and
#     the load vector differ from the hand-built example's, each 0; the difference of the mass bound from the
#     example's (g), 0; the number of faces where the density filter's faces differ, 0;
# (b) (slow) `michell run` on the file from the command line, on one thread: the compliance in result.json,
#     read back with read_result, equals that of the hand-built example's stiff_part run on the same thread
#     count to 1e-9 relative; the largest relative difference of the limit values (mass, member-size value)
#     in result.json from the example's, at most 1e-9; the number of entries of limits_met that differ, 0;
# (c) a copy of the example file with one bad edit for each kind of fault is refused with that fault's code
#     (each line 0 when refused with the expected code, 1 otherwise).
#
# The example scripts state the mass bound as 0.30 times the mass of the whole block; the file states
# max_fraction = 0.30, which prepare turns into the same product.

ROW_33_FILE = Path("bridge_file") / "bridge.toml"
ROW_33_THREADS = 1

# (c): the expected code, the text of the example file replaced, and what replaces it.
ROW_33_EDITS = (
    ("invalid_toml", "[grid]", "[grid"),
    ("unknown_schema", 'schema = "michell/0"', 'schema = "michell/1"'),
    ("unknown_key", "poisson_ratio = 0.3", "poissons_ratio = 0.3"),
    ("missing_key", "voxel_mm = 1.0\n", "\n"),
    ("wrong_type", "voxel_mm = 1.0", 'voxel_mm = "1.0"'),
    ("out_of_range", "poisson_ratio = 0.3", "poisson_ratio = 0.5"),
    ("unknown_name", 'on = "clamp_high"', 'on = "clamp_hihg"'),
    ("duplicate_name", 'load_pad = "load_pad.stl"', 'pad = "load_pad.stl"'),
    ("file_not_found", 'design = "envelope.stl"', 'design = "envelop.stl"'),
    ("not_available_yet", "size_mm = 5.0\n", 'size_mm = 5.0\n\n[[rules]]\nmodule = "symmetry"\nplanes = ["y"]\n'),
)


def _example(examples) -> Path | None:
    if examples is None:
        return None
    path = Path(examples) / ROW_33_FILE
    return path if path.is_file() else None


def row_33_problem_file(examples: Path | str | None = None) -> list[Row]:
    """Lines (a) and (c), from the problem file under ``examples`` (the folder that holds ``bridge_file/``)."""
    file = _example(examples)
    names = [
        "active voxels",
        "held-solid voxels",
        "held degrees of freedom",
        "entries of the load vector",
    ]
    if file is None:
        rows = [
            not_run(
                33,
                f"Problem file: the example bridge prepared from its file, {name} that differ from the "
                "hand-built example's",
                INDEPENDENT_COMPUTATION,
                0.0,
                "example files not found",
            )
            for name in names
        ]
        rows.append(
            not_run(
                33,
                "Problem file: the example bridge, mass bound against the hand-built example's (g)",
                INDEPENDENT_COMPUTATION,
                0.0,
                "example files not found",
            )
        )
        rows.append(
            not_run(
                33,
                "Problem file: the example bridge, faces of the density filter that differ from the "
                "hand-built example's",
                INDEPENDENT_COMPUTATION,
                0.0,
                "example files not found",
            )
        )
        rows += [
            not_run(
                33,
                f"Problem file: a copy of the example with one bad edit is refused with {code}",
                DESIGN_PROPERTY,
                0.0,
                "example files not found",
            )
            for code, _, _ in ROW_33_EDITS
        ]
        return rows

    rows = []
    prepared = prepare(file, threads=ROW_33_THREADS)
    example = example_bridge(ROW_33_THREADS)
    attached = prepared.set_up
    counts = [
        np.count_nonzero((np.asarray(attached.problem.grid.active) == 1.0) != (np.asarray(example.grid.active) == 1.0)),
        np.count_nonzero(attached.keep_solid.mask != example.keep_solid.mask),
        np.count_nonzero(attached.held != example.held),
        np.count_nonzero(attached.problem.cases[0].loads != example.loads),
    ]
    for name, count in zip(names, counts):
        rows.append(
            checked(
                33,
                f"Problem file: the example bridge prepared from its file, {name} that differ from the hand-built "
                "example's",
                INDEPENDENT_COMPUTATION,
                float(count),
                0.0,
                f"{attached.problem.grid.n_elements} voxels from STL regions and marker bodies; "
                f"{int(attached.keep_solid.mask.sum())} held solid, {int(attached.held.sum())} held degrees of freedom",
            )
        )
    rows.append(
        checked(
            33,
            "Problem file: the example bridge, mass bound against the hand-built example's (g)",
            INDEPENDENT_COMPUTATION,
            abs(prepared.mass_g - example.mass_g),
            0.0,
            f"max_fraction 0.30 gives {prepared.mass_g!r} g, the example {example.mass_g!r} g",
        )
    )
    differ = [face for face in example.outside if attached.outside[face] != example.outside[face]]
    rows.append(
        checked(
            33,
            "Problem file: the example bridge, faces of the density filter that differ from the hand-built example's",
            INDEPENDENT_COMPUTATION,
            float(len(differ)),
            0.0,
            ", ".join(f"{face} {word}" for face, word in attached.outside.items()),
        )
    )

    text = file.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as folder:
        for source in file.parent.glob("*.stl"):
            shutil.copy(source, folder)
        for code, old, new in ROW_33_EDITS:
            if text.count(old) != 1:
                raise ValueError(f"the edit for {code} needs {old!r} exactly once in {file}")
            edited = Path(folder) / f"{code}.toml"
            edited.write_text(text.replace(old, new), encoding="utf-8")
            try:
                prepare(edited, threads=ROW_33_THREADS)
                value, note = 1.0, "not refused"
            except EngineError as error:
                got = getattr(error, "code", "")
                value = 0.0 if got == code else 1.0
                where = getattr(error, "path", "")
                edit = f"{old.strip()!r} written {new.strip()!r}" if new.strip() else f"{old.strip()!r} removed"
                note = f"{edit}: {got}" + (f" at {where}" if where else "")
            rows.append(
                checked(
                    33,
                    f"Problem file: a copy of the example with one bad edit is refused with {code} (0 when refused "
                    "with it)",
                    DESIGN_PROPERTY,
                    value,
                    0.0,
                    note,
                )
            )
    return rows


def row_33_run(examples: Path | str | None = None, threads: int = ROW_33_THREADS) -> list[Row]:
    """Line (b): ``michell run`` on the example's problem file from the command line against the hand-built
    example's ``stiff_part`` run, on ``threads`` threads."""
    file = _example(examples)
    names = [
        "compliance in result.json against the hand-built example's",
        "largest relative difference of the limit values in result.json from the hand-built example's",
        "entries of limits_met in result.json that differ from the hand-built example's",
    ]
    limits = [1e-9, 1e-9, 0.0]
    if file is None:
        return [
            not_run(
                33,
                f"Problem file: michell run on the example bridge, {name}",
                INDEPENDENT_COMPUTATION,
                limit,
                "example files not found",
            )
            for name, limit in zip(names, limits)
        ]
    example = example_bridge(threads)
    reference = stiff_part(example.problem, example.mass_g, ROW_32_SIZE, example.keep_solid, outside=example.outside)
    expected = reference.result
    with tempfile.TemporaryDirectory() as folder:
        command = [
            sys.executable,
            "-m",
            "michell.cli",
            "run",
            str(file),
            "--out",
            folder,
            "--threads",
            str(threads),
            "--json",
        ]
        finished = subprocess.run(command, capture_output=True, text=True, check=False)
        if finished.returncode not in (0, 1):
            note = f"michell run exited with {finished.returncode}: {finished.stdout.strip()[-300:]}"
            return [
                checked(
                    33,
                    f"Problem file: michell run on the example bridge, {name}",
                    INDEPENDENT_COMPUTATION,
                    math.inf,
                    limit,
                    note,
                )
                for name, limit in zip(names, limits)
            ]
        reply = json.loads(finished.stdout)
        result = read_result(Path(folder) / "result")
    compliance = result["goal"]
    limit_difference = max(
        abs(value / reference_value - 1.0) for value, reference_value in zip(result["limits"], expected.limits)
    )
    met_differ = sum(1 for a, b in zip(result["limits_met"], expected.limits_met) if a != b)
    met_differ += abs(len(result["limits_met"]) - len(expected.limits_met))
    stated = f"result.json format {result['format']}, goal {result['set_up']['goal']['module']}"
    measured = [abs(compliance / expected.goal - 1.0), limit_difference, float(met_differ)]
    notes = [
        f"{g3(compliance)} against {g3(expected.goal)} N mm, {threads} thread(s), {len(result['passes'])} passes; "
        f"exit status {finished.returncode} ({reply['exit']}); {stated}",
        f"mass {result['limits'][0]!r} against {expected.limits[0]!r} g, member-size value {result['limits'][1]!r} "
        f"against {expected.limits[1]!r}",
        f"limits_met {result['limits_met']} against {list(expected.limits_met)}",
    ]
    return [
        checked(
            33,
            f"Problem file: michell run on the example bridge, {name}",
            INDEPENDENT_COMPUTATION,
            value if math.isfinite(value) else math.inf,
            limit,
            note,
        )
        for name, value, limit, note in zip(names, measured, limits, notes)
    ]


__all__ = ["ROW_33_EDITS", "ROW_33_FILE", "row_33_problem_file", "row_33_run"]
