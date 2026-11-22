"""Fields and results written to files that other programs read.

``write_vtk(path, grid, **fields)`` writes per-element fields (a density, a nominal design, a mask) of a 2D
``Grid`` or a 3D ``Grid3d`` as one file in the legacy VTK format (``.vtk``), which ParaView and VisIt open:
a ``STRUCTURED_POINTS`` data set whose points are the nodes of the grid and whose cells are its elements,
with the fields as ``CELL_DATA``. It needs no package beyond NumPy.

``write_result(path, result, *, record=None)`` writes a ``Result`` of ``optimise`` or a ``StiffPart`` of
``stiff_part`` as two files beside each other, ``<path>.npz`` (every array, NumPy's own format) and
``<path>.json`` (everything else; with ``record``, the record of a run from a problem file as well);
``read_result(path)`` reads both back into a plain dictionary, whichever format the JSON file has. Either
file is read by ``numpy.load`` or ``json.load`` alone.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from importlib.metadata import version

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue
from michell.optimise import Result
from michell.recipes import StiffPart

TITLE = "michell fields"
RESULT_FORMAT = 1
# The format of a result written with the record of a run from a problem file (``michell.run``).
RUN_RESULT_FORMAT = 2


def write_vtk(path: str | os.PathLike, grid, **fields) -> None:
    """Write ``fields`` (each one value per element, in element order) of ``grid`` to ``path`` as a legacy VTK
    file, version 3.0, binary.

    The file holds, in this order: the line ``# vtk DataFile Version 3.0``, the title ``michell fields``,
    ``BINARY``, ``DATASET STRUCTURED_POINTS``, ``DIMENSIONS nx+1 ny+1 nz+1`` (the nodes; ``ny+1 1`` on a 2D
    grid), ``ORIGIN 0 0 0``, ``SPACING h h h`` (mm), ``CELL_DATA n`` (the number of elements) and
    ``FIELD FieldData m`` (the number of fields); then for each field, in the order given, ``<name> 1 n
    double``, followed by its n values as 8-byte floating-point numbers, big-endian as the format
    prescribes, x running fastest (the element order of the grid), and a line end. Every line of text ends
    with a line feed. The fields are field arrays rather than ``SCALARS`` blocks: VTK's own reader of the
    format reads every field array, but only the first ``SCALARS`` block unless told otherwise.

    A field is any array of numbers or booleans (written as 0 and 1) with one entry per element; its name
    (the keyword) is how ParaView lists it: ``write_vtk("bridge.vtk", grid, density=part.nominal,
    held=keep_solid.mask)``. At least one field is needed. An existing file at ``path`` is replaced.
    """
    if isinstance(grid, _engine.Grid3d):
        points = (grid.nx + 1, grid.ny + 1, grid.nz + 1)
    elif isinstance(grid, _engine.Grid):
        points = (grid.nx + 1, grid.ny + 1, 1)
    else:
        raise InvalidType(f"write_vtk takes a michell.Grid or michell.Grid3d; got {type(grid).__name__}")
    if not fields:
        raise InvalidValue("write_vtk needs at least one field, given by name: write_vtk(path, grid, density=rho)")
    blocks = []
    for name, values in fields.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise InvalidValue(f"a field name is a word of letters, digits and underscores; got {name!r}")
        array = np.asarray(values)
        if array.dtype != np.bool_ and not np.issubdtype(array.dtype, np.number):
            raise InvalidType(f"field {name!r} must hold numbers or booleans; got an array of {array.dtype}")
        if np.iscomplexobj(array):
            raise InvalidType(f"field {name!r} must hold real numbers; got complex ones")
        if array.shape != (grid.n_elements,):
            raise InvalidValue(
                f"field {name!r} needs one value per element ({grid.n_elements}), as a flat array; got shape "
                f"{array.shape}"
            )
        header = f"{name} 1 {grid.n_elements} double\n".encode("ascii")
        blocks.append(header + array.astype(">f8").tobytes() + b"\n")
    h = repr(float(grid.h))
    head = (
        f"# vtk DataFile Version 3.0\n{TITLE}\nBINARY\nDATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {points[0]} {points[1]} {points[2]}\nORIGIN 0 0 0\nSPACING {h} {h} {h}\n"
        f"CELL_DATA {grid.n_elements}\nFIELD FieldData {len(blocks)}\n"
    ).encode("ascii")
    with open(path, "wb") as file:
        file.write(head + b"".join(blocks))


def _plain(value):
    """``value`` as plain JSON: tuples and arrays as lists, NumPy numbers as Python numbers, and every number
    that is not finite (NaN, infinity) as ``None``."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _files(path: str | os.PathLike) -> tuple[str, str]:
    base = os.fspath(path)
    return base + ".npz", base + ".json"


def write_result(path: str | os.PathLike, result: Result | StiffPart, *, record: dict | None = None) -> None:
    """Write ``result``, a ``Result`` (``optimise``) or a ``StiffPart`` (``stiff_part``), as two files beside
    each other: ``<path>.npz`` with every array and ``<path>.json`` with everything else. ``path`` is given
    without an extension; existing files of those names are replaced. ``record``, a dictionary of plain data,
    adds its keys to the JSON file, which then has format 2 (below).

    The arrays (``numpy.savez``, uncompressed, the values bit for bit as held): ``x`` and ``density``, and for
    each named design ``design_<name>`` (its field after the last pass). A ``StiffPart`` adds ``filtered``,
    ``nominal``, ``first_step_nominal`` (the nominal design at the end of the robust step) and ``thresholds``
    (the member-size limit's per-element thresholds); its ``x``, ``density`` and designs are those of its
    ``result``.

    The JSON file is one flat object with these keys:

    - ``format``: 1, the version of this layout;
    - ``kind``: ``"Result"`` or ``"StiffPart"``;
    - ``version``: the version of the michell package that wrote the file;
    - ``goal``: the goal's value on the returned design;
    - ``limits``: the limits' values on the returned design, in the order of the run's limits;
    - ``bounds``: the limits' bounds (``at_most``), in the same order, or ``null`` when the object does not
      hold them (a ``Result`` holds the values of its limits but not the limits themselves);
    - ``limits_met``: for each limit whether it was met; ``feasible``: whether every limit was met;
    - ``stop_reason``: ``"passes"``, ``"change"`` or ``"error"`` (``Result.stop_reason``);
    - ``threads``: the number of threads of the run;
    - ``designs``: the names of the named designs, whose arrays are ``design_<name>`` (empty without them);
    - ``passes``: the record of every pass, a list of objects with the fields of ``Pass`` (``number``,
      ``goal``, ``limits``, ``change``, ``solver_steps``, ``design_solver_steps``) for a ``Result``, and with
      those of ``RecipePass`` (``step``, ``number``, ``beta``, ``goal``, ``limits``, ``bounds``, ``change``)
      for a ``StiffPart``, both of its steps in order;
    - for a ``StiffPart`` also ``eps``, ``threshold``, ``value_at_switch`` and ``scale`` (as in ``StiffPart``).

    ``goal``, ``limits`` and ``limits_met`` are ``null`` where the ``Result`` holds ``None`` (a partial result
    whose design could not be evaluated). Every number that is not finite is written as ``null``, so the file
    is strict JSON.

    Format 2, written with ``record`` (``michell.run`` writes it), has every key of format 1, with ``format``
    2, and these new keys, the record of a run from a problem file:

    - ``problem_file``: the problem file's ``name`` and ``sha256`` (of its bytes);
    - ``geometry``: each geometry file, with its ``key`` in the problem file, ``role``, ``name``, ``file``
      (the path as written) and ``sha256``;
    - ``grid``: ``counts``, ``voxel_mm``, ``origin_mm`` (the grid's corner in the meshes' coordinates) and
      the numbers of voxels, active voxels, voxels held solid (with and without the skins) and design voxels;
    - ``volumes``: the volume table of the voxel model (``michell.regions.VolumeRow``, one object per line);
    - ``set_up``: the set-up report (``michell.problem_file.Prepared.summary``): bodies, material, goal,
      mass limit, member size, each support and load (what it acts on, its faces or voxels, nodes, point of
      action, force and the share dropped on held degrees of freedom, the held skin), the density filter's
      faces and the thread count;
    - ``warnings``: each warning given (``category`` and ``message``);
    - ``files``: the names of the files written beside the JSON file, the JSON file among them;
    - ``seconds``: the seconds the preparation and the optimisation took.

    A key of ``record`` that format 1 already has is refused."""
    if isinstance(result, StiffPart):
        part, run = result, result.result
    elif isinstance(result, Result):
        part, run = None, result
    else:
        raise InvalidType(f"write_result takes a michell.Result or a michell.StiffPart; got {type(result).__name__}")
    arrays = {"x": run.x, "density": run.density}
    for name, field in run.designs.items():
        arrays[f"design_{name}"] = field
    summary = {
        "format": RESULT_FORMAT,
        "kind": "Result" if part is None else "StiffPart",
        "version": version("michell"),
        "goal": run.goal,
        "limits": run.limits,
        "bounds": None,
        "limits_met": run.limits_met,
        "feasible": run.feasible,
        "stop_reason": run.stop_reason,
        "threads": run.threads,
        "designs": list(run.designs),
        "passes": [asdict(item) for item in run.passes],
    }
    if part is not None:
        arrays.update(filtered=part.filtered, nominal=part.nominal, first_step_nominal=part.first_step_nominal)
        if part.member_size.thresholds is not None:
            arrays["thresholds"] = part.member_size.thresholds
        if part.passes:
            summary["bounds"] = part.passes[-1].bounds
        summary["passes"] = [asdict(item) for item in part.passes]
        summary.update(eps=part.eps, threshold=part.threshold, value_at_switch=part.value_at_switch, scale=part.scale)
    if record is not None:
        if not isinstance(record, dict):
            raise InvalidType(f"record is a dict of plain data; got {type(record).__name__}")
        taken = sorted(set(record) & set(summary))
        if taken:
            raise InvalidValue(f"record may not replace the keys of format {RESULT_FORMAT}; got {taken}")
        summary["format"] = RUN_RESULT_FORMAT
        summary.update(record)
    arrays_path, summary_path = _files(path)
    with open(arrays_path, "wb") as file:
        np.savez(file, **{name: np.asarray(values) for name, values in arrays.items()})
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(_plain(summary), file, indent=2, allow_nan=False)
        file.write("\n")


def read_result(path: str | os.PathLike) -> dict:
    """Read the two files that ``write_result`` wrote for ``path`` (given without an extension) and return a
    plain dictionary: the content of ``<path>.json`` (format 1 or 2; the keys are listed in ``write_result``),
    with the arrays of ``<path>.npz`` added under the key ``"arrays"``, a dictionary from each array's name to the
    array. It does not rebuild ``Result`` or ``StiffPart`` objects: the responses, filters and limits of the
    run are not in the files."""
    arrays_path, summary_path = _files(path)
    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)
    with np.load(arrays_path, allow_pickle=False) as data:
        summary["arrays"] = {name: data[name] for name in data.files}
    return summary


__all__ = ["RESULT_FORMAT", "RUN_RESULT_FORMAT", "TITLE", "read_result", "write_result", "write_vtk"]
