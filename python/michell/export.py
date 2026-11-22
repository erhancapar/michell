"""Fields written to files that other programs read.

``write_vtk(path, grid, **fields)`` writes per-element fields (a density, a nominal design, a mask) of a 2D
``Grid`` or a 3D ``Grid3d`` as one file in the legacy VTK format (``.vtk``), which ParaView and VisIt open:
a ``STRUCTURED_POINTS`` data set whose points are the nodes of the grid and whose cells are its elements,
with the fields as ``CELL_DATA``. It needs no package beyond NumPy.
"""

from __future__ import annotations

import os
import re

import numpy as np

from michell import _engine
from michell.errors import InvalidType, InvalidValue

TITLE = "michell fields"


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


__all__ = ["TITLE", "write_vtk"]
