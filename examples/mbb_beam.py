"""The MBB beam of the 88-line code on 150 by 50 elements, and a picture of the result.

Half of a simply supported beam with a load at mid-span, cut at the symmetry line: least compliance with
half of the design space filled, the density filter of radius 6 element sides, and the optimality-criteria
optimiser. Run from the repository:

    python examples/mbb_beam.py [picture.png]

The picture (default: docs/images/mbb_150x50.png) is greyscale, solid black and void white, four pixels per
element side, written by the few lines of `write_greyscale_png` below.
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

import michell


def write_greyscale_png(path, image):
    """Writes ``image`` (rows of 8-bit grey values, top row first) as a PNG file: the signature, a header
    chunk, one compressed data chunk and the end chunk, each chunk with its CRC (PNG specification, W3C)."""
    height, width = image.shape

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8 bits, greyscale, no interlace
    rows = b"".join(b"\x00" + bytes(row) for row in image)  # each row starts with filter type 0 (none)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


# Half of the MBB beam: 150 by 50 square elements of side 1 mm, thickness 1 mm.
nx, ny = 150, 50
grid = michell.Grid(nx, ny, 1.0, 1.0)

# x held on the left edge (the symmetry line), y held at the bottom-right corner (the support).
held = np.zeros(grid.n_dofs, dtype=bool)
for j in range(ny + 1):
    held[2 * grid.node(0, j)] = True
held[2 * grid.node(nx, 0) + 1] = True

# A unit load down at the top-left corner.
loads = np.zeros(grid.n_dofs)
loads[2 * grid.node(0, ny) + 1] = -1.0

material = michell.Material(youngs_modulus=1.0, poisson_ratio=0.3, density_g_cm3=1000.0)
problem = michell.Problem(grid, held, material, [michell.LoadCase("load", loads)], penalty=3.0, e_min=1e-9)

# Least compliance with at most half the material, until no design variable changes by more than 0.01.
result = michell.optimise(
    problem,
    goal=michell.Compliance(),
    limits=[michell.Limit(michell.Mass(), at_most=0.5 * nx * ny)],
    filters=[michell.DensityFilter(grid, 6.0)],
    optimiser=michell.OptimalityCriteria(move_limit=0.2),
    x=np.full(grid.n_elements, 0.5),
    passes=1000,
    stop_change=0.01,
)
print(f"{len(result.passes)} passes, compliance {result.goal:.2f} N mm")

# The density, one row of elements per image row with y up, solid black; four pixels per element side.
density = np.clip(result.density, 0.0, 1.0).reshape(ny, nx)[::-1]
grey = np.round(255 * (1.0 - density)).astype(np.uint8)
picture = np.repeat(np.repeat(grey, 4, axis=0), 4, axis=1)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "docs" / "images" / "mbb_150x50.png"
out.parent.mkdir(parents=True, exist_ok=True)
write_greyscale_png(out, picture)
print(f"picture written to {out}")
