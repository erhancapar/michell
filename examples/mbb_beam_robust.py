"""The MBB beam with the robust formulation on 180 by 60 elements, and a picture of its three designs.

The beam of `mbb_beam.py`, made robust against a part that comes out uniformly too thin or too thick (Wang,
Lazarov and Sigmund, "On projection methods, convergence and robust formulations in topology optimization",
Struct. Multidisc. Optim. 43, 2011). One set of design variables passes through the density filter and then
through three projections at the thresholds 0.75, 0.5 and 0.25: the eroded, the nominal and the dilated
design. The goal is the compliance of the eroded design, the weakest of the three; the mass limit sits on the
dilated design, with its bound set so that the nominal design has half the mass of the design space. The
method of moving asymptotes runs every step. Run from the repository:

    python examples/mbb_beam_robust.py [picture.png]

The picture (default: docs/images/mbb_robust.png) shows the eroded, the nominal and the dilated design one
above the other, solid black and void white, three pixels per element side, parted by grey bands.
"""

import struct
import sys
import time
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


# Half of the MBB beam: 180 by 60 square elements of side 1 mm, thickness 1 mm.
nx, ny = 180, 60
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

density_filter = michell.DensityFilter(grid, 0.04 * nx)  # radius 7.2 element sides
asked_mass = 0.5 * nx * ny  # half of the design space, in g (1 g/mm^3 on elements of 1 mm^3)
spread = 0.25

# Continuation and rescaling, after Wang, Lazarov and Sigmund (2011) (the rescaling rule as
# stated by Kumar and Langelaar, Mech. Mach. Theory, 2022, doi:10.1016/j.mechmachtheory.2022.104871): the
# sharpness beta of the projections doubles from 1 to 32, every 50 passes, each step a new call of
# `optimise` from the x the step before ended with. Before each step the bound on the mass of the dilated
# design is set to
#
#     asked mass * mass(dilated) / mass(nominal),
#
# both masses taken at the current x with the new beta, so that the nominal design is steered to the asked
# mass while the limit itself is written on the dilated design.
x = np.full(grid.n_elements, 0.5)
started = time.perf_counter()
for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    designs = michell.robust_designs(beta, spread)
    filtered = density_filter.forward(x)
    nominal_mass = designs["nominal"][0].forward(filtered).sum()
    dilated_mass = designs["dilated"][0].forward(filtered).sum()
    bound = asked_mass * dilated_mass / nominal_mass
    result = michell.optimise(
        problem,
        goal=michell.on("eroded", michell.Compliance()),
        limits=[michell.Limit(michell.on("dilated", michell.Mass()), at_most=bound)],
        filters=[density_filter],
        optimiser=michell.MovingAsymptotes(),
        x=x,
        passes=50,
        designs=designs,
    )
    x = result.x
    print(
        f"beta {beta:4g}: bound on the dilated mass {bound:.1f} g, compliance of the eroded design "
        f"{result.goal:.2f} N mm, mass of the nominal design {result.designs['nominal'].sum():.1f} g"
    )
seconds = time.perf_counter() - started

# The three designs after the last pass, each solved once more for its compliance.
for name in ("eroded", "nominal", "dilated"):
    rho = result.designs[name]
    u, _ = problem.solve(rho, loads)
    print(f"{name:8}: compliance {michell.dot(loads, u):.2f} N mm, mass {rho.sum():.1f} g")
nominal = result.designs["nominal"]
print(f"non-discreteness of the nominal design: {(4 * nominal * (1 - nominal)).mean():.4f}; {seconds:.1f} s")

# The three designs one above the other, eroded at the top, each with y up, solid black, grey between them.
pieces = []
for name in ("eroded", "nominal", "dilated"):
    density = np.clip(result.designs[name], 0.0, 1.0).reshape(ny, nx)[::-1]
    grey = np.round(255 * (1.0 - density)).astype(np.uint8)
    pieces += [np.repeat(np.repeat(grey, 3, axis=0), 3, axis=1), np.full((12, 3 * nx), 160, dtype=np.uint8)]
picture = np.vstack(pieces[:-1])
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "docs" / "images" / "mbb_robust.png"
out.parent.mkdir(parents=True, exist_ok=True)
write_greyscale_png(out, picture)
print(f"picture written to {out}")
