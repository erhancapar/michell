"""A 3D cantilever: the stiffest block that uses at most 15 % of its material, and a picture of the result.

A block of 64 by 32 by 32 cube elements of side 1 mm (steel), clamped on its face x = 0, with a line load of
1000 N down along the lower edge of its free end (x = 64, z = 0). Least compliance with at most 15 % of the
block's mass, the Helmholtz filter of radius 2.5 element sides, the method of moving asymptotes, 120 passes.
Run from the repository:

    python examples/cantilever_3d.py [--nx 64 --ny 32 --nz 32] [--passes 120] [--threads 1] [--out picture.png]
                                     [--robust [--spread 0.25]]

With `--robust` the block is optimised with the robust formulation of Wang, Lazarov and Sigmund ("On
projection methods, convergence and robust formulations in topology optimization", Struct. Multidisc. Optim.
43, 2011): the filtered field passes through three projections at the thresholds 0.5 + spread, 0.5 and
0.5 - spread (the eroded, nominal and dilated designs), the goal is the compliance of the eroded design and
the mass limit sits on the dilated design, rescaled so that the nominal design meets the 15 %; the sharpness
beta doubles from 1 to 32 in six equal steps of the passes. The picture then shows the nominal design and is
written to docs/images/cantilever_3d_robust.png unless `--out` is given.

The picture (default: docs/images/cantilever_3d.png) shows three views side by side: the density summed along
y (seen from the side, z up), along z (seen from above) and along x (seen from the free end, z up), each
scaled so that its densest line is black. It is written by the few lines of `write_greyscale_png` below.
The last line printed is a JSON summary of the run.
"""

import argparse
import json
import struct
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


def three_views(density, nx, ny, nz, scale=4, gap=8):
    """The density summed along y, along z and along x, each scaled to grey (its densest line black, an
    empty line white), ``scale`` pixels per element side, side by side with white gaps."""
    field = density.reshape(nz, ny, nx)  # index [k, j, i]: x runs fastest
    views = [
        field.sum(axis=1)[::-1],  # along y: rows are z (up), columns x
        field.sum(axis=0)[::-1],  # along z: rows are y (up), columns x
        field.sum(axis=2)[::-1],  # along x: rows are z (up), columns y
    ]
    height = max(view.shape[0] for view in views) * scale
    pieces = []
    for view in views:
        grey = np.round(255 * (1.0 - view / max(view.max(), 1e-300))).astype(np.uint8)
        grey = np.repeat(np.repeat(grey, scale, axis=0), scale, axis=1)
        padded = np.full((height, grey.shape[1]), 255, dtype=np.uint8)
        padded[height - grey.shape[0] :] = grey
        pieces += [padded, np.full((height, gap), 255, dtype=np.uint8)]
    return np.hstack(pieces[:-1])


parser = argparse.ArgumentParser(description="Optimise a 3D cantilever and write a picture of the result.")
parser.add_argument("--nx", type=int, default=64)
parser.add_argument("--ny", type=int, default=32)
parser.add_argument("--nz", type=int, default=32)
parser.add_argument("--passes", type=int, default=120)
parser.add_argument("--threads", type=int, default=1, help="the number of threads of the solves and the filter")
parser.add_argument("--out", type=Path, help="the picture (default: docs/images/cantilever_3d.png, or ..._robust.png)")
parser.add_argument("--robust", action="store_true", help="the robust formulation: eroded, nominal, dilated")
parser.add_argument("--spread", type=float, default=0.25, help="with --robust: the thresholds are 0.5 -+ spread")
args = parser.parse_args()
picture_name = "cantilever_3d_robust.png" if args.robust else "cantilever_3d.png"
out = args.out or Path(__file__).parent.parent / "docs" / "images" / picture_name
nx, ny, nz = args.nx, args.ny, args.nz

# The block: nx by ny by nz cube elements of side 1 mm.
grid = michell.Grid3d(nx, ny, nz, 1.0)

# Clamped on the face x = 0: every degree of freedom of its nodes held.
held = np.zeros(grid.n_dofs, dtype=bool)
for k in range(nz + 1):
    for j in range(ny + 1):
        node = grid.node(0, j, k)
        held[3 * node : 3 * node + 3] = True

# 1000 N down (-z) along the lower edge of the free end, as consistent nodal forces of a uniform line load.
loads = np.zeros(grid.n_dofs)
for j in range(ny + 1):
    share = 0.5 if j in (0, ny) else 1.0
    loads[3 * grid.node(nx, j, 0) + 2] = -1000.0 / ny * share

steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_mm3=7.85e-3)
problem = michell.Problem(grid, held, steel, [michell.LoadCase("tip", loads)], threads=args.threads)

# Least compliance with at most 15 % of the block's mass.
block_mass = steel.density_g_mm3 * grid.voxel_volume_mm3 * grid.n_elements
mass_limit = 0.15 * block_mass
pde_filter = michell.HelmholtzFilter(grid, 2.5, threads=args.threads)
started = time.perf_counter()
if not args.robust:
    result = michell.optimise(
        problem,
        goal=michell.Compliance(),
        limits=[michell.Limit(michell.Mass(), at_most=mass_limit)],
        filters=[pde_filter],
        optimiser=michell.MovingAsymptotes(),
        x=np.full(grid.n_elements, 0.15),
        passes=args.passes,
    )
    records, density = result.passes, result.density
else:
    # Continuation and rescaling, after Wang, Lazarov and Sigmund (2011) (the rescaling rule as
    # stated by Kumar and Langelaar, Mech. Mach. Theory, 2022, doi:10.1016/j.mechmachtheory.2022.104871):
    # beta doubles from 1 to 32 in six equal steps, each a new call of `optimise` from the x the step before
    # ended with. Before each step the bound on the mass of the dilated design is set to mass limit *
    # mass(dilated) / mass(nominal), both masses taken at the current x with the new beta, so that the
    # nominal design meets the mass limit.
    x = np.full(grid.n_elements, 0.15)
    records = []
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        designs = michell.robust_designs(beta, args.spread)
        filtered = pde_filter.forward(x)
        nominal_sum = designs["nominal"][0].forward(filtered).sum()
        dilated_sum = designs["dilated"][0].forward(filtered).sum()
        bound = mass_limit * dilated_sum / nominal_sum
        result = michell.optimise(
            problem,
            goal=michell.on("eroded", michell.Compliance()),
            limits=[michell.Limit(michell.on("dilated", michell.Mass()), at_most=bound)],
            filters=[pde_filter],
            optimiser=michell.MovingAsymptotes(),
            x=x,
            passes=args.passes // 6,
            designs=designs,
        )
        x = result.x
        records += result.passes
        print(
            f"beta {beta:4g}: bound on the dilated mass {bound:.4g} g, compliance of the eroded design "
            f"{result.passes[-1].goal:.4g} N mm"
        )
    density = result.designs["nominal"]
    for name in ("eroded", "nominal", "dilated"):  # each design solved once more for its compliance
        u, _ = problem.solve(result.designs[name], loads)
        print(f"{name:8}: compliance {michell.dot(loads, u):.4g} N mm")
seconds = time.perf_counter() - started
first, last = records[0], records[-1]
mass = steel.density_g_mm3 * grid.voxel_volume_mm3 * float(density.sum())
print(
    f"{len(records)} passes in {seconds:.1f} s on {result.threads} threads: compliance {first.goal:.4g} "
    f"-> {last.goal:.4g} N mm, mass{' of the nominal design' if args.robust else ''} {mass:.4g} g of at most "
    f"{mass_limit:.4g} g"
)

out.parent.mkdir(parents=True, exist_ok=True)
write_greyscale_png(out, three_views(density, nx, ny, nz))
print(f"picture written to {out}")
summary = {
    "shape": [nx, ny, nz],
    "robust": args.robust,
    "passes": len(records),
    "threads": result.threads,
    "seconds": seconds,
    "compliance_first": first.goal,
    "compliance_last": last.goal,
    "mass": mass,
    "mass_limit": mass_limit,
    "solver_steps": [p.solver_steps for p in records],
}
print(json.dumps(summary))
