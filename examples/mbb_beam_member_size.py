"""The MBB beam on 180 by 60 elements with a minimum member size of 7 element sides, and a picture of the beam
without and with the member-size limit.

The limit is the geometric constraint of Zhou, Lazarov, Wang and Sigmund, "Minimum length scale in topology
optimization by geometric constraints", Comput. Methods Appl. Mech. Engrg. 293 (2015) 266-282
(``michell.MemberSize``). One set of design variables passes through the density filter (radius R equal to
the member size, as in that paper's set-up); the filtered field is the design "filtered", which the limit
reads, and its projection at 0.5 is the design "nominal", whose compliance is the goal and whose mass is
limited to 0.35 of the design space. The same set-up is verification row 27 (with eps times 1 there). Run
from the repository:

    python examples/mbb_beam_member_size.py [picture.png]

The procedure, step by step (the paper switches the limit on only once a topology has formed, at a high
fixed sharpness of the projection, section 3):

1. Without the limit, the sharpness beta of the projection doubles from 1 to 32, every 50 passes, each step a
   new call of ``optimise`` from the x the step before ended with.
2. At beta 32, the member-size limit is added: eta_e = member_threshold(filter, 7) = 0.780 (the filtered
   value at the centre of a strip 7 elements wide), c = r^4 = 2401 with r = R / h = 7 (section 2.2) and the
   bound eps = 6e-8, read from the paper's figure 8 at r = 7. The limit's ``scale`` is its value at that
   moment, so that MMA first sees it as a relative excess of about 1 (with the default scale, |eps|, it would
   see an excess of thousands, and MMA's subproblem does not converge). 300 passes follow.
3. For the comparison, the same 300 passes at beta 32 run from the same x without the limit.

The picture (default: docs/images/mbb_member_size.png) shows the beam without the limit above the beam with
it, solid black and void white, three pixels per element side, parted by a grey band.

The filter's faces say what lies beyond the grid: the left edge is the symmetry line, closed (a mirror);
the top, bottom and right edges are free, empty (void beyond them), so a member along them is filtered like
one in the interior and the limit sees its true thickness. With empty faces the corner element at the
support cannot become solid, so a pad of 7 by 7 elements there is kept solid (``michell.KeepSolid``, at the
end of the nominal design's chain; Clausen and Andreassen 2017, section 3.3). The limit reshapes members of
a layout that has formed; it does not change the layout, so the script prints whether the bound is met.
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

material = michell.Material(youngs_modulus=1.0, poisson_ratio=0.3, density_g_mm3=1.0)
problem = michell.Problem(grid, held, material, [michell.LoadCase("load", loads)], penalty=3.0, e_min=1e-9)

size = 7.0  # the minimum member size, mm (7 element sides)
# Radius R equal to the size; the symmetry edge closed, the three free edges empty.
outside = {"x-": "closed", "x+": "empty", "y-": "empty", "y+": "empty"}
density_filter = michell.DensityFilter(grid, size, outside=outside)
corner = np.zeros((ny, nx), dtype=bool)
corner[: int(size), nx - int(size) :] = True  # the solid pad at the support, bottom right
pad = michell.KeepSolid(corner.ravel(), grid)
mass_limit = michell.Limit(michell.on("nominal", michell.Mass()), at_most=0.35 * nx * ny)  # unit density
goal = michell.on("nominal", michell.Compliance())

# Step 1: without the limit, beta from 1 to 32.
started = time.perf_counter()
x = np.full(grid.n_elements, 0.5)
for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    projection = michell.Projection(beta, 0.5)
    result = michell.optimise(
        problem, goal, [mass_limit], [density_filter], michell.MovingAsymptotes(), x, passes=50,
        designs={"nominal": [projection, pad]},
    )  # fmt: skip
    x = result.x
    print(f"beta {beta:4g}: compliance {result.passes[-1].goal:.2f} N mm")

# Step 2: the member-size limit switched on at beta 32, on the design "filtered" (the filtered field).
member_size = michell.MemberSize(density_filter, projection, michell.member_threshold(density_filter, size))
eps = 6e-8  # figure 8 of Zhou et al. (2015) at r = 7
at_switch = member_size.evaluate(grid, density_filter.forward(x))[0]
size_limit = michell.Limit(michell.on("filtered", member_size), at_most=eps, scale=max(at_switch, eps))
print(f"limit switched on: eta_e {member_size.threshold:.4f}, c {member_size.c:g}, value {at_switch:.3g}")
with_limit = michell.optimise(
    problem, goal, [mass_limit, size_limit], [density_filter], michell.MovingAsymptotes(), x, passes=300,
    designs={"filtered": [], "nominal": [projection, pad]},
)  # fmt: skip

# Step 3: the same passes without the limit, for the comparison.
without_limit = michell.optimise(
    problem, goal, [mass_limit], [density_filter], michell.MovingAsymptotes(), x, passes=300,
    designs={"nominal": [projection, pad]},
)  # fmt: skip
seconds = time.perf_counter() - started

designs = {}
for label, result in (("without the limit", without_limit), ("with the limit", with_limit)):
    filtered = density_filter.forward(result.x)
    nominal = pad.forward(projection.forward(filtered))
    u, _ = problem.solve(nominal, loads)
    value = member_size.evaluate(grid, filtered)[0]
    met = "met" if value <= 1.1 * eps else "NOT met"
    print(
        f"{label:18}: compliance {michell.dot(loads, u):.2f} N mm, mass {nominal.sum():.1f} g, "
        f"member-size value {value:.3g} (bound {eps:g}, {met} to 10 %)"
    )
    designs[label] = nominal
print(f"{seconds:.1f} s")

# The two designs one above the other, without the limit at the top, each with y up, solid black.
pieces = []
for label in ("without the limit", "with the limit"):
    density = np.clip(designs[label], 0.0, 1.0).reshape(ny, nx)[::-1]
    grey = np.round(255 * (1.0 - density)).astype(np.uint8)
    pieces += [np.repeat(np.repeat(grey, 3, axis=0), 3, axis=1), np.full((12, 3 * nx), 160, dtype=np.uint8)]
picture = np.vstack(pieces[:-1])
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "docs" / "images" / "mbb_member_size.png"
out.parent.mkdir(parents=True, exist_ok=True)
write_greyscale_png(out, picture)
print(f"picture written to {out}")
