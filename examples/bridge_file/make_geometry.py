"""Writes the STL files of the bridge problem file ``bridge.toml`` beside this script, with ``michell.mesh``.

The bridge of ``examples/bridge_3d_member_size.py`` as closed bodies, in mm (x along the span, z up):

- ``envelope.stl``: the design region, the block from the origin to (50, 24, 10);
- ``plate_low.stl`` and ``plate_high.stl``: the end plates, 2 mm thick at x = 0 and x = 50 (solid regions);
- ``pad.stl``: the pad under the load, x from 22 to 28, y from 9 to 15, z from 8 to 10 (a solid region);
- ``clamp_low.stl`` and ``clamp_high.stl``: marker slabs 1 mm thick beyond each end face, overhanging the
  face by 1 mm on every side, where the bridge is held;
- ``load_pad.stl``: a marker box 1 mm thick on top of the pad, of the pad's size, through which it is loaded.

Every coordinate is a whole number of mm, which STL's single-precision numbers hold exactly. Run from the
repository:

    python examples/bridge_file/make_geometry.py
"""

from pathlib import Path

from michell import mesh

HERE = Path(__file__).parent

LENGTH, WIDTH, HEIGHT = 50.0, 24.0, 10.0
PLATE = 2.0
PAD = ((22.0, 9.0, 8.0), (28.0, 15.0, 10.0))
SLAB, OVERHANG = 1.0, 1.0

BODIES = {
    "envelope": ((0.0, 0.0, 0.0), (LENGTH, WIDTH, HEIGHT)),
    "plate_low": ((0.0, 0.0, 0.0), (PLATE, WIDTH, HEIGHT)),
    "plate_high": ((LENGTH - PLATE, 0.0, 0.0), (LENGTH, WIDTH, HEIGHT)),
    "pad": PAD,
    "clamp_low": ((-SLAB, -OVERHANG, -OVERHANG), (0.0, WIDTH + OVERHANG, HEIGHT + OVERHANG)),
    "clamp_high": ((LENGTH, -OVERHANG, -OVERHANG), (LENGTH + SLAB, WIDTH + OVERHANG, HEIGHT + OVERHANG)),
    "load_pad": ((PAD[0][0], PAD[0][1], PAD[1][2]), (PAD[1][0], PAD[1][1], PAD[1][2] + SLAB)),
}

if __name__ == "__main__":
    for name, (low, high) in BODIES.items():
        mesh.write_stl(HERE / f"{name}.stl", mesh.box(low, high))
        print(f"wrote {name}.stl: the box from {low} to {high} mm")
