# The bridge as a problem file

The bridge of [`../bridge_3d_member_size.py`](../bridge_3d_member_size.py), written as a problem file
([`bridge.toml`](bridge.toml)) with its geometry as STL files: a block 50 x 24 x 10 mm of steel, clamped at
both ends, carrying 1000 N down on a pad in the middle of its top face; the stiffest bridge with 30 % of the
block's mass and members at least 5 mm thick, on voxels of 1 mm.

| File | What it is |
|---|---|
| `envelope.stl` | the design region: where material may be |
| `plate_low.stl`, `plate_high.stl` | the end plates, 2 mm thick: solid regions, which stay material |
| `pad.stl` | the pad under the load, 6 x 6 x 2 mm: a solid region |
| `clamp_low.stl`, `clamp_high.stl` | marker bodies: slabs beyond each end face, where the bridge is held |
| `load_pad.stl` | a marker body on top of the pad, through which the load acts |
| `make_geometry.py` | the script that wrote the STL files, with `michell.mesh` |

A support or a load acts where the part touches its marker body, so the file says nothing about nodes or
voxels; [`docs/design/problem-file.md`](../../docs/design/problem-file.md) describes every key and
[`docs/design/geometry.md`](../../docs/design/geometry.md#supports-and-loads) the contact rule.

From the repository:

    michell check examples/bridge_file/bridge.toml     # reads and sets up everything, solves nothing
    michell run examples/bridge_file/bridge.toml       # about a minute on one thread

`check` prints the grid, the volume table, each support and load (its contact faces, nodes and point of
action), the density filter's faces, the mass bound, the member size and the files `run` will write. `run`
prints a line per pass and writes `result.json`, `result.npz`, `result.vtk` (the design, for ParaView) and
the part as a closed mesh, `part.stl` and `part.3mf`, into `out/` beside the file;
`--out DIR` writes them elsewhere and `--json` prints the outcome as JSON. The arrays, the mass bound and
the compliance equal those of the script built by hand (verification row 33).
