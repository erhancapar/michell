# The problem file

Status: **schema `michell/0`, read and run** for parts given as STL files: `michell check` and `michell run`
work on it, and so do `michell.prepare` and `michell.run` in Python (`michell.problem_file`). A STEP file
with named bodies, symmetry, mesh output, estimates of time and memory, a picture of the set-up and a report
are planned in [milestone 3](../roadmap.md) and refused for now with a message that says so (below). A
machine-readable description of what is accepted now is in `schema/problem-0.schema.json`; the reader is
the reference for the exact checks.

## Why one file

The problem file is the whole contract with the outside. The command line and the Python library are thin
layers over it: whatever they do, they first read this file's content, and a run can always be repeated from
the file and the geometry alone. It is TOML because people write it by hand.

## An example

`examples/bridge_file/bridge.toml` is the bridge of `examples/bridge_3d_member_size.py` written as a problem
file, with its geometry as STL files beside it (written by `make_geometry.py` there):

```toml
schema = "michell/0"

[regions]
design = "envelope.stl"                     # where material may be
solid = { plate_low = "plate_low.stl", plate_high = "plate_high.stl", pad = "pad.stl" }   # must stay material

[bodies]                                    # marker bodies: where the part is held and loaded
clamp_low = "clamp_low.stl"
clamp_high = "clamp_high.stl"
load_pad = "load_pad.stl"

[grid]
voxel_mm = 1.0

[material]
name = "steel"
youngs_modulus_mpa = 210000
poisson_ratio = 0.3
density_g_cm3 = 7.85

[[supports]]
on = "clamp_low"                            # held where the part touches this body
fix = ["x", "y", "z"]

[[supports]]
on = "clamp_high"
fix = ["x", "y", "z"]

[[load_cases]]
name = "pad"
loads = [ { on = "load_pad", force_n = [0, 0, -1000] } ]

[goal]
module = "compliance"
sense = "min"

[[limits]]
module = "mass"
max_fraction = 0.30                         # of the whole block's mass, every voxel solid

[[rules]]
module = "min_member"
size_mm = 5.0

[run]
threads = 1

[output]
folder = "out"
fields = ["vtk"]
```

Prepared from this file, the bridge has the same active voxels, voxels held solid, held degrees of freedom,
load vector, density filter faces and mass bound as the script builds by hand, and `michell run` on it
gives the script's compliance (verification row 33).

## What each part means

Units are fixed: mm, N, MPa, g/cm³ for density, g for mass. Every key the format does not have is refused,
anywhere in the file, so a typing mistake is never ignored. Paths of geometry files are relative to the
problem file.

- **`schema`** names the format version, `"michell/0"`. A file with another value is refused, never
  guessed at. It is checked before anything else.
- **`[regions]`**: `design` (one STL file, the space the part may fill), and optionally `solid` and `void`,
  each a table of name to STL file: solid regions must stay material, void regions must stay empty, and
  both override the design region. Every region is a closed body. The voxel model is built from them as
  [geometry.md](geometry.md#the-voxel-model) describes.
- **`[bodies]`** (optional): marker bodies, name to STL file. They are not part of the part: they say where
  it is held or loaded. Every name of a solid region, a void region and a marker body is used once, and
  `design` is the design region's.
- **`[grid] voxel_mm`**: the side of the cube voxels.
- **`[material]`**: `name` (optional), `youngs_modulus_mpa`, `poisson_ratio` (between -1 and 0.5, both
  left out) and `density_g_cm3`. Whole numbers are read as numbers; `true` and `false` are not.
- **`[[supports]]`**: `on`, what it acts on; `fix`, the directions held (any of `"x"`, `"y"`, `"z"`; all
  three by default); `skin`, whether the voxels behind a contact patch on design voxels are held solid
  (`true` by default).
- **`[[load_cases]]`**: `name` (each its own), `weight` in the weighted sum of the cases (1.0 by default,
  at least 0) and `loads`, a list of `{ on = ..., force_n = [fx, fy, fz], skin = true }`.
- **`on`** names a solid region (a load is then a uniform body force over its voxels, a support holds all
  of its nodes) or a marker body, or is a box `{ box_mm = [[x0, y0, z0], [x1, y1, z1]] }` in the meshes'
  coordinates (a load is a uniform traction over the part's boundary faces in contact with it, a support
  holds the nodes of those faces). The rules, with the contact test, the held skin and the filter faces,
  are in [geometry.md](geometry.md#supports-and-loads) (`michell.attach`). Naming the design region or a
  void region is refused.
- **`[goal]`**: `module = "compliance"`, `sense = "min"`, the only goal for now (the weighted sum of the
  load cases' compliances).
- **`[[limits]]`**: exactly one, `module = "mass"`, with exactly one of `max` (g) or `max_fraction` (of the
  mass of the whole part, every active voxel solid: `max_fraction` times the material's mass of the active
  voxels' volume, the product the example scripts write). The bound must lie above the mass of the voxels
  held solid (the solid regions and the skins) and at most at the whole part's; `michell check` says so
  before any pass.
- **`[[rules]]`**: `module = "min_member"` with `size_mm`, the minimum member size (also the radius of the
  density filter). Without it the size is five voxel sides, the smallest the grid resolves, and `check` and
  `result.json` say that it was not given.
- **`[run]`**: `threads` (1 by default; `--threads` overrides it) and optionally `filter_faces`, a table of
  grid face (`"x-"` to `"z+"`) to `"closed"` or `"empty"` that overrides what the density filter sees beyond
  that face of the grid (by default a face is closed only when one support covers its whole active area,
  [geometry.md](geometry.md#supports-and-loads)). There is no random seed, because nothing is random, and no
  pass limit: the procedure fixes its passes.
- **`[output]`**: `folder` (`"out"` by default, relative to the problem file; `--out` overrides it) and
  `fields`, the field files to write besides the result: `["vtk"]` for `result.vtk`.

Every run is the two-step procedure of `michell.recipes.stiff_part` (the robust formulation, then the
member-size limit), through `michell.attach.SetUp.stiff_part`, so the returned design is always a 0/1
design. There is one path and no setting of the procedure in the file.

## The commands

```
michell check part.toml [--threads N] [--json]
michell run part.toml [--out DIR] [--threads N] [--json]
```

`check` reads the file, reads the STL files, builds the voxel model and places the supports and loads, and
solves nothing. It prints the grid (counts, voxel side, the grid's corner, active, held-solid and design
voxels), the bodies of the part, the volume table, the material, each support and load (its contact faces
and area, or its voxels and volume, its nodes, the point of action, the share of a load that falls on held
degrees of freedom, the held skin and the closed faces it lies on), the density filter's faces, the goal,
the mass bound in grams and the member size with whether it was given. `run` prints the same, then a line
per pass and a short summary, and writes the outcome.

In Python, `michell.problem_file.read(path)` gives the checked description (`ProblemFile`),
`michell.prepare(path)` everything `check` prints (`Prepared`, with `summary()` as plain data) and
`michell.run(path, out=None, threads=None, callback=None)` the outcome (`RunOutcome`: the `StiffPart`, the
paths of the files written, the seconds and the warnings).

`--json` makes either command print one JSON object on standard output and nothing else (`run` then prints
its pass lines on standard error). The exit status is 0 when the file is accepted (`check`) or the run met
its limits (`run`); 1 when the run finished but a limit is not met (the design is written all the same); 2
when the input is refused, which is anything refused before the first pass: the problem file, the
geometry, the set-up, a part that is not held against rigid motion (`singular_system`), an output folder
that cannot be made, and the command line itself (a bad option is refused with a usage line on standard
error and nothing on standard output, also with `--json`); 3 when the engine fails during the passes (an
`EngineError` of the engine, such as `solver_not_converged`, rather than a refusal) or anything else goes
wrong (code `internal_error`).

## What comes back

`run` writes into the output folder:

- `result.npz`: every array of the outcome (`michell.export.write_result`: the design variables, the
  filtered field, the nominal design, the design after the first step, the member-size thresholds);
- `result.json`, format 2: everything of format 1 (the package version, the goal's and the limits' values
  on the returned design, their bounds, `limits_met`, the stop reason, the thread count, the record of
  every pass) and the record of the run: the problem file's name and SHA-256, each geometry file's name and
  SHA-256, the grid and its origin, the volume table, the set-up report (bodies, material, goal, mass limit,
  member size, every support and load, the filter faces), the warnings given, the files written and the
  seconds the preparation and the optimisation took. `michell.read_result` reads both formats;
- `result.vtk` with `fields = ["vtk"]`: the nominal design, the design after the robust step, the voxels
  held solid and the active voxels, for ParaView.

A refused input is reported the same way every time: a stable code, the key's place in the file, the file
and a sentence for people (with `--json`, `{"code", "path", "file", "message", "details"}`). The codes of
the file itself (`michell.errors`):

| Code | The fault |
|---|---|
| `invalid_toml` | the file is not TOML (the message gives the line and column), or not UTF-8 |
| `unknown_schema` | `schema` is not `"michell/0"` |
| `unknown_key` | a key the format does not have, anywhere |
| `missing_key` | a key the format needs |
| `wrong_type` | a value of the wrong type or shape (a string for a number, two numbers for three) |
| `out_of_range` | a value outside its range (a voxel side of 0, a direction `"w"`, a mass bound below the held solid) |
| `unknown_name` | `on` names no solid region or marker body; a goal, limit or rule module michell does not have |
| `duplicate_name` | a name given twice (a region and a marker body, two load cases, a direction) |
| `file_not_found` | the problem file or a geometry file it names does not exist or cannot be read |
| `not_available_yet` | a part of the format that is planned and not built yet (below) |

Refusals of the geometry and the set-up by `michell.mesh`, `michell.regions` and `michell.attach` (a mesh
that is not closed, a region too thin for the voxels, a marker body that touches no face of the part, a load
mostly on held degrees of freedom) keep their code, `invalid_input`, and carry the path of the entry that
caused them.

## Refused for now, and why

These are part of the format's plan and are refused with `not_available_yet`, so that a file written for a
later version fails clearly instead of being half read:

- `[geometry] file = "part.step"`, and STEP files named as regions: reading STEP (named bodies, units,
  instances) comes with the optional STEP support; until then every region and marker body is an STL file.
- `[[rules]] module = "symmetry"`: the mirror filter and the checks that the regions, supports and load
  cases mirror are still to be built.
- `[output] mesh`: turning the design into a closed mesh (STL, 3MF) is still to be built.
- The reserved load kinds `moment_n_mm`, `pressure_mpa` and `remote_force_n`.

## Planned

- **STEP input**: `[geometry] file` with named bodies as the regions and marker bodies, by name.
- **Symmetry**: `[[rules]] module = "symmetry"`, `planes = ["y"]`, through the centre of the regions' box.
- **Mesh output**: `[output] mesh = ["stl", "3mf"]`.
- **`check` with estimates and a picture**: an estimate of time and memory, and a picture of the set-up
  (supports and loads drawn on the voxel model).
- **A report**: one self-contained page with the inputs and their hashes, the set-up, views of the result,
  goal and limits with whether each is met and the name of the check behind each number.
- An optional server offers `check` and `run` to programs that call tools.

## Open questions

1. Settled: how "touches a named body" becomes loaded or held nodes. A named solid region of the part is
   loaded as a body force over its voxels or held at all its nodes; any other body, or a box, acts on the
   part's boundary faces in contact with it. The rules, with the held skin under a contact patch and the
   filter faces, are in [geometry.md](geometry.md#supports-and-loads) (`michell.attach`); verification row
   32 checks them against cases with known answers.
2. Load kinds beyond a plain force. Reserved names, not in the first version: `moment_n_mm` about a point,
   `pressure_mpa` normal to the contact face, `remote_force_n` acting at a point away from the part.
3. Settled: one file holds one material and one mass limit. A sweep over several materials or mass targets
   (the "outcomes" run of version 0.1) will be a second, small file that points at this one, so that this
   file stays the description of one run.
4. Settled for STL: STL cannot carry names, so each region and each marker body is one STL file, named in
   the problem file (`[regions]` and `[bodies]`). Whether a STEP file's faces, not only its bodies, can be
   named is checked when STEP reading is written.
