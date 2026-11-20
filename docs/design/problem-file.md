# The problem file

Status: **draft 0**. The format is sketched here so that the engine and the modules are built towards it; it
becomes real in stage 3 and may change until then. A machine-readable draft is in
`schema/problem-0.schema.json`.

## Why one file

The problem file is the whole contract with the outside. The command line, the Python library and the agent
face are thin layers over it: whatever they do, they first build this file's content, and a run can always be
repeated from the file and the geometry alone. It is TOML because people write it by hand.

## An example

```toml
schema = "michell/0"

[geometry]
file = "arm.step"              # one STEP file with named bodies, or STL files per region (below)

[regions]                      # each entry names a body in the STEP file, or an STL file
design = "envelope"            # where material may be
solid  = ["root_floor", "motor_pad"]    # must stay material
void   = ["bolt_holes", "wire_duct"]    # must stay empty

[grid]
voxel_mm = 0.8

[material]
name = "PLA"
youngs_modulus_mpa = 3500
poisson_ratio = 0.35
density_g_cm3 = 1.24

[[supports]]
on = "root_patch"              # a named body; the part is held where it touches this body
fix = ["z"]                    # any of x, y, z: a support can hold one direction only

[[supports]]
on = "root_bore"
fix = ["x", "y"]

[[load_cases]]
name = "thrust"
weight = 1.0
loads = [ { on = "motor_holes", force_n = [0, 0, 12] } ]

[[load_cases]]
name = "sideways"
weight = 1.0
loads = [ { on = "motor_holes", force_n = [0, 12, 0] } ]

[goal]
module = "compliance"
sense = "min"

[[limits]]
module = "mass"
max = 13.27                    # in the module's own unit, here grams

[[rules]]
module = "min_member"
size_mm = 2.5

[[rules]]
module = "symmetry"
planes = ["y"]

[run]
max_passes = 200
threads = 12

[output]
folder = "out"
mesh = ["stl", "3mf"]
```

## What each part means

- **`schema`** names the format version. A file with an unknown version is refused, never guessed at.
- **`geometry` and `regions`**: every region is a closed body. The design region is the space the part may
  fill; solid and void regions override it.
- **`supports` and `loads`** act where the part touches a named body, so they survive a change of voxel size.
  Units are fixed: mm, N, MPa, g/cm³ for density.
- **`goal`, `limits`, `rules`** each name a module (see `modules.md`). Extra keys in the same table are that
  module's own settings; the module checks them.
- **`run`**: the pass limit and the thread count. There is no random seed, because nothing is random.

## The three ways in

```
michell check part.toml        # reads everything, solves nothing: errors, a picture of the set-up,
                               # an estimate of time and memory
michell run part.toml          # the optimisation; writes the mesh, the report and result.json
```

The same two calls exist in Python (`michell.check(problem)`, `michell.run(problem)`), and an optional MCP
server offers them to AI agents. `--json` makes every command answer with structured data only.

## What comes back

`result.json` (its own versioned schema) holds: the versions of michell and the engine, the thread count, a
hash of the inputs, the history of every pass, the final value of the goal and each limit **with the name of
the check that stands behind each number**, the files written, and warnings. Errors are structured the same
way: a stable code, a sentence for people, and the numbers involved.

## Open questions (to settle in stage 3 with test cases)

1. How "touches a named body" becomes loaded or held nodes: only the contact face, or the overlapping volume.
   The two give measurably different stiffness, so the choice is made with test cases that have a known answer.
2. Load kinds beyond a plain force. Reserved names, not in the first version: `moment_n_mm` about a point,
   `pressure_mpa` normal to the contact face, `remote_force_n` acting at a point away from the part.
3. Whether one file may hold several materials or mass targets (the "outcomes" run of version 0.1), or whether
   a sweep is a second small file that points at this one.
4. STL input cannot carry names; regions are then one file each. Whether a STEP file's faces, not only its
   bodies, can be named is checked when STEP reading is written.
