# Estimates of time and memory, and the picture of the set-up

Status: design note, kept in step with the code. `michell check` gives, besides the set-up, an estimate of the
time and the peak memory of the run (`michell.estimate`) and, with `--picture`, a picture of the voxel model
with its supports and loads (`michell.picture`). Both are also in the structured output: `Prepared.estimate`
and `summary()["estimate"]`, the `estimate` object of `michell check --json`, and the `picture` entry naming
the file written.

## What the estimate covers

The estimate is for the passes of `stiff_part` with its default schedule (140 passes): seconds per pass, the
total for the 140 passes, and the peak memory of the process. Each is a range, low to high, with a centre. It
does not cover the preparation (`check` itself has done it and printed its time in the JSON output), nor the
mesh of the part after the passes (`[output] mesh`), which on a part of a million voxels takes a few minutes
more. The output says which machine the estimate is for.

## The model

The measurements are those of [`docs/performance.json`](../performance.json) (see
[`docs/performance.md`](../performance.md)): `stiff_part` on a 3D cantilever at five grid sizes from 93 312 to
1 458 000 voxels, on one machine and 12 threads, with the time and the conjugate-gradient steps of every pass.
A pass is the density filter forward and back, one solve per load case, the gradients and the update. With
`n` active voxels (millions), `s` solver steps in the pass and a density filter whose stencil holds `k` cells
(thousands; the filter sums over `(2 reach + 1)^3` offsets per voxel, `reach = ceil(radius) - 1` in voxel
sides),

    seconds per pass = n (a + b s + c k).

The three numbers are fitted by least squares on the relative error over all 700 measured passes. The filter
term matters: the member size is fixed in millimetres in the measurements, so the filter radius grows from 5.4
to 13.5 voxel sides with the grid, and a model in `n` and `n s` alone is 24 % fast at the largest size. With
the filter term, the fitted time at the mean steps of each size is within 2.2 % of the measured mean at all
five sizes, and within 2.1 % at each size when that size is left out of the fit (fitted to the other four).
The reach is at most the grid's largest count of voxels along an axis, as in the engine.

The steps of a pass are not known before the run: they depend on the shape, the supports and the loads, and
they change as the design forms (in the measurements, from about 16 in the first passes to over 90 in a few).
The estimate takes the steps per pass and load case as the mean over the measured sizes (about 44), and its
range as the lowest and the highest of the sizes' means (about 37 to 50), each times the number of load cases.
The range is then widened by the fit's largest error at a size left out of it (2.1 %). The range is how wide
the measurements say the estimate is, not a bound. Fitted to four sizes and tested on the fifth, each in turn, 3
of the 10 held-out values (five times, five memories) fall outside their range, by at most 1.4 % of the measured
value (see Verification).

The peak memory is `m n_v^p` GB for `n_v` voxels of the grid (millions), fitted in logarithms to the five sizes
(`p` about 0.81). Fitted to four sizes, it is off at the fifth by up to 13.9 % (at the smallest and the largest
size, where it extrapolates), and the range is that much either side.

The filter's term matters little below the measured radii: at the smallest, 5.4 voxel sides (1 331 cells),
and the mean steps, it is 3.1 % of the time of a pass, and at a smaller radius it only shrinks, so the error
it can cause there is at most that share. A radius below the measured ones is therefore not warned about (the
default member size of five voxel sides is one); a radius above them is, since the term grows with the cube of
the reach.

The measurements have one load case. What each further load case holds was counted in the code: in the Python
face, its load vector, its displacements of the pass, those of the previous pass (kept as the first guess of the
next solve), and, while the gradients are formed, the load vector times its weight that the compliance hands on
(the evaluation keeps read-only copies of the displacements, made as it is built, after which the originals go).
That is at most four vectors of three doubles per node: 96 bytes per node, or 0.099 GB per million voxels on a
grid of 1 458 000 voxels (1.03 nodes per voxel). The engine keeps nothing per load case: the vectors of a solve
and its multigrid levels are made and freed within the solve, and the cases are solved one after another.

The memory added by each further case is measured by

    python -m michell.estimate load-cases --size 90 --cases 3 --passes 4 --threads 12

on the cantilever of `michell.performance` at 1 458 000 voxels: one run with its load case and one with three
(the same force along -z, +y and +x at the free end), each in a fresh process, over the first four passes. On
the reference machine the peaks were 2.03 and 2.29 GB, so 0.092 GB per million voxels for each further case,
about 3.7 of the four counted vectors; the peak with three cases was still rising at the fourth pass, so the
number may be low. It is the number the estimate adds. On a larger part not in the repository the same
difference was larger, 0.12 GB per million voxels, more than the count explains; the estimate with the
cantilever's number still held that part's measured peak with three load cases inside its range. On that part
a pass with three load cases took 2.7 times as long as one with one load case; the model gives 2.9.

The fitted numbers ship with the package (`michell/estimate/model.json`), with the reference timing and the
memory per load case; they are written from `docs/performance.json` by

    python -m michell.estimate fit docs/performance.json --out python/michell/estimate/model.json --timing

(the memory per load case is given with `--extra-load-case`, or kept, and `load-cases --store` writes a new
measurement into the model), and a test checks that the shipped fit equals a fresh fit of the committed
measurements. The reference timing is stored with its thread count, the day it was made and the versions of
the package and of the engine it timed, so that a timing of a later engine can be told from it.

## Scaling to the machine at hand

`michell check` times a fixed workload on the problem's thread count: one solve of a cantilever of
48 x 24 x 24 voxels at a uniform density, and one pass of the density filter (radius 4.5 voxel sides) forward
and back on the same grid, each the shortest of repeated runs within 0.3 s. The same workload was timed on the
reference machine when the model was written (`--timing` above). The ratio of the solves scales the terms `a`
and `b`, the ratio of the filters the term `c`. The timing takes about half a second on the reference machine
and about two seconds on one thread of a laptop; `--no-timing` skips it, and the estimate is then the
reference machine's, which the output says.

## Where it stops being trusted

- **What was measured.** The model is fitted between 93 312 and 1 458 000 active voxels, filter radii from
  5.4 to 13.5 voxel sides and one load case; outside the sizes, above the largest radius or with more load
  cases, `check` prints a warning, and so it does when it
  has no timing (`--no-timing`, or `prepare()` without one) and the problem's thread count is not the
  reference's 12, since the estimate is then for the reference machine on its 12 threads. Below the range the fixed costs of a run (the
  interpreter, NumPy, the set-up of each solve) are a larger share than the model holds, and the memory
  estimate in particular is low there: the power law has no floor for the interpreter and the libraries.
- **Other machines.** The timing scales the estimate by a small workload. A small workload uses many
  threads less well than a large problem does, and machines differ in memory bandwidth, which the solve of a
  large grid depends on more than the timing's solve does; so on another machine the estimate is less sure
  than its range says. An example: on the reference machine the timing's solve is 3.8 times slower on one
  thread than on 12, while the example bridge (12 000 voxels, below the measured sizes) ran 2.25 times slower,
  so the timing overstates its time on one thread by 3.8 / 2.25, about 1.7. Its estimate on one thread was
  0.65 s per pass against 0.28 measured, 2.3 times too long; the rest of that error, about 1.36, is the
  bridge's small size, where the estimate on 12 threads (centre 0.171 s) was already above the measured
  0.126 s.
- **Other problems.** The solver steps of the measured cantilever stand for every problem. A part with thin
  regions, many supports or loads far from them can need more steps per solve than the cantilever did.
- **Load cases.** Each load case is assumed to cost one solve per pass, with the steps of the measured
  cantilever; the measurements have one load case, and the memory per further case was measured on one part
  only.
- **Not included.** The preparation, the mesh of the part, and anything else running on the machine.

## The picture

`michell check part.toml --picture set-up.svg` writes one SVG with four views of the voxel model: from the
top (+z), the front (-y), the right (+x) and an oblique view (from +x, -y and +z). The views are rasters drawn
by `michell.picture` with NumPy only and embedded as PNG images (encoded with `zlib`); the view names, the
scale bars in mm, the axes, the legend and the marks of the supports and loads are SVG text and lines over
them.

- **What is drawn.** The design space light, the voxels held solid darker, the voxels of the supports blue
  and those of the loads amber; voids and the space outside the part are not drawn. A voxel belongs to a
  support or a load when it is one of its region's voxels, or when one of its corners is one of its nodes.
- **Seen through.** The held, support and load voxels are also drawn alone, and wherever they show, that
  raster and the whole one are mixed half and half: a solid region inside the part, or bosses on the face
  turned away from every view, show as paler patches of their colours.
- **Marks.** Every support's point of action has a blue square and every load an amber arrow in the
  direction of its force, ending at its point of action and named by its load case; a force along the line of
  sight is a circle with a dot (towards the viewer) or a cross (away). Marks are drawn over the part, so a
  support or load hidden behind it still shows where it is.
- **The renderer.** Every voxel with a face on empty space is projected; each pixel shows the face of the
  nearest voxel that the line of sight through its centre meets (found exactly, per voxel, as the slab where
  the line enters the cube). The face is shaded by its direction to a light above, left of and in front of the
  viewer, nearer faces are lighter, and pixels where the depth jumps by more than a voxel side are darkened
  as outlines. All four views share one scale, chosen so the largest fits 360 pixels (at most 12 pixels per
  voxel side).
- **Blue and amber** are told apart by people who do not see red and green apart.
- **Deterministic on one installation.** The same input gives the same bytes on one installation: the ties
  in depth are broken by the voxel's index, and nothing depends on the time or the machine's name. The PNG
  data are compressed by the zlib library Python is built with, and another (zlib-ng, for one) compresses the
  same pixels to other bytes, so two installations can write different files of the same picture.

## Verification

Row 38 leaves each measured size out in turn: the model is fitted to the other four sizes and its estimate
for the fifth is compared with the measurements there. Its limits were fixed before the numbers were computed:
the time within a factor 1.5 of the centre, the memory within 20 %, and no size outside the range, for time or
for memory. The first two hold (largest factor 1.21, at 194 672 voxels; memory 13.9 %); the third was
missed: the time range misses two sizes (194 672 voxels, 2.45 s against 2.48 to 3.27 s, outside by 1.4 % of
the measured value, and 746 496 voxels, 14.1 s against 11.0 to 14.0 s, by 0.8 %) and the memory range one
(390 224 voxels, 0.964 GB against 0.790 to 0.963 GB, by 0.1 %). The ranges are drawn from the steps of the
other sizes and widened by the errors of the fit alone, so a size whose steps per pass lie outside those of
the others falls outside. Since the range is not a bound, the row keeps as pass or fail lines only the factor
and the memory, and reports these counts in their notes with no limit, saying that the limit was missed. It also checks
that the estimate grows with the size and the load cases. It cannot test the estimate on other problems or
machines.

Row 39 draws, as the picture does (`see_through`), a model of 4 x 3 x 2 voxels with a support, a load and a
voxel held solid, and checks the colour at the centre of every voxel face in the top and front views (from the
front, the support and the load show through the design space, half and half), that the centre of the top face
of the support's voxel in the oblique view is blue and that of the load's amber (blue at least 60 above red,
red at least 60 above blue; the greys of the design space and the held voxels differ by at most 6, and the row
fails on the same model with those two voxels design space or held solid), and that drawing it twice gives the
same bytes.
