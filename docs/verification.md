# Verification

This table checks the 2D engine against closed-form solutions, against central finite differences, against
the published 88-line topology optimisation code (Andreassen et al., Struct. Multidisc. Optim. 43, 2011, run
in GNU Octave; see `tests/reference/top88/PROVENANCE.txt`) and against published optima (Svanberg, Int. J.
Numer. Meth. Eng. 24, 1987). The measure, the set-up and the limit of every row were fixed before its first
result was computed; a limit is never changed to make a row pass. Every limit is an upper bound on the measured
value. Measured values are rounded to three significant digits; values at the level of floating-point
round-off may differ in those digits from one machine to another. The code of every row, with its set-up, is
in `python/michell/verification/`, and continuous integration runs every row on each change.

Version: michell 0.0.1 (engine 0.0.1).

| Row | Check | Kind | Measured | Limit | Result | Note |
|---|---|---|---|---|---|---|
| 1 | Element stiffness matrix against the textbook closed form | closed form | 2.25e-16 | ≤ 1e-12 | pass | largest entry difference relative to the largest entry, worst of 3 sets of modulus, Poisson ratio, side and thickness |
| 2 | Patch test: a linear displacement field and its uniform stress | closed form | 1.01e-15 | ≤ 1e-10 | pass | relative error, the larger of the inside displacements (2.74e-16) and the stress (1.01e-15); 7 x 5 elements |
| 3 | Pure shear: element shear stress equal to G gamma, normal stresses zero | closed form | 1.86e-15 | ≤ 1e-10 | pass | relative to G gamma, the larger of the shear stress error (8.8e-16) and the normal stresses (1.86e-15) |
| 4 | Cantilever, length 8 x depth, tip deflection against Timoshenko beam theory | closed form | 0.0026 | < 0.02 | pass | relative error with 40 elements through the depth; with 10, 20, 40: 0.00757, 0.00364, 0.0026; the error must also shrink: it does |
| 5 | Plate with a hole (a = 16 elements) in tension: stress at 2a against Kirsch | closed form | 0.0293 | < 0.03 | pass | relative error; sx 125 MPa against Kirsch's 122 MPa; this row sits close to its limit |
| 5 | Plate with a hole (a = 16 elements) in tension: stress at 3a against Kirsch | closed form | 0.0213 | < 0.03 | pass | relative error; sx 110 MPa against Kirsch's 107 MPa; the peak on the section, at the stair-stepped hole edge, is 2.54 times the applied stress (Kirsch: 3), reported, not judged |
| 6 | MBB beam 60 x 20, uniform density 0.5: compliance against the 88-line code | reference code | 8.02e-12 | ≤ 1e-08 | pass | relative difference; compliance 1010 N mm |
| 6 | MBB beam 150 x 50, uniform density 0.5: compliance against the 88-line code | reference code | 1.11e-10 | ≤ 1e-08 | pass | relative difference; compliance 1030 N mm |
| 7 | Gradient of compliance through the density filter against central differences | gradient | 1.94e-07 | ≤ 1e-05 | pass | 66 elements with a hole, two load cases, random density; largest difference relative to the gradient's size; worst element on its own scale 2.03e-06 (reported, not judged) |
| 7 | Gradient of mass through the density filter against central differences | gradient | 5.73e-09 | ≤ 1e-05 | pass | 66 elements with a hole, two load cases, random density; largest difference relative to the gradient's size; worst element on its own scale 8.63e-09 (reported, not judged) |
| 8 | MBB beam 60 x 20 with OC: compliance on passes 1 to 10 against the 88-line code | reference code | 8.02e-12 | ≤ 1e-06 | pass | largest relative difference; filter radius 2.4, move limit 0.2 |
| 8 | MBB beam 60 x 20 with OC: compliance on every later pass against the 88-line code | reference code | 6.72e-12 | ≤ 0.001 | pass | largest relative difference over passes 11 to 144; final compliance 234 N mm |
| 8 | MBB beam 150 x 50 with OC: compliance on passes 1 to 10 against the 88-line code | reference code | 1.11e-10 | ≤ 1e-06 | pass | largest relative difference; filter radius 6, move limit 0.2 |
| 8 | MBB beam 150 x 50 with OC: compliance on every later pass against the 88-line code | reference code | 4.72e-11 | ≤ 0.001 | pass | largest relative difference over passes 11 to 362; final compliance 236 N mm |
| 9 | MMA on Svanberg's cantilever: objective against the published optimum and SciPy | published optimum | 3.26e-05 | ≤ 0.001 | pass | the larger relative difference, to the published optimum 1.340 (3.26e-05) and to SciPy's optimum (7.48e-11); 13 updates |
| 9 | MMA on Svanberg's cantilever: worst limit f_i - 1 at the optimum | published optimum | -2.24e-10 | ≤ 1e-06 | pass | every limit written as f_i <= 1 in the published scaling |
| 9 | MMA on Svanberg's two-bar truss: objective against the published optimum and SciPy | published optimum | 0.000892 | ≤ 0.001 | pass | the larger relative difference, to the published optimum 1.51 (0.000892) and to SciPy's optimum (6.64e-11); 16 updates |
| 9 | MMA on Svanberg's two-bar truss: worst limit f_i - 1 at the optimum | published optimum | -6.62e-11 | ≤ 1e-06 | pass | every limit written as f_i <= 1 in the published scaling |
| 10 | MBB beam 60 x 20 with MMA: final compliance over OC's after the same passes | reference code | 0.999 | ≤ 1.02 | pass | 144 passes each; compliance MMA 233 N mm, OC 234 N mm |
| 10 | MBB beam 60 x 20 with MMA: volume excess mass / bound - 1 | reference code | -8.66e-10 | ≤ 1e-06 | pass | larger of the last pass (-8.89e-10) and the returned design (-8.66e-10) |

20 passed, 0 failed, 0 not run.

## Regenerating this page

From the repository, with the package and SciPy installed:

    python -m michell.verification --reference tests/reference --out docs/verification.md

Rows 6, 8 and 10 read the output of the 88-line code from `tests/reference`, and row 9 solves its problems a
second time with SciPy; without them those rows are listed as not run. The command exits with status 1 when a
row fails (and, with `--require-all`, when a row was not run). The test suite runs the same functions.
