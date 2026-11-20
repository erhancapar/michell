# Verification

This table checks the 2D and the 3D analysis, the 2D and the 3D optimisation, the Helmholtz filter and the
projection against closed-form solutions, against an exact integration and an independent assembly of the 3D
stiffness matrix, against a sparse direct solution of the 3D system, against central finite differences,
against the published 88-line topology optimisation code (Andreassen et al., Struct. Multidisc. Optim. 43,
2011, run in GNU Octave; see `tests/reference/top88/PROVENANCE.txt`) and against published optima (Svanberg,
Int. J. Numer. Meth. Eng. 24, 1987); it also runs the 3D analysis and the Helmholtz filter on several thread
counts and twice in a row and compares the numbers, counts the steps the 3D solver takes, and checks the three
designs of a robust optimisation (Wang, Lazarov and Sigmund, Struct. Multidisc. Optim. 43, 2011) for their
mass, their discreteness (the measure of Sigmund, Struct. Multidisc. Optim. 33, 2007) and the order of their
compliances. The measure, the set-up and the limit of every row were fixed before its first result was
computed; a limit is never changed to make a row pass. Every limit is an upper bound on the measured value.
Measured values are rounded to three significant digits; values at the level of floating-point round-off may
differ in those digits from one machine to another. The code of every row, with its set-up, is in
`python/michell/verification/`, and continuous integration runs every row on each change.

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
| 7 | Gradient of mass through the density filter against central differences | gradient | 6.03e-09 | ≤ 1e-05 | pass | 66 elements with a hole, two load cases, random density; largest difference relative to the gradient's size; worst element on its own scale 6.11e-09 (reported, not judged) |
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
| 11 | 3D element stiffness matrix against an exact integration of the shape functions | closed form | 3.93e-16 | ≤ 1e-12 | pass | largest entry difference relative to the largest entry, worst of 3 sets of modulus, Poisson ratio and side |
| 11 | 3D element stiffness matrix is symmetric | closed form | 5.9e-17 | ≤ 1e-14 | pass | largest difference between K and its transpose relative to the largest entry, worst of 3 sets of modulus, Poisson ratio and side |
| 11 | 3D element stiffness matrix has exactly six zero eigenvalues (the rigid motions) | closed form | 2.08e-16 | < 1e-10 | pass | the sixth smallest eigenvalue relative to the largest, worst of 3 sets of modulus, Poisson ratio and side; the seventh must be above the limit: its smallest relative value is 0.0426 |
| 12 | 3D patch test: a linear displacement field and its uniform stress | closed form | 3.97e-15 | ≤ 1e-08 | pass | relative error, the larger of the inside displacements (5.91e-16) and the stress (3.97e-15); 4 x 3 x 5 elements, 1 conjugate-gradient steps |
| 13 | 3D pure shear in the yz plane: shear stress equal to G gamma, other components zero | closed form | 1.6e-15 | ≤ 1e-08 | pass | relative to G gamma, the larger of the shear stress error (5.28e-16) and the other five components (1.6e-15); 4 x 4 x 4 elements |
| 13 | 3D pure shear in the zx plane: shear stress equal to G gamma, other components zero | closed form | 3.12e-15 | ≤ 1e-08 | pass | relative to G gamma, the larger of the shear stress error (8.8e-16) and the other five components (3.12e-15); 4 x 4 x 4 elements |
| 13 | 3D pure shear in the xy plane: shear stress equal to G gamma, other components zero | closed form | 1.09e-15 | ≤ 1e-08 | pass | relative to G gamma, the larger of the shear stress error (5.28e-16) and the other five components (1.09e-15); 4 x 4 x 4 elements |
| 14 | 3D matrix-free product K(rho) x against the matrix assembled with SciPy | independent computation | 4.59e-16 | ≤ 1e-12 | pass | largest entry difference relative to the largest entry; 7 x 5 x 3 elements, 12 outside the part, random densities, 114 held degrees of freedom |
| 15 | 3D analysis and Helmholtz filter give the same numbers on 1, 2, 5 and 12 threads and on a repeated run | repeatability | 0 | ≤ 0 | pass | largest absolute difference from the run on 1 thread, over the product, a complete solve (24 steps to 1e-10: displacements, final residual, step count) the strain energy, sensitivity and stress of its solution, and the Helmholtz filter forward and backward; every array also compared byte for byte; 19 x 15 x 15 elements, 15360 degrees of freedom (15 blocks of a sum), 450 outside the part, random densities, held displacements |
| 16 | 3D solver against a sparse direct solution: uniform, 50 x 20 x 20 | independent computation | 4.97e-12 | ≤ 1e-07 | pass | \|u - u_direct\| / \|u_direct\|; uniform density 1; 3 multigrid levels, 0 coarsest degrees of freedom left out, 24 steps to 1e-10; largest entry difference over the largest entry 4.81e-12 |
| 16 | 3D solver against a sparse direct solution: lattice, 50 x 20 x 20 | independent computation | 1.15e-11 | ≤ 1e-07 | pass | \|u - u_direct\| / \|u_direct\|; lattice of bars, 25% solid; 3 multigrid levels, 0 coarsest degrees of freedom left out, 57 steps to 1e-10; largest entry difference over the largest entry 1.08e-11 |
| 16 | 3D solver against a sparse direct solution: irregular, 50 x 20 x 20 | independent computation | 2.34e-09 | ≤ 1e-07 | pass | \|u - u_direct\| / \|u_direct\|; irregular, cut at 40 %, 44% solid in one face-joined group; 3 multigrid levels, 0 coarsest degrees of freedom left out, 112 steps to 1e-10; largest entry difference over the largest entry 2.41e-09 |
| 16 | 3D solver against a sparse direct solution: L-shaped, 35 x 25 x 27 | independent computation | 4.78e-12 | ≤ 1e-07 | pass | \|u - u_direct\| / \|u_direct\|; L-shaped region of 18100 elements, held on a patch of the face x = 35 that ends one node short of the edge y = 25; 3 multigrid levels, 0 coarsest degrees of freedom left out, 27 steps to 1e-10; largest entry difference over the largest entry 4.73e-12 |
| 17 | 3D solver steps on a uniform cantilever at growing sizes | convergence | 20 | ≤ 30 | pass | largest step count; 24 x 12 x 12: 18 steps, 55 products; 48 x 24 x 24: 19 steps, 58 products; 96 x 48 x 48: 20 steps, 61 products; largest over smallest 1.11 (limit 1.5): within |
| 17 | 3D solver steps on the lattice 0-and-1 design, 48 x 24 x 24 | convergence | 92 | ≤ 300 | pass | lattice of bars, 22% solid; 277 products with the fine stiffness matrix |
| 17 | 3D solver steps on the irregular 0-and-1 design, 48 x 24 x 24 | convergence | 131 | ≤ 600 | pass | irregular, cut at 41 %, 45% solid in one face-joined group; 394 products with the fine stiffness matrix |
| 18 | 3D cantilever, length 8 x depth, square section: deflection against Timoshenko beam theory | closed form | 0.0112 | < 0.02 | pass | relative error with 16 elements through the depth; with 4, 8, 16: 0.0451, 0.0186, 0.0112; the error must also shrink: it does |
| 19 | Gradient of compliance through the Helmholtz filter in 3D against central differences | gradient | 5.95e-08 | ≤ 1e-05 | pass | 120 elements of a 3D block with an inactive pocket, two load cases, random density, solves at 1e-12; largest difference relative to the gradient's size; worst element on its own scale 3.33e-07 (reported, not judged) |
| 19 | Gradient of mass through the Helmholtz filter in 3D against central differences | gradient | 3.25e-08 | ≤ 1e-05 | pass | 120 elements of a 3D block with an inactive pocket, two load cases, random density, solves at 1e-12; largest difference relative to the gradient's size; worst element on its own scale 3.25e-08 (reported, not judged) |
| 20 | Helmholtz filter keeps the sum of the field | closed form | 1.36e-13 | ≤ 1e-08 | pass | relative difference of the sums; 24 x 12 x 10 elements, 480 outside the part, radius 2.5 h, random fields |
| 20 | Helmholtz filter is symmetric: <F x, y> = <x, F y> | closed form | 2.84e-13 | ≤ 1e-08 | pass | relative difference; 24 x 12 x 10 elements, 480 outside the part, radius 2.5 h, random fields |
| 20 | Helmholtz filter on a long bar against the 1D solution h / (2 r) exp(-\|x\| / r) | closed form | 0.0156 | ≤ 0.02 | pass | largest relative difference at r, 2r and 3r on both sides (0.0104, 0.013, 0.0156); r = 4 h, 91 x 1 x 1 elements, 11.2 r on each side; lumped mass matrix, which keeps every filtered field in [0, 1] for any radius |
| 21 | Projection: derivative against central differences | gradient | 3.71e-10 | ≤ 1e-07 | pass | largest difference relative to the derivative's size over beta in 1, 8, 32; eta in 0.25, 0.5, 0.75 (worst: beta 32, eta 0.25); 1000 random values, step 1e-6 |
| 21 | Projection keeps 0 and 1: P(0) = 0, P(1) = 1 | closed form | 0 | ≤ 1e-15 | pass | largest of \|P(0)\| and \|P(1) - 1\| over beta in 1, 8, 32; eta in 0.25, 0.5, 0.75 |
| 22 | Gradient of compliance on the eroded design (Helmholtz filter, projection) against central differences | gradient | 1.73e-07 | ≤ 1e-05 | pass | 120 elements of a 3D block with an inactive pocket, two load cases, random density; Helmholtz filter, then the projection of the eroded design (beta 8, threshold 0.75); solves at 1e-12; worst element on its own scale 1.17e-06 (reported, not judged) |
| 22 | Gradient of mass on the dilated design (Helmholtz filter, projection) against central differences | gradient | 1.46e-07 | ≤ 1e-05 | pass | 120 elements of a 3D block with an inactive pocket, two load cases, random density; Helmholtz filter, then the projection of the dilated design (beta 8, threshold 0.25); solves at 1e-12; worst element on its own scale 3.82e-07 (reported, not judged) |
| 23 | Robust MBB beam 120 x 40: mass of the nominal design against the asked mass | property of the design | 0.00267 | ≤ 0.01 | pass | nominal 2410 g, asked 2400 g; masses eroded, nominal, dilated: 1950, 2410, 2840 g; last bound on the dilated mass 2840 g; MBB beam 120 x 40, density filter radius 4.8, spread 0.25, beta 1 to 32 doubling every 50 passes (300 passes), MMA |
| 23 | Robust MBB beam 120 x 40: non-discreteness of the nominal design, sum 4 rho (1 - rho) / n | property of the design | 0.0131 | ≤ 0.1 | pass | over all 4800 elements (Sigmund 2007); MBB beam 120 x 40, density filter radius 4.8, spread 0.25, beta 1 to 32 doubling every 50 passes (300 passes), MMA |
| 23 | Robust MBB beam 120 x 40: compliance eroded >= nominal >= dilated | property of the design | 0 | ≤ 0 | pass | compliance eroded 236, nominal 195, dilated 175 N mm; measured: the largest relative step out of that order, 0 when in order; MBB beam 120 x 40, density filter radius 4.8, spread 0.25, beta 1 to 32 doubling every 50 passes (300 passes), MMA |

49 passed, 0 failed, 0 not run.

## Regenerating this page

From the repository, with the package and SciPy installed:

    python -m michell.verification --reference tests/reference --out docs/verification.md

Rows 6, 8 and 10 read the output of the 88-line code from `tests/reference`; row 9 solves its problems a
second time with SciPy, row 14 assembles the 3D stiffness matrix with SciPy, and row 16 solves the 3D system
directly with SciPy. Without them those rows are listed as not run. The command exits with status 1 when a row fails (and, with `--require-all`, when a row
was not run). The test suite runs the same functions.
