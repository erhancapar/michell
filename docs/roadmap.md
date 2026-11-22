# Roadmap

michell is developed in milestones. A milestone is complete when its verification cases pass in continuous
integration, and a feature is listed as supported only from that point on. The current state of every case is
in the [verification table](verification.md). No dates are given.

| Milestone | Scope | Completion criteria |
|---|---|---|
| 1. 2D engine | Plane-stress elements, a direct solver, the density filter, the optimality-criteria and MMA optimisers, the compliance and mass responses | The published 88-line MBB result is reproduced; every gradient agrees with finite differences; the closed-form cases pass |
| 2. 3D engine | Matrix-free 3D elements, multithreading, a multigrid solver, the PDE filter, projection and the robust formulation | A 3D cantilever agrees with an independent finite-element code on the same mesh; run time and memory are measured and published |
| 3. Model in, model out | STEP and STL input; symmetry; a problem check with time and memory estimates; STL and 3MF output; a report. Already available: several load cases, regions held solid, elements outside the part, the minimum member size | The quick-start produces a part from a CAD model without further tools; the benchmark parts reproduce their reference results |
| 4. Version 0.1 | Stress and deflection constraints, a buckling check, heat conduction, sweeps over mass targets and materials, a server interface for automated workflows | The closed-form stress and heat-conduction cases pass; the comparison with other codes is published |
| 5. Additive manufacturing | Overhang and build-direction constraints for fused-filament printing; strength across layers | Predicted and measured stiffness of printed test parts agree within a limit stated in advance |
| 6. Further physics | Natural frequency, buckling constraints, thermal stress | Each module passes its verification cases |
| 7. Further manufacturing methods and integration | Constraints for casting, milling and extrusion; a C interface | Each module passes its verification cases |

Status: milestone 1 is complete. Of milestone 2, the 3D engine is available and its verification cases pass,
among them the 3D cantilever against CalculiX on the same mesh (verification row 24); its run time and memory
are not yet published. Milestones 3 to 7 are planned.
