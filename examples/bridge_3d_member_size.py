"""A 3D bridge with a minimum member size of 5 mm, from one call of ``michell.recipes.stiff_part``.

A block of 50 by 24 by 10 cube elements of side 1 mm (steel, x along the span, z up), clamped on its faces
x = 0 and x = 50, carries 1000 N down on a pad of 6 by 6 elements in the middle of its top face. End plates
2 elements thick at the clamped faces and the pad, 6 by 6 by 2 elements, are held solid. The stiffest
bridge with 30 % of the block's mass and members at least 5 mm thick: the robust formulation forms the
layout, then the member-size limit reshapes its members (the procedure is written out in the docstring of
``stiff_part``). What the density filter sees beyond the faces: the clamped x faces continue into the
support, so they are closed; the free y and z faces are empty. Verification row 28 runs the same part.
Run from the repository:

    python examples/bridge_3d_member_size.py [--threads 1] [--vtk bridge.vtk] [--save bridge]

It prints a line per pass and, last, a JSON summary of the run. With ``--vtk`` it also writes the nominal
design, the design at the end of the robust step and the held elements as one VTK file, which ParaView opens
(``michell.export.write_vtk``). With ``--save`` it also writes the whole outcome as ``bridge.npz`` (every
field) and ``bridge.json`` (the values, the limits and whether they were met, why the run stopped and the
record of every pass), which ``michell.export.read_result`` reads back. The supports, the load and the held
elements are picked by position with ``michell.select``.
"""

import argparse
import json
import math
import time

import numpy as np

import michell

parser = argparse.ArgumentParser(description="Optimise a 3D bridge with a minimum member size.")
parser.add_argument("--threads", type=int, default=1, help="the number of threads of the solves and the filter")
parser.add_argument("--vtk", metavar="FILE", help="also write the design as a VTK file (for ParaView) to FILE")
parser.add_argument(
    "--save", metavar="PATH", help="also write the outcome to PATH.npz (the fields) and PATH.json (the rest)"
)
args = parser.parse_args()

nx, ny, nz = 50, 24, 10
grid = michell.Grid3d(nx, ny, nz, 1.0)

# Positions in mm (the elements have a side of 1 mm); a box is a (low, high) pair per direction.
everywhere = (-math.inf, math.inf)
pad = ((22.0, 28.0), (9.0, 15.0), (8.0, 10.0))  # 6 x 6 elements in the middle of the top face, 2 deep

# Clamped on the faces x = 0 and x = nx: every degree of freedom of their nodes held.
supports = np.concatenate([michell.select.face_nodes(grid, "x-"), michell.select.face_nodes(grid, "x+")])
held = michell.select.held(grid, supports)

# 1000 N down (-z) on the nodes of the pad's top, as a uniform pressure: each 1 mm square of the pad gives a
# quarter of its force to each of its corners.
pad_nodes = michell.select.face_nodes(grid, "z+", within=pad)
loads = michell.select.load(grid, pad_nodes, (0.0, 0.0, -1000.0))

# Held solid: the end plates and the pad under the load.
kept = (
    michell.select.elements_in(grid, ((0.0, 2.0), everywhere, everywhere))
    | michell.select.elements_in(grid, ((nx - 2.0, nx), everywhere, everywhere))
    | michell.select.elements_in(grid, pad)
)
keep_solid = michell.KeepSolid(kept, grid)

steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
problem = michell.Problem(grid, held, steel, [michell.LoadCase("pad", loads)], threads=args.threads)
mass = 0.30 * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
outside = {"x-": "closed", "x+": "closed", "y-": "empty", "y+": "empty", "z-": "empty", "z+": "empty"}


def show(step, number, record):
    limits = ", ".join(f"{value:.4g} (at most {bound:.4g})" for value, bound in zip(record.limits, record.bounds))
    print(f"{step:11} pass {number:3}, beta {record.beta:4g}: compliance {record.goal:.5g} N mm, limits {limits}")


started = time.perf_counter()
part = michell.stiff_part(problem, mass, 5.0, keep_solid, outside=outside, callback=show)
seconds = time.perf_counter() - started

result = part.result
compliance_first = michell.dot(loads, problem.solve(part.first_step_nominal, loads)[0])
summary = {
    "passes": len(part.passes),
    "seconds": seconds,
    "threads": result.threads,
    "compliance": result.goal,
    "compliance_after_the_robust_step": compliance_first,
    "mass": result.limits[0],
    "mass_asked": mass,
    "member_size_value": result.limits[1],
    "eps": part.eps,
    "limits_met": result.limits_met,
    "stop_reason": result.stop_reason,
    "solid_elements": int((part.nominal >= 0.5).sum()),
}
if args.vtk:
    michell.write_vtk(
        args.vtk, grid, nominal=part.nominal, robust_step=part.first_step_nominal, held_solid=keep_solid.mask
    )
if args.save:
    michell.write_result(args.save, part)
print(json.dumps(summary))
