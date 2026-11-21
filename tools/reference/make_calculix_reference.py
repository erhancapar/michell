"""Regenerates tests/reference/calculix/ by solving the 3D cantilever of verification row 24 with CalculiX.

CalculiX (G. Dhondt, www.calculix.de) is an independent finite-element program. This script builds the design
of row 24 with the same function the row uses (``michell.verification.calculix_rows.calculix_design``),
writes it as a CalculiX input deck, runs the CalculiX solver ``ccx`` and keeps only numbers:

- ``cantilever_48x24x24_irregular.json``: the grid, the material, the load, the design and its fingerprint,
  the compliance f^T u, CalculiX's total strain energy and reaction (two checks of the run), the solver;
- ``cantilever_48x24x24_irregular_loaded_nodes.csv``: the displacements of the loaded nodes, as CalculiX
  printed them;
- ``PROVENANCE.txt``: the CalculiX version, the date and the commands.

The deck has one C3D8 element (the 8-node brick with 2 x 2 x 2 Gauss integration) per solid voxel and no
element for a void, and only the nodes those elements use. CalculiX node n + 1 is michell node n (CalculiX
numbers from 1), element e + 1 is michell element e. The equations are solved by the direct solver SPOOLES.

Needs the michell package and CalculiX. Usage, from the repository:

    python tools/reference/make_calculix_reference.py [--ccx PATH] [--work FOLDER]

``--ccx`` is the CalculiX solver (default: ``ccx`` on the PATH); ``--work`` is the folder for the deck and
CalculiX's output files (default: a temporary folder, removed afterwards).
"""

import argparse
import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from michell.verification.calculix_rows import CALCULIX_CASE, CALCULIX_SHAPE, calculix_design, design_fingerprint
from michell.verification.designs3d import E, NU

JOB = "cantilever"
OUT = Path(__file__).resolve().parent.parent.parent / "tests" / "reference" / "calculix"

# The eight nodes of a C3D8 element as steps (di, dj, dk) from its lowest corner: the face z = k
# counter-clockwise seen from +z, then the face z = k + 1 in the same order (the node order of the C3D8
# element in the CalculiX manual; CalculiX stops on an element whose Jacobian is not positive).
CORNERS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def lines_of_eight(numbers):
    """Lists of at most eight numbers: CalculiX reads at most 16 entries per line of a set."""
    return [numbers[start : start + 8] for start in range(0, len(numbers), 8)]


def write_deck(path, setup, solid):
    """Writes the input deck. Returns the michell indices of the loaded nodes, their loads in z, the number
    of nodes written and the number of held nodes."""
    grid = setup.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    elements = np.flatnonzero(solid)
    connectivity = []
    for element in elements:
        k, rest = divmod(int(element), nx * ny)
        j, i = divmod(rest, nx)
        connectivity.append([grid.node(i + di, j + dj, k + dk) for di, dj, dk in CORNERS])
    used = sorted({node for nodes in connectivity for node in nodes})
    held = [node for node in used if setup.held[3 * node]]
    loaded = [node for node in used if setup.loads[3 * node + 2] != 0.0]
    if np.count_nonzero(setup.loads) != len(loaded):
        raise ValueError("a load sits on a node that no solid element uses")

    text = ["*HEADING", f"michell verification row 24: {nx} x {ny} x {nz} cantilever, irregular design", "*NODE"]
    for node in used:
        i, rest = node % (nx + 1), node // (nx + 1)
        j, k = rest % (ny + 1), rest // (ny + 1)
        h = grid.h
        text.append(f"{node + 1}, {i * h!r}, {j * h!r}, {k * h!r}")
    text.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for element, nodes in zip(elements, connectivity):
        text.append(", ".join(str(n) for n in [element + 1] + [node + 1 for node in nodes]))
    for name, nodes in (("HELD", held), ("LOADED", loaded)):
        text.append(f"*NSET, NSET={name}")
        text += [", ".join(str(node + 1) for node in line) for line in lines_of_eight(nodes)]
    text += [
        "*BOUNDARY",
        "HELD, 1, 3",
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        f"{E!r}, {NU!r}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
        "*STEP",
        "*STATIC, SOLVER=SPOOLES",
        "*CLOAD",
    ]
    text += [f"{node + 1}, 3, {float(setup.loads[3 * node + 2])!r}" for node in loaded]
    text += [
        "*NODE PRINT, NSET=LOADED",
        "U",
        "*NODE PRINT, NSET=HELD, TOTALS=ONLY",
        "RF",
        "*EL PRINT, ELSET=EALL, TOTALS=ONLY",
        "ELSE",
        "*END STEP",
    ]
    Path(path).write_text("\n".join(text) + "\n", encoding="ascii")
    return np.array(loaded), np.array([setup.loads[3 * node + 2] for node in loaded]), len(used), len(held)


def section(lines, title):
    """The lines of numbers that follow the header line of the .dat file that starts with ``title``."""
    for start, line in enumerate(lines):
        if line.strip().startswith(title):
            break
    else:
        raise ValueError(f"no section {title!r} in CalculiX's output")
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        rows.append(line.split())
    return rows


def significant_digits(text):
    """The number of significant digits of a number printed as 1.234567E-03."""
    mantissa = re.split("[eE]", text)[0].lstrip("+-")
    return len(mantissa.replace(".", "").lstrip("0"))


def main():
    parser = argparse.ArgumentParser(description="Solve the cantilever of verification row 24 with CalculiX.")
    parser.add_argument("--ccx", default=shutil.which("ccx"), help="the CalculiX solver (default: ccx on the PATH)")
    parser.add_argument("--work", type=Path, help="folder for the deck and CalculiX's files (default: temporary)")
    args = parser.parse_args()
    if args.ccx is None:
        sys.exit("CalculiX's solver ccx was not found on the PATH; give it with --ccx")

    setup, solid = calculix_design()
    work = args.work or Path(tempfile.mkdtemp(prefix="michell-calculix-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        loaded, forces, n_nodes, n_held = write_deck(work / f"{JOB}.inp", setup, solid)
        version = subprocess.run([args.ccx, "-v"], check=False, capture_output=True, text=True, cwd=work).stdout.strip()
        run = subprocess.run([args.ccx, "-i", JOB], check=False, capture_output=True, text=True, cwd=work)
        if run.returncode != 0 or "*ERROR" in run.stdout or not (work / f"{JOB}.dat").is_file():
            sys.exit(f"CalculiX failed (status {run.returncode}):\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        if "spooles" not in run.stdout.lower():
            sys.exit(f"CalculiX did not report the SPOOLES solver:\n{run.stdout[-3000:]}")
        lines = (work / f"{JOB}.dat").read_text().splitlines()
    finally:
        if args.work is None:
            shutil.rmtree(work)

    printed = {int(row[0]) - 1: row[1:4] for row in section(lines, "displacements (vx,vy,vz) for set LOADED")}
    if sorted(printed) != sorted(int(node) for node in loaded):
        raise ValueError("CalculiX printed displacements for other nodes than the loaded ones")
    uz = np.array([float(printed[int(node)][2]) for node in loaded])
    compliance = float(forces @ uz)  # the loads act in z only
    energy = float(section(lines, "total internal energy for set EALL")[0][0])
    reaction = [float(value) for value in section(lines, "total force (fx,fy,fz) for set HELD")[0]]
    digits = min(significant_digits(value) for values in printed.values() for value in values)
    if abs(2 * energy - compliance) > 1e-5 * compliance or abs(reaction[2] - 1.0) > 1e-5:
        raise ValueError(f"the run does not check out: 2 x energy {2 * energy}, compliance {compliance}, {reaction}")

    nx, ny, nz = CALCULIX_SHAPE
    program = "CalculiX " + version.split("Version")[-1].strip()  # ccx -v prints "This is Version 2.23"
    case = {
        "program": program,
        "solver": "SPOOLES",
        "shape": list(CALCULIX_SHAPE),
        "voxel_side_mm": setup.grid.h,
        "youngs_modulus": E,
        "poisson_ratio": NU,
        "held": "every displacement of the nodes on the face x = 0",
        "load": f"1 N in -z spread over the face x = {nx} by area (michell.verification.designs3d.face_load)",
        "design": f"michell.verification.designs3d.irregular(({nx}, {ny}, {nz})): {setup.description}",
        "solid_elements": int(solid.sum()),
        "solid_mask_sha256": design_fingerprint(solid),
        "calculix_nodes": n_nodes,
        "held_nodes": n_held,
        "loaded_nodes": len(loaded),
        "printed_digits": digits,
        "compliance": compliance,
        "twice_the_strain_energy": 2 * energy,
        "reaction_z": reaction[2],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{CALCULIX_CASE}.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    with open(OUT / f"{CALCULIX_CASE}_loaded_nodes.csv", "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["node", "ux", "uy", "uz"])
        for node in loaded:
            writer.writerow([int(node)] + printed[int(node)])
    provenance = [
        f"program: {program} (G. Dhondt, www.calculix.de)",
        "solver: SPOOLES (direct), *STATIC, SOLVER=SPOOLES",
        f"date: {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}",
        f"command: ccx -i {JOB}  (in the work folder, on the deck written by this script)",
        "written by: python tools/reference/make_calculix_reference.py [--ccx PATH] [--work FOLDER]",
        f"case: {CALCULIX_CASE}: {case['design']}; {case['solid_elements']} C3D8 elements, {n_nodes} nodes",
    ]
    (OUT / "PROVENANCE.txt").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print(f"{CALCULIX_CASE}: compliance {compliance!r} N mm, 2 x strain energy {2 * energy!r}, reaction {reaction}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
