"""michell.export.write_vtk: the file of a 3 x 2 x 2 grid with two fields parsed here, header and binary
block, its values and their order (x fastest) compared exactly; a 2D grid; refused arguments. And the
GridTooCoarse warning of stiff_part: given below five elements per member size, not at or above.
michell.export.write_result and read_result: a Result of OC, a Result with named designs and a StiffPart
written and read back (the arrays bit for bit, the JSON values equal), numbers that are not finite written as
null, both files read by numpy.load and json.load alone."""

import dataclasses
import json
import math
import warnings

import numpy as np
import pytest

import michell
from michell.gradient_check import small_problem
from michell.verification.recipe_rows import bridge


def parse_vtk(data: bytes):
    """The header lines up to FIELD and the fields of a legacy binary VTK file, parsed by hand."""
    lines, position = [], 0
    while True:
        end = data.index(b"\n", position)
        lines.append(data[position:end].decode("ascii"))
        position = end + 1
        if lines[-1].startswith("FIELD"):
            break
    count = int(lines[-2].split()[1])
    fields = {}
    while position < len(data):
        end = data.index(b"\n", position)
        name, components, tuples, kind = data[position:end].decode("ascii").split()
        assert (components, int(tuples), kind) == ("1", count, "double")
        position = end + 1
        fields[name] = np.frombuffer(data[position : position + 8 * count], dtype=">f8")
        position += 8 * count
        assert data[position : position + 1] == b"\n"
        position += 1
    assert int(lines[-1].split()[2]) == len(fields)
    return lines, fields


def test_write_vtk_of_a_3d_grid_with_two_fields(tmp_path):
    grid = michell.Grid3d(3, 2, 2, 0.5)
    density = np.zeros(grid.n_elements)
    held = np.zeros(grid.n_elements, dtype=bool)
    for k in range(2):
        for j in range(2):
            for i in range(3):
                density[grid.element(i, j, k)] = 0.01 * i + 0.1 * j + 1.0 / 3 * k
                held[grid.element(i, j, k)] = (i + j + k) % 2 == 0
    path = tmp_path / "part.vtk"
    michell.write_vtk(path, grid, density=density, held=held)
    lines, fields = parse_vtk(path.read_bytes())
    assert lines == [
        "# vtk DataFile Version 3.0",
        "michell fields",
        "BINARY",
        "DATASET STRUCTURED_POINTS",
        "DIMENSIONS 4 3 3",
        "ORIGIN 0 0 0",
        "SPACING 0.5 0.5 0.5",
        "CELL_DATA 12",
        "FIELD FieldData 2",
    ]
    assert list(fields) == ["density", "held"]
    # The cell at position p of the file is (i, j, k) with p = i + 3 (j + 2 k): x fastest, then y, then z.
    for p in range(12):
        i, j, k = p % 3, (p // 3) % 2, p // 6
        assert fields["density"][p] == 0.01 * i + 0.1 * j + 1.0 / 3 * k
        assert fields["held"][p] == (1.0 if (i + j + k) % 2 == 0 else 0.0)


def test_write_vtk_of_a_2d_grid(tmp_path):
    grid = michell.Grid(4, 3, 2.0, 1.0)
    rho = np.linspace(0.0, 1.0, grid.n_elements)
    path = tmp_path / "beam.vtk"
    michell.export.write_vtk(str(path), grid, rho=rho)
    lines, fields = parse_vtk(path.read_bytes())
    assert lines[4:7] == ["DIMENSIONS 5 4 1", "ORIGIN 0 0 0", "SPACING 2.0 2.0 2.0"]
    assert np.array_equal(fields["rho"], rho)


def test_write_vtk_refuses(tmp_path):
    grid = michell.Grid3d(3, 2, 2, 1.0)
    path = tmp_path / "x.vtk"
    for call in (
        lambda: michell.write_vtk(path, "grid", rho=np.zeros(12)),
        lambda: michell.write_vtk(path, grid),
        lambda: michell.write_vtk(path, grid, rho=np.zeros(11)),
        lambda: michell.write_vtk(path, grid, rho=np.zeros((2, 6))),
        lambda: michell.write_vtk(path, grid, rho=np.array(["a"] * 12)),
        lambda: michell.write_vtk(path, grid, **{"two words": np.zeros(12)}),
    ):
        with pytest.raises(michell.EngineError) as caught:
            call()
        assert caught.value.code == "invalid_input"
    assert not path.exists()


def coarse_warnings(h, size_mm):
    """The GridTooCoarse warnings of a short stiff_part run on a small 2D cantilever of side h."""
    grid = michell.Grid(16, 8, h, 1.0)
    left = michell.select.face_nodes(grid, "x-")
    tip = michell.select.face_nodes(grid, "x+", within=((0.0, 1e9), (0.0, 0.0)))
    steel = michell.Material(210000.0, 0.3, 7.85)
    loads = michell.select.load(grid, tip, (0.0, -1.0))
    problem = michell.Problem(grid, michell.select.held(grid, left), steel, [michell.LoadCase("tip", loads)])
    mass = 0.5 * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        part = michell.stiff_part(problem, mass, size_mm, schedule=((1.0, 1),), second_step_passes=1)
    assert len(part.passes) == 2  # the run proceeds
    return [w for w in caught if issubclass(w.category, michell.GridTooCoarse)]


def test_stiff_part_warns_below_five_elements_per_member_size():
    assert issubclass(michell.GridTooCoarse, UserWarning)
    (warning,) = coarse_warnings(0.5, 2.4)
    text = str(warning.message)
    assert "4.8 element sides" in text and "thicker than asked" in text and "finer grid" in text
    assert warning.filename == __file__
    assert len(coarse_warnings(1.0, 4.99)) == 1
    assert coarse_warnings(0.5, 2.5) == []  # exactly five elements
    assert coarse_warnings(0.07, 0.35) == []  # five elements, though 0.35 / 0.07 = 4.999999999999999
    assert coarse_warnings(1.0, 6.0) == []


def part_mass(problem):
    grid = problem.grid
    return problem.material.mass_g(grid.voxel_volume_mm3) * float(np.asarray(grid.active).sum())


def oc_result():
    """A Result of 5 OC passes on the small 2D problem of the gradient checks."""
    problem, _, filters = small_problem()
    mass = 0.4 * part_mass(problem)
    return michell.optimise(
        problem, michell.Compliance(), [michell.Limit(michell.Mass(), at_most=mass)], filters,
        michell.OptimalityCriteria(), np.full(problem.grid.n_elements, 0.4), passes=5,
    )  # fmt: skip


def designs_result():
    """A Result of 3 MMA passes with two named designs, the filtered field and its projection."""
    problem, x, filters = small_problem()
    projection = michell.Projection(4.0)
    mass = 0.6 * part_mass(problem)
    return michell.optimise(
        problem, michell.on("nominal", michell.Compliance()),
        [michell.Limit(michell.on("nominal", michell.Mass()), at_most=mass)], filters, michell.MovingAsymptotes(),
        x, passes=3, designs={"filtered": [], "nominal": [projection]},
    )  # fmt: skip


def tiny_stiff_part():
    """stiff_part on the bridge of row 28 at 16 x 8 x 6 elements, a few passes; the member size of 2 elements
    gives GridTooCoarse, asserted here."""
    part = bridge(16, 8, 6)
    with pytest.warns(michell.GridTooCoarse):
        return michell.stiff_part(
            part.problem, part.mass_g / 0.3 * 0.5, 2.0, part.keep_solid, schedule=((1.0, 2), (32.0, 1)),
            second_step_passes=2,
        )  # fmt: skip


def expected_json(result):
    """The JSON content write_result should give, built from the object itself."""
    run = result.result if isinstance(result, michell.StiffPart) else result

    def number(value):
        return None if value is None or not math.isfinite(value) else float(value)

    expected = {
        "format": 1,
        "kind": type(result).__name__,
        "version": michell.__version__,
        "goal": number(run.goal),
        "limits": None if run.limits is None else [number(value) for value in run.limits],
        "bounds": None,
        "limits_met": None if run.limits_met is None else list(run.limits_met),
        "feasible": run.feasible,
        "stop_reason": run.stop_reason,
        "threads": run.threads,
        "designs": list(run.designs),
    }
    if isinstance(result, michell.StiffPart):
        expected["bounds"] = list(result.passes[-1].bounds)
        expected["passes"] = [
            {"step": p.step, "number": p.number, "beta": p.beta, "goal": number(p.goal),
             "limits": [number(v) for v in p.limits], "bounds": list(p.bounds), "change": number(p.change)}
            for p in result.passes
        ]  # fmt: skip
        expected.update(eps=result.eps, threshold=result.threshold, value_at_switch=result.value_at_switch)
        expected["scale"] = result.scale
    else:
        expected["passes"] = [
            {"number": p.number, "goal": number(p.goal), "limits": [number(v) for v in p.limits],
             "change": number(p.change), "solver_steps": p.solver_steps,
             "design_solver_steps": dict(p.design_solver_steps)}
            for p in run.passes
        ]  # fmt: skip
    return expected


def expected_arrays(result):
    run = result.result if isinstance(result, michell.StiffPart) else result
    arrays = {"x": run.x, "density": run.density, **{f"design_{name}": f for name, f in run.designs.items()}}
    if isinstance(result, michell.StiffPart):
        arrays.update(
            filtered=result.filtered, nominal=result.nominal, first_step_nominal=result.first_step_nominal,
            thresholds=result.member_size.thresholds,
        )  # fmt: skip
    return arrays


def assert_round_trip(tmp_path, result, name):
    path = tmp_path / name
    michell.write_result(path, result)
    read = michell.read_result(path)
    arrays = read.pop("arrays")
    assert read == expected_json(result)
    expected = expected_arrays(result)
    assert sorted(arrays) == sorted(expected)
    for key, values in expected.items():
        assert arrays[key].dtype == values.dtype and arrays[key].tobytes() == np.asarray(values).tobytes(), key
    # Either file on its own, by numpy.load and json.load alone.
    with open(tmp_path / f"{name}.json", encoding="utf-8") as file:
        assert json.load(file) == read
    with np.load(tmp_path / f"{name}.npz", allow_pickle=False) as data:
        assert sorted(data.files) == sorted(expected)
        assert all(np.array_equal(data[key], values) for key, values in expected.items())
    return read


def test_write_result_of_an_oc_run(tmp_path):
    result = oc_result()
    assert result.limits_met == (True,)
    read = assert_round_trip(tmp_path, result, "oc")
    assert read["kind"] == "Result" and read["designs"] == [] and len(read["passes"]) == 5
    assert read["bounds"] is None and read["feasible"] is True


def test_write_result_with_named_designs(tmp_path):
    result = designs_result()
    read = assert_round_trip(tmp_path, result, "designs")
    assert read["designs"] == ["filtered", "nominal"]
    assert read["passes"][0]["design_solver_steps"] == {"nominal": 0}


# A run of a few passes does not meet the member-size limit; the warning that says so is expected here.
@pytest.mark.filterwarnings("ignore::michell.LimitNotMet")
def test_write_result_of_a_stiff_part(tmp_path):
    part = tiny_stiff_part()
    read = assert_round_trip(tmp_path, part, "part")
    assert read["kind"] == "StiffPart" and read["bounds"] == [part.passes[-1].bounds[0], part.eps]
    assert [p["step"] for p in read["passes"]] == ["robust"] * 3 + ["member size"] * 2


def test_numbers_that_are_not_finite_are_written_as_null(tmp_path):
    result = dataclasses.replace(oc_result(), goal=float("nan"), limits=(float("inf"),))
    michell.write_result(tmp_path / "odd", result)
    text = (tmp_path / "odd.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    read = michell.read_result(tmp_path / "odd")
    assert read["goal"] is None and read["limits"] == [None]
    assert read["arrays"]["x"].tobytes() == result.x.tobytes()


def test_write_result_refuses_other_objects(tmp_path):
    with pytest.raises(michell.EngineError) as caught:
        michell.write_result(tmp_path / "nothing", {"x": np.zeros(3)})
    assert caught.value.code == "invalid_input"
    assert not (tmp_path / "nothing.json").exists() and not (tmp_path / "nothing.npz").exists()
