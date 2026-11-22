"""michell.export.write_vtk: the file of a 3 x 2 x 2 grid with two fields parsed here, header and binary
block, its values and their order (x fastest) compared exactly; a 2D grid; refused arguments. And the
GridTooCoarse warning of stiff_part: given below five elements per member size, not at or above."""

import warnings

import numpy as np
import pytest

import michell


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
