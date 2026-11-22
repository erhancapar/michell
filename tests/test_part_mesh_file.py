"""The part's mesh through the problem file: ``[output] mesh``, ``michell check`` and ``michell run``, and the
``mesh`` section of ``result.json``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import michell
from michell import cli, part_mesh, problem_file
from michell.mesh import read_3mf, read_stl

# The tests run in importlib mode, in which the tests folder is not on sys.path: the helpers of the problem-file
# tests are loaded from their file.
_spec = importlib.util.spec_from_file_location("problem_file_helpers", Path(__file__).with_name("test_problem_file.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
BAR, edited, write = _helpers.BAR, _helpers.edited, _helpers.write

MESH = 'max_fraction = 0.5\n\n[output]\nmesh = ["stl", "3mf"]\n'


def test_check_says_what_will_be_written(tmp_path, capsys):
    plain = michell.prepare(write(tmp_path, BAR))
    assert plain.problem_file.mesh == () and plain.problem_file.files == ("result.json", "result.npz")
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", MESH), "mesh.toml")
    prepared = michell.prepare(file)
    output = prepared.summary()["output"]
    assert output["mesh"] == ["stl", "3mf"] and output["folder"] == str(tmp_path / "out")
    assert output["files"] == ["result.json", "result.npz", "part.stl", "part.3mf"]
    assert cli.main(["check", str(file)]) == 0
    assert f"run writes into {tmp_path / 'out'}: result.json, result.npz, part.stl, part.3mf" in capsys.readouterr().out


def test_run_writes_the_mesh_and_its_section(tmp_path):
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", MESH))
    outcome = michell.run(file)
    assert outcome.limits_met and outcome.mesh_delivered and outcome.mesh_refused is None
    finished = outcome.finished
    assert all(gate.passed for gate in finished.gates)
    assert set(outcome.paths) == {"json", "npz", "stl", "3mf"}
    stl, three_mf = read_stl(outcome.paths["stl"]), read_3mf(outcome.paths["3mf"])
    assert np.array_equal(three_mf.vertices, finished.mesh.vertices)
    assert np.array_equal(three_mf.triangles, finished.mesh.triangles)
    assert len(stl.triangles) == len(finished.mesh.triangles)
    rounded = finished.mesh.vertices.astype(np.float32).astype(np.float64)
    assert np.array_equal(np.unique(stl.vertices, axis=0), np.unique(rounded, axis=0))
    assert np.array_equal(stl.vertices[stl.triangles], rounded[finished.mesh.triangles])
    result = michell.read_result(tmp_path / "out" / "result")
    section = result["mesh"]
    assert section["written"] and section["triangles"] == len(finished.mesh.triangles)
    assert section["level"] == finished.level and section["delivered"]["within"]
    assert [gate["name"] for gate in section["gates"]] == [gate.name for gate in finished.gates]
    assert section["stl_distinct_float32_vertices"] == len(finished.mesh.vertices)
    assert result["files"] == ["result.json", "result.npz", "part.stl", "part.3mf"]
    assert set(result["seconds"]) == {"prepare", "optimise", "mesh"}


def test_a_refused_mesh_is_not_written_and_the_run_exits_1(tmp_path, monkeypatch, capsys):
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", MESH))

    def refuse(part, setup, threads=None):
        raise part_mesh.MeshRefused("one body", "made to fail", {"bodies": 2.0})

    monkeypatch.setattr(problem_file, "finish", refuse)
    status = cli.main(["run", str(file), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert status == cli.LIMITS_NOT_MET and data["exit"] == 1
    assert data["result"]["mesh"]["written"] is False and data["result"]["mesh"]["refused"]["gate"] == "one body"
    assert {path.name for path in (tmp_path / "out").iterdir()} == {"result.json", "result.npz"}
    section = michell.read_result(tmp_path / "out" / "result")["mesh"]
    assert section["written"] is False and section["refused"]["details"] == {"bodies": 2.0}


def test_a_delivered_part_that_differs_makes_the_run_exit_1(tmp_path, monkeypatch, capsys):
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", MESH))
    monkeypatch.setattr(part_mesh, "DELIVERED_TOLERANCE", 0.0)
    status = cli.main(["run", str(file)])
    out = capsys.readouterr().out
    assert status == cli.LIMITS_NOT_MET
    assert "warning (DeliveredPartDiffers)" in out and "NOT within" in out
    result = michell.read_result(tmp_path / "out" / "result")
    assert result["limits_met"] == [True, True] and not result["mesh"]["delivered"]["within"]
    assert any(w["category"] == "DeliveredPartDiffers" for w in result["warnings"])
    assert (tmp_path / "out" / "part.stl").is_file()


@pytest.mark.parametrize(
    ("value", "code", "path"),
    [
        ('["step"]', "not_available_yet", "output.mesh[0]"),
        ('"stl"', "wrong_type", "output.mesh"),
        ('["obj"]', "out_of_range", "output.mesh[0]"),
        ('["3mf", "3mf"]', "duplicate_name", "output.mesh[1]"),
    ],
)
def test_the_mesh_formats_are_checked(tmp_path, value, code, path):
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", f"max_fraction = 0.5\n\n[output]\nmesh = {value}\n"))
    with pytest.raises(michell.errors.ProblemFileError) as caught:
        problem_file.read(file)
    assert caught.value.code == code and caught.value.path == path
