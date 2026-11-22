"""The problem file: read and checked (``michell.problem_file.read``: every key, the defaults, every refusal
with its code and the key's path), prepared and run (``michell.prepare`` and ``michell.run`` on a bar and on
the example bridge), and the ``michell check`` and ``michell run`` commands (exit status, ``--json`` output,
the files written)."""

import hashlib
import json
import math
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import michell
from michell import cli, errors, mesh, problem_file
from michell.attach import LoadOnSupport
from michell.problem_file import Target, read
from michell.verification.attach_rows import example_bridge

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "bridge_file"

# A bar 12 x 6 x 6 mm held through a wall beyond its face x = 0 and loaded through a box beyond x = 12.
BAR = """\
schema = "michell/0"

[regions]
design = "bar.stl"

[bodies]
wall = "wall.stl"

[grid]
voxel_mm = 1

[material]
youngs_modulus_mpa = 1000
poisson_ratio = 0.3
density_g_cm3 = 1.0

[[supports]]
on = "wall"

[[load_cases]]
name = "tip"
loads = [ { on = { box_mm = [[12, -1, -1], [13, 7, 7]] }, force_n = [0, 0, -1] } ]

[goal]
module = "compliance"
sense = "min"

[[limits]]
module = "mass"
max_fraction = 0.5
"""

# Every key of the format, each set away from its default.
EVERY_KEY = """\
schema = "michell/0"

[regions]
design = "bar.stl"
solid = { root = "root.stl" }
void = { hole = "hole.stl" }

[bodies]
wall = "wall.stl"

[grid]
voxel_mm = 1.0

[material]
name = "aluminium"
youngs_modulus_mpa = 70000.0
poisson_ratio = 0.33
density_g_cm3 = 2.7

[[supports]]
on = "wall"
fix = ["z", "x"]
skin = false

[[supports]]
on = "root"
fix = ["y"]

[[load_cases]]
name = "down"
weight = 2.5
loads = [ { on = { box_mm = [[12, -1, -1], [13, 7, 7]] }, force_n = [0, 0, -1], skin = false } ]

[[load_cases]]
name = "side"
loads = [ { on = "root", force_n = [0, 1, 0] } ]

[goal]
module = "compliance"
sense = "min"

[[limits]]
module = "mass"
max = 0.2

[[rules]]
module = "min_member"
size_mm = 4.5

[run]
threads = 3
filter_faces = { "x-" = "empty", "z+" = "closed" }

[output]
folder = "results/here"
fields = ["vtk"]
"""


def write_geometry(folder: Path) -> None:
    mesh.write_stl(folder / "bar.stl", mesh.box((0, 0, 0), (12, 6, 6)))
    mesh.write_stl(folder / "wall.stl", mesh.box((-1, -1, -1), (0, 7, 7)))
    mesh.write_stl(folder / "root.stl", mesh.box((0, 0, 0), (2, 6, 6)))
    mesh.write_stl(folder / "hole.stl", mesh.box((6, 2, -1), (8, 4, 7)))


def write(folder: Path, text: str, name: str = "part.toml") -> Path:
    write_geometry(folder)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def edited(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


TOP = 'schema = "michell/0"\n'


def applied(text: str, edit) -> str:
    """``edit`` applied: one (old, new) pair, or a list of them."""
    for old, new in edit if isinstance(edit, list) else [edit]:
        text = edited(text, old, new)
    return text


def test_the_example_file_reads_into_its_description():
    problem = read(EXAMPLE / "bridge.toml")
    assert problem.schema == "michell/0" and len(problem.sha256) == 64
    assert problem.design.location == EXAMPLE / "envelope.stl" and problem.design.key == "regions.design"
    assert list(problem.solid) == ["plate_low", "plate_high", "pad"]
    assert problem.solid["pad"].key == "regions.solid.pad" and problem.solid["pad"].role == "solid"
    assert list(problem.bodies) == ["clamp_low", "clamp_high", "load_pad"] and problem.void == {}
    assert [g.key for g in problem.geometry][:2] == ["regions.design", "regions.solid.plate_low"]
    assert problem.voxel_mm == 1.0
    assert problem.material.name == "steel"
    assert problem.material.material() == michell.Material(210000.0, 0.3, 7.85)
    assert [s.on for s in problem.supports] == [Target("body", "clamp_low"), Target("body", "clamp_high")]
    assert all(s.fix == ("x", "y", "z") and s.skin for s in problem.supports)
    (case,) = problem.load_cases
    assert case.name == "pad" and case.weight == 1.0 and case.key == "load_cases[0]"
    (load,) = case.loads
    assert load.on == Target("body", "load_pad") and load.force_n == (0.0, 0.0, -1000.0)
    assert load.key == "load_cases[0].loads[0]"
    assert problem.goal == ("compliance", "min")
    assert problem.mass.max_fraction == 0.30 and problem.mass.max_g is None
    assert problem.member_size_mm == 5.0 and problem.threads == 1
    assert problem.filter_faces == {} and problem.output == EXAMPLE / "out" and problem.fields == ("vtk",)


def test_the_defaults(tmp_path):
    problem = read(write(tmp_path, BAR))
    (support,) = problem.supports
    assert support.fix == ("x", "y", "z") and support.skin is True
    assert problem.load_cases[0].weight == 1.0 and problem.load_cases[0].loads[0].skin is True
    assert problem.load_cases[0].loads[0].on == Target("box", box=((12.0, -1.0, -1.0), (13.0, 7.0, 7.0)))
    assert problem.material.name is None
    assert problem.member_size_mm is None
    assert problem.threads == 1 and problem.filter_faces == {}
    assert problem.output == tmp_path / "out" and problem.fields == ()
    assert problem.solid == {} and problem.void == {}


def test_every_key(tmp_path):
    problem = read(write(tmp_path, EVERY_KEY))
    assert problem.material == michell.problem_file.MaterialEntry("aluminium", 70000.0, 0.33, 2.7)
    assert problem.solid["root"].location == tmp_path / "root.stl"
    assert problem.void["hole"].role == "void"
    first, second = problem.supports
    assert first.fix == ("x", "z") and first.skin is False
    assert second.on == Target("region", "root") and second.fix == ("y",)
    down, side = problem.load_cases
    assert down.weight == 2.5 and down.loads[0].skin is False
    assert side.loads[0].on == Target("region", "root") and side.loads[0].force_n == (0.0, 1.0, 0.0)
    assert problem.mass.max_g == 0.2 and problem.mass.max_fraction is None
    assert problem.member_size_mm == 4.5
    assert problem.threads == 3
    assert problem.filter_faces == {"x-": "empty", "z+": "closed"}
    assert problem.output == tmp_path / "results" / "here" and problem.fields == ("vtk",)


def test_integers_are_read_as_numbers_and_booleans_are_not(tmp_path):
    problem = read(write(tmp_path, BAR))
    assert isinstance(problem.voxel_mm, float) and problem.voxel_mm == 1.0
    assert isinstance(problem.material.youngs_modulus_mpa, float)
    with pytest.raises(errors.WrongType) as caught:
        read(write(tmp_path, edited(BAR, "voxel_mm = 1", "voxel_mm = true")))
    assert caught.value.path == "grid.voxel_mm"


# (the edit, the code, the path)
REFUSALS = [
    # the file and its schema
    (("[grid]", "[grid"), "invalid_toml", ""),
    (("voxel_mm = 1\n", "voxel_mm = 1\nvoxel_mm = 2\n"), "invalid_toml", ""),
    (('schema = "michell/0"\n', ""), "missing_key", "schema"),
    (('schema = "michell/0"', 'schema = "michell/1"'), "unknown_schema", "schema"),
    (('schema = "michell/0"', "schema = 0"), "unknown_schema", "schema"),
    # unknown keys, anywhere
    (("[grid]", '[geometry]\nfile = "part.step"\n\n[grid]'), "not_available_yet", "geometry"),
    (("[grid]", "[mesh]\nx = 1\n\n[grid]"), "unknown_key", "mesh"),
    (('design = "bar.stl"', 'design = "bar.stl"\nsolids = {}'), "unknown_key", "regions.solids"),
    (("voxel_mm = 1", "voxel_mm = 1\nsize = 2"), "unknown_key", "grid.size"),
    (("poisson_ratio = 0.3", "poissons_ratio = 0.3"), "unknown_key", "material.poissons_ratio"),
    (('on = "wall"', 'on = "wall"\nfixed = ["x"]'), "unknown_key", "supports[0].fixed"),
    (('name = "tip"', 'name = "tip"\nweigth = 1'), "unknown_key", "load_cases[0].weigth"),
    (("force_n = [0, 0, -1] }", "force_n = [0, 0, -1], at = 1 }"), "unknown_key", "load_cases[0].loads[0].at"),
    (
        ("force_n = [0, 0, -1] }", "force_n = [0, 0, -1], moment_n_mm = [0, 0, 1] }"),
        "not_available_yet",
        "load_cases[0].loads[0].moment_n_mm",
    ),
    (
        ("force_n = [0, 0, -1] }", "force_n = [0, 0, -1], pressure_mpa = 1 }"),
        "not_available_yet",
        "load_cases[0].loads[0].pressure_mpa",
    ),
    (
        ("box_mm = [[12, -1, -1], [13, 7, 7]]", "box_mm = [[12, -1, -1], [13, 7, 7]], z = 1"),
        "unknown_key",
        "load_cases[0].loads[0].on.z",
    ),
    (('sense = "min"', 'sense = "min"\nweight = 1'), "unknown_key", "goal.weight"),
    (("max_fraction = 0.5", "max_fraction = 0.5\nmin = 0"), "unknown_key", "limits[0].min"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[rules]]\nmodule = "min_member"\nsize_mm = 5\nsize = 4\n'),
        "unknown_key",
        "rules[0].size",
    ),
    (("max_fraction = 0.5\n", "max_fraction = 0.5\n\n[run]\nmax_passes = 200\n"), "unknown_key", "run.max_passes"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[run]\nfilter_faces = { "x" = "closed" }\n'),
        "unknown_key",
        "run.filter_faces.x",
    ),
    (("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nformat = "json"\n'), "unknown_key", "output.format"),
    (("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nmesh = ["stl"]\n'), "not_available_yet", "output.mesh"),
    # missing keys
    (('[regions]\ndesign = "bar.stl"\n', ""), "missing_key", "regions"),
    (('design = "bar.stl"', ""), "missing_key", "regions.design"),
    (("voxel_mm = 1", ""), "missing_key", "grid.voxel_mm"),
    (("density_g_cm3 = 1.0", ""), "missing_key", "material.density_g_cm3"),
    (('[[supports]]\non = "wall"\n', ""), "missing_key", "supports"),
    (('on = "wall"', 'fix = ["x"]'), "missing_key", "supports[0].on"),
    (('name = "tip"\n', ""), "missing_key", "load_cases[0].name"),
    (("force_n = [0, 0, -1] }", "skin = true }"), "missing_key", "load_cases[0].loads[0].force_n"),
    (('sense = "min"', ""), "missing_key", "goal.sense"),
    (('[[limits]]\nmodule = "mass"\nmax_fraction = 0.5\n', ""), "missing_key", "limits"),
    (("max_fraction = 0.5", ""), "missing_key", "limits[0].max"),
    (("max_fraction = 0.5\n", "max_fraction = 0.5\n\n[[rules]]\nsize_mm = 5\n"), "missing_key", "rules[0].module"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[rules]]\nmodule = "min_member"\n'),
        "missing_key",
        "rules[0].size_mm",
    ),
    (("box_mm = [[12, -1, -1], [13, 7, 7]]", "box = 1"), "unknown_key", "load_cases[0].loads[0].on.box"),
    # wrong types and shapes
    (("voxel_mm = 1", 'voxel_mm = "1"'), "wrong_type", "grid.voxel_mm"),
    ([("[grid]\nvoxel_mm = 1\n", ""), (TOP, TOP + "grid = 1\n")], "wrong_type", "grid"),
    (('design = "bar.stl"', "design = 1"), "wrong_type", "regions.design"),
    (('wall = "wall.stl"', "wall = 1"), "wrong_type", "bodies.wall"),
    ([('[bodies]\nwall = "wall.stl"\n', ""), (TOP, TOP + 'bodies = "wall.stl"\n')], "wrong_type", "bodies"),
    (('on = "wall"', 'on = "wall"\nfix = "xyz"'), "wrong_type", "supports[0].fix"),
    (('on = "wall"', 'on = "wall"\nskin = "yes"'), "wrong_type", "supports[0].skin"),
    (('on = "wall"', "on = 3"), "wrong_type", "supports[0].on"),
    (("[[12, -1, -1], [13, 7, 7]]", "[[12, -1], [13, 7]]"), "wrong_type", "load_cases[0].loads[0].on.box_mm"),
    (
        ("[[12, -1, -1], [13, 7, 7]]", '[[12, -1, -1], [13, 7, "7"]]'),
        "wrong_type",
        "load_cases[0].loads[0].on.box_mm[1][2]",
    ),
    (("force_n = [0, 0, -1]", "force_n = [0, -1]"), "wrong_type", "load_cases[0].loads[0].force_n"),
    (("force_n = [0, 0, -1]", 'force_n = [0, 0, "-1"]'), "wrong_type", "load_cases[0].loads[0].force_n[2]"),
    (('name = "tip"', "name = 1"), "wrong_type", "load_cases[0].name"),
    ([('[[supports]]\non = "wall"\n', ""), (TOP, TOP + 'supports = { on = "wall" }\n')], "wrong_type", "supports"),
    (("max_fraction = 0.5", "max_fraction = 0.5\nmax = 1"), "wrong_type", "limits[0]"),
    (("max_fraction = 0.5\n", "max_fraction = 0.5\n\n[run]\nthreads = 2.0\n"), "wrong_type", "run.threads"),
    (("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nfields = "vtk"\n'), "wrong_type", "output.fields"),
    # values out of range
    (("voxel_mm = 1", "voxel_mm = 0"), "out_of_range", "grid.voxel_mm"),
    (("voxel_mm = 1", "voxel_mm = -0.5"), "out_of_range", "grid.voxel_mm"),
    (("voxel_mm = 1", "voxel_mm = inf"), "out_of_range", "grid.voxel_mm"),
    (("voxel_mm = 1", "voxel_mm = nan"), "out_of_range", "grid.voxel_mm"),
    (("youngs_modulus_mpa = 1000", "youngs_modulus_mpa = 0"), "out_of_range", "material.youngs_modulus_mpa"),
    (("poisson_ratio = 0.3", "poisson_ratio = 0.5"), "out_of_range", "material.poisson_ratio"),
    (("poisson_ratio = 0.3", "poisson_ratio = -1"), "out_of_range", "material.poisson_ratio"),
    (("density_g_cm3 = 1.0", "density_g_cm3 = -1.0"), "out_of_range", "material.density_g_cm3"),
    (('on = "wall"', 'on = "wall"\nfix = ["w"]'), "out_of_range", "supports[0].fix[0]"),
    (('on = "wall"', 'on = "wall"\nfix = []'), "out_of_range", "supports[0].fix"),
    (('name = "tip"', 'name = "tip"\nweight = -1'), "out_of_range", "load_cases[0].weight"),
    (('name = "tip"', 'name = ""'), "out_of_range", "load_cases[0].name"),
    (
        ("loads = [ { on = { box_mm = [[12, -1, -1], [13, 7, 7]] }, force_n = [0, 0, -1] } ]", "loads = []"),
        "out_of_range",
        "load_cases[0].loads",
    ),
    (("[[12, -1, -1], [13, 7, 7]]", "[[12, -1, -1], [12, 7, 7]]"), "out_of_range", "load_cases[0].loads[0].on.box_mm"),
    (("force_n = [0, 0, -1]", "force_n = [0, 0, inf]"), "out_of_range", "load_cases[0].loads[0].force_n[2]"),
    (('sense = "min"', 'sense = "max"'), "out_of_range", "goal.sense"),
    (("max_fraction = 0.5", "max_fraction = 0"), "out_of_range", "limits[0].max_fraction"),
    (("max_fraction = 0.5", "max_fraction = 1.5"), "out_of_range", "limits[0].max_fraction"),
    (("max_fraction = 0.5", "max = 0"), "out_of_range", "limits[0].max"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[rules]]\nmodule = "min_member"\nsize_mm = 0\n'),
        "out_of_range",
        "rules[0].size_mm",
    ),
    (("max_fraction = 0.5\n", "max_fraction = 0.5\n\n[run]\nthreads = 0\n"), "out_of_range", "run.threads"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[run]\nfilter_faces = { z- = "open" }\n'),
        "out_of_range",
        "run.filter_faces.z-",
    ),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nfields = ["png"]\n'),
        "out_of_range",
        "output.fields[0]",
    ),
    (("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nfolder = ""\n'), "out_of_range", "output.folder"),
    # names
    (('on = "wall"', 'on = "wal"'), "unknown_name", "supports[0].on"),
    (('on = "wall"', 'on = "design"'), "unknown_name", "supports[0].on"),
    (('module = "compliance"', 'module = "stiffness"'), "unknown_name", "goal.module"),
    (('module = "mass"', 'module = "volume"'), "unknown_name", "limits[0].module"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[rules]]\nmodule = "overhang"\n'),
        "unknown_name",
        "rules[0].module",
    ),
    (('wall = "wall.stl"', 'wall = "wall.stl"\ndesign = "bar.stl"'), "duplicate_name", "bodies.design"),
    (('design = "bar.stl"', 'design = "bar.stl"\nsolid = { wall = "wall.stl" }'), "duplicate_name", "bodies.wall"),
    (('on = "wall"', 'on = "wall"\nfix = ["x", "x"]'), "duplicate_name", "supports[0].fix[1]"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[limits]]\nmodule = "mass"\nmax = 1\n'),
        "duplicate_name",
        "limits[1].module",
    ),
    (
        (
            "max_fraction = 0.5\n",
            'max_fraction = 0.5\n\n[[rules]]\nmodule = "min_member"\nsize_mm = 5\n\n[[rules]]\n'
            'module = "min_member"\nsize_mm = 6\n',
        ),
        "duplicate_name",
        "rules[1].module",
    ),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nfields = ["vtk", "vtk"]\n'),
        "duplicate_name",
        "output.fields[1]",
    ),
    # files
    (('design = "bar.stl"', 'design = "beam.stl"'), "file_not_found", "regions.design"),
    (('wall = "wall.stl"', 'wall = "walls/wall.stl"'), "file_not_found", "bodies.wall"),
    (('design = "bar.stl"', 'design = "bar.step"'), "not_available_yet", "regions.design"),
    (
        ("max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[[rules]]\nmodule = "symmetry"\nplanes = ["y"]\n'),
        "not_available_yet",
        "rules[0].module",
    ),
]


@pytest.mark.parametrize("edit, code, path", REFUSALS, ids=[f"{code}:{path}" for _, code, path in REFUSALS])
def test_a_bad_edit_is_refused_with_its_code_and_path(tmp_path, edit, code, path):
    file = write(tmp_path, applied(BAR, edit))
    with pytest.raises(errors.ProblemFileError) as caught:
        read(file)
    assert (caught.value.code, caught.value.path) == (code, path), caught.value.message
    assert caught.value.file == str(file)
    assert caught.value.message


def test_every_kind_of_fault_has_a_refusal_above():
    assert {code for _, code, _ in REFUSALS} == {
        "invalid_toml",
        "unknown_schema",
        "unknown_key",
        "missing_key",
        "wrong_type",
        "out_of_range",
        "unknown_name",
        "duplicate_name",
        "file_not_found",
        "not_available_yet",
    }


def test_names_of_the_design_and_void_regions_are_not_targets(tmp_path):
    text = edited(BAR, 'design = "bar.stl"', 'design = "bar.stl"\nvoid = { hole = "hole.stl" }')
    for name in ("design", "hole"):
        with pytest.raises(errors.UndefinedName) as caught:
            read(write(tmp_path, edited(text, 'on = "wall"', f'on = "{name}"')))
        assert "region; a support or a load acts on a solid region" in caught.value.message


def test_the_same_load_case_twice_is_refused(tmp_path):
    text = BAR + '\n[[load_cases]]\nname = "tip"\nloads = [ { on = "wall", force_n = [1, 0, 0] } ]\n'
    with pytest.raises(errors.DuplicateName) as caught:
        read(write(tmp_path, text))
    assert caught.value.path == "load_cases[1].name" and "load_cases[0]" in caught.value.message


def test_a_missing_problem_file_and_one_that_is_not_text(tmp_path):
    with pytest.raises(errors.MissingFile) as caught:
        read(tmp_path / "nothing.toml")
    assert caught.value.code == "file_not_found" and caught.value.path == ""
    (tmp_path / "binary.toml").write_bytes(b"\xff\xfe\x00schema")
    with pytest.raises(errors.InvalidToml):
        read(tmp_path / "binary.toml")


def test_an_invalid_toml_file_names_the_line(tmp_path):
    with pytest.raises(errors.InvalidToml) as caught:
        read(write(tmp_path, edited(BAR, "[grid]", "[grid")))
    assert "line" in caught.value.message


def test_a_key_that_is_not_a_bare_key_is_quoted_in_the_path(tmp_path):
    with pytest.raises(errors.UnknownKey) as caught:
        read(write(tmp_path, edited(BAR, "voxel_mm = 1", 'voxel_mm = 1\n"voxel size" = 1')))
    assert caught.value.path == 'grid."voxel size"'


def test_the_errors_are_structured_and_also_built_in_exceptions(tmp_path):
    file = write(tmp_path, edited(BAR, "voxel_mm = 1", 'voxel_mm = "1"'))
    with pytest.raises(TypeError) as caught:
        read(file)
    error = caught.value
    assert isinstance(error, michell.EngineError) and isinstance(error, errors.InvalidType)
    assert str(error) == f"wrong_type at grid.voxel_mm: {error.message}"
    assert error.as_dict() == {
        "code": "wrong_type",
        "path": "grid.voxel_mm",
        "file": str(file),
        "message": error.message,
        "details": {},
    }
    with pytest.raises(ValueError):
        read(write(tmp_path, edited(BAR, "voxel_mm = 1", "voxel_mm = 0")))
    with pytest.raises(KeyError):
        read(write(tmp_path, edited(BAR, 'on = "wall"', 'on = "wal"')))


def test_geometry_paths_are_relative_to_the_problem_file(tmp_path, monkeypatch):
    folder = tmp_path / "case"
    folder.mkdir()
    write(folder, BAR)
    monkeypatch.chdir(tmp_path)
    problem = read(Path("case") / "part.toml")
    assert problem.design.location == folder / "bar.stl" and problem.output == folder / "out"
    assert problem.path == folder / "part.toml"


def test_the_example_s_files_are_those_its_script_writes(tmp_path):
    shutil.copy(EXAMPLE / "make_geometry.py", tmp_path)
    runpy.run_path(str(tmp_path / "make_geometry.py"), run_name="__main__")
    for stl in EXAMPLE.glob("*.stl"):
        assert (tmp_path / stl.name).read_bytes() == stl.read_bytes(), stl.name
    assert sorted(p.name for p in tmp_path.glob("*.stl")) == sorted(p.name for p in EXAMPLE.glob("*.stl"))
    assert math.isclose(mesh.read_stl(EXAMPLE / "envelope.stl").volume_mm3, 50 * 24 * 10)


# ---------------------------------------------------------------------------------------------------------
# Prepare and run


@pytest.fixture
def bar(tmp_path):
    return write(tmp_path, BAR)


def test_the_example_prepared_from_its_file_is_the_hand_built_bridge():
    prepared = michell.prepare(EXAMPLE / "bridge.toml")
    example = example_bridge()
    setup = prepared.set_up
    assert np.array_equal(np.asarray(setup.problem.grid.active), np.asarray(example.grid.active))
    assert np.array_equal(setup.keep_solid.mask, example.keep_solid.mask)
    assert np.array_equal(setup.held, example.held)
    assert np.array_equal(setup.problem.cases[0].loads, example.loads)
    assert setup.outside == example.outside
    # The mass bound to the last bit: max_fraction = 0.30 as the example writes 0.30 * the block's mass.
    steel = michell.Material(youngs_modulus=210000.0, poisson_ratio=0.3, density_g_cm3=7.85)
    grid = example.grid
    assert prepared.mass_g == 0.30 * steel.mass_g(grid.voxel_volume_mm3 * grid.n_elements) == example.mass_g
    assert prepared.size_mm == 5.0 and prepared.size_given
    assert prepared.material == steel and prepared.threads == 1 and prepared.warnings == ()


def test_the_summary_is_plain_data(bar):
    prepared = michell.prepare(bar)
    summary = json.loads(json.dumps(prepared.summary(), allow_nan=False))
    assert summary["problem_file"] == {"name": "part.toml", "sha256": hashlib.sha256(bar.read_bytes()).hexdigest()}
    assert [g["key"] for g in summary["geometry"]] == ["regions.design", "bodies.wall"]
    assert summary["geometry"][0]["sha256"] == hashlib.sha256((bar.parent / "bar.stl").read_bytes()).hexdigest()
    assert summary["grid"]["counts"] == [12, 6, 6] and summary["grid"]["origin_mm"] == [0.0, 0.0, 0.0]
    assert summary["grid"]["active"] == 432
    set_up = summary["set_up"]
    assert set_up["member_size"] == {"size_mm": 5.0, "given": False, "voxels": 5.0}
    assert set_up["mass_limit"]["max_fraction"] == 0.5
    assert set_up["mass_limit"]["max_g"] == pytest.approx(0.5 * 432 * 1e-3)
    (support,) = set_up["supports"]
    assert support["key"] == "supports[0]" and support["on"] == "marker body 'wall'" and support["faces"] == 36
    assert support["closed_faces"] == ["x-"] and set_up["filter_faces"]["x-"] == "closed"
    (load,) = set_up["loads"]
    assert load["key"] == "load_cases[0].loads[0]" and load["force_n"] == [0.0, 0.0, -1.0]
    assert load["skin_voxels"] > 0 and load["on"].startswith("box from (12, -1, -1)")


def test_a_mass_bound_below_the_held_solid_is_refused_with_its_key(tmp_path):
    text = edited(BAR, 'design = "bar.stl"', 'design = "bar.stl"\nsolid = { root = "root.stl" }')
    with pytest.raises(errors.OutOfRange) as caught:
        michell.prepare(write(tmp_path, edited(text, "max_fraction = 0.5", "max_fraction = 0.1")))
    assert caught.value.path == "limits[0].max_fraction" and "held solid" in caught.value.message


def test_refusals_of_the_geometry_and_the_set_up_carry_the_entry_s_path(tmp_path):
    # A solid region too thin for the voxels.
    file = write(tmp_path, edited(BAR, 'design = "bar.stl"', 'design = "bar.stl"\nsolid = { thin = "thin.stl" }'))
    mesh.write_stl(tmp_path / "thin.stl", mesh.box((4, 0, 0), (4.2, 6, 6)))
    with pytest.raises(errors.InvalidValue) as caught:
        michell.prepare(file)
    assert caught.value.code == "invalid_input" and caught.value.path == "regions.solid.thin"
    # A marker body that touches no face of the part.
    file = write(tmp_path, BAR)
    mesh.write_stl(tmp_path / "wall.stl", mesh.box((-5, -1, -1), (-4, 7, 7)))
    with pytest.raises(errors.InvalidValue) as caught:
        michell.prepare(file)
    assert caught.value.path == "supports[0]" and "touches no boundary face" in caught.value.message
    # A load box away from the part.
    file = write(tmp_path, edited(BAR, "[[12, -1, -1], [13, 7, 7]]", "[[20, -1, -1], [21, 7, 7]]"))
    with pytest.raises(errors.InvalidValue) as caught:
        michell.prepare(file)
    assert caught.value.path == "load_cases[0].loads[0]"
    # A file that is not an STL file, and a mesh that is not closed.
    file = write(tmp_path, BAR)
    (tmp_path / "wall.stl").write_text("not a mesh")
    with pytest.raises(errors.InvalidValue) as caught:
        michell.prepare(file)
    assert caught.value.path == "bodies.wall"
    box = mesh.box((-1, -1, -1), (0, 7, 7))
    mesh.write_stl(tmp_path / "wall.stl", mesh.Mesh(box.vertices, box.triangles[:-1]))
    with pytest.raises(errors.InvalidValue) as caught:
        michell.prepare(file)
    assert caught.value.path == "bodies.wall" and "not closed" in caught.value.message


def test_a_part_that_is_not_held_is_refused_by_prepare(tmp_path):
    file = write(tmp_path, edited(BAR, 'on = "wall"', 'on = "wall"\nfix = ["z"]'))
    with pytest.raises(michell.EngineError) as caught:
        michell.prepare(file)
    assert caught.value.code == "singular_system"


def test_warnings_are_kept_and_given_again(tmp_path):
    # A region load on the two layers at the held face: a quarter of it falls on the held nodes.
    text = edited(BAR, 'design = "bar.stl"', 'design = "bar.stl"\nsolid = { root = "root.stl" }')
    text = edited(text, "on = { box_mm = [[12, -1, -1], [13, 7, 7]] }", 'on = "root"')
    with pytest.warns(LoadOnSupport):
        prepared = michell.prepare(write(tmp_path, text))
    (record,) = prepared.warnings
    assert record.category is LoadOnSupport and "25 %" in record.message
    assert prepared.summary()["warnings"] == [{"category": "LoadOnSupport", "message": record.message}]


def test_threads_from_the_caller_and_the_file(tmp_path, bar):
    assert michell.prepare(bar).threads == 1
    assert michell.prepare(bar, threads=2).set_up.problem.threads == 2
    with pytest.raises(errors.OutOfRange):
        michell.prepare(bar, threads=0)


def test_run_on_the_bar_writes_the_outcome(tmp_path):
    file = write(tmp_path, edited(BAR, "max_fraction = 0.5\n", 'max_fraction = 0.5\n\n[output]\nfields = ["vtk"]\n'))
    seen = []
    outcome = michell.run(file, callback=lambda step, number, record: seen.append((step, number)))
    assert isinstance(outcome, michell.RunOutcome) and outcome.limits_met and outcome.warnings == ()
    assert len(seen) == len(outcome.part.passes) == 140
    assert outcome.paths == {kind: tmp_path / "out" / f"result.{kind}" for kind in ("json", "npz", "vtk")}
    assert all(path.is_file() for path in outcome.paths.values())
    result = michell.read_result(tmp_path / "out" / "result")
    assert result["format"] == 2 and result["kind"] == "StiffPart"
    assert result["goal"] == outcome.part.result.goal and result["limits_met"] == [True, True]
    assert result["problem_file"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert [g["file"] for g in result["geometry"]] == ["bar.stl", "wall.stl"]
    assert result["grid"]["counts"] == [12, 6, 6] and result["grid"]["voxel_mm"] == 1.0
    assert result["volumes"][0]["name"] == "part"
    assert result["set_up"]["supports"][0]["key"] == "supports[0]"
    assert result["warnings"] == [] and result["files"] == ["result.json", "result.npz", "result.vtk"]
    assert set(result["seconds"]) == {"prepare", "optimise"}
    assert np.array_equal(result["arrays"]["nominal"], outcome.part.nominal)
    # --out elsewhere, from Python
    other = michell.run(file, out=tmp_path / "elsewhere")
    assert other.paths["json"] == tmp_path / "elsewhere" / "result.json"
    assert other.part.result.goal == outcome.part.result.goal


def test_write_result_keeps_format_1_without_a_record_and_refuses_a_clash(tmp_path):
    outcome = michell.problem_file._optimise(
        michell.problem_file._prepare(write(tmp_path, BAR)),
        tmp_path / "short",
        schedule=((1.0, 2),),
        second_step_passes=1,
    )
    michell.write_result(tmp_path / "plain", outcome.part)
    assert michell.read_result(tmp_path / "plain")["format"] == 1
    with pytest.raises(errors.InvalidValue):
        michell.write_result(tmp_path / "clash", outcome.part, record={"goal": 1.0})


# ---------------------------------------------------------------------------------------------------------
# The command line


def test_check_prints_the_set_up(bar, capsys):
    assert cli.main(["check", str(bar)]) == 0
    out = capsys.readouterr().out
    assert "grid: 12 x 6 x 6 voxels of 1 mm" in out
    assert "supports[0] on marker body 'wall'" in out
    assert "member size: 5 mm, 5 voxels (not given: five voxel sides)" in out


def test_check_json_is_one_json_object(bar, capsys):
    assert cli.main(["check", str(bar), "--json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["command"] == "check" and data["exit"] == 0
    assert data["prepared"]["grid"]["active"] == 432


@pytest.mark.parametrize("json_flag", [False, True])
def test_check_refuses_with_status_2_and_the_error(tmp_path, capsys, json_flag):
    file = write(tmp_path, edited(BAR, "voxel_mm = 1", "voxel_mm = -1"))
    assert cli.main(["check", str(file)] + (["--json"] if json_flag else [])) == 2
    captured = capsys.readouterr()
    if json_flag:
        error = json.loads(captured.out)["error"]
        assert (error["code"], error["path"], error["file"]) == ("out_of_range", "grid.voxel_mm", str(file))
    else:
        assert captured.out == "" and "out_of_range at grid.voxel_mm" in captured.err


def test_a_part_that_is_not_held_exits_with_2(tmp_path, capsys):
    file = write(tmp_path, edited(BAR, 'on = "wall"', 'on = "wall"\nfix = ["z"]'))
    assert cli.main(["check", str(file), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "singular_system"
    assert cli.main(["run", str(file), "--json"]) == 2


def _short(monkeypatch, **settings):
    real = problem_file._optimise
    monkeypatch.setattr(cli, "_optimise", lambda prepared, out, callback: real(prepared, out, callback, **settings))


def test_run_json_gives_one_json_object_and_the_passes_on_standard_error(tmp_path, capsys, monkeypatch):
    _short(monkeypatch, schedule=((1.0, 3), (2.0, 2)), second_step_passes=4)
    file = write(tmp_path, BAR)
    status = cli.main(["run", str(file), "--json", "--out", str(tmp_path / "o"), "--threads", "2"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert status == data["exit"] == (0 if data["result"]["feasible"] else 1)
    assert data["result"]["passes"] == 9 and data["result"]["threads"] == 2
    assert data["result"]["files"]["json"] == str(tmp_path / "o" / "result.json")
    assert captured.err.count(" pass ") == 9


def test_run_exits_with_1_when_a_limit_is_not_met(tmp_path, capsys, monkeypatch):
    # One pass of the member-size step does not bring a uniform start to the member-size bound.
    _short(monkeypatch, schedule=((1.0, 1),), second_step_passes=1)
    file = write(tmp_path, BAR)
    assert cli.main(["run", str(file)]) == 1
    out = capsys.readouterr().out
    assert "NOT met" in out and "warning (LimitNotMet)" in out
    result = michell.read_result(tmp_path / "out" / "result")
    assert False in result["limits_met"] and result["warnings"][0]["category"] == "LimitNotMet"


def test_run_exits_with_3_on_an_engine_error(tmp_path, capsys, monkeypatch):
    def failing(prepared, out, callback):
        error = michell.EngineError("solver_not_converged: the solver did not converge")
        error.code, error.message, error.details = "solver_not_converged", "the solver did not converge", {"steps": 5}
        raise error

    monkeypatch.setattr(cli, "_optimise", failing)
    file = write(tmp_path, BAR)
    assert cli.main(["run", str(file), "--json"]) == 3
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "solver_not_converged" and error["details"] == {"steps": 5}


def test_an_output_folder_that_cannot_be_made_exits_with_2(tmp_path, capsys):
    file = write(tmp_path, BAR)
    (tmp_path / "taken").write_text("a file, not a folder")
    assert cli.main(["run", str(file), "--out", str(tmp_path / "taken"), "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert (error["code"], error["path"]) == ("out_of_range", "output.folder")


def test_anything_else_that_goes_wrong_exits_with_3_and_is_reported(tmp_path, capsys, monkeypatch):
    def broken(prepared, out, callback):
        raise RuntimeError("something broke")

    monkeypatch.setattr(cli, "_optimise", broken)
    file = write(tmp_path, BAR)
    assert cli.main(["run", str(file), "--json"]) == 3
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "internal_error" and "something broke" in error["message"]
    assert cli.main(["run", str(file)]) == 3
    assert "internal_error" in capsys.readouterr().err


def test_run_refused_before_the_first_pass_exits_with_2(tmp_path, capsys):
    file = write(tmp_path, edited(BAR, 'on = "wall"', 'on = "wal"'))
    assert cli.main(["run", str(file)]) == 2
    assert "unknown_name at supports[0].on" in capsys.readouterr().err


def test_the_command_runs_as_a_program(bar):
    finished = subprocess.run(
        [sys.executable, "-m", "michell.cli", "check", str(bar), "--json"], capture_output=True, text=True
    )
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout)["prepared"]["problem_file"]["name"] == "part.toml"


def test_the_version_and_the_help_stay(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.startswith("michell ")
    assert cli.main([]) == 0
    assert "check" in capsys.readouterr().out


@pytest.mark.slow
def test_the_example_runs_from_the_command_line(tmp_path):
    finished = subprocess.run(
        [sys.executable, "-m", "michell.cli", "run", str(EXAMPLE / "bridge.toml"), "--out", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    data = json.loads(finished.stdout)
    assert data["result"]["limits_met"] == [True, True]
    assert {path.name for path in tmp_path.iterdir()} == {"result.json", "result.npz", "result.vtk"}
    assert data["result"]["warnings"] == []
