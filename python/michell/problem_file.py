"""The problem file: a part, its supports and loads, the material, the goal, the limits and the rules in one
TOML file, with the geometry beside it as STL files. The format is described in
``docs/design/problem-file.md``; this module makes it real.

- ``read(path)``: the file read and checked into a ``ProblemFile``, a plain description with every value
  typed and every geometry file found. Nothing is computed.
- ``prepare(path, *, threads=None)``: the geometry read, the voxel model built (``michell.regions``) and the
  supports and loads placed on it (``michell.attach.set_up``), with the mass bound and the member size, as a
  ``Prepared``: everything ``michell check`` shows. It solves nothing.
- ``run(path, *, out=None, threads=None, callback=None)``: ``prepare``, then ``SetUp.stiff_part`` (every run
  is the two-step procedure of ``michell.recipes.stiff_part``), then the outcome written to the output
  folder (``result.json`` and ``result.npz``, and ``result.vtk`` when asked), as a ``RunOutcome``.

Every refusal of the file is a ``michell.errors.ProblemFileError``: one stable code per kind of fault, the
key's place in the file (``path``) and a sentence for people. A refusal of the geometry or of the set-up by
``michell.regions`` or ``michell.attach`` (a region too thin for the voxels, a marker body that touches no
face of the part) keeps their code, ``invalid_input``, and gains the ``path`` of the entry that caused it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import tomllib
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from michell.attach import FACES, Attachment, Case, Load, SetUp, Support, set_up
from michell.errors import (
    DuplicateName,
    InvalidToml,
    InvalidValue,
    MissingFile,
    MissingKey,
    NotAvailableYet,
    OutOfRange,
    UndefinedName,
    UnknownKey,
    UnknownSchema,
    WrongType,
    _Refused,
)
from michell.export import _plain, write_result, write_vtk
from michell.mesh import Mesh, read_stl
from michell.problem import Material
from michell.recipes import RecipePass, StiffPart
from michell.regions import VoxelModel, voxel_model

SCHEMA = "michell/0"

# Without a `min_member` rule, the member size is this many voxel sides: the smallest the grid resolves.
DEFAULT_SIZE_VOXELS = 5

# The output folder, relative to the problem file, when [output] names none.
DEFAULT_OUTPUT = "out"

# The fields [output] fields may ask for, and the name of the files run writes.
FIELDS = ("vtk",)
RESULT_NAME = "result"

GOALS = {"compliance": ("min",)}

# What the format reserves for later, with the sentence a file that uses it is refused with.
_PLANNED_TOP = {
    "geometry": "a STEP file with named bodies ([geometry] file) is planned and not available yet; for now each "
    "region and each marker body is its own STL file, named in [regions] and [bodies]",
}
_PLANNED_LOAD = {
    name: f"the load kind {name} is reserved and not available yet; for now a load is a force, force_n (N)"
    for name in ("moment_n_mm", "pressure_mpa", "remote_force_n")
}
_PLANNED_OUTPUT = {
    "mesh": "mesh output ([output] mesh) is planned and not available yet; for now a run writes result.json, "
    'result.npz and, with fields = ["vtk"], result.vtk',
}
_PLANNED_RULES = {
    "symmetry": "the symmetry rule is planned and not available yet; for now the whole part is optimised without it",
}
_STEP_SUFFIXES = (".step", ".stp")

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _join(path: str, key: str) -> str:
    """The place of ``key`` inside the table at ``path``, in TOML's dotted form (quoted where needed)."""
    part = key if _BARE_KEY.fullmatch(key) else json.dumps(key)
    return f"{path}.{part}" if path else part


def _shown(value) -> str:
    if isinstance(value, dict):
        return "a table"
    if isinstance(value, list):
        return "a list"
    return repr(value)


# ---------------------------------------------------------------------------------------------------------
# The description of a problem file


@dataclass(frozen=True)
class GeometryFile:
    """A geometry file named in the problem file: ``key`` (its place in the file, such as
    ``regions.solid.pad``), ``role`` (``"design"``, ``"solid"``, ``"void"`` or ``"body"``, a marker body),
    ``name`` (the region's or the body's name; ``"design"`` for the design region), ``given`` (the path as
    written) and ``location`` (the path, relative ones taken from the problem file's folder)."""

    key: str
    role: str
    name: str
    given: str
    location: Path


@dataclass(frozen=True)
class Target:
    """What a support or a load acts on: ``kind`` ``"region"`` (a solid region, by ``name``), ``"body"`` (a
    marker body, by ``name``) or ``"box"`` (``box``, its low and high corners in mm)."""

    kind: str
    name: str | None = None
    box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    @property
    def words(self) -> str:
        if self.kind == "region":
            return f"solid region {self.name!r}"
        if self.kind == "body":
            return f"marker body {self.name!r}"
        low, high = self.box
        return f"box from ({', '.join(f'{c:g}' for c in low)}) to ({', '.join(f'{c:g}' for c in high)}) mm"


@dataclass(frozen=True)
class SupportEntry:
    """A ``[[supports]]`` entry: ``key`` (``supports[i]``), ``on``, ``fix`` and ``skin``."""

    key: str
    on: Target
    fix: tuple[str, ...]
    skin: bool


@dataclass(frozen=True)
class LoadEntry:
    """A load of a load case: ``key`` (``load_cases[i].loads[j]``), ``on``, ``force_n`` (N) and ``skin``."""

    key: str
    on: Target
    force_n: tuple[float, float, float]
    skin: bool


@dataclass(frozen=True)
class CaseEntry:
    """A ``[[load_cases]]`` entry: ``key``, ``name``, ``weight`` and its ``loads``."""

    key: str
    name: str
    weight: float
    loads: tuple[LoadEntry, ...]


@dataclass(frozen=True)
class MaterialEntry:
    """``[material]``: ``name`` (or None), the modulus (MPa), Poisson's ratio and the density (g/cm3)."""

    name: str | None
    youngs_modulus_mpa: float
    poisson_ratio: float
    density_g_cm3: float

    def material(self) -> Material:
        return Material(self.youngs_modulus_mpa, self.poisson_ratio, self.density_g_cm3)


@dataclass(frozen=True)
class MassLimit:
    """The ``[[limits]]`` entry of module ``mass``: ``key``, and exactly one of ``max_g`` (grams) and
    ``max_fraction`` (of the mass of the whole part, every active voxel solid)."""

    key: str
    max_g: float | None
    max_fraction: float | None


@dataclass(frozen=True)
class ProblemFile:
    """A problem file read and checked by ``read``. Every value is typed and in range, every name it uses
    is defined once and every geometry file exists; nothing has been computed.

    - ``path`` (absolute), ``sha256`` (of its bytes) and ``schema``.
    - ``design``, ``solid``, ``void`` (by name) and ``bodies`` (the marker bodies, by name): ``GeometryFile``.
    - ``voxel_mm``, ``material`` (``MaterialEntry``), ``supports`` and ``load_cases`` (in the order of the
      file), ``goal`` (module and sense), ``mass`` (``MassLimit``) and ``member_size_mm`` (None when the
      file has no ``min_member`` rule).
    - ``threads``; ``filter_faces``, the density filter's faces the file sets (``"x-"`` to ``"z+"``, each
      ``"closed"`` or ``"empty"``); ``output``, the output folder (absolute); ``fields``.
    """

    path: Path
    sha256: str
    schema: str
    design: GeometryFile
    solid: dict[str, GeometryFile]
    void: dict[str, GeometryFile]
    bodies: dict[str, GeometryFile]
    voxel_mm: float
    material: MaterialEntry
    supports: tuple[SupportEntry, ...]
    load_cases: tuple[CaseEntry, ...]
    goal: tuple[str, str]
    mass: MassLimit
    member_size_mm: float | None
    threads: int
    filter_faces: dict[str, str]
    output: Path
    fields: tuple[str, ...]

    @property
    def geometry(self) -> tuple[GeometryFile, ...]:
        """Every geometry file, in the order design, solid, void, bodies."""
        return (self.design, *self.solid.values(), *self.void.values(), *self.bodies.values())


class _Reader:
    """The checks of one problem file; every refusal names the file and the key."""

    def __init__(self, path: Path):
        self.path = path
        self.folder = path.parent

    def error(self, kind, message: str, key: str, details=None):
        return kind(message, key, self.path, details)

    def table(self, value, key: str, allowed, required=(), planned=None) -> dict:
        label = f"[{key}]" if key else "the file"
        if not isinstance(value, dict):
            raise self.error(WrongType, f"{key} is a table; got {_shown(value)}", key)
        planned = planned or {}
        for name in value:
            if name in planned:
                raise self.error(NotAvailableYet, planned[name], _join(key, name))
            if name not in allowed:
                raise self.error(
                    UnknownKey, f"{label} has no key {name!r}; its keys are {', '.join(allowed)}", _join(key, name)
                )
        for name in required:
            if name not in value:
                raise self.error(MissingKey, f"{label} needs the key {name!r}", _join(key, name))
        return value

    def tables(self, value, key: str, least: int = 1) -> list[dict]:
        """A list of tables (an array of tables, [[key]])."""
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise self.error(WrongType, f"{key} is a list of tables, written [[{key}]]; got {_shown(value)}", key)
        if len(value) < least:
            raise self.error(OutOfRange, f"{key} needs at least {least} entry", key)
        return value

    def number(self, value, key: str, what: str, *, above=None, at_least=None, below=None, at_most=None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.error(WrongType, f"{what} ({key}) is a number; got {_shown(value)}", key)
        number = float(value)
        bad = (
            not math.isfinite(number)
            or (above is not None and not number > above)
            or (at_least is not None and not number >= at_least)
            or (below is not None and not number < below)
            or (at_most is not None and not number <= at_most)
        )
        if bad:
            ranges = [
                f"above {above:g}" if above is not None else "",
                f"at least {at_least:g}" if at_least is not None else "",
                f"below {below:g}" if below is not None else "",
                f"at most {at_most:g}" if at_most is not None else "",
            ]
            wanted = " and ".join(part for part in ranges if part)
            raise self.error(
                OutOfRange, f"{what} ({key}) must be a finite number {wanted}".rstrip() + f"; got {value!r}", key
            )
        return number

    def whole(self, value, key: str, what: str, at_least: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(WrongType, f"{what} ({key}) is a whole number; got {_shown(value)}", key)
        if value < at_least:
            raise self.error(OutOfRange, f"{what} ({key}) must be at least {at_least}; got {value}", key)
        return value

    def string(self, value, key: str, what: str) -> str:
        if not isinstance(value, str):
            raise self.error(WrongType, f"{what} ({key}) is a string; got {_shown(value)}", key)
        if not value:
            raise self.error(OutOfRange, f"{what} ({key}) must not be empty", key)
        return value

    def flag(self, value, key: str, what: str) -> bool:
        if not isinstance(value, bool):
            raise self.error(WrongType, f"{what} ({key}) is true or false; got {_shown(value)}", key)
        return value

    def choice(self, value, key: str, what: str, choices) -> str:
        text = self.string(value, key, what)
        if text not in choices:
            raise self.error(OutOfRange, f"{what} ({key}) is one of {', '.join(map(repr, choices))}; got {text!r}", key)
        return text

    def geometry(self, value, key: str, role: str, name: str) -> GeometryFile:
        given = self.string(value, key, f"the file of the {'marker body' if role == 'body' else role + ' region'}")
        location = self.folder / given
        if location.suffix.lower() in _STEP_SUFFIXES:
            raise self.error(
                NotAvailableYet,
                f"{given!r} is a STEP file: reading STEP is planned and not available yet; for "
                "now each region and each marker body is an STL file",
                key,
            )
        if not location.is_file():
            raise self.error(MissingFile, f"the file {given!r} does not exist (looked for {location})", key)
        return GeometryFile(key, role, name, given, location)

    def named_files(self, value, key: str, role: str, taken: dict[str, str]) -> dict[str, GeometryFile]:
        if not isinstance(value, dict):
            raise self.error(WrongType, f'{key} is a table of name = "file.stl"; got {_shown(value)}', key)
        named = {}
        for name, given in value.items():
            where = _join(key, name)
            if not name:
                raise self.error(OutOfRange, f"a name in {key} must not be empty", where)
            if name in taken:
                raise self.error(DuplicateName, f"the name {name!r} is already used by {taken[name]}", where)
            taken[name] = where
            named[name] = self.geometry(given, where, role, name)
        return named

    def target(self, value, key: str, solid, void, bodies) -> Target:
        if isinstance(value, str):
            if value in solid:
                return Target("region", name=value)
            if value in bodies:
                return Target("body", name=value)
            if value == "design" or value in void:
                role = "design" if value == "design" else "void"
                raise self.error(
                    UndefinedName,
                    f"{value!r} is the {role} region; a support or a load acts on a solid region, a "
                    f"marker body ([bodies]) or a box",
                    key,
                )
            known = [*solid, *bodies]
            raise self.error(
                UndefinedName,
                f"no solid region or marker body is called {value!r}; the file defines "
                f"{', '.join(map(repr, known)) if known else 'none'}",
                key,
            )
        if isinstance(value, dict):
            self.table(value, key, ("box_mm",), ("box_mm",))
            where = _join(key, "box_mm")
            corners = value["box_mm"]
            shaped = (
                isinstance(corners, list)
                and len(corners) == 2
                and all(isinstance(c, list) and len(c) == 3 for c in corners)
            )
            if not shaped:
                raise self.error(
                    WrongType,
                    f"a box ({where}) is its low and high corners, [[x0, y0, z0], [x1, y1, z1]] in mm; got "
                    f"{_shown(corners)}",
                    where,
                )
            low, high = (
                tuple(self.number(c, f"{where}[{i}][{j}]", "a corner coordinate") for j, c in enumerate(corner))
                for i, corner in enumerate(corners)
            )
            if not all(h > lo for lo, h in zip(low, high)):
                raise self.error(
                    OutOfRange,
                    f"each coordinate of a box's high corner must be above that of its low corner; got {corners!r}",
                    where,
                )
            return Target("box", box=(low, high))
        raise self.error(
            WrongType,
            f"on ({key}) names a solid region or a marker body, or is a box {{ box_mm = [[x0, y0, z0], "
            f"[x1, y1, z1]] }}; got {_shown(value)}",
            key,
        )

    def force(self, value, key: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise self.error(WrongType, f"force_n ({key}) is three numbers [fx, fy, fz] in N; got {_shown(value)}", key)
        return tuple(self.number(c, f"{key}[{i}]", "a force component") for i, c in enumerate(value))

    def directions(self, value, key: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(d, str) for d in value):
            raise self.error(
                WrongType, f'fix ({key}) is a list of directions such as ["x", "z"]; got {_shown(value)}', key
            )
        if not value:
            raise self.error(OutOfRange, f"fix ({key}) holds at least one direction", key)
        for i, direction in enumerate(value):
            if direction not in ("x", "y", "z"):
                raise self.error(OutOfRange, f"a direction is 'x', 'y' or 'z'; got {direction!r}", f"{key}[{i}]")
            if direction in value[:i]:
                raise self.error(DuplicateName, f"the direction {direction!r} is given twice", f"{key}[{i}]")
        return tuple(sorted(value, key="xyz".index))


def _file_bytes(path: Path, key: str, owner: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise MissingFile(f"the file {os.fspath(path)!r} does not exist", key, owner) from None
    except IsADirectoryError:
        raise MissingFile(f"{os.fspath(path)!r} is a folder, not a file", key, owner) from None
    except OSError as error:  # no permission, or a folder on systems that say so
        raise MissingFile(
            f"the file {os.fspath(path)!r} cannot be read: {error.strerror or error}", key, owner
        ) from None


def read(path: str | os.PathLike) -> ProblemFile:
    """Read the problem file at ``path`` and check it: TOML (read with the standard library's ``tomllib``),
    ``schema = "michell/0"``, every key known, every needed key there, every value of the right type and in
    range, every name it uses defined exactly once, every geometry file present (paths are relative to the
    problem file). Returns a ``ProblemFile``; refuses with a ``ProblemFileError`` (``michell.errors``: one
    code per kind of fault, the key's ``path`` in the file and a sentence for people). The first fault
    found is the one reported. The format is in ``docs/design/problem-file.md``."""
    path = Path(path).absolute()
    data = _file_bytes(path, "", path)
    r = _Reader(path)
    try:
        content = tomllib.loads(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise r.error(InvalidToml, f"the file is not UTF-8 text ({error.reason} at byte {error.start})", "") from None
    except tomllib.TOMLDecodeError as error:
        raise r.error(InvalidToml, f"the file is not valid TOML: {error}", "") from None

    if "schema" not in content:
        raise r.error(MissingKey, f"the file needs the key 'schema' (schema = \"{SCHEMA}\")", "schema")
    if content["schema"] != SCHEMA:
        raise r.error(UnknownSchema, f'michell reads schema = "{SCHEMA}"; got {content["schema"]!r}', "schema")
    r.table(
        content,
        "",
        (
            "schema",
            "regions",
            "bodies",
            "grid",
            "material",
            "supports",
            "load_cases",
            "goal",
            "limits",
            "rules",
            "run",
            "output",
        ),
        ("regions", "grid", "material", "supports", "load_cases", "goal", "limits"),
        _PLANNED_TOP,
    )

    # [regions] and [bodies]: names, each used once ("design" is the design region's).
    regions = r.table(content["regions"], "regions", ("design", "solid", "void"), ("design",))
    taken = {"design": "regions.design"}
    design = r.geometry(regions["design"], "regions.design", "design", "design")
    solid = r.named_files(regions.get("solid", {}), "regions.solid", "solid", taken)
    void = r.named_files(regions.get("void", {}), "regions.void", "void", taken)
    bodies = r.named_files(content.get("bodies", {}), "bodies", "body", taken)

    grid = r.table(content["grid"], "grid", ("voxel_mm",), ("voxel_mm",))
    voxel_mm = r.number(grid["voxel_mm"], "grid.voxel_mm", "the voxel side in mm", above=0.0)

    material = r.table(
        content["material"],
        "material",
        ("name", "youngs_modulus_mpa", "poisson_ratio", "density_g_cm3"),
        ("youngs_modulus_mpa", "poisson_ratio", "density_g_cm3"),
    )
    entry = MaterialEntry(
        r.string(material["name"], "material.name", "the material's name") if "name" in material else None,
        r.number(material["youngs_modulus_mpa"], "material.youngs_modulus_mpa", "Young's modulus in MPa", above=0.0),
        r.number(material["poisson_ratio"], "material.poisson_ratio", "Poisson's ratio", above=-1.0, below=0.5),
        r.number(material["density_g_cm3"], "material.density_g_cm3", "the density in g/cm3", above=0.0),
    )

    def targeted(value, key):
        return r.target(value, key, solid, void, bodies)

    supports = []
    for i, item in enumerate(r.tables(content["supports"], "supports")):
        key = f"supports[{i}]"
        r.table(item, key, ("on", "fix", "skin"), ("on",))
        supports.append(
            SupportEntry(
                key,
                targeted(item["on"], f"{key}.on"),
                r.directions(item["fix"], f"{key}.fix") if "fix" in item else ("x", "y", "z"),
                r.flag(item["skin"], f"{key}.skin", "skin") if "skin" in item else True,
            )
        )

    cases, case_names = [], {}
    for i, item in enumerate(r.tables(content["load_cases"], "load_cases")):
        key = f"load_cases[{i}]"
        r.table(item, key, ("name", "weight", "loads"), ("name", "loads"))
        name = r.string(item["name"], f"{key}.name", "the load case's name")
        if name in case_names:
            raise r.error(
                DuplicateName, f"the load case {name!r} is already defined at {case_names[name]}", f"{key}.name"
            )
        case_names[name] = key
        weight = r.number(item["weight"], f"{key}.weight", "the weight", at_least=0.0) if "weight" in item else 1.0
        loads = []
        for j, load in enumerate(r.tables(item["loads"], f"{key}.loads")):
            where = f"{key}.loads[{j}]"
            r.table(load, where, ("on", "force_n", "skin"), ("on", "force_n"), _PLANNED_LOAD)
            loads.append(
                LoadEntry(
                    where,
                    targeted(load["on"], f"{where}.on"),
                    r.force(load["force_n"], f"{where}.force_n"),
                    r.flag(load["skin"], f"{where}.skin", "skin") if "skin" in load else True,
                )
            )
        cases.append(CaseEntry(key, name, weight, tuple(loads)))

    goal = r.table(content["goal"], "goal", ("module", "sense"), ("module", "sense"))
    module = r.string(goal["module"], "goal.module", "the goal's module")
    if module not in GOALS:
        raise r.error(
            UndefinedName,
            f"michell has no goal module {module!r}; the goal for now is 'compliance' (sense = \"min\")",
            "goal.module",
        )
    sense = r.choice(goal["sense"], "goal.sense", f"the sense of the goal {module!r}", GOALS[module])

    mass = None
    for i, item in enumerate(r.tables(content["limits"], "limits")):
        key = f"limits[{i}]"
        r.table(item, key, ("module", "max", "max_fraction"), ("module",))
        name = r.string(item["module"], f"{key}.module", "a limit's module")
        if name != "mass":
            raise r.error(
                UndefinedName, f"michell has no limit module {name!r}; the limit for now is 'mass'", f"{key}.module"
            )
        if mass is not None:
            raise r.error(
                DuplicateName, f"the mass is already limited at {mass.key}; one mass limit for now", f"{key}.module"
            )
        if "max" in item and "max_fraction" in item:
            raise r.error(WrongType, "a mass limit has exactly one of max (g) and max_fraction; got both", key)
        if "max" in item:
            mass = MassLimit(key, r.number(item["max"], f"{key}.max", "the mass limit in g", above=0.0), None)
        elif "max_fraction" in item:
            fraction = r.number(
                item["max_fraction"],
                f"{key}.max_fraction",
                "the mass limit as a fraction of the whole part's mass",
                above=0.0,
                at_most=1.0,
            )
            mass = MassLimit(key, None, fraction)
        else:
            raise r.error(
                MissingKey, "a mass limit needs max (g) or max_fraction (of the whole part's mass)", f"{key}.max"
            )

    size, size_key = None, None
    for i, item in enumerate(r.tables(content.get("rules", []), "rules", least=0)):
        key = f"rules[{i}]"
        # The module first: the keys a rule may have depend on it.
        if "module" not in item:
            raise r.error(MissingKey, f"[{key}] needs the key 'module'", f"{key}.module")
        name = r.string(item["module"], f"{key}.module", "a rule's module")
        if name in _PLANNED_RULES:
            raise r.error(NotAvailableYet, _PLANNED_RULES[name], f"{key}.module")
        if name != "min_member":
            raise r.error(
                UndefinedName, f"michell has no rule module {name!r}; the rule for now is 'min_member'", f"{key}.module"
            )
        r.table(item, key, ("module", "size_mm"), ("module", "size_mm"))
        if size is not None:
            raise r.error(DuplicateName, f"the member size is already given at {size_key}", f"{key}.module")
        size, size_key = r.number(item["size_mm"], f"{key}.size_mm", "the member size in mm", above=0.0), key

    run = r.table(content.get("run", {}), "run", ("threads", "filter_faces"))
    threads = r.whole(run["threads"], "run.threads", "the number of threads", 1) if "threads" in run else 1
    faces = r.table(run.get("filter_faces", {}), "run.filter_faces", FACES)
    filter_faces = {
        face: r.choice(word, _join("run.filter_faces", face), f"the filter face {face}", ("closed", "empty"))
        for face, word in faces.items()
    }

    output = r.table(content.get("output", {}), "output", ("folder", "fields"), planned=_PLANNED_OUTPUT)
    folder = r.string(output["folder"], "output.folder", "the output folder") if "folder" in output else DEFAULT_OUTPUT
    fields = output.get("fields", [])
    if not isinstance(fields, list) or any(not isinstance(f, str) for f in fields):
        raise r.error(WrongType, f'output.fields is a list such as ["vtk"]; got {_shown(fields)}', "output.fields")
    for i, field in enumerate(fields):
        r.choice(field, f"output.fields[{i}]", "a field kind", FIELDS)
        if field in fields[:i]:
            raise r.error(DuplicateName, f"the field kind {field!r} is asked for twice", f"output.fields[{i}]")

    return ProblemFile(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        schema=SCHEMA,
        design=design,
        solid=solid,
        void=void,
        bodies=bodies,
        voxel_mm=voxel_mm,
        material=entry,
        supports=tuple(supports),
        load_cases=tuple(cases),
        goal=(module, sense),
        mass=mass,
        member_size_mm=size,
        threads=threads,
        filter_faces=filter_faces,
        output=path.parent / folder,
        fields=tuple(fields),
    )


# ---------------------------------------------------------------------------------------------------------
# Prepare: the geometry, the voxel model and the set-up


@dataclass(frozen=True)
class WarningRecord:
    """A warning given while a problem was prepared or run: ``category`` (its class, such as
    ``michell.attach.LoadOnSupport``) and ``message``."""

    category: type
    message: str

    def as_dict(self) -> dict:
        return {"category": self.category.__name__, "message": self.message}


def _recorded(caught) -> tuple[WarningRecord, ...]:
    return tuple(WarningRecord(item.category, str(item.message)) for item in caught)


def _reissue(records) -> None:
    for record in records:
        warnings.warn(record.message, record.category, stacklevel=3)


@dataclass(frozen=True, eq=False)
class Prepared:
    """What ``prepare`` returns: everything ``michell check`` shows, nothing solved.

    - ``problem_file``: the ``ProblemFile``; ``sha256``: each geometry file's SHA-256, by its key.
    - ``model``: the ``VoxelModel`` (grid, active and held-solid voxels, volume table, bodies); ``set_up``:
      the ``SetUp`` (problem, held degrees of freedom, voxels held solid with the skins, filter faces, the
      report of every support and load).
    - ``material``; ``mass_g``: the mass bound in g; ``part_mass_g`` and ``held_mass_g``: the masses of the
      whole part (every active voxel solid) and of the voxels held solid, between which the bound lies.
    - ``size_mm``: the member size; ``size_given``: whether the file gave it (else five voxel sides).
    - ``threads``; ``warnings``: the warnings given while preparing (``WarningRecord``); ``seconds``.

    ``summary()`` gives all of it as plain data (what ``michell check --json`` prints).
    """

    problem_file: ProblemFile
    sha256: dict[str, str]
    model: VoxelModel
    set_up: SetUp
    material: Material
    mass_g: float
    part_mass_g: float
    held_mass_g: float
    size_mm: float
    size_given: bool
    threads: int
    warnings: tuple[WarningRecord, ...]
    seconds: float

    def summary(self) -> dict:
        """The prepared problem as plain data: ``problem_file``, ``geometry``, ``grid``, ``volumes`` and
        ``set_up`` (bodies, material, goal, mass limit, member size, supports, loads, filter faces,
        threads), and ``warnings``. Numbers that are not finite are None."""
        pf, model, setup = self.problem_file, self.model, self.set_up
        mass = {
            "module": "mass",
            "max_g": self.mass_g,
            "part_mass_g": self.part_mass_g,
            "held_mass_g": self.held_mass_g,
        }
        if pf.mass.max_fraction is not None:
            mass["max_fraction"] = pf.mass.max_fraction
        targets = {s.key: s.on for s in pf.supports} | {load.key: load.on for c in pf.load_cases for load in c.loads}
        keys = [s.key for s in pf.supports] + [load.key for c in pf.load_cases for load in c.loads]
        report = [_attachment(a, key, targets[key]) for a, key in zip(setup.report, keys)]
        summary = {
            "problem_file": {"name": pf.path.name, "sha256": pf.sha256},
            "geometry": [
                {"key": g.key, "role": g.role, "name": g.name, "file": g.given, "sha256": self.sha256[g.key]}
                for g in pf.geometry
            ],
            "grid": {
                "counts": list(model.counts),
                "voxel_mm": model.voxel_mm,
                "origin_mm": list(model.origin_mm),
                "voxels": int(model.grid.n_elements),
                "active": int(np.count_nonzero(model.active)),
                "held_solid": int(np.count_nonzero(setup.keep_solid.mask)),
                "held_solid_regions": int(np.count_nonzero(model.keep_solid)),
                "design": int(np.count_nonzero(model.active & ~setup.keep_solid.mask)),
            },
            "volumes": [asdict(row) for row in model.volumes],
            "set_up": {
                "bodies": {"count": model.bodies, "sizes": list(model.body_sizes)},
                "material": {
                    "name": pf.material.name,
                    "youngs_modulus_mpa": pf.material.youngs_modulus_mpa,
                    "poisson_ratio": pf.material.poisson_ratio,
                    "density_g_cm3": pf.material.density_g_cm3,
                },
                "goal": {"module": pf.goal[0], "sense": pf.goal[1]},
                "mass_limit": mass,
                "member_size": {
                    "size_mm": self.size_mm,
                    "given": self.size_given,
                    "voxels": self.size_mm / model.voxel_mm,
                },
                "supports": report[: len(pf.supports)],
                "loads": report[len(pf.supports) :],
                "filter_faces": setup.outside,
                "filter_faces_given": dict(pf.filter_faces),
                "inactive": setup.inactive,
                "threads": self.threads,
            },
            "warnings": [w.as_dict() for w in self.warnings],
        }
        return _plain(summary)


def _attachment(a: Attachment, key: str, target: Target) -> dict:
    line = {
        "key": key,
        "role": a.role,
        "case": a.case,
        "on": target.words,
        "rule": a.rule,
        "nodes": int(a.nodes.size),
        "point_mm": list(a.point_mm),
        "skin_voxels": int(a.skin.size),
        "skin_mass_g": a.skin_mass_g,
        "closed_faces": list(a.closed_faces),
    }
    if a.rule == "contact":
        line.update(faces=len(a.faces), area_mm2=a.area_mm2)
    else:
        line.update(voxels=int(a.voxels.size), volume_mm3=a.volume_mm3)
    if a.role == "support":
        line["fix"] = list(a.fix)
    else:
        line.update(force_n=list(a.force_n), dropped_n=list(a.dropped_n), dropped_share=a.dropped_share)
    return line


_REGION_IN_MESSAGE = re.compile(r"the (design|solid|void) region '([^']*)'")
_SUPPORT_IN_MESSAGE = re.compile(r"^support (\d+) \(on ")
_LOAD_IN_MESSAGE = re.compile(r"^load (\d+) of load case (['\"])(.*?)\2 \(on ")


def _with_path(error: _Refused, path: str) -> _Refused:
    if not error.path:
        error.path = path
    return error


def _region_path(error: _Refused, pf: ProblemFile) -> _Refused:
    found = _REGION_IN_MESSAGE.search(error.message)
    if found:
        role, name = found.groups()
        return _with_path(error, "regions.design" if role == "design" else _join(f"regions.{role}", name))
    return error


def _set_up_path(error: _Refused, pf: ProblemFile) -> _Refused:
    found = _SUPPORT_IN_MESSAGE.match(error.message)
    if found:
        return _with_path(error, f"supports[{found.group(1)}]")
    found = _LOAD_IN_MESSAGE.match(error.message)
    if found:
        index, name = found.group(1), found.group(3)
        for case in pf.load_cases:
            if case.name == name:
                return _with_path(error, f"{case.key}.loads[{index}]")
    return error


def _read_mesh(item: GeometryFile, owner: Path) -> tuple[Mesh, str]:
    data = _file_bytes(item.location, item.key, owner)
    try:
        mesh = read_stl(item.location).closed()
    except InvalidValue as error:
        raise _with_path(error, item.key) from None
    return mesh, hashlib.sha256(data).hexdigest()


def _threads(threads, pf: ProblemFile) -> int:
    if threads is None:
        return pf.threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise OutOfRange(
            f"the number of threads given by the caller must be a whole number of at least 1; got {threads!r}",
            "run.threads",
            pf.path,
        )
    return threads


def _prepare(file, threads=None) -> Prepared:
    started = time.perf_counter()
    pf = file if isinstance(file, ProblemFile) else read(file)
    threads = _threads(threads, pf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meshes, hashes = {}, {}
        for item in pf.geometry:
            meshes[item.key], hashes[item.key] = _read_mesh(item, pf.path)
        try:
            model = voxel_model(
                meshes[pf.design.key],
                {name: meshes[g.key] for name, g in pf.solid.items()},
                {name: meshes[g.key] for name, g in pf.void.items()},
                voxel_mm=pf.voxel_mm,
                threads=threads,
            )
        except _Refused as error:
            raise _region_path(error, pf) from None

        def target(on: Target):
            if on.kind == "region":
                return on.name
            if on.kind == "body":
                return meshes[pf.bodies[on.name].key]  # one Mesh object per file, so set_up computes it once
            return on.box

        material = pf.material.material()
        size_given = pf.member_size_mm is not None
        size_mm = pf.member_size_mm if size_given else DEFAULT_SIZE_VOXELS * pf.voxel_mm
        supports = [Support(target(s.on), fix=s.fix, skin=s.skin) for s in pf.supports]
        cases = [
            Case(c.name, [Load(target(load.on), load.force_n, skin=load.skin) for load in c.loads], c.weight)
            for c in pf.load_cases
        ]
        try:
            setup = set_up(model, material, supports, cases, size_mm, threads=threads, outside=pf.filter_faces)
        except _Refused as error:
            raise _set_up_path(error, pf) from None

    grid = model.grid
    n_active = int(np.count_nonzero(model.active))
    n_held = int(np.count_nonzero(setup.keep_solid.mask))
    # The bound from a fraction: written as the example scripts write it, fraction * mass of the whole part.
    part_mass = material.mass_g(grid.voxel_volume_mm3 * n_active)
    mass_g = pf.mass.max_g if pf.mass.max_g is not None else pf.mass.max_fraction * part_mass
    # The test of stiff_part, so that a bound it would refuse is refused here, with the key.
    element_mass = float(material.mass_g(grid.voxel_volume_mm3))
    design = n_active - n_held
    fraction = (mass_g / element_mass - n_held) / design if design else 0.0
    held_mass = element_mass * n_held
    if not 0.0 < fraction <= 1.0:
        key = f"{pf.mass.key}.{'max' if pf.mass.max_g is not None else 'max_fraction'}"
        raise OutOfRange(
            f"the mass bound {mass_g:.6g} g must lie above the mass of the voxels held solid ({held_mass:.6g} g: the "
            f"solid regions and the skins under the supports and loads) and at most at the mass of the whole part "
            f"({element_mass * n_active:.6g} g)",
            key,
            pf.path,
            {"mass_g": mass_g, "held_mass_g": held_mass, "part_mass_g": element_mass * n_active},
        )
    return Prepared(
        problem_file=pf,
        sha256=hashes,
        model=model,
        set_up=setup,
        material=material,
        mass_g=float(mass_g),
        part_mass_g=float(part_mass),
        held_mass_g=float(held_mass),
        size_mm=float(size_mm),
        size_given=size_given,
        threads=threads,
        warnings=_recorded(caught),
        seconds=time.perf_counter() - started,
    )


def prepare(file: str | os.PathLike | ProblemFile, *, threads: int | None = None) -> Prepared:
    """Read the problem file ``file`` (``read``), read its STL files, build the voxel model
    (``michell.regions.voxel_model``) and place the supports and loads (``michell.attach.set_up``), on
    ``threads`` threads (the file's ``[run] threads`` by default). Nothing is solved.

    The member size is the ``min_member`` rule's, or five voxel sides without one (``size_given`` says
    which). The mass bound is ``[[limits]] max`` in g, or ``max_fraction`` times the mass of the whole part
    with every active voxel solid; it must lie above the mass of the voxels held solid (the solid regions and
    the skins) and at most at the whole part's, or it is refused (``out_of_range``) before any pass.

    Returns a ``Prepared``. Refuses with a ``ProblemFileError`` (the file) or, for the geometry and the
    set-up, with the refusals of ``michell.mesh``, ``michell.regions`` and ``michell.attach`` (code
    ``invalid_input``, with ``path`` set to the entry that caused it) and the engine's ``singular_system``
    when a body of the part is not held. Warnings given on the way (``VoxelVolumeDiffers``,
    ``LoadOnSupport``) are kept in ``Prepared.warnings`` and issued again."""
    prepared = _prepare(file, threads)
    _reissue(prepared.warnings)
    return prepared


# ---------------------------------------------------------------------------------------------------------
# Run: stiff_part and the files


@dataclass(frozen=True, eq=False)
class RunOutcome:
    """What ``run`` returns.

    - ``prepared``: the ``Prepared`` problem; ``part``: the ``StiffPart`` of ``stiff_part`` (its ``result``
      holds the goal, the limits and ``limits_met``).
    - ``paths``: the files written, by kind (``"json"``, ``"npz"`` and, when asked, ``"vtk"``).
    - ``seconds``: the seconds the optimisation took (``prepared.seconds`` those of the preparation).
    - ``warnings``: every warning given while preparing and running (``WarningRecord``).
    """

    prepared: Prepared
    part: StiffPart
    paths: dict[str, Path]
    seconds: float
    warnings: tuple[WarningRecord, ...]

    @property
    def limits_met(self) -> bool:
        """Whether the returned design meets every limit."""
        return bool(self.part.result.feasible)


def _optimise(
    prepared: Prepared,
    out: str | os.PathLike | None = None,
    callback: Callable[[str, int, RecipePass], None] | None = None,
    **settings,
) -> RunOutcome:
    """``stiff_part`` on a prepared problem and the files. ``settings`` go to ``stiff_part`` (the tests use
    them to shorten the schedule); a run from a problem file passes none."""
    pf, setup = prepared.problem_file, prepared.set_up
    folder = Path(out).absolute() if out is not None else pf.output
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutOfRange(
            f"the output folder {os.fspath(folder)!r} cannot be made: {error.strerror or error}",
            "output.folder",
            pf.path,
        ) from None
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        part = setup.stiff_part(prepared.mass_g, callback=callback, **settings)
    seconds = time.perf_counter() - started
    records = prepared.warnings + _recorded(caught)
    base = folder / RESULT_NAME
    paths = {"json": base.with_suffix(".json"), "npz": base.with_suffix(".npz")}
    if "vtk" in pf.fields:
        paths["vtk"] = base.with_suffix(".vtk")
        write_vtk(
            paths["vtk"],
            setup.problem.grid,
            nominal=part.nominal,
            robust_step=part.first_step_nominal,
            held_solid=setup.keep_solid.mask,
            active=prepared.model.active,
        )
    summary = prepared.summary()
    record = {
        "problem_file": summary["problem_file"],
        "geometry": summary["geometry"],
        "grid": summary["grid"],
        "volumes": summary["volumes"],
        "set_up": summary["set_up"],
        "warnings": [w.as_dict() for w in records],
        "files": [path.name for path in paths.values()],
        "seconds": {"prepare": prepared.seconds, "optimise": seconds},
    }
    write_result(base, part, record=record)
    return RunOutcome(prepared=prepared, part=part, paths=paths, seconds=seconds, warnings=records)


def run(
    file: str | os.PathLike | ProblemFile,
    *,
    out: str | os.PathLike | None = None,
    threads: int | None = None,
    callback: Callable[[str, int, RecipePass], None] | None = None,
) -> RunOutcome:
    """Run the problem file ``file``: ``prepare`` it (on ``threads`` threads, the file's ``[run] threads``
    by default), find the stiffest part within the mass bound and with the member size
    (``Prepared.set_up.stiff_part``, the two-step procedure of ``michell.recipes.stiff_part`` with its
    default settings), and write the outcome into the folder ``out`` (the file's ``[output] folder`` by
    default, relative to the problem file; created if needed):

    - ``result.npz`` and ``result.json`` (``michell.export.write_result``, format 2: the content of format 1
      and the run's record: the problem file's name and SHA-256, each geometry file's, the grid and its
      origin, the volume table, the set-up report, the warnings, the files written and the seconds used);
    - ``result.vtk`` when ``[output] fields`` holds ``"vtk"``: the nominal design, the design after the
      robust step, the voxels held solid and the active voxels (``michell.export.write_vtk``).

    ``callback`` is passed to ``stiff_part`` (called after every pass with the step, the pass number and its
    ``RecipePass``). Returns a ``RunOutcome`` (the outcome and the paths). A design that misses a limit is
    returned all the same (``RunOutcome.limits_met`` is False, and ``stiff_part`` warns ``LimitNotMet``).
    Refusals are those of ``prepare``; an ``EngineError`` during the passes leaves with its
    ``partial_result`` (``stiff_part``), and nothing is written. Warnings are kept in
    ``RunOutcome.warnings`` and in ``result.json``, and issued again."""
    outcome = _optimise(_prepare(file, threads), out, callback)
    _reissue(outcome.warnings)
    return outcome


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SIZE_VOXELS",
    "FIELDS",
    "GOALS",
    "RESULT_NAME",
    "SCHEMA",
    "CaseEntry",
    "GeometryFile",
    "LoadEntry",
    "MassLimit",
    "MaterialEntry",
    "Prepared",
    "ProblemFile",
    "RunOutcome",
    "SupportEntry",
    "Target",
    "WarningRecord",
    "prepare",
    "read",
    "run",
]
