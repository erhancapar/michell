"""Tests for the performance measurement: the page written from a fixed result, the tie between the
committed measurements and the committed page, and a short run in fresh processes."""

import getpass
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from michell.performance import scaled_passes
from michell.performance.page import render

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent / "performance"


def test_the_page_of_the_fixture_is_the_stored_page():
    data = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
    assert render(data) == (HERE / "expected.md").read_text(encoding="utf-8")


def test_the_committed_page_is_written_from_the_committed_measurements():
    measurements, page = ROOT / "docs" / "performance.json", ROOT / "docs" / "performance.md"
    if not measurements.exists() and not page.exists():
        pytest.skip("no measurements committed yet")
    assert measurements.exists() and page.exists(), "the measurements and the page are committed together"
    data = json.loads(measurements.read_text(encoding="utf-8"))
    assert page.read_text(encoding="utf-8") == render(data)


def test_the_page_refuses_another_format():
    with pytest.raises(ValueError, match="format"):
        render({"format": 2})


def test_the_passes_scale_shortens_every_count_to_at_least_one():
    schedule, second = scaled_passes(1.0)
    assert sum(passes for _, passes in schedule) == 100 and second == 40
    schedule, second = scaled_passes(0.01)
    assert all(passes == 1 for _, passes in schedule) and second == 1


SIZE_KEYS = {
    "size", "grid", "elements", "voxel_mm", "filter_radius_elements", "passes", "seconds_per_pass",
    "total_seconds", "peak_memory_gb", "solver_steps_per_solve", "goal", "limits_met", "stop_reason", "warnings",
    "seconds_of_each_pass", "solver_steps_of_each_pass",
}  # fmt: skip
MACHINE_KEYS = {
    "cpu", "logical_cores", "memory_gb", "os", "os_version", "os_distribution", "python", "michell", "engine",
    "threads",
}  # fmt: skip


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


@pytest.mark.slow
def test_a_short_run_writes_every_key_and_nothing_that_names_the_machine(tmp_path):
    out = tmp_path / "performance.json"
    command = [sys.executable, "-m", "michell.performance", "run", "--sizes", "8", "--passes-scale", "0.1"]
    subprocess.run(command + ["--threads", "2", "--out", str(out)], capture_output=True, text=True, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"format", "machine", "problem", "sizes"} and data["format"] == 1
    assert set(data["machine"]) == MACHINE_KEYS and data["machine"]["threads"] == 2
    assert data["problem"]["passes_scale"] == 0.1
    (one,) = data["sizes"]
    assert set(one) == SIZE_KEYS
    assert one["grid"] == [16, 8, 8] and one["elements"] == 1024
    expected = sum(passes for _, passes in data["problem"]["schedule"]) + data["problem"]["second_step_passes"]
    assert one["passes"] == expected == len(one["seconds_of_each_pass"]) == len(one["solver_steps_of_each_pass"])
    assert set(one["seconds_per_pass"]) == {"mean", "first_step_mean", "second_step_mean", "slowest_ten_mean"}
    assert all(value > 0 for value in one["seconds_per_pass"].values())
    assert one["solver_steps_per_solve"]["mean"] > 0 and one["solver_steps_per_solve"]["largest"] > 0
    assert one["peak_memory_gb"] is None or one["peak_memory_gb"] > 0
    assert "GridTooCoarse" in one["warnings"]  # 3 mm on elements of 2.5 mm
    # Whole words only, case kept: a host may be named like its system ("arch" and "Arch Linux").
    names = {socket.gethostname(), socket.gethostname().split(".")[0], getpass.getuser()}
    for text in strings_in(data):
        for name in names:
            assert not re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", text), f"{name!r} in {text!r}"
        assert str(Path.home()) not in text and str(tmp_path) not in text
    assert render(data).startswith("# Performance\n")
