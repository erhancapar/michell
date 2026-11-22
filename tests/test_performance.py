"""Tests for the performance measurement: the page written from a fixed result, the tie between the
committed measurements and the committed page, the record of a run that fails, the saving of the result after
every size, and a short run in fresh processes."""

import getpass
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import michell
from michell import performance
from michell.performance import measure, scaled_passes
from michell.performance.__main__ import main
from michell.performance.page import render
from michell.recipes import stiff_part

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
    "seconds_of_each_pass", "solver_steps_of_each_pass", "error",
}  # fmt: skip
ERROR_KEYS = {"type", "code", "message", "details", "step", "pass_in_step", "pass", "exit_status", "error_output"}
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


def failing_after(passes: int, error: Exception):
    """A ``stiff_part`` whose solves raise ``error`` once ``passes`` passes are complete."""

    def failing(problem, *arguments, callback, **settings):
        completed = []
        solve = problem.solve

        def failing_solve(*solve_arguments, **solve_settings):
            if len(completed) >= passes:
                raise error
            return solve(*solve_arguments, **solve_settings)

        def note(step, number, record):
            completed.append(number)
            callback(step, number, record)

        problem.solve = failing_solve
        return stiff_part(problem, *arguments, callback=note, **settings)

    return failing


def not_converged():
    error = michell.EngineError("solver_not_converged: the solver did not reach the tolerance")
    error.code, error.message = "solver_not_converged", "the solver did not reach the tolerance"
    error.details = {"steps": 5, "relative_residual": float("nan"), "tolerance": 1e-8}
    return error


@pytest.mark.parametrize(
    "passes, error, expected",
    [
        (3, not_converged(), ("EngineError", "solver_not_converged", "robust", 4)),
        (6, MemoryError("out of memory"), ("MemoryError", None, "member size", 1)),
    ],
)
def test_a_run_that_fails_keeps_its_completed_passes_and_the_error(monkeypatch, passes, error, expected):
    # With the passes scaled by 0.05 every count of the schedule is 1: 6 robust passes, then 2.
    monkeypatch.setattr(performance, "stiff_part", failing_after(passes, error))
    one = measure(8, 1, 0.05)
    assert set(one) == SIZE_KEYS and set(one["error"]) == ERROR_KEYS
    assert one["passes"] == passes == len(one["seconds_of_each_pass"]) == len(one["solver_steps_of_each_pass"])
    assert all(seconds > 0 for seconds in one["seconds_of_each_pass"])
    assert one["seconds_per_pass"]["mean"] > 0 and one["solver_steps_per_solve"]["solves"] > 0
    assert one["peak_memory_gb"] is None or one["peak_memory_gb"] > 0
    assert (one["goal"], one["limits_met"], one["stop_reason"]) == (None, None, None)
    found = one["error"]
    assert (found["type"], found["code"], found["step"], found["pass_in_step"]) == expected
    assert found["pass"] == passes + 1
    if expected[1]:
        assert found["message"] == error.message
        assert found["details"] == {"steps": 5, "relative_residual": None, "tolerance": 1e-8}
    else:
        assert found["message"] == str(error) and found["details"] == {}
    assert (found["exit_status"], found["error_output"]) == (None, [])
    json.loads(json.dumps(one, allow_nan=False))  # strict JSON: no NaN reaches the file


def test_the_parent_saves_after_every_size_records_every_failure_and_ends_with_1(monkeypatch, tmp_path, capsys):
    fixture = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
    completed, raised = fixture["sizes"][0], fixture["sizes"][2]
    out = tmp_path / "performance.json"
    host = socket.gethostname()
    seen = []

    def child(command, **settings):
        seen.append(len(json.loads(out.read_text(encoding="utf-8"))["sizes"]) if out.exists() else None)
        size = int(command[3])
        if size == completed["size"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(completed) + "\n", "")
        if size == raised["size"]:
            return subprocess.CompletedProcess(command, 0, "a line before\n" + json.dumps(raised) + "\n", "")
        output = (
            "Traceback (most recent call last):\n"
            '  File "/home/someone/work/michell/recipes.py", line 3, in stiff_part\n'
            f"  on {host}: C:\\Users\\someone\\run.py: 2e5 N/mm\n"
            "MemoryError\n"
        )
        return subprocess.CompletedProcess(command, -9, "", output)

    monkeypatch.setattr(performance.subprocess, "run", child)
    assert main(["run", "--sizes", "20", "60", "7", "--threads", "4", "--out", str(out)]) == 1
    assert seen == [None, 1, 2]  # the file holds every size before the next one starts
    assert list(tmp_path.iterdir()) == [out]  # no temporary file is left
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [one["size"] for one in data["sizes"]] == [20, 60, 7]
    assert data["sizes"][:2] == [completed, raised]
    died = data["sizes"][2]
    assert set(died) == SIZE_KEYS and set(died["error"]) == ERROR_KEYS
    assert died["grid"] == [14, 7, 7] and died["passes"] is None and died["error"]["exit_status"] == -9
    assert died["error"]["error_output"] == [
        "Traceback (most recent call last):",
        '  File "recipes.py", line 3, in stiff_part',
        "  on (host): run.py: 2e5 N/mm",
        "MemoryError",
    ]
    page = " ".join(render(data).split())
    assert (
        "At 120 x 60 x 60 the run stopped in pass 117 of 140, pass 17 of the member-size step, with the error "
        "`solver_not_converged`: the solver did not reach the relative residual 1e-8 within 2000 steps; it "
        "reached 3.2e-7." in page
    )
    assert (
        "At 14 x 7 x 7 the process of the run ended with status -9 (stopped by signal 9) before it gave its "
        "measurements; the last line of its error output: `MemoryError`." in page
    )
    assert "failed: the process ended with status -9" in capsys.readouterr().out
    assert main(["run", "--sizes", "20", "--threads", "4", "--out", str(out)]) == 0


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
    assert set(one) == SIZE_KEYS and one["error"] is None
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
