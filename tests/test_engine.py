"""Tests for the Python face of michell: import, versions and the Rust boundary."""

import subprocess
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import michell
from michell import _engine
from michell.cli import main


def test_package_version_matches_metadata():
    assert michell.__version__ == version("michell") == "0.0.1"


def test_engine_version_is_the_rust_crate_version():
    assert _engine.engine_version() == "0.0.1"


def test_sum_f64_adds_an_array():
    values = np.array([1.0, 2.0, 3.5], dtype=np.float64)
    assert _engine.sum_f64(values) == 6.5


def test_sum_f64_of_empty_array_is_zero():
    assert _engine.sum_f64(np.array([], dtype=np.float64)) == 0.0


def test_sum_f64_matches_numpy_on_many_values():
    values = np.linspace(0.0, 1.0, 1001)
    assert _engine.sum_f64(values) == pytest.approx(values.sum())


def test_sum_f64_refuses_a_non_contiguous_array():
    every_second = np.arange(10, dtype=np.float64)[::2]
    assert not every_second.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        _engine.sum_f64(every_second)
    # The error message's advice works.
    assert _engine.sum_f64(np.ascontiguousarray(every_second)) == 20.0


def test_sum_f64_refuses_other_dtypes():
    with pytest.raises(TypeError):
        _engine.sum_f64(np.array([1, 2, 3], dtype=np.int64))


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "michell 0.0.1 (engine 0.0.1)"


def test_installed_console_script():
    script = "michell.exe" if sys.platform == "win32" else "michell"
    command = Path(sysconfig.get_path("scripts")) / script
    result = subprocess.run([command, "--version"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "michell 0.0.1 (engine 0.0.1)"
