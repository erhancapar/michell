"""The benchmark script runs and prints one JSON line with its measurements."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_the_3d_benchmark_prints_one_json_line():
    command = [sys.executable, str(ROOT / "benchmarks" / "cantilever_3d.py"), "--nx", "8", "--ny", "4", "--nz", "4"]
    result = subprocess.run(command + ["--threads", "2", "--passes", "3"], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["shape"] == [8, 4, 4] and summary["threads"] == 2 and summary["passes"] == 3
    assert len(summary["seconds_per_pass"]) == len(summary["solver_steps_per_pass"]) == 3
    assert all(seconds > 0 for seconds in summary["seconds_per_pass"])
    assert all(steps > 0 for steps in summary["solver_steps_per_pass"])
    assert summary["peak_memory_mb"] is None or summary["peak_memory_mb"] > 0
