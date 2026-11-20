"""Tests for the verification table itself: rows without their data, the page, the exit status, and the
MBB example with its picture."""

import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from michell import verification
from michell.row import checked, not_run
from michell.verification.page import exit_status, render

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("reference", [None, ROOT / "no such folder"], ids=["absent", "missing folder"])
def test_rows_without_reference_data_report_not_run(reference):
    rows = (
        verification.row_6_first_pass(reference)
        + verification.row_8_mbb_with_oc(reference, "mbb_150x50_r6_density")
        + verification.row_10_mbb_with_mma(reference)
    )
    assert len(rows) == 6
    for row in rows:
        assert row.passed is None and row.measured is None and not row.was_run
        assert row.note == "not run: reference data not found"


def test_a_row_computes_its_result_from_the_limit():
    assert checked(1, "a", "closed form", 1e-12, 1e-12, "").passed
    assert not checked(1, "a", "closed form", 1e-12, 1e-12, "", strict=True).passed
    assert not checked(1, "a", "closed form", 2e-12, 1e-12, "").passed


def hand_made_rows():
    return [
        checked(1, "first", "closed form", 2.2449e-16, 1e-12, "a note | with a bar"),
        not_run(6, "second", "reference code", 1e-8, "reference data not found"),
        checked(5, "third", "closed form", 0.029, 0.03, "close", strict=True),
    ]


def test_the_page_rounds_to_three_digits_and_names_the_version_and_commit():
    page = render(hand_made_rows(), "1.2.3", "1.2.3", commit="abc1234")
    assert "michell 1.2.3 (engine 1.2.3), commit `abc1234`." in page
    assert "| 1 | first | closed form | 2.24e-16 | ≤ 1e-12 | pass | a note \\| with a bar |" in page
    assert "| 6 | second | reference code |  | ≤ 1e-08 | not run | not run: reference data not found |" in page
    assert "| 5 | third | closed form | 0.029 | < 0.03 | pass | close |" in page
    assert "2 passed, 0 failed, 1 not run." in page
    assert "python -m michell.verification --reference tests/reference --out docs/verification.md" in page
    assert "Version: michell 1.2.3 (engine 1.2.3).\n" in render(hand_made_rows(), "1.2.3", "1.2.3")


def test_the_exit_status_fails_on_a_failed_row_and_optionally_on_a_row_not_run():
    rows = hand_made_rows()
    assert exit_status(rows) == 0
    assert exit_status(rows, require_all=True) == 1
    failed = rows + [checked(2, "fourth", "closed form", 1.0, 1e-10, "")]
    assert exit_status(failed) == 1
    assert exit_status([rows[0], rows[2]], require_all=True) == 0


def test_the_readme_example_is_taken_from_the_example_script():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("```python\n", 1)[1].split("```", 1)[0]
    script = (ROOT / "examples" / "mbb_beam.py").read_text(encoding="utf-8").splitlines()
    position = 0
    for line in block.splitlines():
        if not line.strip():
            continue
        assert line in script[position:], f"README line not found in order in the example script: {line!r}"
        position = script.index(line, position) + 1


def read_greyscale_png(path):
    """Width, height and pixel bytes of a PNG written by the example (one IDAT chunk, filter type 0)."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, offset = {}, 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(kind + body)
        chunks[kind] = body
        offset += 12 + length
    width, height, depth, colour, _, _, _ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, colour) == (8, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    assert len(raw) == height * (width + 1)
    return width, height, raw


@pytest.mark.slow
def test_the_mbb_example_runs_and_writes_its_picture(tmp_path):
    picture = tmp_path / "mbb.png"
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "mbb_beam.py"), str(picture)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "compliance" in result.stdout
    width, height, raw = read_greyscale_png(picture)
    assert (width, height) == (600, 200)
    pixels = [raw[row * (width + 1) + 1 : (row + 1) * (width + 1)] for row in range(height)]
    assert min(min(row) for row in pixels) < 10 and max(max(row) for row in pixels) > 245  # solid and void
