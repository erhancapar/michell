"""What ``michell check`` adds to the set-up: the estimate of time and memory (``michell.estimate``) and the
picture of the set-up (``michell.picture``), with the command's options and its JSON output, and verification
rows 38 and 39."""

import json
from pathlib import Path

import numpy as np
import pytest

import michell
from michell import cli, estimate, mesh, picture
from michell.estimate.__main__ import main as fit_main
from michell.verification.check_rows import (
    row_38_estimate,
    row_38_held_out,
    row_39_classes,
    row_39_hue,
    row_39_picture,
)

ROOT = Path(__file__).parent.parent

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

[[load_cases]]
name = "side"
loads = [ { on = { box_mm = [[12, -1, -1], [13, 7, 7]] }, force_n = [0, 1, 0] } ]

[goal]
module = "compliance"
sense = "min"

[[limits]]
module = "mass"
max_fraction = 0.5
"""

FAKE_TIMING = {"threads": 1, "solve_seconds": 0.0, "filter_seconds": 0.0, "seconds": 0.0}


@pytest.fixture
def bar(tmp_path):
    mesh.write_stl(tmp_path / "bar.stl", mesh.box((0, 0, 0), (12, 6, 6)))
    mesh.write_stl(tmp_path / "wall.stl", mesh.box((-1, -1, -1), (0, 7, 7)))
    path = tmp_path / "part.toml"
    path.write_text(BAR, encoding="utf-8")
    return path


@pytest.fixture
def doubled_timing(monkeypatch):
    """A timing that reports this machine as twice as slow as the reference at the solve and three times at
    the filter, without timing anything."""
    reference = estimate.load_model()["timing"]
    fake = dict(
        FAKE_TIMING, solve_seconds=2 * reference["solve_seconds"], filter_seconds=3 * reference["filter_seconds"]
    )
    monkeypatch.setattr(estimate, "timing", lambda threads: dict(fake, threads=threads))
    return fake


# ---------------------------------------------------------------------------------------------------------
# The model


def test_the_shipped_model_is_a_fresh_fit_of_the_committed_measurements():
    shipped = estimate.load_model()
    fresh = estimate.fit(json.loads((ROOT / "docs" / "performance.json").read_text(encoding="utf-8")))
    assert set(shipped) - {"timing", "extra_load_case"} == set(fresh)

    def same(a, b, where):
        if isinstance(a, dict):
            assert set(a) == set(b), where
            for key in a:
                same(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), where
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{where}[{i}]")
        elif isinstance(a, float):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), where
        else:
            assert a == b, where

    for key in fresh:
        same(shipped[key], fresh[key], key)
    timing_keys = {"threads", "solve_seconds", "filter_seconds", "seconds", "date", "michell", "engine"}
    assert set(shipped["timing"]) == timing_keys
    assert shipped["extra_load_case"]["measured"].startswith("the difference of the peak memory")


def test_the_fit_reproduces_the_measured_sizes():
    model = estimate.load_model()
    assert model["seconds_per_pass"]["largest_share_off_held_out"] < 0.05
    assert model["peak_memory_gb"]["largest_share_off_held_out"] < 0.2
    held_out = {row["elements"]: row for row in model["held_out"]}
    assert set(held_out) == {row["elements"] for row in model["sizes"]}
    assert model["peak_memory_gb"]["largest_share_off_held_out"] == max(
        abs(row["memory_share_off"]) for row in model["held_out"]
    )
    for row in model["sizes"]:
        assert row["fitted_seconds_per_pass"] == pytest.approx(row["seconds_per_pass"], rel=0.05)


def test_the_fit_refuses_other_measurements():
    data = json.loads((ROOT / "docs" / "performance.json").read_text(encoding="utf-8"))
    with pytest.raises(michell.errors.InvalidValue, match="format 1"):
        estimate.fit(dict(data, format=2))
    failed = dict(data, sizes=[dict(data["sizes"][0], error={"type": "x"})])
    with pytest.raises(michell.errors.InvalidValue, match="completed"):
        estimate.fit(failed)


def test_the_fitting_command_writes_the_same_model(tmp_path, capsys):
    out = tmp_path / "model.json"
    assert fit_main(["fit", str(ROOT / "docs" / "performance.json"), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    shipped = estimate.load_model()
    assert written["timing"] == shipped["timing"]  # kept: no --timing
    assert written["extra_load_case"] == shipped["extra_load_case"]
    assert written["seconds_per_pass"] == pytest.approx(shipped["seconds_per_pass"], rel=1e-9)
    assert "written to" in capsys.readouterr().out


def test_the_stencil_counts_the_cells_the_filter_sums_over():
    assert estimate.stencil_cells(0.5) == 1
    assert estimate.stencil_cells(1.0) == 1  # the neighbour one side away has weight 0
    assert estimate.stencil_cells(1.01) == 27
    assert estimate.stencil_cells(5.4) == 11**3
    assert estimate.stencil_cells(5.0) == 9**3
    assert estimate.stencil_cells(20.0, largest=6) == 13**3  # the reach is at most the grid's largest count
    assert estimate.stencil_cells(5.4, largest=40) == 11**3


def test_the_fit_needs_three_sizes_to_leave_one_out():
    model = estimate.load_model()
    with pytest.raises(michell.errors.InvalidValue, match="three measured sizes"):
        estimate.fit_sizes(model["sizes"][:2], model["reference"], model["passes"])
    four = estimate.fit_sizes(model["sizes"][1:], model["reference"], model["passes"])
    assert len(four["held_out"]) == 4 and four["measured"]["elements"] == [194672, 1458000]


# ---------------------------------------------------------------------------------------------------------
# The estimate


def test_the_estimate_grows_with_the_size_the_load_cases_and_the_filter():
    sizes = np.geomspace(1e4, 3e6, 12).round().astype(int)
    totals = [estimate.estimate(int(n), int(n), 5.0, 1, 12).total_seconds for n in sizes]
    memory = [estimate.estimate(int(n), int(n), 5.0, 1, 12).peak_memory_gb for n in sizes]
    for kind in (totals, memory):
        for low_centre_high in kind:
            assert low_centre_high[0] <= low_centre_high[1] <= low_centre_high[2]
        assert all(b[1] > a[1] for a, b in zip(kind, kind[1:]))
    cases = [estimate.estimate(10**6, 10**6, 5.0, c, 12).seconds_per_pass[1] for c in (1, 2, 3)]
    assert cases[0] < cases[1] < cases[2]
    memory_by_cases = [estimate.estimate(10**6, 10**6, 5.0, c, 12).peak_memory_gb[1] for c in (1, 2, 3)]
    assert memory_by_cases[0] < memory_by_cases[1] < memory_by_cases[2]
    assert estimate.estimate(10**6, 10**6, 5.0, 1, 12).seconds_per_pass[1] < estimate.estimate(
        10**6, 10**6, 9.0, 1, 12
    ).seconds_per_pass[1]


def test_the_estimate_at_the_reference_is_not_scaled_and_says_where_it_holds():
    inside = estimate.estimate(500_000, 500_000, 8.0, 1, 12)
    assert not inside.scaled and inside.timing is None and inside.within_measured and inside.outside == ()
    assert inside.passes == 140 and inside.total_seconds[1] == pytest.approx(140 * inside.seconds_per_pass[1])
    small = estimate.estimate(1000, 1000, 8.0, 1, 12)
    assert not small.within_measured and "1000 active voxels" in small.outside[0]
    wide = estimate.estimate(500_000, 500_000, 20.0, 1, 12)
    assert not wide.within_measured and "filter radius of 20" in wide.outside[0]
    narrow = estimate.estimate(500_000, 500_000, 3.0, 1, 12)
    assert narrow.within_measured  # below the measured radii the filter's term only shrinks: no warning
    cases = estimate.estimate(500_000, 500_000, 8.0, 3, 12)
    assert cases.outside == (cases.outside[0],) and "3 load cases, more than the measured 1" in cases.outside[0]
    threads = estimate.estimate(500_000, 500_000, 8.0, 1, 4)
    assert "reference machine's on its 12 threads, not on the problem's 4" in threads.outside[0]
    timed = dict(estimate.load_model()["timing"], threads=4)
    assert estimate.estimate(500_000, 500_000, 8.0, 1, 4, timed=timed).outside == ()
    data = inside.as_dict()
    assert data["reference"]["threads"] == 12 and data["covers"].startswith("the passes")
    assert json.loads(json.dumps(data, allow_nan=False)) == data


def test_a_timing_scales_the_terms_by_its_ratios(doubled_timing):
    plain = estimate.estimate(500_000, 500_000, 8.0, 2, 12)
    scaled = estimate.estimate(500_000, 500_000, 8.0, 2, 12, timed=True)
    assert scaled.scaled and scaled.timing["solve_ratio"] == pytest.approx(2.0)
    assert scaled.timing["filter_ratio"] == pytest.approx(3.0)
    assert 2.0 * plain.seconds_per_pass[1] < scaled.seconds_per_pass[1] < 3.0 * plain.seconds_per_pass[1]
    assert scaled.peak_memory_gb == plain.peak_memory_gb


def test_the_timing_is_short_and_reports_both_workloads():
    measured = estimate.timing(1, budget=0.05)
    assert measured["threads"] == 1 and measured["solve_seconds"] > 0 and measured["filter_seconds"] > 0
    assert measured["engine"] == michell.engine_version() and measured["michell"] == michell.__version__
    assert len(measured["date"]) == 10
    assert measured["seconds"] < 10
    with pytest.raises(michell.errors.InvalidValue):
        estimate.timing(0)


# ---------------------------------------------------------------------------------------------------------
# prepare and the command line


def test_prepare_gives_the_reference_estimate_unless_asked_to_time(bar, doubled_timing):
    prepared = michell.prepare(bar)
    assert not prepared.estimate.scaled and prepared.estimate.load_cases == 2
    assert prepared.estimate.active == 432 and prepared.estimate.filter_radius_elements == 5.0
    timed = michell.prepare(bar, timing=True)
    assert timed.estimate.scaled and timed.estimate.timing["threads"] == 1
    assert prepared.summary()["estimate"]["scaled"] is False


def test_check_prints_the_estimate_and_its_source(bar, capsys, doubled_timing):
    assert cli.main(["check", str(bar)]) == 0
    out = capsys.readouterr().out
    assert "estimate, scaled to this machine" in out and "solve x2" in out
    assert "s per pass" in out and "peak memory" in out and "not counting the mesh of the part" in out
    assert "warning (estimate): outside what was measured" in out and "not on the problem's" not in out
    assert cli.main(["check", str(bar), "--no-timing"]) == 0
    out = capsys.readouterr().out
    assert "for the reference machine (AMD" in out and "not scaled to this one" in out
    assert "on its 12 threads, not on the problem's 1" in out


def test_run_prints_no_estimate(bar, tmp_path, capsys, monkeypatch):
    def stop(*args, **kwargs):
        raise michell.errors.InvalidValue("stopped before the passes")

    monkeypatch.setattr(cli, "_optimise", stop)
    assert cli.main(["run", str(bar), "--out", str(tmp_path / "out")]) != 0
    out = capsys.readouterr().out
    assert "read and checked" in out and "estimate," not in out and "peak memory" not in out


def test_check_json_holds_the_estimate(bar, capsys):
    assert cli.main(["check", str(bar), "--json", "--no-timing"]) == 0
    data = json.loads(capsys.readouterr().out)["prepared"]["estimate"]
    assert data["scaled"] is False and data["timing"] is None and data["load_cases"] == 2
    for key in ("seconds_per_pass", "total_seconds", "peak_memory_gb"):
        assert data[key]["low"] <= data[key]["centre"] <= data[key]["high"]
    assert data["within_measured"] is False and data["outside"]


def test_check_writes_the_picture_and_names_it(bar, tmp_path, capsys):
    svg = tmp_path / "set-up.svg"
    assert cli.main(["check", str(bar), "--no-timing", "--picture", str(svg), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["prepared"]["picture"] == str(svg)
    text = svg.read_text(encoding="utf-8")
    assert text.startswith("<svg") and text.count("<image ") == 4
    for words in ("top (from +z)", "front (from -y)", "right (from +x)", "oblique", " mm</text>", "tip", "side"):
        assert words in text
    assert picture.SUPPORT_HEX in text and picture.LOAD_HEX in text
    assert cli.main(["check", str(bar), "--no-timing", "--picture", str(svg)]) == 0
    assert f"picture of the set-up written to {svg}" in capsys.readouterr().out


def test_a_picture_that_cannot_be_written_is_refused(bar, tmp_path, capsys):
    target = tmp_path / "no such folder" / "set-up.svg"
    assert cli.main(["check", str(bar), "--no-timing", "--picture", str(target), "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "out_of_range" and "cannot be written" in error["message"]


# ---------------------------------------------------------------------------------------------------------
# The picture


def test_the_picture_is_the_same_bytes_every_time(bar, tmp_path):
    prepared = michell.prepare(bar)
    first = picture.write_picture(prepared, tmp_path / "a.svg").read_bytes()
    again = picture.write_picture(michell.prepare(bar), tmp_path / "b.svg").read_bytes()
    assert first == again


def test_the_classes_mark_the_supports_and_the_loads(bar):
    classes = picture.voxel_classes(michell.prepare(bar))
    assert classes.shape == (6, 6, 12)
    assert np.all(classes[:, :, 0] == picture.SUPPORT)  # the voxels on the wall at x = 0
    assert np.all(classes[:, :, -1] == picture.LOAD)  # the voxels on the loaded end
    assert np.all(classes[:, :, -3:-1] == picture.HELD)  # the skin held solid under the load
    assert np.all(classes[:, :, 1:-3] == picture.DESIGN)


def test_the_renderer_draws_a_single_voxel_as_a_square():
    classes = np.zeros((1, 1, 1), dtype=np.uint8)
    classes[0, 0, 0] = picture.HELD
    top = picture.render(classes, (0, 0, 1), (0, 1, 0), 6.0)
    assert top.rgb.shape == (6, 6, 3)
    inner = top.rgb[1:5, 1:5].reshape(-1, 3)
    assert np.all(inner == inner[0]) and np.all(inner[0] < 255)
    oblique = picture.render(classes, (1, -1, 0.8), (0, 0, 1), 6.0)
    corners = oblique.rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert np.all(corners == 255)  # the corners of the hexagon's box are background


def test_the_png_encoder_writes_a_valid_png():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data = picture.png(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and data.endswith(b"IEND\xaeB`\x82")


# ---------------------------------------------------------------------------------------------------------
# Rows 38 and 39


def test_row_38_leaves_each_size_out_and_states_its_kind():
    rows = row_38_estimate()
    assert len(rows) == 3 and all(row.number == 38 for row in rows)
    assert [row.kind for row in rows] == ["held-out measurement"] * 2 + ["property of the design"]
    assert all("fitted to four measured sizes and tested on the fifth" in row.name for row in rows[:2])
    assert all("reported with no limit" in row.note and "outside the estimated range" in row.note for row in rows[:2])
    held_out = row_38_held_out()
    model = estimate.load_model()
    assert [one["elements"] for one in held_out] == [row["elements"] for row in model["sizes"]]
    for one, row in zip(held_out, model["held_out"]):
        centre = one["estimated_peak_memory_gb"][1]
        assert centre / one["peak_memory_gb"] - 1.0 == pytest.approx(row["memory_share_off"], rel=1e-9)
    assert all(row.passed for row in rows)


def test_row_39_passes():
    rows = row_39_picture()
    assert len(rows) == 3 and all(row.number == 39 and row.passed for row in rows)
    assert row_39_classes().shape == (2, 3, 4)


@pytest.mark.parametrize("instead", [picture.DESIGN, picture.HELD])
def test_row_39_fails_without_the_support_and_the_load(instead):
    classes = row_39_classes()
    classes[(classes == picture.SUPPORT) | (classes == picture.LOAD)] = instead
    row = row_39_hue(classes)
    assert row.passed is False and row.measured == 2


def test_the_load_case_memory_runs_the_cantilever_with_more_cases():
    one = estimate.load_case_memory(8, 1, 1, 1)
    three = estimate.load_case_memory(8, 3, 2, 1)
    assert one["elements"] == three["elements"] == 16 * 8 * 8
    assert len(one["peak_after_each_pass_gb"]) == 1 and len(three["peak_after_each_pass_gb"]) == 2
    with pytest.raises(michell.errors.InvalidValue, match="cases must be 1, 2 or 3"):
        estimate.load_case_memory(8, 4, 1, 1)


def test_the_load_cases_command_stores_a_positive_measurement(tmp_path, monkeypatch, capsys):
    import michell.estimate.__main__ as command

    runs = [{"cases": c, "elements": 1000, "peak_after_each_pass_gb": [0.1 * c]} for c in (1, 3)]
    measured = {"gb_per_million_voxels": 0.09, "measured": "a test", "runs": runs}
    monkeypatch.setattr(command, "measure_extra_load_case", lambda *args: dict(measured))
    store = tmp_path / "model.json"
    store.write_text(json.dumps(estimate.load_model()), encoding="utf-8")
    assert fit_main(["load-cases", "--store", str(store)]) == 0
    assert json.loads(store.read_text(encoding="utf-8"))["extra_load_case"] == {
        "gb_per_million_voxels": 0.09,
        "measured": "a test",
    }
    measured["gb_per_million_voxels"] = -1.0
    assert fit_main(["load-cases", "--store", str(store)]) == 1
    assert "not stored" in capsys.readouterr().out
