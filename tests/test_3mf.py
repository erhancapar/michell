"""michell.mesh: 3MF files written and read back, and the files ``read_3mf`` refuses."""

import zipfile

import numpy as np
import pytest

from michell import mesh
from michell.errors import InvalidType, InvalidValue
from michell.mesh import UNITS_3MF, Mesh, read_3mf, write_3mf

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>'
)
RELS = (
    '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
    'relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
)
TETRAHEDRON = (
    '<vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>'
    '<vertex x="0" y="0" z="1"/></vertices><triangles><triangle v1="0" v2="2" v3="1"/>'
    '<triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles>'
)


def model(body=TETRAHEDRON, *, unit='unit="millimeter"', object_attributes='id="1" type="model"', extra="",
          item='<item objectid="1"/>'):  # fmt: skip
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><model {unit} xmlns="{CORE}"><resources>'
        f"<object {object_attributes}><mesh>{body}</mesh></object>{extra}</resources><build>{item}</build></model>"
    )


def package(path, parts):
    with zipfile.ZipFile(path, "w") as out:
        for name, text in parts.items():
            out.writestr(name, text)
    return path


def a_package(path, model_text=None, **more):
    parts = {"[Content_Types].xml": CONTENT_TYPES, "_rels/.rels": RELS, "3D/3dmodel.model": model_text or model()}
    parts.update(more)
    return package(path, parts)


@pytest.mark.parametrize(
    "made",
    [
        lambda: mesh.torus((0.1, -0.2, 3.0), 12.0, 4.0),
        lambda: mesh.sphere((1.0 / 3.0, 2e-7, -5e12), 7.0, 16, 8),
        lambda: mesh.box((-1e-7, 0.1, 1.0 / 3.0), (1.0 + 2.0**-40, 1e5 / 7.0, 2.0)),
    ],
    ids=["torus", "sphere far away", "box of awkward numbers"],
)
def test_a_3mf_file_reads_back_bit_for_bit(tmp_path, made):
    body = made()
    write_3mf(tmp_path / "body.3mf", body)
    back = read_3mf(tmp_path / "body.3mf")
    assert np.array_equal(back.vertices, body.vertices) and np.array_equal(back.triangles, body.triangles)
    with zipfile.ZipFile(tmp_path / "body.3mf") as written:
        assert written.namelist() == ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in written.infolist())
        assert written.testzip() is None


def test_the_same_mesh_gives_the_same_bytes(tmp_path):
    body = mesh.sphere((0.0, 0.0, 0.0), 2.0)
    write_3mf(tmp_path / "a.3mf", body)
    write_3mf(tmp_path / "b.3mf", body)
    assert (tmp_path / "a.3mf").read_bytes() == (tmp_path / "b.3mf").read_bytes()


@pytest.mark.parametrize("unit", sorted(UNITS_3MF))
def test_other_units_are_converted_both_ways(tmp_path, unit):
    body = mesh.cylinder((1.0, 2.0, 3.0), 2.5, 7.0)
    write_3mf(tmp_path / "c.3mf", body, unit=unit)
    text = zipfile.ZipFile(tmp_path / "c.3mf").read("3D/3dmodel.model").decode()
    assert f'unit="{unit}"' in text
    back = read_3mf(tmp_path / "c.3mf")
    np.testing.assert_allclose(back.vertices, body.vertices, rtol=1e-15, atol=1e-15)
    assert np.array_equal(back.triangles, body.triangles)


def test_only_closed_meshes_are_written_and_they_face_outward(tmp_path):
    box = mesh.box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    with pytest.raises(InvalidValue, match="not closed"):
        write_3mf(tmp_path / "open.3mf", Mesh(box.vertices, box.triangles[1:]))
    assert not (tmp_path / "open.3mf").exists()  # refused before the file is written
    inward = Mesh(box.vertices, box.triangles[:, [0, 2, 1]])
    write_3mf(tmp_path / "turned.3mf", inward)
    assert read_3mf(tmp_path / "turned.3mf").volume_mm3 == 6.0
    with pytest.raises(InvalidValue, match="unit"):
        write_3mf(tmp_path / "x.3mf", box, unit="yard")
    with pytest.raises(InvalidType):
        write_3mf(tmp_path / "x.3mf", np.zeros((3, 3)))


def test_a_hand_written_package_is_read(tmp_path):
    tetrahedron = read_3mf(a_package(tmp_path / "t.3mf", model(unit='unit="centimeter"')))
    assert tetrahedron.closed().volume_mm3 == pytest.approx(1000.0 / 6.0)
    # The unit may be left out (millimetres), and the model part named otherwise.
    rels = RELS.replace("/3D/3dmodel.model", "/3D/part.model")
    named = package(tmp_path / "n.3mf", {"[Content_Types].xml": CONTENT_TYPES, "_rels/.rels": rels,
                                         "3D/part.model": model(unit="")})  # fmt: skip
    assert read_3mf(named).bounds.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize(
    "parts, message",
    [
        ({"3D/Thumbnail/a.png": "png"}, "parts other than the model"),
        ({"3D/3dmodel.model": model(extra='<object id="2" type="model"><mesh></mesh></object>')}, "one object"),
        ({"3D/3dmodel.model": model(extra='<basematerials id="5"/>')}, "one object"),
        ({"3D/3dmodel.model": model(object_attributes='id="1" type="support"')}, "type 'support'"),
        ({"3D/3dmodel.model": model(object_attributes='id="1" type="model" pid="3"')}, "pid"),
        ({"3D/3dmodel.model": model(item='<item objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/>')}, "transform"),
        ({"3D/3dmodel.model": model(item='<item objectid="1"/><item objectid="1"/>')}, "one item"),
        ({"3D/3dmodel.model": model(item='<item objectid="7"/>')}, "one item"),
        ({"3D/3dmodel.model": model(unit='unit="yard"')}, "unknown unit"),
        ({"3D/3dmodel.model": model(unit='unit="millimeter" requiredextensions="p"')}, "requiredextensions"),
        ({"3D/3dmodel.model": model(TETRAHEDRON.replace('v3="3"/>', 'v3="3" p1="0"/>', 1))}, "attributes"),
        ({"3D/3dmodel.model": model(TETRAHEDRON.replace('x="1"', 'x="one"'))}, "not numbers"),
        ({"3D/3dmodel.model": model(TETRAHEDRON.replace("<vertices>", "<vertices><metadata/>"))}, "only vertex"),
        ({"3D/3dmodel.model": model("<components/>")}, "vertices and then triangles"),
        ({"3D/3dmodel.model": model().replace("<build>", "<metadata/><build>")}, "resources and then build"),
        ({"3D/3dmodel.model": model().replace(CORE, "urn:other")}, "core namespace"),
        ({"3D/3dmodel.model": "<model"}, "not well-formed"),
        ({"_rels/.rels": RELS.replace("/3D/3dmodel.model", "/3D/missing.model")}, "is missing"),
        ({"_rels/.rels": RELS.replace("2013/01/3dmodel", "2013/01/thumbnail")}, "exactly one relationship"),
        ({"[Content_Types].xml": "<Other/>"}, "content types"),
    ],
    ids=["thumbnail", "two objects", "materials", "support object", "property", "transform", "two items",
         "wrong object", "unknown unit", "extension", "triangle property", "word for a number", "metadata in the "
         "vertices", "components", "metadata in the model", "other namespace", "broken XML", "missing model",
         "no model relationship", "wrong content types"],  # fmt: skip
)
def test_anything_beyond_one_mesh_is_refused(tmp_path, parts, message):
    path = a_package(tmp_path / "x.3mf", **parts)
    with pytest.raises(InvalidValue, match=message) as refused:
        read_3mf(path)
    assert refused.value.code == "invalid_input" and str(path) in refused.value.message


def test_files_that_are_not_packages_are_refused(tmp_path):
    (tmp_path / "plain.3mf").write_text("solid")
    with pytest.raises(InvalidValue, match="not a zip package"):
        read_3mf(tmp_path / "plain.3mf")
    with pytest.raises(InvalidValue, match="lacks"):
        read_3mf(package(tmp_path / "bare.3mf", {"3D/3dmodel.model": model()}))
    with pytest.raises(FileNotFoundError):
        read_3mf(tmp_path / "absent.3mf")
