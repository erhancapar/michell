"""A picture of a prepared problem: the voxel model with its supports and loads, as one SVG file.

``write_picture(prepared, path)`` writes four views of the voxel model of a ``Prepared`` problem
(``michell.problem_file.prepare``): from the top (+z), the front (-y), the right (+x) and an oblique view
(from +x, -y and +z). The views are rasters drawn here, with NumPy only, and embedded in the SVG as PNG
images (encoded with ``zlib``); the labels, the scale bars, the legend and the arrows of the loads are SVG
text and lines on top of them.

What the views show: the design space light, the voxels held solid darker, the voxels of the supports blue and
those of the loads amber (a voxel is a support's or a load's when it is one of its region's voxels, or when one of
its corners is a node the support holds or the load acts on); voids and everything outside the part are not drawn.
The held, support and load voxels show through the design space as paler patches of their colours
(``see_through``), so regions inside the part or on a face turned away are seen. Over each view, every support's
point of action has a blue mark, and every load an amber arrow in the direction of its force, ending at its point
of action and labelled with its load case; a force along the line of sight is drawn as a circle with a dot
(towards the viewer) or a cross (away). The marks and arrows are drawn over the part, so they show where a support
or a load hidden behind it lies.

Each face of a voxel is shaded by its direction to a light above, left of and in front of the viewer, and
nearer faces are lighter; edges where the depth jumps are darkened. The same input gives the same bytes on one
installation; the PNG data are compressed by the zlib library Python is built with, and another (zlib-ng, for
one) gives other bytes for the same pixels.

``render(classes, direction, up, scale)`` is the renderer alone: it takes an array of voxel classes
(``EMPTY``, ``DESIGN``, ``HELD``, ``SUPPORT``, ``LOAD``) indexed ``[k, j, i]`` and returns the RGB raster.
"""

from __future__ import annotations

import base64
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMPTY, DESIGN, HELD, SUPPORT, LOAD = 0, 1, 2, 3, 4

# Colours: blue and amber for supports and loads (told apart by people who do not see red and green apart).
COLOURS = np.array(
    [
        (255, 255, 255),  # empty: the background
        (214, 214, 208),  # design space
        (118, 118, 124),  # held solid
        (0, 114, 178),  # support: blue
        (230, 159, 0),  # load: amber
    ],
    dtype=float,
)
SUPPORT_HEX, LOAD_HEX, TEXT_HEX = "#0072b2", "#e69f00", "#222222"

LIGHT = np.array([-0.4, 0.5, 1.0])  # towards the light, in the view's terms: (right, up, towards the viewer)
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE, DEPTH_CUE, EDGE = 0.55, 0.45, 0.25, 0.55
PANEL_PIXELS = 360  # the largest extent of a view
MOST_PIXELS_PER_VOXEL = 12.0
ARROW_PIXELS = 44.0


@dataclass(frozen=True)
class View:
    """A view: its ``name``, the unit vector ``direction`` from the model towards the viewer, and ``up``."""

    name: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]


VIEWS = (
    View("top (from +z)", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    View("front (from -y)", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    View("right (from +x)", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    View("oblique (from +x, -y, +z)", (1.0, -1.0, 0.8), (0.0, 0.0, 1.0)),
)


def _basis(direction, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    w = np.asarray(up, dtype=float)
    w = w - (w @ v) * v
    w = w / np.linalg.norm(w)
    right = np.cross(w, v)
    return v, right, w


def _frame(shape, v, right, w):
    """The screen extent (u and w, in voxel sides) of the grid's box in this view."""
    nz, ny, nx = shape
    corners = np.array([(x, y, z) for x in (0, nx) for y in (0, ny) for z in (0, nz)], dtype=float)
    u, h = corners @ right, corners @ w
    return float(u.min()), float(u.max()), float(h.min()), float(h.max())


def pixels_per_voxel(shape) -> float:
    """The scale shared by every view of a grid of ``shape`` ``(nz, ny, nx)`` voxels: its largest view fits
    ``PANEL_PIXELS``, at most ``MOST_PIXELS_PER_VOXEL``."""
    extent = 0.0
    for view in VIEWS:
        u0, u1, h0, h1 = _frame(shape, *_basis(view.direction, view.up))
        extent = max(extent, u1 - u0, h1 - h0)
    return min(MOST_PIXELS_PER_VOXEL, PANEL_PIXELS / extent)


def _exposed(classes: np.ndarray) -> np.ndarray:
    """The voxels of the part with at least one face on empty space (or on the grid's box)."""
    filled = classes != EMPTY
    padded = np.pad(filled, 1, constant_values=False)
    inner = np.ones_like(filled)
    for axis in range(3):
        for step in (-1, 1):
            inner &= np.roll(padded, step, axis=axis)[1:-1, 1:-1, 1:-1]
    return filled & ~inner


class Raster:
    """A rendered view: ``rgb`` (rows, columns, 3; uint8), ``drawn`` (rows, columns; the pixels that show a
    voxel), the scale and the frame, so that points in voxel units can be placed on it (``place``)."""

    def __init__(self, rgb, scale, frame, basis, drawn=None):
        self.rgb, self.scale, self.frame, self.basis = rgb, scale, frame, basis
        self.drawn = np.zeros(rgb.shape[:2], dtype=bool) if drawn is None else drawn

    def place(self, point) -> tuple[float, float]:
        """The pixel position (column, row) of ``point`` (x, y, z in voxel sides from the grid's corner)."""
        _, right, w = self.basis
        u0, _, _, h1 = self.frame
        p = np.asarray(point, dtype=float)
        return float((p @ right - u0) * self.scale), float((h1 - p @ w) * self.scale)

    def screen(self, vector) -> tuple[float, float, float]:
        """A direction in screen terms: (columns, rows, towards the viewer), of unit length in space."""
        v, right, w = self.basis
        d = np.asarray(vector, dtype=float)
        d = d / np.linalg.norm(d)
        return float(d @ right), float(-(d @ w)), float(d @ v)


def render(classes: np.ndarray, direction, up, scale: float) -> Raster:
    """Renders the voxels of ``classes`` (indexed ``[k, j, i]``) seen from ``direction`` with ``up`` at
    ``scale`` pixels per voxel side: each pixel shows the face of the nearest voxel its centre's line of sight
    meets."""
    classes = np.asarray(classes, dtype=np.uint8)
    v, right, w = _basis(direction, up)
    u0, u1, h0, h1 = _frame(classes.shape, v, right, w)
    columns, rows = max(1, math.ceil((u1 - u0) * scale)), max(1, math.ceil((h1 - h0) * scale))
    k, j, i = np.nonzero(_exposed(classes))
    centres = np.stack([i + 0.5, j + 0.5, k + 0.5], axis=1)
    cu, ch, cd = centres @ right, centres @ w, centres @ v
    # A cube's footprint on the screen lies within half the sum of the absolute components around its centre.
    reach_u = 0.5 * float(np.abs(right).sum()) * scale
    reach_h = 0.5 * float(np.abs(w).sum()) * scale
    col_c, row_c = (cu - u0) * scale, (h1 - ch) * scale
    pixel, depth, voxel, face = [], [], [], []
    s = np.sign(v)
    for dc in range(-math.ceil(reach_u) - 1, math.ceil(reach_u) + 2):
        for dr in range(-math.ceil(reach_h) - 1, math.ceil(reach_h) + 2):
            col = np.floor(col_c).astype(np.int64) + dc
            row = np.floor(row_c).astype(np.int64) + dr
            inside = (col >= 0) & (col < columns) & (row >= 0) & (row < rows)
            # The line of sight through the pixel's centre, as an offset from the voxel's centre.
            qu = (col + 0.5) / scale - (cu - u0)
            qh = (h1 - ch) - (row + 0.5) / scale
            q = qu[:, None] * right + qh[:, None] * w  # (voxels, 3), in voxel sides
            enter = np.full(len(q), -np.inf)
            leave = np.full(len(q), np.inf)
            which = np.zeros(len(q), dtype=np.int64)
            for axis in range(3):
                if v[axis] == 0.0:
                    inside &= np.abs(q[:, axis]) <= 0.5
                    continue
                # Along the line X = centre + q - t v, the slab |X_axis - centre_axis| <= 0.5.
                near = (q[:, axis] - 0.5 * s[axis]) / v[axis]
                far = (q[:, axis] + 0.5 * s[axis]) / v[axis]
                later = near > enter
                which = np.where(later, axis, which)
                enter = np.maximum(enter, near)
                leave = np.minimum(leave, far)
            inside &= enter <= leave
            index = np.nonzero(inside)[0]
            pixel.append(row[index] * columns + col[index])
            depth.append(-cd[index] + enter[index])  # distance from the viewer, up to a constant
            voxel.append(index)
            face.append(which[index])
    rgb = np.full((rows, columns, 3), 255, dtype=np.uint8)
    if not pixel:
        return Raster(rgb, scale, (u0, u1, h0, h1), (v, right, w))
    pixel, depth, voxel, face = (np.concatenate(a) for a in (pixel, depth, voxel, face))
    if pixel.size == 0:
        return Raster(rgb, scale, (u0, u1, h0, h1), (v, right, w))
    depth = np.round(depth, 9)  # the same order whatever the last bit of the arithmetic
    order = np.lexsort((voxel, depth, pixel))
    pixel, depth, voxel, face = pixel[order], depth[order], voxel[order], face[order]
    first = np.ones(pixel.size, dtype=bool)
    first[1:] = pixel[1:] != pixel[:-1]
    pixel, depth, voxel, face = pixel[first], depth[first], voxel[first], face[first]
    normal = np.zeros((pixel.size, 3))
    normal[np.arange(pixel.size), face] = s[face]
    light = LIGHT[0] * right + LIGHT[1] * w + LIGHT[2] * v
    shade = AMBIENT + DIFFUSE * np.clip(normal @ light, 0.0, None)
    span = float(depth.max() - depth.min())
    if span > 0:
        shade = shade * (1.0 - DEPTH_CUE * (depth - depth.min()) / span)
    colour = COLOURS[classes[k[voxel], j[voxel], i[voxel]]] * shade[:, None]
    image = np.full((rows * columns, 3), 255.0)
    image[pixel] = colour
    # The outlines: a drawn pixel whose neighbour (across a side) is not drawn, or lies more than a voxel side
    # deeper or nearer, is darkened.
    drawn = np.zeros(rows * columns, dtype=bool)
    drawn[pixel] = True
    drawn = drawn.reshape(rows, columns)
    far = np.zeros(rows * columns)
    far[pixel] = depth
    far = np.pad(far.reshape(rows, columns), 1)
    seen = np.pad(drawn, 1)
    edge = np.zeros((rows, columns), dtype=bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other = far[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + columns]
        other_seen = seen[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + columns]
        edge |= drawn & (~other_seen | (np.abs(other - far[1:-1, 1:-1]) > 1.0))
    image = image.reshape(rows, columns, 3)
    image[edge] *= EDGE
    rgb = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return Raster(rgb, scale, (u0, u1, h0, h1), (v, right, w), drawn)


def see_through(classes: np.ndarray, direction, up, scale: float) -> Raster:
    """``render`` of ``classes`` with what the design space hides shown through it: the voxels held solid and
    those of the supports and loads are drawn alone as well, and wherever they show, the two rasters are mixed
    half and half, so that a region inside the part or behind it shows as a paler patch of its colour."""
    whole = render(classes, direction, up, scale)
    inner = np.where(classes == DESIGN, EMPTY, classes).astype(np.uint8)
    if not np.any(inner):
        return whole
    behind = render(inner, direction, up, scale)
    mixed = np.round(0.5 * whole.rgb.astype(float) + 0.5 * behind.rgb.astype(float)).astype(np.uint8)
    rgb = np.where(behind.drawn[:, :, None], mixed, whole.rgb)
    return Raster(rgb, whole.scale, whole.frame, whole.basis, whole.drawn | behind.drawn)


def png(rgb: np.ndarray) -> bytes:
    """``rgb`` (rows, columns, 3; uint8) as a PNG file: 8-bit RGB, no filter, compressed with ``zlib``."""
    rows, columns, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(rows))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", columns, rows, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def voxel_classes(prepared) -> np.ndarray:
    """The class of every voxel of a ``Prepared`` problem, indexed ``[k, j, i]``: ``DESIGN`` for an active
    voxel, ``HELD`` for one held solid, ``SUPPORT`` and ``LOAD`` for a voxel of a support or a load (a load
    over a support where they meet), ``EMPTY`` elsewhere."""
    model, setup = prepared.model, prepared.set_up
    nx, ny, nz = model.counts
    active = np.asarray(model.active, dtype=bool).reshape(nz, ny, nx)
    classes = np.where(active, DESIGN, EMPTY).astype(np.uint8)
    classes[np.asarray(setup.keep_solid.mask, dtype=bool).reshape(nz, ny, nx) & active] = HELD
    for role, value in (("support", SUPPORT), ("load", LOAD)):
        marked = np.zeros(nz * ny * nx, dtype=bool)
        corners = np.zeros((nz + 1) * (ny + 1) * (nx + 1), dtype=bool)
        for item in setup.report:
            if item.role != role:
                continue
            if item.voxels is not None:
                marked[np.asarray(item.voxels, dtype=np.int64)] = True
            else:
                corners[np.asarray(item.nodes, dtype=np.int64)] = True
        corners = corners.reshape(nz + 1, ny + 1, nx + 1)
        touched = np.zeros((nz, ny, nx), dtype=bool)
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    touched |= corners[dk : dk + nz, dj : dj + ny, di : di + nx]
        classes[(marked.reshape(nz, ny, nx) | touched) & active] = value
    return classes


def _number(value: float) -> str:
    return f"{value:.1f}"


def _round_length(target: float) -> float:
    """The largest of 1, 2 and 5 times a power of ten that is at most ``target`` (at least the smallest)."""
    power = 10.0 ** math.floor(math.log10(target))
    for factor in (5.0, 2.0, 1.0):
        if factor * power <= target:
            return factor * power
    return power


def _length_words(mm: float) -> str:
    return f"{mm:g} mm"


def _arrow(x0, y0, x1, y1, colour) -> list[str]:
    """An arrow from (x0, y0) to its head at (x1, y1)."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    size = 9.0
    left = (x1 - size * ux + 0.5 * size * uy, y1 - size * uy - 0.5 * size * ux)
    right = (x1 - size * ux - 0.5 * size * uy, y1 - size * uy + 0.5 * size * ux)
    return [
        f'<line x1="{_number(x0)}" y1="{_number(y0)}" x2="{_number(x1 - 0.8 * size * ux)}" '
        f'y2="{_number(y1 - 0.8 * size * uy)}" stroke="{colour}" stroke-width="2.5"/>',
        f'<polygon points="{_number(x1)},{_number(y1)} {_number(left[0])},{_number(left[1])} '
        f'{_number(right[0])},{_number(right[1])}" fill="{colour}"/>',
    ]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def picture_svg(prepared) -> str:
    """The picture of a ``Prepared`` problem as SVG text (see the module's description)."""
    model, setup = prepared.model, prepared.set_up
    voxel_mm = float(model.voxel_mm)
    origin = np.asarray(model.origin_mm, dtype=float)
    classes = voxel_classes(prepared)
    scale = pixels_per_voxel(classes.shape)
    rasters = [see_through(classes, view.direction, view.up, scale) for view in VIEWS]
    margin, title, pad, gap = 16, 22, int(ARROW_PIXELS) + 16, 30
    width_cell = max(r.rgb.shape[1] for r in rasters) + 2 * pad + 2 * margin
    height_cell = max(r.rgb.shape[0] for r in rasters) + title + 2 * pad + gap + margin
    legend_height = 110
    width, height = 2 * width_cell, 2 * height_cell + legend_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for index, (view, raster) in enumerate(zip(VIEWS, rasters)):
        x0 = (index % 2) * width_cell + margin
        y0 = (index // 2) * height_cell + margin
        rows, columns, _ = raster.rgb.shape
        image_x, image_y = x0 + pad, y0 + title + pad
        data = base64.b64encode(png(raster.rgb)).decode("ascii")
        parts.append(f'<text x="{x0}" y="{y0 + 14}" fill="{TEXT_HEX}" font-weight="bold">{_escape(view.name)}</text>')
        parts.append(
            f'<image x="{image_x}" y="{image_y}" width="{columns}" height="{rows}" image-rendering="pixelated" '
            f'href="data:image/png;base64,{data}"/>'
        )
        for item in setup.report:
            p = (np.asarray(item.point_mm, dtype=float) - origin) / voxel_mm
            cx, cy = raster.place(p)
            cx, cy = cx + image_x, cy + image_y
            if item.role == "support":
                parts.append(
                    f'<rect x="{_number(cx - 5)}" y="{_number(cy - 5)}" width="10" height="10" fill="none" '
                    f'stroke="{SUPPORT_HEX}" stroke-width="2.5"/>'
                )
                continue
            du, dv, towards = raster.screen(item.force_n) if any(item.force_n) else (0.0, 0.0, 0.0)
            flat = math.hypot(du, dv)
            if flat > 0.2:
                tail = (cx - ARROW_PIXELS * du / flat, cy - ARROW_PIXELS * dv / flat)
                parts.extend(_arrow(tail[0], tail[1], cx, cy, LOAD_HEX))
                label_x, label_y = tail[0] - 4 * du / flat, tail[1] - 4 * dv / flat
            else:
                parts.append(
                    f'<circle cx="{_number(cx)}" cy="{_number(cy)}" r="8" fill="none" stroke="{LOAD_HEX}" '
                    f'stroke-width="2.5"/>'
                )
                if towards > 0:
                    parts.append(f'<circle cx="{_number(cx)}" cy="{_number(cy)}" r="2.5" fill="{LOAD_HEX}"/>')
                else:
                    for a, b in ((-5, -5), (-5, 5)):
                        parts.append(
                            f'<line x1="{_number(cx + a)}" y1="{_number(cy + b)}" x2="{_number(cx - a)}" '
                            f'y2="{_number(cy - b)}" stroke="{LOAD_HEX}" stroke-width="2"/>'
                        )
                label_x, label_y = cx + 14, cy - 12
            parts.append(
                f'<text x="{_number(label_x)}" y="{_number(label_y)}" fill="{TEXT_HEX}" font-size="12" '
                f'text-anchor="middle">{_escape(str(item.case))}</text>'
            )
        # The scale bar, under the view.
        bar_mm = _round_length(0.3 * columns / scale * voxel_mm)
        bar = bar_mm / voxel_mm * scale
        bar_y = image_y + rows + pad + 8
        parts.append(
            f'<line x1="{x0}" y1="{_number(bar_y)}" x2="{_number(x0 + bar)}" y2="{_number(bar_y)}" '
            f'stroke="{TEXT_HEX}" stroke-width="2"/>'
        )
        for x in (x0, x0 + bar):
            parts.append(
                f'<line x1="{_number(x)}" y1="{_number(bar_y - 4)}" x2="{_number(x)}" y2="{_number(bar_y + 4)}" '
                f'stroke="{TEXT_HEX}" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{_number(x0 + bar + 8)}" y="{_number(bar_y + 4)}" fill="{TEXT_HEX}">'
            f"{_length_words(bar_mm)}</text>"
        )
        # The axes of the view, beside the scale bar.
        ax, ay = image_x + columns + pad - 50, bar_y + 4
        for name, axis in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            du, dv, _ = raster.screen(axis)
            if math.hypot(du, dv) < 0.2:
                continue
            parts.extend(_arrow(ax, ay, ax + 22 * du, ay + 22 * dv, TEXT_HEX))
            parts.append(
                f'<text x="{_number(ax + 30 * du)}" y="{_number(ay + 30 * dv + 4)}" fill="{TEXT_HEX}" '
                f'font-size="11" text-anchor="middle">{name}</text>'
            )
    # The legend.
    ly = 2 * height_cell + 18
    entries = [
        ("swatch", COLOURS[DESIGN], "design space"),
        ("swatch", COLOURS[HELD], "held solid"),
        ("swatch", COLOURS[SUPPORT], "support (square: its point of action)"),
        ("swatch", COLOURS[LOAD], "load (arrow: its force, named by load case)"),
    ]
    x = margin
    for index, (_, colour, words) in enumerate(entries):
        lx = margin + (index % 2) * width_cell
        y = ly + (index // 2) * 22
        hexa = "#" + "".join(f"{int(c):02x}" for c in colour)
        parts.append(f'<rect x="{lx}" y="{y - 11}" width="14" height="14" fill="{hexa}" stroke="{TEXT_HEX}"/>')
        parts.append(f'<text x="{lx + 22}" y="{y + 1}" fill="{TEXT_HEX}">{_escape(words)}</text>')
    nx, ny, nz = model.counts
    notes = (
        f"{prepared.problem_file.path.name}: {nx} x {ny} x {nz} voxels of {voxel_mm:g} mm.",
        "Held solid, supports and loads show through the design space, paler; marks and arrows are drawn over "
        "the part.",
        "Voids and the space outside the part are not drawn.",
    )
    for index, words in enumerate(notes):
        parts.append(
            f'<text x="{x}" y="{ly + 48 + 18 * index}" fill="{TEXT_HEX}" font-size="12">{_escape(words)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_picture(prepared, path: str | os.PathLike) -> Path:
    """Writes ``picture_svg(prepared)`` to ``path`` and returns the path."""
    path = Path(path)
    path.write_bytes(picture_svg(prepared).encode("utf-8"))  # UTF-8 and newlines as written, whatever the system
    return path
