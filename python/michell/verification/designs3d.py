"""The 3D set-ups of verification rows 16 and 17: a cantilever block and the designs solved on it.

Every set-up uses unit cube voxels. The cantilever is clamped on its face x = 0 (all three displacements of
every node there) and carries a downward (-z) load of total 1 N spread over its far face x = nx, each node
taking its share of the face's area (a half on an edge, a quarter at a corner). The designs are densities of
0 and 1 (with E_min = 1e-9 E, a contrast of 1e9) or uniform:

- **uniform**: every element at density 1;
- **lattice**: bars two elements thick along x, y and z, repeating every 8 elements, with solid end slabs two
  elements thick at x = 0 and x = nx;
- **irregular**: a fixed sum of four sine waves with unrelated wavelengths, evaluated at the element
  centres. The share s of elements where it is largest is solid, with the end slabs; of that solid only the
  group joined through faces to the held slab is kept (solid joined to the rest through E_min alone would be
  a near-mechanism, which measures the material model, not the solver). s is the smallest whole percentage
  from 40 % up for which that group reaches the loaded slab: near 40 % a field like this one is close to
  the share at which its solid first joins across the block, so a fixed 40 % joins the two slabs at some
  sizes and not at others;
- **islands**: the irregular design with four detached islands of solid, cubes of 2 x 2 x 2 elements in the
  void, joined to the rest only through E_min. Each sits in the middle of a block of 4 x 4 x 4 void elements
  none of which is within two face steps of the irregular design's solid; the blocks are taken in element
  order (x fastest) over the corners 3 <= i < nx - 7, 1 <= j < ny - 4, 1 <= k < nz - 4, each one more than 8
  elements in x, or more than 6 in y or in z, from every block taken before it;
- **load on void**: the irregular design with its load moved onto void: 1 N in -z spread equally over a
  3 x 3 patch of nodes in a plane x = const. The patch is the first, in element order over the corners
  nx / 2 <= i < nx - 3, 1 <= j < ny - 3, 1 <= k < nz - 3, for which every element of the 4 x 4 x 4 block with
  corner (i - 1, j - 1, k - 1) is void and shares no face with solid; the patch's nodes are (i + 1, j + a,
  k + b) for a and b from 0 to 2, so every element they touch lies in that block;
- **L-shaped**: a block with odd counts in every direction, a corner block removed so that the active region
  is an L, all at density 1, held on a support patch on the far face x = nx that ends one node short of the
  odd edge y = ny, loaded on the face x = 0. In a direction with an odd count the last coarse multigrid node
  lies outside the fine grid; on the face x = nx the patch leaves only the fine nodes of the edge y = ny
  free, and on a coarse level the nodes that reach only them can have degrees of freedom that depend on
  each other, so that the coarse matrix is only semi-definite. Whether that happens on the coarsest level
  depends on the size: at 15 x 9 x 11 its direct solve leaves out 12 degrees of freedom, at 35 x 25 x 27
  none (the row reports the number).

These rules were fixed before any solve was run on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from michell.analysis3d import Grid3d

E, NU, E_MIN_RATIO = 210000.0, 0.3, 1e-9

# The length scales a of the four sine waves sin(x / a) of the irregular design, in elements.
IRREGULAR_WAVES = (3.1, 2.3, 2.9, 4.3)
IRREGULAR_SOLID_SHARE = 0.4
LATTICE_PERIOD, BAR_WIDTH, SLAB_WIDTH = 8, 2, 2


@dataclass(frozen=True)
class SetUp:
    """A grid, its held degrees of freedom, its loads and a density field."""

    grid: Grid3d
    held: np.ndarray  # bool, one per degree of freedom
    loads: np.ndarray
    rho: np.ndarray
    description: str


def element_indices(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """i, j and k of every element, in element order (x fastest)."""
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return i.ravel(), j.ravel(), k.ravel()


def face_load(grid, on_face) -> np.ndarray:
    """A total load of 1 N in -z spread over the nodes of the far face x = nx for which ``on_face(j, k)``
    holds, by their share of the face area (a node on the edge of the loaded patch takes half, a corner a
    quarter). The patch must be a rectangle j0..j1, k0..k1 of nodes."""
    loads = np.zeros(grid.n_dofs)
    nodes = [(j, k) for k in range(grid.nz + 1) for j in range(grid.ny + 1) if on_face(j, k)]
    js, ks = [j for j, _ in nodes], [k for _, k in nodes]
    j_ends, k_ends = (min(js), max(js)), (min(ks), max(ks))
    for j, k in nodes:
        share = (0.5 if j in j_ends else 1.0) * (0.5 if k in k_ends else 1.0)
        loads[3 * grid.node(grid.nx, j, k) + 2] = -share
    return loads / -loads.sum()


def clamped_face(grid, on_face=lambda j, k: True) -> np.ndarray:
    """Every displacement held at the nodes of the face x = 0 for which ``on_face(j, k)`` holds."""
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k in range(grid.nz + 1):
        for j in range(grid.ny + 1):
            if on_face(j, k):
                node = grid.node(0, j, k)
                held[3 * node : 3 * node + 3] = True
    return held


def cantilever(shape: tuple[int, int, int], rho: np.ndarray, description: str) -> SetUp:
    grid = Grid3d(*shape, 1.0)
    return SetUp(grid, clamped_face(grid), face_load(grid, lambda j, k: True), rho, description)


def end_slabs(nx: int, i: np.ndarray) -> np.ndarray:
    return (i < SLAB_WIDTH) | (i >= nx - SLAB_WIDTH)


def uniform(shape: tuple[int, int, int]) -> SetUp:
    nx, ny, nz = shape
    return cantilever(shape, np.ones(nx * ny * nz), "uniform density 1")


def lattice(shape: tuple[int, int, int]) -> SetUp:
    nx, ny, nz = shape
    i, j, k = element_indices(nx, ny, nz)
    on = [(a % LATTICE_PERIOD) < BAR_WIDTH for a in (i, j, k)]
    solid = (on[1] & on[2]) | (on[0] & on[2]) | (on[0] & on[1]) | end_slabs(nx, i)
    rho = np.where(solid, 1.0, 0.0)
    return cantilever(shape, rho, f"lattice of bars, {rho.mean():.0%} solid")


def face_joined_group(solid: np.ndarray, shape: tuple[int, int, int], start: int) -> np.ndarray:
    """The solid elements joined to element ``start`` through shared faces: the group grows by its face
    neighbours that are solid until it stops growing."""
    nx, ny, nz = shape
    solid = solid.reshape(nz, ny, nx)
    group = np.zeros_like(solid)
    group.flat[start] = solid.flat[start]
    while True:
        grown = group.copy()
        grown[1:] |= group[:-1]
        grown[:-1] |= group[1:]
        grown[:, 1:] |= group[:, :-1]
        grown[:, :-1] |= group[:, 1:]
        grown[:, :, 1:] |= group[:, :, :-1]
        grown[:, :, :-1] |= group[:, :, 1:]
        grown &= solid
        if np.array_equal(grown, group):
            return group.ravel()
        group = grown


def irregular_design(shape: tuple[int, int, int]) -> tuple[np.ndarray, int, bool]:
    """The irregular density field, its solid share s in percent, and whether its solid joins the held slab
    to the loaded slab (true by construction; the rows check it)."""
    nx, ny, nz = shape
    i, j, k = element_indices(nx, ny, nz)
    x, y, z = i + 0.5, j + 0.5, k + 0.5
    a = IRREGULAR_WAVES
    field = np.sin(x / a[0] + 0.3) + np.sin(y / a[1] + x / 7.7) + np.sin(z / a[2] + 1.1) + np.sin((x + y + z) / a[3])
    order = np.argsort(-field, kind="stable")
    loaded_slab = i == nx - 1

    def held_group(percent: int) -> np.ndarray:
        solid = np.zeros(field.size, dtype=bool)
        solid[order[: round(percent / 100 * field.size)]] = True
        solid |= end_slabs(nx, i)
        return face_joined_group(solid, shape, 0)

    # The smallest share from 40 % that joins: the group only grows with the share, so bisect.
    low, high = round(100 * IRREGULAR_SOLID_SHARE), 100
    if not held_group(low)[loaded_slab].any():
        while high - low > 1:
            middle = (low + high) // 2
            if held_group(middle)[loaded_slab].any():
                high = middle
            else:
                low = middle
        low = high
    group = held_group(low)
    return np.where(group, 1.0, 0.0), low, bool(group[loaded_slab].any())


def irregular(shape: tuple[int, int, int]) -> tuple[SetUp, bool]:
    rho, percent, joined = irregular_design(shape)
    text = f"irregular, cut at {percent} %, {rho.mean():.0%} solid in one face-joined group"
    return cantilever(shape, rho, text), joined


def grown(solid: np.ndarray, steps: int) -> np.ndarray:
    """``solid`` (nz x ny x nx) grown by ``steps`` face steps."""
    out = solid.copy()
    for _ in range(steps):
        g = out.copy()
        g[1:] |= out[:-1]
        g[:-1] |= out[1:]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        g[:, :, 1:] |= out[:, :, :-1]
        g[:, :, :-1] |= out[:, :, 1:]
        out = g
    return out


def islands(shape: tuple[int, int, int]) -> tuple[SetUp, int]:
    """The irregular design with four detached solid islands, and the number of islands placed (fewer than
    four if the void has no room for them)."""
    nx, ny, nz = shape
    base, _ = irregular(shape)
    solid = base.rho.reshape(nz, ny, nx) > 0.5
    near = grown(solid, 2)
    corners: list[tuple[int, int, int]] = []
    for k in range(1, nz - 4):
        for j in range(1, ny - 4):
            for i in range(3, nx - 7):
                if near[k : k + 4, j : j + 4, i : i + 4].any():
                    continue
                if all(abs(i - a) > 8 or abs(j - b) > 6 or abs(k - c) > 6 for a, b, c in corners):
                    corners.append((i, j, k))
    rho = base.rho.reshape(nz, ny, nx).copy()
    for i, j, k in corners[:4]:
        rho[k + 1 : k + 3, j + 1 : j + 3, i + 1 : i + 3] = 1.0
    placed = min(len(corners), 4)
    text = f"{base.description}, with {placed} detached islands of 2 x 2 x 2 solid elements in the void"
    return SetUp(base.grid, base.held, base.loads, rho.ravel(), text), placed


def load_on_void(shape: tuple[int, int, int]) -> SetUp | None:
    """The irregular design with its load on a patch of void nodes; None if no patch qualifies."""
    nx, ny, nz = shape
    base, _ = irregular(shape)
    clear = ~grown(base.rho.reshape(nz, ny, nx) > 0.5, 1)
    for k in range(1, nz - 3):
        for j in range(1, ny - 3):
            for i in range(nx // 2, nx - 3):
                if clear[k - 1 : k + 3, j - 1 : j + 3, i - 1 : i + 3].all():
                    loads = np.zeros(base.grid.n_dofs)
                    for b in range(3):
                        for a in range(3):
                            loads[3 * base.grid.node(i + 1, j + a, k + b) + 2] = -1.0 / 9
                    text = f"{base.description}, loaded on a 3 x 3 patch of void nodes at x = {i + 1}"
                    return SetUp(base.grid, base.held, loads, base.rho, text)
    return None


def l_shaped(shape: tuple[int, int, int]) -> SetUp:
    """The L-shaped region: the elements with i >= ceil(nx / 2) and k >= ceil(nz / 2) are removed; held on
    the face x = nx at the nodes of the part with j <= ny - 1; loaded on the face x = 0."""
    nx, ny, nz = shape
    i, j, k = element_indices(nx, ny, nz)
    i_cut, k_cut = -(-nx // 2), -(-nz // 2)
    active = ~((i >= i_cut) & (k >= k_cut))
    grid = Grid3d(nx, ny, nz, 1.0, active=active)
    held = np.zeros(grid.n_dofs, dtype=bool)
    for k_node in range(k_cut + 1):
        for j_node in range(ny):
            node = grid.node(nx, j_node, k_node)
            held[3 * node : 3 * node + 3] = True
    loads = np.zeros(grid.n_dofs)
    for k_node in range(nz + 1):
        for j_node in range(ny + 1):
            share = (0.5 if j_node in (0, ny) else 1.0) * (0.5 if k_node in (0, nz) else 1.0)
            loads[3 * grid.node(0, j_node, k_node) + 2] = -share
    loads /= -loads.sum()
    text = (
        f"L-shaped region of {int(active.sum())} elements, held on a patch of the face x = {nx} that ends one "
        f"node short of the edge y = {ny}"
    )
    return SetUp(grid, held, loads, np.ones(nx * ny * nz), text)
