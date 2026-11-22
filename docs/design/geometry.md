# Geometry: from closed meshes to the voxel model

Status: design note, kept in step with the code. Closed triangle meshes (`michell.mesh`), the voxeliser
(the engine's `voxel_occupancy`, in Python `michell.regions.occupancy`) and the voxel model of a part
(`michell.regions.voxel_model`) exist. Reading STEP, the problem file and supports and loads on the
model's faces are planned (see the [roadmap](../roadmap.md)).

## Meshes

A region is a closed triangle mesh in mm, `michell.Mesh(vertices, triangles)`, read from an STL file
(`read_stl`, binary or ASCII, with equal corners joined into one vertex) or made from its dimensions
(`box`, `sphere`, `cylinder`, `torus`, each with the exact volume of its facets in its docstring).
`Mesh.closed()` accepts a mesh only when every edge lies on exactly two triangles that run along it in
opposite directions and no triangle is without area; it refuses it otherwise with the counts of each fault.
Nothing is repaired, except that a closed mesh whose volume is negative (all its triangles facing inward)
is turned inside out.

## The voxeliser

The occupancy of a voxel is the share of it that lies inside a body. The voxeliser computes it for every
voxel of a grid at once, for a union of bodies minus another union.

- **Rays along x.** Every voxel column (j, k) is crossed by r x r rays along x (r = 3 by default), at the
  centres of the column's r x r sub-squares. On each ray, each body's inside set is a list of x-intervals;
  the intervals of the adding bodies are joined, those of the subtracting bodies joined and cut out. A
  voxel's share on a ray is the exact length of that set inside the voxel over the voxel side; its occupancy
  is the mean over its column's r^2 rays. So the occupancy is exact in x and sampled at r x r points in y
  and z: a box with its y and z faces on voxel planes comes out exact wherever its x faces lie.
- **Which triangles a ray crosses** is decided exactly. The y and z coordinates of every vertex are put on
  a lattice of h / (2 r 2^20) (h the voxel side), on which every ray point is a lattice point; rounding moves
  a vertex in y and z by less than 1e-7 voxel sides, and the test of a point against a triangle is then
  exact integer arithmetic. **Where** a ray crosses a triangle is computed from the vertices' own
  coordinates.
- **Rays through edges and vertices.** A ray that passes exactly through an edge or a vertex shared by
  several triangles counts that crossing once. The rule is a symbolic perturbation: every ray is treated as
  if moved by an infinitely small step towards +y, and a smaller one still towards +z. Such a ray passes
  through no edge and no vertex, so it crosses exactly one of two triangles that share an edge from both
  sides, and exactly one of a fan of triangles around a vertex; and since every triangle is tested on its
  own against the same moved ray, with exact numbers, the triangles cannot disagree. A point on a boundary
  between two sides thus belongs to the side of larger y (along a boundary that runs along y, to the side of
  larger z). Triangles parallel to the rays are never crossed and are ignored.
- **Inside** is where the winding number of the body along the ray is positive: +1 for each triangle the
  ray enters through, -1 for each it leaves through. A ray whose winding number is not 0 beyond the body has
  found it open, and the body is refused.
- The rule is proved by tests in `crates/michell/src/voxel_rays.rs` and `voxelise.rs`: a box whose edges,
  corners and face diagonals lie on rays, a pyramid with its apex on a ray, an octahedron with both tips on
  one ray, a box turned by 45 degrees with an edge in a plane of rays, and a comparison of the rule with a
  ray moved off every edge by a finite step. Verification row 30 measures the voxeliser against exact
  volumes, its convergence as the voxels are halved, and its thread counts.
- **Threads and order.** Each voxel column is computed on its own from the inputs, so the result is bit for
  bit the same on any thread count; it does not depend on the order of the triangles or of their corners
  either. A ray looks only at the triangles whose projected bounding box touches its column (a bucket per
  column).

## The voxel model

`voxel_model(design, solid=None, void=None, *, voxel_mm, mirror=(), threads=1)` builds the model an
optimisation runs on from a design region (where material may be), solid regions (where it must be) and
void regions (where it must not be), each a closed mesh.

- **The grid** covers the box around the design and solid regions: ceil(size / voxel_mm - 1e-9) voxels along
  each axis, centred on that box. `origin_mm` is the grid's corner in the meshes' coordinates; `to_grid`
  and `to_mesh` move points between the two, so that `michell.select` (which puts the grid's corner at the
  origin) works on the model.
- **The classes come from the union of the regions, not from each region alone.** The part is (design or
  solid) minus void. A voxel is **active** when the part fills at least half of it. An active voxel is
  **held solid** when the solid regions (minus void) fill at least half of what the part fills of it; else
  it is a **design** voxel. A tie at exactly one half counts as inside. So a solid region nested inside the
  design region overrides it, and two regions that meet inside a voxel lose no layer of voxels between them.
  `keep_solid` is the held class as the mask `KeepSolid` takes.
- **The volume table** has a line for the part, the held-solid class and each region: the exact volume of
  the mesh where one mesh defines the line, the sampled volume (the sum of the occupancy times the voxel
  volume), the volume of the class's voxels (for a region, of the voxels it fills by majority on its own),
  and their relative differences. When the active voxels' volume differs from the part's sampled volume by
  more than 5 %, `voxel_model` warns with `VoxelVolumeDiffers`. A region that fills no voxel by majority on
  its own is refused, with its name, its smallest extent and the voxel size: it is never dropped silently.
- **Bodies.** The model counts the bodies of the active set joined through voxel faces, with their sizes
  when there is more than one (the 3D analysis refuses a body that is not held).
- **Mirror.** With `mirror=("y",)` (any of x, y, z; planes through the centre of the grid), each region's
  occupancy is first compared with its mirror image. The volume it has without a mirror image (half the
  sum of |f - M f| times the voxel volume) must be at most 0.5 % of its sampled volume, or the region must
  match the mirror image of another region of the same role as closely (two regions that mirror each other).
  A region that passes neither is refused, naming it, the plane and the volume that differs. Then each
  occupancy field is averaged with its mirror image (a region with a partner with its partner's), so the
  classes are mirror images of themselves exactly, whatever the rays decided on the plane itself.
  Verification row 31 checks the classes, the meeting of two regions, the mirror and the refusals.
