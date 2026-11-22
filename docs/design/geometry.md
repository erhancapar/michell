# Geometry: from closed meshes to the voxel model

Status: design note, kept in step with the code. Closed triangle meshes (`michell.mesh`), the voxeliser
(the engine's `voxel_occupancy`, in Python `michell.regions.occupancy`), the voxel model of a part
(`michell.regions.voxel_model`) and supports and loads on that model (`michell.attach`) exist. Reading
STEP and the problem file are planned (see the [roadmap](../roadmap.md)).

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

## Supports and loads

`michell.attach.set_up(model, material, supports, cases, size_mm, threads=1, outside=None)` turns the voxel
model, the supports (`Support(on, fix, skin)`) and the load cases (`Case(name, loads, weight)`, each load a
`Load(on, force_n, skin)`) into the `Problem`, the voxels held solid (`KeepSolid`) and the density
filter's faces that `stiff_part` takes, with a report per support and per load. A support or a load names
what it acts `on`, and that decides how it is applied. No rule has a setting to tune.

1. **A solid region of the part**, by its name in the model. A load is a uniform body force over the
   region's voxels (the active voxels the region fills by majority): each voxel carries force / voxels,
   one eighth of it at each of its eight nodes. A support holds every node of those voxels, in the
   directions it fixes. This is the way to load or hold a volume.
2. **Any other closed body** (a `Mesh`: a marker drawn where a clamp, a pin or a pad touches the part), or
   **a box** given by its low and high corners in the meshes' coordinates. A load is a uniform traction over
   the part's boundary faces in contact with the body: each face carries force / faces, one quarter of it
   at each of its four corners (on a patch of a face of the grid this is `michell.select.load`, entry for
   entry). A support holds the nodes of those faces. A boundary face lies between an active voxel and one
   that is not active, or the grid's edge. It is **in contact** when, seen along the face's normal, the body
   covers more than half of the face within a **window** two voxel sides deep centred on the face: the
   boundary voxel and the position across it. This is measured on 16 rays along the normal through the
   face, at 1/8, 3/8, 5/8 and 7/8 of its sides (an even count, so that no ray lies on a mid-plane of the
   face): a ray counts when the body's length on it inside the window is at least one twentieth of a voxel
   side, and the face is in contact when more than 8 of its 16 rays count. The lengths come from the
   voxeliser, which is exact along its rays: the body is handed to it with its coordinates turned so that
   the normal becomes x, on a grid four times finer than the model's with one ray per fine column, over the
   box around the body only and in pieces of at most 4 194 304 fine voxels (32 MiB).

   Why this is right: a voxel layer is active when the part fills at least half of it, so a body beyond the
   part's true face covers at most half of that layer's side faces as seen along their normals (exactly
   half in the tie, which "more than half" leaves out), while it covers the end face fully. The window
   reaches into the boundary voxel, so a thin marker lying inside the last layer, where the part's true
   face lies, is still seen; and the length along the normal is measured alike in x, y and z.

           part's true face
                 :
       ----------+--:-------       a clamp drawn larger than the face, beyond the true face (:)
       part      |  :  ######      inside the last voxel layer; window of the end face: |<-->|
       (active   |  :  ######      seen along x, the clamp covers all of it: in contact
       layer) ---+--:--######      window of the side face of the last layer, seen along y:
                 |  :  ######      the clamp covers the half beyond (:), not more: not in contact
       ----------+--:-------

3. A body that touches no boundary face is refused, with the most rays it covers on any face and its
   volume, and the advice that a marker should touch the part or overlap it slightly, and to name a solid
   region when a volume is meant.
4. **Shares on held degrees of freedom.** A load's shares on degrees of freedom that a support holds act on
   the support directly: they are set to zero, not spread over the other nodes, which would add force the
   part never sees. Their sum is reported per load (`dropped_n`, and `dropped_share`, its length over the
   force's); `set_up` warns (`LoadOnSupport`) above 1 % and refuses the load above one half. A region load
   on a bar held at one end, for example, puts 1/80 of the force on the held end nodes of a bar 40 voxels
   long. Before the shares are dropped, the shares of every load sum to its force exactly (`math.fsum`; a
   residue left by rounding goes to the largest entry).
5. **The held skin.** With free filter faces the filtered field cannot pass about 0.6 at a face, and
   `stiff_part` projects its eroded design at 0.75, so a patch on design voxels would load or hold void.
   Where a contact face lies on a design voxel, the active voxels whose centre lies within half the member
   size (at least two voxel sides) of the face's centre are held solid; a centre at exactly that distance
   is inside, so behind a flat patch a member size of 5 voxels holds three layers. Nothing is added, and
   nothing is refused, for contact faces on voxels the model already holds solid or on a face of the grid
   that is closed for the filter (rule 6, after the caller's overrides). With `skin=False` on a support or
   a load, a contact face on a design voxel that is neither held solid nor on a closed face is refused.
   Region supports and loads need no skin. The report gives the skin's voxels and mass per support and load
   (the voxels the model did not hold already).
6. **Filter faces.** Every boundary of the part is a free surface for the density filter: `outside` is
   `"empty"` on every face of the grid and `inactive` is `"empty"`; except that a face of the grid's box
   whose whole active area (every boundary face of an active voxel on it) is in contact with one support is
   `"closed"`: the part continues into what holds it. A face held only in part, or by two supports that
   share it, stays free and relies on the skin. `outside=` overrides any face. A region support closes no
   face.

**Mirror planes.** The checks that supports and load cases are mirror images of each other belong to the
symmetry rule and are not part of `set_up`. Nothing in it stands in their way: a box is turned into a
`Mesh` and goes through the same contact rule as any body, and the contact faces of a body (`Faces`, by
voxel and side, with the ray count of every boundary face in `Contact.rays`) are kept in the report, so the
faces of one body can be compared with the mirror image of another's.

**The cases that decide rule 2**, measured on 1 mm voxels (tests in `tests/test_attach.py`):

- A bar of 10 x 10 mm whose end face lies on a grid plane, half-way, a quarter or three quarters through
  the last voxel layer (a box cut by a void, or a box of 40.5 mm on its grid centred on it): a slab 1 mm
  thick, or a marker a quarter of a voxel thick lying inside the layer, flush with the face's sides or
  overhanging them by three voxels on every side, catches the 100 end faces and nothing else in all 16
  combinations. Where the part fills the last layer to three quarters, the overhanging slab covers 4 of
  the 16 rays of that layer's side faces; in the tie at one half, 8.
- A pin in a bore of radius 3 mm on 32 sides (axis along z, through a block of 40 x 20 x 10 voxels), with a
  clearance or an interference of a hundredth of a voxel in radius, catches all 240 faces of the bore and
  no other face; every bore face has 12 or 16 of its rays covered. A pin of half the bore's diameter covers
  no ray of any face and is refused.
- A marker of L-shaped section wrapped around an edge from outside catches the faces of both sides of the
  edge. A box that overlaps the part deeply also covers, seen along their normals, the faces it passes
  under inside the part (a box through the bar's whole depth catches its top and bottom faces there):
  markers are meant to touch the part or overlap it slightly.
- A slab a twentieth of a voxel thick is seen across x, y and z alike; one a twenty-fifth thick is not.

Verification row 32 checks a bar pulled by a face load with rollers on three faces (F L / (E A)) and a bar
under a region load with Poisson's ratio 0 (F L / (2 E A)), the same bar with its end face half-way and a
quarter through a voxel layer pulled through an overhanging slab (only end faces loaded; the extension
against the true length, off by the voxel model's error in length, at most half a voxel), the force sums,
a bore loaded through a pin with
the held skin in a short run of `stiff_part`, and the example bridge set up from regions and marker bodies,
which gives the example's arrays entry for entry.
