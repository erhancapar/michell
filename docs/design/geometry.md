# Geometry: from closed meshes to the voxel model

Status: design note, kept in step with the code. Closed triangle meshes (`michell.mesh`), the voxeliser
(the engine's `voxel_occupancy`, in Python `michell.regions.occupancy`), the voxel model of a part
(`michell.regions.voxel_model`), supports and loads on that model (`michell.attach`), and the tools that turn
a field sampled on a lattice into a smooth closed surface (`michell.surface`: the iso-surface, refining and
smoothing; the measures of a mesh and 3MF files in `michell.mesh`), and the step from an optimised design
to the part's closed mesh with exact interfaces (`michell.part_mesh`, below) exist. Reading STEP is planned
(see the [roadmap](../roadmap.md)).

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

## From a field to a closed surface

`michell.surface` turns a scalar field sampled on a regular lattice into a closed triangle mesh, and smooths
it; `michell.mesh` measures meshes and writes and reads 3MF files. None of it knows about an optimisation:
which field to mesh, at which level, and how the surface meets the exact regions belong to the step that
builds a part's mesh from a result.

- **The field.** `ScalarField(values, spacing_mm, origin_mm)` holds samples on a lattice of points
  `spacing_mm` apart: `values[k, j, i]` is the sample at `origin_mm + (i, j, k) * spacing_mm`, so x runs
  fastest in memory, as in `Grid3d`. The cells are the cubes between neighbouring points.

### The iso-surface

`iso_surface(field, level, threads=1)` returns the surface where the field equals `level`, with the side of
larger values inside, as a `Mesh` facing outward. The loop over cells is the engine's (`iso_surface.rs`,
`tetrahedra.rs`).

- **Method: marching tetrahedra** (Doi and Koide 1991) on one split of every cell into six tetrahedra, the
  split of Kuhn (1960) and Freudenthal (1942): one tetrahedron per order of the three axes, running from the
  cell's lowest corner one step along each axis in that order to its highest corner, so that all six share
  the body diagonal from the lowest to the highest corner. Each face of a cell is then cut along the diagonal
  through its lowest and highest corners, and so is the same face seen from the neighbouring cell: the
  tetrahedra of the whole lattice meet face to face. Every edge of a tetrahedron runs from a lattice point by
  one of seven offsets: along x, y or z, along a face diagonal (xy, xz, yz) or along the body diagonal (xyz).
- **The level rule.** A sample is inside when it is at least the level (f >= L), outside when it is below
  it; there is no third case. A vertex lies on every edge of the tetrahedra whose two ends are on different
  sides, at the share t = (f_in - L) / (f_in - f_out) of the edge from its inside end (linear
  interpolation), with t kept at least a thousandth of the edge from both ends (`END_GAP` = 1e-3: t is
  raised to 0.001 or lowered to 0.999 where it falls outside). A sample exactly at the level is therefore
  inside, and the surface passes a thousandth of an edge beyond it: a layer of samples at the level gives a
  flat surface just outside that layer, and a sheet one point thick at the level gives a slab two
  thousandths of an edge thick. The end gap keeps every vertex off the lattice points, so vertices on
  different edges never coincide: two vertices near one point lie on edges from that point whose directions
  differ by at least 35 degrees, and are at least about half of END_GAP times the spacing apart. A triangle
  whose vertices crowd towards one point keeps the shape of the tetrahedron's corner, so twice its area over
  its longest side squared stays of the order of END_GAP at least, far above the 1e-12 below which
  `Mesh.defects()` counts a triangle as without area.
- **Closed by construction.** Each vertex is made once, for its edge, and every triangle that uses the edge
  names it by the same index. The piece of surface in a tetrahedron is decided by which of its four corners
  are inside (16 cases): one triangle around a corner that differs from the other three, or a quadrilateral
  cut in two along a fixed diagonal when two corners are inside. Every face of a tetrahedron whose corners
  are not all on one side holds exactly one segment of the surface, between its two edges that change side,
  and the tetrahedron on the other side of the face uses the same two vertices, so it holds the same segment.
  The direction of every triangle is read off the case (every case is a turned copy of one of three forms
  whose direction is fixed in a tetrahedron of positive orientation), never computed from coordinates, so
  the two triangles of a segment run along it in opposite directions, and every triangle faces away from the
  inside corners. On the lattice's outer layer every sample must be below the level, so no surface reaches
  it: `iso_surface` refuses a field whose outer layer reaches the level, naming the first such sample (the
  caller pads the field with a layer below the level), and a field that reaches the level nowhere. Hence every
  edge of the surface lies on exactly two triangles that run along it in opposite directions, whatever the
  field: the surface passes `Mesh.closed()`, and a vertex where two parts of the body touch is never shared,
  since every vertex lies inside an edge.
- **Tests.** The engine's tests check that the six tetrahedra are positive, fill the cell and cut each face
  alike on both sides; that each of the 16 cases faces away from its inside corners in all six tetrahedra;
  that random fields, fields of 0, the level and 1, and fields one unit in the last place either side of the
  level give closed surfaces whose vertex count equals the number of edges that change side; and the thread
  counts. Verification row 34 measures a sphere's distance field at three spacings against the exact volume
  and area (the volume error falls by a factor of 4 per halving), a torus (genus 1), two spheres joined,
  touching and apart (touching at a lattice point exactly at the level, they come out as one body joined at
  that point), fields with whole layers of samples at the level, and the thread count.
- **Order and threads.** The vertices are numbered by the lattice point their edge starts from and then by
  the edge's offset; the triangles come cell by cell in the order of the points, tetrahedron by tetrahedron.
  Every point and every cell is computed from the samples alone and the pieces are joined in that order, so
  the vertices, the triangles and their order are bit for bit the same on every thread count.
- **Which samples are joined: the split is not mirror-symmetric.** Two inside samples are joined by the
  surface's body when they are the ends of an edge of the tetrahedra: neighbours along x, y and z, along the
  face diagonals (1, 1, 0), (1, 0, 1) and (0, 1, 1), and along the body diagonal (1, 1, 1), but not along the
  other diagonals, such as (1, -1, 0) or (1, 1, -1). Two inside samples that meet only across such a diagonal
  (their other neighbours outside) are thus one body in one orientation and two bodies in its mirror image,
  and where they are joined the body is one body with a neck, not two. Beyond that, the split changes only
  where vertices lie within a cell. So the surface of a field that is its own mirror image is not exactly its
  own mirror image, and its number of bodies and its genus can differ from those of a voxel set joined
  through faces only, where samples meet only along a diagonal.

### The measures of a mesh

- `Mesh.bodies()`: the sets of triangles joined through shared edges (two triangles that share only a
  vertex are not joined), each as a `Mesh` with its own vertices, largest first by the volume it encloses. A
  cavity inside a solid is a body of its own, facing inward with a negative volume, so it comes after the
  solids.
- `Mesh.euler_characteristic()` is V - E + F (the vertices the triangles use, the edges counted once, the
  triangles), and `Mesh.genus()` the number of through holes of one closed body, (2 - chi) / 2: 0 for a
  sphere, 1 for a torus. It refuses a mesh that is not closed or that is more than one body.
- `Mesh.roughness_deg()`: the mean angle between the normals of two triangles that share an edge, in
  degrees; it falls as a faceted surface is smoothed.

### Refining a field

`refine(field, factor, order="cubic")` resamples the field on a lattice `factor` times finer: the spacing
divided by `factor`, the same origin, (n - 1) factor + 1 points along an axis of n, so every old point is a
new one and keeps its sample exactly. The new samples are interpolated one axis after the other.

- `"linear"`: linear along each axis, which makes the field trilinear in each cell.
- `"cubic"`: piecewise cubic Hermite interpolation with the slopes of Fritsch and Butland (1984), which meet
  the condition of Fritsch and Carlson (1980) for a monotone interpolant: at a sample, the harmonic mean of
  the differences to its two neighbours, 0 where they differ in sign or one is 0, and the one difference at
  the ends of a line. It reproduces fields that are linear along an axis, and between two samples it never
  leaves their range (each new sample is also held within it, which removes rounding). Its error falls with
  the cube of the spacing where the field is monotone along the line, and with the square in the end
  intervals of a line and next to a sample where the field turns.
- **At a step.** Neither order overshoots. Between the 0 and the 1 of a step (0, 0, 1, 1 along an axis) the
  cubic order gives 3 t^2 - 2 t^3, which crosses 0.5 half-way, and exactly 0 and 1 beyond them; a 0/1 field
  stays within [0, 1]. A cubic convolution (Catmull-Rom) was not used: it overshoots by 2/27 on each side of
  such a step, about 1.24 at a corner of a box, and can lift a sample of the outer layer above the level. With
  the monotone order the new samples of the outer layer come from the old ones of the outer layer alone and
  stay within their range, so a field whose outer layer is below a level stays so after refining.

### Smoothing

`smooth(mesh, iterations, lam=0.5, mu=-0.53, fixed=None)` is Taubin's smoothing (Taubin 1995): each iteration
is a step x += lam L(x) followed by a step x += mu L(x), with L(x) at a vertex the mean of its neighbours
(the vertices it shares an edge with) minus the vertex. The first step smooths and shrinks, the second, with
mu < -lam, swells the surface back, so the shape keeps its size while short ripples die out. Vertices marked
in `fixed` stay where they are and still pull their neighbours. It moves vertices only and never changes the
triangles, so a closed mesh stays closed (a vertex can still be pulled through a neighbouring triangle on a
very thin wall; the closedness of the mesh by its edges is not affected). The result, a `Smoothing`, gives the
largest move of a vertex and the volumes before and after. The neighbours' weights are equal, so the steps
also slide vertices along the surface towards an even spacing: on the torus of row 35, 50 iterations move a
vertex by up to 0.4 mm on a 0.5 mm lattice while the volume changes by 0.4 %.

It is written with NumPy: each step is three sums over the edges (`numpy.bincount`). On a surface of 1.16
million triangles (the sphere-like field on a lattice of 300 x 200 x 200 points) an iteration takes about 0.13
s on one core and 50 iterations 6.4 s, against 0.25 s for the iso-surface itself on 4 threads and the minutes
an optimisation of such a part takes, so it is not moved into the engine.

### 3MF files

`write_3mf(path, mesh, unit="millimeter")` writes a closed mesh (`Mesh.closed()`, which turns a mesh facing
inward the right way out; anything not closed is refused) as a 3MF package (3MF Core Specification 1.4.0)
with one object: a zip archive, deflated, of `[Content_Types].xml` (written first), `_rels/.rels`, whose one
relationship names the model part, and `3D/3dmodel.model`, a `model` in the core namespace
(`http://schemas.microsoft.com/3dmanufacturing/core/2015/02`) with one `object` of type "model" holding the
`mesh` (`vertices` and `triangles`) and a `build` of one `item`. Every part carries the same fixed date, so the
same mesh gives the same bytes. Each coordinate is written as the shortest decimal that reads back to the same
float64 (Python's `repr`), so in millimetres the vertices read back bit for bit; in another unit (`micron`,
`centimeter`, `inch`, `foot`, `meter`) the coordinates are divided by the unit's size in mm and multiplied back on
reading. `read_3mf(path)` reads such a package: exactly those three parts (the model part as the relationship
names it), one object with one mesh, one build item without a transform; anything else (more parts, objects,
components, materials, transforms, extensions, other elements or attributes) is refused with a message that
names it, never skipped. Verification row 35 writes the torus of row 34 and reads it back, and checks the
smoothing of that torus.

## From the result to the mesh

`michell.finish(part, setup)` (`michell.part_mesh.finish`) turns the outcome of `stiff_part` into the part's
closed triangle mesh: the organic surface of the optimised design, cut by the envelope, joined with the
exact solid regions and cut by the exact void regions, so that every face of a region is the region's own.
It returns a `FinishedPart`: the mesh, the level of the surface, the volumes and masses, the loose pieces
dropped, the closed cavities, every gate with its numbers, the genus against the voxel body's, and the
check of the delivered part. The cut and join use [manifold3d](https://github.com/elalish/manifold)
(Apache-2.0, version 3.5.0 or later, a required dependency) through its 64-bit mesh interface (`Mesh64`),
so that the regions' vertices keep their float64 coordinates: a 1e-6 mm interface needs that, since float32
holds about 1e-5 mm on a 100 mm part. `michell run` calls it when the problem file asks for
`[output] mesh` ([problem-file.md](problem-file.md#what-comes-back)).

- **The field.** On the voxel centres: the result's filtered field on the design voxels, since that is
  the field the optimisation drew its designs from. 1 on the voxels of a held skin (held solid without an
  exact region, such as the skin of a load), so that the surface itself holds them. Continued by
  repetition (each new voxel takes the mean of its known face neighbours, one layer at a time) three
  layers into the voxels of the solid regions, so that members run on into the regions and the organic
  body overlaps them instead of ending on their faces; deeper inside the regions 0, so that the organic
  surface never runs close beside an exact face, where the cut would leave slivers. Continued three layers
  into the inactive voxels and beyond the grid, so that the surface overshoots the envelope everywhere and
  is cut by it; then padded with 0, so that the outer layer is below every level.
- **The surface.** The field refined by 2 with the monotone cubic order, `iso_surface` at the level, and 10
  Taubin iterations, all before the cut and join: smoothing slides vertices along the surface, so it must
  never touch an exact face. Measured on the crossing bars of row 36 and on the example bridge (factor 1, 2,
  3 against 0 to 40 iterations): at factor 1 the surface stays rough (7 to 10 degrees between neighbouring
  normals); 5 iterations turn triangles over on the bridge's thin members at factor 1 and 2 (13 and 14),
  10 or more none; at factor 2 with 10 iterations the roughness is 3.8 degrees on the bars and 5.6 on the
  bridge, and the compliance of the mesh voxelised back is within 0.15 % of that at factor 3, which doubles
  the triangles. On the bars, a part symmetric about the plane y = 10, the centre of mass lies 8e-7 mm off that
  plane and the two halves' volumes differ by 2e-7 of each other (1e-9 and 4e-10 with no smoothing, since
  smoothing with equal weights follows the tetrahedra, which are not mirror-symmetric).
- **The cut and join.** (organic body intersected with the envelope) joined with the solid regions, minus
  the void regions, where the envelope is the design and solid regions joined. Each triangle of a region
  carries the face id of its plane, so that the triangles of the result can be traced to the region and
  the face they came from. No step relies on bodies that only touch: the organic body overlaps the solid
  regions and overshoots the envelope by construction, and the envelope is one boolean union whose result
  is checked. Three arrangements pass every gate (row 36): solid regions nested in the design region,
  overlapping it, and touching it on a face. A region that touches the design region on a face is joined by
  the union itself: manifold3d's symbolic perturbation treats the shared face as inside, so the union is one
  body whose volume is the two volumes added (also when the shared face is only part of a face), and the
  organic body crosses that face, since its field runs on into the region. A region that neither overlaps
  nor touches the others cannot be joined: the envelope gate refuses it by name, with the advice to make
  the regions overlap. Where the outline of the organic body's cut meets an edge of a region's face,
  manifold3d can leave a triangle without area (its three corners on that edge); it is removed by flipping
  its longest edge with the neighbour across it, which moves no vertex and keeps the neighbour's plane.
- **The level** is found by a root search on the organic part, measured on the finished body: the
  finished body's volume minus that of the solid regions less the void regions (the finished body holds
  them whole) equals the volume of the returned design's design-class voxels (its nominal design at 0.5 or
  more, not held by a solid region; skin voxels count). The search starts at 0.5, brackets the root by steps
  that double, and closes it by false position (Illinois), until it is within 0.2 %. The exact regions count
  with their exact volumes. Where the volume jumps at a level (a member that closes or opens) and the
  search cannot reach 0.2 %, the closest level is taken and `LevelOffTarget` warns.
- **Loose pieces.** Bodies of the finished mesh with positive volume below 1 % of the whole are dropped and
  listed (`LoosePiece`); a larger second body is refused: the optimisation's design was one body, and a
  second one that size is not noise. Closed cavities (bodies with negative volume, inside another) are kept
  and counted.
- **The gates.** A mesh that fails one is not returned: `MeshRefused` (code `mesh_refused`) names the gate
  and holds the numbers. The **envelope** is one body. The mesh is **closed** after merging equal vertices
  (every edge on exactly two triangles, in opposite directions) and has **no triangle without area**; it is
  **one body**. The centre of every **held-solid voxel** (of the regions and of the skins) is **inside** the
  mesh and the centre of every voxel a **void region** fills by majority is **outside** it (each centre
  tested on the middle ray of the voxelisation below, and where that ray is not conclusive by the parity of
  a ray along a direction off every axis). **Exact interfaces**: every vertex of a triangle the mesh took
  from a region lies on the plane of the region's face it came from within 1e-6 mm, and every vertex inside
  a void region's wall (used only by its triangles) is one of the void mesh's own vertices. Where the wall
  runs on beyond the part, the vertices on its edge with the part's faces are new by necessity; a void
  cylinder from face to face of the part has its own vertices only (row 36). The **genus** of the mesh's
  main body is compared with that of the returned design's voxel body, with a warning (`GenusDiffers`), not
  a refusal. The voxel body is counted as the iso-surface at 0.5 of the returned design's 0/1 voxels, drawn
  by the same tetrahedra, so that the two counts join samples along the same diagonals; a face-joined
  count of the voxels could differ along the others.
- **The delivered part is checked, not assumed.** The finished mesh is voxelised back on the run's grid
  (the voxeliser on a grid three times finer, one ray per fine column; a voxel whose occupancy is at least
  one half is solid); the voxels gained and lost are reported as a share of the returned design's solid
  voxels, and each load case is solved once on that 0/1 design and its compliance (and the weighted goal)
  compared with the returned design's. Within 5 % the result says so; beyond it `DeliveredPartDiffers`
  warns, the numbers are in the result, and `michell run` exits with status 1. On the example bridge the
  mesh voxelised back is 3.3 % stiffer (its compliance is lower): the returned design still holds some
  grey voxels, which carry less than solid ones, while the voxels of the mesh are all solid; and 0.9 % of
  the solid voxels change side, since the surface at the level that matches the design's volume rounds the
  steps of the voxels off (row 37).
- **Without region meshes.** For a result built in Python on a bare `Problem` (no voxel model), with its
  `KeepSolid`, the held voxels are held by the field (1 there), the cut is the grid's box and exactness is
  not claimed (`FinishedPart.exact` is False, and the gate of exact interfaces says so).

STL files store float32 coordinates, so `part.stl` holds the mesh to about 1e-5 mm on a 100 mm part, and
`michell run` warns (`StlRounding`) if that rounding would join two vertices; `part.3mf` keeps the float64
coordinates bit for bit. Time and memory of `finish` (one level search, the gates and the delivered-part
check, which solves each load case twice), on the crossing bars scaled up, voxels of 1 mm: 99 528 voxels,
188 000 triangles, 16 s on one thread with a peak of 264 MB (the two solves 14 s of it); 1.5 million voxels,
1.14 million triangles, 121 s on four threads with a peak of 2.3 GB, of which the field, the level search and
the gates take 39 s and the two solves 82 s.

Verification row 36 checks the three arrangements, the exact face of a solid box and the wall of a void
cylinder, and the refusals on results built by hand; row 37 checks that the held skin of row 32's bore
stays inside the mesh after a short optimisation, and runs `michell run` on the example bridge, which
writes `part.stl` and `part.3mf`.
