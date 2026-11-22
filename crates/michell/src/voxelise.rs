//! The voxeliser: the share of every voxel of a grid that lies inside a combination of closed triangle
//! meshes, the union of some bodies minus the union of others.
//!
//! Rays run along x through every voxel column (j, k), r x r of them at the centres of the column's
//! sub-squares (r = 3 by default). On each ray, each body's inside set is a list of x-intervals
//! (`voxel_rays`, which also holds the rule for rays through edges and vertices); the intervals of the
//! adding bodies are joined, those of the subtracting bodies are joined, and the second set is cut out of
//! the first. A voxel's share on that ray is the exact length of the combined set inside the voxel over the
//! voxel side, and its occupancy is the mean of its shares over the r^2 rays of its column. So nothing is
//! sampled along x: the occupancy is exact in x and sampled at r x r points in y and z.
//!
//! Each ray looks only at the triangles near it: before the rays are cast, every triangle is entered in the
//! buckets of the voxel columns that its projected bounding box touches (a closed box, so that a ray on the
//! box's edge still sees the triangle), and a ray tests only the triangles of its column's bucket.
//!
//! Threads: every voxel column is one piece of the output (its nx entries are contiguous, since x runs
//! fastest), computed from the inputs alone in a fixed order of rays, bodies and intervals. Which thread
//! computes a column cannot change its numbers, so the result is bit for bit the same on every thread count
//! (`threads`). Nor does the result depend on the order of the triangles (the crossings of a ray are sorted
//! by x before they are used) or on which corner of a triangle is listed first (`ProjectedTriangle::new`
//! turns the corners into one order).

use std::sync::Mutex;

use crate::error::EngineError;
use crate::threads::Threads;
use crate::voxel_rays::{
    Crossing, LatticePoint, ProjectedTriangle, inside_intervals, ray_point, snap, subtract, union,
    units_per_voxel,
};

/// The rays per side of a voxel when the caller does not say: 3 x 3 rays per voxel column.
pub const DEFAULT_RAYS_PER_SIDE: usize = 3;

/// The largest number of rays per side accepted.
pub const MAX_RAYS_PER_SIDE: usize = 64;

/// A closed triangle mesh in mm: vertices and triangles as index triples. A triangle's corners run
/// counter-clockwise seen from outside (its normal points out); the voxeliser counts a point as inside where
/// the winding number of the mesh around it is positive.
#[derive(Clone, Debug)]
pub struct Body {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
}

impl Body {
    /// A body from its vertices (mm, finite) and triangles (indices into `vertices`).
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[usize; 3]>) -> Result<Body, EngineError> {
        if let Some(index) = vertices
            .iter()
            .position(|v| !v.iter().all(|c| c.is_finite()))
        {
            return Err(EngineError::invalid_input(
                "the coordinates of a vertex must be finite numbers",
            )
            .with_integer("vertex", index));
        }
        if triangles.len() > u32::MAX as usize {
            return Err(
                EngineError::invalid_input("a body may have at most 4294967295 triangles")
                    .with_integer("triangles", triangles.len()),
            );
        }
        for (index, triangle) in triangles.iter().enumerate() {
            if let Some(&corner) = triangle.iter().find(|&&c| c >= vertices.len()) {
                return Err(EngineError::invalid_input(format!(
                    "triangle {index} names vertex {corner}, but the body has {} vertices",
                    vertices.len()
                ))
                .with_integer("triangle", index)
                .with_integer("vertex", corner)
                .with_integer("vertices", vertices.len()));
            }
        }
        Ok(Body {
            vertices,
            triangles,
        })
    }

    /// The vertices, mm.
    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    /// The triangles, as index triples into the vertices.
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }
}

/// The grid the occupancy is computed on: `counts` voxels along x, y and z, of side `h` mm, with the corner
/// of voxel (0, 0, 0) at `origin` in the meshes' coordinates (mm). Voxel (i, j, k) has index
/// i + nx (j + ny k), as in `Grid3d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    pub counts: [usize; 3],
    pub h: f64,
    pub origin: [f64; 3],
}

/// One body prepared for the rays: its triangles on the lattice, and for each voxel column (j, k) the
/// triangles whose projected bounding box touches it, as compressed rows (`starts[c]..starts[c + 1]` of
/// `entries` for column c = j + ny k).
struct PreparedBody {
    triangles: Vec<ProjectedTriangle>,
    starts: Vec<usize>,
    entries: Vec<u32>,
}

/// Where a ray found a body that is not closed: the first by column, then by ray, then by body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OpenRay {
    column: usize,
    ray: usize,
    subtracting: bool,
    body: usize,
    winding: i32,
}

/// The occupancy of every voxel of `grid`: the share of the voxel inside the union of `adding` minus the
/// union of `subtracting`, one number between 0 and 1 per voxel in the order of `Grid3d`. `rays_per_side`
/// rays per side of a voxel (r x r per column, at least 1), on `threads` threads (at least 1); the numbers do
/// not depend on the thread count.
///
/// Refused with `invalid_input`: a grid without voxels, a voxel side that is not a positive finite number, an
/// origin that is not finite, a vertex whose y or z lies too far from the grid to be put on the lattice,
/// and a body that a ray finds not closed (its winding number is not 0 beyond it), with the body and the ray.
pub fn occupancy(
    grid: &VoxelGrid,
    adding: &[Body],
    subtracting: &[Body],
    rays_per_side: usize,
    threads: usize,
) -> Result<Vec<f64>, EngineError> {
    let [nx, ny, nz] = grid.counts;
    check_grid(grid)?;
    if rays_per_side == 0 || rays_per_side > MAX_RAYS_PER_SIDE {
        return Err(EngineError::invalid_input(format!(
            "the rays per side of a voxel must be between 1 and {MAX_RAYS_PER_SIDE}"
        ))
        .with_integer("rays_per_side", rays_per_side));
    }
    let threads = Threads::new(threads)?;
    let adding = prepare_all(grid, adding, rays_per_side, false)?;
    let subtracting = prepare_all(grid, subtracting, rays_per_side, true)?;
    let first_open: Mutex<Option<OpenRay>> = Mutex::new(None);
    let mut out = vec![0.0; nx * ny * nz];
    let rays = (rays_per_side * rays_per_side) as f64;
    threads.for_each_piece(&mut out, nx, |column, values| {
        let (j, k) = (column % ny, column / ny);
        let mut work = ColumnWork::default();
        for b in 0..rays_per_side {
            for a in 0..rays_per_side {
                let ray = a + rays_per_side * b;
                let p = ray_point(j, k, a, b, rays_per_side);
                let r = rays_per_side as f64;
                let p_yz = [
                    j as f64 + (a as f64 + 0.5) / r,
                    k as f64 + (b as f64 + 0.5) / r,
                ];
                let found = work.combined(&adding, &subtracting, column, p, p_yz);
                if let Err((subtracting, body, winding)) = found {
                    let open = OpenRay {
                        column,
                        ray,
                        subtracting,
                        body,
                        winding,
                    };
                    let mut first = first_open
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if first.is_none_or(|known| open < known) {
                        *first = Some(open);
                    }
                    return;
                }
                add_lengths(&work.combined, grid, values);
            }
        }
        for value in values.iter_mut() {
            // Each ray adds at most 1 per voxel; rounding must not lift a full voxel above 1.
            *value = (*value / rays).min(1.0);
        }
    });
    let first_open = first_open
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(open) = first_open {
        let (j, k) = (open.column % ny, open.column / ny);
        let (a, b) = (open.ray % rays_per_side, open.ray / rays_per_side);
        let r = rays_per_side as f64;
        let y = grid.origin[1] + (j as f64 + (a as f64 + 0.5) / r) * grid.h;
        let z = grid.origin[2] + (k as f64 + (b as f64 + 0.5) / r) * grid.h;
        let list = if open.subtracting {
            "subtracting"
        } else {
            "adding"
        };
        return Err(EngineError::invalid_input(format!(
            "{list} body {} is not closed: the ray along x at y = {y} mm, z = {z} mm ends with winding \
             number {} beyond it",
            open.body, open.winding
        ))
        .with_integer("body", open.body)
        .with_number("y", y)
        .with_number("z", z)
        .with_number("winding", f64::from(open.winding)));
    }
    Ok(out)
}

fn check_grid(grid: &VoxelGrid) -> Result<(), EngineError> {
    let [nx, ny, nz] = grid.counts;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(EngineError::invalid_input(
            "the grid needs at least one voxel along each axis",
        )
        .with_integer("nx", nx)
        .with_integer("ny", ny)
        .with_integer("nz", nz));
    }
    if nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)).is_none() {
        return Err(EngineError::invalid_input("the grid has too many voxels"));
    }
    if !(grid.h.is_finite() && grid.h > 0.0) {
        return Err(EngineError::invalid_input(
            "the voxel side must be a positive finite number (mm)",
        )
        .with_number("h", grid.h));
    }
    if !grid.origin.iter().all(|c| c.is_finite()) {
        return Err(EngineError::invalid_input(
            "the origin of the grid must be finite (mm)",
        ));
    }
    Ok(())
}

fn prepare_all(
    grid: &VoxelGrid,
    bodies: &[Body],
    rays_per_side: usize,
    subtracting: bool,
) -> Result<Vec<PreparedBody>, EngineError> {
    bodies
        .iter()
        .enumerate()
        .map(|(index, body)| {
            prepare(grid, body, rays_per_side).map_err(|vertex| {
                let list = if subtracting { "subtracting" } else { "adding" };
                EngineError::invalid_input(format!(
                    "vertex {vertex} of {list} body {index} lies too far from the grid in y or z"
                ))
                .with_integer("body", index)
                .with_integer("vertex", vertex)
            })
        })
        .collect()
}

/// Puts a body on the lattice and fills the buckets of its columns. `Err(vertex)` names a vertex that does
/// not fit on the lattice.
fn prepare(grid: &VoxelGrid, body: &Body, rays_per_side: usize) -> Result<PreparedBody, usize> {
    let [_, ny, nz] = grid.counts;
    let units = units_per_voxel(rays_per_side);
    let mut lattice = Vec::with_capacity(body.vertices.len());
    for (index, vertex) in body.vertices.iter().enumerate() {
        let u = snap(vertex[1], grid.origin[1], grid.h, units).ok_or(index)?;
        let v = snap(vertex[2], grid.origin[2], grid.h, units).ok_or(index)?;
        lattice.push(LatticePoint { u, v });
    }
    let own = |vertex: &[f64; 3]| {
        [
            (vertex[1] - grid.origin[1]) / grid.h,
            (vertex[2] - grid.origin[2]) / grid.h,
        ]
    };
    let mut triangles = Vec::with_capacity(body.triangles.len());
    for corners in &body.triangles {
        let on_lattice = corners.map(|c| lattice[c]);
        let yz = corners.map(|c| own(&body.vertices[c]));
        let x = corners.map(|c| body.vertices[c][0]);
        if let Some(triangle) = ProjectedTriangle::new(on_lattice, yz, x) {
            triangles.push(triangle);
        }
    }
    // The columns a triangle's closed bounding box touches, clipped to the grid (empty when it misses it).
    let columns_of = |triangle: &ProjectedTriangle| {
        let corners = triangle.corners();
        let u_low = corners.iter().map(|c| c.u).min().unwrap_or(0);
        let u_high = corners.iter().map(|c| c.u).max().unwrap_or(0);
        let v_low = corners.iter().map(|c| c.v).min().unwrap_or(0);
        let v_high = corners.iter().map(|c| c.v).max().unwrap_or(0);
        let range = |low: i64, high: i64, count: usize| {
            let first = low.div_euclid(units).max(0);
            let last = high.div_euclid(units).min(count as i64 - 1);
            if first > last {
                0..0
            } else {
                first as usize..last as usize + 1
            }
        };
        (range(u_low, u_high, ny), range(v_low, v_high, nz))
    };
    let mut starts = vec![0usize; ny * nz + 1];
    for triangle in &triangles {
        let (js, ks) = columns_of(triangle);
        for k in ks {
            for j in js.clone() {
                starts[j + ny * k + 1] += 1;
            }
        }
    }
    for c in 0..ny * nz {
        starts[c + 1] += starts[c];
    }
    let mut filled = starts.clone();
    let mut entries = vec![0u32; starts[ny * nz]];
    for (index, triangle) in triangles.iter().enumerate() {
        let (js, ks) = columns_of(triangle);
        for k in ks {
            for j in js.clone() {
                let c = j + ny * k;
                // At most u32::MAX triangles, checked when the body was built.
                entries[filled[c]] = index as u32;
                filled[c] += 1;
            }
        }
    }
    Ok(PreparedBody {
        triangles,
        starts,
        entries,
    })
}

/// The work space of one column, kept from ray to ray so that nothing is allocated per ray.
#[derive(Default)]
struct ColumnWork {
    crossings: Vec<Crossing>,
    added: Vec<(f64, f64)>,
    removed: Vec<(f64, f64)>,
    combined: Vec<(f64, f64)>,
}

impl ColumnWork {
    /// Fills `combined` with the inside set of the ray through `p` in `column`: the union of the adding
    /// bodies minus the union of the subtracting ones. `Err((subtracting, body, winding))` names a body the
    /// ray finds not closed.
    fn combined(
        &mut self,
        adding: &[PreparedBody],
        subtracting: &[PreparedBody],
        column: usize,
        p: LatticePoint,
        p_yz: [f64; 2],
    ) -> Result<(), (bool, usize, i32)> {
        self.added.clear();
        for (index, body) in adding.iter().enumerate() {
            inside_on_ray(
                body,
                column,
                (p, p_yz),
                &mut self.crossings,
                &mut self.added,
            )
            .map_err(|winding| (false, index, winding))?;
        }
        union(&mut self.added);
        self.removed.clear();
        for (index, body) in subtracting.iter().enumerate() {
            inside_on_ray(
                body,
                column,
                (p, p_yz),
                &mut self.crossings,
                &mut self.removed,
            )
            .map_err(|winding| (true, index, winding))?;
        }
        union(&mut self.removed);
        self.combined.clear();
        subtract(&self.added, &self.removed, &mut self.combined);
        Ok(())
    }
}

/// Appends the inside intervals of one body on the ray through `ray` (in `column`; the point on the lattice
/// and in voxel sides) to `inside`.
fn inside_on_ray(
    body: &PreparedBody,
    column: usize,
    ray: (LatticePoint, [f64; 2]),
    crossings: &mut Vec<Crossing>,
    inside: &mut Vec<(f64, f64)>,
) -> Result<(), i32> {
    crossings.clear();
    for &index in &body.entries[body.starts[column]..body.starts[column + 1]] {
        if let Some(crossing) = body.triangles[index as usize].crossing(ray.0, ray.1) {
            crossings.push(crossing);
        }
    }
    inside_intervals(crossings, inside)
}

/// Adds to `values` (one entry per voxel of a column) the length of each interval inside each voxel, in
/// voxel sides. The positions are taken in voxel sides from the grid's origin, so a voxel runs from i to
/// i + 1 exactly.
fn add_lengths(intervals: &[(f64, f64)], grid: &VoxelGrid, values: &mut [f64]) {
    let nx = grid.counts[0];
    for &(low, high) in intervals {
        let start = ((low - grid.origin[0]) / grid.h).max(0.0);
        let end = ((high - grid.origin[0]) / grid.h).min(nx as f64);
        if end <= start {
            continue;
        }
        // Both are between 0 and nx here, so the casts are exact.
        let first = start.floor() as usize;
        let last = (end.ceil() as usize).min(nx);
        for (i, value) in values.iter_mut().enumerate().take(last).skip(first) {
            let length = end.min((i + 1) as f64) - start.max(i as f64);
            if length > 0.0 {
                *value += length;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 12 triangles of the box from `low` to `high`, outward.
    fn cuboid(low: [f64; 3], high: [f64; 3]) -> Body {
        let mut vertices = Vec::new();
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    let pick =
                        |axis: usize, bit: usize| if bit == 0 { low[axis] } else { high[axis] };
                    vertices.push([pick(0, i), pick(1, j), pick(2, k)]);
                }
            }
        }
        // Corner (i, j, k) is vertex i + 2 j + 4 k; each face as two triangles, outward.
        let faces = [
            [0, 4, 6, 2], // x = low
            [1, 3, 7, 5], // x = high
            [0, 1, 5, 4], // y = low
            [2, 6, 7, 3], // y = high
            [0, 2, 3, 1], // z = low
            [4, 5, 7, 6], // z = high
        ];
        let mut triangles = Vec::new();
        for [a, b, c, d] in faces {
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
        Body::new(vertices, triangles).unwrap()
    }

    /// A body from vertices and triangles given as arrays.
    fn body(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Body {
        Body::new(vertices.to_vec(), triangles.to_vec()).unwrap()
    }

    /// The square pyramid with base the square of half-side `half` around (`base_x`, yc, zc) in the plane
    /// x = base_x and apex at x = base_x + height on the axis, outward.
    fn pyramid(base_x: f64, yc: f64, zc: f64, half: f64, height: f64) -> Body {
        let vertices = [
            [base_x, yc - half, zc - half],
            [base_x, yc + half, zc - half],
            [base_x, yc + half, zc + half],
            [base_x, yc - half, zc + half],
            [base_x + height, yc, zc],
        ];
        let triangles = [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ];
        body(&vertices, &triangles)
    }

    fn volume(values: &[f64], h: f64) -> f64 {
        values.iter().sum::<f64>() * h * h * h
    }

    fn grid(counts: [usize; 3], h: f64, origin: [f64; 3]) -> VoxelGrid {
        VoxelGrid { counts, h, origin }
    }

    fn shifted(body: &Body, by: [f64; 3]) -> Body {
        let vertices = body
            .vertices()
            .iter()
            .map(|v| [v[0] + by[0], v[1] + by[1], v[2] + by[2]])
            .collect();
        Body::new(vertices, body.triangles().to_vec()).unwrap()
    }

    #[test]
    fn a_box_on_voxel_planes_fills_its_voxels_exactly() {
        let g = grid([6, 5, 4], 0.5, [-1.0, -0.5, 0.0]);
        let b = cuboid([-0.5, 0.0, 0.5], [1.5, 1.5, 1.5]);
        let values = occupancy(&g, &[b], &[], 3, 1).unwrap();
        for k in 0..4 {
            for j in 0..5 {
                for i in 0..6 {
                    let inside = (1..5).contains(&i) && (1..4).contains(&j) && (1..3).contains(&k);
                    assert_eq!(values[i + 6 * (j + 5 * k)], f64::from(u8::from(inside)));
                }
            }
        }
    }

    #[test]
    fn a_box_off_the_voxel_planes_along_x_is_exact_in_x() {
        let g = grid([5, 2, 2], 1.0, [0.0, 0.0, 0.0]);
        let b = cuboid([0.3, 0.0, 0.0], [3.55, 2.0, 1.0]);
        let values = occupancy(&g, &[b], &[], 3, 1).unwrap();
        let expected_row = [0.7, 1.0, 1.0, 0.55, 0.0];
        for (i, expected) in expected_row.iter().enumerate() {
            assert!((values[i] - expected).abs() < 1e-14, "{i}: {}", values[i]);
            assert_eq!(values[i + 5 * 2], 0.0);
        }
        assert!((volume(&values, 1.0) - 3.25 * 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_box_whose_edges_lie_on_rays_counts_each_ray_once() {
        // Rays of a column at 1/6, 1/2 and 5/6 of a voxel side. The box's y and z bounds lie on rays, and its
        // end faces are cut into many triangles whose corners lie on rays too (a fan around a ray, and edges
        // through rays): every ray on the half-open square [y0, y1) x [z0, z1) must cross exactly once.
        let h = 1.0;
        let (y0, y1, z0, z1) = (1.0 / 6.0, 13.0 / 6.0, 1.0 / 6.0, 7.0 / 6.0);
        let (x0, x1) = (0.25, 2.75);
        // Corners two ray spacings apart, so that rays also cross edges and diagonals half-way along.
        let ys = [y0, 5.0 / 6.0, 1.5, y1];
        let zs = [z0, 5.0 / 6.0, z1];
        // Vertices of both end faces on the grid of points ys x zs, then the side faces as two triangles.
        let mut vertices = Vec::new();
        for &x in &[x0, x1] {
            for &z in &zs {
                for &y in &ys {
                    vertices.push([x, y, z]);
                }
            }
        }
        let (ny, nz) = (ys.len(), zs.len());
        let index = |end: usize, jy: usize, kz: usize| end * ny * nz + jy + ny * kz;
        let mut triangles = Vec::new();
        for kz in 0..nz - 1 {
            for jy in 0..ny - 1 {
                let corners = |end| {
                    [
                        index(end, jy, kz),
                        index(end, jy + 1, kz),
                        index(end, jy + 1, kz + 1),
                        index(end, jy, kz + 1),
                    ]
                };
                let [a, b, c, d] = corners(0);
                // Seen from -x (outside the face x = x0) the corners run clockwise in (y, z).
                if (jy + kz) % 2 == 0 {
                    triangles.extend([[a, c, b], [a, d, c]]);
                } else {
                    triangles.extend([[a, d, b], [b, d, c]]);
                }
                let [a, b, c, d] = corners(1);
                if (jy + kz) % 2 == 0 {
                    triangles.extend([[a, b, c], [a, c, d]]);
                } else {
                    triangles.extend([[a, b, d], [b, c, d]]);
                }
            }
        }
        // The side faces (parallel to the rays: dropped by the voxeliser, but they close the mesh).
        let ring: Vec<(usize, usize)> = (0..ny - 1)
            .map(|j| (j, 0))
            .chain((0..nz - 1).map(|k| (ny - 1, k)))
            .chain((1..ny).rev().map(|j| (j, nz - 1)))
            .chain((1..nz).rev().map(|k| (0, k)))
            .collect();
        for step in 0..ring.len() {
            let (ja, ka) = ring[step];
            let (jb, kb) = ring[(step + 1) % ring.len()];
            let (a, b) = (index(0, ja, ka), index(0, jb, kb));
            let (c, d) = (index(1, jb, kb), index(1, ja, ka));
            triangles.extend([[a, b, c], [a, c, d]]);
        }
        let b = Body::new(vertices, triangles).unwrap();
        let g = grid([3, 3, 2], h, [0.0, 0.0, 0.0]);
        let values = occupancy(&g, std::slice::from_ref(&b), &[], 3, 1).unwrap();
        // Rays in y at 1/6 + n/3: the box takes n = 0..5 (6 of them), half-open; in z n = 0..2 (3).
        assert!((volume(&values, h) - 2.5 * 2.0 * 1.0).abs() < 1e-12);
        for k in 0..2 {
            for j in 0..3 {
                let rays_y = [3.0, 3.0, 0.0][j];
                let rays_z = [3.0, 0.0][k];
                let share = rays_y * rays_z / 9.0;
                let expected = [0.75 * share, share, 0.75 * share];
                for i in 0..3 {
                    let value = values[i + 3 * (j + 3 * k)];
                    assert!(
                        (value - expected[i]).abs() < 1e-15,
                        "({i}, {j}, {k}): {value}"
                    );
                }
            }
        }
        // The same body with its triangles in another order and their corners rotated: bit for bit equal.
        let mut reordered: Vec<[usize; 3]> = b.triangles().to_vec();
        reordered.reverse();
        let reordered: Vec<[usize; 3]> = reordered
            .iter()
            .enumerate()
            .map(|(n, t)| match n % 3 {
                0 => *t,
                1 => [t[1], t[2], t[0]],
                _ => [t[2], t[0], t[1]],
            })
            .collect();
        let again = Body::new(b.vertices().to_vec(), reordered).unwrap();
        let values_again = occupancy(&g, &[again], &[], 3, 1).unwrap();
        assert!(
            values
                .iter()
                .zip(&values_again)
                .all(|(p, q)| p.to_bits() == q.to_bits())
        );
    }

    #[test]
    fn the_apex_of_a_pyramid_on_a_ray_is_counted_once() {
        // Axis along x through a ray (y = z = 1/2); the base corners on rays as well (1/6 and 5/6).
        let p = pyramid(0.2, 0.5, 0.5, 1.0 / 3.0, 1.7);
        let g = grid([2, 1, 1], 1.0, [0.0, 0.0, 0.0]);
        let values = occupancy(&g, std::slice::from_ref(&p), &[], 3, 1).unwrap();
        // The ray through the axis runs the full height inside; the eight others lie on the base's edges,
        // where the length is 0; half-open, the ray at y = 1/6, z = 1/6 is inside with length 0.
        assert!((values[0] - (0.8 / 9.0)).abs() < 1e-15, "{}", values[0]);
        assert!((values[1] - (0.9 / 9.0)).abs() < 1e-15, "{}", values[1]);
        // Shifted by a small irrational step, so no ray meets an edge or a vertex: nearly the same.
        let step = 1e-5 * std::f64::consts::SQRT_2;
        let moved = shifted(&p, [0.0, step, step * std::f64::consts::PI / 3.0]);
        let values_moved = occupancy(&g, &[moved], &[], 3, 1).unwrap();
        let (sum, sum_moved) = (volume(&values, 1.0), volume(&values_moved, 1.0));
        assert!((sum - sum_moved).abs() / sum < 1e-3, "{sum} {sum_moved}");
    }

    #[test]
    fn a_box_turned_by_45_degrees_with_an_edge_in_a_ray_plane() {
        // A square prism with vertical edges (along z) turned by 45 degrees about z: its cross-section in
        // (x, y) is a square with corners at (xc -+ d, yc) and (xc, yc -+ d). The edge at (xc - d, yc) lies in
        // the plane y = yc of the middle rays; on every ray the length inside is 2 (d - |y - yc|) exactly.
        let (xc, yc, d) = (2.0, 1.5, 1.2);
        let (z0, z1) = (0.0, 1.0);
        let corners_xy = [(xc - d, yc), (xc, yc - d), (xc + d, yc), (xc, yc + d)];
        let mut vertices = Vec::new();
        for &z in &[z0, z1] {
            for &(x, y) in &corners_xy {
                vertices.push([x, y, z]);
            }
        }
        // Corners 0..3 run counter-clockwise seen from +z; bottom 0..3, top 4..7.
        let triangles = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let b = body(&vertices, &triangles);
        let g = grid([4, 3, 1], 1.0, [0.0, 0.0, 0.0]);
        let values = occupancy(&g, &[b], &[], 3, 1).unwrap();
        let mut expected_sum = 0.0;
        for j in 0..3 {
            for a in 0..3 {
                let y = j as f64 + (a as f64 + 0.5) / 3.0;
                expected_sum += (2.0 * (d - (y - yc).abs())).max(0.0) * 3.0;
            }
        }
        let sum: f64 = values.iter().sum::<f64>() * 9.0;
        assert!((sum - expected_sum).abs() < 1e-12, "{sum} {expected_sum}");
    }

    #[test]
    fn an_octahedron_with_both_tips_on_one_ray() {
        // Tips at x = 0.25 and 2.75 on the ray y = z = 1/2; the four other corners on rays as well.
        let (c, s) = (0.5, 1.0 / 3.0);
        let vertices = [
            [0.25, c, c],
            [2.75, c, c],
            [1.5, c - s, c],
            [1.5, c, c - s],
            [1.5, c + s, c],
            [1.5, c, c + s],
        ];
        // The ring 2, 3, 4, 5 runs counter-clockwise in (y, z); the triangles at tip 0 face -x.
        let mut triangles = Vec::new();
        for n in 0..4 {
            let (p, q) = (2 + n, 2 + (n + 1) % 4);
            triangles.push([0, q, p]);
            triangles.push([1, p, q]);
        }
        let b = body(&vertices, &triangles);
        let g = grid([3, 1, 1], 1.0, [0.0, 0.0, 0.0]);
        let values = occupancy(&g, &[b], &[], 3, 1).unwrap();
        // Only the axis ray is inside, over its full length 2.5; the others touch edges with length 0.
        assert!((values.iter().sum::<f64>() * 9.0 - 2.5).abs() < 1e-14);
    }

    #[test]
    fn union_and_subtraction_of_bodies() {
        let g = grid([4, 1, 1], 1.0, [0.0, 0.0, 0.0]);
        let left = cuboid([0.0, 0.0, 0.0], [2.5, 1.0, 1.0]);
        let right = cuboid([1.5, 0.0, 0.0], [3.0, 1.0, 1.0]);
        let hole = cuboid([0.5, 0.0, 0.0], [1.25, 1.0, 1.0]);
        let values = occupancy(
            &g,
            &[left.clone(), right.clone()],
            std::slice::from_ref(&hole),
            3,
            1,
        )
        .unwrap();
        assert_eq!(values, vec![0.5, 0.75, 1.0, 0.0]);
        // A subtracting body outside every adding body removes nothing; nothing added gives nothing.
        let far = cuboid([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]);
        assert_eq!(
            occupancy(&g, std::slice::from_ref(&left), &[far], 3, 1).unwrap(),
            vec![1.0, 1.0, 0.5, 0.0]
        );
        assert_eq!(occupancy(&g, &[], &[hole], 3, 1).unwrap(), vec![0.0; 4]);
        // The rays per side change only the sampling in y and z: a box on voxel planes is the same.
        assert_eq!(
            occupancy(&g, &[left], &[], 1, 1).unwrap(),
            vec![1.0, 1.0, 0.5, 0.0]
        );
    }

    #[test]
    fn the_numbers_do_not_depend_on_the_thread_count() {
        // A torus-like ring of boxes and a pyramid cut out, on a grid of many columns.
        let g = grid([23, 31, 17], 0.37, [-0.3, -0.2, -0.1]);
        let mut adding = Vec::new();
        for n in 0..12 {
            let angle = n as f64 * std::f64::consts::PI / 6.0;
            let (cy, cz) = (5.5 + 3.5 * angle.cos(), 3.0 + 2.0 * angle.sin());
            adding.push(cuboid([0.4, cy - 1.0, cz - 0.7], [7.9, cy + 1.0, cz + 0.7]));
        }
        let cut = [pyramid(1.0, 5.0, 3.0, 2.0, 6.0)];
        let one = occupancy(&g, &adding, &cut, 3, 1).unwrap();
        assert!(one.iter().any(|v| *v > 0.0 && *v < 1.0));
        for threads in [2, 5, 12] {
            let many = occupancy(&g, &adding, &cut, 3, threads).unwrap();
            assert!(
                one.iter()
                    .zip(&many)
                    .all(|(p, q)| p.to_bits() == q.to_bits()),
                "{threads}"
            );
        }
    }

    #[test]
    fn an_open_body_and_bad_arguments_are_refused() {
        let g = grid([2, 2, 2], 1.0, [0.0, 0.0, 0.0]);
        let closed = cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut triangles = closed.triangles().to_vec();
        triangles.remove(1); // a hole in the face x = 0
        let open = Body::new(closed.vertices().to_vec(), triangles).unwrap();
        let error = occupancy(&g, std::slice::from_ref(&closed), &[open], 3, 1).unwrap_err();
        assert!(
            error.message.contains("subtracting body 0 is not closed"),
            "{}",
            error.message
        );
        for bad in [
            grid([0, 2, 2], 1.0, [0.0; 3]),
            grid([2, 2, 2], 0.0, [0.0; 3]),
            grid([2, 2, 2], f64::NAN, [0.0; 3]),
            grid([2, 2, 2], 1.0, [0.0, f64::INFINITY, 0.0]),
        ] {
            assert!(occupancy(&bad, std::slice::from_ref(&closed), &[], 3, 1).is_err());
        }
        assert!(occupancy(&g, std::slice::from_ref(&closed), &[], 0, 1).is_err());
        assert!(occupancy(&g, std::slice::from_ref(&closed), &[], 3, 0).is_err());
        let far = shifted(&closed, [0.0, 1e300, 0.0]);
        assert!(occupancy(&g, &[far], &[], 3, 1).is_err());
        assert!(Body::new(vec![[0.0, f64::NAN, 0.0]], vec![]).is_err());
        assert!(Body::new(vec![[0.0; 3]; 3], vec![[0, 1, 3]]).is_err());
    }
}
