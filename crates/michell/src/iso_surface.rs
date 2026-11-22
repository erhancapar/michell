//! The iso-surface of a scalar field sampled on a regular lattice: the closed triangle surface where the
//! field equals a level, with the side of larger values inside, by marching tetrahedra (Doi and Koide 1991)
//! on the split of every cell into six tetrahedra (`tetrahedra`).
//!
//! **The lattice** has `counts` (nx, ny, nz) points; point (i, j, k) lies at origin + (i, j, k) h and has
//! index i + nx (j + ny k), x fastest. Its cells are the cubes between neighbouring points.
//!
//! **The level rule.** A sample is inside when it is at least the level (f >= L) and outside when it is
//! below it; there is no third case, so a sample exactly at the level is inside. A surface vertex lies on
//! every edge of the tetrahedra (lattice edges, face diagonals and body diagonals) whose ends are on
//! different sides, at the share t = (f_in - L) / (f_in - f_out) of the edge from its inside end (linear
//! interpolation; 0 <= t < 1 since f_in >= L > f_out), kept at least `END_GAP` of the edge from both ends:
//! t is raised to `END_GAP` or lowered to 1 - `END_GAP` where it falls outside. So no vertex lies on a
//! lattice point, two vertices are never at the same place (vertices on edges that meet at a point keep at
//! least END_GAP times the shorter edge times the sine of the edges' angle, at least 35 degrees, between
//! them), and a triangle whose vertices crowd towards one point keeps the shape of the tetrahedron's
//! corner. A sample exactly at the level thus gives a surface a thousandth of an edge beyond it.
//!
//! **Closed by construction.** Every edge of the surface lies in a face of a tetrahedron, and each face
//! whose corners are not all on one side holds one segment, which the two tetrahedra beside the face both
//! use, in opposite directions (`tetrahedra`); a face on the lattice's outer layer has all its corners
//! outside, because the outer layer must be below the level (a field whose outer layer reaches it is
//! refused: the caller pads the field). Each vertex is made once, for its edge, and every triangle that
//! uses the edge names it by the same index. So every edge of the surface lies on exactly two triangles that
//! run along it in opposite directions, whatever the field: the surface passes `Mesh.closed()`. Which
//! corners are inside decides the triangles and their direction; the coordinates only place the vertices.
//!
//! **Order and threads.** Vertices are numbered by the lattice point their edge starts from (the edge's
//! lower end) and then by the edge's offset (x, y, xy, z, xz, yz, xyz); triangles come cell by cell in the
//! order of the points, tetrahedron by tetrahedron in the order of `CELL_TETRAHEDRA`. Every point and every
//! cell is computed from the samples alone, and the pieces are joined in that order, so the vertices, the
//! triangles and their order are bit for bit the same on every thread count.

use crate::error::EngineError;
use crate::tetrahedra::{CELL_TETRAHEDRA, Piece, case_table};
use crate::threads::Threads;

/// The least share of an edge that a vertex keeps from each end of the edge.
pub const END_GAP: f64 = 1e-3;

/// A regular lattice of points: `counts` points along x, y and z, `spacing` mm apart, point (0, 0, 0) at
/// `origin` (mm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    pub counts: [usize; 3],
    pub spacing: f64,
    pub origin: [f64; 3],
}

/// An indexed triangle surface: vertices in mm and triangles as three vertex indices, counter-clockwise
/// seen from outside.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// The offset (0 or 1 along x, y, z) of an edge's upper end from its lower end, by the edge's bits (x 1,
/// y 2, z 4; 1 to 7). The edge of bits b is the point's edge number b - 1.
fn offset(bits: usize) -> [usize; 3] {
    [bits & 1, bits >> 1 & 1, bits >> 2 & 1]
}

/// The surface where `values` (one per point of `lattice`, in its order) equal `level`, on `threads`
/// threads; see the module's notes for the rule. Refused with `invalid_input`: a lattice without points or
/// with a spacing that is not a positive finite number, an origin or a level that is not finite, values of
/// the wrong length or not finite, a sample on the lattice's outer layer that reaches the level (named), and
/// a field that reaches the level nowhere (no surface).
pub fn iso_surface(
    lattice: &Lattice,
    values: &[f64],
    level: f64,
    threads: usize,
) -> Result<Surface, EngineError> {
    check(lattice, values, level)?;
    let threads = Threads::new(threads)?;
    let field = Field {
        lattice,
        values,
        level,
    };
    let n = values.len();
    let masks: Vec<u8> = threads
        .blocks(n, |block| {
            block.map(|p| field.crossing_mask(p)).collect::<Vec<u8>>()
        })
        .concat();
    // The index of the first vertex on each point's edges, in 4 bytes per point (the masks take 1, the
    // samples 8): a surface of more than 4294967295 vertices is refused.
    let mut first = Vec::with_capacity(n);
    let mut count: u64 = 0;
    for mask in &masks {
        let Ok(index) = u32::try_from(count) else {
            return Err(EngineError::invalid_input(
                "the surface would have more than 4294967295 vertices; mesh the field in parts",
            ));
        };
        first.push(index);
        count += u64::from(mask.count_ones());
    }
    if count > u64::from(u32::MAX) {
        return Err(EngineError::invalid_input(
            "the surface would have more than 4294967295 vertices; mesh the field in parts",
        ));
    }
    if count == 0 {
        return Err(EngineError::invalid_input(
            "no sample of the field reaches the level, so there is no surface",
        )
        .with_number("level", level));
    }
    let vertices: Vec<[f64; 3]> = threads
        .blocks(n, |block| {
            let mut out = Vec::new();
            for p in block {
                for bits in 1..8 {
                    if masks[p] >> (bits - 1) & 1 == 1 {
                        out.push(field.vertex(p, bits));
                    }
                }
            }
            out
        })
        .concat();
    let table = case_table();
    let numbering = Numbering {
        masks: &masks,
        first: &first,
    };
    let triangles: Vec<[usize; 3]> = threads
        .blocks(n, |block| {
            let mut out = Vec::new();
            for p in block {
                field.cell_triangles(p, &table, &numbering, &mut out);
            }
            out
        })
        .concat();
    Ok(Surface {
        vertices,
        triangles,
    })
}

fn check(lattice: &Lattice, values: &[f64], level: f64) -> Result<(), EngineError> {
    let [nx, ny, nz] = lattice.counts;
    let Some(n) = nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)) else {
        return Err(EngineError::invalid_input(
            "the lattice has too many points",
        ));
    };
    if n == 0 {
        return Err(EngineError::invalid_input(
            "the lattice needs at least one point along each axis",
        )
        .with_integer("nx", nx)
        .with_integer("ny", ny)
        .with_integer("nz", nz));
    }
    if !(lattice.spacing.is_finite() && lattice.spacing > 0.0) {
        return Err(EngineError::invalid_input(
            "the spacing of the lattice must be a positive finite number (mm)",
        )
        .with_number("spacing", lattice.spacing));
    }
    if !lattice.origin.iter().all(|c| c.is_finite()) {
        return Err(EngineError::invalid_input(
            "the origin of the lattice must be finite (mm)",
        ));
    }
    if !level.is_finite() {
        return Err(
            EngineError::invalid_input("the level must be a finite number")
                .with_number("level", level),
        );
    }
    if values.len() != n {
        return Err(EngineError::invalid_input(format!(
            "the field has {} values; a lattice of {nx} x {ny} x {nz} points needs {n}",
            values.len()
        ))
        .with_integer("values", values.len())
        .with_integer("points", n));
    }
    if let Some(p) = values.iter().position(|v| !v.is_finite()) {
        return Err(EngineError::invalid_input(format!(
            "value {p} of the field is not a finite number"
        ))
        .with_integer("point", p));
    }
    for (p, &value) in values.iter().enumerate() {
        let (i, j, k) = (p % nx, p / nx % ny, p / (nx * ny));
        let outer = i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
        if outer && value >= level {
            return Err(EngineError::invalid_input(format!(
                "the sample at point ({i}, {j}, {k}) on the lattice's outer layer is {value}, at least the \
                 level {level}: every sample of the outer layer must be below the level for the surface to \
                 close (pad the field with a layer below the level)"
            ))
            .with_integer("i", i)
            .with_integer("j", j)
            .with_integer("k", k)
            .with_number("value", value)
            .with_number("level", level));
        }
    }
    Ok(())
}

/// The samples on their lattice, with the level.
struct Field<'a> {
    lattice: &'a Lattice,
    values: &'a [f64],
    level: f64,
}

/// Where the vertex of each crossing edge is in the list of vertices.
struct Numbering<'a> {
    /// Bit b - 1 of `masks[p]` is set when the edge of bits b from point p changes side.
    masks: &'a [u8],
    /// The index of the first vertex on the edges of point p.
    first: &'a [u32],
}

impl Numbering<'_> {
    /// The index of the vertex on the edge of `bits` from point `p` (an edge that changes side).
    fn vertex(&self, p: usize, bits: usize) -> usize {
        let before = self.masks[p] & ((1u8 << (bits - 1)) - 1);
        self.first[p] as usize + before.count_ones() as usize
    }
}

impl Field<'_> {
    fn inside(&self, p: usize) -> bool {
        self.values[p] >= self.level
    }

    /// The point `offset` up from point (i, j, k), or None beyond the lattice.
    fn up(&self, [i, j, k]: [usize; 3], [a, b, c]: [usize; 3]) -> Option<usize> {
        let [nx, ny, nz] = self.lattice.counts;
        let (i, j, k) = (i + a, j + b, k + c);
        (i < nx && j < ny && k < nz).then(|| i + nx * (j + ny * k))
    }

    fn position(&self, p: usize) -> [usize; 3] {
        let [nx, ny, _] = self.lattice.counts;
        [p % nx, p / nx % ny, p / (nx * ny)]
    }

    /// Bit b - 1 set for each edge of bits b from point `p` whose two ends are on different sides.
    fn crossing_mask(&self, p: usize) -> u8 {
        let at = self.position(p);
        let inside = self.inside(p);
        let mut mask = 0;
        for bits in 1..8 {
            if let Some(q) = self.up(at, offset(bits)) {
                if self.inside(q) != inside {
                    mask |= 1 << (bits - 1);
                }
            }
        }
        mask
    }

    /// The vertex on the edge of `bits` from point `p`, by the level rule.
    fn vertex(&self, p: usize, bits: usize) -> [f64; 3] {
        let at = self.position(p);
        let step = offset(bits);
        let q = self
            .up(at, step)
            .expect("a crossing edge ends inside the lattice");
        let (f_p, f_q, level) = (self.values[p], self.values[q], self.level);
        // The share of the edge from its inside end, kept END_GAP from both ends; then measured from p.
        // `max` and `min` (not `clamp`) so that a quotient of two huge numbers that is not a number still
        // gives a share.
        #[allow(clippy::manual_clamp)]
        let from_inside = |f_in: f64, f_out: f64| {
            ((f_in - level) / (f_in - f_out))
                .max(END_GAP)
                .min(1.0 - END_GAP)
        };
        let s = if f_p >= level {
            from_inside(f_p, f_q)
        } else {
            1.0 - from_inside(f_q, f_p)
        };
        let Lattice {
            spacing, origin, ..
        } = *self.lattice;
        [0, 1, 2].map(|axis| origin[axis] + (at[axis] as f64 + s * step[axis] as f64) * spacing)
    }

    /// Appends the triangles of the cell whose lowest corner is point `p` (none when `p` lies on the
    /// lattice's upper faces, where no cell starts, or when the cell's corners are all on one side).
    fn cell_triangles(
        &self,
        p: usize,
        table: &[Piece; 16],
        numbering: &Numbering,
        out: &mut Vec<[usize; 3]>,
    ) {
        let at = self.position(p);
        let mut corners = [0usize; 8];
        for (c, corner) in corners.iter_mut().enumerate() {
            match self.up(at, offset(c)) {
                Some(q) => *corner = q,
                None => return,
            }
        }
        let inside = corners.map(|q| self.inside(q));
        if inside.iter().all(|&s| s == inside[0]) {
            return;
        }
        for tetrahedron in CELL_TETRAHEDRA {
            let case = (0..4).fold(0, |case, q| {
                case | (usize::from(inside[tetrahedron[q]]) << q)
            });
            let piece = &table[case];
            for triangle in &piece.triangles[..piece.count] {
                out.push(triangle.map(|(a, b)| {
                    // The edge between cell corners ca and cb runs up from the one whose offset the other's
                    // holds (the tetrahedron's corners form a chain).
                    let (ca, cb) = (tetrahedron[a], tetrahedron[b]);
                    let (low, high) = if ca & cb == ca { (ca, cb) } else { (cb, ca) };
                    numbering.vertex(corners[low], high ^ low)
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;
    use std::collections::HashMap;

    fn lattice(counts: [usize; 3]) -> Lattice {
        Lattice {
            counts,
            spacing: 0.5,
            origin: [-1.0, 2.0, 0.25],
        }
    }

    /// A field from a function of the point's indices.
    fn field(counts: [usize; 3], mut f: impl FnMut(usize, usize, usize) -> f64) -> Vec<f64> {
        let mut values = Vec::new();
        for k in 0..counts[2] {
            for j in 0..counts[1] {
                for i in 0..counts[0] {
                    values.push(f(i, j, k));
                }
            }
        }
        values
    }

    /// A field of numbers from a small fixed generator (no dependency), each between 0 and 1, zero on
    /// the outer layer.
    fn noise(counts: [usize; 3], seed: u64, values_from: impl Fn(u64) -> f64) -> Vec<f64> {
        let mut state = seed;
        field(counts, |i, j, k| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let outer = i == 0
                || j == 0
                || k == 0
                || i == counts[0] - 1
                || j == counts[1] - 1
                || k == counts[2] - 1;
            if outer { 0.0 } else { values_from(state >> 11) }
        })
    }

    /// The faults `Mesh.closed()` counts, from the triangles alone: edges on one triangle, on more than
    /// two, and edges whose two triangles run along them the same way; and triangles with a repeated vertex.
    fn faults(surface: &Surface) -> (usize, usize, usize, usize) {
        let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
        let mut repeated = 0;
        for t in &surface.triangles {
            if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                repeated += 1;
            }
            for k in 0..3 {
                *directed.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
            }
        }
        let (mut open, mut more, mut same) = (0, 0, 0);
        for (&(a, b), &count) in &directed {
            let back = directed.get(&(b, a)).copied().unwrap_or(0);
            if a < b || back == 0 {
                let total = count + back;
                if total == 1 {
                    open += 1;
                } else if total > 2 {
                    more += 1;
                } else if count == 2 || back == 2 {
                    same += 1;
                }
            }
        }
        (open, more, same, repeated)
    }

    /// The volume the triangles enclose (positive when they face outward).
    fn volume(surface: &Surface) -> f64 {
        let mut six = 0.0;
        for t in &surface.triangles {
            let [a, b, c] = t.map(|v| surface.vertices[v]);
            six += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        six / 6.0
    }

    /// The least of twice a triangle's area over its longest side squared, the measure of
    /// `Mesh.defects()`, and the least distance between two vertices.
    fn shapes(surface: &Surface) -> (f64, f64) {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let norm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let mut least_ratio = f64::INFINITY;
        for t in &surface.triangles {
            let [a, b, c] = t.map(|v| surface.vertices[v]);
            let (u, w) = (sub(b, a), sub(c, a));
            let cross = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            let longest = norm(sub(b, a)).max(norm(sub(c, b))).max(norm(sub(a, c)));
            least_ratio = least_ratio.min(norm(cross) / (longest * longest));
        }
        let mut sorted = surface.vertices.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut least_gap = f64::INFINITY;
        for (n, a) in sorted.iter().enumerate() {
            for b in &sorted[n + 1..] {
                if b[0] - a[0] > least_gap {
                    break;
                }
                least_gap = least_gap.min(norm(sub(*a, *b)));
            }
        }
        (least_ratio, least_gap)
    }

    /// The number of edges of the tetrahedra whose two ends are on different sides, counted directly.
    fn crossing_edges(counts: [usize; 3], values: &[f64], level: f64) -> usize {
        let [nx, ny, nz] = counts;
        let mut count = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    for bits in 1..8 {
                        let [a, b, c] = offset(bits);
                        if i + a < nx && j + b < ny && k + c < nz {
                            let p = i + nx * (j + ny * k);
                            let q = i + a + nx * (j + b + ny * (k + c));
                            if (values[p] >= level) != (values[q] >= level) {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        count
    }

    #[test]
    fn a_single_inside_point_gives_a_closed_octahedron_like_body_facing_outward() {
        let counts = [3, 3, 3];
        let values = field(
            counts,
            |i, j, k| if (i, j, k) == (1, 1, 1) { 1.0 } else { 0.0 },
        );
        let surface = iso_surface(&lattice(counts), &values, 0.5, 1).unwrap();
        // The centre has 14 edges (6 along the axes, 6 face diagonals, 2 body diagonals).
        assert_eq!(surface.vertices.len(), 14);
        assert_eq!(surface.vertices.len(), crossing_edges(counts, &values, 0.5));
        // 24 tetrahedra meet at the centre, one triangle each.
        assert_eq!(surface.triangles.len(), 24);
        assert_eq!(faults(&surface), (0, 0, 0, 0));
        assert!(volume(&surface) > 0.0);
    }

    #[test]
    fn every_vertex_is_made_once_and_random_fields_give_closed_outward_surfaces() {
        for (seed, counts) in [
            (1, [6, 5, 7]),
            (2, [9, 9, 9]),
            (3, [12, 4, 5]),
            (4, [4, 4, 4]),
        ] {
            let values = noise(counts, seed, |r| (r % 1000) as f64 / 1000.0);
            let surface = iso_surface(&lattice(counts), &values, 0.5, 1).unwrap();
            assert_eq!(
                surface.vertices.len(),
                crossing_edges(counts, &values, 0.5),
                "{seed}"
            );
            assert_eq!(faults(&surface), (0, 0, 0, 0), "{seed}");
            // The union of closed outward bodies (and cavities facing inward inside them) has a positive
            // volume.
            assert!(volume(&surface) > 0.0, "{seed}");
            // Every vertex is used.
            let mut used = vec![false; surface.vertices.len()];
            surface
                .triangles
                .iter()
                .flatten()
                .for_each(|&v| used[v] = true);
            assert!(used.iter().all(|&u| u), "{seed}");
        }
    }

    #[test]
    fn samples_exactly_at_the_level_count_as_inside_and_give_no_degenerate_triangle() {
        // Fields of 0, 0.5 and 1 at the level 0.5: many samples exactly at the level; and values one unit
        // in the last place either side of it.
        let level: f64 = 0.5;
        let below = f64::from_bits(level.to_bits() - 1);
        let above = f64::from_bits(level.to_bits() + 1);
        let cases: Vec<(&str, Vec<f64>)> = vec![
            (
                "thirds",
                noise([10, 9, 8], 7, |r| [0.0, 0.5, 1.0][(r % 3) as usize]),
            ),
            (
                "halves",
                noise([10, 9, 8], 8, |r| [0.0, 0.5][(r % 2) as usize]),
            ),
            (
                "ulps",
                noise([10, 9, 8], 9, |r| {
                    [below, level, above, 0.0][(r % 4) as usize]
                }),
            ),
            (
                "sheet at the level",
                field([7, 7, 7], |i, j, k| {
                    if k == 3 && (1..6).contains(&i) && (1..6).contains(&j) {
                        level
                    } else {
                        0.0
                    }
                }),
            ),
        ];
        for (name, values) in cases {
            let counts = if name == "sheet at the level" {
                [7, 7, 7]
            } else {
                [10, 9, 8]
            };
            let surface = iso_surface(&lattice(counts), &values, level, 1).unwrap();
            assert_eq!(faults(&surface), (0, 0, 0, 0), "{name}");
            let (ratio, gap) = shapes(&surface);
            // `Mesh.defects()` counts a triangle as without area at a ratio of 1e-12.
            assert!(ratio > 1e-6, "{name}: {ratio}");
            // At least END_GAP times the spacing times sin(35 degrees) / 2 apart.
            assert!(gap > 0.25 * END_GAP * 0.5, "{name}: {gap}");
        }
    }

    #[test]
    fn a_sample_at_the_level_is_inside_and_vertices_keep_the_end_gap() {
        // One sample at the level, its neighbours below: a small closed body around it, END_GAP of an edge
        // from the sample.
        let counts = [3, 3, 3];
        let values = field(
            counts,
            |i, j, k| if (i, j, k) == (1, 1, 1) { 0.5 } else { 0.0 },
        );
        let lattice = lattice(counts);
        let surface = iso_surface(&lattice, &values, 0.5, 1).unwrap();
        assert_eq!(faults(&surface), (0, 0, 0, 0));
        let centre = [0, 1, 2].map(|a| lattice.origin[a] + lattice.spacing);
        for v in &surface.vertices {
            let d: Vec<f64> = (0..3)
                .map(|a| (v[a] - centre[a]) / lattice.spacing)
                .collect();
            let along = d.iter().map(|x| x.abs()).fold(0.0, f64::max);
            assert!((along - END_GAP).abs() < 1e-15, "{d:?}");
        }
        // Interpolation away from the ends: f = 1 at the centre, 0 around, level 0.25: the vertex lies at
        // three quarters of each edge from the centre.
        let values = field(
            counts,
            |i, j, k| if (i, j, k) == (1, 1, 1) { 1.0 } else { 0.0 },
        );
        let surface = iso_surface(&lattice, &values, 0.25, 1).unwrap();
        for v in &surface.vertices {
            let along = (0..3)
                .map(|a| ((v[a] - centre[a]) / lattice.spacing).abs())
                .fold(0.0, f64::max);
            assert!((along - 0.75).abs() < 1e-12);
        }
    }

    #[test]
    fn the_same_surface_on_every_thread_count() {
        let counts = [40, 37, 33];
        let values = noise(counts, 11, |r| [0.0, 0.5, 1.0, 0.3, 0.7][(r % 5) as usize]);
        let one = iso_surface(&lattice(counts), &values, 0.5, 1).unwrap();
        assert!(one.triangles.len() > 10_000);
        for count in [2, 5, 12] {
            let many = iso_surface(&lattice(counts), &values, 0.5, count).unwrap();
            assert_eq!(many.triangles, one.triangles, "{count}");
            assert!(
                many.vertices
                    .iter()
                    .zip(&one.vertices)
                    .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)),
                "{count}"
            );
            assert_eq!(many.vertices.len(), one.vertices.len());
        }
    }

    #[test]
    fn a_sphere_like_field_gives_one_closed_body_close_to_the_sphere() {
        let counts = [31, 29, 27];
        let lattice = Lattice {
            counts,
            spacing: 1.0,
            origin: [-15.0, -14.0, -13.0],
        };
        let radius = 10.0;
        let values = field(counts, |i, j, k| {
            let p = [
                i as f64 - 15.0 - 0.3,
                j as f64 - 14.0 - 0.2,
                k as f64 - 13.0 - 0.1,
            ];
            radius - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        });
        let surface = iso_surface(&lattice, &values, 0.0, 3).unwrap();
        assert_eq!(faults(&surface), (0, 0, 0, 0));
        let exact = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let v = volume(&surface);
        assert!(v < exact && v > 0.98 * exact, "{v} {exact}");
    }

    #[test]
    fn refusals() {
        let counts = [4, 4, 4];
        let inner = field(counts, |i, j, k| {
            if (1..3).contains(&i) && (1..3).contains(&j) && (1..3).contains(&k) {
                1.0
            } else {
                0.0
            }
        });
        let code = |r: Result<Surface, EngineError>| r.unwrap_err().code;
        // The outer layer reaches the level.
        let mut touching = inner.clone();
        touching[1 + 4 * (2 + 4 * 3)] = 0.5;
        let error = iso_surface(&lattice(counts), &touching, 0.5, 1).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidInput);
        assert!(error.message.contains("(1, 2, 3)"), "{}", error.message);
        // Nothing reaches the level.
        assert_eq!(
            code(iso_surface(&lattice(counts), &inner, 2.0, 1)),
            ErrorCode::InvalidInput
        );
        // Not finite, wrong length, bad lattice, bad level, no threads.
        let mut nan = inner.clone();
        nan[21] = f64::NAN;
        assert_eq!(
            code(iso_surface(&lattice(counts), &nan, 0.5, 1)),
            ErrorCode::InvalidInput
        );
        assert_eq!(
            code(iso_surface(&lattice(counts), &inner[1..], 0.5, 1)),
            ErrorCode::InvalidInput
        );
        assert_eq!(
            code(iso_surface(&lattice([0, 4, 4]), &[], 0.5, 1)),
            ErrorCode::InvalidInput
        );
        let mut flat = lattice(counts);
        flat.spacing = 0.0;
        assert_eq!(
            code(iso_surface(&flat, &inner, 0.5, 1)),
            ErrorCode::InvalidInput
        );
        assert_eq!(
            code(iso_surface(&lattice(counts), &inner, f64::INFINITY, 1)),
            ErrorCode::InvalidInput
        );
        assert_eq!(
            code(iso_surface(&lattice(counts), &inner, 0.5, 0)),
            ErrorCode::InvalidInput
        );
        // Huge values still give vertices inside their edges.
        let huge = field(counts, |i, j, k| {
            if (1..3).contains(&i) && (1..3).contains(&j) && (1..3).contains(&k) {
                f64::MAX
            } else {
                -f64::MAX
            }
        });
        let surface = iso_surface(&lattice(counts), &huge, 0.0, 1).unwrap();
        assert_eq!(faults(&surface), (0, 0, 0, 0));
        assert!(surface.vertices.iter().flatten().all(|c| c.is_finite()));
    }
}
