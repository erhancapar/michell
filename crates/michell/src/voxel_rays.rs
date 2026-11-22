//! One ray of the voxeliser: where a ray along x crosses a closed triangle mesh, which parts of the ray lie
//! inside it, and how the inside sets of several bodies combine on the ray.
//!
//! **The projection and its lattice.** A ray along x is a point of the (y, z) plane, and a triangle meets it
//! when the point lies in the triangle's projection on that plane. The projection is computed on a lattice:
//! the y and z coordinates of every vertex are measured from the grid's origin in units of
//! h / (2 r 2^20), with h the voxel side and r the rays per side of a voxel, and rounded to whole numbers.
//! The rays of a voxel column sit at the centres of its r x r sub-squares, at odd multiples of 2^20 of these
//! units, so every ray point is exactly a lattice point. Rounding moves a vertex in y and z by at most half
//! a unit, under 1e-7 h for r = 3. On the lattice the test of a point against a triangle is exact integer
//! arithmetic (`i128`), so it cannot be wrong by rounding. The lattice decides only WHICH triangles a ray
//! crosses; WHERE it crosses them is computed from the vertices' own coordinates.
//!
//! **The half-open rule.** A ray may pass exactly through an edge or a vertex that several triangles share.
//! It must then count that crossing once (or, where the surface only touches the ray, twice with opposite
//! signs, or not at all). The rule used is a symbolic perturbation (Edelsbrunner and Mücke, "Simulation of
//! simplicity", ACM Trans. Graphics 9, 1990): every ray point P is treated as the point
//! P' = P + (eps, eps^2) for an infinitely small eps > 0. For an edge from A to B and the edge function
//!
//!   e(P) = du (Pv - Av) - dv (Pu - Au),   du = Bu - Au, dv = Bv - Av,
//!
//! (twice the signed area of A, B, P, with u = y and v = z), e(P') = e(P) - eps dv + eps^2 du, whose sign is
//! that of e(P) when e(P) is not 0, else that of -dv when dv is not 0, else that of du. It is never 0 for an
//! edge of non-zero length, so P' lies on no edge and at no vertex: the ray through P' is a generic ray, and
//! it meets the projected surface only inside triangles. Each triangle is tested on its own against the same
//! P', with exact numbers, so the tests of all triangles agree with one another: P' lies in exactly one of
//! two triangles that share an edge and lie on its two sides, and in exactly one triangle of a fan around a
//! vertex. The sign for the edge from B to A is exactly the opposite (e changes sign, and so do du and dv),
//! so it does not matter which triangle names the edge first. A point on a boundary thereby belongs to the
//! side of increasing y, and on a boundary along y to the side of increasing z.
//!
//! **Crossings and the inside set.** A triangle A, B, C whose projection contains P' is crossed at the x of
//! its plane at P: the barycentric weights of P in the triangle's projection, from the vertices' own y and z
//! (measured from the grid's origin in voxel sides), applied to their x, and the result kept within the
//! triangle's range of x (a triangle nearly parallel to the rays can give weights far outside [0, 1]; one
//! whose own projection has no area takes the weights on the lattice). Its projected
//! area 2 S = cross(B - A, C - A) is the x component of the triangle's normal (times its area), so the ray
//! enters the body there when S < 0 (for an outward-facing triangle) and leaves it when S > 0. Triangles whose
//! projection has no area (parallel to the rays, or degenerate) are never crossed by a generic ray and are
//! dropped. Along the ray the winding number (the sum of +1 for every entry and -1 for every exit) is 0 far
//! away on both sides for a closed body; the ray is inside where it is positive. A winding number other than
//! 0 at the far end means the body is not closed, and the caller refuses it.

/// The lattice is finer than the ray spacing by this power of two (see the module).
pub const LATTICE_BITS: u32 = 20;

/// The largest distance of a vertex from the grid's origin, in lattice units, in y or z: 2^60, so that every
/// product of the edge function fits in an `i128`. (For r = 3 rays per side it is 1.8e11 voxel sides.)
pub const LATTICE_LIMIT: f64 = 1_152_921_504_606_846_976.0;

/// A point of the (y, z) plane on the lattice: u along y, v along z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticePoint {
    pub u: i64,
    pub v: i64,
}

/// The lattice units in one voxel side for `rays_per_side` rays per side: 2 r 2^20.
pub fn units_per_voxel(rays_per_side: usize) -> i64 {
    2 * rays_per_side as i64 * (1i64 << LATTICE_BITS)
}

/// The lattice coordinate of `value` (mm), measured from `origin` (mm) in voxel sides of `h` mm times
/// `units_per_voxel`, rounded to the nearest whole number; `None` when it is not finite or lies beyond
/// `LATTICE_LIMIT`.
pub fn snap(value: f64, origin: f64, h: f64, units_per_voxel: i64) -> Option<i64> {
    let scaled = (value - origin) / h * units_per_voxel as f64;
    if !scaled.is_finite() || scaled.abs() > LATTICE_LIMIT {
        return None;
    }
    // The cast is exact: the rounded value is a whole number well inside the range of i64.
    Some(scaled.round() as i64)
}

/// The ray of sub-square (a, b) of voxel column (j, k), on the lattice: at y = (j + (2 a + 1) / (2 r)) h
/// and z = (k + (2 b + 1) / (2 r)) h from the origin.
pub fn ray_point(j: usize, k: usize, a: usize, b: usize, rays_per_side: usize) -> LatticePoint {
    let r = rays_per_side as i64;
    LatticePoint {
        u: (2 * r * j as i64 + 2 * a as i64 + 1) << LATTICE_BITS,
        v: (2 * r * k as i64 + 2 * b as i64 + 1) << LATTICE_BITS,
    }
}

/// e(P) for the edge from `a` to `b`: twice the signed area of the triangle a, b, p.
fn edge_function(a: LatticePoint, b: LatticePoint, p: LatticePoint) -> i128 {
    let du = i128::from(b.u) - i128::from(a.u);
    let dv = i128::from(b.v) - i128::from(a.v);
    du * (i128::from(p.v) - i128::from(a.v)) - dv * (i128::from(p.u) - i128::from(a.u))
}

/// The sign of e(P') for the edge from `a` to `b`, with P' = P + (eps, eps^2) (see the module): +1 or -1,
/// and 0 only when the edge has no length.
pub fn perturbed_side(a: LatticePoint, b: LatticePoint, p: LatticePoint) -> i32 {
    let e = edge_function(a, b, p);
    if e != 0 {
        return e.signum() as i32;
    }
    let du = i128::from(b.u) - i128::from(a.u);
    let dv = i128::from(b.v) - i128::from(a.v);
    if dv != 0 {
        return (-dv).signum() as i32;
    }
    du.signum() as i32
}

/// One crossing of a ray with a body: where along x, and +1 where the ray enters, -1 where it leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    pub x: f64,
    pub change: i32,
}

/// A triangle prepared for the rays: its corners on the lattice, their own y and z (in voxel sides from the
/// grid's origin), their x in mm, and twice its projected area on the lattice.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedTriangle {
    corners: [LatticePoint; 3],
    yz: [[f64; 2]; 3],
    x: [f64; 3],
    /// cross(B - A, C - A) on the lattice; never 0.
    area2: i128,
}

impl ProjectedTriangle {
    /// The triangle with `corners` (on the lattice), `yz` (the same corners' y and z, unrounded, in voxel
    /// sides from the grid's origin) and `x` (mm), or `None` when its projection on the lattice has no area
    /// (a triangle parallel to the rays, or a degenerate one): no generic ray crosses it.
    ///
    /// The corners are first turned round (keeping their cyclic order, so the orientation) until the one
    /// with the smallest (y, z, x) comes first: the crossing is then computed with the same arithmetic
    /// however the triangle's corners were listed.
    pub fn new(
        corners: [LatticePoint; 3],
        yz: [[f64; 2]; 3],
        x: [f64; 3],
    ) -> Option<ProjectedTriangle> {
        let key = |c: usize| (yz[c][0], yz[c][1], x[c]);
        let before = |a: usize, b: usize| {
            let (p, q) = (key(a), key(b));
            p.0.total_cmp(&q.0)
                .then(p.1.total_cmp(&q.1))
                .then(p.2.total_cmp(&q.2))
                .is_lt()
        };
        let mut first = 0;
        for c in 1..3 {
            if before(c, first) {
                first = c;
            }
        }
        let turn = |values: [usize; 3]| values.map(|c| (c + first) % 3);
        let order = turn([0, 1, 2]);
        let corners = order.map(|c| corners[c]);
        let yz = order.map(|c| yz[c]);
        let x = order.map(|c| x[c]);
        let area2 = edge_function(corners[0], corners[1], corners[2]);
        if area2 == 0 {
            return None;
        }
        Some(ProjectedTriangle {
            corners,
            yz,
            x,
            area2,
        })
    }

    /// The corners on the lattice.
    pub fn corners(&self) -> [LatticePoint; 3] {
        self.corners
    }

    /// Where the ray through `p` (on the lattice; `p_yz`, the same point in voxel sides from the grid's
    /// origin) crosses the triangle, if it does (by the half-open rule of the module).
    pub fn crossing(&self, p: LatticePoint, p_yz: [f64; 2]) -> Option<Crossing> {
        let [a, b, c] = self.corners;
        let orientation = self.area2.signum() as i32;
        if perturbed_side(a, b, p) != orientation
            || perturbed_side(b, c, p) != orientation
            || perturbed_side(c, a, p) != orientation
        {
            return None;
        }
        // The barycentric weights of P (not P': the crossing is the limit as eps goes to 0). Written as
        // x_A + w_B (x_B - x_A) + w_C (x_C - x_A), a triangle of constant x gives exactly that x.
        let (weight_b, weight_c) = self.weights(p_yz).unwrap_or_else(|| {
            let area2 = self.area2 as f64;
            (
                edge_function(c, a, p) as f64 / area2,
                edge_function(a, b, p) as f64 / area2,
            )
        });
        let x = self.x[0] + weight_b * (self.x[1] - self.x[0]) + weight_c * (self.x[2] - self.x[0]);
        let low = self.x[0].min(self.x[1]).min(self.x[2]);
        let high = self.x[0].max(self.x[1]).max(self.x[2]);
        let x = x.clamp(low, high);
        Some(Crossing {
            x,
            change: -orientation,
        })
    }

    /// The barycentric weights (w_B, w_C) of the point `p` (y, z) from the corners' own y and z, or `None`
    /// when the triangle's own projection has no area.
    fn weights(&self, p: [f64; 2]) -> Option<(f64, f64)> {
        let [a, b, c] = self.yz;
        let cross = |s: [f64; 2], e: [f64; 2], q: [f64; 2]| {
            (e[0] - s[0]) * (q[1] - s[1]) - (e[1] - s[1]) * (q[0] - s[0])
        };
        let area2 = cross(a, b, c);
        if area2 == 0.0 || !area2.is_finite() {
            return None;
        }
        Some((cross(c, a, p) / area2, cross(a, b, p) / area2))
    }
}

/// Sorts `crossings` along x (an entry and an exit at the same x: the exit first) and writes the intervals
/// of x where the winding number is positive into `inside`, in order, without the empty ones. Returns
/// `Err(winding)` when the winding number at the far end is not 0 (the body is not closed).
pub fn inside_intervals(
    crossings: &mut [Crossing],
    inside: &mut Vec<(f64, f64)>,
) -> Result<(), i32> {
    crossings.sort_by(|p, q| p.x.total_cmp(&q.x).then(p.change.cmp(&q.change)));
    let mut winding = 0;
    let mut start = 0.0;
    for crossing in crossings.iter() {
        let before = winding;
        winding += crossing.change;
        if before <= 0 && winding > 0 {
            start = crossing.x;
        } else if before > 0 && winding <= 0 && crossing.x > start {
            inside.push((start, crossing.x));
        }
    }
    if winding != 0 {
        return Err(winding);
    }
    Ok(())
}

/// Turns `intervals` (in any order, possibly overlapping) into their union: sorted, disjoint, and with no
/// two touching.
pub fn union(intervals: &mut Vec<(f64, f64)>) {
    intervals.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
    let mut merged = 0;
    for index in 0..intervals.len() {
        let (start, end) = intervals[index];
        if merged > 0 && start <= intervals[merged - 1].1 {
            if end > intervals[merged - 1].1 {
                intervals[merged - 1].1 = end;
            }
        } else {
            intervals[merged] = (start, end);
            merged += 1;
        }
    }
    intervals.truncate(merged);
}

/// Writes `kept` minus `removed` into `out`; both must be sorted and disjoint (as `union` leaves them), and
/// so is the result.
pub fn subtract(kept: &[(f64, f64)], removed: &[(f64, f64)], out: &mut Vec<(f64, f64)>) {
    let mut first = 0;
    for &(low, high) in kept {
        // Removed intervals that end before this one starts end before every later one too.
        while first < removed.len() && removed[first].1 <= low {
            first += 1;
        }
        let mut start = low;
        for &(cut_low, cut_high) in &removed[first..] {
            if cut_low >= high {
                break;
            }
            if cut_low > start {
                out.push((start, cut_low));
            }
            if cut_high > start {
                start = cut_high;
            }
            if start >= high {
                break;
            }
        }
        if start < high {
            out.push((start, high));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(u: i64, v: i64) -> LatticePoint {
        LatticePoint { u, v }
    }

    /// A point of the lattice as unrounded coordinates (lattice units taken as voxel sides, for the tests).
    fn yz(p: LatticePoint) -> [f64; 2] {
        [p.u as f64, p.v as f64]
    }

    /// A triangle whose own coordinates are its lattice coordinates.
    fn on_lattice(corners: [LatticePoint; 3], x: [f64; 3]) -> Option<ProjectedTriangle> {
        ProjectedTriangle::new(corners, corners.map(yz), x)
    }

    /// How many of `triangles` the ray through `p` crosses, with the signed sum of their changes.
    fn hits(triangles: &[ProjectedTriangle], p: LatticePoint) -> (usize, i32) {
        let crossings: Vec<Crossing> = triangles
            .iter()
            .filter_map(|t| t.crossing(p, yz(p)))
            .collect();
        (crossings.len(), crossings.iter().map(|c| c.change).sum())
    }

    fn triangle(a: LatticePoint, b: LatticePoint, c: LatticePoint) -> ProjectedTriangle {
        on_lattice([a, b, c], [0.0; 3]).unwrap()
    }

    #[test]
    fn the_perturbed_side_is_never_zero_and_flips_with_the_edge() {
        let corners = [
            point(0, 0),
            point(4, 0),
            point(4, 4),
            point(0, 4),
            point(2, 2),
        ];
        for &a in &corners {
            for &b in &corners {
                if a == b {
                    continue;
                }
                for u in -1..=5 {
                    for v in -1..=5 {
                        let p = point(u, v);
                        let side = perturbed_side(a, b, p);
                        assert!(side == 1 || side == -1);
                        assert_eq!(side, -perturbed_side(b, a, p));
                    }
                }
            }
        }
    }

    #[test]
    fn a_ray_on_a_shared_edge_or_vertex_counts_once() {
        // A square of side 4 cut into 8 triangles around its centre (2, 2) and through the midpoints of
        // its sides; every lattice point of the closed square, corners, edges and the centre included, is
        // tried, with both orientations of the whole fan.
        let ring = [
            point(0, 0),
            point(2, 0),
            point(4, 0),
            point(4, 2),
            point(4, 4),
            point(2, 4),
            point(0, 4),
            point(0, 2),
        ];
        let centre = point(2, 2);
        for flip in [false, true] {
            let fan: Vec<ProjectedTriangle> = (0..8)
                .map(|i| {
                    let (a, b) = (ring[i], ring[(i + 1) % 8]);
                    if flip {
                        triangle(centre, b, a)
                    } else {
                        triangle(centre, a, b)
                    }
                })
                .collect();
            for u in -1..=5 {
                for v in -1..=5 {
                    let (count, _) = hits(&fan, point(u, v));
                    // Half-open: the square [0, 4) x [0, 4) holds the perturbed point.
                    let inside = (0..4).contains(&u) && (0..4).contains(&v);
                    assert_eq!(count, usize::from(inside), "({u}, {v}), flipped {flip}");
                }
            }
        }
    }

    #[test]
    fn the_rule_agrees_with_a_ray_moved_off_every_edge() {
        // On a lattice of spacing M = 2^20 (the spacing of the rays), the ray moved by (K, 1) units with
        // 1 << K << M meets no edge, and the decision of the moved ray must match the rule's decision at
        // the ray itself, for every triangle with corners on a 5 x 5 patch of the coarse lattice.
        let m = 1i64 << LATTICE_BITS;
        let k = 1i64 << 10;
        let coarse: Vec<LatticePoint> = (0..5)
            .flat_map(|i| (0..5).map(move |j| point(i * m, j * m)))
            .collect();
        let mut tried = 0;
        for (index_a, &a) in coarse.iter().enumerate() {
            for (index_b, &b) in coarse.iter().enumerate().skip(index_a + 1) {
                for &c in coarse.iter().skip(index_b + 1) {
                    let Some(t) = on_lattice([a, b, c], [0.0; 3]) else {
                        continue;
                    };
                    for &p in &coarse {
                        let moved = point(p.u + k, p.v + 1);
                        let orientation = t.area2.signum();
                        let strictly_inside = [(a, b), (b, c), (c, a)]
                            .iter()
                            .all(|&(s, e)| edge_function(s, e, moved).signum() == orientation);
                        assert_eq!(t.crossing(p, yz(p)).is_some(), strictly_inside);
                        tried += 1;
                    }
                }
            }
        }
        assert!(tried > 40_000);
    }

    #[test]
    fn the_crossing_lies_on_the_plane_and_its_sign_follows_the_normal() {
        // The plane x = 1 + 0.5 y + 0.25 z over a triangle, with lattice units of 1 mm.
        let corners = [point(0, 0), point(8, 0), point(0, 8)];
        let x = corners.map(|c| 1.0 + 0.5 * c.u as f64 + 0.25 * c.v as f64);
        let t = on_lattice(corners, x).unwrap();
        let p = point(2, 4);
        let crossing = t.crossing(p, yz(p)).unwrap();
        assert_eq!(crossing.x, 1.0 + 1.0 + 1.0);
        // Counter-clockwise in (y, z): the normal points along +x, the ray leaves the body.
        assert_eq!(crossing.change, -1);
        let reversed = on_lattice([corners[0], corners[2], corners[1]], x).unwrap();
        assert_eq!(reversed.crossing(p, yz(p)).unwrap().change, 1);
        // A triangle of constant x gives that x exactly.
        let flat = on_lattice(corners, [0.3; 3]).unwrap();
        assert_eq!(flat.crossing(point(1, 3), [1.0, 3.0]).unwrap().x, 0.3);
        // The crossing is taken from the corners' own coordinates, not from the lattice: here the own
        // corners lie a quarter unit off the lattice ones, and the crossing follows them.
        let own = [[0.25, 0.0], [8.25, 0.0], [0.25, 8.0]];
        let shifted = ProjectedTriangle::new(corners, own, x).unwrap();
        assert_eq!(shifted.crossing(p, yz(p)).unwrap().x, 3.0 - 0.125);
        // A triangle nearly parallel to the rays: the crossing stays within its range of x.
        let thin =
            ProjectedTriangle::new(corners, [[0.0, 0.0], [1e-300, 0.0], [0.0, 1e-300]], x).unwrap();
        let far = thin.crossing(p, yz(p)).unwrap().x;
        assert!((1.0..=5.0).contains(&far));
        // A triangle parallel to the rays is dropped.
        assert!(on_lattice([point(0, 0), point(0, 0), point(0, 8)], x).is_none());
        assert!(on_lattice([point(0, 0), point(4, 4), point(8, 8)], x).is_none());
    }

    #[test]
    fn the_crossing_does_not_depend_on_which_corner_is_listed_first() {
        let corners = [point(0, 0), point(9, 1), point(2, 7)];
        let x = [0.1, 1.7, -0.9];
        let own = [[0.013, 0.0], [9.0, 1.0], [2.0, 7.03]];
        let p = point(3, 3);
        let reference = ProjectedTriangle::new(corners, own, x)
            .unwrap()
            .crossing(p, [3.0, 3.0])
            .unwrap();
        for first in 1..3 {
            let turn = |c: usize| (c + first) % 3;
            let t = ProjectedTriangle::new(
                [0, 1, 2].map(|c| corners[turn(c)]),
                [0, 1, 2].map(|c| own[turn(c)]),
                [0, 1, 2].map(|c| x[turn(c)]),
            )
            .unwrap();
            let crossing = t.crossing(p, [3.0, 3.0]).unwrap();
            assert_eq!(crossing.x.to_bits(), reference.x.to_bits());
            assert_eq!(crossing.change, reference.change);
        }
    }

    #[test]
    fn inside_intervals_follow_the_winding_number() {
        let c = |x: f64, change: i32| Crossing { x, change };
        let mut inside = Vec::new();
        // Two overlapping bodies in one mesh, in any order: winding 1, 2, 1, 0.
        let mut crossings = vec![c(3.0, -1), c(0.0, 1), c(5.0, -1), c(1.0, 1)];
        inside_intervals(&mut crossings, &mut inside).unwrap();
        assert_eq!(inside, vec![(0.0, 5.0)]);
        // A surface that touches the ray: entry and exit at the same x give nothing.
        inside.clear();
        let mut touching = vec![c(2.0, 1), c(2.0, -1)];
        inside_intervals(&mut touching, &mut inside).unwrap();
        assert!(inside.is_empty());
        // Two bodies that meet at x = 1: the exit is taken first, two intervals.
        inside.clear();
        let mut meeting = vec![c(0.0, 1), c(1.0, 1), c(1.0, -1), c(2.0, -1)];
        inside_intervals(&mut meeting, &mut inside).unwrap();
        assert_eq!(inside, vec![(0.0, 1.0), (1.0, 2.0)]);
        // An open body.
        let mut open = vec![c(0.0, 1)];
        assert_eq!(inside_intervals(&mut open, &mut inside), Err(1));
    }

    #[test]
    fn union_and_subtraction_of_intervals() {
        let mut intervals = vec![(4.0, 6.0), (0.0, 1.0), (1.0, 2.0), (5.0, 7.0), (8.0, 9.0)];
        union(&mut intervals);
        assert_eq!(intervals, vec![(0.0, 2.0), (4.0, 7.0), (8.0, 9.0)]);
        let removed = [(-1.0, 0.5), (1.0, 1.5), (3.0, 4.5), (6.0, 8.5)];
        let mut out = Vec::new();
        subtract(&intervals, &removed, &mut out);
        assert_eq!(out, vec![(0.5, 1.0), (1.5, 2.0), (4.5, 6.0), (8.5, 9.0)]);
        out.clear();
        subtract(&intervals, &[], &mut out);
        assert_eq!(out, intervals);
        out.clear();
        subtract(&intervals, &[(-10.0, 10.0)], &mut out);
        assert!(out.is_empty());
        out.clear();
        subtract(&[(0.0, 10.0)], &[(2.0, 3.0), (5.0, 6.0)], &mut out);
        assert_eq!(out, vec![(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]);
    }

    #[test]
    fn snapping_refuses_what_does_not_fit() {
        let units = units_per_voxel(3);
        assert_eq!(units, 6 << LATTICE_BITS);
        assert_eq!(snap(2.5, 0.5, 0.5, units), Some(4 * units));
        assert_eq!(snap(f64::NAN, 0.0, 1.0, units), None);
        assert_eq!(snap(1e300, 0.0, 1.0, units), None);
        let ray = ray_point(2, 0, 1, 2, 3);
        assert_eq!(ray.u, (12 + 3) << LATTICE_BITS);
        assert_eq!(ray.v, 5 << LATTICE_BITS);
    }
}
