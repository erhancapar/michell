//! The split of every lattice cell into six tetrahedra, and the piece of an iso-surface inside one
//! tetrahedron: the two building blocks of marching tetrahedra (`iso_surface`).
//!
//! Corner c of a cell (0 to 7) lies at the offset (c & 1, (c >> 1) & 1, (c >> 2) & 1) from the cell's lowest
//! corner: bit 0 is x, bit 1 is y, bit 2 is z.
//!
//! **The split** (Kuhn 1960; Freudenthal 1942). There is one tetrahedron per order of the three axes: the
//! tetrahedron of the order (a, b, c) runs from corner 0 one step along a, then one along b, then one along
//! c to corner 7, so all six share the body diagonal from corner 0 to corner 7. Each face of the cell is cut
//! by them along the diagonal through the face's lowest and highest corners, and the cell on the other side
//! of the face cuts it along the same diagonal, so the tetrahedra of neighbouring cells meet face to face:
//! the tetrahedra of the whole lattice fill it without gaps or overlaps. The corners of each tetrahedron
//! form a chain in which each corner's offset contains the previous one's; every edge of a tetrahedron
//! therefore runs from a corner up by one of the seven offsets x, y, xy, z, xz, yz, xyz.
//!
//! **Orientation.** Each tetrahedron is listed in positive order: with corners (p0, p1, p2, p3), the
//! determinant of (p1 - p0, p2 - p0, p3 - p0) is positive. The path of an even order of the axes is
//! positive as it stands; for an odd order its middle two corners are swapped.
//!
//! **The piece of surface** in one tetrahedron depends only on which of its four corners are inside, 16
//! cases (`case_table`). Its vertices lie on the edges with one end inside and one outside. With one corner
//! inside it is one triangle on the three edges from that corner, with three inside one triangle on the
//! three edges to the corner outside, with two inside a quadrilateral on the four edges between the two
//! pairs, cut in two along a fixed diagonal. The direction of each triangle is read off the case, never
//! computed from coordinates: for a tetrahedron in positive order (q0, q1, q2, q3), the triangle on the
//! edges q0q1, q0q2, q0q3 runs counter-clockwise seen from the side away from q0, and the quadrilateral on
//! q0q2, q0q3, q1q3, q1q2 counter-clockwise seen from the side of q2 and q3. Every case is first turned by an
//! even permutation of the corners (which keeps the order positive) so that it takes one of these forms,
//! with the triangles facing away from the inside corners. A face of a tetrahedron whose corners are not
//! all on one side holds exactly one segment of the surface, between the two of its edges that change
//! side, and the tetrahedron on the other side of the face crosses the same two edges: it holds the same
//! segment, run the other way.

/// The six tetrahedra of a cell, as cell corners, each in positive order.
pub const CELL_TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7], // x, y, z (even)
    [0, 5, 1, 7], // x, z, y (odd: 0, 1, 5, 7 with its middle corners swapped)
    [0, 3, 2, 7], // y, x, z (odd)
    [0, 2, 6, 7], // y, z, x (even)
    [0, 4, 5, 7], // z, x, y (even)
    [0, 6, 4, 7], // z, y, x (odd)
];

/// An edge of a tetrahedron, as two of its corners by their place (0 to 3) in the tetrahedron's list.
pub type Edge = (usize, usize);

/// The piece of surface in one tetrahedron: `count` triangles (0, 1 or 2), each as three edges in the
/// order that faces it away from the inside corners.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Piece {
    pub triangles: [[Edge; 3]; 2],
    pub count: usize,
}

/// The pieces of all 16 cases. Case `c` has corner `q` of the tetrahedron inside where bit `q` of `c` is
/// set.
pub fn case_table() -> [Piece; 16] {
    let mut table = [Piece::default(); 16];
    for (case, piece) in table.iter_mut().enumerate() {
        let inside: Vec<usize> = (0..4).filter(|q| case >> q & 1 == 1).collect();
        let outside: Vec<usize> = (0..4).filter(|q| case >> q & 1 == 0).collect();
        match inside.len() {
            1 => {
                let q = even_order(inside[0], outside[0], outside[1], outside[2]);
                piece.triangles[0] = [(q[0], q[1]), (q[0], q[2]), (q[0], q[3])];
                piece.count = 1;
            }
            3 => {
                // The triangle around the corner outside, run the other way: it faces that corner.
                let q = even_order(outside[0], inside[0], inside[1], inside[2]);
                piece.triangles[0] = [(q[0], q[1]), (q[0], q[3]), (q[0], q[2])];
                piece.count = 1;
            }
            2 => {
                let q = even_order(inside[0], inside[1], outside[0], outside[1]);
                piece.triangles[0] = [(q[0], q[2]), (q[0], q[3]), (q[1], q[3])];
                piece.triangles[1] = [(q[0], q[2]), (q[1], q[3]), (q[1], q[2])];
                piece.count = 2;
            }
            _ => {} // all inside or all outside: no surface
        }
    }
    table
}

/// The order (a, b, c, d), or (a, b, d, c) when that is an odd permutation of (0, 1, 2, 3): an even
/// permutation that starts with a and b.
fn even_order(a: usize, b: usize, c: usize, d: usize) -> [usize; 4] {
    let order = [a, b, c, d];
    let mut inversions = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            if order[i] > order[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        order
    } else {
        [a, b, d, c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner(c: usize) -> [f64; 3] {
        [(c & 1) as f64, (c >> 1 & 1) as f64, (c >> 2 & 1) as f64]
    }

    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn the_six_tetrahedra_are_positive_chains_that_fill_the_cell() {
        let mut volume = 0.0;
        for tetrahedron in CELL_TETRAHEDRA {
            let p = tetrahedron.map(corner);
            let det = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])));
            assert_eq!(det, 1.0, "{tetrahedron:?}");
            volume += det;
            // A chain: every corner's offset holds the offsets of the corners before it.
            let mut sorted = tetrahedron;
            sorted.sort_by_key(|c| c.count_ones());
            assert!(sorted.windows(2).all(|w| w[0] & w[1] == w[0]));
            assert_eq!((sorted[0], sorted[3]), (0, 7));
        }
        // Six tetrahedra of a sixth of the cell each.
        assert_eq!(volume, 6.0);
        // Six different tetrahedra.
        let mut sets: Vec<[usize; 4]> = CELL_TETRAHEDRA
            .iter()
            .map(|t| {
                let mut set = *t;
                set.sort_unstable();
                set
            })
            .collect();
        sets.sort_unstable();
        sets.dedup();
        assert_eq!(sets.len(), 6);
    }

    #[test]
    fn every_face_of_the_cell_is_cut_along_its_diagonal_from_lowest_to_highest_corner() {
        // The triangles the tetrahedra put on the face x = 0 (corners with bit 0 clear) and on x = 1 (bit 0
        // set) must be the same pattern moved by one step along x, so the cells on both sides agree.
        for (axis, bit) in [(0, 1usize), (1, 2), (2, 4)] {
            let mut low = Vec::new();
            let mut high = Vec::new();
            for tetrahedron in CELL_TETRAHEDRA {
                let on_low: Vec<usize> = tetrahedron
                    .iter()
                    .copied()
                    .filter(|c| c & bit == 0)
                    .collect();
                let on_high: Vec<usize> = tetrahedron
                    .iter()
                    .copied()
                    .filter(|c| c & bit != 0)
                    .collect();
                if on_low.len() == 3 {
                    let mut face = on_low.clone();
                    face.sort_unstable();
                    low.push(face);
                }
                if on_high.len() == 3 {
                    let mut face: Vec<usize> = on_high.iter().map(|c| c & !bit).collect();
                    face.sort_unstable();
                    high.push(face);
                }
            }
            low.sort();
            high.sort();
            assert_eq!(low.len(), 2, "axis {axis}");
            assert_eq!(low, high, "axis {axis}");
        }
    }

    /// The point of edge (a, b) of tetrahedron `p` at the share `t` from a.
    fn on_edge(p: &[[f64; 3]; 4], (a, b): Edge, t: f64) -> [f64; 3] {
        let d = sub(p[b], p[a]);
        [p[a][0] + t * d[0], p[a][1] + t * d[1], p[a][2] + t * d[2]]
    }

    #[test]
    fn every_case_faces_away_from_the_inside_corners_in_every_tetrahedron() {
        let table = case_table();
        for tetrahedron in CELL_TETRAHEDRA {
            let p = tetrahedron.map(corner);
            for (case, piece) in table.iter().enumerate() {
                let inside: Vec<usize> = (0..4).filter(|q| case >> q & 1 == 1).collect();
                let expected = match inside.len() {
                    0 | 4 => 0,
                    2 => 2,
                    _ => 1,
                };
                assert_eq!(piece.count, expected, "case {case}");
                // Away from the inside corners: towards the mean of the outside corners minus that of the
                // inside ones.
                let mean = |corners: &mut dyn Iterator<Item = usize>| {
                    let (mut sum, mut n) = ([0.0; 3], 0.0);
                    for q in corners {
                        sum = [sum[0] + p[q][0], sum[1] + p[q][1], sum[2] + p[q][2]];
                        n += 1.0;
                    }
                    [sum[0] / n, sum[1] / n, sum[2] / n]
                };
                if piece.count == 0 {
                    continue;
                }
                let towards_outside = sub(
                    mean(&mut (0..4).filter(|q| case >> q & 1 == 0)),
                    mean(&mut (0..4).filter(|q| case >> q & 1 == 1)),
                );
                for triangle in &piece.triangles[..piece.count] {
                    for edge in triangle {
                        // Each vertex lies on an edge that changes side.
                        assert_ne!(case >> edge.0 & 1, case >> edge.1 & 1, "case {case}");
                    }
                    // At several places along the edges, not only at their midpoints.
                    for t in [0.5, 0.1, 0.9, 0.3] {
                        let v = triangle.map(|edge| on_edge(&p, edge, t));
                        let normal = cross(sub(v[1], v[0]), sub(v[2], v[0]));
                        assert!(
                            dot(normal, towards_outside) > 0.0,
                            "case {case} {tetrahedron:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_segments_of_a_piece_close_up_inside_the_tetrahedron() {
        // Each piece's own edges between two of its vertices: the edge of the quadrilateral's diagonal
        // appears in both directions, and every other appears once; there is one segment per face that
        // changes side.
        for (case, piece) in case_table().iter().enumerate() {
            let mut segments = Vec::new();
            for triangle in &piece.triangles[..piece.count] {
                for k in 0..3 {
                    segments.push((triangle[k], triangle[(k + 1) % 3]));
                }
            }
            let inner: Vec<_> = segments
                .iter()
                .filter(|(a, b)| segments.contains(&(*b, *a)))
                .collect();
            assert_eq!(
                inner.len(),
                if piece.count == 2 { 2 } else { 0 },
                "case {case}"
            );
            let mixed_faces = (0..4)
                .filter(|&left_out| {
                    let face: Vec<usize> = (0..4).filter(|&q| q != left_out).collect();
                    let n_inside = face.iter().filter(|&&q| case >> q & 1 == 1).count();
                    n_inside == 1 || n_inside == 2
                })
                .count();
            assert_eq!(segments.len() - inner.len(), mixed_faces, "case {case}");
        }
    }
}
