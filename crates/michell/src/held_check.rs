//! The check, when a 3D analysis object is built, that every part is held against all six rigid-body
//! motions. A part that is not held has a singular stiffness matrix whatever the density; this check
//! finds it without factoring anything, and does not depend on the density.
//!
//! 1. The active elements are split into bodies: groups joined through shared faces, found by flood fill.
//!    Elements that meet only along an edge or at a corner share no face; such a joint is a hinge (in the
//!    8-node element it carries almost no moment), so they count as separate bodies.
//! 2. A body is held when its held degrees of freedom stop every rigid motion of it. A rigid motion is a
//!    translation t plus a small rotation w: the displacement of a point at d is u = t + w x d. Held
//!    degree of freedom c at a node at d (measured from the body's centre and divided by its size, so the
//!    numbers do not depend on units or on where the body is) stops the motions with
//!    r . (t, w) = 0, where r is the row of six numbers: e_c for t, and component c of (e_a x d) for each
//!    w_a. The motions that no held degree of freedom stops are the null space of G = sum of r r^T, a
//!    6 by 6 matrix: the body is held when G has full rank.
//! 3. Rank is judged from the eigenvalues of G (Jacobi's method, which finds them to the accuracy of the
//!    largest one): held when the smallest is above `RANK_RATIO` times the largest.
//!
//! The rule is stricter than the mathematics in one rare case: a body that is held only through a hinge
//! to another, held body is refused, though the hinge would stop some of its motions.
//!
//! Sources: rigid-body motions of a solid, Cook, Malkus, Plesha and Witt, "Concepts and Applications of
//! Finite Element Analysis", 4th ed. (2002), section 2.9; Jacobi's eigenvalue method, Golub and Van Loan,
//! "Matrix Computations", 4th ed. (2013), section 8.5.

use crate::error::{EngineError, ErrorCode};
use crate::grid3d::Grid3d;

/// A body is held when the smallest eigenvalue of its 6 by 6 matrix G is above this fraction of the
/// largest. Measured with this rule: a beam one element square held on one edge line and on one degree of
/// freedom beside it (barely held) gives 1e-6 at 100 elements long, 1e-9 at 1000 and 1.6e-11 at 4000; a
/// block held at a patch of 2 by 2 nodes gives 1e-6; bodies that are not held give 1e-16 or less, the
/// rounding of G. (The 2D threshold, 1e-10, would refuse the held beam of 4000.)
pub const RANK_RATIO: f64 = 1e-13;

/// Returns `singular_system` for the first body that is not held. `held` has one entry per degree of
/// freedom; only those on nodes of active elements count.
pub fn check_held(grid: &Grid3d, held: &[bool]) -> Result<(), EngineError> {
    let (body, bodies) = bodies(grid);
    // For each node, the last body that counted it, so that a node is counted once per body.
    let mut counted = vec![usize::MAX; grid.n_nodes()];
    let mut members = vec![Vec::new(); bodies];
    for (element, &b) in body.iter().enumerate() {
        if b != usize::MAX {
            members[b].push(element);
        }
    }
    for (b, elements) in members.iter().enumerate() {
        let (centre, size) = centre_and_size(grid, elements);
        let mut g = [[0.0; 6]; 6];
        for &element in elements {
            for node in grid.element_nodes(element) {
                if counted[node] == b {
                    continue;
                }
                counted[node] = b;
                let (i, j, k) = grid.node_position(node);
                let position = [i, j, k].map(|n| n as f64 * grid.h());
                let d = [0, 1, 2].map(|a| (position[a] - centre[a]) / size);
                for c in 0..3 {
                    if held[3 * node + c] {
                        let row = rigid_motion_row(c, d);
                        for (p, g_p) in g.iter_mut().enumerate() {
                            for (q, g_pq) in g_p.iter_mut().enumerate() {
                                *g_pq += row[p] * row[q];
                            }
                        }
                    }
                }
            }
        }
        let eigenvalues = jacobi_eigenvalues(g);
        let largest = eigenvalues.iter().fold(0.0f64, |a, b| a.max(*b));
        let smallest = eigenvalues.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        if !(largest > 0.0 && smallest > RANK_RATIO * largest) {
            let element = elements[0];
            let (i, j, k) = grid.element_position(element);
            return Err(EngineError::new(
                ErrorCode::SingularSystem,
                format!(
                    "a body of {} face-joined elements (one of them element {element}, at i={i}, j={j}, \
                     k={k}) is not held against every rigid motion",
                    elements.len()
                ),
            )
            .with_integer("elements", elements.len())
            .with_integer("element", element)
            .with_number(
                "eigenvalue_ratio",
                if largest > 0.0 { smallest / largest } else { 0.0 },
            ));
        }
    }
    Ok(())
}

/// The body of every active element (inactive elements get `usize::MAX`) and the number of bodies.
/// Bodies are numbered in the order of their lowest element.
fn bodies(grid: &Grid3d) -> (Vec<usize>, usize) {
    let mut body = vec![usize::MAX; grid.n_elements()];
    let mut count = 0;
    let mut stack = Vec::new();
    for start in 0..grid.n_elements() {
        if !grid.is_active(start) || body[start] != usize::MAX {
            continue;
        }
        body[start] = count;
        stack.push(start);
        while let Some(element) = stack.pop() {
            for neighbour in face_neighbours(grid, element).into_iter().flatten() {
                if grid.is_active(neighbour) && body[neighbour] == usize::MAX {
                    body[neighbour] = count;
                    stack.push(neighbour);
                }
            }
        }
        count += 1;
    }
    (body, count)
}

/// The (up to six) elements that share a face with `element`.
fn face_neighbours(grid: &Grid3d, element: usize) -> [Option<usize>; 6] {
    let (i, j, k) = grid.element_position(element);
    let index = |i: usize, j: usize, k: usize| Some(grid.element_index(i, j, k));
    [
        if i > 0 { index(i - 1, j, k) } else { None },
        if i + 1 < grid.nx() {
            index(i + 1, j, k)
        } else {
            None
        },
        if j > 0 { index(i, j - 1, k) } else { None },
        if j + 1 < grid.ny() {
            index(i, j + 1, k)
        } else {
            None
        },
        if k > 0 { index(i, j, k - 1) } else { None },
        if k + 1 < grid.nz() {
            index(i, j, k + 1)
        } else {
            None
        },
    ]
}

/// The mean of the element centres and the longest side of the box around the elements, in mm.
fn centre_and_size(grid: &Grid3d, elements: &[usize]) -> ([f64; 3], f64) {
    let mut sum = [0.0; 3];
    let mut low = [usize::MAX; 3];
    let mut high = [0; 3];
    for &element in elements {
        let (i, j, k) = grid.element_position(element);
        for (a, n) in [i, j, k].into_iter().enumerate() {
            sum[a] += (n as f64 + 0.5) * grid.h();
            low[a] = low[a].min(n);
            high[a] = high[a].max(n + 1);
        }
    }
    let centre = sum.map(|s| s / elements.len() as f64);
    let size = (0..3)
        .map(|a| (high[a] - low[a]) as f64 * grid.h())
        .fold(0.0, f64::max);
    (centre, size)
}

/// The row r of degree of freedom `c` at the (scaled) position `d`: displacement component c of each of
/// the six rigid motions, the translations along x, y, z, then the rotations about x, y, z (e_a x d).
fn rigid_motion_row(c: usize, d: [f64; 3]) -> [f64; 6] {
    let mut row = [0.0; 6];
    row[c] = 1.0;
    // e_x x d = (0, -d_z, d_y), e_y x d = (d_z, 0, -d_x), e_z x d = (-d_y, d_x, 0).
    let rotations = [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]];
    for (a, rotation) in rotations.iter().enumerate() {
        row[3 + a] = rotation[c];
    }
    row
}

/// The eigenvalues of a symmetric 6 by 6 matrix by the cyclic Jacobi method: plane rotations that set one
/// off-diagonal entry to zero at a time, until every off-diagonal entry is negligible.
fn jacobi_eigenvalues(mut a: [[f64; 6]; 6]) -> [f64; 6] {
    for _sweep in 0..100 {
        let off: f64 = (0..6)
            .flat_map(|p| (0..6).map(move |q| (p, q)))
            .filter(|(p, q)| p != q)
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        let diagonal: f64 = (0..6).map(|p| a[p][p] * a[p][p]).sum();
        if off <= 1e-36 * diagonal || off == 0.0 {
            break;
        }
        for p in 0..5 {
            for q in p + 1..6 {
                if a[p][q] == 0.0 {
                    continue;
                }
                // The rotation (c, s) that sets a[p][q] to zero (Golub and Van Loan, algorithm 8.5.1).
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for k in 0..6 {
                    a[p][k] = c * row_p[k] - s * row_q[k];
                    a[q][k] = s * row_p[k] + c * row_q[k];
                }
            }
        }
    }
    [0, 1, 2, 3, 4, 5].map(|p| a[p][p])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hold_node(held: &mut [bool], node: usize) {
        held[3 * node..3 * node + 3].fill(true);
    }

    fn clamped_face(grid: &Grid3d) -> Vec<bool> {
        let mut held = vec![false; grid.n_dofs()];
        for node in 0..grid.n_nodes() {
            if grid.node_position(node).0 == 0 {
                hold_node(&mut held, node);
            }
        }
        held
    }

    #[test]
    fn jacobi_finds_the_eigenvalues_of_a_known_matrix() {
        // Q diag(1..6) Q^T for a rotation Q in the (0, 1) and (2, 5) planes.
        let mut a = [[0.0; 6]; 6];
        for (p, row) in a.iter_mut().enumerate() {
            row[p] = (p + 1) as f64;
        }
        let (c, s) = (0.6, 0.8);
        let (a00, a11) = (a[0][0], a[1][1]);
        a[0][0] = c * c * a00 + s * s * a11;
        a[1][1] = s * s * a00 + c * c * a11;
        a[0][1] = c * s * (a00 - a11);
        a[1][0] = a[0][1];
        let mut values = jacobi_eigenvalues(a);
        values.sort_by(f64::total_cmp);
        for (p, v) in values.iter().enumerate() {
            assert!((v - (p + 1) as f64).abs() < 1e-13);
        }
    }

    #[test]
    fn a_clamped_block_is_held_and_its_absence_or_one_node_is_not() {
        let grid = Grid3d::full(4, 2, 3, 1.0).unwrap();
        assert!(check_held(&grid, &clamped_face(&grid)).is_ok());
        let error = check_held(&grid, &vec![false; grid.n_dofs()]).unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
        let mut one = vec![false; grid.n_dofs()];
        hold_node(&mut one, 0);
        assert!(check_held(&grid, &one).is_err());
        // Three nodes not on one line hold it (each node all three directions).
        let mut three = one.clone();
        hold_node(&mut three, grid.node_index(4, 0, 0));
        hold_node(&mut three, grid.node_index(0, 2, 3));
        assert!(check_held(&grid, &three).is_ok());
    }

    #[test]
    fn a_block_on_an_edge_or_a_loose_second_body_is_not_held() {
        // 4 x 2 x 2: a clamped bar of elements j = 0, k = 0, and one more element at (3, 1, 1), which
        // meets the bar only along an edge.
        let mut active = vec![false; 16];
        active[..4].fill(true);
        active[3 + 4 * (1 + 2)] = true;
        let grid = Grid3d::new(4, 2, 2, 1.0, active).unwrap();
        let held = clamped_face(&grid);
        let error = check_held(&grid, &held).unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
        assert_eq!(
            error.details[0],
            ("elements".to_string(), crate::DetailValue::Integer(1))
        );
        // Joined through a face (element (3, 1, 0) added), the whole part is held.
        let mut joined = grid.active().to_vec();
        joined[3 + 4] = true;
        let grid = Grid3d::new(4, 2, 2, 1.0, joined).unwrap();
        assert!(check_held(&grid, &held).is_ok());
    }

    #[test]
    fn a_long_slender_beam_held_on_a_line_and_one_more_dof_is_held() {
        // 1000 x 1 x 1 held on its edge line y = 0, z = 0 (a hinge: rotation about x is free), then with
        // the z displacement of one node beside the line held too.
        let grid = Grid3d::full(1000, 1, 1, 1.0).unwrap();
        let mut held = vec![false; grid.n_dofs()];
        for i in 0..=1000 {
            hold_node(&mut held, grid.node_index(i, 0, 0));
        }
        assert!(check_held(&grid, &held).is_err());
        held[3 * grid.node_index(1000, 1, 0) + 2] = true;
        assert!(check_held(&grid, &held).is_ok());
    }
}
