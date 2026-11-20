//! The transfer of a vector between two levels of the multigrid: interpolation from the coarser grid to
//! the finer one (prolongation, P), and its transpose (restriction, P^T).
//!
//! Interpolation is trilinear and node based, the same for the x, y and z displacement. In one direction
//! that is halved, fine node i takes the value of coarse node i / 2 when i is even, and half of each of its
//! two neighbours (i - 1) / 2 and (i + 1) / 2 when i is odd: the coarse element of side 2h, interpolated
//! linearly at its own nodes and at its middle. In a direction that is not halved (one element across),
//! fine node i takes coarse node i. The weight of a fine node in 3D is the product of its three weights.
//! At the far edge of a direction with an odd number of elements, the last fine node is odd: it lies in
//! the middle of the last coarse element, whose far node lies outside the fine grid.
//!
//! Degrees of freedom that are fixed (held, or touching no active element) are left out on both sides:
//! interpolation never moves a fixed fine degree of freedom, and a fixed coarse degree of freedom (one
//! that no free fine degree of freedom takes a value from; see `multigrid_levels`) passes nothing on.
//! Restriction is exactly the transpose of that interpolation, P^T.
//!
//! Both run in gather form, like the product of `elasticity3d`: each output node collects from the input
//! nodes that concern it and writes only its own three entries, so the work is shared out over the
//! threads without two threads writing the same entry, and the result does not depend on the thread
//! count.
//!
//! Source: Trottenberg, Oosterlee and Schueller, "Multigrid" (2001), section 2.3 (linear interpolation and
//! its transpose, full weighting) and section 2.7 (Galerkin coarse operators).

use crate::grid3d::{Grid3d, NODE_STEPS};
use crate::threads::Threads;

/// Whether a direction of `n` elements is halved on the next coarser level: every direction with more
/// than one element is.
pub fn halves(n: usize) -> bool {
    n > 1
}

/// The number of elements, on the next coarser level, of a direction that has `n`: n / 2 rounded up when
/// it is halved (an odd count leaves a last coarse element with one child in that direction).
pub fn coarser_count(n: usize) -> usize {
    if halves(n) { n.div_ceil(2) } else { n }
}

/// The weight with which coarse node `coarse` enters fine node `fine`, both counted along one direction.
pub fn weight_1d(fine: usize, coarse: usize, halved: bool) -> f64 {
    if !halved {
        return if fine == coarse { 1.0 } else { 0.0 };
    }
    match fine.abs_diff(2 * coarse) {
        0 => 1.0,
        1 => 0.5,
        _ => 0.0,
    }
}

/// Along one direction: the coarse nodes that fine node `fine` takes its value from, with their weights.
/// The first `count` entries are used (1 or 2).
fn parents_1d(fine: usize, halved: bool) -> ([(usize, f64); 2], usize) {
    if !halved {
        ([(fine, 1.0), (0, 0.0)], 1)
    } else if fine % 2 == 0 {
        ([(fine / 2, 1.0), (0, 0.0)], 1)
    } else {
        ([((fine - 1) / 2, 0.5), (fine.div_ceil(2), 0.5)], 2)
    }
}

/// Along one direction: the fine nodes that take a value from coarse node `coarse`, with their weights,
/// on a direction of `fine_nodes` fine nodes. The first `count` entries are used (1 to 3).
fn children_1d(coarse: usize, halved: bool, fine_nodes: usize) -> ([(usize, f64); 3], usize) {
    let mut found = [(0, 0.0); 3];
    let mut count = 0;
    if !halved {
        found[0] = (coarse, 1.0);
        return (found, 1);
    }
    let centre = 2 * coarse;
    for fine in [centre.wrapping_sub(1), centre, centre + 1] {
        // `wrapping_sub` turns -1 into a huge number, which the range check below drops.
        if fine < fine_nodes {
            found[count] = (fine, weight_1d(fine, coarse, true));
            count += 1;
        }
    }
    (found, count)
}

/// Whether each direction of `fine` (x, y, z) is halved on the next level.
pub fn halved_directions(fine: &Grid3d) -> [bool; 3] {
    [halves(fine.nx()), halves(fine.ny()), halves(fine.nz())]
}

/// The interpolation inside one coarse element, at the level of nodes: entry `[a][b]` is the weight of the
/// coarse element's node b in the node a of its child at `position` (0 or 1 per direction: which half of
/// the coarse element the child fills; always 0 in a direction that is not halved). Nodes in the order of
/// `NODE_STEPS`. The interpolation of the 24 degrees of freedom is this, the same for x, y and z.
pub fn local_weights(position: [usize; 3], halved: [bool; 3]) -> [[f64; 8]; 8] {
    let mut weights = [[0.0; 8]; 8];
    for (a, fine_steps) in NODE_STEPS.iter().enumerate() {
        for (b, coarse_steps) in NODE_STEPS.iter().enumerate() {
            let mut weight = 1.0;
            for d in 0..3 {
                // Fine node a sits at fine index 2I + position + step; coarse node b at coarse index I + step,
                // so their distance, counted in fine steps, does not depend on I.
                let fine = position[d] + fine_steps[d];
                weight *= weight_1d(fine, coarse_steps[d], halved[d]);
            }
            weights[a][b] = weight;
        }
    }
    weights
}

/// The interpolation between two neighbouring levels, with the fixed degrees of freedom of both.
pub struct Transfer<'a> {
    pub fine: &'a Grid3d,
    pub fine_fixed: &'a [bool],
    pub coarse: &'a Grid3d,
    pub coarse_fixed: &'a [bool],
}

impl Transfer<'_> {
    /// fine_out = P coarse_x: each fine node gathers from its (at most eight) coarse parents. Fixed fine
    /// degrees of freedom get 0; fixed coarse ones pass nothing on.
    pub fn prolong(&self, threads: &Threads, coarse_x: &[f64], fine_out: &mut [f64]) {
        let halved = halved_directions(self.fine);
        threads.for_each_piece(fine_out, 3, |node, values| {
            let (i, j, k) = self.fine.node_position(node);
            let (px, nx) = parents_1d(i, halved[0]);
            let (py, ny) = parents_1d(j, halved[1]);
            let (pz, nz) = parents_1d(k, halved[2]);
            let mut sum = [0.0; 3];
            for &(ck, wk) in &pz[..nz] {
                for &(cj, wj) in &py[..ny] {
                    for &(ci, wi) in &px[..nx] {
                        let parent = self.coarse.node_index(ci, cj, ck);
                        let weight = wi * wj * wk;
                        for (c, entry) in sum.iter_mut().enumerate() {
                            if !self.coarse_fixed[3 * parent + c] {
                                *entry += weight * coarse_x[3 * parent + c];
                            }
                        }
                    }
                }
            }
            for (c, value) in values.iter_mut().enumerate() {
                *value = if self.fine_fixed[3 * node + c] {
                    0.0
                } else {
                    sum[c]
                };
            }
        });
    }

    /// coarse_out = P^T fine_r: each coarse node gathers from the (at most 27) fine nodes that take a
    /// value from it, with the same weights. Fixed coarse degrees of freedom get 0; fixed fine ones pass
    /// nothing on.
    pub fn restrict(&self, threads: &Threads, fine_r: &[f64], coarse_out: &mut [f64]) {
        let halved = halved_directions(self.fine);
        let (fine_x, fine_y, fine_z) = (self.fine.nx() + 1, self.fine.ny() + 1, self.fine.nz() + 1);
        threads.for_each_piece(coarse_out, 3, |node, values| {
            let (i, j, k) = self.coarse.node_position(node);
            let (cx, nx) = children_1d(i, halved[0], fine_x);
            let (cy, ny) = children_1d(j, halved[1], fine_y);
            let (cz, nz) = children_1d(k, halved[2], fine_z);
            let mut sum = [0.0; 3];
            for &(fk, wk) in &cz[..nz] {
                for &(fj, wj) in &cy[..ny] {
                    for &(fi, wi) in &cx[..nx] {
                        let child = self.fine.node_index(fi, fj, fk);
                        let weight = wi * wj * wk;
                        for (c, entry) in sum.iter_mut().enumerate() {
                            if !self.fine_fixed[3 * child + c] {
                                *entry += weight * fine_r[3 * child + c];
                            }
                        }
                    }
                }
            }
            for (c, value) in values.iter_mut().enumerate() {
                *value = if self.coarse_fixed[3 * node + c] {
                    0.0
                } else {
                    sum[c]
                };
            }
        });
    }
}

/// Whether coarse degree of freedom `dof` passes a value to at least one free fine degree of freedom:
/// the rule that decides which coarse degrees of freedom are free (`multigrid_levels`).
pub fn reaches_a_free_fine_dof(
    fine: &Grid3d,
    fine_fixed: &[bool],
    coarse: &Grid3d,
    dof: usize,
) -> bool {
    let halved = halved_directions(fine);
    let (node, c) = (dof / 3, dof % 3);
    let (i, j, k) = coarse.node_position(node);
    let (cx, nx) = children_1d(i, halved[0], fine.nx() + 1);
    let (cy, ny) = children_1d(j, halved[1], fine.ny() + 1);
    let (cz, nz) = children_1d(k, halved[2], fine.nz() + 1);
    for &(fk, _) in &cz[..nz] {
        for &(fj, _) in &cy[..ny] {
            for &(fi, _) in &cx[..nx] {
                if !fine_fixed[3 * fine.node_index(fi, fj, fk) + c] {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_halve_and_round_up_and_one_element_stays() {
        assert_eq!([1, 2, 3, 4, 5, 48].map(coarser_count), [1, 1, 2, 2, 3, 24]);
    }

    #[test]
    fn local_weights_reproduce_a_linear_field_and_each_row_sums_to_one() {
        // A linear field on the coarse element (side 2) must give the same field at the child's nodes.
        let field = |x: f64, y: f64, z: f64| 1.0 + 2.0 * x - 3.0 * y + 0.5 * z;
        for halved in [[true, true, true], [true, false, true]] {
            for position in [[0, 0, 0], [1, 0, 1], [1, 0, 0]] {
                let w = local_weights(position, halved);
                for (a, fine_steps) in NODE_STEPS.iter().enumerate() {
                    let coordinate = |d: usize| {
                        let fine = (position[d] + fine_steps[d]) as f64;
                        if halved[d] { fine / 2.0 } else { fine }
                    };
                    let mut value = 0.0;
                    for (b, s) in NODE_STEPS.iter().enumerate() {
                        value += w[a][b] * field(s[0] as f64, s[1] as f64, s[2] as f64);
                    }
                    let expected = field(coordinate(0), coordinate(1), coordinate(2));
                    assert!((value - expected).abs() < 1e-14);
                    assert!((w[a].iter().sum::<f64>() - 1.0).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn restriction_is_the_transpose_of_prolongation() {
        // 5 x 1 x 4 fine elements (odd, one, even), some fixed degrees of freedom on both levels.
        let fine = Grid3d::full(5, 1, 4, 1.0).unwrap();
        let coarse = Grid3d::full(3, 1, 2, 1.0).unwrap();
        let fine_fixed: Vec<bool> = (0..fine.n_dofs()).map(|d| d % 7 == 0).collect();
        let coarse_fixed: Vec<bool> = (0..coarse.n_dofs()).map(|d| d % 5 == 1).collect();
        let transfer = Transfer {
            fine: &fine,
            fine_fixed: &fine_fixed,
            coarse: &coarse,
            coarse_fixed: &coarse_fixed,
        };
        let threads = Threads::new(1).unwrap();
        let x: Vec<f64> = (0..coarse.n_dofs())
            .map(|d| (0.7 * d as f64).sin())
            .collect();
        let y: Vec<f64> = (0..fine.n_dofs()).map(|d| (0.3 * d as f64).cos()).collect();
        let mut px = vec![0.0; fine.n_dofs()];
        let mut pty = vec![0.0; coarse.n_dofs()];
        transfer.prolong(&threads, &x, &mut px);
        transfer.restrict(&threads, &y, &mut pty);
        let left: f64 = px.iter().zip(&y).map(|(a, b)| a * b).sum();
        let right: f64 = x.iter().zip(&pty).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-12 * left.abs().max(1.0));
        // Fixed entries stay 0 on both sides.
        assert!(fine_fixed.iter().zip(&px).all(|(f, v)| !*f || *v == 0.0));
        assert!(coarse_fixed.iter().zip(&pty).all(|(f, v)| !*f || *v == 0.0));
        // The last fine node of the odd direction (i = 5) lies halfway between coarse nodes 2 and 3.
        let free_fixed = vec![false; fine.n_dofs()];
        let free_coarse = vec![false; coarse.n_dofs()];
        let open = Transfer {
            fine: &fine,
            fine_fixed: &free_fixed,
            coarse: &coarse,
            coarse_fixed: &free_coarse,
        };
        let mut ones = vec![0.0; coarse.n_dofs()];
        ones[3 * coarse.node_index(3, 0, 0)] = 1.0;
        open.prolong(&threads, &ones, &mut px);
        assert_eq!(px[3 * fine.node_index(5, 0, 0)], 0.5);
        assert_eq!(px[3 * fine.node_index(4, 0, 0)], 0.0);
    }
}
