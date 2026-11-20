//! The levels of the multigrid: the grids from the finest (the analysis grid) to the coarsest, which of
//! their elements are active and which of their degrees of freedom are fixed. None of this depends on
//! the density, so it is built once, with the analysis object.
//!
//! - Each coarser level halves the grid: a direction of n elements gets n / 2 rounded up, so a coarse
//!   element at the far edge of an odd direction has one child in that direction instead of two. A
//!   direction that is down to one element is not halved again (its two node layers map one to one),
//!   while the other directions keep halving: a thin plate does not stop the hierarchy early. The fine
//!   grid is never padded.
//! - A coarse element is active when at least one of its children is.
//! - A coarse degree of freedom is free when at least one free fine degree of freedom takes a value from
//!   it by the interpolation (`multigrid_transfer`); otherwise it is fixed, like a held one. These are
//!   exactly the coarse degrees of freedom whose row of the coarse matrix P^T K P would be empty.
//! - Levels are added until the coarsest has at most `COARSEST_MAX_DOFS` degrees of freedom (it is then
//!   solved directly) or no direction can be halved any more.
//!
//! Numbering on every level is that of `grid3d`; a level is described by a `Grid3d` of its own. The size
//! `h` of a coarse level's grid is that of the finest and is not used: the coarse matrices come from the
//! Galerkin product (`multigrid_coarse`), not from an element of a given size.

use crate::error::EngineError;
use crate::grid3d::Grid3d;
use crate::multigrid_transfer::{
    coarser_count, halved_directions, halves, reaches_a_free_fine_dof,
};

/// The coarsest level holds at most about this many degrees of freedom: the levels stop at the first one
/// at or below it. Its matrix is factored by the banded Cholesky solver on every solve.
pub const COARSEST_MAX_DOFS: usize = 3000;

/// One level of the multigrid.
#[derive(Clone, Debug)]
pub struct Level {
    /// The grid of this level, with its active elements.
    pub grid: Grid3d,
    /// One entry per degree of freedom: held, touching no active element, or (on a coarse level)
    /// passing no value to a free degree of freedom of the finer level.
    pub fixed: Vec<bool>,
}

/// The levels, finest first. `fine_fixed` is the fixed set of the analysis (held or touching no active
/// element).
pub fn build_levels(fine: &Grid3d, fine_fixed: &[bool]) -> Result<Vec<Level>, EngineError> {
    let mut levels = vec![Level {
        grid: fine.clone(),
        fixed: fine_fixed.to_vec(),
    }];
    loop {
        let finer = &levels[levels.len() - 1];
        let can_halve =
            halves(finer.grid.nx()) || halves(finer.grid.ny()) || halves(finer.grid.nz());
        if finer.grid.n_dofs() <= COARSEST_MAX_DOFS || !can_halve {
            return Ok(levels);
        }
        let grid = coarser_grid(&finer.grid)?;
        let fixed = (0..grid.n_dofs())
            .map(|dof| !reaches_a_free_fine_dof(&finer.grid, &finer.fixed, &grid, dof))
            .collect();
        levels.push(Level { grid, fixed });
    }
}

/// The grid of the next coarser level, its elements active where at least one child is.
fn coarser_grid(fine: &Grid3d) -> Result<Grid3d, EngineError> {
    let (nx, ny, nz) = (
        coarser_count(fine.nx()),
        coarser_count(fine.ny()),
        coarser_count(fine.nz()),
    );
    let mut coarse = Grid3d::full(nx, ny, nz, fine.h())?;
    let active = (0..coarse.n_elements())
        .map(|element| {
            let (found, count) = children(fine, &coarse, element);
            found[..count]
                .iter()
                .any(|(child, _)| fine.is_active(*child))
        })
        .collect();
    coarse = Grid3d::new(nx, ny, nz, fine.h(), active)?;
    Ok(coarse)
}

/// The fine elements inside coarse element `coarse_element` (active or not), each with its position in
/// it: 0 or 1 per direction (which half), always 0 in a direction that is not halved. The first `count`
/// entries are used (1 to 8).
pub fn children(
    fine: &Grid3d,
    coarse: &Grid3d,
    coarse_element: usize,
) -> ([(usize, [usize; 3]); 8], usize) {
    let halved = halved_directions(fine);
    let (ci, cj, ck) = coarse.element_position(coarse_element);
    let along = |index: usize, halved: bool, n: usize| -> ([(usize, usize); 2], usize) {
        if !halved {
            ([(index, 0), (0, 0)], 1)
        } else if 2 * index + 1 < n {
            ([(2 * index, 0), (2 * index + 1, 1)], 2)
        } else {
            ([(2 * index, 0), (0, 0)], 1)
        }
    };
    let (xs, nx) = along(ci, halved[0], fine.nx());
    let (ys, ny) = along(cj, halved[1], fine.ny());
    let (zs, nz) = along(ck, halved[2], fine.nz());
    let mut found = [(0, [0; 3]); 8];
    let mut count = 0;
    for &(k, pk) in &zs[..nz] {
        for &(j, pj) in &ys[..ny] {
            for &(i, pi) in &xs[..nx] {
                found[count] = (fine.element_index(i, j, k), [pi, pj, pk]);
                count += 1;
            }
        }
    }
    (found, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thin_plate_keeps_halving_in_its_plane() {
        // 64 x 64 x 2: the depth goes to 1 and stays there while the plane keeps halving.
        let grid = Grid3d::full(64, 64, 2, 1.0).unwrap();
        let levels = build_levels(&grid, &vec![false; grid.n_dofs()]).unwrap();
        let shapes: Vec<_> = levels
            .iter()
            .map(|l| (l.grid.nx(), l.grid.ny(), l.grid.nz()))
            .collect();
        assert_eq!(shapes, vec![(64, 64, 2), (32, 32, 1), (16, 16, 1)]);
        assert!(levels.last().unwrap().grid.n_dofs() <= COARSEST_MAX_DOFS);
    }

    #[test]
    fn odd_counts_give_coarse_elements_with_fewer_children() {
        let fine = Grid3d::full(5, 1, 3, 1.0).unwrap();
        let coarse = Grid3d::full(3, 1, 2, 1.0).unwrap();
        // Coarse element (2, 0, 1) holds fine element (4, 0, 2) only.
        let (found, count) = children(&fine, &coarse, coarse.element_index(2, 0, 1));
        assert_eq!(count, 1);
        assert_eq!(found[0], (fine.element_index(4, 0, 2), [0, 0, 0]));
        // Coarse element (0, 0, 0) holds four: x and z halved, y not.
        let (found, count) = children(&fine, &coarse, 0);
        assert_eq!(count, 4);
        assert_eq!(found[3], (fine.element_index(1, 0, 1), [1, 0, 1]));
        // Every fine element is the child of exactly one coarse element.
        let mut seen = vec![0; fine.n_elements()];
        for element in 0..coarse.n_elements() {
            let (found, count) = children(&fine, &coarse, element);
            for (child, _) in &found[..count] {
                seen[*child] += 1;
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
    }

    #[test]
    fn coarse_activity_and_fixed_follow_the_rules() {
        // 6 x 4 x 4 with the part only in the elements i < 3: the coarse elements i >= 2 are inactive,
        // coarse element i = 1 is active (one active child column).
        let active: Vec<bool> = (0..96).map(|element| element % 6 < 3).collect();
        let fine = Grid3d::new(6, 4, 4, 1.0, active).unwrap();
        let mut fixed = vec![false; fine.n_dofs()];
        for node in 0..fine.n_nodes() {
            let (i, _, _) = fine.node_position(node);
            // x = 0 held; nodes beyond i = 3 touch no active element.
            if i == 0 || i > 3 {
                fixed[3 * node..3 * node + 3].fill(true);
            }
        }
        let grid_levels = build_levels(&fine, &fixed);
        let coarse_grid = coarser_grid(&fine).unwrap();
        assert_eq!(
            (0..coarse_grid.n_elements())
                .filter(|&e| coarse_grid.is_active(e))
                .count(),
            2 * 2 * 2
        );
        // Without a level limit the 6 x 4 x 4 grid is already small: one level.
        assert_eq!(grid_levels.unwrap().len(), 1);
        let coarse_fixed: Vec<bool> = (0..coarse_grid.n_dofs())
            .map(|dof| !reaches_a_free_fine_dof(&fine, &fixed, &coarse_grid, dof))
            .collect();
        for node in 0..coarse_grid.n_nodes() {
            let (i, _, _) = coarse_grid.node_position(node);
            // Coarse node i = 0 sees fine nodes 0 (held) and 1 (free); i = 1 sees 1, 2, 3 (free);
            // i = 2 sees 3 (free), 4, 5; i = 3 sees 5 and 6 (fixed).
            let expected = i == 3;
            assert_eq!(coarse_fixed[3 * node], expected, "node {node}");
        }
    }
}
