//! The coarsest level of the multigrid, solved directly: its element matrices are assembled into the
//! banded Cholesky solver of `banded_cholesky` and factored on every solve.
//!
//! The coarse matrices may be positive semi-definite (`multigrid_coarse`): some combinations of coarse
//! degrees of freedom are mapped to zero by the interpolation. Those combinations do not depend on the
//! density, so they are found once, when the analysis object is built: the coarsest matrix with every
//! active element at modulus 1 is factored, and every degree of freedom whose pivot is zero (not above
//! `ZERO_PIVOT_RATIO` times its row's diagonal) is taken out: its row and column are set to zero and its
//! diagonal to 1, and it is kept at 0 in the solve. The same set is taken out of every later
//! factorisation. Since a zero pivot of a positive semi-definite matrix means the degree of freedom
//! depends on the ones before it, taking it out removes nothing the interpolation can see: the coarse
//! correction P (P^T A P)^+ P^T is the same.
//!
//! At run time every other pivot must be positive. One that is not would mean the factorisation has met
//! a dependence the full-density matrix did not have, which the mathematics rules out; it is reported as
//! `singular_system` with the numbers, never clamped to a small value.
//!
//! Numbering inside the solver: the nodes are counted with the direction that has the fewest nodes
//! fastest and the one with the most nodes slowest, which keeps the band narrow (in the outside numbering
//! x always runs fastest). Fixed degrees of freedom stay in the matrix as identity rows.

use crate::banded_cholesky::BandedMatrix;
use crate::error::{EngineError, ErrorCode};
use crate::multigrid_coarse::LevelMatrices;
use crate::multigrid_levels::Level;

/// A pivot of the full-density factorisation counts as zero when it is not above this fraction of its
/// row's diagonal: the threshold of the 2D check that a part is held (`HELD_CHECK_PIVOT_RATIO` in
/// `elasticity2d`). The coarsest level is small (at most a few thousand degrees of freedom), and there a
/// dependent row leaves a pivot of the order of 1e-15 of its diagonal while an independent one stays far
/// above 1e-10.
pub const ZERO_PIVOT_RATIO: f64 = 1e-10;

/// What the coarsest level keeps from the build of the analysis object: its numbering and the degrees of
/// freedom taken out.
#[derive(Clone, Debug)]
pub struct CoarsestShape {
    /// For each degree of freedom (outside numbering of the level), its row in the solver.
    solver_dof: Vec<usize>,
    half_band: usize,
    /// One entry per degree of freedom (outside numbering): taken out of every factorisation.
    dropped: Vec<bool>,
}

/// The coarsest matrix for one density, factored.
pub struct Coarsest<'a> {
    shape: &'a CoarsestShape,
    factor: BandedMatrix,
}

impl CoarsestShape {
    /// The numbering of `level`, and the set taken out, from a factorisation of its matrix at full
    /// density (`full_density`: every active element at the same modulus).
    pub fn new(level: &Level, full_density: LevelMatrices) -> CoarsestShape {
        let (solver_dof, half_band) = solver_numbering(level);
        let mut shape = CoarsestShape {
            solver_dof,
            half_band,
            dropped: vec![false; level.grid.n_dofs()],
        };
        let mut matrix = shape.assemble(level, full_density);
        let rows = matrix.factor_recording_zero_pivots(ZERO_PIVOT_RATIO);
        let mut dropped_row = vec![false; level.grid.n_dofs()];
        for row in rows {
            dropped_row[row] = true;
        }
        for dof in 0..level.grid.n_dofs() {
            shape.dropped[dof] = dropped_row[shape.solver_dof[dof]];
        }
        shape
    }

    /// The number of degrees of freedom taken out.
    pub fn dropped_count(&self) -> usize {
        self.dropped.iter().filter(|d| **d).count()
    }

    /// The level's matrix in the solver numbering, with identity rows and columns for fixed and dropped
    /// degrees of freedom.
    fn assemble(&self, level: &Level, matrices: LevelMatrices) -> BandedMatrix {
        let n = level.grid.n_dofs();
        let mut matrix = BandedMatrix::zeros(n, self.half_band);
        let left_out = |dof: usize| level.fixed[dof] || self.dropped[dof];
        for element in 0..level.grid.n_elements() {
            let Some(values) = matrices.element_matrix(level, element) else {
                continue;
            };
            let dofs = level.grid.element_dofs(element);
            for (a, &dof_a) in dofs.iter().enumerate() {
                if left_out(dof_a) {
                    continue;
                }
                for (b, &dof_b) in dofs.iter().enumerate() {
                    let (row, col) = (self.solver_dof[dof_a], self.solver_dof[dof_b]);
                    // Each pair once: the lower triangle (row >= col) holds the matrix.
                    if !left_out(dof_b) && row >= col {
                        matrix.add(row, col, values[24 * a + b]);
                    }
                }
            }
        }
        for (dof, &row) in self.solver_dof.iter().enumerate() {
            if left_out(dof) {
                matrix.add(row, row, 1.0);
            }
        }
        matrix
    }

    /// Assembles and factors the matrix of `level` for the element matrices `matrices`, with the set
    /// found at full density taken out. A pivot that is not positive outside that set gives
    /// `singular_system` with the numbers.
    pub fn factor(
        &self,
        level: &Level,
        matrices: LevelMatrices,
    ) -> Result<Coarsest<'_>, EngineError> {
        let mut factor = self.assemble(level, matrices);
        let mut dropped_row = vec![false; level.grid.n_dofs()];
        for dof in 0..level.grid.n_dofs() {
            dropped_row[self.solver_dof[dof]] = self.dropped[dof];
        }
        if let Err(failure) = factor.factor_without(&dropped_row, 0.0) {
            let dof = self
                .solver_dof
                .iter()
                .position(|&row| row == failure.row)
                .unwrap_or(failure.row);
            return Err(EngineError::new(
                ErrorCode::SingularSystem,
                format!(
                    "the coarsest multigrid matrix has a pivot that is not positive at its degree of \
                     freedom {dof}, outside the set found at full density"
                ),
            )
            .with_integer("coarsest_dof", dof)
            .with_integer("coarsest_dofs", level.grid.n_dofs())
            .with_number("pivot", failure.pivot)
            .with_number("diagonal", failure.diagonal));
        }
        Ok(Coarsest {
            shape: self,
            factor,
        })
    }
}

impl Coarsest<'_> {
    /// x = A^-1 r on the coarsest level, with the fixed and dropped degrees of freedom at 0. `fixed` is
    /// the level's fixed set.
    pub fn solve(&self, fixed: &[bool], r: &[f64], x: &mut [f64]) {
        let shape = self.shape;
        let mut rhs = vec![0.0; r.len()];
        for (dof, value) in r.iter().enumerate() {
            if !(fixed[dof] || shape.dropped[dof]) {
                rhs[shape.solver_dof[dof]] = *value;
            }
        }
        self.factor.solve_factored(&mut rhs);
        for (dof, value) in x.iter_mut().enumerate() {
            *value = if fixed[dof] || shape.dropped[dof] {
                0.0
            } else {
                rhs[shape.solver_dof[dof]]
            };
        }
    }
}

/// For each degree of freedom of the level, its row in the solver, and the half band that numbering
/// gives: nodes counted with the direction of fewest nodes fastest, the three of a node together.
fn solver_numbering(level: &Level) -> (Vec<usize>, usize) {
    let grid = &level.grid;
    let nodes = [grid.nx() + 1, grid.ny() + 1, grid.nz() + 1];
    // The directions from the fewest nodes to the most (a stable sort keeps x before y before z on ties).
    let mut order = [0, 1, 2];
    order.sort_by_key(|&d| nodes[d]);
    let (fast, middle) = (nodes[order[0]], nodes[order[1]]);
    let mut solver_dof = vec![0; grid.n_dofs()];
    for node in 0..grid.n_nodes() {
        let (i, j, k) = grid.node_position(node);
        let at = [i, j, k];
        let rank = at[order[0]] + fast * (at[order[1]] + middle * at[order[2]]);
        for c in 0..3 {
            solver_dof[3 * node + c] = 3 * rank + c;
        }
    }
    // Two nodes of one element differ by at most 1 in each direction.
    let half_band = 3 * (1 + fast + fast * middle) + 2;
    (solver_dof, half_band)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid3d::Grid3d;
    use crate::hex8;

    #[test]
    fn a_held_block_solves_exactly_and_drops_nothing() {
        // One level, 3 x 2 x 4, x = 0 clamped: the coarsest is the fine grid and is positive definite.
        let grid = Grid3d::full(3, 2, 4, 1.0).unwrap();
        let mut fixed = vec![false; grid.n_dofs()];
        for node in 0..grid.n_nodes() {
            if grid.node_position(node).0 == 0 {
                fixed[3 * node..3 * node + 3].fill(true);
            }
        }
        let level = Level {
            grid: grid.clone(),
            fixed: fixed.clone(),
        };
        let k_unit = hex8::stiffness(1.0, 0.3, 1.0);
        let ones = vec![1.0; grid.n_elements()];
        let shape = CoarsestShape::new(
            &level,
            LevelMatrices::Fine {
                moduli: &ones,
                k_unit: &k_unit,
            },
        );
        assert_eq!(shape.dropped_count(), 0);
        let moduli: Vec<f64> = (0..grid.n_elements())
            .map(|e| 0.2 + (e % 3) as f64)
            .collect();
        let source = LevelMatrices::Fine {
            moduli: &moduli,
            k_unit: &k_unit,
        };
        let coarsest = shape.factor(&level, source).unwrap();
        // A x = r against the dense product.
        let r: Vec<f64> = (0..grid.n_dofs())
            .map(|d| {
                if fixed[d] {
                    0.0
                } else {
                    (0.3 * d as f64).sin()
                }
            })
            .collect();
        let mut x = vec![0.0; grid.n_dofs()];
        coarsest.solve(&fixed, &r, &mut x);
        let mut ax = vec![0.0; grid.n_dofs()];
        for element in 0..grid.n_elements() {
            let m = source.element_matrix(&level, element).unwrap();
            let dofs = grid.element_dofs(element);
            for a in 0..24 {
                for b in 0..24 {
                    ax[dofs[a]] += m[24 * a + b] * x[dofs[b]];
                }
            }
        }
        for dof in 0..grid.n_dofs() {
            if !fixed[dof] {
                assert!(
                    (ax[dof] - r[dof]).abs() < 1e-6 * r.iter().map(|v| v.abs()).fold(0.0, f64::max)
                );
            } else {
                assert_eq!(x[dof], 0.0);
            }
        }
    }

    #[test]
    fn a_pivot_that_is_not_positive_at_run_time_is_an_error() {
        // The set found at full density is empty; moduli of zero then leave a zero pivot, which must be
        // reported as singular_system, never clamped.
        let grid = Grid3d::full(2, 2, 2, 1.0).unwrap();
        let mut fixed = vec![false; grid.n_dofs()];
        for node in 0..grid.n_nodes() {
            if grid.node_position(node).0 == 0 {
                fixed[3 * node..3 * node + 3].fill(true);
            }
        }
        let level = Level { grid, fixed };
        let k_unit = hex8::stiffness(1.0, 0.3, 1.0);
        let ones = vec![1.0; level.grid.n_elements()];
        let shape = CoarsestShape::new(
            &level,
            LevelMatrices::Fine {
                moduli: &ones,
                k_unit: &k_unit,
            },
        );
        assert_eq!(shape.dropped_count(), 0);
        let zeros = vec![0.0; level.grid.n_elements()];
        let source = LevelMatrices::Fine {
            moduli: &zeros,
            k_unit: &k_unit,
        };
        let error = shape.factor(&level, source).err().unwrap();
        assert_eq!(error.code, ErrorCode::SingularSystem);
        assert!(error.details.iter().any(|(key, _)| key == "pivot"));
    }

    #[test]
    fn the_numbering_puts_the_fewest_nodes_fastest() {
        let grid = Grid3d::full(6, 1, 3, 1.0).unwrap();
        let level = Level {
            fixed: vec![false; grid.n_dofs()],
            grid,
        };
        let (solver_dof, half_band) = solver_numbering(&level);
        // y has 2 nodes (fastest), then z (4), then x (7).
        assert_eq!(half_band, 3 * (1 + 2 + 2 * 4) + 2);
        let mut sorted = solver_dof.clone();
        sorted.sort();
        assert!(sorted.iter().enumerate().all(|(i, d)| i == *d));
        assert_eq!(solver_dof[3 * level.grid.node_index(0, 1, 0)], 3);
    }
}
