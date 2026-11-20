//! The PDE filter (Helmholtz filter) on the 3D grid: the filtered density is the solution of a
//! Helmholtz-type equation with the design field as its source,
//!
//!   -r^2 lap(psi) + psi = x     on the active elements,     d psi / dn = 0 on their boundary,
//!
//! solved by finite elements on the same grid, with the radius `R` of the filter in mm and
//! `r = R / (2 sqrt 3)`. Source: Lazarov and Sigmund, "Filters in topology optimization based on
//! Helmholtz-type differential equations", Int. J. Numer. Meth. Eng. 86 (2011), which gives the equation,
//! its zero-flux boundary, the relation r = R / (2 sqrt 3) and the finite-element form below.
//!
//! Discretisation: psi is trilinear on 8-node cube elements (the shape functions of
//! `hex8`, one value per node); x is constant on each element. With
//!
//!   K_e = integral of grad N^T grad N,   M_e = integral of N^T N   (8 by 8)
//!
//! over one cube, both integrated by 2 x 2 x 2 Gauss points (exact here: every entry is at most quadratic
//! in each coordinate), and the mass matrix lumped, L_e = diag(row sums of M_e) = h^3 / 8 times the
//! identity, the equations are
//!
//!   A psi = T x,    A = sum over active elements of (r^2 K_e + L_e),    (T x)_n = sum over the active
//!   elements e around node n of h^3 / 8 * x_e,
//!
//! the last because each shape function integrates to h^3 / 8 over the cube. The filtered value of an
//! element is the mean of its eight node values, rho_e = 1/8 sum over its nodes of psi_n. The whole map is
//! rho = (1/8) Q^T A^-1 (h^3 / 8) Q x, with Q the node-by-element incidence matrix: symmetric, so the chain
//! rule back through the filter (its transpose) is the same map, and `backward` is the same solve.
//!
//! Two properties follow and are checked in verification row 20. The sum is kept: K_e times a constant is
//! zero and the diagonal of L_e is h^3 / 8 per node, so summing the equations gives sum_e rho_e =
//! sum_e x_e. A uniform field stays uniform: psi = c solves A psi = T (c, ..., c).
//!
//! Inactive elements take no part: their input is ignored and their output is 0. A node that touches no
//! active element acts as an identity row and column with a zero right-hand side.
//!
//! Solve: the preconditioned conjugate-gradient method of `conjugate_gradient` with the diagonal of A as
//! the preconditioner, stopped at the relative residual `tolerance` (1e-10 by default). The product A psi
//! is computed node by node in gather form, like the product of `elasticity3d`; the right-hand side and
//! the mean back to the elements likewise, one node or one element per piece, and the solver's sums run
//! over the fixed blocks of `threads`. The numbers therefore do not depend on the thread count.
//!
//! The range [0, 1], and why the mass matrix is lumped. Every off-diagonal entry of K_e is zero (two nodes
//! on one edge) or negative (-h/12 for two nodes across a face or across the cube; a unit test checks
//! this), and the rows of K_e sum to zero. With the lumped, diagonal mass matrix, A therefore has a
//! positive diagonal, no positive entry off it, and every row strictly diagonally dominant (the diagonal
//! exceeds the sum of the others by h^3 / 8 times the number of active elements at the node): A is an
//! M-matrix, and its inverse has no negative entry (a property of M-matrices; Varga, "Matrix Iterative Analysis", 1962).
//! The whole map then has no negative entry and maps a constant to the same constant, so a field in [0, 1]
//! stays in [0, 1], for any radius and any active region. The consistent mass matrix M_e has positive
//! entries between neighbouring nodes and loses this: measured on a 9 x 9 x 9 grid, its map has entries
//! down to -3.3e-3 at R = 1.5 h, and the response to one solid element in an empty block falls 2.3 %
//! below zero at R = 0.3 h; it keeps the range only from about R = 2 h.
//!
//! The iterative solve leaves an error of the order of its tolerance, which can put a value just outside
//! [0, 1]. `forward` puts a value that lies outside by at most `ROUND_OFF` back on the bound (the density
//! of an element must lie in [0, 1]); a larger excursion can only come from an input outside [0, 1] and is
//! returned as it is. `backward` is the linear map itself; the values put back move by less than
//! `ROUND_OFF`, so the gradient is not changed by more than that.

use crate::conjugate_gradient::{self, SolveReport};
use crate::error::{EngineError, ErrorCode};
use crate::grid3d::{Grid3d, NODE_STEPS};
use crate::threads::Threads;

/// The relative residual at which the solve stops, unless the filter is built with another.
pub const DEFAULT_TOLERANCE: f64 = 1e-10;

/// The most conjugate-gradient steps a solve may take. The diagonally preconditioned matrix has a
/// condition number of about 1 + 12 (r / h)^2, so a solve to 1e-10 takes some tens of steps for the radii
/// of practice and a few hundred for a radius of tens of elements; more than this means something is
/// wrong, and gives `solver_not_converged`.
pub const MAX_STEPS: usize = 10_000;

/// A filtered value outside [0, 1] by at most this much is solver round-off and is put on the bound (see
/// the file header). The error of a solve at the relative residual 1e-10 is at most the condition number
/// of A times that, about 1e-8 for a radius of ten elements; this bound leaves a margin of a hundred.
pub const ROUND_OFF: f64 = 1e-6;

/// The Helmholtz filter on one 3D grid.
#[derive(Clone, Debug)]
pub struct HelmholtzFilter {
    grid: Grid3d,
    /// The filter radius R, mm.
    radius: f64,
    /// The length scale r = R / (2 sqrt 3) of the equation, mm.
    length: f64,
    /// r^2 K_e + L_e for one cube element (L_e the lumped mass matrix).
    element_matrix: [[f64; 8]; 8],
    /// One entry per node: true when no active element touches it (an identity row and column).
    outside: Vec<bool>,
    /// The diagonal of A (1 on the nodes outside), the preconditioner.
    diagonal: Vec<f64>,
    tolerance: f64,
    threads: Threads,
}

impl HelmholtzFilter {
    /// The filter of radius `radius` mm on `grid`, solving to the relative residual `tolerance` on
    /// `threads` threads.
    pub fn new(
        grid: &Grid3d,
        radius: f64,
        tolerance: f64,
        threads: usize,
    ) -> Result<HelmholtzFilter, EngineError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(
                EngineError::invalid_input("the filter radius must be a positive number")
                    .with_number("radius", radius),
            );
        }
        conjugate_gradient::check_tolerance(tolerance)?;
        let threads = Threads::new(threads)?;
        let length = radius / (2.0 * 3.0_f64.sqrt());
        let (stiffness, mass) = element_matrices(grid.h());
        let mut element_matrix = [[0.0; 8]; 8];
        for a in 0..8 {
            for b in 0..8 {
                element_matrix[a][b] = length * length * stiffness[a][b];
            }
            // The lumped mass matrix: each row of M_e summed onto its diagonal.
            element_matrix[a][a] += mass[a].iter().sum::<f64>();
        }
        let mut outside = vec![true; grid.n_nodes()];
        for element in 0..grid.n_elements() {
            if grid.is_active(element) {
                for node in grid.element_nodes(element) {
                    outside[node] = false;
                }
            }
        }
        let mut filter = HelmholtzFilter {
            grid: grid.clone(),
            radius,
            length,
            element_matrix,
            outside,
            diagonal: Vec::new(),
            tolerance,
            threads,
        };
        filter.diagonal = filter.assembled_diagonal();
        Ok(filter)
    }

    /// The radius R, mm.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The length scale r = R / (2 sqrt 3) of the equation, mm.
    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The number of threads every call runs on.
    pub fn threads(&self) -> usize {
        self.threads.count()
    }

    pub fn n_elements(&self) -> usize {
        self.grid.n_elements()
    }

    /// The filtered density of `x`, one value per element, 0 on inactive elements. A value outside
    /// [0, 1] by at most `ROUND_OFF` (solver round-off) is put on the bound; for an input in [0, 1] no
    /// other value can lie outside (file header).
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        let (mut rho, _) = self.apply(x)?;
        for value in rho.iter_mut() {
            if (-ROUND_OFF..0.0).contains(value) {
                *value = 0.0;
            } else if *value > 1.0 && *value <= 1.0 + ROUND_OFF {
                *value = 1.0;
            }
        }
        Ok(rho)
    }

    /// dg/dx from dg/drho (`grad`): the chain rule back through the filter. The map is symmetric, so
    /// this is the map itself applied to `grad`.
    pub fn backward(&self, grad: &[f64]) -> Result<Vec<f64>, EngineError> {
        Ok(self.apply(grad)?.0)
    }

    /// The linear map (1/8) Q^T A^-1 (h^3 / 8) Q applied to `field` (one value per element), with how its
    /// solve ended. Inactive elements are ignored on input and 0 on output.
    pub fn apply(&self, field: &[f64]) -> Result<(Vec<f64>, SolveReport), EngineError> {
        self.check_field(field)?;
        let grid = &self.grid;
        let share = grid.element_volume() / 8.0;

        // Right-hand side, node by node: h^3 / 8 times the sum of the active elements around the node.
        let mut b = vec![0.0; grid.n_nodes()];
        self.threads.for_each_piece(&mut b, 1, |node, value| {
            let mut sum = 0.0;
            for element in grid.node_elements(node).into_iter().flatten() {
                if grid.is_active(element) {
                    sum += field[element];
                }
            }
            value[0] = share * sum;
        });

        let mut psi = vec![0.0; grid.n_nodes()];
        let report = conjugate_gradient::solve(
            &self.threads,
            |p, q| self.multiply(p, q),
            |r, z| {
                self.threads
                    .update(z, |node, z_n| *z_n = r[node] / self.diagonal[node])
            },
            &b,
            &mut psi,
            self.tolerance,
            MAX_STEPS,
        )?;

        // Back to the elements: the mean of the eight node values.
        let mut out = vec![0.0; grid.n_elements()];
        self.threads.update(&mut out, |element, value| {
            if grid.is_active(element) {
                let mut sum = 0.0;
                for node in grid.element_nodes(element) {
                    sum += psi[node];
                }
                *value = sum / 8.0;
            }
        });
        Ok((out, report))
    }

    /// q = A p, node by node: entry n gathers, over the active elements around node n (in which n is node
    /// `place`), the row `place` of the element matrix times the element's node values. Nodes outside
    /// the part are identity rows.
    fn multiply(&self, p: &[f64], q: &mut [f64]) {
        let grid = &self.grid;
        self.threads.for_each_piece(q, 1, |node, value| {
            if self.outside[node] {
                value[0] = p[node];
                return;
            }
            let mut sum = 0.0;
            for (place, element) in grid.node_elements(node).iter().enumerate() {
                let Some(element) = *element else { continue };
                if !grid.is_active(element) {
                    continue;
                }
                let row = &self.element_matrix[place];
                for (column, other) in grid.element_nodes(element).iter().enumerate() {
                    sum += row[column] * p[*other];
                }
            }
            value[0] = sum;
        });
    }

    /// The diagonal of A, node by node; 1 on the nodes outside the part. Positive on every other node: it
    /// sums positive diagonal entries of r^2 K_e + L_e.
    fn assembled_diagonal(&self) -> Vec<f64> {
        let grid = &self.grid;
        let mut diagonal = vec![1.0; grid.n_nodes()];
        self.threads
            .for_each_piece(&mut diagonal, 1, |node, value| {
                if self.outside[node] {
                    return;
                }
                let mut sum = 0.0;
                for (place, element) in grid.node_elements(node).iter().enumerate() {
                    let Some(element) = *element else { continue };
                    if grid.is_active(element) {
                        sum += self.element_matrix[place][place];
                    }
                }
                value[0] = sum;
            });
        diagonal
    }

    fn check_field(&self, field: &[f64]) -> Result<(), EngineError> {
        if field.len() != self.grid.n_elements() {
            return Err(
                EngineError::invalid_input("the field needs one entry per element")
                    .with_integer("expected", self.grid.n_elements())
                    .with_integer("got", field.len()),
            );
        }
        if let Some(element) = field.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::new(
                ErrorCode::InvalidInput,
                "the field holds a value that is not a finite number",
            )
            .with_integer("element", element)
            .with_number("value", field[element]));
        }
        Ok(())
    }
}

/// The scalar stiffness matrix K_e = integral of grad N_a . grad N_b and the consistent mass matrix
/// M_e = integral of N_a N_b over a cube of side h (the filter lumps M_e), both by 2 x 2 x 2 Gauss points (each of weight 1, at
/// +-1/sqrt 3). The shape functions are those of `hex8`: N_a = (1 + xi xi_a)(1 + eta eta_a)(1 + zeta
/// zeta_a) / 8, nodes in the order of `NODE_STEPS`, d/dx = (2 / h) d/dxi, dV = (h / 2)^3 dxi deta dzeta.
pub fn element_matrices(h: f64) -> ([[f64; 8]; 8], [[f64; 8]; 8]) {
    let natural = |a: usize, direction: usize| 2.0 * NODE_STEPS[a][direction] as f64 - 1.0;
    let g = 1.0 / 3.0_f64.sqrt();
    let volume_factor = (h / 2.0).powi(3);
    let mut stiffness = [[0.0; 8]; 8];
    let mut mass = [[0.0; 8]; 8];
    for point in 0..8 {
        // The Gauss points lie in the pattern of the nodes, at +-g instead of +-1.
        let (xi, eta, zeta) = (
            g * natural(point, 0),
            g * natural(point, 1),
            g * natural(point, 2),
        );
        let mut n = [0.0; 8];
        let mut gradient = [[0.0; 3]; 8];
        for a in 0..8 {
            let (xa, ya, za) = (natural(a, 0), natural(a, 1), natural(a, 2));
            n[a] = (1.0 + xi * xa) * (1.0 + eta * ya) * (1.0 + zeta * za) / 8.0;
            gradient[a] = [
                xa * (1.0 + eta * ya) * (1.0 + zeta * za) / 8.0 * 2.0 / h,
                ya * (1.0 + xi * xa) * (1.0 + zeta * za) / 8.0 * 2.0 / h,
                za * (1.0 + xi * xa) * (1.0 + eta * ya) / 8.0 * 2.0 / h,
            ];
        }
        for a in 0..8 {
            for b in 0..8 {
                let dot = gradient[a][0] * gradient[b][0]
                    + gradient[a][1] * gradient[b][1]
                    + gradient[a][2] * gradient[b][2];
                stiffness[a][b] += dot * volume_factor;
                mass[a][b] += n[a] * n[b] * volume_factor;
            }
        }
    }
    (stiffness, mass)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_field(n: usize, seed: u64) -> Vec<f64> {
        // A small linear congruential generator: values in [0, 1), the same on every machine.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    /// A 7 x 5 x 4 grid with a pocket of inactive elements.
    fn grid_with_a_pocket() -> Grid3d {
        let (nx, ny, nz) = (7, 5, 4);
        let mut active = vec![true; nx * ny * nz];
        for k in 1..3 {
            for j in 1..3 {
                for i in 2..5 {
                    active[i + nx * (j + ny * k)] = false;
                }
            }
        }
        Grid3d::new(nx, ny, nz, 1.5, active).unwrap()
    }

    #[test]
    fn the_element_matrices_have_the_textbook_entries() {
        // Unit cube: M_e = (1/216) * 8, 4, 2, 1 for a node with itself, an edge neighbour, a face diagonal
        // and the opposite corner; K_e = 1/3, 0, -1/12, -1/12 for the same pairs.
        let (k, m) = element_matrices(1.0);
        let pairs = [(0, 0), (0, 1), (0, 2), (0, 6)];
        let mass = [8.0 / 216.0, 4.0 / 216.0, 2.0 / 216.0, 1.0 / 216.0];
        let stiffness = [1.0 / 3.0, 0.0, -1.0 / 12.0, -1.0 / 12.0];
        for (index, (a, b)) in pairs.iter().enumerate() {
            assert!((m[*a][*b] - mass[index]).abs() < 1e-15);
            assert!((k[*a][*b] - stiffness[index]).abs() < 1e-15);
        }
        // Scaling with h: M with h^3, K with h.
        let (k2, m2) = element_matrices(2.0);
        assert!((m2[0][0] - 8.0 * m[0][0]).abs() < 1e-14);
        assert!((k2[0][0] - 2.0 * k[0][0]).abs() < 1e-14);
        // Rows of K sum to 0, rows of M to h^3 / 8.
        for a in 0..8 {
            assert!(k[a].iter().sum::<f64>().abs() < 1e-15);
            assert!((m[a].iter().sum::<f64>() - 0.125).abs() < 1e-15);
        }
    }

    #[test]
    fn a_uniform_field_stays_uniform_and_the_sum_is_kept() {
        let grid = grid_with_a_pocket();
        let filter = HelmholtzFilter::new(&grid, 4.0, 1e-12, 1).unwrap();
        let uniform = filter.forward(&vec![0.4; grid.n_elements()]).unwrap();
        for (element, value) in uniform.iter().enumerate() {
            let expected = if grid.is_active(element) { 0.4 } else { 0.0 };
            assert!((value - expected).abs() < 1e-11, "{element}: {value}");
        }
        let x = random_field(grid.n_elements(), 3);
        let rho = filter.forward(&x).unwrap();
        let before: f64 = (0..grid.n_elements())
            .filter(|e| grid.is_active(*e))
            .map(|e| x[e])
            .sum();
        let after: f64 = rho.iter().sum();
        assert!((after - before).abs() < 1e-10 * before);
    }

    #[test]
    fn backward_is_the_transpose() {
        let grid = grid_with_a_pocket();
        let filter = HelmholtzFilter::new(&grid, 3.0, 1e-12, 1).unwrap();
        let x = random_field(grid.n_elements(), 5);
        let y = random_field(grid.n_elements(), 7);
        let fx = filter.forward(&x).unwrap();
        let fy = filter.backward(&y).unwrap();
        let left: f64 = fx.iter().zip(&y).map(|(a, b)| a * b).sum();
        let right: f64 = x.iter().zip(&fy).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-10 * left.abs());
    }

    #[test]
    fn the_numbers_do_not_depend_on_the_thread_count() {
        // Enough nodes for several blocks of a sum.
        let grid = Grid3d::full(16, 12, 10, 1.0).unwrap();
        let x = random_field(grid.n_elements(), 11);
        let one = HelmholtzFilter::new(&grid, 2.5, 1e-10, 1).unwrap();
        let (expected, report) = one.apply(&x).unwrap();
        for count in [2, 5, 12] {
            let filter = HelmholtzFilter::new(&grid, 2.5, 1e-10, count).unwrap();
            let (values, other) = filter.apply(&x).unwrap();
            assert_eq!(other.steps, report.steps);
            assert!(
                values
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn no_off_diagonal_entry_of_the_element_stiffness_is_positive() {
        // The ground of the range argument in the file header: K_e has no positive entry off its diagonal.
        for h in [0.5, 1.0, 3.0] {
            let (k, _) = element_matrices(h);
            for (a, row) in k.iter().enumerate() {
                for (b, entry) in row.iter().enumerate() {
                    if a != b {
                        assert!(*entry <= 0.0, "h {h}: K[{a}][{b}] = {entry}");
                    }
                }
            }
        }
    }

    #[test]
    fn one_solid_element_stays_in_range_for_every_radius_and_keeps_its_sum() {
        // One solid element in an empty block, radii from far below an element to above it. The linear map
        // itself (before any clip) must stay within [0, 1] to the solver's tolerance.
        let grid = Grid3d::full(9, 9, 9, 1.0).unwrap();
        let mut x = vec![0.0; grid.n_elements()];
        x[grid.element_index(4, 4, 4)] = 1.0;
        for radius in [0.3, 1.0, 1.5, 2.5] {
            let filter = HelmholtzFilter::new(&grid, radius, 1e-10, 1).unwrap();
            let (values, _) = filter.apply(&x).unwrap();
            let lowest = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let highest = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                lowest >= -1e-10 && highest <= 1.0 + 1e-10,
                "R {radius}: {lowest} .. {highest}"
            );
            let sum: f64 = values.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "R {radius}: sum {sum}");
            let rho = filter.forward(&x).unwrap();
            assert!(rho.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        // The opposite field, a block with one empty element, stays at or below 1.
        let holed: Vec<f64> = x.iter().map(|v| 1.0 - v).collect();
        let filter = HelmholtzFilter::new(&grid, 0.3, 1e-10, 1).unwrap();
        let (values, _) = filter.apply(&holed).unwrap();
        assert!(values.iter().all(|v| *v <= 1.0 + 1e-10 && *v >= 0.0));
    }

    #[test]
    fn wrong_input_is_refused() {
        let grid = Grid3d::full(3, 3, 3, 1.0).unwrap();
        assert!(HelmholtzFilter::new(&grid, 0.0, 1e-10, 1).is_err());
        assert!(HelmholtzFilter::new(&grid, f64::NAN, 1e-10, 1).is_err());
        assert!(HelmholtzFilter::new(&grid, 2.0, 0.0, 1).is_err());
        assert!(HelmholtzFilter::new(&grid, 2.0, 1e-10, 0).is_err());
        let filter = HelmholtzFilter::new(&grid, 2.0, 1e-10, 1).unwrap();
        assert!(filter.forward(&[0.5; 26]).is_err());
        let mut x = vec![0.5; 27];
        x[4] = f64::INFINITY;
        assert!(filter.forward(&x).is_err());
    }
}
